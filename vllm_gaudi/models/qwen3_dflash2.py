# SPDX-License-Identifier: Apache-2.0
"""DFlash2 drafter on HPU: the upstream context K/V precompute uses CUDA ops (ops.rms_norm with stacked weights,
ops.rotary_embedding, do_kv_cache_update); this runs it in torch and writes the HPU paged cache directly. The grouped
convolution already falls back to torch off CUDA."""
from typing import Optional

import torch
from vllm.model_executor.models.qwen3_dflash2 import DFlash2Qwen3ForCausalLM, DFlash2Qwen3Model

from vllm_gaudi.extension.utils import index_copy_indices
from vllm_gaudi.v1.spec_decode.dflash_ops import (chunked_topk, compute_context_kv, greedy_candidate_walk,
                                                  score_edges)


class HpuDFlash2Qwen3Model(DFlash2Qwen3Model):

    def _build_fused_kv_buffers(self) -> None:
        super()._build_fused_kv_buffers()
        assert self._fused_kv_weight is not None, "HPU DFlash needs unquantized draft K/V projections"
        attn0 = self.layers[0].self_attn
        object.__setattr__(self, "_rotary_emb", attn0.rotary_emb)  # a reference, not a submodule
        self._v_scale = attn0.v_scale

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor,
                input_embeds: Optional[torch.Tensor] = None) -> torch.Tensor:
        if input_embeds is not None:
            return self._decode_stack(input_embeds, positions)
        return self.decode(input_ids, positions)

    def decode(self, input_ids: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """Embedding, decoder layers and final norm, one compiled region on HPU (HpuDFlashProposer.compile_regions)."""
        return self._decode_stack(self.embed_input_ids(input_ids), positions)

    def _decode_stack(self, hidden_states: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(positions=positions, hidden_states=hidden_states, residual=residual)
        return self.norm(hidden_states, residual)[0]

    def _rope(self, positions: torch.Tensor, keys: torch.Tensor) -> torch.Tensor:
        # NeoX RoPE in plain torch: RotaryEmbedding.forward_native fails to launch here in eager mode
        # (synStatus 30, cause not determined).
        r = self._rotary_emb
        assert r.is_neox_style
        cos, sin = r.cos_sin_cache.index_select(0, positions).unsqueeze(1).chunk(2, dim=-1)
        x = keys.view(keys.shape[0], -1, r.head_size)
        x1, x2 = x[..., :r.rotary_dim].chunk(2, dim=-1)
        return torch.cat((x1 * cos - x2 * sin, x2 * cos + x1 * sin, x[..., r.rotary_dim:]), dim=-1).view_as(keys)

    def compute_context_kv(self, context_states: torch.Tensor,
                           context_positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """[N, hidden] combined target states -> K, V of every layer, each [L, N, num_kv_heads, head_dim]."""
        if not hasattr(self, "_rotary_emb"):  # dummy weights skip load_weights
            self._build_fused_kv_buffers()
        return compute_context_kv(context_states,
                                  context_positions,
                                  norm_weight=self.hidden_norm.weight,
                                  eps=self._rms_norm_eps,
                                  kv_weight=self._fused_kv_weight,
                                  kv_bias=self._fused_kv_bias,
                                  k_norm_weight=self._k_norm_weights,
                                  num_kv_heads=self._num_kv_heads,
                                  head_dim=self._head_dim,
                                  rope=self._rope,
                                  v_scale=self._v_scale)

    def store_context_kv(self, keys: torch.Tensor, values: torch.Tensor, slot_mapping: torch.Tensor) -> None:
        slot_mapping = index_copy_indices(slot_mapping)
        for i, attn in enumerate(self._attn_layers):
            kv_cache = attn.kv_cache
            if not isinstance(kv_cache, (tuple, list)):
                return  # memory profiling: no KV cache yet
            attn.impl.k_cache(keys[i], kv_cache[0], slot_mapping)
            attn.impl.v_cache(values[i], kv_cache[1], slot_mapping)

    def precompute_and_store_context_kv(self, context_states, context_positions, context_slot_mapping=None) -> None:
        assert not isinstance(context_slot_mapping, (list, tuple)), "HPU DFlash shares one slot mapping per layer"
        keys, values = self.compute_context_kv(context_states, context_positions)
        if context_slot_mapping is not None:
            self.store_context_kv(keys, values, context_slot_mapping)


class HpuDFlash2Qwen3ForCausalLM(DFlash2Qwen3ForCausalLM):
    model_cls = HpuDFlash2Qwen3Model

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor, inputs_embeds=None) -> torch.Tensor:
        # The grouped convolution needs [num_rows, hidden] with rows packed per request; the HPU runner passes
        # decode rows as [num_rows, 1].
        return self.model(input_ids.reshape(-1), positions.reshape(-1), inputs_embeds)

    def compute_context_kv(self, target_hidden_states: torch.Tensor,
                           context_positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Concatenated aux hidden states of the target [N, L_aux * hidden] -> per-layer context K, V."""
        return self.model.compute_context_kv(self.combine_hidden_states(target_hidden_states), context_positions)

    def store_context_kv(self, keys: torch.Tensor, values: torch.Tensor, slot_mapping: torch.Tensor) -> None:
        self.model.store_context_kv(keys, values, slot_mapping)

    def compute_candidates(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """LogitsProcessor.get_top_k_tokens (TP 1) with the vocab top-k by chunk maxima: torch.topk over the
        whole vocab is a radix sort on HPU (~1.3 ms for 7 rows, the chunked one ~0.09 ms)."""
        processor, lm_head = self.candidate_logits_processor, self.lm_head
        assert lm_head.tp_size == 1, "HPU DFlash candidates assume TP 1"
        logits = processor._apply_head(lm_head, hidden_states, None)
        num_pad = lm_head.shard_indices.num_org_vocab_padding
        if num_pad > 0:
            logits[..., -num_pad:] = -float("inf")
        top_k = self.model.candidate_selector.top_k
        if logits.shape[-1] % 128 == 0:
            values, ids = chunked_topk(logits, top_k)
        else:
            values, ids = torch.topk(logits, top_k, dim=-1)
        values = values.float()
        if processor.scale != 1.0:
            values = values * processor.scale
        if processor.soft_cap is not None:
            values = torch.tanh(values / processor.soft_cap) * processor.soft_cap
        return ids.to(torch.int64), values

    def propose_draft_tokens(self, hidden_states: torch.Tensor, anchor_token_ids: torch.Tensor) -> torch.Tensor:
        """Greedy DFlash2 drafts: [B * T, hidden] drafter outputs (row 0 of each block is the anchor) and the anchor
        (bonus) tokens [B] -> draft tokens [B, T - 1]."""
        top_k = self.model.candidate_selector.top_k
        hidden = hidden_states.view(anchor_token_ids.shape[0], -1, hidden_states.shape[-1])[:, 1:]
        batch_size, num_steps, hidden_size = hidden.shape
        candidate_ids, unary_logits = self.compute_candidates(hidden.reshape(-1, hidden_size))
        candidate_ids = candidate_ids.view(batch_size, num_steps, top_k)
        selector = self.model.candidate_selector
        scores = score_edges(selector.predecessor_codebook, selector.successor_codebook, candidate_ids,
                             unary_logits.view(batch_size, num_steps, top_k), selector.hidden_projection(hidden),
                             anchor_token_ids)
        return greedy_candidate_walk(scores, candidate_ids)
