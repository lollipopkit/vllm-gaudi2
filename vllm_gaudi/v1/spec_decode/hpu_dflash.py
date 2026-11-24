# SPDX-License-Identifier: Apache-2.0
import dataclasses
from collections.abc import Callable

import torch
from vllm.v1.spec_decode.dflash import DFlashProposer

from vllm_gaudi.utils import async_h2d_copy, slot_mapping_dtype
from vllm_gaudi.v1.attention.backends.hpu_attn import HPUAttentionMetadataV1
from vllm_gaudi.v1.spec_decode.dflash_ops import (block_mapping_from_groups, build_query_layout, window_block_bias)


class HpuDFlashProposer(DFlashProposer):
    """DFlash2 drafting on HPU (greedy).

    Every target step writes the drafter's context K/V (from the target's aux hidden states) at the target's slots;
    the drafter then runs one query block per request, the bonus token plus mask tokens at p .. p + k, as k + 1
    virtual decode rows that all see the whole block, and the candidate walk turns the mask rows into k drafts.
    The drafter layers share the target attention group's block table.
    """

    def load_model(self, target_model: torch.nn.Module) -> None:
        super().load_model(target_model)
        # The runner wraps self.model in an HpuModelAdapter; keep the drafter module for the non-forward paths.
        self.draft_model = self.model
        attns = [layer.self_attn for layer in self.draft_model.model.layers]
        windows = {attn.sliding_window for attn in attns}
        causal = {attn.causal for attn in attns}
        assert len(windows) == 1 and len(causal) == 1, "HPU DFlash needs one attention mode for all draft layers"
        self.sliding_window = windows.pop()
        self.causal = causal.pop()
        self._compute_context_kv = self.draft_model.compute_context_kv
        self._propose_draft_tokens = self.draft_model.propose_draft_tokens

    def row_block_buckets(self, block_size: int) -> list[int]:
        """Padded blocks per query row of the drafter's attention: powers of two up to the sliding window's
        block count. The drafter graph's block bucket is rows x this: a few variants, all reached by the decode
        warmup's context lengths."""
        if self.sliding_window is None:
            limit = -(-self.max_model_len // block_size)
        else:
            limit = -(-self.sliding_window // block_size) + 1
        buckets = []
        n = 2
        while n < limit:
            buckets.append(n)
            n *= 2
        return buckets + [limit]

    def compile_regions(self, compile_fn: Callable) -> None:
        """Compile the context K/V precompute, the embedding + decoder stack as one region (one per layer costs host time, and
        the async step is host-bound) and the candidate walk."""
        model = self.draft_model.model
        model.decode = compile_fn(model.decode)
        self._compute_context_kv = compile_fn(self.draft_model.compute_context_kv)
        self._propose_draft_tokens = compile_fn(self.draft_model.propose_draft_tokens)

    def precompute_context(self, aux_hidden_states: list[torch.Tensor], positions: torch.Tensor,
                           slot_mapping: torch.Tensor) -> None:
        """Write the drafter's K/V for every row the target just ran. Rejected verify rows sit inside the next query
        block, whose own K/V overwrite them before any query reads them."""
        states = torch.cat([h.reshape(-1, h.shape[-1]) for h in aux_hidden_states], dim=-1)
        num_ctx = states.shape[0]
        # mrope positions are [3, N]; text positions share all sections. Padding positions can be -1.
        positions = positions.reshape(-1, num_ctx)[0].clamp(min=0)
        keys, values = self._compute_context_kv(states, positions)
        self.draft_model.store_context_kv(keys, values, slot_mapping.reshape(-1))

    def propose(self, req_indices: list[int], num_reqs: int, input_batch, block_table_cpu: torch.Tensor,
                model_runner) -> torch.Tensor:
        """Drafts [num_reqs, k] for the requests in req_indices (others get zeros: prefill chunks without a sampled
        token). Positions and bonus tokens come from the CPU batch state, so warmup's compile-only runs work too."""
        k = self.num_speculative_tokens
        t = 1 + k
        drafts = torch.zeros(num_reqs, k, dtype=torch.int64, device=self.device)
        if not req_indices:
            return drafts
        block_size = model_runner.attn_block_size
        positions = (input_batch.num_tokens_no_spec[req_indices] - 1).clip(min=0).tolist()
        bonus = input_batch.token_ids_cpu[req_indices, positions].clip(0, self.draft_model.config.vocab_size - 1)
        width = block_table_cpu.shape[1]
        block_tables = model_runner._resolve_all_blocks([
            block_table_cpu[i, :min(width, (p + t + block_size - 1) // block_size)].tolist()
            for i, p in zip(req_indices, positions)
        ])
        batch_size = len(req_indices)
        padded_batch_size = batch_size
        if model_runner.enable_bucketing:
            padded_batch_size = model_runner.bucketing_manager.find_decode_bucket(batch_size,
                                                                                  sum(len(b) for b in block_tables),
                                                                                  True)[0]
        layout = build_query_layout(positions,
                                    bonus.tolist(),
                                    block_tables,
                                    num_query_per_req=t,
                                    mask_token_id=self.parallel_drafting_token_id,
                                    block_size=block_size,
                                    padded_batch_size=padded_batch_size,
                                    pad_slot_id=model_runner._PAD_SLOT_ID,
                                    sliding_window=self.sliding_window,
                                    causal=self.causal)
        num_rows = padded_batch_size * t
        # Graph block bucket: rows x the smallest row bucket holding the longest row (warmed up per bucket).
        longest = max((len(b) for b in layout.row_block_tables), default=1)
        row_blocks = next((n for n in self.row_block_buckets(block_size) if n >= longest), longest)
        # Sliding-window rows go through the window block path (gather by block id): build plain block lists.
        block_list, block_groups, block_usage = model_runner.get_habana_paged_attn_buffers(
            layout.row_block_tables,
            layout.row_last_slots,
            num_rows,
            block_size=block_size,
            force_non_contiguous=self.sliding_window is not None,
            block_bucket_size=num_rows * row_blocks if self.sliding_window is not None else None)
        block_list_device = async_h2d_copy(block_list, device=self.device)
        block_usage_device = async_h2d_copy(block_usage, device=self.device)
        block_groups_device = async_h2d_copy(block_groups, device=self.device)
        attn_metadata = HPUAttentionMetadataV1.make_decode_metadata(
            block_list=block_list_device,
            block_usage=block_usage_device,
            block_groups=block_groups_device,
            input_positions=None,
            slot_mapping=async_h2d_copy(layout.slot_mapping.to(slot_mapping_dtype()), device=self.device),
            block_size=block_size,
            window_block_list=None,
            window_block_usage=None,
            window_block_groups=None,
            chunked_block_list=None,
            chunked_block_usage=None,
            chunked_block_groups=None,
        )
        if self.sliding_window is not None:
            bias = window_block_bias(layout.row_block_tables, layout.row_first_offsets, block_usage, block_size,
                                     self.dtype)
            attn_metadata = dataclasses.replace(
                attn_metadata,
                window_block_list=block_list_device,
                window_block_usage=block_usage_device,
                window_block_groups=block_groups_device,
                window_block_mapping=async_h2d_copy(block_mapping_from_groups(block_groups, num_rows, self.dtype),
                                                    device=self.device),
                window_attn_bias=async_h2d_copy(bias, device=self.device),
            )
        input_ids = async_h2d_copy(layout.input_ids, device=self.device)
        hidden_states = self.model(input_ids=input_ids,
                                   positions=async_h2d_copy(layout.positions, device=self.device),
                                   attn_metadata=attn_metadata)
        anchors = input_ids.view(padded_batch_size, t)[:, 0].long()
        tokens = self._propose_draft_tokens(hidden_states, anchors)[:batch_size]
        if req_indices == list(range(num_reqs)):
            return tokens
        drafts[async_h2d_copy(torch.tensor(req_indices, dtype=torch.int64), device=self.device)] = tokens
        return drafts
