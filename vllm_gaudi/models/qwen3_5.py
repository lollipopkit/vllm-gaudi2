import torch
from vllm.model_executor.layers.fusion.quant_activation import QuantizedActivation, get_input_quant_key
from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import QwenGatedDeltaNetAttention
from vllm.model_executor.layers.quantization.utils.quant_utils import kFp8DynamicTokenSym
from vllm.forward_context import get_forward_context

import vllm_gaudi.extension.ops as hpu_ops
from vllm_gaudi.ops.causal_conv1d_pytorch import (
    hpu_causal_conv1d_fn,
    hpu_causal_conv1d_update,
)
from vllm_gaudi.ops.hpu_gaudi2_fp8_tpc import has_gdn_gated_rms_norm_f8, has_gdn_prep_decode
from vllm_gaudi.ops.hpu_gdn_pytorch import (
    _l2norm_last_dim,
    hpu_chunk_gated_delta_rule,
    hpu_fused_gdn_gating,
    hpu_fused_recurrent_gated_delta_rule,
)
from vllm_gaudi.ops import hpu_gdn_tpc
from vllm_gaudi.ops.hpu_gdn_spec import hpu_causal_conv1d_update_spec, hpu_gdn_spec_verify
from vllm_gaudi.ops.flashinfer_gaudi_adapter import (
    conv_taps,
    flashinfer_gdn_enabled,
    flashinfer_gdn_fused_decode_enabled,
    flashinfer_gdn_prefill_enabled,
    maybe_run_gdn_decode_packed,
    maybe_run_gdn_fused_decode_step,
    maybe_run_gdn_prefill,
)


def _save_ssm_state(core_attn_out, final_state, ssm_state, state_indices):
    """Persist GDN final_state into ssm_state cache for chunked prefill.

    Must be @torch._dynamo.disable because HPU torch.compile silently
    drops in-place index_copy_ to aliased state tensors.  Returns
    core_attn_out as a pass-through so the compiled graph consumes
    the call — HPU drops dynamo-disabled calls whose results are unused.
    """
    safe_si = torch.remainder(state_indices, ssm_state.shape[0]).long()
    ssm_state.index_copy_(0, safe_si, hpu_gdn_tpc.from_vk(final_state.to(device=ssm_state.device,
                                                                         dtype=ssm_state.dtype)))
    return core_attn_out


class HPUGatedDeltaNetAttention(QwenGatedDeltaNetAttention):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # cache_group_idx: set later by model runner for hybrid cache
        # lookup.  Stored as tensor so torch.compile treats it as dynamic.
        self.cache_group_idx = None
        # Compact GDN group-major layout (set by the model runner): this
        # layer's group position and the number of GDN groups sharing a pool.
        self.compact_state_group_offset = None
        self.compact_state_group_count = None
        # Resolved once so compiled graphs never read the environment.
        self.gdn_flashinfer_decode = flashinfer_gdn_enabled()
        self.gdn_flashinfer_fused_decode = flashinfer_gdn_fused_decode_enabled()
        self.gdn_flashinfer_prefill = flashinfer_gdn_prefill_enabled()
        # FP32 [4, width] conv taps & precomputed gating constants for the fused decode step.
        self.conv_taps = None
        self.neg_exp_A = None
        self.dt_bias_f32 = None
        self.conv_bias_f32 = None
        self._conv_bias_dummy = None
        self._norm_weight_f32 = None
        self._use_tpc_gdn_prep = False
        self._use_tpc_gdn_gated_norm = False
        self._ba_consumes_prequant = False

        # mamba_chunk_size: use explicit config value or default to 128
        # for HPU bucket alignment.
        hf_text_config = getattr(self.model_config, "hf_text_config", None)
        has_explicit = (hf_text_config is not None and (getattr(hf_text_config, "mamba_chunk_size", None) is not None
                                                        or getattr(hf_text_config, "chunk_size", None) is not None))
        self.mamba_chunk_size = (self.model_config.get_mamba_chunk_size() if has_explicit else 128)

        self.qkv_size = (self.key_dim * 2 + self.value_dim) // self.tp_size
        self.z_size = self.value_dim // self.tp_size

    def prepare_hpu_constants(self):
        """Called by the model runner after loading (weights on device)."""
        if self.gdn_flashinfer_fused_decode:
            w = self.conv1d.weight
            self.conv_taps = conv_taps(w.view(w.size(0), w.size(2)))
            self.neg_exp_A = (-torch.exp(self.A_log.to(torch.float32))).contiguous()
            self.dt_bias_f32 = self.dt_bias.to(torch.float32).contiguous()
            if self.conv1d.bias is not None:
                self.conv_bias_f32 = self.conv1d.bias.to(torch.float32).contiguous()
        num_v_heads_local = self.num_v_heads // self.tp_size
        self._use_tpc_gdn_prep = (
            self.gdn_flashinfer_fused_decode
            and hpu_gdn_tpc.STATE_KV
            and has_gdn_prep_decode()
            and self.conv_taps is not None
            and self.conv_taps.shape[0] == 4
            and self.activation in ("silu", "swish")
            and self.head_k_dim == 128
            and self.head_v_dim == 128
            and self.num_v_heads == 3 * self.num_k_heads
            and num_v_heads_local <= 64
        )
        if self._use_tpc_gdn_prep and self.conv_bias_f32 is None:
            self._conv_bias_dummy = torch.zeros(
                self.conv_taps.shape[1], dtype=torch.float32, device=self.conv_taps.device
            )
        self._use_tpc_gdn_gated_norm = (
            has_gdn_gated_rms_norm_f8()
            and get_input_quant_key(self.out_proj) == kFp8DynamicTokenSym
            and self.head_v_dim == 128
            and num_v_heads_local <= 128
            and getattr(self.norm, "group_size", None) is None
            and getattr(self.norm, "bias", None) is None
            and getattr(self.norm, "norm_before_gate", True)
            and getattr(self.norm, "activation", "silu") in ("silu", "swish")
        )
        if self._use_tpc_gdn_gated_norm:
            self._norm_weight_f32 = self.norm.weight.float().contiguous()
        self._ba_consumes_prequant = (
            hasattr(self, "in_proj_ba") and get_input_quant_key(self.in_proj_ba) == kFp8DynamicTokenSym
        )

    def rearrange_mixed_qkv(self, mixed_qkv):
        """Pure-torch rearrange – avoids einops graph breaks on HPU."""
        if mixed_qkv is None:
            return None, None, None
        key_dim_local = self.key_dim // self.tp_size
        val_dim_local = self.value_dim // self.tp_size
        query, key, value = torch.split(
            mixed_qkv,
            [key_dim_local, key_dim_local, val_dim_local],
            dim=-1,
        )
        n_tok = query.size(0)
        query = query.reshape(1, n_tok, key_dim_local // self.head_k_dim, self.head_k_dim).contiguous()
        key = key.reshape(1, n_tok, key_dim_local // self.head_k_dim, self.head_k_dim).contiguous()
        value = value.reshape(1, n_tok, val_dim_local // self.head_v_dim, self.head_v_dim).contiguous()
        return query, key, value

    def _select_cache_group(self, indices):
        if indices is not None and indices.dim() > 1:
            cg = self.cache_group_idx
            assert cg is not None
            indices = indices.index_select(0, cg.view(1)).squeeze(0)
        return indices

    def _resolve_state_indices(self, attn_metadata):
        """Slot to READ the incoming recurrent state from (last computed block)."""
        return self._select_cache_group(attn_metadata.load_indices_tensor)

    def _resolve_store_indices(self, attn_metadata):
        """Slot to WRITE the outgoing recurrent state to (last scheduled block).

        With prefix caching off the runner sets ``load_indices_tensor`` and
        ``store_indices_tensor`` to the same thing, so this used to be assumed
        identical to the load slot. With prefix caching on they differ: load is
        ``block_idx_last_computed_token`` while store is
        ``block_idx_last_scheduled_token``. Writing the final state to the load
        slot then leaves the block the next step reads from never written, which
        corrupts generation for any prompt spanning more than one mamba block.
        """
        store = getattr(attn_metadata, "store_indices_tensor", None)
        if store is None:
            return self._resolve_state_indices(attn_metadata)
        return self._select_cache_group(store)

    def _extract_metadata(self, num_tokens):
        """Extract forward-context metadata into plain tensors.

        Dynamo graph-breaks naturally on ``get_forward_context()``; no
        ``@dynamo.disable`` needed.
        """
        forward_context = get_forward_context()
        attn_metadata = forward_context.attn_metadata
        if attn_metadata is None:
            return (False, None, None, None, None, None, None, 0, 0, 0, 0, None, None, False, None)

        is_prompt = bool(getattr(attn_metadata, "is_prompt", False))
        state_indices = self._resolve_state_indices(attn_metadata)
        store_indices = self._resolve_store_indices(attn_metadata)

        conv_state = self.kv_cache[0]
        ssm_state = self.kv_cache[1]

        query_start_loc = attn_metadata.query_start_loc_p
        has_initial_state = getattr(attn_metadata, "has_initial_states_p", None)
        padding_mask_flat = getattr(attn_metadata, "padding_mask_flat", None)
        # Runner proof that request i owns row i of every compact GDN group.
        direct_gdn_state = (not is_prompt and self.gdn_flashinfer_decode
                            and bool(getattr(attn_metadata, "direct_gdn_state", False)))

        if not is_prompt:
            num_decodes = (state_indices.numel() if state_indices is not None else
                           (query_start_loc.numel() - 1 if query_start_loc is not None else num_tokens))
        else:
            num_decodes = 0

        mamba_block_size = (self.cache_config.mamba_block_size if is_prompt else 0)

        # Prefill-specific metadata (Python ints for torch.compile)
        prefill_num_seqs = 0
        prefill_seq_len = 0
        initial_state = None
        if is_prompt and state_indices is not None:
            prefill_num_seqs = int(state_indices.numel())
            prefill_seq_len = (num_tokens // prefill_num_seqs if prefill_num_seqs > 0 else 0)
            initial_state = hpu_gdn_tpc.to_vk(ssm_state[state_indices]).contiguous()
            if has_initial_state is not None:
                # Select instead of multiply: 0 * NaN/Inf cannot discard
                # non-finite cache rows for requests without initial state.
                # Block 0 is the reserved null block (BlockPool.null_block);
                # its state is never legitimate, so zero it even when
                # has_initial_state is True (prefix-cache-hit request whose
                # load index resolves to a null-filled block-table entry).
                keep = (has_initial_state.bool().view(-1) & (state_indices.reshape(-1) != 0))
                initial_state = torch.where(keep.view(-1, 1, 1, 1), initial_state, torch.zeros_like(initial_state))

        return (is_prompt, conv_state, ssm_state, state_indices, query_start_loc, has_initial_state, padding_mask_flat,
                num_decodes, mamba_block_size, prefill_num_seqs, prefill_seq_len, initial_state, store_indices,
                direct_gdn_state, getattr(attn_metadata, "gdn_num_accepted", None))

    def _spec_verify(self, mixed_qkv, a, b, conv_state, conv_weights, ssm_state, load_rows, base_rows, num_accepted,
                     batch, tokens):
        """Verify 1 + k tokens per request (see ops/hpu_gdn_spec.py).

        A request owns rows ``base_rows + t`` (t = 0..k) of the compact pool: the SSM state after token t is
        stored in row t and read back from row ``num_accepted - 1`` (``load_rows``); the conv state lives in
        row k, where prefill and non-spec decode keep it. Padding requests (base -1) stay on the sentinel.
        """
        g, beta = hpu_fused_gdn_gating(self.A_log, a, b, self.dt_bias)
        valid = (base_rows.view(-1, 1) >= 0).to(base_rows.dtype)  # arithmetic: HPU miscompiles int where()
        offsets = torch.arange(tokens, device=base_rows.device, dtype=base_rows.dtype) * valid
        store_rows = base_rows.view(-1, 1) + offsets
        conv = hpu_causal_conv1d_update_spec(mixed_qkv.view(batch, tokens, -1), conv_state, conv_weights,
                                             self.conv1d.bias, self.activation, store_rows[:, -1], num_accepted)
        query, key, value = self.rearrange_mixed_qkv(conv.reshape(batch * tokens, -1))
        if hpu_gdn_tpc.STATE_KV:
            # One TPC pass per request and head: the state stays in VLM across the T tokens.
            pool_rows = ssm_state.shape[0]
            qk = _l2norm_last_dim(torch.stack((query, key)).float())
            out = hpu_gdn_tpc.gdn_tpc_verify(qk[0].view(batch, tokens, -1, self.head_k_dim),
                                             qk[1].view(batch, tokens, -1, self.head_k_dim),
                                             value.view(batch, tokens, -1, self.head_v_dim),
                                             g.view(batch, tokens, -1), beta.view(batch, tokens, -1), ssm_state,
                                             torch.remainder(load_rows.reshape(-1), pool_rows),
                                             torch.remainder(store_rows, pool_rows), self.head_k_dim**-0.5,
                                             self.num_v_heads // self.num_k_heads)
            return out.to(value.dtype).reshape(1, batch * tokens, *out.shape[2:])
        out = hpu_gdn_spec_verify(query.view(batch, tokens, -1, self.head_k_dim),
                                  key.view(batch, tokens, -1, self.head_k_dim),
                                  value.view(batch, tokens, -1, self.head_v_dim), g.view(batch, tokens, -1),
                                  beta.view(batch, tokens, -1), ssm_state, load_rows, store_rows,
                                  scale=self.head_k_dim**-0.5)
        return out.reshape(1, batch * tokens, *out.shape[2:])

    def _indexed_decode_kv(self, query, key, value, g, beta, ssm_state, state_indices, store_indices,
                           query_start_loc, num_decodes):
        """Indexed (non-direct) decode on the KV-major pool of VLLM_HPU_GDN_TPC_DECODE: gather, run the VK
        recurrence, scatter back transposed. Only steps whose requests do not own contiguous rows land here."""
        rows = ssm_state.shape[0]
        load = torch.remainder(state_indices.reshape(-1).long(), rows)
        store = load if store_indices is None else torch.remainder(store_indices.reshape(-1).long(), rows)
        out, final_state = hpu_fused_recurrent_gated_delta_rule(
            q=query, k=key, v=value, g=g, beta=beta,
            initial_state=hpu_gdn_tpc.to_vk(ssm_state[load]).contiguous(),
            inplace_final_state=False,
            cu_seqlens=(query_start_loc[:num_decodes + 1] if query_start_loc is not None else None),
            use_qk_l2norm_in_kernel=True,
        )
        return _save_ssm_state(out, final_state, ssm_state, store)

    def forward(
        self,
        hidden_states: torch.Tensor,
        prequant_f8: torch.Tensor | None = None,
        prequant_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """HPU compile-friendly GDN forward.

        Bypasses the upstream ``gdn_attention_core`` custom-op and
        drives the HPU conv1d + GDN kernels directly with
        ``HPUAttentionMetadataV1``.

        Return-based since upstream vLLM #46998 (300e33797f) dropped the
        ``output`` in-place buffer; caller now does
        ``hidden_states = self.linear_attn(hidden_states=...)``.
        """
        orig_shape = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_states.size(-1))
        num_tokens = hidden_states.size(0)

        # === Metadata extraction (natural graph break) ===============
        (is_prompt, conv_state, ssm_state, state_indices, query_start_loc, has_initial_state, padding_mask_flat,
         num_decodes, mamba_block_size, prefill_num_seqs, prefill_seq_len, initial_state, store_indices,
         direct_gdn_state, num_accepted) = self._extract_metadata(num_tokens)

        # === Part 1: Input Projection ================================
        num_v_heads_local = self.num_v_heads // self.tp_size
        key_dim_local = self.key_dim // self.tp_size
        val_dim_local = self.value_dim // self.tp_size
        if hasattr(self, 'in_proj_qkv'):
            # LoRA path (Qwen3.5 only): separate in_proj_qkv and in_proj_z
            mixed_qkv, _ = self.in_proj_qkv(hidden_states)
            ba, _ = self.in_proj_ba(hidden_states)
            z, _ = self.in_proj_z(hidden_states)
            b, a = ba.chunk(2, dim=-1)
        else:
            if prequant_f8 is not None and prequant_scale is not None:
                qkvz_in = QuantizedActivation(
                    data=prequant_f8,
                    scale=prequant_scale,
                    orig_dtype=hidden_states.dtype,
                    orig_shape=(num_tokens, hidden_states.size(-1)),
                    quant_key=kFp8DynamicTokenSym,
                )
            else:
                qkvz_in = hidden_states
            mixed_qkvz, _ = self.in_proj_qkvz(qkvz_in)
            ba, _ = self.in_proj_ba(qkvz_in if (self._ba_consumes_prequant and qkvz_in is not hidden_states)
                                    else hidden_states)

            if self.gqa_interleaved_layout:
                # Qwen3-Next: unpack the interleaved GQA layout with explicit head dims (avoid HPU -1 -> 0 reshape)
                num_k_heads_local = key_dim_local // self.head_k_dim
                v_per_k = num_v_heads_local // num_k_heads_local
                qkvz_group_dim = 2 * self.head_k_dim + 2 * v_per_k * self.head_v_dim
                mixed_qkvz_3d = mixed_qkvz.reshape(num_tokens, num_k_heads_local, qkvz_group_dim)
                ba_3d = ba.reshape(num_tokens, num_k_heads_local, 2 * v_per_k)
                query, key, value, z = torch.split(
                    mixed_qkvz_3d,
                    [self.head_k_dim, self.head_k_dim, v_per_k * self.head_v_dim, v_per_k * self.head_v_dim],
                    dim=2,
                )
                b, a = torch.split(ba_3d, [v_per_k, v_per_k], dim=2)
                b = b.reshape(num_tokens, num_v_heads_local)
                a = a.reshape(num_tokens, num_v_heads_local)
                query = query.reshape(num_tokens, key_dim_local)
                key = key.reshape(num_tokens, key_dim_local)
                value = value.reshape(num_tokens, val_dim_local)
                mixed_qkv = torch.cat((query, key, value), dim=-1)
            else:
                # Qwen3.5: weights already in [q, k, v, z] and [b, a] order
                mixed_qkv, z = mixed_qkvz.split([self.qkv_size, self.z_size], dim=-1)
                b, a = ba.chunk(2, dim=-1)

        out_shape = (num_tokens, num_v_heads_local, self.head_v_dim)

        if conv_state is None:
            # No attn_metadata — skip core attention (profile run)
            core_attn_out = torch.zeros(out_shape, dtype=hidden_states.dtype, device=hidden_states.device)
        elif is_prompt:
            # === Part 2a: Prefill ====================================
            if (padding_mask_flat is not None and padding_mask_flat.numel() == num_tokens):
                token_mask_flat = padding_mask_flat.view(-1, 1).to(dtype=mixed_qkv.dtype)
                mixed_qkv = mixed_qkv * token_mask_flat
            else:
                token_mask_flat = None

            g, beta = hpu_fused_gdn_gating(self.A_log, a, b, self.dt_bias)

            conv_weights = self.conv1d.weight.view(self.conv1d.weight.size(0), self.conv1d.weight.size(2))
            mixed_qkv_conv = hpu_causal_conv1d_fn(
                x=mixed_qkv.transpose(0, 1),
                weight=conv_weights,
                bias=self.conv1d.bias,
                activation=self.activation,
                conv_states=conv_state,
                has_initial_state=has_initial_state,
                cache_indices=store_indices,
                block_idx_first_scheduled_token=None,
                block_idx_last_scheduled_token=None,
                # Read the incoming state from the load slot while cache_indices
                # writes the outgoing state to the store slot.
                initial_state_idx=state_indices,
                query_start_loc=query_start_loc,
                block_size_to_align=mamba_block_size,
                num_computed_tokens=None,
                metadata=None,
                is_prompt=True,
            ).transpose(0, 1)

            if token_mask_flat is not None:
                mixed_qkv_conv = mixed_qkv_conv * token_mask_flat

            query, key, value = self.rearrange_mixed_qkv(mixed_qkv_conv)

            if token_mask_flat is not None:
                token_mask_h = token_mask_flat.view(1, -1, 1).to(dtype=g.dtype)
                g = g * token_mask_h
                beta = beta * token_mask_h

            prefill_result = None
            if self.gdn_flashinfer_prefill:
                prefill_result = maybe_run_gdn_prefill(
                    q=query,
                    k=key,
                    v=value,
                    log_decay=g,
                    beta=beta,
                    initial_state=initial_state,
                    output_final_state=True,
                    use_qk_l2norm_in_kernel=True,
                    chunk_size=self.mamba_chunk_size,
                    prefill_num_seqs=prefill_num_seqs,
                    prefill_seq_len=prefill_seq_len,
                    scale=self.head_k_dim**-0.5,
                )
            if prefill_result is not None:
                core_attn_out_result, final_state = prefill_result
            else:
                core_attn_out_result, final_state = hpu_chunk_gated_delta_rule(
                    q=query,
                    k=key,
                    v=value,
                    g=g,
                    beta=beta,
                    initial_state=initial_state,
                    output_final_state=True,
                    use_qk_l2norm_in_kernel=True,
                    chunk_size=self.mamba_chunk_size,
                    prefill_num_seqs=prefill_num_seqs,
                    prefill_seq_len=prefill_seq_len,
                )
            # State save in dynamo-disabled wrapper — index_copy_ is
            # silently dropped by HPU torch.compile on aliased tensors.
            core_attn_out_result = _save_ssm_state(
                core_attn_out_result,
                final_state,
                ssm_state,
                store_indices,
            )

            non_spec_out = core_attn_out_result.squeeze(0)
            if non_spec_out.shape[0] == num_tokens:
                core_attn_out = non_spec_out
            else:
                core_attn_out = torch.zeros(out_shape, dtype=hidden_states.dtype, device=hidden_states.device)
                n = min(non_spec_out.shape[0], num_tokens)
                core_attn_out[:n] = non_spec_out[:n]

        else:
            # === Part 2b: Decode =====================================
            conv_weights = self.conv1d.weight.view(self.conv1d.weight.size(0), self.conv1d.weight.size(2))
            decode_result = None
            # Spec-decode verify: 1 + k tokens per request (the runner flattens [bs, 1 + k] to bs * (1 + k)).
            spec_tokens = num_tokens // num_decodes if num_decodes else 1
            if spec_tokens > 1:
                decode_result = (self._spec_verify(mixed_qkv, a, b, conv_state, conv_weights, ssm_state, state_indices,
                                                   store_indices, num_accepted, num_decodes, spec_tokens), )
            if decode_result is None and direct_gdn_state and self.conv_taps is not None:
                group_count = self.compact_state_group_count
                group_offset = self.compact_state_group_offset
                if (self._use_tpc_gdn_prep and group_count is not None and group_count > 0
                        and group_offset is not None and 0 <= group_offset < group_count
                        and conv_state.ndim == 3 and conv_state.shape[1] >= 3 and conv_state.is_contiguous()
                        and conv_state.dtype == torch.bfloat16 and ssm_state.ndim == 4
                        and ssm_state.is_contiguous() and ssm_state.dtype == torch.float32
                        and tuple(ssm_state.shape[2:]) == (128, 128)
                        and conv_state.shape[0] == ssm_state.shape[0]
                        and mixed_qkv.dtype == torch.bfloat16 and a.dtype == torch.bfloat16
                        and b.dtype == torch.bfloat16 and mixed_qkv.shape[1] == conv_state.shape[2]
                        and a.shape[1] == ssm_state.shape[1] and b.shape[1] == ssm_state.shape[1]):
                    span, rem = divmod(ssm_state.shape[0] - 2, group_count)
                    if rem == 0 and 0 < num_tokens <= span:
                        row_offset = group_offset * span + 1
                        conv_col_offset = conv_state.shape[1] - 3
                        has_bias = 1 if self.conv_bias_f32 is not None else 0
                        bias_t = self.conv_bias_f32 if has_bias else self._conv_bias_dummy
                        qk, v, decay, beta, kqdot = torch.ops.custom_op.gdn_prep_decode(
                            mixed_qkv.contiguous(),
                            a.contiguous(),
                            b.contiguous(),
                            self.neg_exp_A,
                            self.dt_bias_f32,
                            conv_state,
                            self.conv_taps,
                            bias_t,
                            row_offset,
                            conv_col_offset,
                            has_bias,
                        )
                        out_f32 = torch.ops.custom_op.gdn_decode(
                            ssm_state, qk, v, decay, beta, kqdot, row_offset, self.head_k_dim**-0.5
                        )
                        if self._use_tpc_gdn_gated_norm and z.dtype == torch.bfloat16:
                            z_2d = z.reshape(num_tokens, num_v_heads_local * self.head_v_dim).contiguous()
                            out_f8, out_sc = torch.ops.custom_op.gdn_gated_rms_norm_f8(
                                out_f32,
                                z_2d,
                                self._norm_weight_f32,
                                float(self.norm.eps),
                                float(hpu_ops.FP8_MAX),
                            )
                            qa = QuantizedActivation(
                                data=out_f8,
                                scale=out_sc,
                                orig_dtype=hidden_states.dtype,
                                orig_shape=(num_tokens, num_v_heads_local * self.head_v_dim),
                                quant_key=kFp8DynamicTokenSym,
                            )
                            output, _ = self.out_proj(qa)
                            return output.view(orig_shape)
                        decode_result = (out_f32.to(hidden_states.dtype).unsqueeze(0), ssm_state)
                if decode_result is None:
                    # Gating, conv and recurrence in one graph on contiguous
                    # group-major state views (no index_select/index_copy).
                    decode_result = maybe_run_gdn_fused_decode_step(
                        mixed_qkv,
                        a,
                        b,
                        self.A_log,
                        self.dt_bias,
                        conv_state,
                        self.conv_taps,
                        self.conv1d.bias,
                        ssm_state,
                        direct_state_group_count=group_count,
                        direct_state_group_offset=group_offset,
                        scale=self.head_k_dim**-0.5,
                        neg_exp_A=self.neg_exp_A,
                        dt_bias_f32=self.dt_bias_f32,
                        conv_bias_f32=self.conv_bias_f32,
                    )
            if decode_result is None:
                g, beta = hpu_fused_gdn_gating(self.A_log, a, b, self.dt_bias)

                mixed_qkv_conv = hpu_causal_conv1d_update(
                    x=mixed_qkv,
                    conv_state=conv_state,
                    weight=conv_weights,
                    bias=self.conv1d.bias,
                    activation=self.activation,
                    conv_state_indices=(state_indices[:num_decodes] if state_indices is not None else state_indices),
                    block_idx_last_scheduled_token=None,
                    initial_state_idx=None,
                    query_start_loc=query_start_loc,
                    validate_data=False,
                    # Read from the load slot, write to the store slot: they differ on
                    # the step whose token opens a new mamba block.
                    store_cache_indices=(store_indices[:num_decodes] if store_indices is not None else None),
                )

                if direct_gdn_state:
                    decode_result = maybe_run_gdn_decode_packed(
                        mixed_qkv_conv,
                        g,
                        beta,
                        ssm_state,
                        use_qk_l2norm=True,
                        scale=self.head_k_dim**-0.5,
                        direct_state_group_count=self.compact_state_group_count,
                        direct_state_group_offset=self.compact_state_group_offset,
                    )
            if decode_result is not None:
                core_attn_out_result = decode_result[0]
            else:
                query, key, value = self.rearrange_mixed_qkv(mixed_qkv_conv)

                if hpu_gdn_tpc.STATE_KV:
                    core_attn_out_result = self._indexed_decode_kv(query, key, value, g, beta, ssm_state,
                                                                   state_indices, store_indices, query_start_loc,
                                                                   num_decodes)
                else:
                    core_attn_out_result, _ = \
                    hpu_fused_recurrent_gated_delta_rule(
                        q=query, k=key, v=value, g=g, beta=beta,
                        initial_state=ssm_state,
                        inplace_final_state=True,
                        cu_seqlens=(
                            query_start_loc[:num_decodes + 1]
                            if query_start_loc is not None else None),
                        ssm_state_indices=state_indices,
                        use_qk_l2norm_in_kernel=True,
                        ssm_store_indices=store_indices,
                    )

            non_spec_out = core_attn_out_result.squeeze(0)
            if non_spec_out.shape[0] == num_tokens:
                core_attn_out = non_spec_out
            else:
                core_attn_out = torch.zeros(out_shape, dtype=hidden_states.dtype, device=hidden_states.device)
                n = min(non_spec_out.shape[0], num_tokens)
                core_attn_out[:n] = non_spec_out[:n]

        # === Part 3: Output Projection ===============================
        # Explicit head dims avoid HPU symbolic-shape inference turning -1 / flatten(-2) into 0 across prefill buckets.
        core_attn_out = core_attn_out.reshape(num_tokens * num_v_heads_local, self.head_v_dim)
        z = z.reshape(num_tokens * num_v_heads_local, self.head_v_dim)
        core_attn_out = self.norm(core_attn_out, z)
        core_attn_out = core_attn_out.reshape(num_tokens, num_v_heads_local * self.head_v_dim)

        output, _ = self.out_proj(core_attn_out)
        # Restore caller's original layout (2-D flat or 3-D [B, L, H]) so
        # the residual add in the decoder layer stays shape-consistent.
        return output.view(orig_shape)


# Replace the class in the upstream modules so that both Qwen3-Next and
# Qwen3.5 model definitions instantiate HPUGatedDeltaNetAttention.
import vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn as _gdn_module  # noqa: E402
import vllm.model_executor.models.qwen3_next as _qwen3_next_module  # noqa: E402
import vllm.model_executor.models.qwen3_5 as _qwen3_5_module  # noqa: E402

_gdn_module.QwenGatedDeltaNetAttention = HPUGatedDeltaNetAttention
_qwen3_next_module.QwenGatedDeltaNetAttention = HPUGatedDeltaNetAttention
_qwen3_5_module.QwenGatedDeltaNetAttention = HPUGatedDeltaNetAttention
