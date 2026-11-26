"""LFM2 / LFM3 gated short convolution on HPU.

Upstream ``ShortConv.forward`` dispatches to CUDA / Triton causal conv kernels through a custom op; this
replacement drives the HPU causal conv1d (the same calls as the GDN layer in models/qwen3_5.py) from
``HPUAttentionMetadataV1``: padded prefill buckets with ``query_start_loc_p`` / ``padding_mask_flat``, and
load / store state slots selected per KV cache group.

For single-token decode steps, ``_decode_fast`` runs directly on the native ``[slots, state_len, dim]``
cache layout using pre-transposed contiguous FP32 ``conv_taps`` (``[kernel, dim]``), avoiding two
``[B, state_len, dim] <-> [B, dim, state_len]`` DMA transposes, ``kernel`` strided column slices, and a
dead ``zeros_like`` allocation on every layer while keeping bit-identical FP32 tap accumulation.
"""

import os

import torch
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.mamba.short_conv import ShortConv

from vllm_gaudi.ops.causal_conv1d_pytorch import hpu_causal_conv1d_fn, hpu_causal_conv1d_update
from vllm_gaudi.ops.hpu_gaudi2_fp8_tpc import has_lfm3_short_conv_decode

_FAST_SHORT_CONV_DECODE = os.environ.get("VLLM_HPU_SHORT_CONV_FAST_DECODE", "1") != "0"


@ShortConv.register_oot
class HPUShortConv(ShortConv):
    _returns_output = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 0-dim tensor set by maybe_set_mamba_kv_cache_groups_ids (see hpu_mamba_mixer2 for why not an int).
        self.cache_group_idx: torch.Tensor | None = None
        # Populated only when decode runs exclusively through per-group code objects (see HpuLfm3Model).
        self._decode_group_int: int | None = None
        self.conv_taps: torch.Tensor | None = None
        self.conv_bias_f32: torch.Tensor | None = None
        self._conv_bias_dummy: torch.Tensor | None = None
        self._use_tpc_short_conv: bool = False

    def prepare_hpu_constants(self) -> None:
        if not _FAST_SHORT_CONV_DECODE:
            return
        w = self.conv.weight.view(self.conv.weight.size(0), self.conv.weight.size(2))
        self.conv_taps = w.t().to(torch.float32).contiguous()
        if self.conv.bias is not None:
            self.conv_bias_f32 = self.conv.bias.to(torch.float32).contiguous()
        self._use_tpc_short_conv = (
            has_lfm3_short_conv_decode()
            and self.conv_taps.shape[0] > 1
            and (self.conv_taps.shape[1] % 64) == 0
        )
        if self._use_tpc_short_conv and self.conv_bias_f32 is None:
            self._conv_bias_dummy = torch.zeros(
                self.conv_taps.shape[1], dtype=torch.float32, device=self.conv_taps.device
            )

    def _select_cache_group(self, indices, *, is_decode: bool = False):
        if indices is None or indices.dim() <= 1:
            return indices
        if is_decode and self._decode_group_int is not None:
            return indices[self._decode_group_int]
        assert self.cache_group_idx is not None
        return indices.index_select(0, self.cache_group_idx.view(1)).squeeze(0)

    def _decode_fast(
        self,
        bx: torch.Tensor,
        conv_state: torch.Tensor,
        load_idx: torch.Tensor,
        store_idx: torch.Tensor | None,
    ) -> torch.Tensor:
        taps = self.conv_taps
        assert taps is not None
        state_len = taps.shape[0] - 1
        num_slots = conv_state.shape[0]
        # Route padding (-1) to the trailing garbage slot via remainder (torch.where on ints miscompiles on HPU).
        safe_load = torch.remainder(load_idx.reshape(-1), num_slots)
        safe_store = (safe_load if store_idx is None else torch.remainder(store_idx.reshape(-1), num_slots))

        bx_work = bx.to(conv_state.dtype)
        init_state = (conv_state[safe_load]
                      if conv_state.shape[1] == state_len else conv_state[safe_load, -state_len:, :])

        acc = init_state[:, 0, :] * taps[0]
        for k in range(1, state_len):
            acc = acc + init_state[:, k, :] * taps[k]
        acc = acc + bx_work * taps[state_len]
        if self.conv_bias_f32 is not None:
            acc = acc + self.conv_bias_f32

        new_state = torch.cat((init_state[:, 1:, :], bx_work.unsqueeze(1)), dim=1)
        with torch.no_grad():
            if conv_state.shape[1] == state_len:
                conv_state[safe_store] = new_state
            else:
                conv_state[safe_store, -state_len:, :] = new_state
        return acc.to(bx.dtype)

    def forward(self, hidden_states: torch.Tensor, output: torch.Tensor | None = None) -> torch.Tensor:
        # The HPU runner feeds padded [batch, seq, hidden] buckets; the conv works on flat tokens.
        orig_shape = hidden_states.shape
        hidden_states = hidden_states.view(-1, orig_shape[-1])
        num_tokens = hidden_states.shape[0]
        attn_metadata = get_forward_context().attn_metadata
        BCx, _ = self.in_proj(hidden_states)

        if attn_metadata is None:  # profile run
            B, C, x = BCx.chunk(3, dim=-1)
            y = C * (B * x)
        elif attn_metadata.is_prompt:
            B, C, x = BCx.chunk(3, dim=-1)
            Bx = B * x
            padding_mask = getattr(attn_metadata, "padding_mask_flat", None)
            if padding_mask is not None and padding_mask.numel() == num_tokens:
                Bx = Bx * padding_mask.view(-1, 1).to(Bx.dtype)
            conv_weights = self.conv.weight.view(self.conv.weight.size(0), self.conv.weight.size(2))
            conv_out = hpu_causal_conv1d_fn(
                x=Bx.transpose(0, 1),
                weight=conv_weights,
                bias=self.conv.bias,
                activation=None,
                conv_states=self.kv_cache[0],
                has_initial_state=getattr(attn_metadata, "has_initial_states_p", None),
                cache_indices=self._select_cache_group(attn_metadata.store_indices_tensor),
                initial_state_idx=self._select_cache_group(attn_metadata.load_indices_tensor),
                query_start_loc=attn_metadata.query_start_loc_p,
                block_size_to_align=self.cache_config.mamba_block_size,
                is_prompt=True,
            ).transpose(0, 1)
            y = C * conv_out
        else:
            load_raw = attn_metadata.load_indices_tensor
            store_raw = attn_metadata.store_indices_tensor
            load = self._select_cache_group(load_raw, is_decode=True)
            same_slot = store_raw is None or store_raw is load_raw
            store = None if same_slot else self._select_cache_group(store_raw, is_decode=True)
            num_decodes = load.numel()
            conv_state = self.kv_cache[0]
            if (self._use_tpc_short_conv and self.conv_taps is not None and num_tokens > 0
                    and num_tokens == num_decodes and BCx.dtype == torch.bfloat16 and conv_state.ndim == 3
                    and conv_state.dtype == torch.bfloat16 and conv_state.is_contiguous()
                    and conv_state.shape[1] >= self.conv_taps.shape[0] - 1
                    and conv_state.shape[2] == self.conv_taps.shape[1]):
                num_slots = conv_state.shape[0]
                state_len = self.conv_taps.shape[0] - 1
                # Route negative padding indices to the trailing garbage slot via remainder, matching _decode_fast.
                safe_load = torch.remainder(load[:num_decodes].reshape(-1), num_slots)
                safe_store = (safe_load if store is None else torch.remainder(store[:num_decodes].reshape(-1), num_slots))
                load_i32 = safe_load.to(torch.int32).contiguous()
                has_bias = 1 if self.conv_bias_f32 is not None else 0
                bias_t = self.conv_bias_f32 if has_bias else self._conv_bias_dummy
                y, new_state = torch.ops.custom_op.lfm3_short_conv_decode(
                    BCx.contiguous(),
                    conv_state,
                    load_i32,
                    self.conv_taps,
                    bias_t,
                    has_bias,
                )
                with torch.no_grad():
                    if conv_state.shape[1] == state_len:
                        conv_state[safe_store] = new_state
                    else:
                        conv_state[safe_store, -state_len:, :] = new_state
            else:
                B, C, x = BCx.chunk(3, dim=-1)
                Bx = B * x
                if self.conv_taps is not None and self.conv_taps.shape[0] > 1 and num_tokens == num_decodes:
                    conv_out = self._decode_fast(
                        Bx,
                        conv_state,
                        load[:num_decodes],
                        store[:num_decodes] if store is not None else None,
                    )
                else:
                    conv_weights = self.conv.weight.view(self.conv.weight.size(0), self.conv.weight.size(2))
                    store_fallback = load[:num_decodes] if store_raw is load_raw else (
                        store[:num_decodes] if store is not None else None)
                    conv_out = hpu_causal_conv1d_update(
                        x=Bx,
                        conv_state=conv_state,
                        weight=conv_weights,
                        bias=self.conv.bias,
                        activation=None,
                        conv_state_indices=load[:num_decodes],
                        query_start_loc=attn_metadata.query_start_loc_p,
                        store_cache_indices=store_fallback,
                    )
                y = C * conv_out
        out, _ = self.out_proj(y)
        out = out.view(orig_shape)
        if output is not None:
            output.copy_(out)
        return out
