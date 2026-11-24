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

_FAST_SHORT_CONV_DECODE = os.environ.get("VLLM_HPU_SHORT_CONV_FAST_DECODE", "1") != "0"


@ShortConv.register_oot
class HPUShortConv(ShortConv):
    _returns_output = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 0-dim tensor set by maybe_set_mamba_kv_cache_groups_ids (see hpu_mamba_mixer2 for why not an int).
        self.cache_group_idx: torch.Tensor | None = None
        self.conv_taps: torch.Tensor | None = None
        self.conv_bias_f32: torch.Tensor | None = None

    def prepare_hpu_constants(self) -> None:
        if not _FAST_SHORT_CONV_DECODE:
            return
        w = self.conv.weight.view(self.conv.weight.size(0), self.conv.weight.size(2))
        self.conv_taps = w.t().to(torch.float32).contiguous()
        if self.conv.bias is not None:
            self.conv_bias_f32 = self.conv.bias.to(torch.float32).contiguous()

    def _select_cache_group(self, indices):
        if indices is None or indices.dim() <= 1:
            return indices
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
        B, C, x = BCx.chunk(3, dim=-1)
        Bx = B * x

        if attn_metadata is None:  # profile run
            y = C * Bx
        elif attn_metadata.is_prompt:
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
            load = self._select_cache_group(load_raw)
            same_slot = store_raw is None or store_raw is load_raw
            store = None if same_slot else self._select_cache_group(store_raw)
            num_decodes = load.numel()
            if self.conv_taps is not None and self.conv_taps.shape[0] > 1 and num_tokens == num_decodes:
                conv_out = self._decode_fast(
                    Bx,
                    self.kv_cache[0],
                    load[:num_decodes],
                    store[:num_decodes] if store is not None else None,
                )
            else:
                conv_weights = self.conv.weight.view(self.conv.weight.size(0), self.conv.weight.size(2))
                store_fallback = load[:num_decodes] if store_raw is load_raw else (
                    store[:num_decodes] if store is not None else None)
                conv_out = hpu_causal_conv1d_update(
                    x=Bx,
                    conv_state=self.kv_cache[0],
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
