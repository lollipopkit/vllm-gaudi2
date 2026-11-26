# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inkling2 4-stream short convolution on HPU using MambaSpec state cache.

Each layer-pass owns one `Inkling2HPUConvState` registered as a `MambaSpec`
layer with a single contiguous state tensor `[num_slots, W - 1, total_dim]`
holding all four streams `(K, V, attn_out, mlp_out)` packed along the last dim.
Both prefill and decode paths are 100% static-shape and zero-host-sync so they
are `torch.compile` and HPU Graph compatible.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from vllm.config import VllmConfig, get_current_vllm_config
from vllm.distributed import get_tensor_model_parallel_rank
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.model_executor.utils import set_weight_attrs
from vllm.v1.attention.backend import AttentionBackend
from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum
from vllm.v1.attention.selector import get_mamba_attn_backend
from vllm.v1.kv_cache_interface import KVCacheSpec, MambaSpec

from vllm_gaudi.extension.utils import index_copy_indices

from .tpc_ops import has_tpc_sconv

K_STREAM, V_STREAM, ATTN_STREAM, MLP_STREAM = range(4)


class Inkling2HPUConvState(nn.Module, AttentionLayerBase):
    """Per-layer-pass MambaSpec state holder for the 4 Inkling2 short-conv streams."""

    def __init__(
        self,
        *,
        kv_size: int,
        hidden_size: int,
        kernel_size: int,
        prefix: str,
    ) -> None:
        super().__init__()
        self.prefix = prefix
        self.kernel_size = kernel_size
        self.kv_size = kv_size
        self.hidden_size = hidden_size
        self.total_dim = 2 * kv_size + 2 * hidden_size
        self.stream_sizes: tuple[int, int, int, int] = (
            kv_size,
            kv_size,
            hidden_size,
            hidden_size,
        )
        # Bound by HPUModelRunner via bind_kv_cache: list of 1 tensor
        # `[num_slots, kernel_size - 1, total_dim]` in bfloat16.
        self.kv_cache: list[torch.Tensor] = [torch.tensor([])]
        self.cache_group_idx: torch.Tensor | None = None

        vllm_config = get_current_vllm_config()
        self.cache_config = vllm_config.cache_config
        if getattr(self.cache_config, "enable_prefix_caching", False):
            raise ValueError(
                "Inkling2HPUConvState does not yet support prefix caching; "
                "pass `--no-enable-prefix-caching`."
            )
        self._dtype = vllm_config.model_config.dtype
        compilation_config = vllm_config.compilation_config
        if prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        compilation_config.static_forward_context[prefix] = self

    @property
    def mamba_type(self) -> MambaAttentionBackendEnum:
        return MambaAttentionBackendEnum.SHORT_CONV

    def get_attn_backend(self) -> type[AttentionBackend]:
        return get_mamba_attn_backend(self.mamba_type)

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec:
        cache_cfg = vllm_config.cache_config
        attn_bs = cache_cfg.block_size
        mamba_bs = cache_cfg.mamba_block_size
        if mamba_bs is None:
            if cache_cfg.mamba_cache_mode == "align":
                mamba_bs = attn_bs
            else:
                # In non-align mode MambaSpec allocates 1 block per request regardless of
                # sequence length, so block_size must cover max_model_len (aligned to attn_bs).
                max_len = vllm_config.model_config.max_model_len
                mamba_bs = ((max_len + attn_bs - 1) // attn_bs) * attn_bs
        return MambaSpec(
            shapes=((self.kernel_size - 1, self.total_dim),),
            dtypes=(self._dtype,),
            block_size=mamba_bs,
            mamba_type=self.mamba_type,
            mamba_cache_mode=cache_cfg.mamba_cache_mode,
        )

    def _select_indices(self, indices: torch.Tensor | None) -> torch.Tensor | None:
        if indices is None or indices.dim() <= 1:
            return indices
        assert self.cache_group_idx is not None
        return indices.index_select(0, self.cache_group_idx.view(1)).squeeze(0)

    def _state_buffer(self) -> torch.Tensor:
        return self.kv_cache[0] if isinstance(self.kv_cache, (list, tuple)) else self.kv_cache

    def resolve_cache_refs(
        self, attn_metadata
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        """Resolve `(state_buf, load_idx, store_idx)` outside `torch.compile` (zero per-pass guard)."""
        state_buf = self._state_buffer()
        if attn_metadata is None or state_buf.numel() == 0:
            return state_buf, None, None
        raw_load = getattr(attn_metadata, "load_indices_tensor", None)
        raw_store = getattr(attn_metadata, "store_indices_tensor", None)
        cg_int = getattr(self, "_cache_group_int", None)
        if cg_int is None and self.cache_group_idx is not None:
            cg_int = int(self.cache_group_idx.item())
            object.__setattr__(self, "_cache_group_int", cg_int)
        if cg_int is not None:
            load_idx = raw_load[cg_int] if (raw_load is not None and raw_load.ndim > 1) else raw_load
            store_idx = (
                raw_store[cg_int] if (raw_store is not None and raw_store.ndim > 1) else raw_store
            )
        else:
            load_idx = self._select_indices(raw_load)
            store_idx = self._select_indices(raw_store)
        return state_buf, load_idx, store_idx

    @staticmethod
    def load_streams_from_buffers(
        state_buf: torch.Tensor,
        load_idx: torch.Tensor | None,
        store_idx: torch.Tensor | None,
        stream_sizes: tuple[int, int, int, int],
        attn_metadata,
        batch_size: int = 1,
    ) -> tuple[
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None,
        torch.Tensor | None,
        bool,
        torch.Tensor | None,
        torch.Tensor | None,
    ]:
        """Pure-tensor conv stream loader that avoids any module/pass_idx Dynamo guards."""
        if attn_metadata is None or state_buf.numel() == 0:
            return None, None, True, None, None
        num_slots = state_buf.shape[0]
        idx_dtype = torch.int32 if state_buf.device.type == "hpu" else torch.long
        if attn_metadata.is_prompt:
            b = max(1, batch_size)
            pad_mask = getattr(attn_metadata, "padding_mask_flat", None)
            seq_lens = getattr(attn_metadata, "seq_lens_tensor", None)
            if seq_lens is not None:
                seq_lens = seq_lens.reshape(-1)[:b]
            prev_streams = None
            has_init = getattr(attn_metadata, "has_initial_states_p", None)
            if load_idx is not None and has_init is not None:
                safe_load = torch.remainder(load_idx.reshape(-1)[:b].to(idx_dtype), num_slots)
                loaded = state_buf.index_select(0, safe_load)
                init_mask = has_init.reshape(-1)[:b].bool().view(b, 1, 1)
                loaded = torch.where(init_mask, loaded, torch.zeros_like(loaded))
                prev_streams = loaded.split(stream_sizes, dim=-1)
            safe_store = (
                torch.remainder(store_idx.reshape(-1)[:b].to(idx_dtype), num_slots)
                if store_idx is not None
                else None
            )
            return prev_streams, safe_store, True, pad_mask, seq_lens

        assert load_idx is not None
        safe_load = torch.remainder(load_idx.reshape(-1).to(idx_dtype), num_slots)
        safe_store = (
            torch.remainder(store_idx.reshape(-1).to(idx_dtype), num_slots)
            if store_idx is not None
            else safe_load
        )
        prev_streams = state_buf.index_select(0, safe_load).split(stream_sizes, dim=-1)
        return prev_streams, safe_store, False, None, None

    @staticmethod
    def store_streams_to_buffer(
        state_buf: torch.Tensor,
        safe_store: torch.Tensor | None,
        parts: tuple[torch.Tensor, ...],
    ) -> None:
        if safe_store is None:
            return
        state_buf.index_copy_(0, index_copy_indices(safe_store), torch.cat(parts, dim=-1))

    def load_streams(
        self,
        attn_metadata,
        batch_size: int = 1,
        decode_ctx=None,
    ) -> tuple[
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None,
        torch.Tensor | None,
        bool,
        torch.Tensor | None,
        torch.Tensor | None,
    ]:
        """Load all 4 conv streams in one `index_select` per layer.

        Returns `(prev_streams, safe_store, is_prompt, pad_mask, seq_lens)`.
        Pairs with `store_streams(safe_store, parts)` which performs one
        contiguous `index_copy_` at layer end (zero `aten.index_put` splits).
        """
        state_buf = self._state_buffer()
        if attn_metadata is None or state_buf.numel() == 0:
            return None, None, True, None, None
        if isinstance(attn_metadata, dict):
            attn_metadata = attn_metadata.get(
                self.prefix, next(iter(attn_metadata.values()), None)
            )
        if attn_metadata is None:
            return None, None, True, None, None
        num_slots = state_buf.shape[0]

        if not attn_metadata.is_prompt and decode_ctx is not None:
            hoisted_load = getattr(decode_ctx, "conv_load_slots", None)
            hoisted_store = getattr(decode_ctx, "conv_store_slots", None)
            hoisted_n = getattr(decode_ctx, "conv_num_slots", 0)
            if hoisted_load is not None and hoisted_n == num_slots:
                index_dtype = torch.int32 if state_buf.device.type == "hpu" else torch.int64
                if hoisted_load.ndim > 1:
                    cg_int = getattr(self, "_cache_group_int", None)
                    if cg_int is not None:
                        safe_load = hoisted_load[cg_int]
                        safe_store = (
                            hoisted_store[cg_int]
                            if hoisted_store is not None
                            else safe_load.to(index_dtype)
                        )
                    else:
                        assert self.cache_group_idx is not None
                        safe_load = hoisted_load.index_select(0, self.cache_group_idx.view(1)).squeeze(0)
                        safe_store = (
                            hoisted_store.index_select(0, self.cache_group_idx.view(1)).squeeze(0)
                            if hoisted_store is not None
                            else safe_load.to(index_dtype)
                        )
                else:
                    safe_load = hoisted_load
                    safe_store = (
                        hoisted_store if hoisted_store is not None else safe_load.to(index_dtype)
                    )
                prev_streams = state_buf.index_select(0, safe_load).split(self.stream_sizes, dim=-1)
                return prev_streams, safe_store, False, None, None

        load_idx = self._select_indices(getattr(attn_metadata, "load_indices_tensor", None))
        store_idx = self._select_indices(getattr(attn_metadata, "store_indices_tensor", None))

        idx_dtype = torch.int32 if state_buf.device.type == "hpu" else torch.long
        if attn_metadata.is_prompt:
            b = max(1, batch_size)
            pad_mask = getattr(attn_metadata, "padding_mask_flat", None)
            seq_lens = getattr(attn_metadata, "seq_lens_tensor", None)
            if seq_lens is not None:
                seq_lens = seq_lens.reshape(-1)[:b]
            prev_streams = None
            has_init = getattr(attn_metadata, "has_initial_states_p", None)
            if load_idx is not None and has_init is not None:
                safe_load = torch.remainder(load_idx.reshape(-1)[:b].to(idx_dtype), num_slots)
                loaded = state_buf.index_select(0, safe_load)
                init_mask = has_init.reshape(-1)[:b].bool().view(b, 1, 1)
                loaded = torch.where(init_mask, loaded, torch.zeros_like(loaded))
                prev_streams = loaded.split(self.stream_sizes, dim=-1)
            safe_store = (
                torch.remainder(store_idx.reshape(-1)[:b].to(idx_dtype), num_slots)
                if store_idx is not None
                else None
            )
            return prev_streams, safe_store, True, pad_mask, seq_lens

        assert load_idx is not None
        safe_load = torch.remainder(load_idx.reshape(-1).to(idx_dtype), num_slots)
        safe_store = (
            torch.remainder(store_idx.reshape(-1).to(idx_dtype), num_slots)
            if store_idx is not None
            else safe_load
        )
        prev_streams = state_buf.index_select(0, safe_load).split(self.stream_sizes, dim=-1)
        return prev_streams, safe_store, False, None, None

    def get_decode_cache_refs(
        self,
        attn_metadata,
        decode_ctx=None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
        """Return `(state_buf, safe_load_i32, safe_store)` for zero-gather TPC decode.

        Performs zero HPU kernels when `decode_ctx` has hoisted `conv_load_slots` and
        `conv_store_slots`. Returns `None` on prompt steps or non-bf16/empty state.
        """
        state_buf = self._state_buffer()
        if (
            attn_metadata is None
            or state_buf.numel() == 0
            or state_buf.dtype != torch.bfloat16
            or state_buf.ndim != 3
            or state_buf.shape[1] != 3
            or not state_buf.is_contiguous()
        ):
            return None
        if isinstance(attn_metadata, dict):
            attn_metadata = attn_metadata.get(
                self.prefix, next(iter(attn_metadata.values()), None)
            )
        if attn_metadata is None or attn_metadata.is_prompt:
            return None
        num_slots = state_buf.shape[0]
        if decode_ctx is not None:
            hoisted_load = getattr(decode_ctx, "conv_load_slots", None)
            hoisted_store = getattr(decode_ctx, "conv_store_slots", None)
            hoisted_n = getattr(decode_ctx, "conv_num_slots", 0)
            if hoisted_load is not None and hoisted_n == num_slots:
                index_dtype = torch.int32 if state_buf.device.type == "hpu" else torch.int64
                if hoisted_load.ndim > 1:
                    cg_int = getattr(self, "_cache_group_int", None)
                    if cg_int is not None:
                        safe_load = hoisted_load[cg_int]
                        safe_store = (
                            hoisted_store[cg_int]
                            if hoisted_store is not None
                            else safe_load.to(index_dtype)
                        )
                    else:
                        assert self.cache_group_idx is not None
                        safe_load = hoisted_load.index_select(0, self.cache_group_idx.view(1)).squeeze(0)
                        safe_store = (
                            hoisted_store.index_select(0, self.cache_group_idx.view(1)).squeeze(0)
                            if hoisted_store is not None
                            else safe_load.to(index_dtype)
                        )
                else:
                    safe_load = hoisted_load
                    safe_store = (
                        hoisted_store if hoisted_store is not None else safe_load.to(index_dtype)
                    )
                return state_buf, safe_load.to(torch.int32).contiguous(), safe_store

        load_idx = self._select_indices(getattr(attn_metadata, "load_indices_tensor", None))
        store_idx = self._select_indices(getattr(attn_metadata, "store_indices_tensor", None))
        if load_idx is None:
            return None
        idx_dtype = torch.int32 if state_buf.device.type == "hpu" else torch.long
        safe_load = torch.remainder(load_idx.reshape(-1).to(idx_dtype), num_slots)
        safe_store = (
            torch.remainder(store_idx.reshape(-1).to(idx_dtype), num_slots)
            if store_idx is not None
            else safe_load
        )
        return state_buf, safe_load.to(torch.int32).contiguous(), safe_store

    def store_streams(
        self,
        safe_store: torch.Tensor | None,
        parts: tuple[torch.Tensor, ...],
    ) -> None:
        """Write updated conv streams in one fused `index_copy_`."""
        if safe_store is None:
            return
        state_buf = self._state_buffer()
        state_buf.index_copy_(0, index_copy_indices(safe_store), torch.cat(parts, dim=-1))

    def store_packed_streams(
        self,
        safe_store: torch.Tensor | None,
        full_state: torch.Tensor,
    ) -> None:
        """Write pre-packed `[B, 3, total_dim]` conv state without `torch.cat` DmaMemcpy."""
        if safe_store is None:
            return
        state_buf = self._state_buffer()
        state_buf.index_copy_(0, index_copy_indices(safe_store), full_state)


def _causal_conv_prefill_batch(
    x: torch.Tensor,
    init_state: torch.Tensor | None,
    static_weight: torch.Tensor,
    dyn_taps: torch.Tensor | None,
    num_heads: int,
    kernel_size: int,
    seq_lens: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Vectorized zero-sync Inkling2 causal conv on `[B, S, C]`.

    Returns `(y [B, S, C] bf16, new_state [B, W - 1, C] bf16)` where `new_state`
    captures the last `W - 1` real tokens of each row using `seq_lens` (or `S`).
    """
    b, s, c = x.shape
    w = kernel_size
    state_len = w - 1
    if init_state is None:
        pad = torch.zeros((b, state_len, c), dtype=x.dtype, device=x.device)
    else:
        pad = init_state.to(x.dtype)
    hist = torch.cat([pad, x], dim=1)  # [B, S + W - 1, C]

    if seq_lens is None:
        new_state = hist[:, -state_len:, :].clone() if state_len > 0 else hist[:, :0, :]
    else:
        # Real token t sits at hist index (state_len + t); last state_len real tokens
        # for length L sit at hist indices [L, L + 1, ..., L + state_len - 1].
        offs = torch.arange(state_len, device=x.device, dtype=torch.long).unsqueeze(0)
        gather_idx = (seq_lens.long().unsqueeze(1) + offs).clamp(0, s + state_len - 1)
        new_state = hist.gather(1, gather_idx.unsqueeze(-1).expand(-1, -1, c))

    hist_f = hist.float()
    if dyn_taps is None:
        # Static conv: oldest tap first (k=0 -> lag=W-1, k=W-1 -> lag=0).
        acc = x.float()
        sw = static_weight.float()  # [C, W]
        for k in range(w):
            lag = w - 1 - k
            start = state_len - lag
            acc = acc + hist_f[:, start : start + s, :] * sw[:, k].view(1, 1, c)
        y = acc * ((w + 1) ** -0.5)
    else:
        # Dynamic conv: newest tap first (lag=0..W-1).
        head_dim = c // num_heads
        dyn_f = dyn_taps.view(b, s, num_heads, w).float()
        sw = static_weight.float()  # [W, C]
        acc = x.float()
        for lag in range(w):
            start = state_len - lag
            tap_w = sw[lag].view(1, 1, c) + dyn_f[:, :, :, lag].repeat_interleave(head_dim, dim=-1)
            acc = acc + hist_f[:, start : start + s, :] * tap_w
        y = acc * ((2 * w + 1) ** -0.5)
    return y.to(x.dtype), new_state


def _causal_conv_prefill_single(
    x: torch.Tensor,
    init_state: torch.Tensor | None,
    static_weight: torch.Tensor,
    dyn_taps: torch.Tensor | None,
    num_heads: int,
    kernel_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Single-sequence wrapper `[L, C]` -> `(y [L, C], new_state [W - 1, C])`."""
    y, st = _causal_conv_prefill_batch(
        x.unsqueeze(0),
        None if init_state is None else init_state.unsqueeze(0),
        static_weight,
        None if dyn_taps is None else dyn_taps.unsqueeze(0),
        num_heads,
        kernel_size,
    )
    return y.squeeze(0), st.squeeze(0)


def _causal_conv_decode_batch(
    x: torch.Tensor,
    prev_state: torch.Tensor,
    static_weight: torch.Tensor,
    dyn_taps: torch.Tensor | None,
    num_heads: int,
    kernel_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Zero-sync Inkling2 causal conv on a decode batch `[B, C]` with `prev_state [B, W - 1, C]`."""
    b, c = x.shape
    w = kernel_size
    hist = torch.cat([prev_state.to(x.dtype), x.unsqueeze(1)], dim=1)  # [B, W, C]
    new_state = hist[:, 1:, :].clone() if w > 1 else hist[:, :0, :]

    hist_f = hist.float()
    if dyn_taps is None:
        acc = (hist_f * static_weight.float().t().unsqueeze(0)).sum(dim=1) + x.float()
        y = acc * ((w + 1) ** -0.5)
    else:
        # Vectorized dynamic conv: newest tap first (lag 0 -> hist[:, W-1], lag W-1 -> hist[:, 0]).
        head_dim = c // num_heads
        # Reverse hist along tap dim so index `lag` aligns with `static_weight[lag]` and `dyn_f[:, :, lag]`.
        hist_rev = hist_f.flip(1).view(b, w, num_heads, head_dim)
        sw = static_weight.float().view(1, w, num_heads, head_dim)
        dyn_f = dyn_taps.view(b, num_heads, w).float().transpose(1, 2).unsqueeze(-1)
        acc = (hist_rev * (sw + dyn_f)).sum(dim=1).reshape(b, c) + x.float()
        y = acc * ((2 * w + 1) ** -0.5)
    return y.to(x.dtype), new_state


class Inkling2HPUShortConv(nn.Module):
    """Causal depthwise short convolution (+ residual, scaled) on HPU."""

    def __init__(
        self,
        dim: int,
        kernel_size: int,
        owner: Inkling2HPUConvState,
        stream: int,
        *,
        dynamic_input_size: int | None = None,
        num_heads: int = 1,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.kernel_size = kernel_size
        self.owner = owner
        self.stream = stream
        self.dynamic = dynamic_input_size is not None
        self.num_heads = num_heads
        self.tp_rank = get_tensor_model_parallel_rank()
        if dynamic_input_size is not None:
            self.static_w = nn.Parameter(
                torch.empty(kernel_size, dim, dtype=torch.bfloat16), requires_grad=False
            )
            self.weight_proj = nn.Parameter(
                torch.empty(num_heads * kernel_size, dynamic_input_size, dtype=torch.bfloat16),
                requires_grad=False,
            )
            set_weight_attrs(self.static_w, {"weight_loader": self._load_dim1})
            set_weight_attrs(self.weight_proj, {"weight_loader": self._load_dim0})
        else:
            self.weight = nn.Parameter(
                torch.empty(dim, 1, kernel_size, dtype=torch.bfloat16), requires_grad=False
            )
            set_weight_attrs(self.weight, {"weight_loader": self._load_dim0})
        self._weight_t_ready = False

    def process_weights_after_loading(self) -> None:
        if not self.dynamic and not self._weight_t_ready:
            self.register_buffer(
                "_weight_t",
                self.weight.squeeze(1).t().contiguous(),
                persistent=False,
            )
            self._weight_t_ready = True

    def _load(self, param: nn.Parameter, loaded: torch.Tensor, dim: int) -> None:
        size = param.shape[dim]
        if loaded.shape[dim] != size:
            loaded = loaded.narrow(dim, self.tp_rank * size, size)
        param.data.copy_(loaded)

    def _load_dim0(self, param: nn.Parameter, loaded: torch.Tensor) -> None:
        self._load(param, loaded, 0)

    def _load_dim1(self, param: nn.Parameter, loaded: torch.Tensor) -> None:
        self._load(param, loaded, 1)

    @property
    def conv_weight(self) -> torch.Tensor:
        return self.static_w if self.dynamic else self.weight.squeeze(1)

    def dynamic_taps(self, x: torch.Tensor) -> torch.Tensor | None:
        if not self.dynamic:
            return None
        return F.linear(x, self.weight_proj)

    def apply_with_state(
        self,
        x: torch.Tensor,
        prev_state: torch.Tensor | None,
        *,
        dyn_taps: torch.Tensor | None = None,
        is_prompt: bool = False,
        batch_size: int = 1,
        pad_mask: torch.Tensor | None = None,
        seq_lens: torch.Tensor | None = None,
        conv_state: torch.Tensor | None = None,
        load_slots: torch.Tensor | None = None,
        stream_offset: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Pure conv step returning `(y, new_state)` without touching `state_buf`."""
        if is_prompt:
            x_in = x
            if pad_mask is not None and pad_mask.numel() == x.shape[0]:
                x_in = x * pad_mask.view(-1, 1).to(x.dtype)
            b = max(1, batch_size)
            s = x_in.shape[0] // b
            x_3d = x_in.view(b, s, self.dim)
            dyn_3d = None if dyn_taps is None else dyn_taps.view(b, s, -1)
            y_3d, new_state = _causal_conv_prefill_batch(
                x_3d,
                prev_state,
                self.conv_weight,
                dyn_3d,
                self.num_heads,
                self.kernel_size,
                seq_lens=seq_lens,
            )
            y_flat = y_3d.reshape(x.shape)
            if pad_mask is not None and pad_mask.numel() == x.shape[0]:
                y_flat = y_flat * pad_mask.view(-1, 1).to(y_flat.dtype)
            return y_flat, new_state

        if (
            not self.dynamic
            and dyn_taps is None
            and x.device.type == "hpu"
            and x.dtype == torch.bfloat16
            and self.kernel_size == 4
            and (self.dim % 64) == 0
            and has_tpc_sconv()
        ):
            w_t = (
                self._weight_t
                if getattr(self, "_weight_t_ready", False)
                else self.conv_weight.t().contiguous()
            )
            if conv_state is not None and load_slots is not None:
                return torch.ops.custom_op.inkling2_sconv_static_decode(
                    x.contiguous(),
                    conv_state,
                    load_slots.contiguous(),
                    w_t,
                    float((self.kernel_size + 1) ** -0.5),
                    int(stream_offset),
                )
            assert prev_state is not None
            return torch.ops.custom_op.inkling2_sconv_static_decode(
                x.contiguous(),
                prev_state.to(torch.bfloat16).contiguous(),
                torch.arange(x.shape[0], device=x.device, dtype=torch.int32),
                w_t,
                float((self.kernel_size + 1) ** -0.5),
                0,
            )
        assert prev_state is not None
        return _causal_conv_decode_batch(
            x, prev_state, self.conv_weight, dyn_taps, self.num_heads, self.kernel_size
        )

    def forward(
        self,
        x: torch.Tensor,
        *,
        dyn_taps: torch.Tensor | None = None,
        owner: Inkling2HPUConvState | None = None,
        batch_size: int = 1,
    ) -> torch.Tensor:
        """Standalone conv entry point (delegates to fused `load_streams`/`store_streams`)."""
        if owner is None:
            owner = self.owner
        attn_metadata = get_forward_context().attn_metadata
        prev_streams, safe_store, is_prompt, pad_mask, seq_lens = owner.load_streams(
            attn_metadata, batch_size=batch_size
        )
        prev_st = None if prev_streams is None else prev_streams[self.stream]
        y, new_st = self.apply_with_state(
            x,
            prev_st,
            dyn_taps=dyn_taps,
            is_prompt=is_prompt,
            batch_size=batch_size,
            pad_mask=pad_mask,
            seq_lens=seq_lens,
        )
        if safe_store is not None:
            state_buf = owner._state_buffer()
            cur_parts = list(state_buf.index_select(0, safe_store).split(owner.stream_sizes, dim=-1))
            cur_parts[self.stream] = new_st
            owner.store_streams(
                safe_store, (cur_parts[0], cur_parts[1], cur_parts[2], cur_parts[3])
            )
        return y
