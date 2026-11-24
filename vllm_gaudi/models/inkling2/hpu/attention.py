# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inkling2 relative-bias GQA attention on Gaudi2 HPU.

Zero-host-sync, static-shape prefill and paged decode paths compatible with
`torch.compile` and HPU Graph capture:
  1. Q / R / gate / sink / K / V projection (merged or split for full/loop layers).
  2. 4-stream short-conv on K and V (static for local layers, dynamic for full layers).
  3. L2-clipped Q, R (scaled by q_scale/r_scale and optional log-scaling tau),
     L2-clipped K, and weightless RMSNorm V.
  4. Relative-position logits `rel = r @ rel_logits_proj` added to scaled QK^T
     with causal + optional sliding-window mask, returning both output and LSE.
  5. Per-token gated sink scaling `out * sigmoid(gate) * sigmoid(lse - sinks)`
     followed by `wo_ud` output projection.
"""

from __future__ import annotations

import os
from typing import NamedTuple

import torch
import torch.nn.functional as F
from torch import nn

from vllm.config import VllmConfig, get_current_vllm_config
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.utils import set_weight_attrs
from vllm.utils.torch_utils import kv_cache_dtype_str_to_dtype
from vllm.v1.attention.backend import AttentionBackend
from vllm.v1.attention.selector import get_attn_backend
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheSpec,
)

from ..configs import Inkling2ModelConfig
from .ops import GAUDI2_FP8_MAX, fp8_linear, gate_sink, l2_clip, quantize_fp8_hpu, weightless_rms_norm
from .sconv import Inkling2HPUConvState, Inkling2HPUShortConv, K_STREAM, V_STREAM
from .tpc_ops import has_tpc_paged_attn, has_tpc_qkvr_prep


def _make_kv_cache_spec(
    vllm_config: VllmConfig,
    num_kv_heads: int,
    head_dim: int,
    sliding_window: int | None,
) -> KVCacheSpec:
    # HPUModelRunner.initialize_kv_cache allocates FullAttentionSpec for all
    # attention layers; sliding-window masking is enforced inside attention.
    del sliding_window
    dtype = kv_cache_dtype_str_to_dtype(
        vllm_config.cache_config.cache_dtype, vllm_config.model_config
    )
    block_size = vllm_config.cache_config.block_size
    return FullAttentionSpec(
        block_size=block_size,
        num_kv_heads=num_kv_heads,
        head_size=head_dim,
        dtype=dtype,
    )


class DecodeAttnContext(NamedTuple):
    """Per-step decode attention metadata hoisted once across all 48 layer visits."""

    safe_grp: torch.Tensor | None  # [NB] long (None when TPC paged attn handles all layers)
    valid_slot_global: torch.Tensor | None  # [NB, 1, BS] bool
    valid_slot_local: torch.Tensor | None  # [NB, 1, BS] bool
    clamped_global: torch.Tensor | None  # [NB * BS] long
    clamped_local: torch.Tensor | None  # [NB * BS] long
    grp_mask: torch.Tensor | None  # [NB, B, 1] bool
    grp_w_t: torch.Tensor | None  # [B, NB] fp32
    neg_large: torch.Tensor | None  # [] fp32 (-1e30)
    block_groups_i32: torch.Tensor  # [NB] int32
    block_usage_i32: torch.Tensor  # [NB] int32
    block_start_pos_i32: torch.Tensor  # [NB] int32
    positions_i32: torch.Tensor  # [B] int32
    block_list_i32: torch.Tensor | None = None  # [NB] int32
    kv_slots: torch.Tensor | None = None  # [B] int32 on HPU, int64 on CPU
    kv_num_slots: int = 0
    conv_load_slots: torch.Tensor | None = None  # [G, B] or [B] int32
    conv_store_slots: torch.Tensor | None = None  # [G, B] or [B] int32 on HPU, int64 on CPU
    conv_num_slots: int = 0


def compute_decode_attn_context_tensors(
    block_groups: torch.Tensor,
    block_usage: torch.Tensor,
    positions: torch.Tensor,
    batch_size: int,
    *,
    block_size: int,
    sliding_window: int,
    rel_extent: int,
    block_logical_idx: torch.Tensor | None = None,
    block_list: torch.Tensor | None = None,
    slot_mapping: torch.Tensor | None = None,
    kv_num_slots: int = 0,
    load_indices: torch.Tensor | None = None,
    store_indices: torch.Tensor | None = None,
    conv_num_slots: int = 0,
    tpc_only: bool = False,
) -> DecodeAttnContext:
    """Pure-tensor builder for `DecodeAttnContext` (compile-friendly, zero sync).

    When `block_logical_idx` (`[NB]`) is provided by `HPUModelRunner`, uses it
    directly so block start positions stay exact even if physical blocks are
    reordered (e.g. under `use_contiguous_pa=True`). Otherwise falls back to
    per-group ascending order (valid for non-contiguous PA).
    """
    device = positions.device
    if block_logical_idx is not None:
        block_start_pos = block_logical_idx.long() * block_size
    else:
        nb = block_groups.shape[0]
        same = block_groups.unsqueeze(1) == block_groups.unsqueeze(0)
        strict_lower = torch.tril(
            torch.ones((nb, nb), dtype=torch.bool, device=device),
            diagonal=-1,
        )
        block_start_pos = (same & strict_lower).sum(dim=1, dtype=torch.long) * block_size

    index_copy_dtype = torch.int32 if device.type == "hpu" else torch.int64
    kv_slots = (
        torch.remainder(slot_mapping.reshape(-1).long(), kv_num_slots).to(index_copy_dtype)
        if slot_mapping is not None and kv_num_slots > 0
        else None
    )
    conv_load_slots_i64 = (
        torch.remainder(load_indices.long(), conv_num_slots)
        if load_indices is not None and conv_num_slots > 0
        else None
    )
    conv_load_slots = None if conv_load_slots_i64 is None else conv_load_slots_i64.to(torch.int32)
    conv_store_slots = (
        torch.remainder(store_indices.long(), conv_num_slots).to(index_copy_dtype)
        if store_indices is not None and conv_num_slots > 0
        else (None if conv_load_slots_i64 is None else conv_load_slots_i64.to(index_copy_dtype))
    )

    bg_i32 = block_groups.to(torch.int32).contiguous()
    bu_i32 = block_usage.to(torch.int32).contiguous()
    bsp_i32 = block_start_pos.to(torch.int32).contiguous()
    pos_i32 = positions.to(torch.int32).contiguous()
    bl_i32 = None if block_list is None else block_list.to(torch.int32).contiguous()

    if tpc_only:
        return DecodeAttnContext(
            safe_grp=None,
            valid_slot_global=None,
            valid_slot_local=None,
            clamped_global=None,
            clamped_local=None,
            grp_mask=None,
            grp_w_t=None,
            neg_large=None,
            block_groups_i32=bg_i32,
            block_usage_i32=bu_i32,
            block_start_pos_i32=bsp_i32,
            positions_i32=pos_i32,
            block_list_i32=bl_i32,
            kv_slots=kv_slots,
            kv_num_slots=kv_num_slots,
            conv_load_slots=conv_load_slots,
            conv_store_slots=conv_store_slots,
            conv_num_slots=conv_num_slots,
        )

    valid_blk = block_groups >= 0  # [NB]
    safe_grp = torch.remainder(block_groups.long(), batch_size)  # [NB]
    slot_off = torch.arange(block_size, device=device, dtype=torch.long)
    k_pos = block_start_pos.unsqueeze(1) + slot_off.unsqueeze(0)  # [NB, BS]
    q_pos = positions.index_select(0, safe_grp).long().unsqueeze(1)  # [NB, 1]
    dist = q_pos - k_pos  # [NB, BS]

    valid_global = (
        valid_blk.unsqueeze(1)
        & (slot_off.unsqueeze(0) < block_usage.long().unsqueeze(1))
        & (dist >= 0)
    )
    valid_local = valid_global & (dist < sliding_window)

    clamped_global = dist.clamp(0, rel_extent - 1).reshape(-1)
    clamped_local = dist.clamp(0, sliding_window - 1).reshape(-1)

    req_ids = torch.arange(batch_size, device=device, dtype=block_groups.dtype)
    grp_mask_2d = block_groups.unsqueeze(1) == req_ids.unsqueeze(0)  # [NB, B]
    neg_large = torch.full((), -1e30, dtype=torch.float32, device=device)
    return DecodeAttnContext(
        safe_grp=safe_grp,
        valid_slot_global=valid_global.unsqueeze(1),
        valid_slot_local=valid_local.unsqueeze(1),
        clamped_global=clamped_global,
        clamped_local=clamped_local,
        grp_mask=grp_mask_2d.unsqueeze(-1),
        grp_w_t=grp_mask_2d.float().t().contiguous(),
        neg_large=neg_large,
        block_groups_i32=bg_i32,
        block_usage_i32=bu_i32,
        block_start_pos_i32=bsp_i32,
        positions_i32=pos_i32,
        block_list_i32=bl_i32,
        kv_slots=kv_slots,
        kv_num_slots=kv_num_slots,
        conv_load_slots=conv_load_slots,
        conv_store_slots=conv_store_slots,
        conv_num_slots=conv_num_slots,
    )


def build_decode_attn_context(
    attn_metadata,
    positions: torch.Tensor,
    batch_size: int,
    *,
    sliding_window: int,
    rel_extent: int,
) -> DecodeAttnContext | None:
    """Precompute shared decode index/mask tensors once per step (zero host sync)."""
    if attn_metadata is None or getattr(attn_metadata, "is_prompt", True):
        return None
    block_groups = getattr(attn_metadata, "block_groups", None)
    block_usage = getattr(attn_metadata, "block_usage", None)
    if block_groups is None or block_usage is None:
        return None
    block_size = int(getattr(attn_metadata, "block_size", 128))
    block_logical_idx = getattr(attn_metadata, "block_logical_idx", None)
    block_list = getattr(attn_metadata, "block_list", None)
    return compute_decode_attn_context_tensors(
        block_groups,
        block_usage,
        positions,
        batch_size,
        block_size=block_size,
        sliding_window=sliding_window,
        rel_extent=rel_extent,
        block_logical_idx=block_logical_idx,
        block_list=block_list,
    )


def _sdpa_with_rel_bias_and_lse(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    rel_logits: torch.Tensor,
    q_positions: torch.Tensor,
    k_positions: torch.Tensor,
    *,
    scale: float,
    sliding_window: int | None,
    enforce_index_causal: bool = False,
    k_valid_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Exact relative-bias GQA for one sequence returning `(out [Lq, H, D], lse [Lq, H])`."""
    lq, h, d = q.shape
    lk, h_kv, _ = k.shape
    extent = rel_logits.shape[-1]
    groups = h // h_kv

    # Grouped BF16 MME matmul: [H_kv, G, Lq, D] @ [H_kv, 1, D, Lk] -> [H, Lq, Lk] fp32
    q_g = q.view(lq, h_kv, groups, d).permute(1, 2, 0, 3)
    k_g = k.permute(1, 2, 0).unsqueeze(1)  # [H_kv, 1, D, Lk]
    scores = torch.matmul(q_g, k_g).float().reshape(h, lq, lk) * scale

    dist = q_positions.unsqueeze(1) - k_positions.unsqueeze(0)  # [Lq, Lk]
    invalid = dist < 0
    if sliding_window is not None:
        invalid = invalid | (dist >= sliding_window)
    if enforce_index_causal and lq == lk:
        idx = torch.arange(lq, device=q.device, dtype=torch.long)
        invalid = invalid | (idx.unsqueeze(0) > idx.unsqueeze(1))
    if k_valid_mask is not None:
        invalid = invalid | (~k_valid_mask.unsqueeze(0))

    clamped = dist.clamp(0, extent - 1)
    rel_h = rel_logits.transpose(0, 1).float()  # [H, Lq, E]
    bias = rel_h.gather(-1, clamped.unsqueeze(0).expand(h, -1, -1))
    scores = scores + bias

    neg_large = torch.full((), -1e30, dtype=torch.float32, device=q.device)
    scores_masked = torch.where(invalid.unsqueeze(0), neg_large, scores)
    m = scores_masked.amax(dim=-1, keepdim=True)  # [H, Lq, 1]
    exp_s = torch.exp((scores_masked - m).clamp(min=-80.0, max=0.0)) * (~invalid).unsqueeze(0).float()
    l = exp_s.sum(dim=-1)  # [H, Lq]
    probs = exp_s / l.clamp(min=1e-30).unsqueeze(-1)
    lse = m.squeeze(-1) + torch.log(l.clamp(min=1e-30))

    v_g = v.transpose(0, 1).unsqueeze(1)  # [H_kv, 1, Lk, D]
    out = torch.matmul(probs.to(v.dtype).view(h_kv, groups, lq, lk), v_g).float().reshape(h, lq, d)
    return out.transpose(0, 1), lse.transpose(0, 1)


def _write_kv_to_paged_cache(
    k: torch.Tensor,
    v: torch.Tensor,
    kv_cache: torch.Tensor | tuple[torch.Tensor, torch.Tensor] | list[torch.Tensor],
    slot_mapping: torch.Tensor,
    *,
    precomputed_slots: torch.Tensor | None = None,
    precomputed_num_slots: int = 0,
) -> None:
    """Zero-sync write of `k, v` `[T, H_kv, D]` into HPU paged KV cache."""
    key_cache, value_cache = kv_cache[0], kv_cache[1]

    if key_cache.ndim == 3:
        # HPU native 3D layout: [num_blocks * block_size, H_kv, D].
        # Route any negative pad slots to the last dummy slot via torch.remainder.
        if precomputed_slots is not None and precomputed_num_slots == key_cache.shape[0]:
            slots = precomputed_slots
        else:
            slots = torch.remainder(slot_mapping.reshape(-1).long(), key_cache.shape[0])
            if key_cache.device.type == "hpu":
                slots = slots.to(torch.int32)
        key_cache.index_copy_(0, slots, k.to(key_cache.dtype))
        value_cache.index_copy_(0, slots, v.to(value_cache.dtype))
        return

    # 4D fallback (unit tests).
    flat_slots = slot_mapping.reshape(-1).long()
    valid = flat_slots >= 0
    slots = flat_slots[valid]
    k_valid = k[valid].to(key_cache.dtype)
    v_valid = v[valid].to(value_cache.dtype)
    if key_cache.shape[2] == k.shape[1]:
        block_size = key_cache.shape[1]
        key_cache[slots // block_size, slots % block_size] = k_valid
        value_cache[slots // block_size, slots % block_size] = v_valid
    else:
        block_size = key_cache.shape[2]
        key_cache[slots // block_size, :, slots % block_size, :] = k_valid
        value_cache[slots // block_size, :, slots % block_size, :] = v_valid


def _gather_paged_kv(
    kv_cache: torch.Tensor | tuple[torch.Tensor, torch.Tensor] | list[torch.Tensor],
    block_ids: torch.Tensor,
    seq_len: int,
    num_kv_heads: int,
    head_dim: int,
    block_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gather `[seq_len, H_kv, D]` keys and values from paged cache (chunked prefill fallback)."""
    key_cache, value_cache = kv_cache[0], kv_cache[1]
    num_blks = (seq_len + block_size - 1) // block_size
    blks = block_ids[:num_blks].long()

    if key_cache.ndim == 3:
        k_4d = key_cache.view(-1, block_size, num_kv_heads, head_dim)
        v_4d = value_cache.view(-1, block_size, num_kv_heads, head_dim)
        k = k_4d.index_select(0, blks).reshape(-1, num_kv_heads, head_dim)[:seq_len]
        v = v_4d.index_select(0, blks).reshape(-1, num_kv_heads, head_dim)[:seq_len]
    elif key_cache.ndim == 4 and key_cache.shape[2] == num_kv_heads:
        k = key_cache.index_select(0, blks).reshape(-1, num_kv_heads, head_dim)[:seq_len]
        v = value_cache.index_select(0, blks).reshape(-1, num_kv_heads, head_dim)[:seq_len]
    else:
        k = key_cache.index_select(0, blks).transpose(1, 2).reshape(-1, num_kv_heads, head_dim)[:seq_len]
        v = value_cache.index_select(0, blks).transpose(1, 2).reshape(-1, num_kv_heads, head_dim)[:seq_len]
    return k, v


class Inkling2HPUAttentionPassCache(nn.Module, AttentionLayerBase):
    """Per-pass KV cache registration for loop passes > 0 on HPU."""

    def __init__(
        self,
        prefix: str,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        sliding_window: int | None,
    ) -> None:
        super().__init__()
        self.prefix = prefix
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.sliding_window = sliding_window
        self.kv_cache: list[torch.Tensor] = [torch.tensor([])]

        vllm_config = get_current_vllm_config()
        self._attn_backend = get_attn_backend(
            head_dim,
            vllm_config.model_config.dtype,
            vllm_config.cache_config.cache_dtype,
            vllm_config.cache_config.block_size,
        )
        fc = vllm_config.compilation_config.static_forward_context
        if prefix in fc:
            raise ValueError(f"Duplicate layer name: {prefix}")
        fc[prefix] = self

    def get_attn_backend(self) -> type[AttentionBackend]:
        return self._attn_backend

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec:
        return _make_kv_cache_spec(
            vllm_config, self.num_kv_heads, self.head_dim, self.sliding_window
        )


class Inkling2HPUAttention(nn.Module, AttentionLayerBase):
    """Inkling2 relative-bias GQA attention layer on Gaudi2 HPU."""

    def __init__(
        self,
        config: Inkling2ModelConfig,
        is_local: bool,
        conv_state: Inkling2HPUConvState,
        prefix: str,
        pass_conv_states: tuple[Inkling2HPUConvState, ...] = (),
    ) -> None:
        super().__init__()
        self.prefix = prefix
        self.is_local = is_local
        num_heads = config.swa_num_attention_heads if is_local else config.num_attention_heads
        num_kv_heads = config.swa_num_key_value_heads if is_local else config.num_key_value_heads
        self.head_dim = config.swa_head_dim if is_local else config.head_dim
        self.d_rel = config.d_rel
        tp_size = get_tensor_model_parallel_world_size()
        tp_rank = get_tensor_model_parallel_rank()
        assert num_heads % tp_size == 0 and num_kv_heads % tp_size == 0
        self.num_heads = num_heads // tp_size
        self.num_kv_heads = num_kv_heads // tp_size
        self.scaling = self.head_dim**-0.5
        self.sliding_window = config.sliding_window_size if is_local else None
        self.rel_extent = config.sliding_window_size if is_local else config.rel_extent
        self.qk_clip = float(config.qk_l2_clip_threshold)
        self.rel_clip = float(config.rel_l2_clip_threshold)
        self.eps = float(config.rms_norm_eps)
        assert config.use_value_norm_no_gamma

        q_size = self.num_heads * self.head_dim
        kv_size = self.num_kv_heads * self.head_dim
        self.split_sizes = [
            q_size,
            self.num_heads * self.d_rel,
            q_size,
            self.num_heads,
            kv_size,
            kv_size,
        ]
        self.q_side_size = sum(self.split_sizes[:4])

        self.qkvr = MergedColumnParallelLinear(
            config.hidden_size,
            [size * tp_size for size in self.split_sizes],
            bias=False,
            prefix=f"{prefix}.qkvr",
        )
        self.wo_ud = RowParallelLinear(
            q_size * tp_size,
            config.hidden_size,
            bias=False,
            reduce_results=False,
            prefix=f"{prefix}.wo_ud",
        )
        self.rel_logits_proj = nn.Parameter(
            torch.empty(self.d_rel, self.rel_extent, dtype=torch.bfloat16),
            requires_grad=False,
        )
        self.register_buffer(
            "rel_logits_proj_t",
            torch.empty(self.rel_extent, self.d_rel, dtype=torch.bfloat16),
            persistent=False,
        )
        self.q_scale = nn.Parameter(
            torch.empty(self.num_heads, self.head_dim, dtype=torch.bfloat16),
            requires_grad=False,
        )
        self.r_scale = nn.Parameter(
            torch.empty(self.num_heads, self.d_rel, dtype=torch.bfloat16),
            requires_grad=False,
        )
        for param in (self.q_scale, self.r_scale):
            set_weight_attrs(
                param,
                {
                    "weight_loader": lambda p, w: p.data.copy_(
                        w.narrow(0, tp_rank * p.shape[0], p.shape[0])
                    )
                },
            )

        self.conv_state = conv_state
        self.conv_states = (conv_state, *pass_conv_states)
        self.pass_caches = nn.ModuleDict(
            {
                str(p - 1): Inkling2HPUAttentionPassCache(
                    f"{prefix}.pass{p}",
                    self.num_heads,
                    self.num_kv_heads,
                    self.head_dim,
                    self.sliding_window,
                )
                for p in range(1, 1 if config.loop_share_cache else len(self.conv_states))
            }
        )
        self.k_sconv, self.v_sconv = (
            Inkling2HPUShortConv(
                kv_size,
                config.sconv_kernel_size,
                conv_state,
                stream,
                dynamic_input_size=None if is_local else config.hidden_size,
                num_heads=self.num_kv_heads,
            )
            for stream in (K_STREAM, V_STREAM)
        )

        vllm_config = get_current_vllm_config()
        self._attn_backend = get_attn_backend(
            self.head_dim,
            vllm_config.model_config.dtype,
            vllm_config.cache_config.cache_dtype,
            vllm_config.cache_config.block_size,
        )
        self.kv_cache: list[torch.Tensor] = [torch.tensor([])]
        self._use_fp8_linear = False
        self._rel_proj_t_ready = False
        self._kvw_t_ready = False
        self.fp8_max = GAUDI2_FP8_MAX
        fc = vllm_config.compilation_config.static_forward_context
        if prefix in fc:
            raise ValueError(f"Duplicate layer name: {prefix}")
        fc[prefix] = self

    def process_weights_after_loading(self) -> None:
        """Prepare transposed rel-bias / sconv weights and quantize BF16 qkvr/wo_ud to FP8@240."""
        self.rel_logits_proj_t.copy_(self.rel_logits_proj.t().contiguous())
        self._rel_proj_t_ready = True
        self.k_sconv.process_weights_after_loading()
        self.v_sconv.process_weights_after_loading()

        if not self._kvw_t_ready:
            kw = self.k_sconv.conv_weight if self.k_sconv.dynamic else self.k_sconv.conv_weight.t()
            vw = self.v_sconv.conv_weight if self.v_sconv.dynamic else self.v_sconv.conv_weight.t()
            kvw_t = torch.stack(
                [
                    kw.contiguous().view(self.k_sconv.kernel_size, self.num_kv_heads, self.head_dim),
                    vw.contiguous().view(self.v_sconv.kernel_size, self.num_kv_heads, self.head_dim),
                ],
                dim=1,
            ).contiguous()
            self.register_buffer("_kvw_t", kvw_t, persistent=False)
            if self.k_sconv.dynamic:
                kv_wp_t = torch.stack(
                    [self.k_sconv.weight_proj.t(), self.v_sconv.weight_proj.t()], dim=0
                ).contiguous()
                self.register_buffer("_kv_weight_proj_t", kv_wp_t, persistent=False)
            self._kvw_t_ready = True

        if os.environ.get("INKLING2_FP8_ATTN", "1") == "0":
            return
        if self._use_fp8_linear or self.qkvr.weight.numel() == 0:
            return
        for src_name, w_buf, s_buf in (
            ("qkvr", "qkvr_w_fp8", "qkvr_scale"),
            ("wo_ud", "wo_w_fp8", "wo_scale"),
        ):
            mod = getattr(self, src_name)
            w_f = mod.weight.data.float()
            scale = (w_f.abs().amax(dim=-1) / self.fp8_max).clamp(min=1e-12)
            q = (w_f / scale.unsqueeze(-1)).clamp(-self.fp8_max, self.fp8_max).to(torch.float8_e4m3fn)
            self.register_buffer(w_buf, q, persistent=False)
            self.register_buffer(s_buf, scale.to(torch.float32), persistent=False)
            # Release BF16 weight HBM once quantized.
            mod.weight.data = mod.weight.data.new_empty((0,))
        self._use_fp8_linear = True

    def get_attn_backend(self) -> type[AttentionBackend]:
        return self._attn_backend

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec:
        return _make_kv_cache_spec(
            vllm_config, self.num_kv_heads, self.head_dim, self.sliding_window
        )

    def _pass_cache(self, pass_idx: int) -> tuple[str, list[torch.Tensor] | torch.Tensor]:
        if pass_idx == 0 or not self.pass_caches:
            return self.prefix, self.kv_cache
        cache = self.pass_caches[str(pass_idx - 1)]
        return cache.prefix, cache.kv_cache

    def _project(
        self,
        x_q: torch.Tensor,
        x_k: torch.Tensor,
        x_v: torch.Tensor,
        *,
        return_packed_proj: bool = False,
        x_f8: torch.Tensor | None = None,
        x_sc: torch.Tensor | None = None,
    ) -> list[torch.Tensor] | tuple[torch.Tensor, torch.Tensor, int]:
        q_side = self.q_side_size
        kv_size = self.split_sizes[4]
        if self._use_fp8_linear:
            w = self.qkvr_w_fp8
            s = self.qkvr_scale
            if x_k is x_q and x_v is x_q:
                if x_f8 is not None and x_sc is not None and x_q.device.type == "hpu":
                    projected = torch.ops.hpu.fp8_gemm_v2(
                        A=x_f8,
                        trans_A=False,
                        B=w,
                        trans_B=True,
                        D=None,
                        out_dtype=torch.bfloat16,
                        A_scale_inv=x_sc.view(-1, 1),
                        B_scale_inv=s,
                        bias=None,
                        accumulate=False,
                    )
                else:
                    projected = fp8_linear(x_q, w, s, fp8_max=self.fp8_max)
                if return_packed_proj:
                    # Zero split/copy: qkvr_prep & rel_paged_attn index projected directly!
                    return projected, projected.unsqueeze(0), q_side
                return list(projected.split(self.split_sizes, dim=-1))
            if x_q.device.type == "hpu":
                b = x_q.shape[0]
                if x_f8 is not None and x_sc is not None:
                    x_q_f8 = x_f8[0]
                    x_q_sc = x_sc[0].view(b, 1)
                    x_kv_f8 = x_f8[1:]
                    x_kv_sc = x_sc[1:].view(2, b, 1)
                else:
                    # Batch activation quantization across (x_q, x_k, x_v) once and run K/V as 1 batched FP8 GEMM.
                    x_fp8_all, x_sc_all = quantize_fp8_hpu(
                        torch.cat([x_q, x_k, x_v], dim=0),
                        fp8_max=self.fp8_max,
                        per_row=True,
                        use_cguid=True,
                    )
                    x_q_f8 = x_fp8_all[:b]
                    x_q_sc = x_sc_all[:b]
                    x_kv_f8 = x_fp8_all[b:].view(2, b, -1)
                    x_kv_sc = x_sc_all[b:].view(2, b, 1)
                projected_q = torch.ops.hpu.fp8_gemm_v2(
                    A=x_q_f8,
                    trans_A=False,
                    B=w[:q_side],
                    trans_B=True,
                    D=None,
                    out_dtype=torch.bfloat16,
                    A_scale_inv=x_q_sc,
                    B_scale_inv=s[:q_side],
                    bias=None,
                    accumulate=False,
                )
                kv_2 = torch.ops.hpu.fp8_gemm_v2(
                    A=x_kv_f8,
                    trans_A=False,
                    B=w[q_side : q_side + 2 * kv_size].view(2, kv_size, -1),
                    trans_B=True,
                    D=None,
                    out_dtype=torch.bfloat16,
                    A_scale_inv=x_kv_sc,
                    B_scale_inv=s[q_side : q_side + 2 * kv_size].view(2, 1, kv_size),
                    bias=None,
                    accumulate=False,
                )  # [2, B, kv_size]
                if return_packed_proj:
                    return projected_q, kv_2, 0
                return [*projected_q.split(self.split_sizes[:4], dim=-1), kv_2[0], kv_2[1]]
            projected_q = fp8_linear(x_q, w[:q_side], s[:q_side], fp8_max=self.fp8_max)
            k = fp8_linear(
                x_k, w[q_side : q_side + kv_size], s[q_side : q_side + kv_size], fp8_max=self.fp8_max
            )
            v = fp8_linear(
                x_v,
                w[q_side + kv_size : q_side + 2 * kv_size],
                s[q_side + kv_size : q_side + 2 * kv_size],
                fp8_max=self.fp8_max,
            )
            if return_packed_proj:
                return projected_q, torch.stack([k, v], dim=0), 0
            return [*projected_q.split(self.split_sizes[:4], dim=-1), k, v]

        weight = self.qkvr.weight
        if x_k is x_q and x_v is x_q:
            projected = F.linear(x_q, weight)
            if return_packed_proj:
                return projected, projected.unsqueeze(0), q_side
            return list(projected.split(self.split_sizes, dim=-1))
        projected_q = F.linear(x_q, weight[:q_side])
        k = F.linear(x_k, weight[q_side : q_side + kv_size])
        v = F.linear(x_v, weight[q_side + kv_size : q_side + 2 * kv_size])
        if return_packed_proj:
            return projected_q, torch.stack([k, v], dim=0), 0
        return [*projected_q.split(self.split_sizes[:4], dim=-1), k, v]

    def forward(
        self,
        positions: torch.Tensor,
        x_q: torch.Tensor,
        x_k: torch.Tensor | None = None,
        x_v: torch.Tensor | None = None,
        tau: torch.Tensor | None = None,
        pass_idx: int = 0,
        batch_size: int = 1,
        decode_ctx: DecodeAttnContext | None = None,
        kv_conv_state: tuple[
            torch.Tensor | None,
            torch.Tensor | None,
            bool,
            torch.Tensor | None,
            torch.Tensor | None,
        ]
        | None = None,
        direct_conv_cache: tuple[torch.Tensor, torch.Tensor] | None = None,
        x_f8: torch.Tensor | None = None,
        x_sc: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.is_local and x_k is None:
            x_k = x_v = x_q
        assert x_k is not None and x_v is not None

        num_tokens = x_q.shape[0]
        use_tpc_prep = (
            (
                direct_conv_cache is not None
                or (
                    kv_conv_state is not None
                    and not kv_conv_state[2]
                    and kv_conv_state[0] is not None
                    and kv_conv_state[1] is not None
                )
            )
            and x_q.device.type == "hpu"
            and x_q.dtype == torch.bfloat16
            and self.head_dim == 128
            and self.d_rel <= 64
            and self.k_sconv.kernel_size == 4
            and has_tpc_qkvr_prep()
        )
        proj_q: torch.Tensor | None = None
        gate: torch.Tensor | None = None
        sinks: torch.Tensor | None = None
        if use_tpc_prep:
            proj_q, kv_src, kv_col_offset = self._project(
                x_q, x_k, x_v, return_packed_proj=True, x_f8=x_f8, x_sc=x_sc
            )
            q = r = k = v = proj_q  # overwritten by inkling2_qkvr_prep_decode below
        else:
            q, r, gate, sinks, k, v = self._project(x_q, x_k, x_v, x_f8=x_f8, x_sc=x_sc)
            kv_src = None
            kv_col_offset = 0

        new_k_state: torch.Tensor | None = None
        new_v_state: torch.Tensor | None = None
        new_kv_state: torch.Tensor | None = None
        if use_tpc_prep:
            assert proj_q is not None and kv_src is not None
            h_kv, d = self.num_kv_heads, self.head_dim
            if direct_conv_cache is not None:
                conv_state_buf, load_slots_i32 = direct_conv_cache
            else:
                assert kv_conv_state is not None
                prev_k, prev_v = kv_conv_state[0], kv_conv_state[1]
                assert prev_k is not None and prev_v is not None
                conv_state_buf = torch.cat(
                    [prev_k.to(torch.bfloat16), prev_v.to(torch.bfloat16)], dim=-1
                ).contiguous()
                load_slots_i32 = torch.arange(num_tokens, device=proj_q.device, dtype=torch.int32)
            if getattr(self, "_kvw_t_ready", False):
                kvw_t = self._kvw_t
            else:
                kw = self.k_sconv.conv_weight if self.k_sconv.dynamic else self.k_sconv.conv_weight.t()
                vw = self.v_sconv.conv_weight if self.v_sconv.dynamic else self.v_sconv.conv_weight.t()
                kvw_t = torch.stack(
                    [kw.contiguous().view(4, h_kv, d), vw.contiguous().view(4, h_kv, d)], dim=1
                ).contiguous()
            is_dyn = 1 if self.k_sconv.dynamic else 0
            if is_dyn:
                if hasattr(self, "_kv_weight_proj_t"):
                    x_kv_2 = torch.stack([x_k, x_v], dim=0)
                    kv_dyn = torch.bmm(x_kv_2, self._kv_weight_proj_t).view(2, num_tokens, h_kv, 4)
                else:
                    kd = self.k_sconv.dynamic_taps(x_k)
                    vd = self.v_sconv.dynamic_taps(x_v)
                    assert kd is not None and vd is not None
                    kv_dyn = torch.stack(
                        [kd.view(num_tokens, h_kv, 4), vd.view(num_tokens, h_kv, 4)], dim=0
                    ).contiguous()
            else:
                kv_dyn = kvw_t  # unused by static ELF; avoids dummy slice copy
            has_tau = 0 if tau is None else 1
            if tau is None:
                tau_in = (
                    decode_ctx.positions_i32.view(torch.float32)
                    if decode_ctx is not None
                    else positions.float().contiguous()
                )
            else:
                tau_in = tau.float().contiguous()
            conv_scale = float((2 * 4 + 1) ** -0.5 if is_dyn else (4 + 1) ** -0.5)
            q, r, k, v, new_kv_5d = torch.ops.custom_op.inkling2_qkvr_prep_decode(
                proj_q.contiguous(),
                self.q_scale.contiguous(),
                self.r_scale.contiguous(),
                tau_in,
                kv_src.contiguous(),
                conv_state_buf,
                load_slots_i32,
                kvw_t,
                kv_dyn,
                kv_col_offset,
                has_tau,
                is_dyn,
                conv_scale,
                self.qk_clip,
                self.rel_clip,
                1e-6,
                self.eps,
            )
            if direct_conv_cache is not None:
                new_kv_state = new_kv_5d.view(num_tokens, 3, 2 * h_kv * d)
            else:
                new_k_state = new_kv_5d[:, :, 0].reshape(num_tokens, 3, h_kv * d)
                new_v_state = new_kv_5d[:, :, 1].reshape(num_tokens, 3, h_kv * d)
        else:
            if kv_conv_state is not None:
                prev_k, prev_v, conv_is_prompt, pad_mask, seq_lens = kv_conv_state
                k, new_k_state = self.k_sconv.apply_with_state(
                    k,
                    prev_k,
                    dyn_taps=self.k_sconv.dynamic_taps(x_k),
                    is_prompt=conv_is_prompt,
                    batch_size=batch_size,
                    pad_mask=pad_mask,
                    seq_lens=seq_lens,
                )
                v, new_v_state = self.v_sconv.apply_with_state(
                    v,
                    prev_v,
                    dyn_taps=self.v_sconv.dynamic_taps(x_v),
                    is_prompt=conv_is_prompt,
                    batch_size=batch_size,
                    pad_mask=pad_mask,
                    seq_lens=seq_lens,
                )
            else:
                conv_owner = self.conv_states[pass_idx]
                k = self.k_sconv(
                    k, dyn_taps=self.k_sconv.dynamic_taps(x_k), owner=conv_owner, batch_size=batch_size
                )
                v = self.v_sconv(
                    v, dyn_taps=self.v_sconv.dynamic_taps(x_v), owner=conv_owner, batch_size=batch_size
                )

            q = l2_clip(
                q.view(num_tokens, self.num_heads, self.head_dim),
                self.q_scale,
                self.qk_clip,
                eps=1e-6,
                tau=tau,
            )
            r = l2_clip(
                r.view(num_tokens, self.num_heads, self.d_rel),
                self.r_scale,
                self.rel_clip,
                eps=1e-6,
                tau=tau,
            )
            k = l2_clip(
                k.view(num_tokens, self.num_kv_heads, self.head_dim),
                None,
                self.qk_clip,
                eps=1e-6,
                tau=None,
            )
            v = weightless_rms_norm(
                v.view(num_tokens, self.num_kv_heads, self.head_dim),
                eps=self.eps,
            )

        layer_name, kv_cache = self._pass_cache(pass_idx)
        attn_metadata = get_forward_context().attn_metadata
        if isinstance(attn_metadata, dict):
            attn_metadata = attn_metadata.get(layer_name)

        has_cache = (
            attn_metadata is not None
            and (
                (isinstance(kv_cache, (list, tuple)) and len(kv_cache) > 0 and kv_cache[0].numel() > 0)
                or (isinstance(kv_cache, torch.Tensor) and kv_cache.numel() > 0)
            )
        )

        gated: torch.Tensor | None = None
        tpc_gated = False
        deferred_kv_slot_mapping: torch.Tensor | None = None
        if not has_cache:
            rel_logits = torch.matmul(r.float(), self.rel_logits_proj.float()).to(q.dtype)
            attn_out, lse = _sdpa_with_rel_bias_and_lse(
                q,
                k,
                v,
                rel_logits,
                positions.long(),
                positions.long(),
                scale=self.scaling,
                sliding_window=self.sliding_window,
                enforce_index_causal=True,
            )
        else:
            slot_mapping = getattr(attn_metadata, "slot_mapping", None)
            if attn_metadata.is_prompt:
                if slot_mapping is not None:
                    _write_kv_to_paged_cache(
                        k,
                        v,
                        kv_cache,
                        slot_mapping,
                        precomputed_slots=None if decode_ctx is None else decode_ctx.kv_slots,
                        precomputed_num_slots=0 if decode_ctx is None else decode_ctx.kv_num_slots,
                    )
                rel_logits = torch.matmul(r.float(), self.rel_logits_proj.float()).to(q.dtype)
                attn_out, lse = self._prefill_attention(
                    q, k, v, rel_logits, positions, kv_cache, attn_metadata, batch_size=batch_size
                )
            else:
                block_list = getattr(attn_metadata, "block_list", None)
                key_cache, value_cache = kv_cache[0], kv_cache[1]
                block_size = int(getattr(attn_metadata, "block_size", 128))
                can_tpc_pa = (
                    q.device.type == "hpu"
                    and q.dtype == torch.bfloat16
                    and key_cache.dtype == torch.bfloat16
                    and value_cache.dtype == torch.bfloat16
                    and self.head_dim == 128
                    and self.num_heads <= 64
                    and self.d_rel <= 64
                    and block_list is not None
                    and 0 < block_list.shape[0] <= 512
                    and (
                        key_cache.ndim == 3
                        or (key_cache.ndim == 4 and key_cache.shape[2] == self.num_kv_heads)
                    )
                    and has_tpc_paged_attn()
                )
                if can_tpc_pa:
                    assert block_list is not None
                    ctx = decode_ctx
                    if ctx is None:
                        ctx = build_decode_attn_context(
                            attn_metadata,
                            positions,
                            num_tokens,
                            sliding_window=self.sliding_window or self.rel_extent,
                            rel_extent=self.rel_extent,
                        )
                    assert ctx is not None
                    k_4d = (
                        key_cache.view(-1, block_size, self.num_kv_heads, self.head_dim)
                        if key_cache.ndim == 3
                        else key_cache
                    )
                    v_4d = (
                        value_cache.view(-1, block_size, self.num_kv_heads, self.head_dim)
                        if value_cache.ndim == 3
                        else value_cache
                    )
                    proj_t = (
                        self.rel_logits_proj_t
                        if getattr(self, "_rel_proj_t_ready", False)
                        else self.rel_logits_proj.t().contiguous()
                    )
                    bl_i32 = (
                        ctx.block_list_i32
                        if ctx.block_list_i32 is not None
                        else block_list.to(torch.int32).contiguous()
                    )
                    if proj_q is not None:
                        pq_pa = proj_q.contiguous()
                    else:
                        assert gate is not None and sinks is not None
                        pq_pa = torch.cat(
                            [q.reshape(num_tokens, -1), r.reshape(num_tokens, -1), gate, sinks], dim=-1
                        ).contiguous()
                    gated = torch.ops.custom_op.inkling2_rel_paged_attn_gate_sink(
                        q.contiguous(),
                        r.contiguous(),
                        pq_pa,
                        proj_t,
                        k_4d,
                        v_4d,
                        k.contiguous(),
                        v.contiguous(),
                        bl_i32,
                        ctx.block_groups_i32,
                        ctx.block_usage_i32,
                        ctx.block_start_pos_i32,
                        ctx.positions_i32,
                        float(self.scaling),
                        int(self.sliding_window or 0),
                        int(self.rel_extent),
                    ).reshape(num_tokens, self.num_heads * self.head_dim)
                    tpc_gated = True
                    deferred_kv_slot_mapping = slot_mapping
                else:
                    if slot_mapping is not None:
                        _write_kv_to_paged_cache(
                            k,
                            v,
                            kv_cache,
                            slot_mapping,
                            precomputed_slots=None if decode_ctx is None else decode_ctx.kv_slots,
                            precomputed_num_slots=0 if decode_ctx is None else decode_ctx.kv_num_slots,
                        )
                    attn_out, lse = self._decode_attention(
                        q,
                        r,
                        positions,
                        kv_cache,
                        attn_metadata,
                        decode_ctx=decode_ctx,
                    )

        if gated is None:
            if gate is None or sinks is None:
                assert proj_q is not None
                g_start = self.num_heads * (self.head_dim + self.d_rel)
                g_end = g_start + self.num_heads * self.head_dim
                gate = proj_q[:, g_start:g_end]
                sinks = proj_q[:, g_end : g_end + self.num_heads]
            gated = gate_sink(attn_out, gate, lse, sinks)
        if self._use_fp8_linear:
            # Custom TPC paged-attn hides gate_sink inside an opaque ELF so CGUID dynamic quant
            # is safe and avoids 4 manual quant ops/layer; fallback PyTorch gate_sink keeps False.
            out = fp8_linear(
                gated, self.wo_w_fp8, self.wo_scale, fp8_max=self.fp8_max, use_cguid=tpc_gated
            )
        else:
            out = F.linear(gated, self.wo_ud.weight)
        # Write current token (k, v) AFTER wo_ud so wo_ud FP8 quant + MME launches immediately
        # after rel_paged_attn without being blocked by KV cache index_copy_.
        if deferred_kv_slot_mapping is not None:
            _write_kv_to_paged_cache(
                k,
                v,
                kv_cache,
                deferred_kv_slot_mapping,
                precomputed_slots=None if decode_ctx is None else decode_ctx.kv_slots,
                precomputed_num_slots=0 if decode_ctx is None else decode_ctx.kv_num_slots,
            )
        if direct_conv_cache is not None:
            assert new_kv_state is not None
            return out, new_kv_state
        if kv_conv_state is not None:
            assert new_k_state is not None and new_v_state is not None
            return out, new_k_state, new_v_state
        return out

    def _prefill_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        rel_logits: torch.Tensor,
        positions: torch.Tensor,
        kv_cache: list[torch.Tensor] | torch.Tensor,
        attn_metadata,
        *,
        batch_size: int = 1,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        block_list = getattr(attn_metadata, "block_list", None)
        pad_mask = getattr(attn_metadata, "padding_mask_flat", None)

        # Fast zero-sync path for non-chunked prefill (block_list is None).
        if block_list is None:
            b = max(1, batch_size)
            s = q.shape[0] // b
            if b == 1:
                k_valid = (
                    pad_mask.reshape(-1).bool()
                    if pad_mask is not None and pad_mask.numel() == s
                    else None
                )
                return _sdpa_with_rel_bias_and_lse(
                    q,
                    k,
                    v,
                    rel_logits,
                    positions.long(),
                    positions.long(),
                    scale=self.scaling,
                    sliding_window=self.sliding_window,
                    enforce_index_causal=True,
                    k_valid_mask=k_valid,
                )
            outs, lses = [], []
            for r in range(b):
                st, en = r * s, (r + 1) * s
                k_valid = (
                    pad_mask.reshape(-1)[st:en].bool()
                    if pad_mask is not None and pad_mask.numel() == b * s
                    else None
                )
                o_r, lse_r = _sdpa_with_rel_bias_and_lse(
                    q[st:en],
                    k[st:en],
                    v[st:en],
                    rel_logits[st:en],
                    positions[st:en].long(),
                    positions[st:en].long(),
                    scale=self.scaling,
                    sliding_window=self.sliding_window,
                    enforce_index_causal=True,
                    k_valid_mask=k_valid,
                )
                outs.append(o_r)
                lses.append(lse_r)
            return torch.cat(outs, dim=0), torch.cat(lses, dim=0)

        # Chunked prefill / prefix blocks path (block_list is not None).
        num_tokens = q.shape[0]
        out = torch.zeros((num_tokens, self.num_heads, self.head_dim), dtype=torch.float32, device=q.device)
        lse = torch.zeros((num_tokens, self.num_heads), dtype=torch.float32, device=q.device)
        block_size = int(getattr(attn_metadata, "block_size", 128))
        q_starts = getattr(attn_metadata, "query_start_loc_p", None)
        if q_starts is None:
            q_starts = getattr(attn_metadata, "query_start_loc", None)
        starts_cpu = q_starts.tolist() if q_starts is not None else [0, num_tokens]
        ctx_lens = getattr(attn_metadata, "context_lens_tensor", None)
        ctx_list = ctx_lens.tolist() if ctx_lens is not None else [0] * (len(starts_cpu) - 1)

        blk_cursor = 0
        num_reqs = len(starts_cpu) - 1
        row_stride = num_tokens // max(1, batch_size) if batch_size > 1 else 0
        for r in range(num_reqs):
            if row_stride > 0 and num_reqs == batch_size:
                q_len = starts_cpu[r + 1] - starts_cpu[r]
                s, e = r * row_stride, r * row_stride + q_len
            else:
                s, e = starts_cpu[r], starts_cpu[r + 1]
            if e <= s:
                continue
            q_r = q[s:e]
            rel_r = rel_logits[s:e]
            q_pos = positions[s:e].long()
            past = ctx_list[r] if r < len(ctx_list) else 0
            q_valid = (
                pad_mask.reshape(-1)[s:e].bool()
                if pad_mask is not None and pad_mask.numel() == num_tokens
                else None
            )
            if past > 0:
                num_ctx_blks = (past + block_size - 1) // block_size
                req_blks = block_list[blk_cursor : blk_cursor + num_ctx_blks]
                blk_cursor += num_ctx_blks
                k_ctx, v_ctx = _gather_paged_kv(
                    kv_cache, req_blks, past, self.num_kv_heads, self.head_dim, block_size
                )
                k_r = torch.cat([k_ctx.to(k.dtype), k[s:e]], dim=0)
                v_r = torch.cat([v_ctx.to(v.dtype), v[s:e]], dim=0)
                k_pos = torch.cat(
                    [torch.arange(past, device=q.device, dtype=torch.long), q_pos], dim=0
                )
                k_valid = (
                    torch.cat(
                        [torch.ones(past, dtype=torch.bool, device=q.device), q_valid],
                        dim=0,
                    )
                    if q_valid is not None
                    else None
                )
            else:
                k_r, v_r = k[s:e], v[s:e]
                k_pos = q_pos
                k_valid = q_valid
            o_r, lse_r = _sdpa_with_rel_bias_and_lse(
                q_r,
                k_r,
                v_r,
                rel_r,
                q_pos,
                k_pos,
                scale=self.scaling,
                sliding_window=self.sliding_window,
                enforce_index_causal=(past == 0),
                k_valid_mask=k_valid,
            )
            out[s:e] = o_r
            lse[s:e] = lse_r
        return out, lse

    def _decode_attention(
        self,
        q: torch.Tensor,
        r: torch.Tensor,
        positions: torch.Tensor,
        kv_cache: list[torch.Tensor] | torch.Tensor,
        attn_metadata,
        *,
        decode_ctx: DecodeAttnContext | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Vectorized, zero-host-sync paged relative-bias decode attention on HPU."""
        b = q.shape[0]
        h, h_kv, d = self.num_heads, self.num_kv_heads, self.head_dim
        groups = h // h_kv
        block_size = int(getattr(attn_metadata, "block_size", 128))
        block_list = getattr(attn_metadata, "block_list", None)
        assert block_list is not None
        nb = block_list.shape[0]

        if decode_ctx is None or decode_ctx.safe_grp is None:
            decode_ctx = build_decode_attn_context(
                attn_metadata,
                positions,
                b,
                sliding_window=self.sliding_window or self.rel_extent,
                rel_extent=self.rel_extent,
            )
        assert decode_ctx is not None
        assert (
            decode_ctx.safe_grp is not None
            and decode_ctx.valid_slot_local is not None
            and decode_ctx.valid_slot_global is not None
            and decode_ctx.clamped_local is not None
            and decode_ctx.clamped_global is not None
            and decode_ctx.grp_mask is not None
            and decode_ctx.grp_w_t is not None
            and decode_ctx.neg_large is not None
        )

        safe_grp = decode_ctx.safe_grp
        valid_slot_h = (
            decode_ctx.valid_slot_local if self.is_local else decode_ctx.valid_slot_global
        )
        clamped_flat = (
            decode_ctx.clamped_local if self.is_local else decode_ctx.clamped_global
        )

        key_cache, value_cache = kv_cache[0], kv_cache[1]
        if key_cache.ndim == 3:
            k_4d = key_cache.view(-1, block_size, h_kv, d)
            v_4d = value_cache.view(-1, block_size, h_kv, d)
        else:
            k_4d = key_cache if key_cache.shape[2] == h_kv else key_cache.transpose(1, 2)
            v_4d = value_cache if value_cache.shape[2] == h_kv else value_cache.transpose(1, 2)

        num_phys_blocks = k_4d.shape[0]
        safe_blks = torch.remainder(block_list.long(), num_phys_blocks)  # [NB]

        k_blks = k_4d.index_select(0, safe_blks)  # [NB, BS, H_kv, D]
        v_blks = v_4d.index_select(0, safe_blks)  # [NB, BS, H_kv, D]

        # Grouped BF16 MME QK scores: [NB, H_kv, G, D] @ [NB, H_kv, D, BS] -> [NB, H, BS] fp32
        q_blk = q.index_select(0, safe_grp).view(nb, h_kv, groups, d)
        scores = torch.matmul(
            q_blk, k_blks.permute(0, 2, 3, 1).to(q.dtype)
        ).float().reshape(nb, h, block_size) * self.scaling

        # Relative position bias directly via gathered projection rows (avoids [NB, H, 4096] FP32 gather):
        # r_blk [NB, H, d_rel] @ proj_blk^T [NB, d_rel, BS] -> [NB, H, BS] fp32.
        r_blk = r.index_select(0, safe_grp)  # [NB, H, d_rel] bf16
        proj_t = (
            self.rel_logits_proj_t
            if getattr(self, "_rel_proj_t_ready", False)
            else self.rel_logits_proj.t().contiguous()
        )
        proj_blk = proj_t.index_select(0, clamped_flat).view(nb, block_size, self.d_rel)
        bias = torch.matmul(r_blk, proj_blk.transpose(1, 2)).float()
        scores = scores + bias

        # Exact multi-block online softmax reduction across block_groups -> [B, H, D]
        neg_large = decode_ctx.neg_large
        scores_masked = torch.where(valid_slot_h, scores, neg_large)
        m_blk = scores_masked.amax(dim=-1)  # [NB, H]

        m_req = torch.where(
            decode_ctx.grp_mask, m_blk.unsqueeze(1), neg_large
        ).amax(dim=0)  # [B, H]

        m_blk_ref = m_req.index_select(0, safe_grp).unsqueeze(-1)  # [NB, H, 1]
        exp_s = (
            torch.exp((scores_masked - m_blk_ref).clamp(min=-80.0, max=0.0))
            * valid_slot_h.float()
        )  # [NB, H, BS]
        l_blk = exp_s.sum(dim=-1)  # [NB, H]
        o_blk = torch.matmul(
            exp_s.to(v_blks.dtype).view(nb, h_kv, groups, block_size),
            v_blks.transpose(1, 2),
        ).float().reshape(nb, h, d)

        l_req = torch.matmul(decode_ctx.grp_w_t, l_blk)  # [B, H]
        o_req = torch.matmul(decode_ctx.grp_w_t, o_blk.reshape(nb, h * d)).view(b, h, d)

        out = o_req / l_req.clamp(min=1e-30).unsqueeze(-1)
        lse = m_req + torch.log(l_req.clamp(min=1e-30))
        return out, lse
