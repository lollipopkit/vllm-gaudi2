# SPDX-License-Identifier: Apache-2.0
"""Single-pass GDN decode recurrence on the custom Gaudi2 TPC kernel (``tpc/`` in the repo root).

With ``VLLM_HPU_GDN_TPC_DECODE=1`` the f32 SSM state pool is KV-major (``[slots, HV, K, V]``): the kernel's
projections are lane-parallel over the value index. Every other reader/writer of the pool (prefill
initial/final state, the indexed decode fallback) transposes at its boundary. The kernel library must be on
``GC_KERNEL_PATH`` before the process opens the device (``host/vm/serve.sh`` sets it).
"""

from __future__ import annotations

import glob
import os
from pathlib import Path

import torch

from vllm_gaudi import envs

STATE_KV = envs.VLLM_HPU_GDN_TPC_DECODE


def _load_torch_op() -> None:
    path = os.environ.get("VLLM_HPU_GDN_TPC_TORCH_LIB")
    if not path:
        resolved = Path(__file__).resolve()
        built: list[str] = []
        for depth in (2, 3):
            if len(resolved.parents) > depth:
                built = sorted(glob.glob(str(resolved.parents[depth] / "tpc" / "torch" / "gdn_tpc*.so")))
                if built:
                    break
        if not built:
            raise RuntimeError("VLLM_HPU_GDN_TPC_DECODE=1 but tpc/torch/gdn_tpc*.so is not built "
                               "(cd tpc/torch && python setup.py build_ext --inplace)")
        path = built[0]
    torch.ops.load_library(path)


if STATE_KV:
    _load_torch_op()


def to_vk(state: torch.Tensor) -> torch.Tensor:
    """Pool layout -> the VK layout of the torch GDN recurrences."""
    return state.transpose(-1, -2) if STATE_KV else state


def from_vk(state: torch.Tensor) -> torch.Tensor:
    """VK state -> pool layout."""
    return state.transpose(-1, -2).contiguous() if STATE_KV else state


def gdn_tpc_decode_direct(qk: torch.Tensor, value: torch.Tensor, log_decay: torch.Tensor, beta: torch.Tensor,
                          pool: torch.Tensor, row_offset: int, scale: float) -> torch.Tensor:
    """One decode step for the B requests owning pool rows row_offset .. row_offset + B - 1 (KV-major pool,
    updated in place by the kernel; no copy back).

    qk [B, 2H, K] f32 L2-normalized (q heads, then k heads; q unscaled); value [B, HV, V]; log_decay/beta
    [B, HV]. Returns out [B, HV, V] f32. The kernel reads these directly: a stack / broadcast here would add a
    dependent node per layer.
    """
    q, k = qk.chunk(2, dim=1)
    kq_dot = (k * q).sum(-1)  # [B, H]
    return torch.ops.custom_op.gdn_decode(pool, qk, value.float(), torch.exp(log_decay.float()), beta.float(), kq_dot,
                                          row_offset, scale)


def gdn_tpc_verify(q: torch.Tensor, k: torch.Tensor, value: torch.Tensor, log_decay: torch.Tensor,
                   beta: torch.Tensor, pool: torch.Tensor, load_rows: torch.Tensor, store_rows: torch.Tensor,
                   scale: float, heads_per_key: int) -> torch.Tensor:
    """Spec-decode verification of T = 1 + k tokens per request on the KV-major pool (updated in place):
    the initial state from row load_rows[b], the state after token t into row store_rows[b, t].

    q/k [B, T, H, K] f32 L2-normalized; value [B, T, HV, V]; log_decay/beta [B, T, HV]; load_rows [B] and
    store_rows [B, T] already inside the pool. Returns out [B, T, HV, V] f32.
    """
    q = q * scale
    kq = torch.stack((k, q), dim=3)  # [B, T, H, 2, K]
    kq_dot = (k * q).sum(-1).repeat_interleave(heads_per_key, dim=2)
    coeff = torch.stack((torch.exp(log_decay.float()), beta.float(), kq_dot), dim=-1)
    rows = torch.cat((load_rows.reshape(-1, 1), store_rows), dim=1).to(torch.int32)
    return torch.ops.custom_op.gdn_verify(pool, kq, value.float(), coeff, rows)


def packed_decode_kv(packed_qkv: torch.Tensor, log_decay: torch.Tensor, beta: torch.Tensor, pool: torch.Tensor,
                     scale: float, row_offset: int) -> tuple[torch.Tensor, torch.Tensor]:
    """The fused recipe's recurrence on the KV-major pool (Qwen3.8 TP1 shapes)."""
    from flashinfer_gaudi._reference import (QWEN38_DIM, QWEN38_KEY_HEADS, QWEN38_QK_WIDTH, QWEN38_VALUE_HEADS,
                                             l2_normalize_rsqrt)
    batch = packed_qkv.shape[0]
    qk = l2_normalize_rsqrt(packed_qkv[:, :QWEN38_QK_WIDTH].reshape(batch, 2 * QWEN38_KEY_HEADS,
                                                                    QWEN38_DIM).to(torch.float32))
    value = packed_qkv[:, QWEN38_QK_WIDTH:].reshape(batch, QWEN38_VALUE_HEADS, QWEN38_DIM)
    out = gdn_tpc_decode_direct(qk, value, log_decay.reshape(batch, QWEN38_VALUE_HEADS),
                                beta.reshape(batch, QWEN38_VALUE_HEADS), pool, row_offset, scale)
    return out.to(packed_qkv.dtype), pool
