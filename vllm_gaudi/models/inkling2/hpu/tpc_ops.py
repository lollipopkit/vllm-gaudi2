# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Lazy loader and feature flags for custom Gaudi2 TPC C++ kernels used by Inkling2 HPU."""

from __future__ import annotations

import glob
import os
from pathlib import Path

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

_LOADED: bool | None = None

_REQUIRED_OPS = (
    "inkling2_moe_w13_situ_f8",
    "inkling2_moe_w2_f8",
    "inkling2_moe_group_experts",
    "inkling2_moe_w13_grouped_f8",
    "inkling2_moe_w2_grouped_f8",
    "inkling2_moe_gate_topk",
    "inkling2_moe_gate_gemv_splitk",
    "inkling2_moe_sum_sconv_res_decode",
    "inkling2_attn_res_norm_bf16",
    "inkling2_attn_res_norm_f8",
    "inkling2_attn_res_f32",
    "inkling2_sconv_static_decode",
    "inkling2_qkvr_prep_decode",
    "inkling2_pair_rms_norm",
    "inkling2_blend_rms_norm",
    "inkling2_blend_rms_norm_f8",
    "inkling2_rel_paged_attn_gate_sink",
)


def _env_enabled(name: str, default: str = "1") -> bool:
    return os.environ.get(name, default).strip().lower() not in ("0", "false", "no")


def _find_op_library() -> str | None:
    explicit = os.environ.get("INKLING2_TPC_OP_LIB", "").strip()
    if explicit:
        return explicit if os.path.isfile(explicit) else None

    resolved = Path(__file__).resolve()
    candidates: list[str] = []
    for depth in (4, 5):
        if len(resolved.parents) > depth:
            candidates.append(
                str(resolved.parents[depth] / "tpc" / "torch" / "inkling2_tpc*.so")
            )
    for pattern in candidates:
        matches = sorted(glob.glob(pattern))
        if matches:
            return matches[0]
    return None


def _gc_kernel_path_ready() -> bool:
    gc_path = os.environ.get("GC_KERNEL_PATH", "").strip()
    if not gc_path:
        return False
    return any(
        os.path.isfile(p) and ("gaudi2_tpc_proxy" in p or "gaudi2_tpc_kernels" in p)
        for p in gc_path.split(":")
    )


def ensure_tpc_ops_loaded() -> bool:
    """Load `inkling2_tpc*.so` once if `GC_KERNEL_PATH` exposes our custom TPC kernels."""
    global _LOADED
    if _LOADED is not None:
        return _LOADED

    if not _env_enabled("INKLING2_TPC_OPS"):
        _LOADED = False
        return False

    if not _gc_kernel_path_ready():
        _LOADED = False
        return False

    so_path = _find_op_library()
    if so_path is None:
        logger.warning(
            "GC_KERNEL_PATH includes custom Gaudi2 TPC kernels, but inkling2_tpc*.so "
            "was not found; falling back to PyTorch HPU ops."
        )
        _LOADED = False
        return False

    try:
        torch.ops.load_library(so_path)
        for op_name in _REQUIRED_OPS:
            getattr(torch.ops.custom_op, op_name)
    except Exception as exc:
        logger.warning("Failed to load Inkling2 TPC custom ops (%s): %s", so_path, exc)
        _LOADED = False
        return False

    logger.info("Loaded Inkling2 Gaudi2 TPC custom ops (%d ops) from %s", len(_REQUIRED_OPS), so_path)
    _LOADED = True
    return True


def has_tpc_moe() -> bool:
    return _env_enabled("INKLING2_TPC_MOE") and ensure_tpc_ops_loaded()


def has_tpc_moe_grouped() -> bool:
    return _env_enabled("INKLING2_TPC_MOE_GROUPED") and has_tpc_moe()


def has_tpc_attn_res() -> bool:
    return _env_enabled("INKLING2_TPC_ATTN_RES") and ensure_tpc_ops_loaded()


def has_tpc_sconv() -> bool:
    return _env_enabled("INKLING2_TPC_SCONV") and ensure_tpc_ops_loaded()


def has_tpc_qkvr_prep() -> bool:
    return _env_enabled("INKLING2_TPC_QKVR_PREP") and ensure_tpc_ops_loaded()


def has_tpc_blend() -> bool:
    return _env_enabled("INKLING2_TPC_BLEND") and ensure_tpc_ops_loaded()


def has_tpc_paged_attn() -> bool:
    return _env_enabled("INKLING2_TPC_PAGED_ATTN") and ensure_tpc_ops_loaded()


def has_tpc_gate() -> bool:
    return _env_enabled("INKLING2_TPC_GATE") and ensure_tpc_ops_loaded()


def has_tpc_gate_splitk() -> bool:
    return _env_enabled("INKLING2_TPC_GATE_SPLITK") and ensure_tpc_ops_loaded()


def has_tpc_attn_res_f8() -> bool:
    return _env_enabled("INKLING2_TPC_ATTN_RES_F8") and has_tpc_attn_res()


def has_tpc_blend_f8() -> bool:
    return _env_enabled("INKLING2_TPC_BLEND_F8") and has_tpc_blend()


def has_tpc_moe_post() -> bool:
    return _env_enabled("INKLING2_TPC_MOE_POST") and ensure_tpc_ops_loaded()


def has_tpc_moe_pack() -> bool:
    return (
        _env_enabled("INKLING2_TPC_MOE_PACK")
        and has_tpc_moe_post()
        and hasattr(torch.ops.custom_op, "inkling2_moe_sum_sconv_pack_decode")
    )
