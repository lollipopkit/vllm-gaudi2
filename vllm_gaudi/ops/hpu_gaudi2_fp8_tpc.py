# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Lazy loader, feature flags, and fusion helpers for custom Gaudi2 FP8/GDN/ShortConv TPC kernels.

All kernels live in ``tpc/`` and are exposed via ``libgaudi2_tpc_proxy.so`` (or
``libgaudi2_tpc_kernels.so``) on ``GC_KERNEL_PATH`` plus ``tpc/torch/gaudi2_fp8_tpc*.so``.
Every helper degrades cleanly to PyTorch HPU ops when the shared libraries are absent or disabled.
"""

from __future__ import annotations

import glob
import os
from pathlib import Path

import torch
from vllm.logger import init_logger
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.fusion import fused_act_quant
from vllm.model_executor.layers.fusion.quant_activation import QuantizedActivation
from vllm.model_executor.layers.layernorm import GemmaRMSNorm, RMSNorm
from vllm.model_executor.layers.linear import LinearBase
from vllm.model_executor.layers.quantization.utils.quant_utils import kFp8DynamicTokenSym

import vllm_gaudi.envs as gaudi_envs
import vllm_gaudi.extension.ops as hpu_ops

logger = init_logger(__name__)

_LOADED: bool | None = None
_SILU_REGISTERED: bool = False
# Cached as plain ints so torch.compile graphs never read os.environ inside forward().
_MIN_ROWS: int = gaudi_envs.VLLM_HPU_GAUDI2_FP8_TPC_MIN_ROWS
_MAX_ROWS: int = gaudi_envs.VLLM_HPU_GAUDI2_FP8_TPC_MAX_ROWS

MAX_RMS_NORM_F8_DIM = 12288
MAX_SILU_MUL_F8_DIM = 28672

_REQUIRED_OPS = (
    "gaudi2_add_rms_norm_f8",
    "gaudi2_add_rms_norm_f8_bf16",
    "gaudi2_sandwich_rms_norm_f8",
    "gaudi2_silu_mul_f8",
    "gdn_prep_decode",
    "gdn_gated_rms_norm_f8",
    "lfm3_short_conv_decode",
)


_REQUIRED_GUIDS = (
    b"gaudi2_add_rms_norm_f8_gaudi2\x00",
    b"gaudi2_sandwich_rms_norm_f8_gaudi2\x00",
    b"gaudi2_silu_mul_f8_gaudi2\x00",
    b"gdn_prep_decode_gaudi2\x00",
    b"gdn_gated_rms_norm_f8_gaudi2\x00",
    b"lfm3_short_conv_decode_gaudi2\x00",
)


def _gc_kernel_path_ready() -> bool:
    gc_path = os.environ.get("GC_KERNEL_PATH", "").strip()
    if not gc_path:
        return False
    for entry in gc_path.split(":"):
        if not entry or not os.path.isfile(entry):
            continue
        if "gaudi2_tpc_proxy" not in entry and "gaudi2_tpc_kernels" not in entry:
            continue
        try:
            blob = Path(entry).read_bytes()
        except OSError:
            continue
        if all(guid in blob for guid in _REQUIRED_GUIDS):
            return True
    return False


def _find_op_library() -> str | None:
    explicit = gaudi_envs.VLLM_HPU_GAUDI2_FP8_TPC_TORCH_LIB.strip()
    if explicit:
        return explicit if os.path.isfile(explicit) else None

    resolved = Path(__file__).resolve()
    for depth in (2, 3):
        if len(resolved.parents) > depth:
            matches = sorted(glob.glob(str(resolved.parents[depth] / "tpc" / "torch" / "gaudi2_fp8_tpc*.so")))
            if matches:
                return matches[0]
    return None


def ensure_gaudi2_fp8_tpc_loaded() -> bool:
    """Load ``gaudi2_fp8_tpc*.so`` once if running on Gaudi2 with custom TPC kernels on GC_KERNEL_PATH."""
    global _LOADED, _MIN_ROWS, _MAX_ROWS
    if _LOADED is not None:
        return _LOADED

    _MIN_ROWS = gaudi_envs.VLLM_HPU_GAUDI2_FP8_TPC_MIN_ROWS
    _MAX_ROWS = gaudi_envs.VLLM_HPU_GAUDI2_FP8_TPC_MAX_ROWS

    if not hpu_ops.is_hpu_gaudi2 or not gaudi_envs.VLLM_HPU_GAUDI2_FP8_TPC:
        _LOADED = False
        return False

    if not _gc_kernel_path_ready():
        _LOADED = False
        return False

    so_path = _find_op_library()
    if so_path is None:
        logger.warning(
            "GC_KERNEL_PATH includes custom Gaudi2 TPC kernels, but gaudi2_fp8_tpc*.so "
            "was not found; falling back to PyTorch HPU ops."
        )
        _LOADED = False
        return False

    try:
        torch.ops.load_library(so_path)
        for op_name in _REQUIRED_OPS:
            getattr(torch.ops.custom_op, op_name)
    except Exception as exc:
        logger.warning("Failed to load Gaudi2 FP8 TPC custom ops (%s): %s", so_path, exc)
        _LOADED = False
        return False

    logger.info(
        "Loaded Gaudi2 FP8/GDN/ShortConv TPC custom ops (%d ops, rows=%d..%d) from %s",
        len(_REQUIRED_OPS),
        _MIN_ROWS,
        _MAX_ROWS,
        so_path,
    )
    _LOADED = True
    return True


def within_fp8_tpc_max_rows(x: torch.Tensor) -> bool:
    """Return whether ``x``'s flattened row count falls within the decode-focused TPC fusion range."""
    if x.ndim < 2 or x.shape[-1] <= 0:
        return False
    if x.ndim == 2:
        rows = x.shape[0]
    elif x.ndim == 3:
        rows = x.shape[0] * x.shape[1]
    else:
        rows = x.numel() // x.shape[-1]
    if rows <= 0 or rows < _MIN_ROWS:
        return False
    return _MAX_ROWS <= 0 or rows <= _MAX_ROWS


def has_add_rms_norm_f8() -> bool:
    return gaudi_envs.VLLM_HPU_TPC_ADD_RMS_NORM_F8 and ensure_gaudi2_fp8_tpc_loaded()


def has_sandwich_rms_norm_f8() -> bool:
    return gaudi_envs.VLLM_HPU_TPC_SANDWICH_RMS_NORM_F8 and ensure_gaudi2_fp8_tpc_loaded()


def has_silu_mul_f8() -> bool:
    return gaudi_envs.VLLM_HPU_TPC_SILU_MUL_F8 and ensure_gaudi2_fp8_tpc_loaded()


def has_gdn_prep_decode() -> bool:
    return gaudi_envs.VLLM_HPU_TPC_GDN_PREP and ensure_gaudi2_fp8_tpc_loaded()


def has_gdn_gated_rms_norm_f8() -> bool:
    return gaudi_envs.VLLM_HPU_TPC_GDN_GATED_NORM_F8 and ensure_gaudi2_fp8_tpc_loaded()


def has_lfm3_short_conv_decode() -> bool:
    return gaudi_envs.VLLM_HPU_TPC_LFM3_SHORT_CONV and ensure_gaudi2_fp8_tpc_loaded()


def _silu_mul_f8_supported(act_fn: torch.nn.Module, x: torch.Tensor, linear: LinearBase) -> bool:
    if not isinstance(act_fn, SiluAndMul) or x.dtype != torch.bfloat16 or x.device.type != "hpu":
        return False
    if not within_fp8_tpc_max_rows(x):
        return False
    width = x.shape[-1]
    if width <= 0 or (width % 128) != 0:
        return False
    return (width // 2) <= MAX_SILU_MUL_F8_DIM


def _silu_mul_f8_producer(x: torch.Tensor, linear: LinearBase) -> QuantizedActivation:
    orig_shape = x.shape[:-1] + (x.shape[-1] // 2, )
    x_2d = x.reshape(-1, x.shape[-1]).contiguous()
    out_f8, out_sc = torch.ops.custom_op.gaudi2_silu_mul_f8(x_2d, float(hpu_ops.FP8_MAX))
    return QuantizedActivation(
        data=out_f8,
        scale=out_sc,
        orig_dtype=x.dtype,
        orig_shape=orig_shape,
        quant_key=kFp8DynamicTokenSym,
    )


def register_fused_silu_mul_f8() -> bool:
    """Register the Gaudi2 SiluAndMul+FP8 dynamic per-token quant producer with vLLM's fusion registry."""
    global _SILU_REGISTERED
    if _SILU_REGISTERED:
        return True
    if not has_silu_mul_f8():
        return False
    key = (SiluAndMul, kFp8DynamicTokenSym)
    fused_act_quant._FUSED_ACT_QUANT[key] = _silu_mul_f8_producer
    fused_act_quant._FUSED_ACT_QUANT_SUPPORT[key] = _silu_mul_f8_supported
    _SILU_REGISTERED = True
    return True


def can_fuse_add_rms_norm_f8(norm_mod: torch.nn.Module, hidden_size: int) -> bool:
    """Return whether ``norm_mod`` can run through ``gaudi2_add_rms_norm_f8``."""
    if not isinstance(norm_mod, (RMSNorm, GemmaRMSNorm)):
        return False
    if getattr(norm_mod, "has_weight", True) is False or getattr(norm_mod, "weight", None) is None:
        return False
    if not isinstance(norm_mod, GemmaRMSNorm) and norm_mod.weight.dtype not in (torch.bfloat16, torch.float32):
        return False
    return hidden_size > 0 and (hidden_size % 64) == 0 and hidden_size <= MAX_RMS_NORM_F8_DIM


def can_fuse_sandwich_rms_norm_f8(post_norm: torch.nn.Module, pre_norm: torch.nn.Module, hidden_size: int) -> bool:
    """Return whether ``post_norm`` + residual add + ``pre_norm`` can run through ``gaudi2_sandwich_rms_norm_f8``."""
    if isinstance(post_norm, GemmaRMSNorm) or isinstance(pre_norm, GemmaRMSNorm):
        return False
    if not isinstance(post_norm, RMSNorm) or not isinstance(pre_norm, RMSNorm):
        return False
    if getattr(post_norm, "weight", None) is None or getattr(pre_norm, "weight", None) is None:
        return False
    if post_norm.weight.dtype != torch.bfloat16 or pre_norm.weight.dtype != torch.bfloat16:
        return False
    if float(post_norm.variance_epsilon) != float(pre_norm.variance_epsilon):
        return False
    return hidden_size > 0 and (hidden_size % 64) == 0 and hidden_size <= MAX_RMS_NORM_F8_DIM


def _norm_weight_and_eps(norm_mod: torch.nn.Module) -> tuple[torch.Tensor, float]:
    if isinstance(norm_mod, GemmaRMSNorm):
        w = getattr(norm_mod, "gemma_weight", None)
        if w is None:
            w = norm_mod.weight.float() + 1.0
        return w, float(norm_mod.variance_epsilon)
    return norm_mod.weight, float(norm_mod.variance_epsilon)


def fused_add_rms_norm_f8(
    norm_mod: torch.nn.Module,
    x: torch.Tensor,
    residual: torch.Tensor,
) -> tuple[QuantizedActivation, torch.Tensor]:
    """Fused residual add + RMSNorm/GemmaRMSNorm + per-row FP8 E4M3 dynamic quantization."""
    orig_shape = x.shape
    d = orig_shape[-1]
    weight, eps = _norm_weight_and_eps(norm_mod)
    out_f8, out_sc, out_res = torch.ops.custom_op.gaudi2_add_rms_norm_f8(
        x.reshape(-1, d).contiguous(),
        residual.reshape(-1, d).contiguous(),
        weight,
        eps,
        float(hpu_ops.FP8_MAX),
    )
    qa = QuantizedActivation(
        data=out_f8,
        scale=out_sc,
        orig_dtype=x.dtype,
        orig_shape=orig_shape,
        quant_key=kFp8DynamicTokenSym,
    )
    return qa, out_res.reshape(residual.shape)


def fused_add_rms_norm_f8_bf16(
    norm_mod: torch.nn.Module,
    x: torch.Tensor,
    residual: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Dual-output fused add + RMSNorm + FP8 quant returning plain tensors ``(out_f8, out_sc, out_res, out_bf16)``."""
    orig_shape = x.shape
    d = orig_shape[-1]
    weight, eps = _norm_weight_and_eps(norm_mod)
    out_f8, out_sc, out_res, out_bf16 = torch.ops.custom_op.gaudi2_add_rms_norm_f8_bf16(
        x.reshape(-1, d).contiguous(),
        residual.reshape(-1, d).contiguous(),
        weight,
        eps,
        float(hpu_ops.FP8_MAX),
    )
    return out_f8, out_sc, out_res.reshape(residual.shape), out_bf16.reshape(orig_shape)


def fused_sandwich_rms_norm_f8(
    post_norm: torch.nn.Module,
    pre_norm: torch.nn.Module,
    x: torch.Tensor,
    residual: torch.Tensor,
) -> tuple[QuantizedActivation, torch.Tensor]:
    """Fused post-RMSNorm + residual add + pre-RMSNorm + per-row FP8 E4M3 dynamic quantization."""
    orig_shape = x.shape
    d = orig_shape[-1]
    out_f8, out_sc, out_res = torch.ops.custom_op.gaudi2_sandwich_rms_norm_f8(
        x.reshape(-1, d).contiguous(),
        residual.reshape(-1, d).contiguous(),
        post_norm.weight,
        pre_norm.weight,
        float(pre_norm.variance_epsilon),
        float(hpu_ops.FP8_MAX),
    )
    qa = QuantizedActivation(
        data=out_f8,
        scale=out_sc,
        orig_dtype=x.dtype,
        orig_shape=orig_shape,
        quant_key=kFp8DynamicTokenSym,
    )
    return qa, out_res.reshape(residual.shape)


register_fused_silu_mul_f8()
