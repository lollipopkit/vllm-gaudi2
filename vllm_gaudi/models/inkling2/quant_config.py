# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Quantization config for Inkling2 on Gaudi2 HPU (FP8 E4M3 per-channel)."""

from __future__ import annotations

from typing import Any

import torch

from vllm.model_executor.layers.quantization import QuantizationMethods
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)
from .configs import INKLING2_ARCHITECTURES

GAUDI2_FP8_MAX = 240.0


class Inkling2FP8QuantConfig(QuantizationConfig):
    """Per-channel FP8 E4M3 quantization for Inkling2 routed experts on Gaudi2.

    Produced by `tools/convert_inkling2_fp8.py` from the upstream NVFP4/MXFP8
    checkpoint. Only the routed experts are quantized; attention, short-conv,
    loop controllers, vision encoder, and embeddings remain BF16/FP32.
    """

    def __init__(
        self,
        fp8_max: float = GAUDI2_FP8_MAX,
        scheme: str = "per_channel",
    ) -> None:
        super().__init__()
        self.fp8_max = float(fp8_max)
        self.scheme = scheme

    @classmethod
    def get_name(cls) -> QuantizationMethods:
        return "inkling2_fp8"  # type: ignore[return-value]

    @classmethod
    def get_supported_act_dtypes(cls) -> list[torch.dtype]:
        return [torch.bfloat16]

    @classmethod
    def get_min_capability(cls) -> int:
        return 0

    @staticmethod
    def get_config_filenames() -> list[str]:
        return []

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> Inkling2FP8QuantConfig:
        return cls(
            fp8_max=config.get("fp8_max", GAUDI2_FP8_MAX),
            scheme=config.get("scheme", "per_channel"),
        )

    @classmethod
    def override_quantization_method(
        cls,
        hf_quant_cfg: dict[str, Any],
        user_quant: str | None,
        hf_config: Any = None,
    ) -> QuantizationMethods | None:
        architectures = getattr(hf_config, "architectures", None) or []
        if not (INKLING2_ARCHITECTURES & set(architectures)):
            return None
        method = hf_quant_cfg.get("quant_method")
        if method == "inkling2_fp8":
            return "inkling2_fp8"  # type: ignore[return-value]
        if method == "modelopt" and hf_quant_cfg.get("quant_algo") == "NVFP4":
            raise ValueError(
                "Raw Inkling2 NVFP4/MXFP8 checkpoints cannot run directly on Gaudi2. "
                "Convert the checkpoint to Gaudi2 FP8 first with:\n"
                "  python tools/convert_inkling2_fp8.py <src_dir> <dst_fp8_dir>"
            )
        return None

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> QuantizeMethodBase | None:
        # Routed experts are handled directly by Inkling2HPUMoE.
        return None
