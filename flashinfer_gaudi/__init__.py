# SPDX-License-Identifier: Apache-2.0
# Derived from 1CatAI/1Cat-vLLM-Gaudi (Apache-2.0) at e2266e0, trimmed to the
# compiled PyTorch tactics used by the Qwen3.5/3.8 TP1 GDN path.
"""Gaudi graph tactics for FlashInfer-compatible GDN operations.

Only the compile-friendly PyTorch tactics are present. Native TPC kernels,
the private-ABI bridge, MTP/DFlash2 and the public FlashInfer API surface
from the source fork are not included.
"""

from __future__ import annotations

from flashinfer_gaudi._config import get_backend_policy

__version__ = "0.1.0"

__all__ = ["get_backend_policy"]
