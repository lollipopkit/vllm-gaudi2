# SPDX-License-Identifier: Apache-2.0
# Derived from 1CatAI/1Cat-vLLM-Gaudi (Apache-2.0) at e2266e0.
"""Runtime configuration for the Gaudi FlashInfer compatibility layer."""

from __future__ import annotations

import os
from typing import Literal

BackendPolicy = Literal["auto"]

_BACKEND_ENV = "FLASHINFER_GAUDI_BACKEND"
# The native/public/bridge policies of the source fork need TPC libraries that
# are not part of this port; reject them instead of silently using PyTorch.
_VALID_BACKENDS = frozenset(("auto", ))


def get_backend_policy() -> BackendPolicy:
    """Return the process-wide backend policy (only ``auto`` is supported)."""
    value = os.environ.get(_BACKEND_ENV, "auto").strip().lower()
    if value not in _VALID_BACKENDS:
        raise ValueError(f"Invalid {_BACKEND_ENV}={value!r}; only 'auto' (compiled PyTorch tactics) is available.")
    return "auto"
