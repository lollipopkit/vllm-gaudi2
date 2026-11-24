# SPDX-License-Identifier: Apache-2.0
# Derived from 1CatAI/1Cat-vLLM-Gaudi (Apache-2.0) at e2266e0
# (flashinfer_gaudi/tactics/gaudi2_qwen38_gdn.json, schema 4).
"""Offline-promoted Gaudi2 tactics for deterministic serving startup."""

from __future__ import annotations

import copy
from typing import Any

_MANIFEST: dict[str, Any] = {
    "schema_version": 4,
    "device": "gaudi2",
    "gdn_fused_decode": {
        "promoted": True,
        "implementation": "qwen38_direct_graph",
        "model_shape": {
            "tensor_parallel_size": 1,
            "key_heads": 16,
            "value_heads": 48,
            "key_dim": 128,
            "value_dim": 128,
            "packed_qkv_dim": 10240,
            "conv_width": 4,
            "input_dtype": "bfloat16",
            "state_dtype": "float32",
            "state_layout": "VK",
            "direct_state_layout": True,
        },
        "tuning": {
            "static_manual_conv": True,
            "recurrent_state_update": "out_of_place_then_copy",
        },
    },
    "gdn_prefill": {
        "promoted": True,
        "implementation": "flashqla_graph",
        "model_shape": {
            "key_heads": 16,
            "value_heads": 48,
            "key_dim": 128,
            "value_dim": 128,
            "chunk_size": 128,
            "input_dtype": "bfloat16",
            "state_layout": "VK",
            "uniform_sequences": True,
        },
        "tuning": {
            "neumann_iters": 14,
            "fused_state_matmul": True,
            "recursive_solver_base": 16,
            "compact_repeated_kkt": True,
            "flashqla_reformulation": True,
            "deferred_output_add": True,
            "solve_in_fp32": True,
            "state_in_fp32": True,
            "compute_dtype": "float32",
            "preserve_compact_qk": True,
            "masked_triangular_decay": True,
        },
    },
}


def gdn_prefill_tactic() -> dict[str, Any]:
    """Return the offline-promoted Gaudi2 GDN prefill tactic."""
    return copy.deepcopy(_MANIFEST["gdn_prefill"])


def gdn_fused_decode_tactic() -> dict[str, Any]:
    """Return the offline-promoted Gaudi2 fused GDN decode tactic."""
    return copy.deepcopy(_MANIFEST["gdn_fused_decode"])
