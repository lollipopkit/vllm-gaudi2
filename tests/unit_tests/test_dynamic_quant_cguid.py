# SPDX-License-Identifier: Apache-2.0
# Derived from 1CatAI/1Cat-vLLM-Gaudi (Apache-2.0) at e2266e0.
"""VLLM_HPU_CGUID_DYNAMIC_QUANT selection in extension.ops.dynamic_quant."""

from unittest import mock

import torch

from vllm_gaudi.extension import ops


def _patch(monkeypatch, enabled=True, max_rows=32):
    calculate = mock.Mock(side_effect=lambda data, *args: torch.zeros(data.shape[0], 1, dtype=data.dtype))
    cast = mock.Mock(side_effect=lambda data, *args: (data, ))
    monkeypatch.setattr(ops, "_CGUID_DYNAMIC_QUANT", enabled)
    monkeypatch.setattr(ops, "_CGUID_DYNAMIC_QUANT_MAX_ROWS", max_rows)
    monkeypatch.setattr(ops.torch.ops.hpu, "calculate_scale_for_cast", calculate)
    monkeypatch.setattr(ops.torch.ops.hpu, "cast_to_fp8_v2", cast)
    return calculate


def test_decode_rows_use_cguid_and_keep_zero_row_epsilon(monkeypatch):
    calculate = _patch(monkeypatch)
    data = torch.zeros((2, 8), dtype=torch.bfloat16)

    quantized, scale = ops.dynamic_quant(data)

    assert quantized is data
    calculate.assert_called_once_with(data, 2, 0, -1, True, float(ops.FP8_MAX), 1.0)
    torch.testing.assert_close(scale, torch.full((2, 1), 1e-8 / ops.FP8_MAX, dtype=torch.bfloat16).float())


def test_prefill_rows_keep_abs_max_path(monkeypatch):
    calculate = _patch(monkeypatch, max_rows=1)
    data = torch.tensor([[1.0, -2.0], [3.0, 0.5]], dtype=torch.bfloat16)

    _, scale = ops.dynamic_quant(data)

    calculate.assert_not_called()
    expected = (data.abs().max(dim=-1).values + 1e-8) / ops.FP8_MAX
    torch.testing.assert_close(scale, expected.unsqueeze(-1).float())


def test_disabled_flag_and_weight_path_bypass_cguid(monkeypatch):
    data = torch.tensor([[1.0, -2.0]], dtype=torch.bfloat16)
    calculate = _patch(monkeypatch, enabled=False)
    ops.dynamic_quant(data)
    calculate.assert_not_called()

    calculate = _patch(monkeypatch, enabled=True)
    ops.dynamic_quant(data, use_cguid=False)
    calculate.assert_not_called()
