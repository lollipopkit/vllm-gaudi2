# SPDX-License-Identifier: Apache-2.0
"""FP8 weight halving (scale_adjustment) is bit-exact with the CPU cast it replaces."""
import pytest
import torch


@pytest.mark.parametrize("device", ["cpu"] + (["hpu"] if hasattr(torch, "hpu") and torch.hpu.is_available() else []))
def test_halve_fp8_bytes_matches_cpu_cast(device):
    from vllm_gaudi.extension.ops import _halve_fp8_bytes
    all_bytes = torch.arange(256, dtype=torch.uint8).view(torch.float8_e4m3fn)
    ref = (all_bytes.float() * 0.5).to(torch.float8_e4m3fn).view(torch.uint8)
    got = _halve_fp8_bytes(all_bytes.to(device).reshape(16, 16), chunk=100).cpu().view(torch.uint8).reshape(-1)
    assert torch.equal(got, ref)
