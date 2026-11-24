# SPDX-License-Identifier: Apache-2.0
"""Symbolic shape helpers that keep decode graphs from specializing on the batch size."""
import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.experimental.symbolic_shapes import ShapeEnv

from vllm_gaudi.patches import _fp8_gemm_v2_out_shape, _patch_fp8_gemm_v2_meta


@pytest.mark.parametrize("sa, ta, sb, tb", [
    ((12, 256), False, (256, 64), False),
    ((12, 256), False, (64, 256), True),
    ((256, 12), True, (256, 64), False),
    ((4, 12, 128), False, (4, 128, 64), False),
    ((4, 12, 128), False, (128, 64), False),
    ((2, 4, 12, 128), False, (4, 64, 128), True),
])
def test_fp8_gemm_v2_out_shape_matches_matmul(sa, ta, sb, tb):
    a, b = torch.empty(sa, device="meta"), torch.empty(sb, device="meta")
    ref = torch.matmul(a.transpose(-1, -2) if ta else a, b.transpose(-1, -2) if tb else b).shape
    assert list(_fp8_gemm_v2_out_shape(a, ta, b, tb)) == list(ref)


def test_fp8_gemm_v2_meta_keeps_batch_symbolic():
    if not hasattr(torch.ops.hpu, "fp8_gemm_v2"):
        pytest.skip("habana_frameworks without fp8_gemm_v2")
    _patch_fp8_gemm_v2_meta()
    shape_env = ShapeEnv()
    with FakeTensorMode(shape_env=shape_env):
        bs = shape_env.create_unbacked_symint()
        out = torch.ops.hpu.fp8_gemm_v2(A=torch.empty(bs, 256, dtype=torch.float8_e4m3fn), trans_A=False,
                                        B=torch.empty(64, 256, dtype=torch.float8_e4m3fn), trans_B=True, D=None,
                                        out_dtype=torch.bfloat16)
    assert isinstance(out.shape[0], torch.SymInt) and out.shape[1] == 64 and out.dtype == torch.bfloat16
    assert not shape_env.guards

