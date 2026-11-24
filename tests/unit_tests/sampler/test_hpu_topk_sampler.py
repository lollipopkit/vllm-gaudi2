# SPDX-License-Identifier: Apache-2.0
"""The capped top-k/top-p sampler keeps exactly the native path's candidate distribution."""
import pytest
import torch

from vllm_gaudi.v1._vendored.sample.ops.topk_topp_sampler import apply_top_k_top_p_pytorch
from vllm_gaudi.ops import hpu_sampler


def _capped_probs(logits, k, p, cap):
    vals, idx = hpu_sampler.capped_candidates(logits, k, p, cap)
    out = torch.zeros_like(logits, dtype=torch.float32)
    return out.scatter_(1, idx.long(), vals.softmax(dim=-1, dtype=torch.float32))


@pytest.mark.parametrize("vocab", [5000, 256 * 97])  # single topk; two-stage grouped topk
@pytest.mark.parametrize("use_p", [True, False])
def test_candidate_distribution_matches_native(use_p, vocab):
    torch.manual_seed(0)
    logits = torch.randn(16, vocab) * 3
    k = torch.randint(1, hpu_sampler.TOPK_CAP + 1, (16, ), dtype=torch.int32)
    p = torch.rand(16) * 0.9 + 0.05 if use_p else None
    native = apply_top_k_top_p_pytorch(logits.clone(), k, p).softmax(dim=-1, dtype=torch.float32)
    torch.testing.assert_close(_capped_probs(logits, k, p, hpu_sampler.TOPK_CAP), native, atol=1e-6, rtol=1e-5)


def test_sample_capped_draws_only_candidates():
    torch.manual_seed(1)
    logits = torch.randn(8, 3000)
    k = torch.full((8, ), 20, dtype=torch.int32)
    p = torch.full((8, ), 0.95)
    allowed = apply_top_k_top_p_pytorch(logits.clone(), k, p) > -float("inf")
    for _ in range(50):
        tok = hpu_sampler.sample_capped(logits, k, p, hpu_sampler.TOPK_CAP, {})
        assert allowed[torch.arange(8), tok].all()


class _Batch:

    def __init__(self, top_k, n):
        self.num_reqs = n
        self.top_k_cpu = top_k
        self.top_k_reqs = set(range(sum(0 < x < 1000 for x in top_k[:n])))


def test_topk_cap_gate():
    import numpy as np
    assert hpu_sampler.topk_cap(_Batch(np.array([20, 20, 0]), 2)) == hpu_sampler.TOPK_CAP
    assert hpu_sampler.topk_cap(_Batch(np.array([20, 100]), 2)) is None
    assert hpu_sampler.topk_cap(_Batch(np.array([20, 1000]), 2)) is None  # one row without top-k
