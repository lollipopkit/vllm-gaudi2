# SPDX-License-Identifier: Apache-2.0
"""CPU tests for the speculative-decode GDN ops.

Every verify output must match feeding the 1 + k tokens one at a time through the non-spec decode ops,
and for every acceptance count a the state the next step reads (conv window at offset a - 1, SSM row
a - 1) must equal the non-spec state after a tokens.
"""

from __future__ import annotations

import pytest
import torch

from vllm_gaudi.ops.causal_conv1d_pytorch import hpu_causal_conv1d_update
from vllm_gaudi.ops.hpu_gdn_pytorch import hpu_fused_recurrent_gated_delta_rule
from vllm_gaudi.ops.hpu_gdn_spec import hpu_causal_conv1d_update_spec, hpu_gdn_spec_verify


@pytest.mark.parametrize("k", [1, 3])
@pytest.mark.parametrize("accepted", ["first", "last", "mixed"])
def test_conv_update_spec_matches_sequential(k, accepted):
    torch.manual_seed(0)
    B, D, width, slots = 3, 16, 4, 6
    T, hist = 1 + k, width - 1
    weight, bias = torch.randn(D, width), torch.randn(D)
    x = torch.randn(B, T, D)
    idx = torch.tensor([4, 1, 2])
    num_acc = {"first": torch.ones(B), "last": torch.full((B, ), T), "mixed": torch.arange(B) % T + 1}[accepted]
    num_acc = num_acc.long()
    spec_state = torch.randn(slots, hist + k, D)
    ref_state = torch.zeros(slots, hist, D)
    for b in range(B):
        a = int(num_acc[b])
        ref_state[idx[b]] = spec_state[idx[b], a - 1:a - 1 + hist]

    out = hpu_causal_conv1d_update_spec(x, spec_state, weight, bias, "silu", idx, num_acc)

    ref_states_after = []
    for t in range(T):
        ref_out = hpu_causal_conv1d_update(x[:, t], ref_state, weight, bias, "silu", conv_state_indices=idx,
                                           query_start_loc=torch.arange(B + 1, dtype=torch.int32))
        torch.testing.assert_close(out[:, t], ref_out, rtol=1e-5, atol=1e-5)
        ref_states_after.append(ref_state[idx].clone())
    for a in range(1, T + 1):
        torch.testing.assert_close(spec_state[idx, a - 1:a - 1 + hist], ref_states_after[a - 1])


def test_conv_update_spec_leaves_other_slots():
    torch.manual_seed(1)
    B, D, width, k, slots = 2, 8, 4, 2, 5
    state = torch.randn(slots, width - 1 + k, D)
    before = state.clone()
    idx = torch.tensor([0, 3])
    hpu_causal_conv1d_update_spec(torch.randn(B, 1 + k, D), state, torch.randn(D, width), None, None, idx,
                                  torch.tensor([1, 3]))
    for s in (1, 2, 4):
        torch.testing.assert_close(state[s], before[s], rtol=0, atol=0)


@pytest.mark.parametrize("k", [1, 2, 3])
def test_gdn_spec_verify_matches_sequential(k):
    torch.manual_seed(2)
    B, H, HV, K, V = 2, 2, 6, 16, 16
    T = 1 + k
    rows = 1 + 2 * T + 2  # sentinel, two requests x T rows, two scratch rows
    q, kk = torch.randn(B, T, H, K), torch.randn(B, T, H, K)
    v = torch.randn(B, T, HV, V)
    g = -torch.rand(B, T, HV)
    beta = torch.rand(B, T, HV)
    scale = K**-0.5
    pool = torch.randn(rows, HV, V, K)
    # Request b owns rows base[b] + t; it accepted num_acc[b] tokens last step.
    base = torch.tensor([1, 1 + T])
    num_acc = torch.tensor([1, T])
    load = base + num_acc - 1
    store = base.unsqueeze(1) + torch.arange(T)

    ref_pool = pool.clone()
    spec_pool = pool.clone()
    out = hpu_gdn_spec_verify(q, kk, v, g, beta, spec_pool, load, store, scale)

    ref_rows = torch.tensor([rows - 1, rows - 2])  # scratch rows for the sequential reference
    ref_pool[ref_rows] = pool[load]
    for t in range(T):
        ref_out, _ = hpu_fused_recurrent_gated_delta_rule(q[:, t:t + 1], kk[:, t:t + 1], v[:, t:t + 1],
                                                          g[:, t:t + 1], beta[:, t:t + 1], scale=scale,
                                                          initial_state=ref_pool, inplace_final_state=True,
                                                          ssm_state_indices=ref_rows,
                                                          use_qk_l2norm_in_kernel=True)
        torch.testing.assert_close(out[:, t], ref_out[:, 0], rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(spec_pool[store[:, t]], ref_pool[ref_rows], rtol=1e-5, atol=1e-5)
