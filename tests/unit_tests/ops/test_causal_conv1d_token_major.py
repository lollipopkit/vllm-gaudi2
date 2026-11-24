# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The token-major prefill conv path matches the channel-first path (output and conv states)."""

from __future__ import annotations

import pytest
import torch

from vllm_gaudi.ops import causal_conv1d_pytorch as conv


def _run(token_major, x_tok, weight, bias, states, qsl, idx, has_init):
    # A channel-last view takes the token-major path, a contiguous (dim, T) tensor the channel-first one.
    x = x_tok.transpose(0, 1) if token_major else x_tok.transpose(0, 1).contiguous()
    states = states.clone()
    out = conv.hpu_causal_conv1d_fn(x=x, weight=weight, bias=bias, conv_states=states,
                                    query_start_loc=qsl, cache_indices=idx, has_initial_state=has_init,
                                    initial_state_idx=idx, activation="silu")
    return out.transpose(0, 1), states


@pytest.mark.parametrize("batch,seq_len,lens", [(1, 16, [16]), (3, 8, [8, 5, 0])])
def test_token_major_matches_channel_first(batch, seq_len, lens):
    torch.manual_seed(0)
    dim, width, slots = 24, 4, 6
    x_tok = torch.randn(batch * seq_len, dim)
    weight, bias = torch.randn(dim, width), torch.randn(dim)
    states = torch.randn(slots, width - 1, dim)
    qsl = torch.tensor([0] + [sum(lens[:i + 1]) for i in range(batch)])
    idx = torch.tensor([1, 3, -1][:batch])
    has_init = torch.tensor([1, 0, 1][:batch])
    ref_out, ref_states = _run(False, x_tok, weight, bias, states, qsl, idx, has_init)
    out, new_states = _run(True, x_tok, weight, bias, states, qsl, idx, has_init)
    torch.testing.assert_close(out, ref_out, rtol=0, atol=0)
    torch.testing.assert_close(new_states, ref_states, rtol=0, atol=0)
