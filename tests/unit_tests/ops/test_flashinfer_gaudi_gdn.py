# SPDX-License-Identifier: Apache-2.0
# Partly derived from 1CatAI/1Cat-vLLM-Gaudi (Apache-2.0) at e2266e0.
"""CPU tests for the FlashInfer-Gaudi GDN tactics and their vLLM adapter.

The direct-state recipes must match the general indexed HPU decode chain
(gating + causal_conv1d_update + fused recurrent) on the same rows of a
group-major compact pool, and must leave every other row untouched.
"""

from __future__ import annotations

from unittest import mock

import pytest
import torch
import torch.nn.functional as F

from flashinfer_gaudi._reference import l2_normalize_rsqrt
from vllm_gaudi.ops.causal_conv1d_pytorch import hpu_causal_conv1d_update
from vllm_gaudi.ops.flashinfer_gaudi_adapter import (
    conv_taps,
    direct_state_rows,
    maybe_run_gdn_decode_packed,
    maybe_run_gdn_fused_decode_step,
    maybe_run_gdn_prefill,
)
from vllm_gaudi.ops.hpu_gdn_pytorch import (
    hpu_chunk_gated_delta_rule,
    hpu_fused_gdn_gating,
    hpu_fused_recurrent_gated_delta_rule,
)
from vllm_gaudi.v1.worker.gdn_state_layout import compact_gdn_state_index, has_direct_gdn_state

Q_HEADS, V_HEADS, DIM = 16, 48, 128
WIDTH = (2 * Q_HEADS + V_HEADS) * DIM


def _reference_decode(packed, a, b, A_log, dt_bias, conv_pool, conv_weight, conv_bias, ssm_pool, indices, heads):
    """The general indexed decode chain used by HPUGatedDeltaNetAttention."""
    q_heads, v_heads = heads
    g, beta = hpu_fused_gdn_gating(A_log, a, b, dt_bias)
    batch = packed.shape[0]
    convolved = hpu_causal_conv1d_update(
        x=packed,
        conv_state=conv_pool,
        weight=conv_weight,
        bias=conv_bias,
        activation="silu",
        conv_state_indices=indices,
        query_start_loc=torch.arange(batch + 1, dtype=torch.int32),
    )
    q, k, v = convolved.split([q_heads * DIM, q_heads * DIM, v_heads * DIM], dim=-1)
    output, _ = hpu_fused_recurrent_gated_delta_rule(
        q=q.reshape(1, batch, q_heads, DIM),
        k=k.reshape(1, batch, q_heads, DIM),
        v=v.reshape(1, batch, v_heads, DIM),
        g=g,
        beta=beta,
        initial_state=ssm_pool,
        inplace_final_state=True,
        cu_seqlens=torch.arange(batch + 1, dtype=torch.int32),
        ssm_state_indices=indices,
        use_qk_l2norm_in_kernel=True,
    )
    return output, convolved, g, beta


def _decode_inputs(batch, groups, max_reqs, seed, with_bias=True):
    gen = torch.Generator().manual_seed(seed)
    slots = groups * max_reqs + 2
    return dict(
        packed=torch.randn(batch, WIDTH, generator=gen).to(torch.bfloat16) * 0.1,
        a=(torch.randn(batch, V_HEADS, generator=gen) * 0.1).to(torch.bfloat16),
        b=(torch.randn(batch, V_HEADS, generator=gen) * 0.1).to(torch.bfloat16),
        A_log=torch.randn(V_HEADS, generator=gen) * 0.1 - 2.0,
        dt_bias=(torch.randn(V_HEADS, generator=gen) * 0.1).to(torch.bfloat16),
        conv_weight=(torch.randn(WIDTH, 4, generator=gen) * 0.05).to(torch.bfloat16),
        conv_bias=(torch.randn(WIDTH, generator=gen) * 0.01).to(torch.bfloat16) if with_bias else None,
        conv_pool=(torch.randn(slots, 3, WIDTH, generator=gen) * 0.1).to(torch.bfloat16),
        ssm_pool=torch.randn(slots, V_HEADS, DIM, DIM, generator=gen) * 0.01,
    )


@pytest.mark.parametrize("magnitude", [0.0, 1e-6, 1e-4, 0.1, 1.0])
def test_gdn_normalization_adds_epsilon_to_squared_norm(magnitude):
    value = torch.tensor([[1.0, -2.0, 3.0, -4.0]], dtype=torch.float32) * magnitude
    expected = value / torch.sqrt(torch.sum(value.square(), dim=-1, keepdim=True) + 1e-6)
    actual = l2_normalize_rsqrt(value)
    torch.testing.assert_close(actual, expected, rtol=2e-7, atol=1e-8)
    if 0 < magnitude <= 1e-4:
        assert not torch.allclose(actual, F.normalize(value, dim=-1, eps=1e-6))


def test_group_major_index_and_direct_rows_agree():
    groups, max_reqs, rows = 3, 8, 4
    pool = torch.arange(groups * max_reqs + 2)
    for offset in range(groups):
        view = direct_state_rows(pool, rows, groups, offset)
        expected = torch.tensor([compact_gdn_state_index(offset, slot, max_reqs) for slot in range(rows)])
        torch.testing.assert_close(view, expected)
    assert direct_state_rows(pool, max_reqs + 1, groups, 0) is None
    assert direct_state_rows(pool, rows, groups, groups) is None
    assert direct_state_rows(pool, rows, None, 0) is None


def test_direct_state_proof_requires_input_order_and_free_padding():
    groups, max_reqs = {0: 0, 3: 1}, 8
    rows = 4
    indices = torch.zeros(4, rows, dtype=torch.int32)
    for group_idx, offset in groups.items():
        indices[group_idx] = torch.tensor([compact_gdn_state_index(offset, s, max_reqs) for s in range(rows)])
    assert has_direct_gdn_state(indices, groups, max_reqs, rows, rows)

    swapped = indices.clone()
    swapped[3, [0, 1]] = swapped[3, [1, 0]]
    assert not has_direct_gdn_state(swapped, groups, max_reqs, rows, rows)
    # Padded batches mutate the padding rows of the contiguous view: allowed only onto free slots.
    padded = indices.clone()
    padded[:, rows - 1] = -1
    assert not has_direct_gdn_state(padded, groups, max_reqs, rows - 1, rows)
    assert not has_direct_gdn_state(padded, groups, max_reqs, rows - 1, rows, free_slots={rows - 2})
    assert has_direct_gdn_state(padded, groups, max_reqs, rows - 1, rows, free_slots={rows - 1})
    assert not has_direct_gdn_state(indices, {}, max_reqs, rows, rows)
    assert not has_direct_gdn_state(indices, groups, rows - 1, rows, rows)


@pytest.mark.parametrize("batch", [1, 3, 8])  # 3: decode buckets include sizes outside powers of two
@pytest.mark.parametrize("with_bias", [False, True])
def test_fused_direct_step_matches_indexed_decode_chain(batch, with_bias):
    groups, max_reqs, offset = 2, 8, 1
    inputs = _decode_inputs(batch, groups, max_reqs, seed=127 + batch, with_bias=with_bias)
    indices = torch.tensor([compact_gdn_state_index(offset, s, max_reqs) for s in range(batch)], dtype=torch.int32)

    ref_conv, ref_ssm = inputs["conv_pool"].clone(), inputs["ssm_pool"].clone()
    expected, _, _, _ = _reference_decode(inputs["packed"], inputs["a"], inputs["b"], inputs["A_log"],
                                          inputs["dt_bias"], ref_conv, inputs["conv_weight"], inputs["conv_bias"],
                                          ref_ssm, indices, (Q_HEADS, V_HEADS))

    conv_pool, ssm_pool = inputs["conv_pool"].clone(), inputs["ssm_pool"].clone()
    result = maybe_run_gdn_fused_decode_step(
        inputs["packed"],
        inputs["a"],
        inputs["b"],
        inputs["A_log"],
        inputs["dt_bias"],
        conv_pool,
        conv_taps(inputs["conv_weight"]),
        inputs["conv_bias"],
        ssm_pool,
        direct_state_group_count=groups,
        direct_state_group_offset=offset,
        scale=DIM**-0.5,
    )
    assert result is not None
    output, updated = result
    assert output.shape == (1, batch, V_HEADS, DIM)
    assert updated.untyped_storage().data_ptr() == ssm_pool.untyped_storage().data_ptr()

    torch.testing.assert_close(output, expected, atol=2e-3, rtol=2e-2)
    torch.testing.assert_close(conv_pool, ref_conv, atol=0, rtol=0)
    torch.testing.assert_close(ssm_pool, ref_ssm, atol=2e-5, rtol=2e-4)
    # Rows outside the active span are untouched.
    untouched = torch.ones(ssm_pool.shape[0], dtype=torch.bool)
    untouched[indices.long()] = False
    torch.testing.assert_close(ssm_pool[untouched], inputs["ssm_pool"][untouched], atol=0, rtol=0)


def test_fused_direct_step_rejects_bf16_state():
    inputs = _decode_inputs(1, 1, 4, seed=6)
    result = maybe_run_gdn_fused_decode_step(
        inputs["packed"],
        inputs["a"],
        inputs["b"],
        inputs["A_log"],
        inputs["dt_bias"],
        inputs["conv_pool"],
        conv_taps(inputs["conv_weight"]),
        inputs["conv_bias"],
        inputs["ssm_pool"].to(torch.bfloat16),
        direct_state_group_count=1,
        direct_state_group_offset=0,
        scale=DIM**-0.5,
    )
    assert result is None


@pytest.mark.parametrize(("q_heads", "v_heads", "dim", "batch"), [(16, 48, 128, 2), (2, 4, 8, 3)])
def test_packed_direct_decode_matches_indexed_recurrent(q_heads, v_heads, dim, batch):
    gen = torch.Generator().manual_seed(53)
    groups, max_reqs, offset = 3, 4, 2
    width = (2 * q_heads + v_heads) * dim
    packed = torch.randn(batch, width, generator=gen).to(torch.bfloat16)
    g = -torch.rand(1, batch, v_heads, generator=gen)
    beta = torch.sigmoid(torch.randn(1, batch, v_heads, generator=gen)).to(torch.bfloat16)
    pool = torch.randn(groups * max_reqs + 2, v_heads, dim, dim, generator=gen)
    indices = torch.tensor([compact_gdn_state_index(offset, s, max_reqs) for s in range(batch)], dtype=torch.int32)

    ref_pool = pool.clone()
    q, k, v = packed.split([q_heads * dim, q_heads * dim, v_heads * dim], dim=-1)
    expected, _ = hpu_fused_recurrent_gated_delta_rule(
        q=q.reshape(1, batch, q_heads, dim),
        k=k.reshape(1, batch, q_heads, dim),
        v=v.reshape(1, batch, v_heads, dim),
        g=g,
        beta=beta,
        initial_state=ref_pool,
        inplace_final_state=True,
        cu_seqlens=torch.arange(batch + 1, dtype=torch.int32),
        ssm_state_indices=indices,
        use_qk_l2norm_in_kernel=True,
    )

    result = maybe_run_gdn_decode_packed(
        packed,
        g,
        beta,
        pool,
        use_qk_l2norm=True,
        scale=dim**-0.5,
        direct_state_group_count=groups,
        direct_state_group_offset=offset,
    )
    assert result is not None
    torch.testing.assert_close(result[0], expected, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(pool, ref_pool, atol=2e-5, rtol=2e-4)


def test_prefill_adapter_selects_promoted_tactic():
    tokens = 64
    q = torch.zeros(1, tokens, Q_HEADS, DIM, dtype=torch.bfloat16)
    v = torch.zeros(1, tokens, V_HEADS, DIM, dtype=torch.bfloat16)
    log_decay = torch.zeros(1, tokens, V_HEADS, dtype=torch.float32)
    beta = torch.ones(1, tokens, V_HEADS, dtype=torch.bfloat16)
    state = torch.zeros(1, V_HEADS, DIM, DIM)
    expected = (torch.empty_like(v), state.clone())
    with mock.patch("vllm_gaudi.ops.flashinfer_gaudi_adapter.chunk_gated_delta_rule_log_gate",
                    return_value=expected) as run:
        result = maybe_run_gdn_prefill(q,
                                       q.clone(),
                                       v,
                                       log_decay,
                                       beta,
                                       state,
                                       output_final_state=True,
                                       use_qk_l2norm_in_kernel=True,
                                       chunk_size=128,
                                       prefill_num_seqs=1,
                                       prefill_seq_len=tokens)
    assert result is expected
    kwargs = run.call_args.kwargs
    assert kwargs["flashqla_reformulation"] is True
    assert kwargs["deferred_output_add"] is True
    assert kwargs["fused_state_matmul"] is True
    assert kwargs["recursive_solver_base"] == 16
    assert kwargs["compact_repeated_kkt"] is True
    assert kwargs["solve_in_fp32"] is True
    assert kwargs["state_in_fp32"] is True
    assert kwargs["compute_dtype"] == torch.float32
    assert kwargs["preserve_compact_qk"] is True
    assert kwargs["masked_triangular_decay"] is True


@pytest.mark.parametrize(("q_heads", "v_heads", "chunk", "seqs"), [(8, 24, 128, 1), (16, 48, 64, 1), (16, 48, 128, 2)])
def test_prefill_adapter_falls_back_for_unqualified_shapes(q_heads, v_heads, chunk, seqs):
    tokens = 64
    q = torch.zeros(1, tokens, q_heads, DIM, dtype=torch.bfloat16)
    v = torch.zeros(1, tokens, v_heads, DIM, dtype=torch.bfloat16)
    result = maybe_run_gdn_prefill(q,
                                   q.clone(),
                                   v,
                                   torch.zeros(1, tokens, v_heads),
                                   torch.ones(1, tokens, v_heads, dtype=torch.bfloat16),
                                   None,
                                   output_final_state=True,
                                   use_qk_l2norm_in_kernel=True,
                                   chunk_size=chunk,
                                   prefill_num_seqs=seqs,
                                   prefill_seq_len=tokens // seqs)
    assert result is None


def test_prefill_tactic_matches_general_path():
    gen = torch.Generator().manual_seed(11)
    tokens = 160  # two chunks, the second one padded
    q = torch.randn(1, tokens, Q_HEADS, DIM, generator=gen).to(torch.bfloat16)
    k = torch.randn(1, tokens, Q_HEADS, DIM, generator=gen).to(torch.bfloat16)
    v = (torch.randn(1, tokens, V_HEADS, DIM, generator=gen) * 0.5).to(torch.bfloat16)
    g = -torch.rand(1, tokens, V_HEADS, generator=gen) * 0.5
    beta = torch.sigmoid(torch.randn(1, tokens, V_HEADS, generator=gen)).to(torch.bfloat16)
    state = torch.randn(1, V_HEADS, DIM, DIM, generator=gen) * 0.01
    common = dict(initial_state=state,
                  output_final_state=True,
                  use_qk_l2norm_in_kernel=True,
                  chunk_size=128,
                  prefill_num_seqs=1,
                  prefill_seq_len=tokens)

    expected_out, expected_state = hpu_chunk_gated_delta_rule(q=q, k=k, v=v, g=g, beta=beta, **common)
    result = maybe_run_gdn_prefill(q, k, v, g, beta, scale=DIM**-0.5, **common)
    assert result is not None
    actual_out, actual_state = result
    assert torch.isfinite(actual_out.float()).all() and torch.isfinite(actual_state).all()
    torch.testing.assert_close(actual_out.float(), expected_out.float(), atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(actual_state, expected_state, atol=2e-3, rtol=2e-2)
