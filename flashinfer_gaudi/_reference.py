# SPDX-License-Identifier: Apache-2.0
# Derived from 1CatAI/1Cat-vLLM-Gaudi (Apache-2.0) at e2266e0, reduced to the
# direct (contiguous group-major) TP1 decode recipes.
"""Compile-friendly PyTorch GDN decode recipes for Gaudi."""

from __future__ import annotations

import torch
import torch.nn.functional as F

QWEN38_KEY_HEADS = 16
QWEN38_VALUE_HEADS = 48
QWEN38_DIM = 128
QWEN38_HEAD_REPEAT = QWEN38_VALUE_HEADS // QWEN38_KEY_HEADS
QWEN38_QK_WIDTH = 2 * QWEN38_KEY_HEADS * QWEN38_DIM
QWEN38_PACKED_WIDTH = (QWEN38_VALUE_HEADS + 2 * QWEN38_KEY_HEADS) * QWEN38_DIM


def l2_normalize_rsqrt(value: torch.Tensor) -> torch.Tensor:
    """Normalize with the additive epsilon of the GDN kernel contract.

    This is not ``F.normalize``: clamping the norm changes small Q/K vectors
    and makes decode disagree with prefill.
    """
    norm_squared = torch.sum(value * value, dim=-1, keepdim=True)
    return value * torch.rsqrt(norm_squared + 1e-6)


def _direct_single_token_decode_core(
    q_work: torch.Tensor,
    k_work: torch.Tensor,
    value_work: torch.Tensor,
    log_decay: torch.Tensor,
    beta: torch.Tensor,
    state_pool: torch.Tensor,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Update contiguous state after Q/K conversion and normalization."""
    batch, q_heads, key_dim = q_work.shape
    value_heads, value_dim = value_work.shape[1:]
    repeat = value_heads // q_heads
    output_dtype = value_work.dtype
    k_grouped = k_work.unsqueeze(2)

    state = state_pool.to(torch.float32).reshape(batch, q_heads, repeat, value_dim, key_dim)
    decay = torch.exp(log_decay.reshape(batch, q_heads, repeat).to(torch.float32)).reshape(batch, q_heads, repeat, 1, 1)
    beta_work = beta.reshape(batch, q_heads, repeat).to(torch.float32).unsqueeze(-1)
    value_work = value_work.to(torch.float32).reshape(batch, q_heads, repeat, value_dim)

    # Two width-one projections are faster than one width-two MME on Gaudi2
    # for this state shape. Keep the update order explicit so the compiler can
    # fuse the surrounding decay and rank-one update.
    decayed_state = state * decay
    projection = torch.matmul(decayed_state, k_grouped.unsqueeze(-1)).squeeze(-1)
    delta = (value_work - projection) * beta_work
    updated_state = torch.addcmul(decayed_state, delta.unsqueeze(-1), k_grouped.unsqueeze(-2))
    output = torch.matmul(updated_state, (q_work.unsqueeze(2) * scale).unsqueeze(-1)).squeeze(-1)

    state_pool.copy_(updated_state.reshape_as(state_pool).to(state_pool.dtype))
    return output.reshape(batch, value_heads, value_dim).to(output_dtype), state_pool


def _direct_single_token_packed_decode(
    packed_qkv: torch.Tensor,
    log_decay: torch.Tensor,
    beta: torch.Tensor,
    state_pool: torch.Tensor,
    scale: float | None,
    use_qk_l2norm: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """General-shape packed decode; Q and K share one normalization launch."""
    if packed_qkv.ndim != 2:
        raise ValueError(f"packed_qkv must have [B,width] shape, got {packed_qkv.shape}.")
    if state_pool.ndim != 4:
        raise ValueError(f"state must have [B,HV,V,K] shape, got {state_pool.shape}.")
    batch = packed_qkv.shape[0]
    slots, value_heads, value_dim, key_dim = state_pool.shape
    if slots != batch:
        raise ValueError(f"Direct state requires one contiguous row per request; got slots={slots}, B={batch}.")
    value_width = value_heads * value_dim
    qk_width = packed_qkv.shape[1] - value_width
    if qk_width <= 0 or qk_width % (2 * key_dim):
        raise ValueError(f"Cannot infer Q/K heads from packed width {packed_qkv.shape[1]}, "
                         f"HV={value_heads}, K={key_dim}, V={value_dim}.")
    q_heads = qk_width // (2 * key_dim)
    if value_heads % q_heads:
        raise ValueError(f"Value heads ({value_heads}) must be divisible by inferred Q/K heads ({q_heads}).")

    qk_work = packed_qkv[:, :qk_width].reshape(batch, 2 * q_heads, key_dim).to(torch.float32)
    if use_qk_l2norm:
        qk_work = l2_normalize_rsqrt(qk_work)
    q_work, k_work = qk_work.split(q_heads, dim=1)
    value_work = packed_qkv[:, qk_width:].reshape(batch, value_heads, value_dim)
    return _direct_single_token_decode_core(
        q_work,
        k_work,
        value_work,
        log_decay,
        beta,
        state_pool,
        key_dim**-0.5 if scale is None else scale,
    )


def _direct_qwen38_single_token_packed_decode(
    packed_qkv: torch.Tensor,
    log_decay: torch.Tensor,
    beta: torch.Tensor,
    state_pool: torch.Tensor,
    scale: float,
    use_qk_l2norm: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Keep the production Qwen3.8 TP1 shape static for the graph compiler."""
    batch = packed_qkv.shape[0]
    qk_work = packed_qkv[:, :QWEN38_QK_WIDTH].reshape(batch, 2 * QWEN38_KEY_HEADS, QWEN38_DIM).to(torch.float32)
    if use_qk_l2norm:
        qk_work = l2_normalize_rsqrt(qk_work)
    q_work, k_work = qk_work.split(QWEN38_KEY_HEADS, dim=1)
    k_grouped = k_work.unsqueeze(2)
    value_work = packed_qkv[:, QWEN38_QK_WIDTH:].reshape(batch, QWEN38_KEY_HEADS, QWEN38_HEAD_REPEAT,
                                                         QWEN38_DIM).to(torch.float32)
    state = state_pool.reshape(batch, QWEN38_KEY_HEADS, QWEN38_HEAD_REPEAT, QWEN38_DIM, QWEN38_DIM)
    decay = torch.exp(log_decay.reshape(batch, QWEN38_KEY_HEADS, QWEN38_HEAD_REPEAT).to(torch.float32)).reshape(
        batch, QWEN38_KEY_HEADS, QWEN38_HEAD_REPEAT, 1, 1)
    beta_work = beta.reshape(batch, QWEN38_KEY_HEADS, QWEN38_HEAD_REPEAT).to(torch.float32).unsqueeze(-1)

    decayed_state = state * decay
    projection = torch.matmul(decayed_state, k_grouped.unsqueeze(-1)).squeeze(-1)
    delta = (value_work - projection) * beta_work
    updated_state = torch.addcmul(decayed_state, delta.unsqueeze(-1), k_grouped.unsqueeze(-2))
    output = torch.matmul(updated_state, (q_work.unsqueeze(2) * scale).unsqueeze(-1)).squeeze(-1)
    # Out-of-place update followed by one copy into the owned cache view keeps
    # the regional graph free of in-place aliasing on the recurrent state.
    state_pool.copy_(updated_state.reshape_as(state_pool))

    output = output.reshape(batch, QWEN38_VALUE_HEADS, QWEN38_DIM).to(packed_qkv.dtype)
    return output, state_pool


def conv_taps(conv_weight: torch.Tensor) -> torch.Tensor:
    """``[width, 4]`` conv weight -> contiguous FP32 ``[4, width]`` taps for the fused decode step."""
    return conv_weight.t().to(torch.float32).contiguous()


def qwen38_fused_decode_step_direct(
    packed_qkv: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    conv_state: torch.Tensor,
    conv_taps: torch.Tensor,
    conv_bias: torch.Tensor | None,
    ssm_state: torch.Tensor,
    scale: float,
    recurrence=None,
    *,
    neg_exp_A: torch.Tensor | None = None,
    dt_bias_f32: torch.Tensor | None = None,
    conv_bias_f32: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fuse gating, width-4 convolution and recurrence for Qwen3.8 TP1 decode.

    ``recurrence(convolved, log_decay, beta, ssm_state, scale) -> (out, state)`` replaces the torch
    recurrence (the custom TPC kernel on a KV-major state).

    ``conv_taps`` is the width-4 conv weight as contiguous FP32 ``[4, width]`` (see ``conv_taps``): column
    slices of the ``[width, 4]`` weight are strided DMA transposes on every step.
    ``conv_state`` is the contiguous ``[B, 3, width]`` history view and
    ``ssm_state`` the contiguous ``[B, HV, V, K]`` FP32 state view; both are
    updated in place. MME still executes the two recurrent projections.
    """
    bias_f32 = dt_bias_f32 if dt_bias_f32 is not None else dt_bias.to(torch.float32)
    gate_input = a.to(torch.float32) + bias_f32
    # log(1 + e^x), not log1p: log1p is a separate unfused node on HPU (one more dependent hop per layer);
    # where they differ (e^x < 2^-24) exp(-exp(A_log) * softplus) rounds to the same f32 decay. Decode only:
    # the same change in the prefill gating (hpu_fused_gdn_gating) moved prompt NLL by 0.004.
    softplus = torch.where(gate_input <= 20.0, torch.log(1.0 + torch.exp(gate_input)), gate_input)
    decay_base = neg_exp_A if neg_exp_A is not None else -torch.exp(A_log.to(torch.float32))
    log_decay = decay_base * softplus
    beta = torch.sigmoid(b.to(torch.float32)).to(b.dtype)

    convolved = conv_state[:, 0, :] * conv_taps[0]
    convolved = convolved + conv_state[:, 1, :] * conv_taps[1]
    convolved = convolved + conv_state[:, 2, :] * conv_taps[2]
    convolved = convolved + packed_qkv * conv_taps[3]
    if conv_bias_f32 is not None:
        convolved = convolved + conv_bias_f32
    elif conv_bias is not None:
        convolved = convolved + conv_bias.to(torch.float32)
    convolved = F.silu(convolved.to(torch.bfloat16))
    conv_state.copy_(torch.cat((conv_state[:, 1:, :], packed_qkv.unsqueeze(1)), dim=1))

    if recurrence is not None:
        output, next_state = recurrence(convolved, log_decay, beta, ssm_state, scale)
    else:
        output, next_state = _direct_qwen38_single_token_packed_decode(convolved, log_decay, beta, ssm_state, scale,
                                                                       True)
    return output, conv_state, next_state


def packed_recurrent_decode_direct(
    packed_qkv: torch.Tensor,
    log_decay: torch.Tensor,
    beta: torch.Tensor,
    state_pool: torch.Tensor,
    scale: float | None,
    use_qk_l2norm: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Single-token packed decode on a contiguous ``[B, HV, V, K]`` state view."""
    if (packed_qkv.ndim == 2 and state_pool.ndim == 4 and state_pool.dtype == torch.float32
            and packed_qkv.shape[1] == QWEN38_PACKED_WIDTH
            and tuple(state_pool.shape[1:]) == (QWEN38_VALUE_HEADS, QWEN38_DIM, QWEN38_DIM)):
        if state_pool.shape[0] != packed_qkv.shape[0]:
            raise ValueError("Direct state requires one contiguous row per request; "
                             f"got slots={state_pool.shape[0]}, B={packed_qkv.shape[0]}.")
        return _direct_qwen38_single_token_packed_decode(
            packed_qkv,
            log_decay,
            beta,
            state_pool,
            QWEN38_DIM**-0.5 if scale is None else scale,
            use_qk_l2norm,
        )
    return _direct_single_token_packed_decode(packed_qkv, log_decay, beta, state_pool, scale, use_qk_l2norm)
