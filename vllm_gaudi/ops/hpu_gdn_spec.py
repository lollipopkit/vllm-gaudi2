# SPDX-License-Identifier: Apache-2.0
"""Gated DeltaNet ops for speculative-decode verification (1 + k tokens per request).

State semantics follow upstream vLLM (fused_sigmoid_gating / causal_conv1d spec paths): a request owns
k + 1 SSM state rows and a conv state of width (K - 1) + k. The verify step writes the SSM state after
token t into row t and the conv history shifted by the 1 + k new tokens. Rollback is free: the next step
reads its initial SSM state from row ``num_accepted - 1`` and its conv window at offset
``num_accepted - 1``. With ``num_accepted == k + 1`` (prefill or non-spec decode) the window is the last
K - 1 conv columns, so the non-spec paths stay unchanged.
"""

import torch

from vllm_gaudi.ops.causal_conv1d_pytorch import (_apply_activation, _depthwise_conv1d_tpc,
                                                  _normalize_activation)
from vllm_gaudi.ops.hpu_gdn_pytorch import _GDN_COMPUTE_DTYPE, _gdn_token_step, _l2norm_last_dim


def hpu_causal_conv1d_update_spec(
    x: torch.Tensor,
    conv_state: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    activation: bool | str | None,
    state_indices: torch.Tensor,
    num_accepted: torch.Tensor,
) -> torch.Tensor:
    """Causal conv over 1 + k tokens per request, reading the window of the last accepted token.

    x [B, T, D] with T = 1 + k; conv_state [slots, K - 1 + k, D] (SD layout); state_indices [B]
    (-1 = padding, routed to the last slot); num_accepted [B] in 1..T. Returns [B, T, D].
    """
    activation = _normalize_activation(activation)
    B, T, D = x.shape
    width = weight.shape[1]
    hist = width - 1
    if conv_state.shape[1] != hist + T - 1:
        raise ValueError(f"conv_state width {conv_state.shape[1]} != (K - 1) + k = {hist + T - 1}")
    work_dtype = conv_state.dtype
    slots = torch.remainder(state_indices.reshape(-1).long(), conv_state.shape[0])
    cols = (num_accepted.reshape(-1, 1).long() - 1) + torch.arange(hist, device=x.device)
    window = conv_state[slots.unsqueeze(1), cols]  # [B, K - 1, D]
    seq = torch.cat([window, x.to(work_dtype)], dim=1).transpose(1, 2)  # [B, D, K - 1 + T]
    out = _depthwise_conv1d_tpc(seq, weight.to(work_dtype), bias.to(work_dtype) if bias is not None else None)
    out = _apply_activation(out, activation)  # [B, D, T]
    with torch.no_grad():
        conv_state[slots] = seq[:, :, 1:].transpose(1, 2)
    return out.transpose(1, 2).to(x.dtype)


def hpu_gdn_spec_verify(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    ssm_state: torch.Tensor,
    load_rows: torch.Tensor,
    store_rows: torch.Tensor,
    scale: float,
    use_qk_l2norm: bool = True,
) -> torch.Tensor:
    """Gated delta rule over 1 + k tokens per request, keeping the state after every token.

    q/k [B, T, H, K]; v [B, T, HV, V]; g/beta [B, T, HV]; ssm_state [rows, HV, V, K].
    load_rows [B]: row of the initial state (the request's row ``num_accepted - 1``).
    store_rows [B, T]: row receiving the state after token t. Returns out [B, T, HV, V].
    """
    B, T, H, Kdim = q.shape
    HV = v.shape[2]
    rows = ssm_state.shape[0]
    h = ssm_state.index_select(0, torch.remainder(load_rows.reshape(-1).long(), rows)).to(_GDN_COMPUTE_DTYPE)
    qf, kf = q.to(_GDN_COMPUTE_DTYPE), k.to(_GDN_COMPUTE_DTYPE)
    if use_qk_l2norm:
        qf, kf = _l2norm_last_dim(qf), _l2norm_last_dim(kf)
    if HV != H:
        qf = qf.repeat_interleave(HV // H, dim=2)
        kf = kf.repeat_interleave(HV // H, dim=2)
    vf = v.to(_GDN_COMPUTE_DTYPE)
    gf = g.to(torch.float32)
    bf = beta.to(_GDN_COMPUTE_DTYPE)
    outs, states = [], []
    for t in range(T):  # static T = 1 + k: unrolled in the compiled graph
        out_t, h = _gdn_token_step(h, qf[:, t], kf[:, t], vf[:, t], gf[:, t], bf[:, t], scale)
        outs.append(out_t)
        states.append(h)
    new_states = torch.stack(states, dim=1).reshape(B * T, *h.shape[1:]).to(ssm_state.dtype)
    ssm_state.index_copy_(0, torch.remainder(store_rows.reshape(-1).long(), rows), new_states)
    return torch.stack(outs, dim=1).to(v.dtype)
