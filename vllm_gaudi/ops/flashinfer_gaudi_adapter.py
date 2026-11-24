# SPDX-License-Identifier: Apache-2.0
# Derived from 1CatAI/1Cat-vLLM-Gaudi (Apache-2.0) at e2266e0, reduced to the
# Qwen3.5/3.8 TP1 direct-state decode and prefill tactics.
"""vLLM-facing adapter for the in-tree Gaudi FlashInfer GDN tactics.

Flags are resolved once by the caller (see ``flashinfer_gdn_*_enabled``) so
compiled graphs never read the environment. The ``maybe_run_*`` entry points
return ``None`` for unqualified shapes so callers keep the general HPU path.
"""

from __future__ import annotations

import functools
import json

import torch

from flashinfer_gaudi._config import get_backend_policy
from flashinfer_gaudi._reference import (QWEN38_PACKED_WIDTH, QWEN38_VALUE_HEADS, conv_taps,
                                         packed_recurrent_decode_direct, qwen38_fused_decode_step_direct)
from flashinfer_gaudi._tactics import gdn_fused_decode_tactic, gdn_prefill_tactic
from flashinfer_gaudi.gdn_prefill import chunk_gated_delta_rule_log_gate
from vllm_gaudi import envs
from vllm_gaudi.ops import hpu_gdn_tpc
from vllm_gaudi.ops.hpu_gdn_pytorch import COMPILE_QK_L2NORM

_GDN_PREFILL_TACTIC = gdn_prefill_tactic()
if envs.VLLM_HPU_GDN_PREFILL_TUNING:
    _GDN_PREFILL_TACTIC["tuning"].update(json.loads(envs.VLLM_HPU_GDN_PREFILL_TUNING))
_GDN_FUSED_DECODE_TACTIC = gdn_fused_decode_tactic()
_GDN_PREFILL_DTYPES = {"bfloat16": torch.bfloat16, "float32": torch.float32}


def flashinfer_gdn_enabled() -> bool:
    if not envs.VLLM_HPU_FLASHINFER_GDN:
        return False
    get_backend_policy()  # rejects backends that are not ported
    return True


def flashinfer_gdn_prefill_enabled() -> bool:
    if not (envs.VLLM_HPU_FLASHINFER_GDN_PREFILL and bool(_GDN_PREFILL_TACTIC.get("promoted", False))):
        return False
    get_backend_policy()
    return True


def flashinfer_gdn_fused_decode_enabled() -> bool:
    return (flashinfer_gdn_enabled() and envs.VLLM_HPU_FLASHINFER_GDN_FUSED_DECODE
            and bool(_GDN_FUSED_DECODE_TACTIC.get("promoted", False)))


def direct_state_rows(
    state_pool: torch.Tensor,
    rows: int,
    group_count: int | None,
    group_offset: int | None,
) -> torch.Tensor | None:
    """Return the contiguous rows ``[1 + offset * span, +rows)`` of a group-major pool.

    The compact GDN pool holds ``span * group_count + 2`` rows: a leading and a
    trailing sentinel and one ``span``-row block per GDN cache group. The
    runner only marks a batch direct when request ``i`` owns row ``i`` of the
    block in every group.
    """
    if group_count is None or group_count <= 0 or group_offset is None or not 0 <= group_offset < group_count:
        return None
    span, remainder = divmod(state_pool.shape[0] - 2, group_count)
    if remainder or not 0 < rows <= span:
        return None
    return state_pool.narrow(0, group_offset * span + 1, rows)


def maybe_run_gdn_prefill(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    log_decay: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor | None,
    *,
    output_final_state: bool,
    use_qk_l2norm_in_kernel: bool,
    chunk_size: int,
    prefill_num_seqs: int,
    prefill_seq_len: int,
    scale: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None] | None:
    """Run the single-sequence Qwen3.8 TP1 FlashQLA prefill tactic."""
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        return None
    if q.dtype != torch.bfloat16 or k.dtype != torch.bfloat16 or v.dtype != torch.bfloat16:
        return None
    if tuple(q.shape) != tuple(k.shape) or q.shape[0] != 1 or v.shape[0] != 1:
        return None
    shape = _GDN_PREFILL_TACTIC["model_shape"]
    if (q.shape[1] != v.shape[1] or q.shape[2] != shape["key_heads"] or v.shape[2] != shape["value_heads"]
            or q.shape[3] != shape["key_dim"] or v.shape[3] != shape["value_dim"]):
        return None
    value_heads = v.shape[2]
    if tuple(log_decay.shape) != (1, q.shape[1], value_heads) or tuple(beta.shape) != (1, q.shape[1], value_heads):
        return None
    if log_decay.dtype != torch.float32:
        return None
    if chunk_size != shape["chunk_size"] or prefill_num_seqs != 1 or prefill_seq_len != q.shape[1]:
        return None

    tuning = _GDN_PREFILL_TACTIC["tuning"]
    return chunk_gated_delta_rule_log_gate(
        q,
        k,
        v,
        log_decay,
        beta,
        scale=scale,
        initial_state=initial_state,
        output_final_state=output_final_state,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        chunk_size=chunk_size,
        prefill_num_seqs=prefill_num_seqs,
        prefill_seq_len=prefill_seq_len,
        neumann_iters=int(tuning["neumann_iters"]),
        fused_state_matmul=bool(tuning["fused_state_matmul"]),
        recursive_solver_base=int(tuning["recursive_solver_base"]),
        compact_repeated_kkt=bool(tuning["compact_repeated_kkt"]),
        # Shared with the general path: an eager (dynamo-disabled) l2norm splits every GDN decoder layer graph.
        compile_qk_l2norm=COMPILE_QK_L2NORM,
        flashqla_reformulation=bool(tuning["flashqla_reformulation"]),
        deferred_output_add=bool(tuning["deferred_output_add"]),
        compute_dtype=_GDN_PREFILL_DTYPES[tuning["compute_dtype"]],
        solve_in_fp32=bool(tuning["solve_in_fp32"]),
        state_in_fp32=bool(tuning["state_in_fp32"]),
        preserve_compact_qk=bool(tuning["preserve_compact_qk"]),
        masked_triangular_decay=bool(tuning["masked_triangular_decay"]),
    )


def maybe_run_gdn_decode_packed(
    mixed_qkv: torch.Tensor,
    log_decay: torch.Tensor,
    beta: torch.Tensor,
    state_pool: torch.Tensor | None,
    *,
    use_qk_l2norm: bool,
    scale: float | None,
    direct_state_group_count: int | None,
    direct_state_group_offset: int | None,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Run single-token recurrence on the contiguous group-major state view.

    Only call this for a batch the runner marked ``direct_gdn_state``. The
    indexed reference of the source fork is not ported: the extra
    gather/scatter is a measured regression versus the general decode path.
    """
    if state_pool is None or mixed_qkv.ndim != 2 or hpu_gdn_tpc.STATE_KV:  # VK math only
        return None
    rows = mixed_qkv.shape[0]
    state_view = direct_state_rows(state_pool, rows, direct_state_group_count, direct_state_group_offset)
    if state_view is None:
        return None
    output, updated_state = packed_recurrent_decode_direct(
        mixed_qkv,
        log_decay,
        beta,
        state_view,
        scale,
        use_qk_l2norm,
    )
    return output.unsqueeze(0), updated_state


def maybe_run_gdn_fused_decode_step(
    mixed_qkv: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    conv_state: torch.Tensor,
    conv_taps: torch.Tensor,
    conv_bias: torch.Tensor | None,
    ssm_state: torch.Tensor,
    *,
    direct_state_group_count: int | None,
    direct_state_group_offset: int | None,
    scale: float,
    neg_exp_A: torch.Tensor | None = None,
    dt_bias_f32: torch.Tensor | None = None,
    conv_bias_f32: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Run the fused gating + conv + recurrence recipe on direct state views.

    ``conv_state`` and ``ssm_state`` are the full compact pools. Only call
    this for a batch the runner marked ``direct_gdn_state``.
    """
    # Any batch size: the recipe is batch-agnostic, and decode buckets include sizes such as 12 and 20.
    if mixed_qkv.ndim != 2:
        return None
    rows, packed_width = mixed_qkv.shape
    value_heads = A_log.numel()
    if (packed_width, value_heads) != (QWEN38_PACKED_WIDTH, QWEN38_VALUE_HEADS):
        return None
    if (mixed_qkv.dtype != torch.bfloat16 or a.dtype != torch.bfloat16 or b.dtype != torch.bfloat16
            or tuple(a.shape) != (rows, value_heads) or tuple(b.shape) != (rows, value_heads)):
        return None
    if tuple(conv_taps.shape) != (4, packed_width) or conv_taps.dtype != torch.float32:
        return None
    if conv_bias is not None and (tuple(conv_bias.shape) != (packed_width, ) or conv_bias.dtype != torch.bfloat16):
        return None
    if (tuple(A_log.shape) != (value_heads, ) or tuple(dt_bias.shape) != (value_heads, ) or A_log.dtype != torch.float32
            or dt_bias.dtype != torch.bfloat16):
        return None
    if (conv_state.ndim != 3 or conv_state.shape[1] < 3 or conv_state.shape[2] != packed_width
            or conv_state.dtype != torch.bfloat16):
        return None
    if ssm_state.ndim != 4 or tuple(ssm_state.shape[1:]) != (value_heads, 128, 128) or ssm_state.dtype != torch.float32:
        return None

    conv_view = direct_state_rows(conv_state, rows, direct_state_group_count, direct_state_group_offset)
    ssm_view = direct_state_rows(ssm_state, rows, direct_state_group_count, direct_state_group_offset)
    if conv_view is None or ssm_view is None:
        return None
    recurrence = None
    if hpu_gdn_tpc.STATE_KV:
        # The TPC kernel updates the whole pool in place at the rows direct_state_rows() selected.
        row_offset = direct_state_group_offset * ((ssm_state.shape[0] - 2) // direct_state_group_count) + 1
        recurrence = functools.partial(hpu_gdn_tpc.packed_decode_kv, row_offset=row_offset)
        ssm_view = ssm_state
    # The recipe consumes the canonical width-1 history (last three columns).
    conv_view = conv_view[:, -3:, :]
    output, _, updated_state = qwen38_fused_decode_step_direct(
        mixed_qkv,
        a,
        b,
        A_log,
        dt_bias,
        conv_view,
        conv_taps,
        conv_bias,
        ssm_view,
        scale,
        recurrence=recurrence,
        neg_exp_A=neg_exp_A,
        dt_bias_f32=dt_bias_f32,
        conv_bias_f32=conv_bias_f32,
    )
    return output.unsqueeze(0), updated_state


__all__ = [
    "conv_taps",
    "direct_state_rows",
    "flashinfer_gdn_enabled",
    "flashinfer_gdn_fused_decode_enabled",
    "flashinfer_gdn_prefill_enabled",
    "maybe_run_gdn_decode_packed",
    "maybe_run_gdn_fused_decode_step",
    "maybe_run_gdn_prefill",
]
