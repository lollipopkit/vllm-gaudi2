# SPDX-License-Identifier: Apache-2.0
# Derived from 1CatAI/1Cat-vLLM-Gaudi (Apache-2.0) at e2266e0.
"""Gated Delta Rule prefill tactic for Intel Gaudi."""

from __future__ import annotations

import torch


def chunk_gated_delta_rule_log_gate(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    log_decay: torch.Tensor,
    beta: torch.Tensor,
    *,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
    chunk_size: int = 64,
    prefill_num_seqs: int | None = None,
    prefill_seq_len: int | None = None,
    neumann_iters: int = 14,
    fused_state_matmul: bool = False,
    recursive_solver_base: int = 0,
    compact_repeated_kkt: bool = False,
    compile_qk_l2norm: bool = False,
    flashqla_reformulation: bool = False,
    deferred_output_add: bool = False,
    compute_dtype: torch.dtype | None = None,
    solve_in_fp32: bool = False,
    state_in_fp32: bool = False,
    preserve_compact_qk: bool = False,
    masked_triangular_decay: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Log-gate entry point used by the vLLM adapter.

    vLLM already computes the forget gate in log space, so this avoids an
    ``exp`` followed by ``log`` at every GDN layer.
    """
    from vllm_gaudi.ops.hpu_gdn_pytorch import hpu_chunk_gated_delta_rule

    return hpu_chunk_gated_delta_rule(
        q=q,
        k=k,
        v=v,
        g=log_decay,
        beta=beta,
        scale=scale,
        initial_state=initial_state,
        output_final_state=output_final_state,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        chunk_size=chunk_size,
        prefill_num_seqs=prefill_num_seqs,
        prefill_seq_len=prefill_seq_len,
        neumann_iters=neumann_iters,
        fused_state_matmul=fused_state_matmul,
        recursive_solver_base=recursive_solver_base,
        compact_repeated_kkt=compact_repeated_kkt,
        compile_qk_l2norm=compile_qk_l2norm,
        flashqla_reformulation=flashqla_reformulation,
        deferred_output_add=deferred_output_add,
        compute_dtype=compute_dtype,
        solve_in_fp32=solve_in_fp32,
        state_in_fp32=state_in_fp32,
        preserve_compact_qk=preserve_compact_qk,
        masked_triangular_decay=masked_triangular_decay,
    )


__all__ = ["chunk_gated_delta_rule_log_gate"]
