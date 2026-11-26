# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pure PyTorch / HPU math primitives for Inkling2 on Gaudi2.

All formulas mirror the upstream NVIDIA/Triton reference (`vllm/models/inkling2/nvidia/ops/`)
in fp32 accumulation, rounding once to storage dtype.
"""

from __future__ import annotations

import math
import os

import torch
import torch.nn.functional as F

from .tpc_ops import (
    has_tpc_attn_res,
    has_tpc_attn_res_f8,
    has_tpc_blend,
    has_tpc_blend_f8,
)

GAUDI2_FP8_MAX = 240.0
_CGUID_DYNAMIC_QUANT = os.environ.get("VLLM_HPU_CGUID_DYNAMIC_QUANT", "1").strip().lower() not in (
    "0",
    "false",
    "no",
)
_CGUID_DYNAMIC_QUANT_MAX_ROWS = int(
    os.environ.get("VLLM_HPU_CGUID_DYNAMIC_QUANT_MAX_ROWS", "32")
)


def rms_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    out_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """RMSNorm over the last dim in fp32, cast to `out_dtype or x.dtype`."""
    xf = x.float()
    rstd = torch.rsqrt(xf.square().mean(dim=-1, keepdim=True) + eps)
    y = xf * rstd * weight.float()
    return y.to(out_dtype if out_dtype is not None else x.dtype)


def weightless_rms_norm(x: torch.Tensor, eps: float) -> torch.Tensor:
    """Weightless RMSNorm over the last dim in fp32, cast back to `x.dtype`."""
    xf = x.float()
    rstd = torch.rsqrt(xf.square().mean(dim=-1, keepdim=True) + eps)
    return (xf * rstd).to(x.dtype)


def l2_clip(
    x: torch.Tensor,
    scale: torch.Tensor | None,
    threshold: float,
    eps: float = 1e-6,
    tau: torch.Tensor | None = None,
) -> torch.Tensor:
    """Per-head L2 clipping of `x (* scale)`, optionally scaled by `tau`."""
    xf = x.float()
    if scale is not None:
        xf = xf * scale.float()
    sum_sq = xf.square().sum(dim=-1, keepdim=True)
    norm = torch.sqrt(sum_sq.clamp(min=0.0))
    factor = (threshold / (norm + eps)).clamp(max=1.0)
    y = xf * factor
    if tau is not None:
        # Match upstream: round clipped value to storage dtype before multiplying tau.
        y = y.to(x.dtype).float() * tau.view(-1, *([1] * (x.ndim - 1)))
    return y.to(x.dtype)


def compute_log_scaling_tau(
    positions: torch.Tensor, n_floor: int | None, alpha: float
) -> torch.Tensor | None:
    """Return fp32 `1 + alpha * log(max((positions + 1) / n_floor, 1))`."""
    if n_floor is None or alpha == 0.0:
        return None
    ratio = ((positions.float() + 1.0) / float(n_floor)).clamp(min=1.0)
    return 1.0 + alpha * torch.log(ratio)


def compact_inkling2_hpu_buckets(
    prompt_buckets: list[tuple[int, int, int]],
    decode_buckets: list[tuple[int, int, int]],
    *,
    max_model_len: int,
    max_num_batched_tokens: int,
    max_num_seqs: int,
    block_size: int,
    num_hpu_blocks: int | None,
    use_contiguous_pa: bool = False,
    use_merged_prefill: bool = False,
) -> tuple[list[tuple[int, int, int]], list[tuple[int, int, int]]]:
    """Compact prompt/decode buckets for Inkling2 static-window & TPC paged attention."""
    if os.environ.get("INKLING2_COMPACT_BUCKETS", "1") == "0":
        return prompt_buckets, decode_buckets

    new_prompt = prompt_buckets
    if prompt_buckets and not use_merged_prefill:
        bss = sorted({bs for bs, _, _ in prompt_buckets})
        orig_qs = sorted({q for _, q, _ in prompt_buckets})
        min_q, max_q = orig_qs[0], orig_qs[-1]
        cand_qs: set[int] = set()
        q_pow = max(1, block_size)
        while q_pow <= max_q * 2:
            cand_qs.add(q_pow)
            if q_pow >= 1024:
                cand_qs.add((q_pow * 3) // 2)
            q_pow *= 2
        aligned_qs = {
            math.ceil(q / block_size) * block_size
            for q in cand_qs
            if min_q <= math.ceil(q / block_size) * block_size <= max_q
        } | {min_q, max_q}
        qs = sorted(q for q in aligned_qs if q <= max_model_len)
        orig_max_ctx = max(ctx for _, _, ctx in prompt_buckets)
        if orig_max_ctx <= 0 or max_model_len <= block_size:
            ctxs = [0]
        else:
            max_prompt_ctx = math.ceil((max_model_len - 1) / block_size)
            if num_hpu_blocks is not None and num_hpu_blocks > 0:
                max_prompt_ctx = min(max_prompt_ctx, num_hpu_blocks)
            ctxs = [0, max_prompt_ctx] if max_prompt_ctx > 0 else [0]
        prompt_set: set[tuple[int, int, int]] = set()
        for bs in bss:
            bs_max_q = min(max_model_len, (max_num_batched_tokens // bs // block_size) * block_size)
            bs_qs = {q for q in qs if bs * q <= max_num_batched_tokens}
            if bs_max_q >= min_q:
                bs_qs.add(bs_max_q)
            bs_ctxs = ctxs if bs == 1 else [0]
            for q in bs_qs:
                for ctx in bs_ctxs:
                    prompt_set.add((bs, q, ctx))
        new_prompt = sorted(prompt_set)

    new_decode = decode_buckets
    if decode_buckets and not use_contiguous_pa:
        orig_bss = sorted({bs for bs, _, _ in decode_buckets})
        min_bs, max_bs = orig_bss[0], orig_bss[-1]
        pow2_bss = set()
        b = 1
        while b < max_bs:
            if b >= min_bs:
                pow2_bss.add(b)
            b *= 2
        pow2_bss.update({min_bs, max_bs})
        orig_max_dec_ctx = max(ctx for _, _, ctx in decode_buckets)
        max_blocks_per_req = math.ceil(max_model_len / block_size)
        cand_blks = [1, 8, 32]
        blk_pow = 64
        while blk_pow <= max(64, orig_max_dec_ctx):
            cand_blks.append(blk_pow)
            blk_pow *= 2
        dec_set: set[tuple[int, int, int]] = set()
        for bs in sorted(pow2_bss):
            bs_max = min(orig_max_dec_ctx, bs * max_blocks_per_req)
            bs_orig = [ctx for b_i, _, ctx in decode_buckets if b_i == bs]
            bs_min = min(bs_orig) if bs_orig else max(1, bs)
            for ctx in cand_blks:
                if bs <= ctx <= bs_max:
                    dec_set.add((bs, 1, ctx))
            dec_set.add((bs, 1, max(bs, min(bs_min, bs_max))))
            dec_set.add((bs, 1, max(bs, bs_max)))
        new_decode = sorted(dec_set)

    return new_prompt, new_decode


def attn_res(
    stack: torch.Tensor,
    last: torch.Tensor,
    addend: torch.Tensor | None,
    queries: torch.Tensor,
    norm_weight: torch.Tensor | None,
    norm_eps: float,
    normalize_values: bool = True,
    eps: float = 1e-6,
    *,
    stack_valid: torch.Tensor | None = None,
) -> torch.Tensor:
    """Block attention residual mix over `[stack..., last (+ addend)]`.

    Args:
        stack: `[N_stack, T, D]` completed blocks (bf16 or fp32).
        last: `[T, D]` running partial block (bf16 or fp32).
        addend: Optional `[T, D]` added to `last` in fp32.
        queries: `[Q, D]` pre-scaled pseudo-queries (fp32).
        norm_weight: Optional `[Q, D]` RMSNorm weights; if None, returns fp32.
        norm_eps: RMSNorm epsilon when `norm_weight` is provided.
        normalize_values: If True, mix RMS-normalized source blocks.
        stack_valid: Optional `[N_stack]` bool mask for fixed-capacity prefill stacks.

    Returns:
        `[Q, T, D]` tensor (bf16 if `norm_weight` is not None, else fp32).
    """
    if (
        stack_valid is None
        and last.device.type == "hpu"
        and last.dtype == torch.bfloat16
        and stack.dtype == torch.bfloat16
        and (addend is None or addend.dtype == torch.bfloat16)
        and last.ndim == 2
        and (last.shape[1] % 64) == 0
        and last.shape[1] <= 40 * 64
        and stack.shape[0] < 16
        and 1 <= queries.shape[0] <= 4
        and (
            norm_weight is None
            or (norm_weight.ndim == 2 and norm_weight.shape == queries.shape)
        )
        and has_tpc_attn_res()
    ):
        n_stack = int(stack.shape[0])
        last_c = last.contiguous()
        stack_c = stack.contiguous() if n_stack > 0 else last_c.unsqueeze(0)
        add_c = last_c if addend is None else addend.contiguous()
        q_c = queries.float().contiguous()
        has_add = 0 if addend is None else 1
        norm_val = 1 if normalize_values else 0
        if norm_weight is not None:
            return torch.ops.custom_op.inkling2_attn_res_norm_bf16(
                stack_c,
                last_c,
                add_c,
                q_c,
                norm_weight.to(torch.bfloat16).contiguous(),
                n_stack,
                has_add,
                norm_val,
                float(eps),
                float(norm_eps),
            )
        return torch.ops.custom_op.inkling2_attn_res_f32(
            stack_c,
            last_c,
            add_c,
            q_c,
            n_stack,
            has_add,
            norm_val,
            float(eps),
        )

    last_f = last.float() if addend is None else (last.float() + addend.float())
    if stack.shape[0] > 0:
        sources = torch.cat([stack.float(), last_f.unsqueeze(0)], dim=0)  # [N, T, D]
    else:
        sources = last_f.unsqueeze(0)  # [1, T, D]

    src_valid: torch.Tensor | None = None
    if stack_valid is not None and stack.shape[0] > 0:
        sv = stack_valid.bool()
        src_valid = torch.cat([sv, sv.new_ones((1,))], dim=0)  # [N]
        sources = torch.where(src_valid.view(-1, 1, 1), sources, 0.0)

    inv_rms = torch.rsqrt(sources.square().mean(dim=-1, keepdim=True) + eps)  # [N, T, 1]
    normed_sources = sources * inv_rms  # [N, T, D]

    # logits[q, n, t] = dot(queries[q], normed_sources[n, t]) via explicit matmul
    # (avoids Habana dynamic-shape einsum Reshape bug under torch.compile).
    logits = torch.matmul(normed_sources, queries.float().t()).permute(2, 0, 1)  # [Q, N, T]
    if src_valid is not None:
        neg_large = torch.full((), -1e30, dtype=logits.dtype, device=logits.device)
        logits = torch.where(src_valid.view(1, -1, 1), logits, neg_large)
    probs = torch.softmax(logits, dim=1)  # [Q, N, T]
    if src_valid is not None:
        probs = probs * src_valid.view(1, -1, 1).float()
        probs = probs / probs.sum(dim=1, keepdim=True).clamp(min=1e-30)

    values = normed_sources if normalize_values else sources  # [N, T, D]
    # [T, Q, N] @ [T, N, D] -> [T, Q, D] -> [Q, T, D]
    mixed = torch.matmul(probs.permute(2, 0, 1), values.transpose(0, 1)).transpose(0, 1)

    if norm_weight is None:
        return mixed

    rstd = torch.rsqrt(mixed.square().mean(dim=-1, keepdim=True) + norm_eps)
    out = mixed * rstd * norm_weight.float().unsqueeze(1)
    return out.to(torch.bfloat16)


def attn_res_f8(
    stack: torch.Tensor,
    last: torch.Tensor,
    addend: torch.Tensor | None,
    queries: torch.Tensor,
    norm_weight: torch.Tensor,
    norm_eps: float,
    normalize_values: bool = True,
    eps: float = 1e-6,
    fp8_max: float = GAUDI2_FP8_MAX,
    *,
    stack_valid: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    """Fused block attention residual + RMSNorm + per-row FP8 E4M3 quantization.

    Returns `(out_bf16 [Q, T, D], out_f8 [Q, T, D] | None, out_scale [Q, T] | None)`.
    """
    if (
        stack_valid is None
        and last.device.type == "hpu"
        and last.dtype == torch.bfloat16
        and stack.dtype == torch.bfloat16
        and (addend is None or addend.dtype == torch.bfloat16)
        and last.ndim == 2
        and (last.shape[1] % 64) == 0
        and last.shape[1] <= 40 * 64
        and stack.shape[0] < 16
        and 1 <= queries.shape[0] <= 4
        and norm_weight.ndim == 2
        and norm_weight.shape == queries.shape
        and has_tpc_attn_res_f8()
    ):
        n_stack = int(stack.shape[0])
        last_c = last.contiguous()
        stack_c = stack.contiguous() if n_stack > 0 else last_c.unsqueeze(0)
        add_c = last_c if addend is None else addend.contiguous()
        q_c = queries.float().contiguous()
        has_add = 0 if addend is None else 1
        norm_val = 1 if normalize_values else 0
        return torch.ops.custom_op.inkling2_attn_res_norm_f8(
            stack_c,
            last_c,
            add_c,
            q_c,
            norm_weight.to(torch.bfloat16).contiguous(),
            n_stack,
            has_add,
            norm_val,
            float(eps),
            float(norm_eps),
            float(fp8_max),
        )
    out = attn_res(
        stack,
        last,
        addend,
        queries,
        norm_weight,
        norm_eps,
        normalize_values,
        eps,
        stack_valid=stack_valid,
    )
    return out, None, None


def pair_rms_norm(dest: torch.Tensor, message: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """bf16 `(S, T, 2D)` `[rms(dest), rms(message)]` of fp32 `(S, T, D)` inputs."""
    if (
        dest.device.type == "hpu"
        and dest.dtype == torch.float32
        and message.dtype == torch.float32
        and dest.ndim == 3
        and (dest.shape[-1] % 64) == 0
        and has_tpc_blend()
    ):
        return torch.ops.custom_op.inkling2_pair_rms_norm(
            dest.contiguous(), message.contiguous(), float(eps)
        )
    d_norm = dest * torch.rsqrt(dest.square().mean(dim=-1, keepdim=True) + eps)
    m_norm = message * torch.rsqrt(message.square().mean(dim=-1, keepdim=True) + eps)
    return torch.cat([d_norm.to(torch.bfloat16), m_norm.to(torch.bfloat16)], dim=-1)


def blend_rms_norm(
    dest: torch.Tensor,
    message: torch.Tensor,
    gain_forget: torch.Tensor,
    norm_weight: torch.Tensor,
    norm_eps: float,
) -> torch.Tensor:
    """bf16 `(S, T, D)` RMSNorm of `(1 - tanh(forget)) * dest + tanh(gain) * message`."""
    if (
        dest.device.type == "hpu"
        and dest.dtype == torch.float32
        and message.dtype == torch.float32
        and gain_forget.dtype == torch.bfloat16
        and dest.ndim == 3
        and (dest.shape[-1] % 64) == 0
        and dest.shape[-1] <= 40 * 64
        and has_tpc_blend()
    ):
        return torch.ops.custom_op.inkling2_blend_rms_norm(
            dest.contiguous(),
            message.contiguous(),
            gain_forget.contiguous(),
            norm_weight.to(torch.bfloat16).contiguous(),
            float(norm_eps),
        )
    d = message.shape[-1]
    gf = gain_forget.float()
    gain = torch.tanh(gf[..., :d])
    forget = torch.tanh(gf[..., d:])
    x = (1.0 - forget) * dest + gain * message
    rstd = torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + norm_eps)
    y = x * rstd * norm_weight.float().unsqueeze(1)
    return y.to(torch.bfloat16)


def blend_rms_norm_f8(
    dest: torch.Tensor,
    message: torch.Tensor,
    gain_forget: torch.Tensor,
    norm_weight: torch.Tensor,
    norm_eps: float,
    fp8_max: float = GAUDI2_FP8_MAX,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    """Fused blend_rms_norm + per-row FP8 E4M3 quantization -> `(out_bf16, out_f8, out_scale)`."""
    if (
        dest.device.type == "hpu"
        and dest.dtype == torch.float32
        and message.dtype == torch.float32
        and gain_forget.dtype == torch.bfloat16
        and dest.ndim == 3
        and (dest.shape[-1] % 64) == 0
        and dest.shape[-1] <= 40 * 64
        and has_tpc_blend_f8()
    ):
        return torch.ops.custom_op.inkling2_blend_rms_norm_f8(
            dest.contiguous(),
            message.contiguous(),
            gain_forget.contiguous(),
            norm_weight.to(torch.bfloat16).contiguous(),
            float(norm_eps),
            float(fp8_max),
        )
    out = blend_rms_norm(dest, message, gain_forget, norm_weight, norm_eps)
    return out, None, None


def softcapped_situ(
    gate_up: torch.Tensor,
    routing_weight: torch.Tensor,
    gate_cap: float,
    up_cap: float,
) -> torch.Tensor:
    """Softcapped SiTU on interleaved `(gate, up)` columns `[..., 2*I]` -> `[..., I]` bf16.

    Matches `nvidia/ops/activation.py:_situ`:
        sigmoid = sigmoid(gate)
        gate = gate_cap * tanh(gate / gate_cap)  (if gate_cap > 0)
        up = up_cap * tanh(up / up_cap)          (if up_cap > 0)
        out = ((gate * sigmoid) * up) * routing_weight
    """
    gate = gate_up[..., 0::2].float()
    up = gate_up[..., 1::2].float()
    sig = torch.sigmoid(gate)
    if gate_cap > 0.0:
        gate = gate_cap * torch.tanh(gate / gate_cap)
    if up_cap > 0.0:
        up = up_cap * torch.tanh(up / up_cap)
    return (((gate * sig) * up) * routing_weight.float()).to(torch.bfloat16)


def gate_sink(
    attn_out: torch.Tensor,
    gate: torch.Tensor,
    lse: torch.Tensor,
    sinks: torch.Tensor,
) -> torch.Tensor:
    """bf16 `attn_out * sigmoid(gate) * sigmoid(lse - sinks)`.

    Args:
        attn_out: `[T, H, D]` fp32 or bf16.
        gate: `[T, H * D]` bf16.
        lse: `[T, H]` fp32 natural-log LSE.
        sinks: `[T, H]` bf16/fp32 per-token sink logits.
    """
    t, h, d = attn_out.shape
    sink_scale = torch.sigmoid(lse.float() - sinks.float()).unsqueeze(-1)  # [T, H, 1]
    gated = attn_out.float() * torch.sigmoid(gate.view(t, h, d).float()) * sink_scale
    return gated.reshape(t, h * d).to(torch.bfloat16)


def quantize_fp8_hpu(
    x: torch.Tensor,
    fp8_max: float = GAUDI2_FP8_MAX,
    *,
    per_row: bool = True,
    use_cguid: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize bf16/fp32 tensor `x` to Gaudi2 `float8_e4m3fn` with dequant scale.

    Uses Habana's hardware-fused `cast_to_fp8_v2` (and `calculate_scale_for_cast`
    CGUID for decode-sized 2D activations when `use_cguid=True`). Pass
    `use_cguid=False` immediately after gated activations on Gaudi2 to avoid the
    known gate + scale-CGUID + fp8_gemm fusion quirk.
    """
    if per_row and x.ndim == 2:
        if use_cguid and _CGUID_DYNAMIC_QUANT and x.shape[0] <= _CGUID_DYNAMIC_QUANT_MAX_ROWS:
            scale = (
                torch.ops.hpu.calculate_scale_for_cast(
                    x,
                    2,  # MAX_ABS_PCS_CALCULATION
                    0,  # NO_SCALE_ROUNDING
                    -1,
                    True,
                    float(fp8_max),
                    1.0,
                )
                + (1e-8 / fp8_max)
            ).float()
        else:
            scale = ((x.abs().amax(dim=-1, keepdim=True) + 1e-8) / fp8_max).float()
    else:
        scale = ((x.abs().amax() + 1e-8) / fp8_max).float()
    x_fp8 = torch.ops.hpu.cast_to_fp8_v2(
        x, torch.reciprocal(scale), False, False, torch.float8_e4m3fn
    )[0]
    return x_fp8, scale


def fp8_linear(
    x: torch.Tensor,
    weight_fp8: torch.Tensor,
    weight_scale: torch.Tensor,
    fp8_max: float = GAUDI2_FP8_MAX,
    *,
    use_cguid: bool = True,
) -> torch.Tensor:
    """Linear `x @ weight_fp8.T` using Gaudi2 `fp8_gemm_v2` on HPU, CPU fallback otherwise.

    Args:
        x: `[M, K]` bf16 activation.
        weight_fp8: `[N, K]` float8_e4m3fn (scaled to `[-fp8_max, fp8_max]`).
        weight_scale: `[N]` float32 per-channel weight scale.
    """
    if x.device.type == "hpu":
        x_fp8, x_scale = quantize_fp8_hpu(
            x, fp8_max=fp8_max, per_row=True, use_cguid=use_cguid
        )
        return torch.ops.hpu.fp8_gemm_v2(
            A=x_fp8,
            trans_A=False,
            B=weight_fp8,
            trans_B=True,
            D=None,
            out_dtype=torch.bfloat16,
            A_scale_inv=x_scale,
            B_scale_inv=weight_scale,
            bias=None,
            accumulate=False,
        )
    # CPU / non-HPU reference fallback for unit tests.
    w = weight_fp8.float() * weight_scale.float().unsqueeze(-1)
    return F.linear(x.float(), w).to(torch.bfloat16)


def fp8_bmm(
    x: torch.Tensor,
    weight_fp8: torch.Tensor,
    weight_scale: torch.Tensor,
    fp8_max: float = GAUDI2_FP8_MAX,
    *,
    x_scale: torch.Tensor | None = None,
    use_cguid: bool = False,
) -> torch.Tensor:
    """Batched 3D FP8 GEMM `[R, M, K] @ [R, N, K].T -> [R, M, N]` bf16.

    If `x` is already `torch.float8_e4m3fn` and `x_scale` is provided, skips
    activation quantization (avoids redundant quantizing after repeat_interleave).
    """
    if x.device.type == "hpu":
        if x.dtype == torch.float8_e4m3fn and x_scale is not None:
            x_fp8 = x
            a_scale = x_scale
        else:
            r, m, k = x.shape
            x_2d = x.reshape(r * m, k)
            x_fp8_2d, sc_2d = quantize_fp8_hpu(
                x_2d, fp8_max=fp8_max, per_row=True, use_cguid=use_cguid
            )
            x_fp8 = x_fp8_2d.view(r, m, k)
            a_scale = sc_2d.view(r, m, 1)
        return torch.ops.hpu.fp8_gemm_v2(
            A=x_fp8,
            trans_A=False,
            B=weight_fp8,
            trans_B=True,
            D=None,
            out_dtype=torch.bfloat16,
            A_scale_inv=a_scale,
            B_scale_inv=weight_scale.unsqueeze(1),
            bias=None,
            accumulate=False,
        )
    if x.dtype == torch.float8_e4m3fn and x_scale is not None:
        xf = x.float() * x_scale.float()
    else:
        xf = x.float()
    w = weight_fp8.float() * weight_scale.float().unsqueeze(-1)
    return torch.bmm(xf, w.transpose(1, 2)).to(torch.bfloat16)
