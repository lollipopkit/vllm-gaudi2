// SPDX-License-Identifier: Apache-2.0
// Generic zero-LUT fused Add + RMSNorm + Per-Row FP8 E4M3 Dynamic Quantization for Gaudi2 TPC:
//   res_f32 = residual.f32 + x.f32
//   out_res = bf16(res_f32)
//   y_bf16  = bf16((res_f32 * rsqrt(mean(res_f32^2) + eps)) * weight)
//   scale   = (max(|y_bf16|) + 1e-8) / fp8_max
//   out_f8  = f8_143(clamp(y_bf16 / scale, -fp8_max, fp8_max))
//   (when OUT_BF16 == 1, also writes out_bf16 = y_bf16 for dual-consumer layers like Qwen GDN / LFM3 Attn)
//
// Zero SFU LUTs -> full 64 KiB VLM available. 4-vector (256-elem) unrolling across all passes,
// packed BF16 VLM + 128-wide BF16 amax in Pass 2/3, and VLM weight hoisting across rows for prefill
// (when rows > 1 and weight fits alongside the row buffer: D <= 7680 for F32w, D <= 10240 for BF16w).
//
// Tensors (TPC dim order, dim0 fastest):
//   x        [D, M] bf16
//   residual [D, M] bf16
//   weight   [D]    f32 (WEIGHT_F32=1, e.g. GemmaRMSNorm precomputed 1+w) or bf16 (WEIGHT_F32=0)
//   out_f8   [D, M] f8_143
//   out_sc   [1, M] f32
//   out_res  [D, M] bf16
//   out_bf16 [D, M] bf16 (only when OUT_BF16 == 1)
//
// Index space: [1, M] (dim0 = full row reduction, dim1 = row slice -> clean SRAM slicing along M).
#pragma once

#include "gaudi2_fp8_common.h"

#ifndef WEIGHT_F32
#define WEIGHT_F32 1
#endif
#ifndef OUT_BF16
#define OUT_BF16 0
#endif

// Shared VLM scratch: holds up to D = 12288 FP32 row elements (192 slots = 48 KiB) plus optional hoisted weights.
#define MAX_NORM_VLM_SLOTS 240  // 240 * 256 B = 60 KiB VLM
__local__ float64 g_norm_vlm[MAX_NORM_VLM_SLOTS];

static inline bfloat128 vlm_read_bf16(int idx)
{
    union {
        float64 f;
        bfloat128 b;
    } cvt;
    cvt.f = g_norm_vlm[idx];
    return cvt.b;
}

static inline void vlm_write_bf16(int idx, bfloat128 v)
{
    union {
        float64 f;
        bfloat128 b;
    } cvt;
    cvt.b = v;
    g_norm_vlm[idx] = cvt.f;
}

#if OUT_BF16
void main(tensor x, tensor residual, tensor weight,
          tensor out_f8, tensor out_sc, tensor out_res, tensor out_bf16,
          float eps, float inv_D, float fp8_max)
#else
void main(tensor x, tensor residual, tensor weight,
          tensor out_f8, tensor out_sc, tensor out_res,
          float eps, float inv_D, float fp8_max)
#endif
{
    const int D = get_dim_size(x, 0);
    const int nchunks = D >> 6;  // D / 64
    const int D256 = D & ~255;
    const int D128 = D & ~127;
    const int5 is_start = get_index_space_offset();
    const int5 is_end = get_index_space_size() + is_start;
    const tpc_xor_masks_t xmasks = make_xor_masks_nolut(read_lane_id_4b_b());
    const float inv_fp8_max = 1.0f / fp8_max;

    // Hoist weight into VLM once outside the row loop whenever this core handles > 1 row (prefill).
    const int rows = is_end[1] - is_start[1];
#if WEIGHT_F32
    const int w_slots = nchunks;
#else
    const int w_slots = (nchunks + 1) >> 1;
#endif
    const int w_base = nchunks;
    const int hoist_w = (rows > 1) && (w_base + w_slots <= MAX_NORM_VLM_SLOTS);
    if (hoist_w)
    {
        int5 wc0 = {0, 0, 0, 0, 0}, wc1 = {64, 0, 0, 0, 0};
#if WEIGHT_F32
        int ch = 0;
        for (int d = 0; d < D128; d += 128, ch += 2)
        {
            g_norm_vlm[w_base + ch + 0] = v_f32_ld_tnsr_b(wc0, weight);
            g_norm_vlm[w_base + ch + 1] = v_f32_ld_tnsr_b(wc1, weight);
            wc0[0] += 128; wc1[0] += 128;
        }
        if (D & 64)
        {
            wc0[0] = D128;
            g_norm_vlm[w_base + ch] = v_f32_ld_tnsr_b(wc0, weight);
        }
#else
        int ws = 0;
        for (int d = 0; d < D128; d += 128, ws++)
        {
            vlm_write_bf16(w_base + ws,
                           pack_two_f32_to_bf16_vec(ld_bf16_as_f32(wc0, weight), ld_bf16_as_f32(wc1, weight)));
            wc0[0] += 128; wc1[0] += 128;
        }
        if (D & 64)
        {
            wc0[0] = D128;
            vlm_write_bf16(w_base + ws, pack_two_f32_to_bf16_vec(ld_bf16_as_f32(wc0, weight), (float64)0.0f));
        }
#endif
    }

    for (int m = is_start[1]; m < is_end[1]; m++)
    {
        // Pass 1: 4-way unrolled FP32 residual sum, store updated BF16 residual, cache FP32 sum in VLM.
        float64 msq0 = 0.0f, msq1 = 0.0f, msq2 = 0.0f, msq3 = 0.0f;
        int5 c0 = {0, m, 0, 0, 0}, c1 = {64, m, 0, 0, 0}, c2 = {128, m, 0, 0, 0}, c3 = {192, m, 0, 0, 0};
        int ch = 0;
        for (int d = 0; d < D256; d += 256, ch += 4)
        {
            const float64 s0 = ld_bf16_as_f32(c0, residual) + ld_bf16_as_f32(c0, x);
            const float64 s1 = ld_bf16_as_f32(c1, residual) + ld_bf16_as_f32(c1, x);
            const float64 s2 = ld_bf16_as_f32(c2, residual) + ld_bf16_as_f32(c2, x);
            const float64 s3 = ld_bf16_as_f32(c3, residual) + ld_bf16_as_f32(c3, x);
            st_f32_as_bf16(c0, out_res, s0);
            st_f32_as_bf16(c1, out_res, s1);
            st_f32_as_bf16(c2, out_res, s2);
            st_f32_as_bf16(c3, out_res, s3);
            c0[0] += 256; c1[0] += 256; c2[0] += 256; c3[0] += 256;
            g_norm_vlm[ch + 0] = s0;
            g_norm_vlm[ch + 1] = s1;
            g_norm_vlm[ch + 2] = s2;
            g_norm_vlm[ch + 3] = s3;
            msq0 = v_f32_mac_b(s0, s0, msq0);
            msq1 = v_f32_mac_b(s1, s1, msq1);
            msq2 = v_f32_mac_b(s2, s2, msq2);
            msq3 = v_f32_mac_b(s3, s3, msq3);
        }
        for (int d = D256; d < D; d += 64, ch++)
        {
            c0[0] = d;
            const float64 s0 = ld_bf16_as_f32(c0, residual) + ld_bf16_as_f32(c0, x);
            st_f32_as_bf16(c0, out_res, s0);
            g_norm_vlm[ch] = s0;
            msq0 = v_f32_mac_b(s0, s0, msq0);
        }
        const float64 rstd = rsqrt_nolut(reduce_add_fast_nolut((msq0 + msq1) + (msq2 + msq3), xmasks) * inv_D + eps);

        // Pass 2: 4-way unrolled norm * weight, convert once to BF16, pack 128 elems/slot into VLM[0..D/128-1]
        // (strictly behind read pointer ch >= 2*slot so overwriting g_norm_vlm in-place is race-free).
        bfloat128 am0 = 0, am1 = 0;
        int5 wc0 = {0, 0, 0, 0, 0}, wc1 = {64, 0, 0, 0, 0}, wc2 = {128, 0, 0, 0, 0}, wc3 = {192, 0, 0, 0, 0};
#if OUT_BF16
        int5 bc0 = {0, m, 0, 0, 0}, bc1 = {64, m, 0, 0, 0}, bc2 = {128, m, 0, 0, 0}, bc3 = {192, m, 0, 0, 0};
#endif
        ch = 0;
        int slot = 0;
        for (int d = 0; d < D256; d += 256, ch += 4, slot += 2)
        {
            float64 w0, w1, w2, w3;
            if (hoist_w)
            {
#if WEIGHT_F32
                w0 = g_norm_vlm[w_base + ch + 0];
                w1 = g_norm_vlm[w_base + ch + 1];
                w2 = g_norm_vlm[w_base + ch + 2];
                w3 = g_norm_vlm[w_base + ch + 3];
#else
                const bfloat128 wp0 = vlm_read_bf16(w_base + slot + 0);
                const bfloat128 wp1 = vlm_read_bf16(w_base + slot + 1);
                w0 = unpack_bf16_lo(wp0);
                w1 = unpack_bf16_hi(wp0);
                w2 = unpack_bf16_lo(wp1);
                w3 = unpack_bf16_hi(wp1);
#endif
            }
            else
            {
#if WEIGHT_F32
                w0 = v_f32_ld_tnsr_b(wc0, weight);
                w1 = v_f32_ld_tnsr_b(wc1, weight);
                w2 = v_f32_ld_tnsr_b(wc2, weight);
                w3 = v_f32_ld_tnsr_b(wc3, weight);
#else
                w0 = ld_bf16_as_f32(wc0, weight);
                w1 = ld_bf16_as_f32(wc1, weight);
                w2 = ld_bf16_as_f32(wc2, weight);
                w3 = ld_bf16_as_f32(wc3, weight);
#endif
                wc0[0] += 256; wc1[0] += 256; wc2[0] += 256; wc3[0] += 256;
            }
            const float64 y0 = (g_norm_vlm[ch + 0] * rstd) * w0;
            const float64 y1 = (g_norm_vlm[ch + 1] * rstd) * w1;
            const float64 y2 = (g_norm_vlm[ch + 2] * rstd) * w2;
            const float64 y3 = (g_norm_vlm[ch + 3] * rstd) * w3;
#if OUT_BF16
            const bfloat128 b0 = v_convert_f32_to_bf16_b(y0, 0, 0, 0, 1, 0);
            const bfloat128 b1 = v_convert_f32_to_bf16_b(y1, 0, 0, 0, 1, 0);
            const bfloat128 b2 = v_convert_f32_to_bf16_b(y2, 0, 0, 0, 1, 0);
            const bfloat128 b3 = v_convert_f32_to_bf16_b(y3, 0, 0, 0, 1, 0);
            st_bf16_lo_vec(bc0, out_bf16, b0);
            st_bf16_lo_vec(bc1, out_bf16, b1);
            st_bf16_lo_vec(bc2, out_bf16, b2);
            st_bf16_lo_vec(bc3, out_bf16, b3);
            bc0[0] += 256; bc1[0] += 256; bc2[0] += 256; bc3[0] += 256;
            const bfloat128 pk0 = merge_two_bf16_lo_vecs(b0, b1);
            const bfloat128 pk1 = merge_two_bf16_lo_vecs(b2, b3);
#else
            const bfloat128 pk0 = pack_two_f32_to_bf16_vec(y0, y1);
            const bfloat128 pk1 = pack_two_f32_to_bf16_vec(y2, y3);
#endif
            vlm_write_bf16(slot + 0, pk0);
            vlm_write_bf16(slot + 1, pk1);
            am0 = v_bf16_max_b(am0, v_bf16_abs_b(pk0));
            am1 = v_bf16_max_b(am1, v_bf16_abs_b(pk1));
        }

        // Tail for D - D256 in {64, 128, 192}.
        bfloat128 pending_b = 0;
        int has_lo = 0;
        for (int d = D256; d < D; d += 64, ch++)
        {
            float64 w0;
            if (hoist_w)
            {
#if WEIGHT_F32
                w0 = g_norm_vlm[w_base + ch];
#else
                const bfloat128 wp = vlm_read_bf16(w_base + (ch >> 1));
                w0 = (ch & 1) ? unpack_bf16_hi(wp) : unpack_bf16_lo(wp);
#endif
            }
            else
            {
                wc0[0] = d;
#if WEIGHT_F32
                w0 = v_f32_ld_tnsr_b(wc0, weight);
#else
                w0 = ld_bf16_as_f32(wc0, weight);
#endif
            }
            const bfloat128 b0 = v_convert_f32_to_bf16_b((g_norm_vlm[ch] * rstd) * w0, 0, 0, 0, 1, 0);
#if OUT_BF16
            bc0[0] = d;
            st_bf16_lo_vec(bc0, out_bf16, b0);
#endif
            if (!has_lo)
            {
                pending_b = b0;
                has_lo = 1;
            }
            else
            {
                const bfloat128 pk = merge_two_bf16_lo_vecs(pending_b, b0);
                vlm_write_bf16(slot++, pk);
                am0 = v_bf16_max_b(am0, v_bf16_abs_b(pk));
                has_lo = 0;
            }
        }
        if (has_lo)
        {
            const bfloat128 pk = merge_two_bf16_lo_vecs(pending_b, (bfloat128)0);
            vlm_write_bf16(slot, pk);
            am0 = v_bf16_max_b(am0, v_bf16_abs_b(pk));
        }

        // Match dynamic_quant: scale = (amax + 1e-8) / fp8_max.
        const float64 amax = reduce_max_fast_nolut(bf16_vec_max_to_f32(v_bf16_max_b(am0, am1)), xmasks);
        const float64 sc_v = (amax + 1e-8f) * inv_fp8_max;
        const float64 inv_sc = rcp_pos_nolut(sc_v);
        int5 scc = {0, m, 0, 0, 0};
        v_f32_st_tnsr_partial(scc, out_sc, sc_v, 0, 0);

        // Pass 3: 4-way unrolled quantize from packed BF16 VLM and store FP8 E4M3.
        int5 fc0 = {0, m, 0, 0, 0}, fc1 = {64, m, 0, 0, 0}, fc2 = {128, m, 0, 0, 0}, fc3 = {192, m, 0, 0, 0};
        slot = 0;
        for (int d = 0; d < D256; d += 256, slot += 2)
        {
            const bfloat128 pk0 = vlm_read_bf16(slot + 0);
            const bfloat128 pk1 = vlm_read_bf16(slot + 1);
            const float64 q0 = v_f32_min_b(v_f32_max_b(unpack_bf16_lo(pk0) * inv_sc, -fp8_max), fp8_max);
            const float64 q1 = v_f32_min_b(v_f32_max_b(unpack_bf16_hi(pk0) * inv_sc, -fp8_max), fp8_max);
            const float64 q2 = v_f32_min_b(v_f32_max_b(unpack_bf16_lo(pk1) * inv_sc, -fp8_max), fp8_max);
            const float64 q3 = v_f32_min_b(v_f32_max_b(unpack_bf16_hi(pk1) * inv_sc, -fp8_max), fp8_max);
            st_f32_as_f8(fc0, out_f8, q0);
            st_f32_as_f8(fc1, out_f8, q1);
            st_f32_as_f8(fc2, out_f8, q2);
            st_f32_as_f8(fc3, out_f8, q3);
            fc0[0] += 256; fc1[0] += 256; fc2[0] += 256; fc3[0] += 256;
        }
        for (int d = D256; d < D128; d += 128, slot++)
        {
            const bfloat128 pk = vlm_read_bf16(slot);
            fc0[0] = d; fc1[0] = d + 64;
            st_f32_as_f8(fc0, out_f8, v_f32_min_b(v_f32_max_b(unpack_bf16_lo(pk) * inv_sc, -fp8_max), fp8_max));
            st_f32_as_f8(fc1, out_f8, v_f32_min_b(v_f32_max_b(unpack_bf16_hi(pk) * inv_sc, -fp8_max), fp8_max));
        }
        if (D & 64)
        {
            fc0[0] = D128;
            st_f32_as_f8(fc0, out_f8,
                         v_f32_min_b(v_f32_max_b(unpack_bf16_lo(vlm_read_bf16(slot)) * inv_sc, -fp8_max), fp8_max));
        }
    }
}
