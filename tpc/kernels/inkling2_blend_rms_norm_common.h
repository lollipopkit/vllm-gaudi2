// SPDX-License-Identifier: Apache-2.0
// Fused blend_rms_norm for Inkling 2 Mini loop message controller on Gaudi2 TPC:
//   x = (1 - tanh(forget)) * dest + tanh(gain) * message
//   out = bf16(x * rsqrt(mean(x^2) + norm_eps) * norm_weight)
//   (when OUTPUT_F8 == 1, also outputs natural-order per-row FP8 E4M3 + scale directly from VLM!)
//
// Wave 7 upgrade: 2x-unrolled Pass 1/2 eliminates all 13 VRF spills from 4x unrolling while
// keeping ONLY 1 SFU LUT (tanh_f32, 0x0) + zero-LUT rsqrt_nolut/rcp_pos_nolut and 10 KB VLM.
//
// Tensors (TPC dim order, dim0 fastest):
//   dest        [D, T, S] f32     (PyTorch [S, T, D])
//   message     [D, T, S] f32     (PyTorch [S, T, D])
//   gain_forget [2*D, T, S] bf16  (PyTorch [S, T, 2*D] from BF16 MME bmm)
//   norm_weight [D, S] bf16       (PyTorch [S, D])
//   out         [D, T, S] bf16    (PyTorch [S, T, D])
//   out_f8      [D, T, S] f8_143  (only when OUTPUT_F8 == 1)
//   out_scale   [T, S] f32        (only when OUTPUT_F8 == 1)
//
// Index space: [T, S] (2D).
#pragma once

#include "inkling2_bf16_common.h"

#ifndef OUTPUT_F8
#define OUTPUT_F8 0
#endif

#define MAX_CHUNKS 40
__local__ float64 g_blended[MAX_CHUNKS];

#if OUTPUT_F8
void main(tensor dest, tensor message, tensor gain_forget, tensor norm_weight,
          tensor out, tensor out_f8, tensor out_scale,
          float norm_eps, float inv_D, float fp8_max)
#else
void main(tensor dest, tensor message, tensor gain_forget, tensor norm_weight, tensor out,
          float norm_eps, float inv_D)
#endif
{
    const int D = get_dim_size(dest, 0);
    const int D128 = D & ~127;
    const int5 is_start = get_index_space_offset();
    const int5 is_end = get_index_space_size() + is_start;
    const tpc_xor_masks_t xmasks = make_xor_masks_nolut(read_lane_id_4b_b());
#if OUTPUT_F8
    const float inv_fp8_max = 1.0f / fp8_max;
#endif

    for (int s = is_start[1]; s < is_end[1]; s++)
    for (int t = is_start[0]; t < is_end[0]; t++)
    {
        float64 msq0 = 0.0f, msq1 = 0.0f;
        int5 c0 = {0, t, s, 0, 0}, c1 = {64, t, s, 0, 0};
        int5 f0 = {D, t, s, 0, 0}, f1 = {D + 64, t, s, 0, 0};
        int ch = 0;
        for (int d = 0; d < D128; d += 128, ch += 2)
        {
            const float64 dv0 = v_f32_ld_tnsr_b(c0, dest);
            const float64 dv1 = v_f32_ld_tnsr_b(c1, dest);
            const float64 mv0 = v_f32_ld_tnsr_b(c0, message);
            const float64 mv1 = v_f32_ld_tnsr_b(c1, message);
            const float64 g0 = tanh_f32(ld_bf16_as_f32(c0, gain_forget));
            const float64 g1 = tanh_f32(ld_bf16_as_f32(c1, gain_forget));
            const float64 fg0 = tanh_f32(ld_bf16_as_f32(f0, gain_forget));
            const float64 fg1 = tanh_f32(ld_bf16_as_f32(f1, gain_forget));
            c0[0] += 128; c1[0] += 128;
            f0[0] += 128; f1[0] += 128;
            const float64 x0 = v_f32_mac_b(g0, mv0, (1.0f - fg0) * dv0);
            const float64 x1 = v_f32_mac_b(g1, mv1, (1.0f - fg1) * dv1);
            g_blended[ch + 0] = x0;
            g_blended[ch + 1] = x1;
            msq0 = v_f32_mac_b(x0, x0, msq0);
            msq1 = v_f32_mac_b(x1, x1, msq1);
        }
        if (D & 64)
        {
            c0[0] = D128;
            f0[0] = D + D128;
            const float64 dv = v_f32_ld_tnsr_b(c0, dest);
            const float64 mv = v_f32_ld_tnsr_b(c0, message);
            const float64 g = tanh_f32(ld_bf16_as_f32(c0, gain_forget));
            const float64 fg = tanh_f32(ld_bf16_as_f32(f0, gain_forget));
            const float64 x = v_f32_mac_b(g, mv, (1.0f - fg) * dv);
            g_blended[ch] = x;
            msq0 = v_f32_mac_b(x, x, msq0);
        }
        const float64 rstd = rsqrt_nolut(reduce_add_fast_nolut(msq0 + msq1, xmasks) * inv_D + norm_eps);

        ch = 0;
        int5 oc0 = {0, t, s, 0, 0}, oc1 = {64, t, s, 0, 0};
        int5 wc0 = {0, s, 0, 0, 0}, wc1 = {64, s, 0, 0, 0};
#if OUTPUT_F8
        float64 am0 = 0.0f, am1 = 0.0f;
#endif
        for (int d = 0; d < D128; d += 128, ch += 2)
        {
            const float64 w0 = ld_bf16_as_f32(wc0, norm_weight);
            const float64 w1 = ld_bf16_as_f32(wc1, norm_weight);
            wc0[0] += 128; wc1[0] += 128;
            const float64 b0 = (g_blended[ch + 0] * rstd) * w0;
            const float64 b1 = (g_blended[ch + 1] * rstd) * w1;
            st_f32_as_bf16(oc0, out, b0);
            st_f32_as_bf16(oc1, out, b1);
            oc0[0] += 128; oc1[0] += 128;
#if OUTPUT_F8
            g_blended[ch + 0] = b0;
            g_blended[ch + 1] = b1;
            am0 = v_f32_max_b(am0, v_f32_abs_b(b0));
            am1 = v_f32_max_b(am1, v_f32_abs_b(b1));
#endif
        }
        if (D & 64)
        {
            oc0[0] = D128;
            wc0[0] = D128;
            const float64 b0 = (g_blended[ch] * rstd) * ld_bf16_as_f32(wc0, norm_weight);
            st_f32_as_bf16(oc0, out, b0);
#if OUTPUT_F8
            g_blended[ch] = b0;
            am0 = v_f32_max_b(am0, v_f32_abs_b(b0));
#endif
        }
#if OUTPUT_F8
        const float64 amax = reduce_max_fast_nolut(v_f32_max_b(am0, am1), xmasks);
        const float64 sc_v = v_f32_max_b(amax * inv_fp8_max, 1e-12f);
        const float64 inv_sc = rcp_pos_nolut(sc_v);
        int5 scc = {t, s, 0, 0, 0};
        v_f32_st_tnsr_partial(scc, out_scale, sc_v, 0, 0);

        int5 fc0 = {0, t, s, 0, 0}, fc1 = {64, t, s, 0, 0};
        ch = 0;
        for (int d = 0; d < D128; d += 128, ch += 2)
        {
            const float64 q0 = v_f32_min_b(v_f32_max_b(g_blended[ch + 0] * inv_sc, -fp8_max), fp8_max);
            const float64 q1 = v_f32_min_b(v_f32_max_b(g_blended[ch + 1] * inv_sc, -fp8_max), fp8_max);
            st_f32_as_f8(fc0, out_f8, q0);
            st_f32_as_f8(fc1, out_f8, q1);
            fc0[0] += 128; fc1[0] += 128;
        }
        if (D & 64)
        {
            fc0[0] = D128;
            st_f32_as_f8(fc0, out_f8, v_f32_min_b(v_f32_max_b(g_blended[ch] * inv_sc, -fp8_max), fp8_max));
        }
#endif
    }
}
