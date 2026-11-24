// SPDX-License-Identifier: Apache-2.0
// Fused Inkling2 Block Attention Residual (attn_res) implementation for Gaudi2 TPC.
//
// Design:
//   - Zero VRF spills (2x/4x-unrolled Pass 1/2/3 fits comfortably inside 32 VRF regs and hides
//     the 4-cycle VPU latency chain).
//   - Interleaved reduce_add_pair_nolut computes (sum_sq, dot) per source in ONE 6-step butterfly.
//   - Optional OUTPUT_F8=1 mode (inkling2_attn_res_norm_f8_gaudi2) fuses per-row FP8 E4M3 quant
//     directly on VLM g_buf (0 extra HBM reads), eliminating 47+ quantize_fp8 kernels + gaps.
//   - Uses ONLY 1 SFU LUT (exp_f32=0x100) + zero-LUT rsqrt_nolut/rcp_pos_nolut (ZERO LUT misses)
//     and keeps VLM <= 14 KB inside the 16 KB SFU-safe budget.
//
// Included by:
//   - inkling2_attn_res_norm_bf16_gaudi2.c (HAS_NORM=1, OUTPUT_F8=0)
//   - inkling2_attn_res_norm_f8_gaudi2.c   (HAS_NORM=1, OUTPUT_F8=1)
//   - inkling2_attn_res_f32_gaudi2.c       (HAS_NORM=0, OUTPUT_F8=0)
//
// Tensors (TPC dim order, dim0 fastest):
//   stack       [D, T, N_stack_dim] bf16  (dummy [D, T, 1] when n_stack == 0)
//   last        [D, T] bf16
//   addend      [D, T] bf16               (dummy [D, T] when has_addend == 0)
//   queries     [D, Q] f32                (pre-scaled by D^-0.5)
//   norm_weight [D, Q] bf16               (only when HAS_NORM == 1)
//   out         [D, T, Q] bf16 (HAS_NORM=1) or f32 (HAS_NORM=0)
//   out_f8      [D, T, Q] f8_143          (only when OUTPUT_F8 == 1)
//   out_scale   [T, Q] f32                (only when OUTPUT_F8 == 1)
//
// Index space: [T, Q] (2D).
#pragma once

#include "inkling2_bf16_common.h"

#ifndef OUTPUT_F8
#define OUTPUT_F8 0
#endif

#define MAX_CHUNKS 40
// 40 * 256 B = 10 KB + 16 * 256 B = 4 KB = 14 KB VLM (<= 16 KB SFU-safe budget).
__local__ float64 g_buf[MAX_CHUNKS];
__local__ float64 g_coeff[16];

#if HAS_NORM && OUTPUT_F8
void main(tensor stack, tensor last, tensor addend, tensor queries, tensor norm_weight,
          tensor out, tensor out_f8, tensor out_scale,
          int n_stack, int has_addend, int normalize_values, float eps, float norm_eps, float inv_D, float fp8_max)
#elif HAS_NORM
void main(tensor stack, tensor last, tensor addend, tensor queries, tensor norm_weight, tensor out,
          int n_stack, int has_addend, int normalize_values, float eps, float norm_eps, float inv_D)
#else
void main(tensor stack, tensor last, tensor addend, tensor queries, tensor out,
          int n_stack, int has_addend, int normalize_values, float eps, float inv_D)
#endif
{
    const int D = get_dim_size(last, 0);
    const int D128 = D & ~127;
    const int N = n_stack + 1;
    const int5 is_start = get_index_space_offset();
    const int5 is_end = get_index_space_size() + is_start;
    const uint64 lane_id = read_lane_id_4b_b();
    const bool64 valid_n = v_u32_cmp_less_b(lane_id, (unsigned int)N);
    const tpc_xor_masks_t xmasks = make_xor_masks_nolut(lane_id);
#if OUTPUT_F8
    const float inv_fp8_max = 1.0f / fp8_max;
#endif

    for (int q = is_start[1]; q < is_end[1]; q++)
    {
#if !HAS_NORM
        {
            int5 qc0 = {0, q, 0, 0, 0}, qc1 = {64, q, 0, 0, 0};
            int ch = 0;
            for (int d = 0; d < D128; d += 128, ch += 2)
            {
                g_buf[ch + 0] = v_f32_ld_tnsr_b(qc0, queries);
                g_buf[ch + 1] = v_f32_ld_tnsr_b(qc1, queries);
                qc0[0] += 128; qc1[0] += 128;
            }
            if (D & 64)
            {
                qc0[0] = D128;
                g_buf[ch] = v_f32_ld_tnsr_b(qc0, queries);
            }
        }
#endif

        for (int t = is_start[0]; t < is_end[0]; t++)
        {
#if HAS_NORM
            {
                int5 qc0 = {0, q, 0, 0, 0}, qc1 = {64, q, 0, 0, 0};
                int ch = 0;
                for (int d = 0; d < D128; d += 128, ch += 2)
                {
                    g_buf[ch + 0] = v_f32_ld_tnsr_b(qc0, queries);
                    g_buf[ch + 1] = v_f32_ld_tnsr_b(qc1, queries);
                    qc0[0] += 128; qc1[0] += 128;
                }
                if (D & 64)
                {
                    qc0[0] = D128;
                    g_buf[ch] = v_f32_ld_tnsr_b(qc0, queries);
                }
            }
#endif

            float64 sum_sq_v = 0.0f;
            float64 dot_v = 0.0f;

            // Pass 1a: stack[0 .. n_stack - 1] (4x unrolled for !HAS_NORM, 2x unrolled for HAS_NORM).
            for (int n = 0; n < n_stack; n++)
            {
#if !HAS_NORM
                const int D256 = D & ~255;
                float64 s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
                float64 d0 = 0.0f, d1 = 0.0f, d2 = 0.0f, d3 = 0.0f;
                int5 sc0 = {0, t, n, 0, 0}, sc1 = {64, t, n, 0, 0};
                int5 sc2 = {128, t, n, 0, 0}, sc3 = {192, t, n, 0, 0};
                int ch = 0;
                for (int d = 0; d < D256; d += 256, ch += 4)
                {
                    const float64 x0 = ld_bf16_as_f32(sc0, stack);
                    const float64 x1 = ld_bf16_as_f32(sc1, stack);
                    const float64 x2 = ld_bf16_as_f32(sc2, stack);
                    const float64 x3 = ld_bf16_as_f32(sc3, stack);
                    sc0[0] += 256; sc1[0] += 256; sc2[0] += 256; sc3[0] += 256;
                    s0 = v_f32_mac_b(x0, x0, s0); d0 = v_f32_mac_b(x0, g_buf[ch + 0], d0);
                    s1 = v_f32_mac_b(x1, x1, s1); d1 = v_f32_mac_b(x1, g_buf[ch + 1], d1);
                    s2 = v_f32_mac_b(x2, x2, s2); d2 = v_f32_mac_b(x2, g_buf[ch + 2], d2);
                    s3 = v_f32_mac_b(x3, x3, s3); d3 = v_f32_mac_b(x3, g_buf[ch + 3], d3);
                }
                for (int d = D256; d < D; d += 64, ch++)
                {
                    sc0[0] = d;
                    const float64 x0 = ld_bf16_as_f32(sc0, stack);
                    s0 = v_f32_mac_b(x0, x0, s0);
                    d0 = v_f32_mac_b(x0, g_buf[ch], d0);
                }
                const float64_pair_t sd = reduce_add_pair_nolut((s0 + s1) + (s2 + s3), (d0 + d1) + (d2 + d3), xmasks);
#else
                float64 s0 = 0.0f, s1 = 0.0f;
                float64 d0 = 0.0f, d1 = 0.0f;
                int5 sc0 = {0, t, n, 0, 0}, sc1 = {64, t, n, 0, 0};
                int ch = 0;
                for (int d = 0; d < D128; d += 128, ch += 2)
                {
                    const float64 x0 = ld_bf16_as_f32(sc0, stack);
                    const float64 x1 = ld_bf16_as_f32(sc1, stack);
                    const float64 qv0 = g_buf[ch + 0];
                    const float64 qv1 = g_buf[ch + 1];
                    sc0[0] += 128; sc1[0] += 128;
                    s0 = v_f32_mac_b(x0, x0, s0);
                    s1 = v_f32_mac_b(x1, x1, s1);
                    d0 = v_f32_mac_b(x0, qv0, d0);
                    d1 = v_f32_mac_b(x1, qv1, d1);
                }
                if (D & 64)
                {
                    sc0[0] = D128;
                    const float64 x0 = ld_bf16_as_f32(sc0, stack);
                    const float64 qv0 = g_buf[ch];
                    s0 = v_f32_mac_b(x0, x0, s0);
                    d0 = v_f32_mac_b(x0, qv0, d0);
                }
                const float64_pair_t sd = reduce_add_pair_nolut(s0 + s1, d0 + d1, xmasks);
#endif
                const bool64 is_n = v_u32_cmp_eq_b(lane_id, (unsigned int)n);
                sum_sq_v = v_f32_mov_vb(sd.v1, 0, sum_sq_v, is_n, 0);
                dot_v = v_f32_mov_vb(sd.v2, 0, dot_v, is_n, 0);
            }

            // Pass 1b: final source n = n_stack (last [+ addend]), unified loop.
            {
                float64 s0 = 0.0f, s1 = 0.0f;
                float64 d0 = 0.0f, d1 = 0.0f;
                int5 lc0 = {0, t, 0, 0, 0}, lc1 = {64, t, 0, 0, 0};
                int ch = 0;
                for (int d = 0; d < D128; d += 128, ch += 2)
                {
                    float64 x0 = ld_bf16_as_f32(lc0, last);
                    float64 x1 = ld_bf16_as_f32(lc1, last);
                    if (has_addend)
                    {
                        x0 += ld_bf16_as_f32(lc0, addend);
                        x1 += ld_bf16_as_f32(lc1, addend);
                    }
                    const float64 qv0 = g_buf[ch + 0];
                    const float64 qv1 = g_buf[ch + 1];
                    lc0[0] += 128; lc1[0] += 128;
                    s0 = v_f32_mac_b(x0, x0, s0);
                    s1 = v_f32_mac_b(x1, x1, s1);
                    d0 = v_f32_mac_b(x0, qv0, d0);
                    d1 = v_f32_mac_b(x1, qv1, d1);
                }
                if (D & 64)
                {
                    lc0[0] = D128;
                    float64 x0 = ld_bf16_as_f32(lc0, last);
                    if (has_addend)
                        x0 += ld_bf16_as_f32(lc0, addend);
                    const float64 qv0 = g_buf[ch];
                    s0 = v_f32_mac_b(x0, x0, s0);
                    d0 = v_f32_mac_b(x0, qv0, d0);
                }
                const float64_pair_t sd = reduce_add_pair_nolut(s0 + s1, d0 + d1, xmasks);
                const bool64 is_n = v_u32_cmp_eq_b(lane_id, (unsigned int)n_stack);
                sum_sq_v = v_f32_mov_vb(sd.v1, 0, sum_sq_v, is_n, 0);
                dot_v = v_f32_mov_vb(sd.v2, 0, dot_v, is_n, 0);
            }

            // Vectorized softmax across all N sources (ONLY 1 SFU LUT: exp_f32=0x100).
            const float64 inv_rms_v = rsqrt_nolut(sum_sq_v * inv_D + eps);
            const float64 logits_v = dot_v * inv_rms_v;
            const float64 masked = v_f32_mov_vb(-1e30f, 0, logits_v, valid_n, 1);
            const float64 m = reduce_max_fast_nolut(masked, xmasks);
            const float64 e = v_f32_mov_vb(0.0f, 0, exp_f32(v_f32_max_b(masked - m, -80.0f)), valid_n, 1);
            const float64 inv_d = rcp_pos_nolut(reduce_add_fast_nolut(e, xmasks));
            const float64 p = e * inv_d;
            const float64 coeff_v = normalize_values ? (p * inv_rms_v) : p;

            for (int n = 0; n < n_stack; n++)
                g_coeff[n] = v_broadcast_element_f32(coeff_v, n);
            const float64 c_last = v_broadcast_element_f32(coeff_v, n_stack);

            // Pass 2: d OUTER, n INNER -> 0 VRF spills & 0 inner g_buf traffic!
#if HAS_NORM
            float64 msq0 = 0.0f, msq1 = 0.0f;
            {
                int5 lc0 = {0, t, 0, 0, 0}, lc1 = {64, t, 0, 0, 0};
                int ch = 0;
                for (int d = 0; d < D128; d += 128, ch += 2)
                {
                    float64 x0 = ld_bf16_as_f32(lc0, last);
                    float64 x1 = ld_bf16_as_f32(lc1, last);
                    if (has_addend)
                    {
                        x0 += ld_bf16_as_f32(lc0, addend);
                        x1 += ld_bf16_as_f32(lc1, addend);
                    }
                    float64 m0 = x0 * c_last;
                    float64 m1 = x1 * c_last;
                    lc0[0] += 128; lc1[0] += 128;

                    int5 sc0 = {d, t, 0, 0, 0}, sc1 = {d + 64, t, 0, 0, 0};
                    for (int n = 0; n < n_stack; n++)
                    {
                        const float64 c_n = g_coeff[n];
                        m0 = v_f32_mac_b(ld_bf16_as_f32(sc0, stack), c_n, m0);
                        m1 = v_f32_mac_b(ld_bf16_as_f32(sc1, stack), c_n, m1);
                        sc0[2] += 1; sc1[2] += 1;
                    }
                    msq0 = v_f32_mac_b(m0, m0, msq0);
                    msq1 = v_f32_mac_b(m1, m1, msq1);
                    g_buf[ch + 0] = m0;
                    g_buf[ch + 1] = m1;
                }
                if (D & 64)
                {
                    lc0[0] = D128;
                    float64 m0 = has_addend
                                     ? (ld_bf16_as_f32(lc0, last) + ld_bf16_as_f32(lc0, addend)) * c_last
                                     : ld_bf16_as_f32(lc0, last) * c_last;
                    int5 sc0 = {D128, t, 0, 0, 0};
                    for (int n = 0; n < n_stack; n++)
                    {
                        m0 = v_f32_mac_b(ld_bf16_as_f32(sc0, stack), g_coeff[n], m0);
                        sc0[2] += 1;
                    }
                    msq0 = v_f32_mac_b(m0, m0, msq0);
                    g_buf[ch] = m0;
                }
            }
#else
            {
                const int D256 = D & ~255;
                int5 lc0 = {0, t, 0, 0, 0}, lc1 = {64, t, 0, 0, 0};
                int5 lc2 = {128, t, 0, 0, 0}, lc3 = {192, t, 0, 0, 0};
                int5 oc0 = {0, t, q, 0, 0}, oc1 = {64, t, q, 0, 0};
                int5 oc2 = {128, t, q, 0, 0}, oc3 = {192, t, q, 0, 0};
                for (int d = 0; d < D256; d += 256)
                {
                    float64 x0 = ld_bf16_as_f32(lc0, last), x1 = ld_bf16_as_f32(lc1, last);
                    float64 x2 = ld_bf16_as_f32(lc2, last), x3 = ld_bf16_as_f32(lc3, last);
                    if (has_addend)
                    {
                        x0 += ld_bf16_as_f32(lc0, addend); x1 += ld_bf16_as_f32(lc1, addend);
                        x2 += ld_bf16_as_f32(lc2, addend); x3 += ld_bf16_as_f32(lc3, addend);
                    }
                    float64 m0 = x0 * c_last, m1 = x1 * c_last, m2 = x2 * c_last, m3 = x3 * c_last;
                    lc0[0] += 256; lc1[0] += 256; lc2[0] += 256; lc3[0] += 256;

                    int5 sc0 = {d, t, 0, 0, 0}, sc1 = {d + 64, t, 0, 0, 0};
                    int5 sc2 = {d + 128, t, 0, 0, 0}, sc3 = {d + 192, t, 0, 0, 0};
                    for (int n = 0; n < n_stack; n++)
                    {
                        const float64 c_n = g_coeff[n];
                        m0 = v_f32_mac_b(ld_bf16_as_f32(sc0, stack), c_n, m0);
                        m1 = v_f32_mac_b(ld_bf16_as_f32(sc1, stack), c_n, m1);
                        m2 = v_f32_mac_b(ld_bf16_as_f32(sc2, stack), c_n, m2);
                        m3 = v_f32_mac_b(ld_bf16_as_f32(sc3, stack), c_n, m3);
                        sc0[2] += 1; sc1[2] += 1; sc2[2] += 1; sc3[2] += 1;
                    }
                    v_f32_st_tnsr(oc0, out, m0); v_f32_st_tnsr(oc1, out, m1);
                    v_f32_st_tnsr(oc2, out, m2); v_f32_st_tnsr(oc3, out, m3);
                    oc0[0] += 256; oc1[0] += 256; oc2[0] += 256; oc3[0] += 256;
                }
                for (int d = D256; d < D; d += 64)
                {
                    lc0[0] = d;
                    float64 m0 = has_addend
                                     ? (ld_bf16_as_f32(lc0, last) + ld_bf16_as_f32(lc0, addend)) * c_last
                                     : ld_bf16_as_f32(lc0, last) * c_last;
                    int5 sc0 = {d, t, 0, 0, 0};
                    for (int n = 0; n < n_stack; n++)
                    {
                        m0 = v_f32_mac_b(ld_bf16_as_f32(sc0, stack), g_coeff[n], m0);
                        sc0[2] += 1;
                    }
                    oc0[0] = d;
                    v_f32_st_tnsr(oc0, out, m0);
                }
            }
#endif

#if HAS_NORM
            // Pass 3: rstd + 2x-unrolled RMSNorm -> bf16 (+ optional fused FP8 E4M3 quantization).
            const float64 rstd = rsqrt_nolut(reduce_add_fast_nolut(msq0 + msq1, xmasks) * inv_D + norm_eps);
            int5 wc0 = {0, q, 0, 0, 0}, wc1 = {64, q, 0, 0, 0};
            int5 oc0 = {0, t, q, 0, 0}, oc1 = {64, t, q, 0, 0};
#if OUTPUT_F8
            float64 am0 = 0.0f, am1 = 0.0f;
#endif
            int ch = 0;
            for (int d = 0; d < D128; d += 128, ch += 2)
            {
                const float64 w0 = ld_bf16_as_f32(wc0, norm_weight);
                const float64 w1 = ld_bf16_as_f32(wc1, norm_weight);
                wc0[0] += 128; wc1[0] += 128;
                const float64 b0 = (g_buf[ch + 0] * rstd) * w0;
                const float64 b1 = (g_buf[ch + 1] * rstd) * w1;
                st_f32_as_bf16(oc0, out, b0);
                st_f32_as_bf16(oc1, out, b1);
                oc0[0] += 128; oc1[0] += 128;
#if OUTPUT_F8
                g_buf[ch + 0] = b0;
                g_buf[ch + 1] = b1;
                am0 = v_f32_max_b(am0, v_f32_abs_b(b0));
                am1 = v_f32_max_b(am1, v_f32_abs_b(b1));
#endif
            }
            if (D & 64)
            {
                wc0[0] = D128;
                oc0[0] = D128;
                const float64 b0 = (g_buf[ch] * rstd) * ld_bf16_as_f32(wc0, norm_weight);
                st_f32_as_bf16(oc0, out, b0);
#if OUTPUT_F8
                g_buf[ch] = b0;
                am0 = v_f32_max_b(am0, v_f32_abs_b(b0));
#endif
            }
#if OUTPUT_F8
            // Pass 3b (pure VLM -> FP8 store, ZERO extra HBM reads!).
            const float64 amax = reduce_max_fast_nolut(v_f32_max_b(am0, am1), xmasks);
            const float64 sc_v = v_f32_max_b(amax * inv_fp8_max, 1e-12f);
            const float64 inv_sc = rcp_pos_nolut(sc_v);
            int5 scc = {t, q, 0, 0, 0};
            v_f32_st_tnsr_partial(scc, out_scale, sc_v, 0, 0);

            int5 fc0 = {0, t, q, 0, 0}, fc1 = {64, t, q, 0, 0};
            ch = 0;
            for (int d = 0; d < D128; d += 128, ch += 2)
            {
                const float64 q0 = v_f32_min_b(v_f32_max_b(g_buf[ch + 0] * inv_sc, -fp8_max), fp8_max);
                const float64 q1 = v_f32_min_b(v_f32_max_b(g_buf[ch + 1] * inv_sc, -fp8_max), fp8_max);
                st_f32_as_f8(fc0, out_f8, q0);
                st_f32_as_f8(fc1, out_f8, q1);
                fc0[0] += 128; fc1[0] += 128;
            }
            if (D & 64)
            {
                fc0[0] = D128;
                st_f32_as_f8(fc0, out_f8, v_f32_min_b(v_f32_max_b(g_buf[ch] * inv_sc, -fp8_max), fp8_max));
            }
#endif
#endif
        }
    }
}
