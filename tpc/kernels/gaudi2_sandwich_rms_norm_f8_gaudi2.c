// SPDX-License-Identifier: Apache-2.0
// Generic zero-LUT fused Sandwich RMSNorm + Residual Add + Pre-RMSNorm + Per-Row FP8 E4M3 Quantization:
// Replaces 2 FusedRMSNorms + residual add + dynamic_quant (e.g. LFM3 post_operator_norm + ffn_norm -> w13):
//   x_post  = bf16((x.f32 * rsqrt(mean(x.f32^2) + eps)) * post_w.f32)
//   out_res = bf16(residual.f32 + x_post)
//   y       = bf16((out_res.f32 * rsqrt(mean(out_res.f32^2) + eps)) * pre_w.f32)
//   out_sc  = (max(|y|) + 1e-8) / fp8_max
//   out_f8  = f8_143(clamp(y / out_sc, -fp8_max, fp8_max))
//
// Zero SFU LUTs -> full 64 KiB VLM available. 4-vector (256-elem) unrolling across all 4 passes,
// single BF16 conversion on out_res and packed BF16 VLM on y (halves VLM traffic in Pass 4), and
// VLM hoisting of both post_w and pre_w across rows for prefill (D <= 7680).
//
// Tensors (TPC dim order, dim0 fastest):
//   x        [D, M] bf16
//   residual [D, M] bf16
//   post_w   [D]    bf16
//   pre_w    [D]    bf16
//   out_f8   [D, M] f8_143
//   out_sc   [1, M] f32
//   out_res  [D, M] bf16
//
// Index space: [1, M] (dim0 = full row reduction, dim1 = row slice -> clean SRAM slicing along M).
#include "gaudi2_fp8_common.h"

#define MAX_SANDWICH_VLM_SLOTS 240  // 240 * 256 B = 60 KiB VLM
__local__ float64 g_sw_vlm[MAX_SANDWICH_VLM_SLOTS];

static inline bfloat128 sw_vlm_read_bf16(int idx)
{
    union {
        float64 f;
        bfloat128 b;
    } cvt;
    cvt.f = g_sw_vlm[idx];
    return cvt.b;
}

static inline void sw_vlm_write_bf16(int idx, bfloat128 v)
{
    union {
        float64 f;
        bfloat128 b;
    } cvt;
    cvt.b = v;
    g_sw_vlm[idx] = cvt.f;
}

void main(tensor x, tensor residual, tensor post_w, tensor pre_w,
          tensor out_f8, tensor out_sc, tensor out_res,
          float eps, float inv_D, float fp8_max)
{
    const int D = get_dim_size(x, 0);
    const int nchunks = D >> 6;  // D / 64
    const int D256 = D & ~255;
    const int D128 = D & ~127;
    const int5 is_start = get_index_space_offset();
    const int5 is_end = get_index_space_size() + is_start;
    const tpc_xor_masks_t xmasks = make_xor_masks_nolut(read_lane_id_4b_b());
    const float inv_fp8_max = 1.0f / fp8_max;

    const int rows = is_end[1] - is_start[1];
    const int w_slots = (nchunks + 1) >> 1;
    const int post_base = nchunks;
    const int pre_base = post_base + w_slots;
    const int hoist_w = (rows > 1) && (pre_base + w_slots <= MAX_SANDWICH_VLM_SLOTS);
    if (hoist_w)
    {
        int5 wc0 = {0, 0, 0, 0, 0}, wc1 = {64, 0, 0, 0, 0};
        int ws = 0;
        for (int d = 0; d < D128; d += 128, ws++)
        {
            sw_vlm_write_bf16(post_base + ws,
                              pack_two_f32_to_bf16_vec(ld_bf16_as_f32(wc0, post_w), ld_bf16_as_f32(wc1, post_w)));
            sw_vlm_write_bf16(pre_base + ws,
                              pack_two_f32_to_bf16_vec(ld_bf16_as_f32(wc0, pre_w), ld_bf16_as_f32(wc1, pre_w)));
            wc0[0] += 128; wc1[0] += 128;
        }
        if (D & 64)
        {
            wc0[0] = D128;
            sw_vlm_write_bf16(post_base + ws, pack_two_f32_to_bf16_vec(ld_bf16_as_f32(wc0, post_w), (float64)0.0f));
            sw_vlm_write_bf16(pre_base + ws, pack_two_f32_to_bf16_vec(ld_bf16_as_f32(wc0, pre_w), (float64)0.0f));
        }
    }

    for (int m = is_start[1]; m < is_end[1]; m++)
    {
        // Pass 1: 4-way unrolled load x, cache FP32 in VLM, compute post-norm rstd.
        float64 msq0 = 0.0f, msq1 = 0.0f, msq2 = 0.0f, msq3 = 0.0f;
        int5 c0 = {0, m, 0, 0, 0}, c1 = {64, m, 0, 0, 0}, c2 = {128, m, 0, 0, 0}, c3 = {192, m, 0, 0, 0};
        int ch = 0;
        for (int d = 0; d < D256; d += 256, ch += 4)
        {
            const float64 x0 = ld_bf16_as_f32(c0, x);
            const float64 x1 = ld_bf16_as_f32(c1, x);
            const float64 x2 = ld_bf16_as_f32(c2, x);
            const float64 x3 = ld_bf16_as_f32(c3, x);
            c0[0] += 256; c1[0] += 256; c2[0] += 256; c3[0] += 256;
            g_sw_vlm[ch + 0] = x0;
            g_sw_vlm[ch + 1] = x1;
            g_sw_vlm[ch + 2] = x2;
            g_sw_vlm[ch + 3] = x3;
            msq0 = v_f32_mac_b(x0, x0, msq0);
            msq1 = v_f32_mac_b(x1, x1, msq1);
            msq2 = v_f32_mac_b(x2, x2, msq2);
            msq3 = v_f32_mac_b(x3, x3, msq3);
        }
        for (int d = D256; d < D; d += 64, ch++)
        {
            c0[0] = d;
            const float64 x0 = ld_bf16_as_f32(c0, x);
            g_sw_vlm[ch] = x0;
            msq0 = v_f32_mac_b(x0, x0, msq0);
        }
        const float64 rstd_post = rsqrt_nolut(reduce_add_fast_nolut((msq0 + msq1) + (msq2 + msq3), xmasks) * inv_D + eps);

        // Pass 2: 4-way unrolled post_w + residual add, convert r to BF16 ONCE to store out_res and unpack back.
        float64 rsq0 = 0.0f, rsq1 = 0.0f, rsq2 = 0.0f, rsq3 = 0.0f;
        int5 wc0 = {0, 0, 0, 0, 0}, wc1 = {64, 0, 0, 0, 0}, wc2 = {128, 0, 0, 0, 0}, wc3 = {192, 0, 0, 0, 0};
        c0[0] = 0; c1[0] = 64; c2[0] = 128; c3[0] = 192;
        ch = 0;
        int ws = 0;
        for (int d = 0; d < D256; d += 256, ch += 4, ws += 2)
        {
            float64 pw0, pw1, pw2, pw3;
            if (hoist_w)
            {
                const bfloat128 wp0 = sw_vlm_read_bf16(post_base + ws + 0);
                const bfloat128 wp1 = sw_vlm_read_bf16(post_base + ws + 1);
                pw0 = unpack_bf16_lo(wp0); pw1 = unpack_bf16_hi(wp0);
                pw2 = unpack_bf16_lo(wp1); pw3 = unpack_bf16_hi(wp1);
            }
            else
            {
                pw0 = ld_bf16_as_f32(wc0, post_w); pw1 = ld_bf16_as_f32(wc1, post_w);
                pw2 = ld_bf16_as_f32(wc2, post_w); pw3 = ld_bf16_as_f32(wc3, post_w);
                wc0[0] += 256; wc1[0] += 256; wc2[0] += 256; wc3[0] += 256;
            }
            const float64 xp0 = round_f32_via_bf16((g_sw_vlm[ch + 0] * rstd_post) * pw0);
            const float64 xp1 = round_f32_via_bf16((g_sw_vlm[ch + 1] * rstd_post) * pw1);
            const float64 xp2 = round_f32_via_bf16((g_sw_vlm[ch + 2] * rstd_post) * pw2);
            const float64 xp3 = round_f32_via_bf16((g_sw_vlm[ch + 3] * rstd_post) * pw3);
            const bfloat128 br0 = v_convert_f32_to_bf16_b(ld_bf16_as_f32(c0, residual) + xp0, 0, 0, 0, 1, 0);
            const bfloat128 br1 = v_convert_f32_to_bf16_b(ld_bf16_as_f32(c1, residual) + xp1, 0, 0, 0, 1, 0);
            const bfloat128 br2 = v_convert_f32_to_bf16_b(ld_bf16_as_f32(c2, residual) + xp2, 0, 0, 0, 1, 0);
            const bfloat128 br3 = v_convert_f32_to_bf16_b(ld_bf16_as_f32(c3, residual) + xp3, 0, 0, 0, 1, 0);
            st_bf16_lo_vec(c0, out_res, br0);
            st_bf16_lo_vec(c1, out_res, br1);
            st_bf16_lo_vec(c2, out_res, br2);
            st_bf16_lo_vec(c3, out_res, br3);
            c0[0] += 256; c1[0] += 256; c2[0] += 256; c3[0] += 256;
            const float64 r0 = unpack_bf16_lo(br0);
            const float64 r1 = unpack_bf16_lo(br1);
            const float64 r2 = unpack_bf16_lo(br2);
            const float64 r3 = unpack_bf16_lo(br3);
            g_sw_vlm[ch + 0] = r0;
            g_sw_vlm[ch + 1] = r1;
            g_sw_vlm[ch + 2] = r2;
            g_sw_vlm[ch + 3] = r3;
            rsq0 = v_f32_mac_b(r0, r0, rsq0);
            rsq1 = v_f32_mac_b(r1, r1, rsq1);
            rsq2 = v_f32_mac_b(r2, r2, rsq2);
            rsq3 = v_f32_mac_b(r3, r3, rsq3);
        }
        for (int d = D256; d < D; d += 64, ch++)
        {
            float64 pw0;
            if (hoist_w)
            {
                const bfloat128 wp = sw_vlm_read_bf16(post_base + (ch >> 1));
                pw0 = (ch & 1) ? unpack_bf16_hi(wp) : unpack_bf16_lo(wp);
            }
            else
            {
                wc0[0] = d;
                pw0 = ld_bf16_as_f32(wc0, post_w);
            }
            c0[0] = d;
            const float64 xp0 = round_f32_via_bf16((g_sw_vlm[ch] * rstd_post) * pw0);
            const bfloat128 br0 = v_convert_f32_to_bf16_b(ld_bf16_as_f32(c0, residual) + xp0, 0, 0, 0, 1, 0);
            st_bf16_lo_vec(c0, out_res, br0);
            const float64 r0 = unpack_bf16_lo(br0);
            g_sw_vlm[ch] = r0;
            rsq0 = v_f32_mac_b(r0, r0, rsq0);
        }
        const float64 rstd_pre = rsqrt_nolut(reduce_add_fast_nolut((rsq0 + rsq1) + (rsq2 + rsq3), xmasks) * inv_D + eps);

        // Pass 3: 4-way unrolled pre_w, convert once to BF16, pack into VLM[0..D/128-1], 128-wide BF16 amax.
        bfloat128 am0 = 0, am1 = 0;
        wc0[0] = 0; wc1[0] = 64; wc2[0] = 128; wc3[0] = 192;
        ch = 0;
        int slot = 0;
        for (int d = 0; d < D256; d += 256, ch += 4, slot += 2)
        {
            float64 nw0, nw1, nw2, nw3;
            if (hoist_w)
            {
                const bfloat128 wp0 = sw_vlm_read_bf16(pre_base + slot + 0);
                const bfloat128 wp1 = sw_vlm_read_bf16(pre_base + slot + 1);
                nw0 = unpack_bf16_lo(wp0); nw1 = unpack_bf16_hi(wp0);
                nw2 = unpack_bf16_lo(wp1); nw3 = unpack_bf16_hi(wp1);
            }
            else
            {
                nw0 = ld_bf16_as_f32(wc0, pre_w); nw1 = ld_bf16_as_f32(wc1, pre_w);
                nw2 = ld_bf16_as_f32(wc2, pre_w); nw3 = ld_bf16_as_f32(wc3, pre_w);
                wc0[0] += 256; wc1[0] += 256; wc2[0] += 256; wc3[0] += 256;
            }
            const bfloat128 pk0 = pack_two_f32_to_bf16_vec((g_sw_vlm[ch + 0] * rstd_pre) * nw0,
                                                           (g_sw_vlm[ch + 1] * rstd_pre) * nw1);
            const bfloat128 pk1 = pack_two_f32_to_bf16_vec((g_sw_vlm[ch + 2] * rstd_pre) * nw2,
                                                           (g_sw_vlm[ch + 3] * rstd_pre) * nw3);
            sw_vlm_write_bf16(slot + 0, pk0);
            sw_vlm_write_bf16(slot + 1, pk1);
            am0 = v_bf16_max_b(am0, v_bf16_abs_b(pk0));
            am1 = v_bf16_max_b(am1, v_bf16_abs_b(pk1));
        }
        float64 pending_lo = 0.0f;
        int has_lo = 0;
        for (int d = D256; d < D; d += 64, ch++)
        {
            float64 nw0;
            if (hoist_w)
            {
                const bfloat128 wp = sw_vlm_read_bf16(pre_base + (ch >> 1));
                nw0 = (ch & 1) ? unpack_bf16_hi(wp) : unpack_bf16_lo(wp);
            }
            else
            {
                wc0[0] = d;
                nw0 = ld_bf16_as_f32(wc0, pre_w);
            }
            const float64 y0 = (g_sw_vlm[ch] * rstd_pre) * nw0;
            if (!has_lo)
            {
                pending_lo = y0;
                has_lo = 1;
            }
            else
            {
                const bfloat128 pk = pack_two_f32_to_bf16_vec(pending_lo, y0);
                sw_vlm_write_bf16(slot++, pk);
                am0 = v_bf16_max_b(am0, v_bf16_abs_b(pk));
                has_lo = 0;
            }
        }
        if (has_lo)
        {
            const bfloat128 pk = pack_two_f32_to_bf16_vec(pending_lo, (float64)0.0f);
            sw_vlm_write_bf16(slot, pk);
            am0 = v_bf16_max_b(am0, v_bf16_abs_b(pk));
        }

        const float64 amax = reduce_max_fast_nolut(bf16_vec_max_to_f32(v_bf16_max_b(am0, am1)), xmasks);
        const float64 sc_v = (amax + 1e-8f) * inv_fp8_max;
        const float64 inv_sc = rcp_pos_nolut(sc_v);
        int5 scc = {0, m, 0, 0, 0};
        v_f32_st_tnsr_partial(scc, out_sc, sc_v, 0, 0);

        // Pass 4: 4-way unrolled quantize from packed BF16 VLM and store FP8 E4M3.
        int5 fc0 = {0, m, 0, 0, 0}, fc1 = {64, m, 0, 0, 0}, fc2 = {128, m, 0, 0, 0}, fc3 = {192, m, 0, 0, 0};
        slot = 0;
        for (int d = 0; d < D256; d += 256, slot += 2)
        {
            const bfloat128 pk0 = sw_vlm_read_bf16(slot + 0);
            const bfloat128 pk1 = sw_vlm_read_bf16(slot + 1);
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
            const bfloat128 pk = sw_vlm_read_bf16(slot);
            fc0[0] = d; fc1[0] = d + 64;
            st_f32_as_f8(fc0, out_f8, v_f32_min_b(v_f32_max_b(unpack_bf16_lo(pk) * inv_sc, -fp8_max), fp8_max));
            st_f32_as_f8(fc1, out_f8, v_f32_min_b(v_f32_max_b(unpack_bf16_hi(pk) * inv_sc, -fp8_max), fp8_max));
        }
        if (D & 64)
        {
            fc0[0] = D128;
            st_f32_as_f8(fc0, out_f8,
                         v_f32_min_b(v_f32_max_b(unpack_bf16_lo(sw_vlm_read_bf16(slot)) * inv_sc, -fp8_max), fp8_max));
        }
    }
}
