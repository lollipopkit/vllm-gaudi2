// SPDX-License-Identifier: Apache-2.0
// Fused Q/R L2-clip (with q_scale/r_scale/tau) + direct conv_cache K/V short-conv (W=4, static/dynamic)
// + K L2-clip + V weightless RMSNorm for Inkling 2 Mini decode on Gaudi2 TPC.
//
// Wave 8 upgrades:
//   - Direct proj_q [Q_cols, B] and kv_src [KV_cols, B, KV_streams] reads (ZERO .split/.transpose copies!).
//   - Dedicated static (IS_DYNAMIC=0) and dynamic (IS_DYNAMIC=1) ELFs + single-call-site KV loop +
//     hardware rsqrt_f32 (single SFU LUT 0x180, ZERO other LUTs) -> ~4x smaller .text (~6 KB),
//     cutting cold I-cache fetch overhead across 48 decode visits!
//
// Tensors (TPC dim order, dim0 fastest):
//   proj_q     [Q_cols, B] bf16                 (PyTorch [B, Q_cols]; q at [0..H*128), r at [H*128..))
//   q_scale    [128, H] bf16                    (PyTorch [H, 128])
//   r_scale    [d_rel, H] bf16                  (PyTorch [H, d_rel])
//   tau        [B] f32                          (PyTorch [B], dummy when has_tau == 0)
//   kv_src     [KV_cols, B, KV_streams] bf16    (PyTorch [KV_streams, B, KV_cols], KV_streams in {1, 2})
//   conv_state [total_dim, 3, num_slots] bf16   (PyTorch [num_slots, 3, total_dim]; [K, V, attn, mlp])
//   load_slots [B] i32                          (PyTorch [B])
//   kvw_t      [128, H_kv, 2, 4] bf16           (PyTorch [4, 2, H_kv, 128])
//   kv_dyn     [4, H_kv, B, 2] bf16             (PyTorch [2, B, H_kv, 4], unused when IS_DYNAMIC == 0)
//   q_out      [128, H, B] bf16                 (PyTorch [B, H, 128])
//   r_out      [d_rel, H, B] bf16               (PyTorch [B, H, d_rel])
//   k_out      [128, H_kv, B] bf16              (PyTorch [B, H_kv, 128])
//   v_out      [128, H_kv, B] bf16              (PyTorch [B, H_kv, 128])
//   new_kv     [128, H_kv, 2, 3, B] bf16        (PyTorch [B, 3, 2, H_kv, 128])
//
// Index space: [H, B] (2D).
#pragma once

#include "inkling2_bf16_common.h"

static inline float64 l2_clip_factor_fast(float64 ssq, float threshold, float eps)
{
    // Hardware rsqrt_f32 (ONLY SFU LUT in this kernel -> zero LUT eviction) + compact 3-iter rcp.
    const float64 norm = ssq * rsqrt_f32(v_f32_max_b(ssq, 1e-30f));
    return v_f32_min_b(threshold * rcp_pos_nolut(norm + eps), 1.0f);
}

void main(tensor proj_q, tensor q_scale, tensor r_scale, tensor tau,
          tensor kv_src, tensor conv_state, tensor load_slots, tensor kvw_t, tensor kv_dyn,
          tensor q_out, tensor r_out, tensor k_out, tensor v_out, tensor new_kv,
          int kv_col_offset, int has_tau, float conv_scale,
          float qk_clip, float rel_clip, float clip_eps, float rms_eps)
{
    const int H = get_dim_size(q_out, 1);
    const int d_rel = get_dim_size(r_out, 0);
    const int H_kv = get_dim_size(k_out, 1);
    const int kv_size = H_kv << 7;
    const int kv_separate = (get_dim_size(kv_src, 2) == 2);
    const int split_kv = (H >= 2 * H_kv);
    const int r_base = H << 7;
    const int5 is_start = get_index_space_offset();
    const int5 is_end = get_index_space_size() + is_start;
    const float inv_128 = 0.0078125f;
    const uint64 lane_id = read_lane_id_4b_b();
    const bool64 valid_r = v_u32_cmp_less_b(lane_id, (unsigned int)d_rel);
    const tpc_xor_masks_t xmasks = make_xor_masks_nolut(lane_id);

    for (int b = is_start[1]; b < is_end[1]; b++)
    {
        int5 lsc = {b, 0, 0, 0, 0};
        const int slot = s_i32_ld_g((__global__ int*)gen_addr(lsc, load_slots));
        float tau_s = 1.0f;
        if (has_tau)
        {
            int5 tc = {b, 0, 0, 0, 0};
            tau_s = s_f32_ld_g((__global__ float*)gen_addr(tc, tau));
        }

        for (int h = is_start[0]; h < is_end[0]; h++)
        {
            int5 qin_c = {h << 7, b, 0, 0, 0};
            int5 qsc = {0, h, 0, 0, 0};
            const float64_pair_t qin = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(qin_c, proj_q));
            const float64_pair_t qs = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(qsc, q_scale));
            const float64 q0 = qin.v1 * qs.v1;
            const float64 q1 = qin.v2 * qs.v2;

            int5 rin_c = {r_base + h * d_rel, b, 0, 0, 0};
            int5 rsc = {0, h, 0, 0, 0};
            const float64 r_raw = v_f32_mov_vb(0.0f, 0, ld_bf16_as_f32(rin_c, proj_q), valid_r, 1);
            const float64 rv = r_raw * ld_bf16_as_f32(rsc, r_scale);

            const float64_pair_t ssq_qr = reduce_add_pair_nolut(v_f32_mac_b(q1, q1, q0 * q0), rv * rv, xmasks);
            const float64 q_fac = l2_clip_factor_fast(ssq_qr.v1, qk_clip, clip_eps);
            const float64 r_fac = l2_clip_factor_fast(ssq_qr.v2, rel_clip, clip_eps);

            float64_pair_t qo = {q0 * q_fac, q1 * q_fac};
            float64 ro = rv * r_fac;
            if (has_tau)
            {
                const float64_pair_t q_rnd = v_convert_bf16_to_f32_all_b(v_convert_f32_to_bf16_all_b(qo));
                qo.v1 = q_rnd.v1 * tau_s;
                qo.v2 = q_rnd.v2 * tau_s;
                ro = round_f32_via_bf16(ro) * tau_s;
            }
            int5 qoc = {0, h, b, 0, 0};
            v_bf16_st_tnsr(qoc, q_out, v_convert_f32_to_bf16_all_b(qo));
            st_f32_as_bf16(qoc, r_out, ro);

            // Single-call-site KV sconv + post-norm loop:
            //   - When H >= 2*H_kv (Inkling2: 24 vs 8): cores [0..7] do K (1 iter), cores [8..15] do V (1 iter).
            //   - When H < 2*H_kv: cores [0..H_kv-1] do K then V (2 iters).
            const int kv_begin = split_kv ? ((h < H_kv) ? 0 : 1) : 0;
            const int kv_end = split_kv ? ((h < H_kv) ? 1 : ((h < 2 * H_kv) ? 2 : 1)) : ((h < H_kv) ? 2 : 0);
            #pragma clang loop unroll(disable)
            for (int kv_idx = kv_begin; kv_idx < kv_end; kv_idx++)
            {
                const int kv_h = (split_kv && kv_idx == 1) ? (h - H_kv) : h;
                const int kv_col = kv_col_offset + (kv_separate ? 0 : (kv_idx * kv_size)) + (kv_h << 7);
                int5 xc = {kv_col, b, kv_separate ? kv_idx : 0, 0, 0};
                const bfloat128 x_raw = v_bf16_ld_tnsr_b(xc, kv_src);

                const int base_c = kv_idx * kv_size + (kv_h << 7);
                int5 sc0 = {base_c, 0, slot, 0, 0};
                int5 sc1 = {base_c, 1, slot, 0, 0};
                int5 sc2 = {base_c, 2, slot, 0, 0};
                const bfloat128 p0_raw = v_bf16_ld_tnsr_b(sc0, conv_state);
                const bfloat128 p1_raw = v_bf16_ld_tnsr_b(sc1, conv_state);
                const bfloat128 p2_raw = v_bf16_ld_tnsr_b(sc2, conv_state);

                int5 noc = {0, kv_h, kv_idx, 0, b};
                v_bf16_st_tnsr(noc, new_kv, p1_raw);
                noc[3] = 1;
                v_bf16_st_tnsr(noc, new_kv, p2_raw);
                noc[3] = 2;
                v_bf16_st_tnsr(noc, new_kv, x_raw);

                const float64_pair_t xp = v_convert_bf16_to_f32_all_b(x_raw);
                const float64_pair_t p0 = v_convert_bf16_to_f32_all_b(p0_raw);
                const float64_pair_t p1 = v_convert_bf16_to_f32_all_b(p1_raw);
                const float64_pair_t p2 = v_convert_bf16_to_f32_all_b(p2_raw);

                int5 wc0 = {0, kv_h, kv_idx, 0, 0};
                int5 wc1 = {0, kv_h, kv_idx, 1, 0};
                int5 wc2 = {0, kv_h, kv_idx, 2, 0};
                int5 wc3 = {0, kv_h, kv_idx, 3, 0};
                const float64_pair_t w0 = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(wc0, kvw_t));
                const float64_pair_t w1 = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(wc1, kvw_t));
                const float64_pair_t w2 = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(wc2, kvw_t));
                const float64_pair_t w3 = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(wc3, kvw_t));

                float64 acc0 = xp.v1;
                float64 acc1 = xp.v2;
#if !IS_DYNAMIC
                (void)kv_dyn;
                acc0 = v_f32_mac_b(p0.v1, w0.v1, acc0);
                acc1 = v_f32_mac_b(p0.v2, w0.v2, acc1);
                acc0 = v_f32_mac_b(p1.v1, w1.v1, acc0);
                acc1 = v_f32_mac_b(p1.v2, w1.v2, acc1);
                acc0 = v_f32_mac_b(p2.v1, w2.v1, acc0);
                acc1 = v_f32_mac_b(p2.v2, w2.v2, acc1);
                acc0 = v_f32_mac_b(xp.v1, w3.v1, acc0);
                acc1 = v_f32_mac_b(xp.v2, w3.v2, acc1);
#else
                int5 dc = {0, kv_h, b, kv_idx, 0};
                const float64 dv = ld_bf16_as_f32(dc, kv_dyn);
                const float64 d0 = v_broadcast_element_f32(dv, 0);
                const float64 d1 = v_broadcast_element_f32(dv, 1);
                const float64 d2 = v_broadcast_element_f32(dv, 2);
                const float64 d3 = v_broadcast_element_f32(dv, 3);

                acc0 = v_f32_mac_b(xp.v1, w0.v1 + d0, acc0);
                acc1 = v_f32_mac_b(xp.v2, w0.v2 + d0, acc1);
                acc0 = v_f32_mac_b(p2.v1, w1.v1 + d1, acc0);
                acc1 = v_f32_mac_b(p2.v2, w1.v2 + d1, acc1);
                acc0 = v_f32_mac_b(p1.v1, w2.v1 + d2, acc0);
                acc1 = v_f32_mac_b(p1.v2, w2.v2 + d2, acc1);
                acc0 = v_f32_mac_b(p0.v1, w3.v1 + d3, acc0);
                acc1 = v_f32_mac_b(p0.v2, w3.v2 + d3, acc1);
#endif
                acc0 *= conv_scale;
                acc1 *= conv_scale;
                const float64 ssq = reduce_add_fast_nolut(v_f32_mac_b(acc1, acc1, acc0 * acc0), xmasks);
                int5 kvc = {0, kv_h, b, 0, 0};
                if (kv_idx == 0)
                {
                    const float64 k_fac = l2_clip_factor_fast(ssq, qk_clip, clip_eps);
                    const float64_pair_t ko = {acc0 * k_fac, acc1 * k_fac};
                    v_bf16_st_tnsr(kvc, k_out, v_convert_f32_to_bf16_all_b(ko));
                }
                else
                {
                    const float64 v_rstd = rsqrt_f32(ssq * inv_128 + rms_eps);
                    const float64_pair_t vo = {acc0 * v_rstd, acc1 * v_rstd};
                    v_bf16_st_tnsr(kvc, v_out, v_convert_f32_to_bf16_all_b(vo));
                }
            }
        }
    }
}
