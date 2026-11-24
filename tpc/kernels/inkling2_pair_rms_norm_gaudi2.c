// SPDX-License-Identifier: Apache-2.0
// Fused pair_rms_norm for Inkling 2 Mini loop message controller on Gaudi2 TPC:
//   pair[s, t, :D]  = bf16(dest[s, t] * rsqrt(mean(dest[s, t]^2) + eps))
//   pair[s, t, D:]  = bf16(message[s, t] * rsqrt(mean(message[s, t]^2) + eps))
//
// Wave 7: interleaved reduce_add_pair_nolut computes both mean-squares in ONE 6-step butterfly,
// 2x unrolled for zero VRF spills, ZERO SFU LUTs, 0 VLM.
//
// Tensors (TPC dim order, dim0 fastest):
//   dest    [D, T, S] f32     (PyTorch [S, T, D])
//   message [D, T, S] f32     (PyTorch [S, T, D])
//   pair    [2*D, T, S] bf16  (PyTorch [S, T, 2*D])
//
// Index space: [T, S] (2D).

#include "inkling2_bf16_common.h"

void main(tensor dest, tensor message, tensor pair, float eps, float inv_D)
{
    const int D = get_dim_size(dest, 0);
    const int D128 = D & ~127;
    const int5 is_start = get_index_space_offset();
    const int5 is_end = get_index_space_size() + is_start;
    const tpc_xor_masks_t xmasks = make_xor_masks_nolut(read_lane_id_4b_b());

    for (int s = is_start[1]; s < is_end[1]; s++)
    for (int t = is_start[0]; t < is_end[0]; t++)
    {
        float64 sd0 = 0.0f, sd1 = 0.0f;
        float64 sm0 = 0.0f, sm1 = 0.0f;
        int5 c0 = {0, t, s, 0, 0}, c1 = {64, t, s, 0, 0};
        for (int d = 0; d < D128; d += 128)
        {
            const float64 d0 = v_f32_ld_tnsr_b(c0, dest);
            const float64 d1 = v_f32_ld_tnsr_b(c1, dest);
            const float64 m0 = v_f32_ld_tnsr_b(c0, message);
            const float64 m1 = v_f32_ld_tnsr_b(c1, message);
            c0[0] += 128; c1[0] += 128;
            sd0 = v_f32_mac_b(d0, d0, sd0);
            sd1 = v_f32_mac_b(d1, d1, sd1);
            sm0 = v_f32_mac_b(m0, m0, sm0);
            sm1 = v_f32_mac_b(m1, m1, sm1);
        }
        if (D & 64)
        {
            c0[0] = D128;
            const float64 d0 = v_f32_ld_tnsr_b(c0, dest);
            const float64 m0 = v_f32_ld_tnsr_b(c0, message);
            sd0 = v_f32_mac_b(d0, d0, sd0);
            sm0 = v_f32_mac_b(m0, m0, sm0);
        }
        const float64_pair_t sums = reduce_add_pair_nolut(sd0 + sd1, sm0 + sm1, xmasks);
        const float64 rstd_d = rsqrt_nolut(sums.v1 * inv_D + eps);
        const float64 rstd_m = rsqrt_nolut(sums.v2 * inv_D + eps);

        int5 ic0 = {0, t, s, 0, 0}, ic1 = {64, t, s, 0, 0};
        int5 od0 = {0, t, s, 0, 0}, od1 = {64, t, s, 0, 0};
        int5 om0 = {D, t, s, 0, 0}, om1 = {D + 64, t, s, 0, 0};
        for (int i = 0; i < D128; i += 128)
        {
            const float64 d0 = v_f32_ld_tnsr_b(ic0, dest) * rstd_d;
            const float64 d1 = v_f32_ld_tnsr_b(ic1, dest) * rstd_d;
            const float64 m0 = v_f32_ld_tnsr_b(ic0, message) * rstd_m;
            const float64 m1 = v_f32_ld_tnsr_b(ic1, message) * rstd_m;
            ic0[0] += 128; ic1[0] += 128;
            st_f32_as_bf16(od0, pair, d0);
            st_f32_as_bf16(od1, pair, d1);
            st_f32_as_bf16(om0, pair, m0);
            st_f32_as_bf16(om1, pair, m1);
            od0[0] += 128; od1[0] += 128;
            om0[0] += 128; om1[0] += 128;
        }
        if (D & 64)
        {
            ic0[0] = D128;
            od0[0] = D128;
            om0[0] = D + D128;
            st_f32_as_bf16(od0, pair, v_f32_ld_tnsr_b(ic0, dest) * rstd_d);
            st_f32_as_bf16(om0, pair, v_f32_ld_tnsr_b(ic0, message) * rstd_m);
        }
    }
}
