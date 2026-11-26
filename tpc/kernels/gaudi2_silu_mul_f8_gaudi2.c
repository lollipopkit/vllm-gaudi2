// SPDX-License-Identifier: Apache-2.0
// Generic zero-LUT fused SiLU-and-Mul + Per-Row FP8 E4M3 Dynamic Quantization for Gaudi2 TPC:
//   gate = gate_up[:, :N], up = gate_up[:, N:]
//   y    = bf16(silu(gate) * up)
//   sc   = (max(|y|) + 1e-8) / fp8_max
//   out  = f8_143(clamp(y / sc, -fp8_max, fp8_max))
//
// Zero SFU LUTs -> full 64 KiB VLM available. 4-vector (256-element) VPU-pipelined SiLU + single-pass
// BF16 packing (128 BF16 elements per 256-byte VLM slot, up to N = 28672 in 56 KiB) + 128-wide BF16
// amax tracking + 4-way unrolled quantize store eliminate VPU stalls and redundant BF16 conversions.
//
// Tensors (TPC dim order, dim0 fastest):
//   gate_up  [2*N, M] bf16
//   out_f8   [N, M]   f8_143
//   out_sc   [1, M]   f32
//
// Index space: [1, M] (dim0 = full row reduction, dim1 = row slice -> clean SRAM slicing along M).
#include "gaudi2_fp8_common.h"

#define MAX_SILU_PACKED_SLOTS 224  // 224 * 128 = 28672 BF16 activations (56 KiB VLM)
__local__ bfloat128 g_silu_bf16[MAX_SILU_PACKED_SLOTS];

void main(tensor gate_up, tensor out_f8, tensor out_sc, float fp8_max)
{
    const int N = get_dim_size(out_f8, 0);
    const int N256 = N & ~255;
    const int N128 = N & ~127;
    const int5 is_start = get_index_space_offset();
    const int5 is_end = get_index_space_size() + is_start;
    const tpc_xor_masks_t xmasks = make_xor_masks_nolut(read_lane_id_4b_b());
    const float inv_fp8_max = 1.0f / fp8_max;

    for (int m = is_start[1]; m < is_end[1]; m++)
    {
        bfloat128 am0 = 0, am1 = 0;
        int5 g0 = {0, m, 0, 0, 0}, g1 = {64, m, 0, 0, 0}, g2 = {128, m, 0, 0, 0}, g3 = {192, m, 0, 0, 0};
        int5 u0 = {N, m, 0, 0, 0}, u1 = {N + 64, m, 0, 0, 0}, u2 = {N + 128, m, 0, 0, 0}, u3 = {N + 192, m, 0, 0, 0};
        int slot = 0;

        for (int n = 0; n < N256; n += 256, slot += 2)
        {
            const float64 gv0 = ld_bf16_as_f32(g0, gate_up);
            const float64 gv1 = ld_bf16_as_f32(g1, gate_up);
            const float64 gv2 = ld_bf16_as_f32(g2, gate_up);
            const float64 gv3 = ld_bf16_as_f32(g3, gate_up);
            g0[0] += 256; g1[0] += 256; g2[0] += 256; g3[0] += 256;

            const float64_quad_t sv = silu_fast_bf16_x4(gv0, gv1, gv2, gv3);

            const float64 uv0 = ld_bf16_as_f32(u0, gate_up);
            const float64 uv1 = ld_bf16_as_f32(u1, gate_up);
            const float64 uv2 = ld_bf16_as_f32(u2, gate_up);
            const float64 uv3 = ld_bf16_as_f32(u3, gate_up);
            u0[0] += 256; u1[0] += 256; u2[0] += 256; u3[0] += 256;

            const bfloat128 pk0 = pack_two_f32_to_bf16_vec(sv.v0 * uv0, sv.v1 * uv1);
            const bfloat128 pk1 = pack_two_f32_to_bf16_vec(sv.v2 * uv2, sv.v3 * uv3);
            g_silu_bf16[slot + 0] = pk0;
            g_silu_bf16[slot + 1] = pk1;
            am0 = v_bf16_max_b(am0, v_bf16_abs_b(pk0));
            am1 = v_bf16_max_b(am1, v_bf16_abs_b(pk1));
        }

        // Single-copy tail handles any remainder N - N256 in {64, 128, 192}.
        float64 pending_lo = 0.0f;
        int has_lo = 0;
        for (int n = N256; n < N; n += 64)
        {
            g0[0] = n;
            u0[0] = N + n;
            const float64 y = silu_fast_bf16(ld_bf16_as_f32(g0, gate_up)) * ld_bf16_as_f32(u0, gate_up);
            if (!has_lo)
            {
                pending_lo = y;
                has_lo = 1;
            }
            else
            {
                const bfloat128 pk = pack_two_f32_to_bf16_vec(pending_lo, y);
                g_silu_bf16[slot++] = pk;
                am0 = v_bf16_max_b(am0, v_bf16_abs_b(pk));
                has_lo = 0;
            }
        }
        if (has_lo)
        {
            const bfloat128 pk = pack_two_f32_to_bf16_vec(pending_lo, (float64)0.0f);
            g_silu_bf16[slot] = pk;
            am0 = v_bf16_max_b(am0, v_bf16_abs_b(pk));
        }

        const float64 amax = reduce_max_fast_nolut(bf16_vec_max_to_f32(v_bf16_max_b(am0, am1)), xmasks);
        const float64 sc_v = (amax + 1e-8f) * inv_fp8_max;
        const float64 inv_sc = rcp_pos_nolut(sc_v);
        int5 scc = {0, m, 0, 0, 0};
        v_f32_st_tnsr_partial(scc, out_sc, sc_v, 0, 0);

        // Pass 2: 4-vector (256-elem) unrolled quantize & store from VLM.
        int5 fc0 = {0, m, 0, 0, 0}, fc1 = {64, m, 0, 0, 0}, fc2 = {128, m, 0, 0, 0}, fc3 = {192, m, 0, 0, 0};
        slot = 0;
        for (int n = 0; n < N256; n += 256, slot += 2)
        {
            const bfloat128 pk0 = g_silu_bf16[slot + 0];
            const bfloat128 pk1 = g_silu_bf16[slot + 1];
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
        for (int n = N256; n < N128; n += 128, slot++)
        {
            const bfloat128 pk = g_silu_bf16[slot];
            fc0[0] = n; fc1[0] = n + 64;
            st_f32_as_f8(fc0, out_f8, v_f32_min_b(v_f32_max_b(unpack_bf16_lo(pk) * inv_sc, -fp8_max), fp8_max));
            st_f32_as_f8(fc1, out_f8, v_f32_min_b(v_f32_max_b(unpack_bf16_hi(pk) * inv_sc, -fp8_max), fp8_max));
        }
        if (N & 64)
        {
            fc0[0] = N128;
            const float64 q0 = v_f32_min_b(
                v_f32_max_b(unpack_bf16_lo(g_silu_bf16[slot]) * inv_sc, -fp8_max), fp8_max);
            st_f32_as_f8(fc0, out_f8, q0);
        }
    }
}
