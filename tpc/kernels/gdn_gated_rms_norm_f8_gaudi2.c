// SPDX-License-Identifier: Apache-2.0
// Fused Qwen3.8 post-GDN Gated RMSNorm + Per-Row FP8 E4M3 Dynamic Quantization for Gaudi2 TPC:
// Consumes the FP32 [B, HV=48, D=128] output of gdn_decode directly (zero BF16 round-trip), applies
// per-head RMSNorm(x, eps) * weight * SiLU(z), computes per-row amax across all HV*D=6144 elements,
// and writes natural-order FP8 E4M3 + scale directly for out_proj fp8_gemm_v2.
//
// Zero SFU LUTs -> full 64 KiB VLM available (supports up to HV=128). Processes 2 value heads per
// iteration, evaluating silu_fast_bf16_x4(z0..z3) BEFORE loading x0..x3 to eliminate VRF spills,
// packing 128 BF16 elements per VLM slot with 128-wide BF16 amax tracking and 4-way quantize store.
//
// Tensors (TPC dim order, dim0 fastest):
//   x         [D=128, HV, B] f32
//   z         [HV*D, B]      bf16
//   weight    [D=128]        f32
//   out_f8    [HV*D, B]      f8_143
//   out_sc    [1, B]         f32
//
// Index space: [1, 1, B] (dim0 = head_dim, dim1 = HV reduction, dim2 = batch -> zero duplicated dims).
#include "gaudi2_fp8_common.h"

#define GDN_MAX_VHEADS 128
__local__ bfloat128 g_gated_bf16[GDN_MAX_VHEADS];

void main(tensor x, tensor z, tensor weight, tensor out_f8, tensor out_sc, float eps, float fp8_max)
{
    const int HV = get_dim_size(x, 1);
    const int HV2 = HV & ~1;
    const int5 is_start = get_index_space_offset();
    const int5 is_end = get_index_space_size() + is_start;
    const tpc_xor_masks_t xmasks = make_xor_masks_nolut(read_lane_id_4b_b());
    const float inv_128 = 1.0f / 128.0f;
    const float inv_fp8_max = 1.0f / fp8_max;

    int5 wc0 = {0, 0, 0, 0, 0}, wc1 = {64, 0, 0, 0, 0};
    const float64 w0 = v_f32_ld_tnsr_b(wc0, weight);
    const float64 w1 = v_f32_ld_tnsr_b(wc1, weight);

    for (int b = is_start[2]; b < is_end[2]; b++)
    {
        bfloat128 am0 = 0, am1 = 0;
        int5 zc0 = {0, b, 0, 0, 0}, zc1 = {64, b, 0, 0, 0};
        int5 zc2 = {128, b, 0, 0, 0}, zc3 = {192, b, 0, 0, 0};

        for (int h = 0; h < HV2; h += 2)
        {
            // Evaluate 4-way pipelined SiLU(z) first while x0..x3 are not live -> zero VRF spills.
            const float64 z0 = ld_bf16_as_f32(zc0, z);
            const float64 z1 = ld_bf16_as_f32(zc1, z);
            const float64 z2 = ld_bf16_as_f32(zc2, z);
            const float64 z3 = ld_bf16_as_f32(zc3, z);
            zc0[0] += 256; zc1[0] += 256; zc2[0] += 256; zc3[0] += 256;
            const float64_quad_t sz = silu_fast_bf16_x4(z0, z1, z2, z3);

            int5 xa0 = {0, h, b, 0, 0}, xa1 = {64, h, b, 0, 0};
            int5 xb0 = {0, h + 1, b, 0, 0}, xb1 = {64, h + 1, b, 0, 0};
            const float64 x0 = v_f32_ld_tnsr_b(xa0, x);
            const float64 x1 = v_f32_ld_tnsr_b(xa1, x);
            const float64 x2 = v_f32_ld_tnsr_b(xb0, x);
            const float64 x3 = v_f32_ld_tnsr_b(xb1, x);

            const float64 sq_a = v_f32_mac_b(x1, x1, x0 * x0);
            const float64 sq_b = v_f32_mac_b(x3, x3, x2 * x2);
            const float64_pair_t sums = reduce_add_pair_nolut(sq_a, sq_b, xmasks);
            const float64 rstd_a = rsqrt_nolut(sums.v1 * inv_128 + eps);
            const float64 rstd_b = rsqrt_nolut(sums.v2 * inv_128 + eps);

            const bfloat128 pk0 = pack_two_f32_to_bf16_vec(((x0 * rstd_a) * w0) * sz.v0,
                                                           ((x1 * rstd_a) * w1) * sz.v1);
            const bfloat128 pk1 = pack_two_f32_to_bf16_vec(((x2 * rstd_b) * w0) * sz.v2,
                                                           ((x3 * rstd_b) * w1) * sz.v3);
            g_gated_bf16[h + 0] = pk0;
            g_gated_bf16[h + 1] = pk1;
            am0 = v_bf16_max_b(am0, v_bf16_abs_b(pk0));
            am1 = v_bf16_max_b(am1, v_bf16_abs_b(pk1));
        }
        if (HV & 1)
        {
            const int h = HV2;
            const float64_pair_t sz = silu_fast_bf16_x2(ld_bf16_as_f32(zc0, z), ld_bf16_as_f32(zc1, z));
            int5 xa0 = {0, h, b, 0, 0}, xa1 = {64, h, b, 0, 0};
            const float64 x0 = v_f32_ld_tnsr_b(xa0, x);
            const float64 x1 = v_f32_ld_tnsr_b(xa1, x);
            const float64 rstd = rsqrt_nolut(reduce_add_fast_nolut(v_f32_mac_b(x1, x1, x0 * x0), xmasks) * inv_128 + eps);
            const bfloat128 pk = pack_two_f32_to_bf16_vec(((x0 * rstd) * w0) * sz.v1,
                                                          ((x1 * rstd) * w1) * sz.v2);
            g_gated_bf16[h] = pk;
            am0 = v_bf16_max_b(am0, v_bf16_abs_b(pk));
        }

        const float64 amax = reduce_max_fast_nolut(bf16_vec_max_to_f32(v_bf16_max_b(am0, am1)), xmasks);
        const float64 sc_v = (amax + 1e-8f) * inv_fp8_max;
        const float64 inv_sc = rcp_pos_nolut(sc_v);
        int5 scc = {0, b, 0, 0, 0};
        v_f32_st_tnsr_partial(scc, out_sc, sc_v, 0, 0);

        int5 fc0 = {0, b, 0, 0, 0}, fc1 = {64, b, 0, 0, 0}, fc2 = {128, b, 0, 0, 0}, fc3 = {192, b, 0, 0, 0};
        for (int h = 0; h < HV2; h += 2)
        {
            const bfloat128 pk0 = g_gated_bf16[h + 0];
            const bfloat128 pk1 = g_gated_bf16[h + 1];
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
        if (HV & 1)
        {
            const bfloat128 pk = g_gated_bf16[HV2];
            st_f32_as_f8(fc0, out_f8, v_f32_min_b(v_f32_max_b(unpack_bf16_lo(pk) * inv_sc, -fp8_max), fp8_max));
            st_f32_as_f8(fc1, out_f8, v_f32_min_b(v_f32_max_b(unpack_bf16_hi(pk) * inv_sc, -fp8_max), fp8_max));
        }
    }
}
