// SPDX-License-Identifier: Apache-2.0
// Expert-Grouped Zero-Copy Indirect FP8 MoE w13 GEMV + Softcapped SiTU + Compact FP8 Quant
// for Inkling 2 Mini on Gaudi2 TPC (B >= 2).
//
// Reads each unique expert's w13 weight tile ONCE from HBM across up to 4 tokens sharing that
// expert with ZERO VRF spills across all four specialized inner K-loops (cnt = 1, 2, 3, 4) and
// cost-weighted core partitioning (via precomputed group_meta row 6) so all 24 TPC cores finish in lockstep.
//
// Tensors (TPC dim order, dim0 fastest):
//   x_i32        [K/4, B] i32
//   x_scale      [B] f32
//   w13_tiled    [256, K, nBlks, E] f8_143
//   s13_packed   [2*I, E] f32
//   topk_weights [K_top, B] f32
//   group_meta   [128, 8] i32                     (PyTorch [8, 128]; rows: e, cnt, r0..3, w13_bounds, w2_bounds)
//   group_count  [2] i32                          (PyTorch [2]; [G, total_w])
//   act_f8       [128, nBlks * K_top * B] f8_143
//   act_scale    [nBlks * K_top * B] f32
//
// Index space: [num_cores] (1D; cost-weighted dynamic split of [0, G * nBlks) across cores).

#include "inkling2_bf16_common.h"

#define MAX_GRP_K4 640
#ifndef W_LD_SW
#define W_LD_SW SW_L0CD
#endif

static __local__ float64 g_res[4][4];

static inline void epilogue_store_one(float64 g_even, float64 g_odd, float64 u_even, float64 u_odd,
                                      float inv_gcap, float inv_ucap, float scale_gamma,
                                      float fp8_max, float inv_fp8_max, tpc_xor_masks_t xmasks,
                                      int out_id, tensor act_f8, tensor act_scale)
{
    const float64 act_even = softcapped_situ_vec(g_even, u_even, inv_gcap, inv_ucap, scale_gamma);
    const float64 act_odd = softcapped_situ_vec(g_odd, u_odd, inv_gcap, inv_ucap, scale_gamma);

    const float64 abs_max_v = v_f32_max_b(v_f32_abs_b(act_even), v_f32_abs_b(act_odd));
    const float64 red_max = reduce_max_fast_nolut(abs_max_v, xmasks);
    const float64 a_sc_v = v_f32_max_b(red_max * inv_fp8_max, 1e-12f);
    const float64 inv_a_sc_v = rcp_pos_nolut(a_sc_v);

    const float64 q_even = v_f32_min_b(v_f32_max_b(act_even * inv_a_sc_v, -fp8_max), fp8_max);
    const float64 q_odd = v_f32_min_b(v_f32_max_b(act_odd * inv_a_sc_v, -fp8_max), fp8_max);

    int5 oc = {0, out_id, 0, 0, 0};
    st_f32_as_f8(oc, act_f8, q_even);
    oc[0] = 64;
    st_f32_as_f8(oc, act_f8, q_odd);

    int5 asc = {out_id, 0, 0, 0, 0};
    v_f32_st_tnsr_partial(asc, act_scale, a_sc_v, 0, 0);
}

void main(tensor x_i32, tensor x_scale, tensor w13_tiled, tensor s13_packed,
          tensor topk_weights, tensor group_meta, tensor group_count,
          tensor act_f8, tensor act_scale,
          float gate_cap, float up_cap, float fp8_max, int num_cores)
{
    (void)group_count;
    (void)num_cores;
    const int K4 = get_dim_size(x_i32, 0);
    const int nBlks = get_dim_size(w13_tiled, 2);
    const int topK = get_dim_size(topk_weights, 0);
    const int5 is_start = get_index_space_offset();
    const int5 is_end = get_index_space_size() + is_start;

    // O(1) startup: read precomputed cost-weighted tile interval [w0_idx, w1_idx) from row 6!
    int5 b0c = {is_start[0], 6, 0, 0, 0};
    int5 b1c = {is_end[0], 6, 0, 0, 0};
    const int w0_idx = s_i32_ld_g((__global__ int*)gen_addr(b0c, group_meta));
    const int w1_idx = s_i32_ld_g((__global__ int*)gen_addr(b1c, group_meta));
    if (w0_idx >= w1_idx)
        return;

    const float inv_gcap = 1.0f / gate_cap;
    const float inv_ucap = 1.0f / up_cap;
    const float gcap_ucap = gate_cap * up_cap;
    const float inv_fp8_max = 1.0f / fp8_max;
    const tpc_xor_masks_t xmasks = make_xor_masks_nolut(read_lane_id_4b_b());

    unsigned int x_slm[MAX_GRP_K4][4];
    int cached_g = -1;
    int g_e = 0, g_cnt = 0;
    int g_rs[4];
    float g_xsc[4], g_gamma[4];

    int blk = w0_idx % nBlks;
    int g = w0_idx / nBlks;

    for (int wid = w0_idx; wid < w1_idx; wid++)
    {
        if (g != cached_g)
        {
            int5 mc = {g, 0, 0, 0, 0};
            g_e = s_i32_ld_g((__global__ int*)gen_addr(mc, group_meta));
            mc[1] = 1;
            g_cnt = s_i32_ld_g((__global__ int*)gen_addr(mc, group_meta));
            for (int j = 0; j < g_cnt; j++)
            {
                mc[1] = 2 + j;
                const int rs = s_i32_ld_g((__global__ int*)gen_addr(mc, group_meta));
                const int b = rs / topK;
                const int slot = rs % topK;
                g_rs[j] = rs;
                int5 sc = {b, 0, 0, 0, 0};
                g_xsc[j] = s_f32_ld_g((__global__ float*)gen_addr(sc, x_scale));
                int5 wc = {slot, b, 0, 0, 0};
                g_gamma[j] = s_f32_ld_g((__global__ float*)gen_addr(wc, topk_weights));

                int5 pf = {0, b, 0, 0, 0};
                for (int k4 = 0; k4 < K4; k4 += 16)
                {
                    pf[0] = k4;
                    prefetch(gen_addr(pf, x_i32));
                }
                const int K4_4 = K4 & ~3;
                int5 xc0 = {0, b, 0, 0, 0}, xc1 = {1, b, 0, 0, 0};
                int5 xc2 = {2, b, 0, 0, 0}, xc3 = {3, b, 0, 0, 0};
                for (int k4 = 0; k4 < K4_4; k4 += 4)
                {
                    const unsigned int v0 = (unsigned int)s_i32_ld_g((__global__ int*)gen_addr(xc0, x_i32));
                    const unsigned int v1 = (unsigned int)s_i32_ld_g((__global__ int*)gen_addr(xc1, x_i32));
                    const unsigned int v2 = (unsigned int)s_i32_ld_g((__global__ int*)gen_addr(xc2, x_i32));
                    const unsigned int v3 = (unsigned int)s_i32_ld_g((__global__ int*)gen_addr(xc3, x_i32));
                    xc0[0] += 4; xc1[0] += 4; xc2[0] += 4; xc3[0] += 4;
                    x_slm[k4 + 0][j] = v0; x_slm[k4 + 1][j] = v1;
                    x_slm[k4 + 2][j] = v2; x_slm[k4 + 3][j] = v3;
                }
                for (int k4 = K4_4; k4 < K4; k4++)
                {
                    xc0[0] = k4;
                    x_slm[k4][j] = (unsigned int)s_i32_ld_g((__global__ int*)gen_addr(xc0, x_i32));
                }
            }
            cached_g = g;
        }

        const int e = g_e;
        const int n_off = blk << 8;
        int5 wc0 = {0, 0, blk, e, 0}, wc1 = {0, 1, blk, e, 0};
        int5 wc2 = {0, 2, blk, e, 0}, wc3 = {0, 3, blk, e, 0};

        int5 s13c = {n_off, e, 0, 0, 0};
        const float64 sge_base = v_f32_ld_tnsr_b(s13c, s13_packed);
        s13c[0] = n_off + 64;
        const float64 sgo_base = v_f32_ld_tnsr_b(s13c, s13_packed);
        s13c[0] = n_off + 128;
        const float64 sue_base = v_f32_ld_tnsr_b(s13c, s13_packed);
        s13c[0] = n_off + 192;
        const float64 suo_base = v_f32_ld_tnsr_b(s13c, s13_packed);

        if (g_cnt == 1)
        {
            float256 acc0 = {0.0f, 0.0f, 0.0f, 0.0f};
            float256 acc1 = {0.0f, 0.0f, 0.0f, 0.0f};
            float256 acc2 = {0.0f, 0.0f, 0.0f, 0.0f};
            float256 acc3 = {0.0f, 0.0f, 0.0f, 0.0f};

            const int K4_even = K4 & ~1;
            if (K4_even >= 2)
            {
                unsigned int xwa = x_slm[0][0];
                minifloat256 wa0 = v_f8_ld_tnsr_b(wc0, w13_tiled, W_LD_SW, 0, 1, 0);
                minifloat256 wa1 = v_f8_ld_tnsr_b(wc1, w13_tiled, W_LD_SW, 0, 1, 0);
                minifloat256 wa2 = v_f8_ld_tnsr_b(wc2, w13_tiled, W_LD_SW, 0, 1, 0);
                minifloat256 wa3 = v_f8_ld_tnsr_b(wc3, w13_tiled, W_LD_SW, 0, 1, 0);

                unsigned int xwb = x_slm[1][0];
                wc0[1] += 4; wc1[1] += 4; wc2[1] += 4; wc3[1] += 4;
                minifloat256 wb0 = v_f8_ld_tnsr_b(wc0, w13_tiled, W_LD_SW, 0, 1, 0);
                minifloat256 wb1 = v_f8_ld_tnsr_b(wc1, w13_tiled, W_LD_SW, 0, 1, 0);
                minifloat256 wb2 = v_f8_ld_tnsr_b(wc2, w13_tiled, W_LD_SW, 0, 1, 0);
                minifloat256 wb3 = v_f8_ld_tnsr_b(wc3, w13_tiled, W_LD_SW, 0, 1, 0);

                for (int k4 = 2; k4 < K4_even; k4 += 2)
                {
                    wc0[1] += 4; wc1[1] += 4; wc2[1] += 4; wc3[1] += 4;
                    acc0 = v_f8_mac_acc32_b(wa0, bits_to_f8(xwa), acc0);
                    wa0 = v_f8_ld_tnsr_b(wc0, w13_tiled, W_LD_SW, 0, 1, 0);
                    acc1 = v_f8_mac_acc32_b(wa1, bits_to_f8(xwa >> 8), acc1);
                    wa1 = v_f8_ld_tnsr_b(wc1, w13_tiled, W_LD_SW, 0, 1, 0);
                    acc2 = v_f8_mac_acc32_b(wa2, bits_to_f8(xwa >> 16), acc2);
                    wa2 = v_f8_ld_tnsr_b(wc2, w13_tiled, W_LD_SW, 0, 1, 0);
                    acc3 = v_f8_mac_acc32_b(wa3, bits_to_f8(xwa >> 24), acc3);
                    wa3 = v_f8_ld_tnsr_b(wc3, w13_tiled, W_LD_SW, 0, 1, 0);
                    xwa = x_slm[k4][0];

                    wc0[1] += 4; wc1[1] += 4; wc2[1] += 4; wc3[1] += 4;
                    acc0 = v_f8_mac_acc32_b(wb0, bits_to_f8(xwb), acc0);
                    wb0 = v_f8_ld_tnsr_b(wc0, w13_tiled, W_LD_SW, 0, 1, 0);
                    acc1 = v_f8_mac_acc32_b(wb1, bits_to_f8(xwb >> 8), acc1);
                    wb1 = v_f8_ld_tnsr_b(wc1, w13_tiled, W_LD_SW, 0, 1, 0);
                    acc2 = v_f8_mac_acc32_b(wb2, bits_to_f8(xwb >> 16), acc2);
                    wb2 = v_f8_ld_tnsr_b(wc2, w13_tiled, W_LD_SW, 0, 1, 0);
                    acc3 = v_f8_mac_acc32_b(wb3, bits_to_f8(xwb >> 24), acc3);
                    wb3 = v_f8_ld_tnsr_b(wc3, w13_tiled, W_LD_SW, 0, 1, 0);
                    xwb = x_slm[k4 + 1][0];
                }
                acc0 = v_f8_mac_acc32_b(wa0, bits_to_f8(xwa), acc0);
                acc1 = v_f8_mac_acc32_b(wa1, bits_to_f8(xwa >> 8), acc1);
                acc2 = v_f8_mac_acc32_b(wa2, bits_to_f8(xwa >> 16), acc2);
                acc3 = v_f8_mac_acc32_b(wa3, bits_to_f8(xwa >> 24), acc3);
                acc0 = v_f8_mac_acc32_b(wb0, bits_to_f8(xwb), acc0);
                acc1 = v_f8_mac_acc32_b(wb1, bits_to_f8(xwb >> 8), acc1);
                acc2 = v_f8_mac_acc32_b(wb2, bits_to_f8(xwb >> 16), acc2);
                acc3 = v_f8_mac_acc32_b(wb3, bits_to_f8(xwb >> 24), acc3);
                wc0[1] += 4; wc1[1] += 4; wc2[1] += 4; wc3[1] += 4;
            }
            if (K4 & 1)
            {
                const unsigned int xw = x_slm[K4_even][0];
                acc0 = v_f8_mac_acc32_b(v_f8_ld_tnsr_b(wc0, w13_tiled, W_LD_SW, 0, 1, 0), bits_to_f8(xw), acc0);
                acc1 = v_f8_mac_acc32_b(v_f8_ld_tnsr_b(wc1, w13_tiled, W_LD_SW, 0, 1, 0), bits_to_f8(xw >> 8), acc1);
                acc2 = v_f8_mac_acc32_b(v_f8_ld_tnsr_b(wc2, w13_tiled, W_LD_SW, 0, 1, 0), bits_to_f8(xw >> 16), acc2);
                acc3 = v_f8_mac_acc32_b(v_f8_ld_tnsr_b(wc3, w13_tiled, W_LD_SW, 0, 1, 0), bits_to_f8(xw >> 24), acc3);
            }
            g_res[0][0] = (acc0.v1 + acc1.v1) + (acc2.v1 + acc3.v1);
            g_res[0][1] = (acc0.v2 + acc1.v2) + (acc2.v2 + acc3.v2);
            g_res[0][2] = (acc0.v3 + acc1.v3) + (acc2.v3 + acc3.v3);
            g_res[0][3] = (acc0.v4 + acc1.v4) + (acc2.v4 + acc3.v4);
        }
        else if (g_cnt == 2)
        {
            float256 a0a = {0.0f, 0.0f, 0.0f, 0.0f}, a0b = {0.0f, 0.0f, 0.0f, 0.0f};
            float256 a1a = {0.0f, 0.0f, 0.0f, 0.0f}, a1b = {0.0f, 0.0f, 0.0f, 0.0f};

            unsigned int x0 = x_slm[0][0], x1 = x_slm[0][1];
            minifloat256 w0 = v_f8_ld_tnsr_b(wc0, w13_tiled, W_LD_SW, 0, 1, 0);
            minifloat256 w1 = v_f8_ld_tnsr_b(wc1, w13_tiled, W_LD_SW, 0, 1, 0);
            minifloat256 w2 = v_f8_ld_tnsr_b(wc2, w13_tiled, W_LD_SW, 0, 1, 0);
            minifloat256 w3 = v_f8_ld_tnsr_b(wc3, w13_tiled, W_LD_SW, 0, 1, 0);

            for (int k4 = 0; k4 < K4 - 1; k4++)
            {
                const minifloat b00 = bits_to_f8(x0), b01 = bits_to_f8(x1);
                const minifloat b10 = bits_to_f8(x0 >> 8), b11 = bits_to_f8(x1 >> 8);
                const minifloat b20 = bits_to_f8(x0 >> 16), b21 = bits_to_f8(x1 >> 16);
                const minifloat b30 = bits_to_f8(x0 >> 24), b31 = bits_to_f8(x1 >> 24);
                x0 = x_slm[k4 + 1][0];
                x1 = x_slm[k4 + 1][1];
                wc0[1] += 4; wc1[1] += 4; wc2[1] += 4; wc3[1] += 4;

                a0a = v_f8_mac_acc32_b(w0, b00, a0a); a1a = v_f8_mac_acc32_b(w0, b01, a1a);
                w0 = v_f8_ld_tnsr_b(wc0, w13_tiled, W_LD_SW, 0, 1, 0);
                a0b = v_f8_mac_acc32_b(w1, b10, a0b); a1b = v_f8_mac_acc32_b(w1, b11, a1b);
                w1 = v_f8_ld_tnsr_b(wc1, w13_tiled, W_LD_SW, 0, 1, 0);
                a0a = v_f8_mac_acc32_b(w2, b20, a0a); a1a = v_f8_mac_acc32_b(w2, b21, a1a);
                w2 = v_f8_ld_tnsr_b(wc2, w13_tiled, W_LD_SW, 0, 1, 0);
                a0b = v_f8_mac_acc32_b(w3, b30, a0b); a1b = v_f8_mac_acc32_b(w3, b31, a1b);
                w3 = v_f8_ld_tnsr_b(wc3, w13_tiled, W_LD_SW, 0, 1, 0);
            }
            {
                a0a = v_f8_mac_acc32_b(w0, bits_to_f8(x0), a0a); a1a = v_f8_mac_acc32_b(w0, bits_to_f8(x1), a1a);
                a0b = v_f8_mac_acc32_b(w1, bits_to_f8(x0 >> 8), a0b); a1b = v_f8_mac_acc32_b(w1, bits_to_f8(x1 >> 8), a1b);
                a0a = v_f8_mac_acc32_b(w2, bits_to_f8(x0 >> 16), a0a); a1a = v_f8_mac_acc32_b(w2, bits_to_f8(x1 >> 16), a1a);
                a0b = v_f8_mac_acc32_b(w3, bits_to_f8(x0 >> 24), a0b); a1b = v_f8_mac_acc32_b(w3, bits_to_f8(x1 >> 24), a1b);
            }
            g_res[0][0] = a0a.v1 + a0b.v1; g_res[0][1] = a0a.v2 + a0b.v2;
            g_res[0][2] = a0a.v3 + a0b.v3; g_res[0][3] = a0a.v4 + a0b.v4;
            g_res[1][0] = a1a.v1 + a1b.v1; g_res[1][1] = a1a.v2 + a1b.v2;
            g_res[1][2] = a1a.v3 + a1b.v3; g_res[1][3] = a1a.v4 + a1b.v4;
        }
        else if (g_cnt == 3)
        {
            // Dedicated 3-token loop: 12 MACs/k4 (~25 cyc/k4 vs 34 cyc/k4 for 4-token!).
            float256 a0 = {0.0f, 0.0f, 0.0f, 0.0f};
            float256 a1 = {0.0f, 0.0f, 0.0f, 0.0f};
            float256 a2 = {0.0f, 0.0f, 0.0f, 0.0f};

            unsigned int x0 = x_slm[0][0], x1 = x_slm[0][1], x2 = x_slm[0][2];
            minifloat256 w0 = v_f8_ld_tnsr_b(wc0, w13_tiled, W_LD_SW, 0, 1, 0);
            minifloat256 w1 = v_f8_ld_tnsr_b(wc1, w13_tiled, W_LD_SW, 0, 1, 0);
            minifloat256 w2 = v_f8_ld_tnsr_b(wc2, w13_tiled, W_LD_SW, 0, 1, 0);
            minifloat256 w3 = v_f8_ld_tnsr_b(wc3, w13_tiled, W_LD_SW, 0, 1, 0);

            for (int k4 = 0; k4 < K4 - 1; k4++)
            {
                const minifloat b00 = bits_to_f8(x0), b01 = bits_to_f8(x1), b02 = bits_to_f8(x2);
                const minifloat b10 = bits_to_f8(x0 >> 8), b11 = bits_to_f8(x1 >> 8), b12 = bits_to_f8(x2 >> 8);
                wc0[1] += 4; wc1[1] += 4; wc2[1] += 4; wc3[1] += 4;

                a0 = v_f8_mac_acc32_b(w0, b00, a0); a1 = v_f8_mac_acc32_b(w0, b01, a1); a2 = v_f8_mac_acc32_b(w0, b02, a2);
                w0 = v_f8_ld_tnsr_b(wc0, w13_tiled, W_LD_SW, 0, 1, 0);

                const minifloat b20 = bits_to_f8(x0 >> 16), b21 = bits_to_f8(x1 >> 16), b22 = bits_to_f8(x2 >> 16);
                a0 = v_f8_mac_acc32_b(w1, b10, a0); a1 = v_f8_mac_acc32_b(w1, b11, a1); a2 = v_f8_mac_acc32_b(w1, b12, a2);
                w1 = v_f8_ld_tnsr_b(wc1, w13_tiled, W_LD_SW, 0, 1, 0);

                const minifloat b30 = bits_to_f8(x0 >> 24), b31 = bits_to_f8(x1 >> 24), b32 = bits_to_f8(x2 >> 24);
                a0 = v_f8_mac_acc32_b(w2, b20, a0); a1 = v_f8_mac_acc32_b(w2, b21, a1); a2 = v_f8_mac_acc32_b(w2, b22, a2);
                w2 = v_f8_ld_tnsr_b(wc2, w13_tiled, W_LD_SW, 0, 1, 0);

                x0 = x_slm[k4 + 1][0]; x1 = x_slm[k4 + 1][1]; x2 = x_slm[k4 + 1][2];
                a0 = v_f8_mac_acc32_b(w3, b30, a0); a1 = v_f8_mac_acc32_b(w3, b31, a1); a2 = v_f8_mac_acc32_b(w3, b32, a2);
                w3 = v_f8_ld_tnsr_b(wc3, w13_tiled, W_LD_SW, 0, 1, 0);
            }
            {
                a0 = v_f8_mac_acc32_b(w0, bits_to_f8(x0), a0); a1 = v_f8_mac_acc32_b(w0, bits_to_f8(x1), a1); a2 = v_f8_mac_acc32_b(w0, bits_to_f8(x2), a2);
                a0 = v_f8_mac_acc32_b(w1, bits_to_f8(x0 >> 8), a0); a1 = v_f8_mac_acc32_b(w1, bits_to_f8(x1 >> 8), a1); a2 = v_f8_mac_acc32_b(w1, bits_to_f8(x2 >> 8), a2);
                a0 = v_f8_mac_acc32_b(w2, bits_to_f8(x0 >> 16), a0); a1 = v_f8_mac_acc32_b(w2, bits_to_f8(x1 >> 16), a1); a2 = v_f8_mac_acc32_b(w2, bits_to_f8(x2 >> 16), a2);
                a0 = v_f8_mac_acc32_b(w3, bits_to_f8(x0 >> 24), a0); a1 = v_f8_mac_acc32_b(w3, bits_to_f8(x1 >> 24), a1); a2 = v_f8_mac_acc32_b(w3, bits_to_f8(x2 >> 24), a2);
            }
            g_res[0][0] = a0.v1; g_res[0][1] = a0.v2; g_res[0][2] = a0.v3; g_res[0][3] = a0.v4;
            g_res[1][0] = a1.v1; g_res[1][1] = a1.v2; g_res[1][2] = a1.v3; g_res[1][3] = a1.v4;
            g_res[2][0] = a2.v1; g_res[2][1] = a2.v2; g_res[2][2] = a2.v3; g_res[2][3] = a2.v4;
        }
        else
        {
            float256 a0 = {0.0f, 0.0f, 0.0f, 0.0f}, a1 = {0.0f, 0.0f, 0.0f, 0.0f};
            float256 a2 = {0.0f, 0.0f, 0.0f, 0.0f}, a3 = {0.0f, 0.0f, 0.0f, 0.0f};

            unsigned int x0 = x_slm[0][0], x1 = x_slm[0][1], x2 = x_slm[0][2], x3 = x_slm[0][3];
            minifloat256 w0 = v_f8_ld_tnsr_b(wc0, w13_tiled, W_LD_SW, 0, 1, 0);
            minifloat256 w1 = v_f8_ld_tnsr_b(wc1, w13_tiled, W_LD_SW, 0, 1, 0);
            minifloat256 w2 = v_f8_ld_tnsr_b(wc2, w13_tiled, W_LD_SW, 0, 1, 0);
            minifloat256 w3 = v_f8_ld_tnsr_b(wc3, w13_tiled, W_LD_SW, 0, 1, 0);

            for (int k4 = 0; k4 < K4 - 1; k4++)
            {
                const minifloat b00 = bits_to_f8(x0), b01 = bits_to_f8(x1), b02 = bits_to_f8(x2), b03 = bits_to_f8(x3);
                const minifloat b10 = bits_to_f8(x0 >> 8), b11 = bits_to_f8(x1 >> 8), b12 = bits_to_f8(x2 >> 8), b13 = bits_to_f8(x3 >> 8);
                wc0[1] += 4; wc1[1] += 4; wc2[1] += 4; wc3[1] += 4;

                a0 = v_f8_mac_acc32_b(w0, b00, a0); a1 = v_f8_mac_acc32_b(w0, b01, a1);
                a2 = v_f8_mac_acc32_b(w0, b02, a2); a3 = v_f8_mac_acc32_b(w0, b03, a3);
                w0 = v_f8_ld_tnsr_b(wc0, w13_tiled, W_LD_SW, 0, 1, 0);

                const minifloat b20 = bits_to_f8(x0 >> 16), b21 = bits_to_f8(x1 >> 16), b22 = bits_to_f8(x2 >> 16), b23 = bits_to_f8(x3 >> 16);
                a0 = v_f8_mac_acc32_b(w1, b10, a0); a1 = v_f8_mac_acc32_b(w1, b11, a1);
                a2 = v_f8_mac_acc32_b(w1, b12, a2); a3 = v_f8_mac_acc32_b(w1, b13, a3);
                w1 = v_f8_ld_tnsr_b(wc1, w13_tiled, W_LD_SW, 0, 1, 0);

                const minifloat b30 = bits_to_f8(x0 >> 24), b31 = bits_to_f8(x1 >> 24), b32 = bits_to_f8(x2 >> 24), b33 = bits_to_f8(x3 >> 24);
                a0 = v_f8_mac_acc32_b(w2, b20, a0); a1 = v_f8_mac_acc32_b(w2, b21, a1);
                a2 = v_f8_mac_acc32_b(w2, b22, a2); a3 = v_f8_mac_acc32_b(w2, b23, a3);
                w2 = v_f8_ld_tnsr_b(wc2, w13_tiled, W_LD_SW, 0, 1, 0);

                x0 = x_slm[k4 + 1][0]; x1 = x_slm[k4 + 1][1]; x2 = x_slm[k4 + 1][2]; x3 = x_slm[k4 + 1][3];
                a0 = v_f8_mac_acc32_b(w3, b30, a0); a1 = v_f8_mac_acc32_b(w3, b31, a1);
                a2 = v_f8_mac_acc32_b(w3, b32, a2); a3 = v_f8_mac_acc32_b(w3, b33, a3);
                w3 = v_f8_ld_tnsr_b(wc3, w13_tiled, W_LD_SW, 0, 1, 0);
            }
            {
                a0 = v_f8_mac_acc32_b(w0, bits_to_f8(x0), a0); a1 = v_f8_mac_acc32_b(w0, bits_to_f8(x1), a1);
                a2 = v_f8_mac_acc32_b(w0, bits_to_f8(x2), a2); a3 = v_f8_mac_acc32_b(w0, bits_to_f8(x3), a3);
                a0 = v_f8_mac_acc32_b(w1, bits_to_f8(x0 >> 8), a0); a1 = v_f8_mac_acc32_b(w1, bits_to_f8(x1 >> 8), a1);
                a2 = v_f8_mac_acc32_b(w1, bits_to_f8(x2 >> 8), a2); a3 = v_f8_mac_acc32_b(w1, bits_to_f8(x3 >> 8), a3);
                a0 = v_f8_mac_acc32_b(w2, bits_to_f8(x0 >> 16), a0); a1 = v_f8_mac_acc32_b(w2, bits_to_f8(x1 >> 16), a1);
                a2 = v_f8_mac_acc32_b(w2, bits_to_f8(x2 >> 16), a2); a3 = v_f8_mac_acc32_b(w2, bits_to_f8(x3 >> 16), a3);
                a0 = v_f8_mac_acc32_b(w3, bits_to_f8(x0 >> 24), a0); a1 = v_f8_mac_acc32_b(w3, bits_to_f8(x1 >> 24), a1);
                a2 = v_f8_mac_acc32_b(w3, bits_to_f8(x2 >> 24), a2); a3 = v_f8_mac_acc32_b(w3, bits_to_f8(x3 >> 24), a3);
            }
            g_res[0][0] = a0.v1; g_res[0][1] = a0.v2; g_res[0][2] = a0.v3; g_res[0][3] = a0.v4;
            g_res[1][0] = a1.v1; g_res[1][1] = a1.v2; g_res[1][2] = a1.v3; g_res[1][3] = a1.v4;
            g_res[2][0] = a2.v1; g_res[2][1] = a2.v2; g_res[2][2] = a2.v3; g_res[2][3] = a2.v4;
            g_res[3][0] = a3.v1; g_res[3][1] = a3.v2; g_res[3][2] = a3.v3; g_res[3][3] = a3.v4;
        }

        #pragma clang loop unroll(disable)
        for (int j = 0; j < g_cnt; j++)
        {
            const float xsc = g_xsc[j];
            epilogue_store_one(g_res[j][0] * (sge_base * xsc), g_res[j][1] * (sgo_base * xsc),
                               g_res[j][2] * (sue_base * xsc), g_res[j][3] * (suo_base * xsc),
                               inv_gcap, inv_ucap, gcap_ucap * g_gamma[j], fp8_max, inv_fp8_max,
                               xmasks, g_rs[j] * nBlks + blk, act_f8, act_scale);
        }

        blk++;
        if (blk == nBlks)
        {
            blk = 0;
            g++;
        }
    }
}
