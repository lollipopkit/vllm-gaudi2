// SPDX-License-Identifier: Apache-2.0
// Expert-Grouped Zero-Copy Indirect FP8 MoE w2 GEMV for Inkling 2 Mini on Gaudi2 TPC (B >= 2).
//
// Reads each unique expert's w2 weight tile ONCE from HBM across up to 4 tokens sharing that
// expert with ZERO VRF spills and ZERO SFU LUTs across all four specialized loops (cnt = 1, 2, 3, 4)
// and cost-weighted core partitioning (via precomputed group_meta row 7) so all 24 TPC cores finish in lockstep.
//
// Tensors (TPC dim order, dim0 fastest):
//   act_i32      [I/4, K_top, B] i32
//   act_scale    [I/128, K_top, B] f32
//   w2_tiled     [256, I, nBlks, E] f8_143
//   s2_packed    [H, E] f32
//   group_meta   [128, 8] i32                     (PyTorch [8, 128]; rows: e, cnt, r0..3, w13_bounds, w2_bounds)
//   group_count  [2] i32                          (PyTorch [2]; [G, total_w])
//   out_bf16     [256, nBlks * K_top * B] bf16
//
// Index space: [num_cores] (1D; cost-weighted dynamic split of [0, G * nBlks) across cores).

#include "inkling2_bf16_common.h"

#define MAX_GRP_I4 384
#define MAX_GROUPS 32
#ifndef W_LD_SW
#define W_LD_SW SW_L0CD
#endif

static __local__ float64 g_tot[4][4];

void main(tensor act_i32, tensor act_scale, tensor w2_tiled, tensor s2_packed,
          tensor group_meta, tensor group_count, tensor out_bf16, int num_cores)
{
    (void)group_count;
    (void)num_cores;
    const int I4 = get_dim_size(act_i32, 0);
    const int num_groups = get_dim_size(act_scale, 0);
    const int topK = get_dim_size(act_i32, 1);
    const int nBlks = get_dim_size(w2_tiled, 2);
    const int5 is_start = get_index_space_offset();
    const int5 is_end = get_index_space_size() + is_start;

    // O(1) startup: read precomputed cost-weighted tile interval [w0_idx, w1_idx) from row 7!
    int5 b0c = {is_start[0], 7, 0, 0, 0};
    int5 b1c = {is_end[0], 7, 0, 0, 0};
    const int w0_idx = s_i32_ld_g((__global__ int*)gen_addr(b0c, group_meta));
    const int w1_idx = s_i32_ld_g((__global__ int*)gen_addr(b1c, group_meta));
    if (w0_idx >= w1_idx)
        return;

    unsigned int act_slm[MAX_GRP_I4][4];
    float sc_slm[MAX_GROUPS][4];
    int cached_g = -1;
    int g_e = 0, g_cnt = 0;
    int g_rs[4];

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

                int5 pf = {0, slot, b, 0, 0};
                for (int i = 0; i < I4; i += 16)
                {
                    pf[0] = i;
                    prefetch(gen_addr(pf, act_i32));
                }
                int5 asc = {0, slot, b, 0, 0};
                prefetch(gen_addr(asc, act_scale));
                const int I4_4 = I4 & ~3;
                int5 ac0 = {0, slot, b, 0, 0}, ac1 = {1, slot, b, 0, 0};
                int5 ac2 = {2, slot, b, 0, 0}, ac3 = {3, slot, b, 0, 0};
                for (int i = 0; i < I4_4; i += 4)
                {
                    const unsigned int v0 = (unsigned int)s_i32_ld_g((__global__ int*)gen_addr(ac0, act_i32));
                    const unsigned int v1 = (unsigned int)s_i32_ld_g((__global__ int*)gen_addr(ac1, act_i32));
                    const unsigned int v2 = (unsigned int)s_i32_ld_g((__global__ int*)gen_addr(ac2, act_i32));
                    const unsigned int v3 = (unsigned int)s_i32_ld_g((__global__ int*)gen_addr(ac3, act_i32));
                    ac0[0] += 4; ac1[0] += 4; ac2[0] += 4; ac3[0] += 4;
                    act_slm[i + 0][j] = v0; act_slm[i + 1][j] = v1;
                    act_slm[i + 2][j] = v2; act_slm[i + 3][j] = v3;
                }
                for (int i = I4_4; i < I4; i++)
                {
                    ac0[0] = i;
                    act_slm[i][j] = (unsigned int)s_i32_ld_g((__global__ int*)gen_addr(ac0, act_i32));
                }
                for (int sg = 0; sg < num_groups; sg++)
                {
                    asc[0] = sg;
                    sc_slm[sg][j] = s_f32_ld_g((__global__ float*)gen_addr(asc, act_scale));
                }
            }
            cached_g = g;
        }

        const int e = g_e;
        const int n_off = blk << 8;
        int5 wc0 = {0, 0, blk, e, 0}, wc1 = {0, 1, blk, e, 0};
        int5 wc2 = {0, 2, blk, e, 0}, wc3 = {0, 3, blk, e, 0};

        int5 s2c = {n_off, e, 0, 0, 0};
        const float64 s2_0 = v_f32_ld_tnsr_b(s2c, s2_packed);
        s2c[0] = n_off + 64;
        const float64 s2_1 = v_f32_ld_tnsr_b(s2c, s2_packed);
        s2c[0] = n_off + 128;
        const float64 s2_2 = v_f32_ld_tnsr_b(s2c, s2_packed);
        s2c[0] = n_off + 192;
        const float64 s2_3 = v_f32_ld_tnsr_b(s2c, s2_packed);

        if (g_cnt == 1)
        {
            float64 tot1 = 0.0f, tot2 = 0.0f, tot3 = 0.0f, tot4 = 0.0f;
            for (int sg = 0; sg < num_groups; sg++)
            {
                const float g_sc = sc_slm[sg][0];
                const int w_base = sg << 5;
                float256 acc0 = {0.0f, 0.0f, 0.0f, 0.0f}, acc1 = {0.0f, 0.0f, 0.0f, 0.0f};
                float256 acc2 = {0.0f, 0.0f, 0.0f, 0.0f}, acc3 = {0.0f, 0.0f, 0.0f, 0.0f};

                unsigned int awa = act_slm[w_base][0];
                minifloat256 wa0 = v_f8_ld_tnsr_b(wc0, w2_tiled, W_LD_SW, 0, 1, 0);
                minifloat256 wa1 = v_f8_ld_tnsr_b(wc1, w2_tiled, W_LD_SW, 0, 1, 0);
                minifloat256 wa2 = v_f8_ld_tnsr_b(wc2, w2_tiled, W_LD_SW, 0, 1, 0);
                minifloat256 wa3 = v_f8_ld_tnsr_b(wc3, w2_tiled, W_LD_SW, 0, 1, 0);

                unsigned int awb = act_slm[w_base + 1][0];
                wc0[1] += 4; wc1[1] += 4; wc2[1] += 4; wc3[1] += 4;
                minifloat256 wb0 = v_f8_ld_tnsr_b(wc0, w2_tiled, W_LD_SW, 0, 1, 0);
                minifloat256 wb1 = v_f8_ld_tnsr_b(wc1, w2_tiled, W_LD_SW, 0, 1, 0);
                minifloat256 wb2 = v_f8_ld_tnsr_b(wc2, w2_tiled, W_LD_SW, 0, 1, 0);
                minifloat256 wb3 = v_f8_ld_tnsr_b(wc3, w2_tiled, W_LD_SW, 0, 1, 0);

                for (int w = w_base + 2; w < w_base + 32; w += 2)
                {
                    wc0[1] += 4; wc1[1] += 4; wc2[1] += 4; wc3[1] += 4;
                    acc0 = v_f8_mac_acc32_b(wa0, bits_to_f8(awa), acc0);
                    wa0 = v_f8_ld_tnsr_b(wc0, w2_tiled, W_LD_SW, 0, 1, 0);
                    acc1 = v_f8_mac_acc32_b(wa1, bits_to_f8(awa >> 8), acc1);
                    wa1 = v_f8_ld_tnsr_b(wc1, w2_tiled, W_LD_SW, 0, 1, 0);
                    acc2 = v_f8_mac_acc32_b(wa2, bits_to_f8(awa >> 16), acc2);
                    wa2 = v_f8_ld_tnsr_b(wc2, w2_tiled, W_LD_SW, 0, 1, 0);
                    acc3 = v_f8_mac_acc32_b(wa3, bits_to_f8(awa >> 24), acc3);
                    wa3 = v_f8_ld_tnsr_b(wc3, w2_tiled, W_LD_SW, 0, 1, 0);
                    awa = act_slm[w][0];

                    wc0[1] += 4; wc1[1] += 4; wc2[1] += 4; wc3[1] += 4;
                    acc0 = v_f8_mac_acc32_b(wb0, bits_to_f8(awb), acc0);
                    wb0 = v_f8_ld_tnsr_b(wc0, w2_tiled, W_LD_SW, 0, 1, 0);
                    acc1 = v_f8_mac_acc32_b(wb1, bits_to_f8(awb >> 8), acc1);
                    wb1 = v_f8_ld_tnsr_b(wc1, w2_tiled, W_LD_SW, 0, 1, 0);
                    acc2 = v_f8_mac_acc32_b(wb2, bits_to_f8(awb >> 16), acc2);
                    wb2 = v_f8_ld_tnsr_b(wc2, w2_tiled, W_LD_SW, 0, 1, 0);
                    acc3 = v_f8_mac_acc32_b(wb3, bits_to_f8(awb >> 24), acc3);
                    wb3 = v_f8_ld_tnsr_b(wc3, w2_tiled, W_LD_SW, 0, 1, 0);
                    awb = act_slm[w + 1][0];
                }
                acc0 = v_f8_mac_acc32_b(wa0, bits_to_f8(awa), acc0);
                acc1 = v_f8_mac_acc32_b(wa1, bits_to_f8(awa >> 8), acc1);
                acc2 = v_f8_mac_acc32_b(wa2, bits_to_f8(awa >> 16), acc2);
                acc3 = v_f8_mac_acc32_b(wa3, bits_to_f8(awa >> 24), acc3);
                acc0 = v_f8_mac_acc32_b(wb0, bits_to_f8(awb), acc0);
                acc1 = v_f8_mac_acc32_b(wb1, bits_to_f8(awb >> 8), acc1);
                acc2 = v_f8_mac_acc32_b(wb2, bits_to_f8(awb >> 16), acc2);
                acc3 = v_f8_mac_acc32_b(wb3, bits_to_f8(awb >> 24), acc3);
                wc0[1] += 4; wc1[1] += 4; wc2[1] += 4; wc3[1] += 4;

                tot1 = v_f32_mac_b((acc0.v1 + acc1.v1) + (acc2.v1 + acc3.v1), g_sc, tot1);
                tot2 = v_f32_mac_b((acc0.v2 + acc1.v2) + (acc2.v2 + acc3.v2), g_sc, tot2);
                tot3 = v_f32_mac_b((acc0.v3 + acc1.v3) + (acc2.v3 + acc3.v3), g_sc, tot3);
                tot4 = v_f32_mac_b((acc0.v4 + acc1.v4) + (acc2.v4 + acc3.v4), g_sc, tot4);
            }
            g_tot[0][0] = tot1; g_tot[0][1] = tot2; g_tot[0][2] = tot3; g_tot[0][3] = tot4;
        }
        else if (g_cnt == 2)
        {
            g_tot[0][0] = 0.0f; g_tot[0][1] = 0.0f; g_tot[0][2] = 0.0f; g_tot[0][3] = 0.0f;
            g_tot[1][0] = 0.0f; g_tot[1][1] = 0.0f; g_tot[1][2] = 0.0f; g_tot[1][3] = 0.0f;
            for (int sg = 0; sg < num_groups; sg++)
            {
                const int w_base = sg << 5;
                float256 a0a = {0.0f, 0.0f, 0.0f, 0.0f}, a0b = {0.0f, 0.0f, 0.0f, 0.0f};
                float256 a1a = {0.0f, 0.0f, 0.0f, 0.0f}, a1b = {0.0f, 0.0f, 0.0f, 0.0f};

                unsigned int x0 = act_slm[w_base][0], x1 = act_slm[w_base][1];
                minifloat256 w0 = v_f8_ld_tnsr_b(wc0, w2_tiled, W_LD_SW, 0, 1, 0);
                minifloat256 w1 = v_f8_ld_tnsr_b(wc1, w2_tiled, W_LD_SW, 0, 1, 0);
                minifloat256 w2 = v_f8_ld_tnsr_b(wc2, w2_tiled, W_LD_SW, 0, 1, 0);
                minifloat256 w3 = v_f8_ld_tnsr_b(wc3, w2_tiled, W_LD_SW, 0, 1, 0);

                for (int w = w_base; w < w_base + 31; w++)
                {
                    const minifloat b00 = bits_to_f8(x0), b01 = bits_to_f8(x1);
                    const minifloat b10 = bits_to_f8(x0 >> 8), b11 = bits_to_f8(x1 >> 8);
                    const minifloat b20 = bits_to_f8(x0 >> 16), b21 = bits_to_f8(x1 >> 16);
                    const minifloat b30 = bits_to_f8(x0 >> 24), b31 = bits_to_f8(x1 >> 24);
                    x0 = act_slm[w + 1][0];
                    x1 = act_slm[w + 1][1];
                    wc0[1] += 4; wc1[1] += 4; wc2[1] += 4; wc3[1] += 4;

                    a0a = v_f8_mac_acc32_b(w0, b00, a0a); a1a = v_f8_mac_acc32_b(w0, b01, a1a);
                    w0 = v_f8_ld_tnsr_b(wc0, w2_tiled, W_LD_SW, 0, 1, 0);
                    a0b = v_f8_mac_acc32_b(w1, b10, a0b); a1b = v_f8_mac_acc32_b(w1, b11, a1b);
                    w1 = v_f8_ld_tnsr_b(wc1, w2_tiled, W_LD_SW, 0, 1, 0);
                    a0a = v_f8_mac_acc32_b(w2, b20, a0a); a1a = v_f8_mac_acc32_b(w2, b21, a1a);
                    w2 = v_f8_ld_tnsr_b(wc2, w2_tiled, W_LD_SW, 0, 1, 0);
                    a0b = v_f8_mac_acc32_b(w3, b30, a0b); a1b = v_f8_mac_acc32_b(w3, b31, a1b);
                    w3 = v_f8_ld_tnsr_b(wc3, w2_tiled, W_LD_SW, 0, 1, 0);
                }
                {
                    a0a = v_f8_mac_acc32_b(w0, bits_to_f8(x0), a0a); a1a = v_f8_mac_acc32_b(w0, bits_to_f8(x1), a1a);
                    a0b = v_f8_mac_acc32_b(w1, bits_to_f8(x0 >> 8), a0b); a1b = v_f8_mac_acc32_b(w1, bits_to_f8(x1 >> 8), a1b);
                    a0a = v_f8_mac_acc32_b(w2, bits_to_f8(x0 >> 16), a0a); a1a = v_f8_mac_acc32_b(w2, bits_to_f8(x1 >> 16), a1a);
                    a0b = v_f8_mac_acc32_b(w3, bits_to_f8(x0 >> 24), a0b); a1b = v_f8_mac_acc32_b(w3, bits_to_f8(x1 >> 24), a1b);
                    wc0[1] += 4; wc1[1] += 4; wc2[1] += 4; wc3[1] += 4;
                }
                const float sc0 = sc_slm[sg][0], sc1 = sc_slm[sg][1];
                g_tot[0][0] = v_f32_mac_b(a0a.v1 + a0b.v1, sc0, g_tot[0][0]);
                g_tot[0][1] = v_f32_mac_b(a0a.v2 + a0b.v2, sc0, g_tot[0][1]);
                g_tot[0][2] = v_f32_mac_b(a0a.v3 + a0b.v3, sc0, g_tot[0][2]);
                g_tot[0][3] = v_f32_mac_b(a0a.v4 + a0b.v4, sc0, g_tot[0][3]);
                g_tot[1][0] = v_f32_mac_b(a1a.v1 + a1b.v1, sc1, g_tot[1][0]);
                g_tot[1][1] = v_f32_mac_b(a1a.v2 + a1b.v2, sc1, g_tot[1][1]);
                g_tot[1][2] = v_f32_mac_b(a1a.v3 + a1b.v3, sc1, g_tot[1][2]);
                g_tot[1][3] = v_f32_mac_b(a1a.v4 + a1b.v4, sc1, g_tot[1][3]);
            }
        }
        else if (g_cnt == 3)
        {
            for (int j = 0; j < 3; j++)
            {
                g_tot[j][0] = 0.0f; g_tot[j][1] = 0.0f; g_tot[j][2] = 0.0f; g_tot[j][3] = 0.0f;
            }
            for (int sg = 0; sg < num_groups; sg++)
            {
                const int w_base = sg << 5;
                float256 a0 = {0.0f, 0.0f, 0.0f, 0.0f};
                float256 a1 = {0.0f, 0.0f, 0.0f, 0.0f};
                float256 a2 = {0.0f, 0.0f, 0.0f, 0.0f};

                unsigned int x0 = act_slm[w_base][0], x1 = act_slm[w_base][1], x2 = act_slm[w_base][2];
                minifloat256 w0 = v_f8_ld_tnsr_b(wc0, w2_tiled, W_LD_SW, 0, 1, 0);
                minifloat256 w1 = v_f8_ld_tnsr_b(wc1, w2_tiled, W_LD_SW, 0, 1, 0);
                minifloat256 w2 = v_f8_ld_tnsr_b(wc2, w2_tiled, W_LD_SW, 0, 1, 0);
                minifloat256 w3 = v_f8_ld_tnsr_b(wc3, w2_tiled, W_LD_SW, 0, 1, 0);

                for (int w = w_base; w < w_base + 31; w++)
                {
                    const minifloat b00 = bits_to_f8(x0), b01 = bits_to_f8(x1), b02 = bits_to_f8(x2);
                    const minifloat b10 = bits_to_f8(x0 >> 8), b11 = bits_to_f8(x1 >> 8), b12 = bits_to_f8(x2 >> 8);
                    wc0[1] += 4; wc1[1] += 4; wc2[1] += 4; wc3[1] += 4;

                    a0 = v_f8_mac_acc32_b(w0, b00, a0); a1 = v_f8_mac_acc32_b(w0, b01, a1); a2 = v_f8_mac_acc32_b(w0, b02, a2);
                    w0 = v_f8_ld_tnsr_b(wc0, w2_tiled, W_LD_SW, 0, 1, 0);

                    const minifloat b20 = bits_to_f8(x0 >> 16), b21 = bits_to_f8(x1 >> 16), b22 = bits_to_f8(x2 >> 16);
                    a0 = v_f8_mac_acc32_b(w1, b10, a0); a1 = v_f8_mac_acc32_b(w1, b11, a1); a2 = v_f8_mac_acc32_b(w1, b12, a2);
                    w1 = v_f8_ld_tnsr_b(wc1, w2_tiled, W_LD_SW, 0, 1, 0);

                    const minifloat b30 = bits_to_f8(x0 >> 24), b31 = bits_to_f8(x1 >> 24), b32 = bits_to_f8(x2 >> 24);
                    a0 = v_f8_mac_acc32_b(w2, b20, a0); a1 = v_f8_mac_acc32_b(w2, b21, a1); a2 = v_f8_mac_acc32_b(w2, b22, a2);
                    w2 = v_f8_ld_tnsr_b(wc2, w2_tiled, W_LD_SW, 0, 1, 0);

                    x0 = act_slm[w + 1][0]; x1 = act_slm[w + 1][1]; x2 = act_slm[w + 1][2];
                    a0 = v_f8_mac_acc32_b(w3, b30, a0); a1 = v_f8_mac_acc32_b(w3, b31, a1); a2 = v_f8_mac_acc32_b(w3, b32, a2);
                    w3 = v_f8_ld_tnsr_b(wc3, w2_tiled, W_LD_SW, 0, 1, 0);
                }
                {
                    a0 = v_f8_mac_acc32_b(w0, bits_to_f8(x0), a0); a1 = v_f8_mac_acc32_b(w0, bits_to_f8(x1), a1); a2 = v_f8_mac_acc32_b(w0, bits_to_f8(x2), a2);
                    a0 = v_f8_mac_acc32_b(w1, bits_to_f8(x0 >> 8), a0); a1 = v_f8_mac_acc32_b(w1, bits_to_f8(x1 >> 8), a1); a2 = v_f8_mac_acc32_b(w1, bits_to_f8(x2 >> 8), a2);
                    a0 = v_f8_mac_acc32_b(w2, bits_to_f8(x0 >> 16), a0); a1 = v_f8_mac_acc32_b(w2, bits_to_f8(x1 >> 16), a1); a2 = v_f8_mac_acc32_b(w2, bits_to_f8(x2 >> 16), a2);
                    a0 = v_f8_mac_acc32_b(w3, bits_to_f8(x0 >> 24), a0); a1 = v_f8_mac_acc32_b(w3, bits_to_f8(x1 >> 24), a1); a2 = v_f8_mac_acc32_b(w3, bits_to_f8(x2 >> 24), a2);
                    wc0[1] += 4; wc1[1] += 4; wc2[1] += 4; wc3[1] += 4;
                }
                const float sc0 = sc_slm[sg][0], sc1 = sc_slm[sg][1], sc2 = sc_slm[sg][2];
                g_tot[0][0] = v_f32_mac_b(a0.v1, sc0, g_tot[0][0]); g_tot[0][1] = v_f32_mac_b(a0.v2, sc0, g_tot[0][1]);
                g_tot[0][2] = v_f32_mac_b(a0.v3, sc0, g_tot[0][2]); g_tot[0][3] = v_f32_mac_b(a0.v4, sc0, g_tot[0][3]);
                g_tot[1][0] = v_f32_mac_b(a1.v1, sc1, g_tot[1][0]); g_tot[1][1] = v_f32_mac_b(a1.v2, sc1, g_tot[1][1]);
                g_tot[1][2] = v_f32_mac_b(a1.v3, sc1, g_tot[1][2]); g_tot[1][3] = v_f32_mac_b(a1.v4, sc1, g_tot[1][3]);
                g_tot[2][0] = v_f32_mac_b(a2.v1, sc2, g_tot[2][0]); g_tot[2][1] = v_f32_mac_b(a2.v2, sc2, g_tot[2][1]);
                g_tot[2][2] = v_f32_mac_b(a2.v3, sc2, g_tot[2][2]); g_tot[2][3] = v_f32_mac_b(a2.v4, sc2, g_tot[2][3]);
            }
        }
        else
        {
            for (int j = 0; j < 4; j++)
            {
                g_tot[j][0] = 0.0f; g_tot[j][1] = 0.0f; g_tot[j][2] = 0.0f; g_tot[j][3] = 0.0f;
            }
            for (int sg = 0; sg < num_groups; sg++)
            {
                const int w_base = sg << 5;
                float256 a0 = {0.0f, 0.0f, 0.0f, 0.0f}, a1 = {0.0f, 0.0f, 0.0f, 0.0f};
                float256 a2 = {0.0f, 0.0f, 0.0f, 0.0f}, a3 = {0.0f, 0.0f, 0.0f, 0.0f};

                unsigned int x0 = act_slm[w_base][0], x1 = act_slm[w_base][1];
                unsigned int x2 = act_slm[w_base][2], x3 = act_slm[w_base][3];
                minifloat256 w0 = v_f8_ld_tnsr_b(wc0, w2_tiled, W_LD_SW, 0, 1, 0);
                minifloat256 w1 = v_f8_ld_tnsr_b(wc1, w2_tiled, W_LD_SW, 0, 1, 0);
                minifloat256 w2 = v_f8_ld_tnsr_b(wc2, w2_tiled, W_LD_SW, 0, 1, 0);
                minifloat256 w3 = v_f8_ld_tnsr_b(wc3, w2_tiled, W_LD_SW, 0, 1, 0);

                for (int w = w_base; w < w_base + 31; w++)
                {
                    const minifloat b00 = bits_to_f8(x0), b01 = bits_to_f8(x1), b02 = bits_to_f8(x2), b03 = bits_to_f8(x3);
                    const minifloat b10 = bits_to_f8(x0 >> 8), b11 = bits_to_f8(x1 >> 8), b12 = bits_to_f8(x2 >> 8), b13 = bits_to_f8(x3 >> 8);
                    wc0[1] += 4; wc1[1] += 4; wc2[1] += 4; wc3[1] += 4;

                    a0 = v_f8_mac_acc32_b(w0, b00, a0); a1 = v_f8_mac_acc32_b(w0, b01, a1);
                    a2 = v_f8_mac_acc32_b(w0, b02, a2); a3 = v_f8_mac_acc32_b(w0, b03, a3);
                    w0 = v_f8_ld_tnsr_b(wc0, w2_tiled, W_LD_SW, 0, 1, 0);

                    const minifloat b20 = bits_to_f8(x0 >> 16), b21 = bits_to_f8(x1 >> 16), b22 = bits_to_f8(x2 >> 16), b23 = bits_to_f8(x3 >> 16);
                    a0 = v_f8_mac_acc32_b(w1, b10, a0); a1 = v_f8_mac_acc32_b(w1, b11, a1);
                    a2 = v_f8_mac_acc32_b(w1, b12, a2); a3 = v_f8_mac_acc32_b(w1, b13, a3);
                    w1 = v_f8_ld_tnsr_b(wc1, w2_tiled, W_LD_SW, 0, 1, 0);

                    const minifloat b30 = bits_to_f8(x0 >> 24), b31 = bits_to_f8(x1 >> 24), b32 = bits_to_f8(x2 >> 24), b33 = bits_to_f8(x3 >> 24);
                    a0 = v_f8_mac_acc32_b(w2, b20, a0); a1 = v_f8_mac_acc32_b(w2, b21, a1);
                    a2 = v_f8_mac_acc32_b(w2, b22, a2); a3 = v_f8_mac_acc32_b(w2, b23, a3);
                    w2 = v_f8_ld_tnsr_b(wc2, w2_tiled, W_LD_SW, 0, 1, 0);

                    x0 = act_slm[w + 1][0]; x1 = act_slm[w + 1][1]; x2 = act_slm[w + 1][2]; x3 = act_slm[w + 1][3];
                    a0 = v_f8_mac_acc32_b(w3, b30, a0); a1 = v_f8_mac_acc32_b(w3, b31, a1);
                    a2 = v_f8_mac_acc32_b(w3, b32, a2); a3 = v_f8_mac_acc32_b(w3, b33, a3);
                    w3 = v_f8_ld_tnsr_b(wc3, w2_tiled, W_LD_SW, 0, 1, 0);
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
                    wc0[1] += 4; wc1[1] += 4; wc2[1] += 4; wc3[1] += 4;
                }
                const float sc0 = sc_slm[sg][0], sc1 = sc_slm[sg][1], sc2 = sc_slm[sg][2], sc3 = sc_slm[sg][3];
                g_tot[0][0] = v_f32_mac_b(a0.v1, sc0, g_tot[0][0]); g_tot[0][1] = v_f32_mac_b(a0.v2, sc0, g_tot[0][1]);
                g_tot[0][2] = v_f32_mac_b(a0.v3, sc0, g_tot[0][2]); g_tot[0][3] = v_f32_mac_b(a0.v4, sc0, g_tot[0][3]);
                g_tot[1][0] = v_f32_mac_b(a1.v1, sc1, g_tot[1][0]); g_tot[1][1] = v_f32_mac_b(a1.v2, sc1, g_tot[1][1]);
                g_tot[1][2] = v_f32_mac_b(a1.v3, sc1, g_tot[1][2]); g_tot[1][3] = v_f32_mac_b(a1.v4, sc1, g_tot[1][3]);
                g_tot[2][0] = v_f32_mac_b(a2.v1, sc2, g_tot[2][0]); g_tot[2][1] = v_f32_mac_b(a2.v2, sc2, g_tot[2][1]);
                g_tot[2][2] = v_f32_mac_b(a2.v3, sc2, g_tot[2][2]); g_tot[2][3] = v_f32_mac_b(a2.v4, sc2, g_tot[2][3]);
                g_tot[3][0] = v_f32_mac_b(a3.v1, sc3, g_tot[3][0]); g_tot[3][1] = v_f32_mac_b(a3.v2, sc3, g_tot[3][1]);
                g_tot[3][2] = v_f32_mac_b(a3.v3, sc3, g_tot[3][2]); g_tot[3][3] = v_f32_mac_b(a3.v4, sc3, g_tot[3][3]);
            }
        }

        #pragma clang loop unroll(disable)
        for (int j = 0; j < g_cnt; j++)
        {
            const float64_pair_t p_lo = {g_tot[j][0] * s2_0, g_tot[j][1] * s2_1};
            const float64_pair_t p_hi = {g_tot[j][2] * s2_2, g_tot[j][3] * s2_3};
            int5 oc = {0, g_rs[j] * nBlks + blk, 0, 0, 0};
            v_bf16_st_tnsr(oc, out_bf16, v_convert_f32_to_bf16_all_b(p_lo));
            oc[0] = 128;
            v_bf16_st_tnsr(oc, out_bf16, v_convert_f32_to_bf16_all_b(p_hi));
        }

        blk++;
        if (blk == nBlks)
        {
            blk = 0;
            g++;
        }
    }
}
