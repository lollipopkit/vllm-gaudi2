// SPDX-License-Identifier: Apache-2.0
// Zero-copy indirect FP8 MoE w13 GEMV + Softcapped SiTU + per-128-group compact FP8 activation quant
// for Inkling 2 Mini on Gaudi2 TPC.
//
// Dispatches totalItems = nBlks * topK * B independent work items (chunk_size = 1) across 24 TPC cores.
// Outputs COMPACT 128B act_f8 ([q_even[0..63], q_odd[0..63]]) via st_f32_as_f8, uses incremental
// blk/slot/b counters (0 inner udiv), and ONLY 1 SFU LUT (0x0 tanh/sigmoid).
//
// Tensors (TPC dim order, dim0 fastest):
//   x_i32        [K/4, B] i32
//   x_scale      [B] f32
//   w13_tiled    [256, K, nBlks, E] f8_143  (PyTorch [E, nBlks, K, 256])
//   s13_packed   [2*I, E] f32               (PyTorch [E, 2*I])
//   topk_ids     [K_top, B] i32
//   topk_weights [K_top, B] f32
//   act_f8       [128, nBlks * K_top * B] f8_143  (PyTorch [B, K_top, I] viewed as [B*K_top*nBlks, 128])
//   act_scale    [nBlks * K_top * B] f32          (PyTorch [B, K_top, nBlks] viewed as [B*K_top*nBlks])
//
// Index space: [totalItems] (1D).

#include "inkling2_bf16_common.h"

#define MAX_K4 1024
#ifndef W_LD_SW
#define W_LD_SW SW_L0CD
#endif

void main(tensor x_i32, tensor x_scale, tensor w13_tiled, tensor s13_packed,
          tensor topk_ids, tensor topk_weights, tensor act_f8, tensor act_scale,
          float gate_cap, float up_cap, float fp8_max, int total_items, int chunk_size)
{
    const int K4 = get_dim_size(x_i32, 0);
    const int nBlks = get_dim_size(w13_tiled, 2);
    const int topK = get_dim_size(topk_ids, 0);
    const int5 is_start = get_index_space_offset();
    const int5 is_end = get_index_space_size() + is_start;

    const float inv_gcap = 1.0f / gate_cap;
    const float inv_ucap = 1.0f / up_cap;
    const float gcap_ucap = gate_cap * up_cap;
    const float inv_fp8_max = 1.0f / fp8_max;

    unsigned int x_slm[MAX_K4];
    int cached_b = -1;
    float x_sc = 0.0f;

    const int id0 = is_start[0] * chunk_size;
    int id_end = is_end[0] * chunk_size;
    if (id_end > total_items)
        id_end = total_items;

    // Initialize incremental (blk, slot, b) counters ONCE per core (zero inner udiv).
    int blk = id0 % nBlks;
    int rs0 = id0 / nBlks;
    int slot = rs0 % topK;
    int b = rs0 / topK;

    for (int id = id0; id < id_end; id++)
    {
        if (b != cached_b)
        {
            int5 sc = {b, 0, 0, 0, 0};
            x_sc = s_f32_ld_g((__global__ float*)gen_addr(sc, x_scale));
            int5 pf = {0, b, 0, 0, 0};
            for (int k4 = 0; k4 < K4; k4 += 16)
            {
                pf[0] = k4;
                prefetch(gen_addr(pf, x_i32));
            }
            const int K4_8 = K4 & ~7;
            int5 xc0 = {0, b, 0, 0, 0}, xc1 = {1, b, 0, 0, 0}, xc2 = {2, b, 0, 0, 0}, xc3 = {3, b, 0, 0, 0};
            int5 xc4 = {4, b, 0, 0, 0}, xc5 = {5, b, 0, 0, 0}, xc6 = {6, b, 0, 0, 0}, xc7 = {7, b, 0, 0, 0};
            for (int k4 = 0; k4 < K4_8; k4 += 8)
            {
                const unsigned int v0 = (unsigned int)s_i32_ld_g((__global__ int*)gen_addr(xc0, x_i32));
                const unsigned int v1 = (unsigned int)s_i32_ld_g((__global__ int*)gen_addr(xc1, x_i32));
                const unsigned int v2 = (unsigned int)s_i32_ld_g((__global__ int*)gen_addr(xc2, x_i32));
                const unsigned int v3 = (unsigned int)s_i32_ld_g((__global__ int*)gen_addr(xc3, x_i32));
                const unsigned int v4 = (unsigned int)s_i32_ld_g((__global__ int*)gen_addr(xc4, x_i32));
                const unsigned int v5 = (unsigned int)s_i32_ld_g((__global__ int*)gen_addr(xc5, x_i32));
                const unsigned int v6 = (unsigned int)s_i32_ld_g((__global__ int*)gen_addr(xc6, x_i32));
                const unsigned int v7 = (unsigned int)s_i32_ld_g((__global__ int*)gen_addr(xc7, x_i32));
                xc0[0] += 8; xc1[0] += 8; xc2[0] += 8; xc3[0] += 8;
                xc4[0] += 8; xc5[0] += 8; xc6[0] += 8; xc7[0] += 8;
                x_slm[k4 + 0] = v0; x_slm[k4 + 1] = v1; x_slm[k4 + 2] = v2; x_slm[k4 + 3] = v3;
                x_slm[k4 + 4] = v4; x_slm[k4 + 5] = v5; x_slm[k4 + 6] = v6; x_slm[k4 + 7] = v7;
            }
            for (int k4 = K4_8; k4 < K4; k4++)
            {
                xc0[0] = k4;
                x_slm[k4] = (unsigned int)s_i32_ld_g((__global__ int*)gen_addr(xc0, x_i32));
            }
            cached_b = b;
        }

        int5 tc = {slot, b, 0, 0, 0};
        const int e = s_i32_ld_g((__global__ int*)gen_addr(tc, topk_ids));
        const float gamma = s_f32_ld_g((__global__ float*)gen_addr(tc, topk_weights));

        const int n_off = blk << 8;
        float256 acc0 = {0}, acc1 = {0}, acc2 = {0}, acc3 = {0};
        int5 wc0 = {0, 0, blk, e, 0}, wc1 = {0, 1, blk, e, 0}, wc2 = {0, 2, blk, e, 0}, wc3 = {0, 3, blk, e, 0};

        const int K4_even = K4 & ~1;
        if (K4_even >= 2)
        {
            unsigned int xwa = x_slm[0];
            minifloat256 wa0 = v_f8_ld_tnsr_b(wc0, w13_tiled, W_LD_SW, 0, 1, 0);
            minifloat256 wa1 = v_f8_ld_tnsr_b(wc1, w13_tiled, W_LD_SW, 0, 1, 0);
            minifloat256 wa2 = v_f8_ld_tnsr_b(wc2, w13_tiled, W_LD_SW, 0, 1, 0);
            minifloat256 wa3 = v_f8_ld_tnsr_b(wc3, w13_tiled, W_LD_SW, 0, 1, 0);

            unsigned int xwb = x_slm[1];
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
                xwa = x_slm[k4];

                wc0[1] += 4; wc1[1] += 4; wc2[1] += 4; wc3[1] += 4;
                acc0 = v_f8_mac_acc32_b(wb0, bits_to_f8(xwb), acc0);
                wb0 = v_f8_ld_tnsr_b(wc0, w13_tiled, W_LD_SW, 0, 1, 0);
                acc1 = v_f8_mac_acc32_b(wb1, bits_to_f8(xwb >> 8), acc1);
                wb1 = v_f8_ld_tnsr_b(wc1, w13_tiled, W_LD_SW, 0, 1, 0);
                acc2 = v_f8_mac_acc32_b(wb2, bits_to_f8(xwb >> 16), acc2);
                wb2 = v_f8_ld_tnsr_b(wc2, w13_tiled, W_LD_SW, 0, 1, 0);
                acc3 = v_f8_mac_acc32_b(wb3, bits_to_f8(xwb >> 24), acc3);
                wb3 = v_f8_ld_tnsr_b(wc3, w13_tiled, W_LD_SW, 0, 1, 0);
                xwb = x_slm[k4 + 1];
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
            const unsigned int xw = x_slm[K4_even];
            acc0 = v_f8_mac_acc32_b(v_f8_ld_tnsr_b(wc0, w13_tiled, W_LD_SW, 0, 1, 0), bits_to_f8(xw), acc0);
            acc1 = v_f8_mac_acc32_b(v_f8_ld_tnsr_b(wc1, w13_tiled, W_LD_SW, 0, 1, 0), bits_to_f8(xw >> 8), acc1);
            acc2 = v_f8_mac_acc32_b(v_f8_ld_tnsr_b(wc2, w13_tiled, W_LD_SW, 0, 1, 0), bits_to_f8(xw >> 16), acc2);
            acc3 = v_f8_mac_acc32_b(v_f8_ld_tnsr_b(wc3, w13_tiled, W_LD_SW, 0, 1, 0), bits_to_f8(xw >> 24), acc3);
        }

        int5 s13c = {n_off, e, 0, 0, 0};
        const float64 sg_even = v_f32_ld_tnsr_b(s13c, s13_packed) * x_sc;
        s13c[0] = n_off + 64;
        const float64 sg_odd = v_f32_ld_tnsr_b(s13c, s13_packed) * x_sc;
        s13c[0] = n_off + 128;
        const float64 su_even = v_f32_ld_tnsr_b(s13c, s13_packed) * x_sc;
        s13c[0] = n_off + 192;
        const float64 su_odd = v_f32_ld_tnsr_b(s13c, s13_packed) * x_sc;

        const float64 g_even = ((acc0.v1 + acc1.v1) + (acc2.v1 + acc3.v1)) * sg_even;
        const float64 g_odd = ((acc0.v2 + acc1.v2) + (acc2.v2 + acc3.v2)) * sg_odd;
        const float64 u_even = ((acc0.v3 + acc1.v3) + (acc2.v3 + acc3.v3)) * su_even;
        const float64 u_odd = ((acc0.v4 + acc1.v4) + (acc2.v4 + acc3.v4)) * su_odd;

        const float scale_gamma = gcap_ucap * gamma;
        const float64 act_even = softcapped_situ_vec(g_even, u_even, inv_gcap, inv_ucap, scale_gamma);
        const float64 act_odd = softcapped_situ_vec(g_odd, u_odd, inv_gcap, inv_ucap, scale_gamma);

        const float64 abs_max_v = v_f32_max_b(v_f32_abs_b(act_even), v_f32_abs_b(act_odd));
        const float64 red_max = v_f32_reduce_nolut_max(abs_max_v);
        const float64 a_sc_v = v_f32_max_b(red_max * inv_fp8_max, 1e-12f);
        const float64 inv_a_sc_v = rcp_pos_nolut(a_sc_v);

        const float64 q_even = v_f32_min_b(v_f32_max_b(act_even * inv_a_sc_v, -fp8_max), fp8_max);
        const float64 q_odd = v_f32_min_b(v_f32_max_b(act_odd * inv_a_sc_v, -fp8_max), fp8_max);

        // Compact 128B natural-order store: [q_even[0..63], q_odd[0..63]].
        int5 oc = {0, id, 0, 0, 0};
        st_f32_as_f8(oc, act_f8, q_even);
        oc[0] = 64;
        st_f32_as_f8(oc, act_f8, q_odd);

        int5 asc = {id, 0, 0, 0, 0};
        v_f32_st_tnsr_partial(asc, act_scale, a_sc_v, 0, 0);

        blk++;
        if (blk == nBlks)
        {
            blk = 0;
            slot++;
            if (slot == topK)
            {
                slot = 0;
                b++;
            }
        }
    }
}
