// SPDX-License-Identifier: Apache-2.0
// Zero-copy indirect FP8 MoE w2 GEMV consuming compact per-128-group quantized FP8 activations
// from inkling2_moe_w13_situ_f8_gaudi2 on Gaudi2 TPC.
//
// Dispatches totalItems = nBlks * topK * B independent work items (chunk_size = 1) across 24 TPC cores;
// compact act_i32 [I/4, K_top, B] halves scalar SLM traffic; ZERO SFU LUTs.
//
// Tensors (TPC dim order, dim0 fastest):
//   act_i32      [I/4, K_top, B] i32         (PyTorch [B, K_top, I] f8 viewed as i32; each 128B group = [even[0..63], odd[0..63]])
//   act_scale    [I/128, K_top, B] f32
//   w2_tiled     [256, I, nBlks, E] f8_143   (PyTorch [E, nBlks, I, 256] matching compact [even, odd] I order)
//   s2_packed    [H, E] f32                  (PyTorch [E, H])
//   topk_ids     [K_top, B] i32
//   out_bf16     [256, nBlks * K_top * B] bf16 (PyTorch [B, K_top, H] viewed as [B*K_top*nBlks, 256])
//
// Index space: [totalItems] (1D).

#include "inkling2_bf16_common.h"

#define MAX_I4 512
#define MAX_GROUPS 32
#ifndef W_LD_SW
#define W_LD_SW SW_L0CD
#endif

void main(tensor act_i32, tensor act_scale, tensor w2_tiled, tensor s2_packed,
          tensor topk_ids, tensor out_bf16, int total_items, int chunk_size)
{
    const int I4 = get_dim_size(act_i32, 0);           // I / 4
    const int num_groups = get_dim_size(act_scale, 0); // I / 128
    const int nBlks = get_dim_size(w2_tiled, 2);       // H / 256
    const int topK = get_dim_size(topk_ids, 0);
    const int5 is_start = get_index_space_offset();
    const int5 is_end = get_index_space_size() + is_start;

    unsigned int act_slm[MAX_I4];
    float sc_slm[MAX_GROUPS];
    int cached_rs = -1;
    int cached_e = 0;

    const int id0 = is_start[0] * chunk_size;
    int id_end = is_end[0] * chunk_size;
    if (id_end > total_items)
        id_end = total_items;

    int blk = id0 % nBlks;
    int rs = id0 / nBlks;
    int slot = rs % topK;
    int b = rs / topK;

    for (int id = id0; id < id_end; id++)
    {
        if (rs != cached_rs)
        {
            int5 tc = {slot, b, 0, 0, 0};
            cached_e = s_i32_ld_g((__global__ int*)gen_addr(tc, topk_ids));
            int5 pf = {0, slot, b, 0, 0};
            for (int i = 0; i < I4; i += 16)
            {
                pf[0] = i;
                prefetch(gen_addr(pf, act_i32));
            }
            int5 asc = {0, slot, b, 0, 0};
            prefetch(gen_addr(asc, act_scale));
            const int I4_8 = I4 & ~7;
            int5 ac0 = {0, slot, b, 0, 0}, ac1 = {1, slot, b, 0, 0}, ac2 = {2, slot, b, 0, 0}, ac3 = {3, slot, b, 0, 0};
            int5 ac4 = {4, slot, b, 0, 0}, ac5 = {5, slot, b, 0, 0}, ac6 = {6, slot, b, 0, 0}, ac7 = {7, slot, b, 0, 0};
            for (int i = 0; i < I4_8; i += 8)
            {
                const unsigned int v0 = (unsigned int)s_i32_ld_g((__global__ int*)gen_addr(ac0, act_i32));
                const unsigned int v1 = (unsigned int)s_i32_ld_g((__global__ int*)gen_addr(ac1, act_i32));
                const unsigned int v2 = (unsigned int)s_i32_ld_g((__global__ int*)gen_addr(ac2, act_i32));
                const unsigned int v3 = (unsigned int)s_i32_ld_g((__global__ int*)gen_addr(ac3, act_i32));
                const unsigned int v4 = (unsigned int)s_i32_ld_g((__global__ int*)gen_addr(ac4, act_i32));
                const unsigned int v5 = (unsigned int)s_i32_ld_g((__global__ int*)gen_addr(ac5, act_i32));
                const unsigned int v6 = (unsigned int)s_i32_ld_g((__global__ int*)gen_addr(ac6, act_i32));
                const unsigned int v7 = (unsigned int)s_i32_ld_g((__global__ int*)gen_addr(ac7, act_i32));
                ac0[0] += 8; ac1[0] += 8; ac2[0] += 8; ac3[0] += 8;
                ac4[0] += 8; ac5[0] += 8; ac6[0] += 8; ac7[0] += 8;
                act_slm[i + 0] = v0; act_slm[i + 1] = v1; act_slm[i + 2] = v2; act_slm[i + 3] = v3;
                act_slm[i + 4] = v4; act_slm[i + 5] = v5; act_slm[i + 6] = v6; act_slm[i + 7] = v7;
            }
            for (int i = I4_8; i < I4; i++)
            {
                ac0[0] = i;
                act_slm[i] = (unsigned int)s_i32_ld_g((__global__ int*)gen_addr(ac0, act_i32));
            }
            for (int g = 0; g < num_groups; g++)
            {
                asc[0] = g;
                sc_slm[g] = s_f32_ld_g((__global__ float*)gen_addr(asc, act_scale));
            }
            cached_rs = rs;
        }
        const int e = cached_e;

        const int n_off = blk << 8;
        float64 tot1 = 0.0f, tot2 = 0.0f, tot3 = 0.0f, tot4 = 0.0f;
        int5 wc0 = {0, 0, blk, e, 0}, wc1 = {0, 1, blk, e, 0}, wc2 = {0, 2, blk, e, 0}, wc3 = {0, 3, blk, e, 0};

        for (int g = 0; g < num_groups; g++)
        {
            const float g_sc = sc_slm[g];
            float256 acc0 = {0}, acc1 = {0}, acc2 = {0}, acc3 = {0};
            const int w_base = g << 5;  // 32 uint32 = 128 fp8 bytes per group

            unsigned int awa = act_slm[w_base];
            minifloat256 wa0 = v_f8_ld_tnsr_b(wc0, w2_tiled, W_LD_SW, 0, 1, 0);
            minifloat256 wa1 = v_f8_ld_tnsr_b(wc1, w2_tiled, W_LD_SW, 0, 1, 0);
            minifloat256 wa2 = v_f8_ld_tnsr_b(wc2, w2_tiled, W_LD_SW, 0, 1, 0);
            minifloat256 wa3 = v_f8_ld_tnsr_b(wc3, w2_tiled, W_LD_SW, 0, 1, 0);

            unsigned int awb = act_slm[w_base + 1];
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
                awa = act_slm[w];

                wc0[1] += 4; wc1[1] += 4; wc2[1] += 4; wc3[1] += 4;
                acc0 = v_f8_mac_acc32_b(wb0, bits_to_f8(awb), acc0);
                wb0 = v_f8_ld_tnsr_b(wc0, w2_tiled, W_LD_SW, 0, 1, 0);
                acc1 = v_f8_mac_acc32_b(wb1, bits_to_f8(awb >> 8), acc1);
                wb1 = v_f8_ld_tnsr_b(wc1, w2_tiled, W_LD_SW, 0, 1, 0);
                acc2 = v_f8_mac_acc32_b(wb2, bits_to_f8(awb >> 16), acc2);
                wb2 = v_f8_ld_tnsr_b(wc2, w2_tiled, W_LD_SW, 0, 1, 0);
                acc3 = v_f8_mac_acc32_b(wb3, bits_to_f8(awb >> 24), acc3);
                wb3 = v_f8_ld_tnsr_b(wc3, w2_tiled, W_LD_SW, 0, 1, 0);
                awb = act_slm[w + 1];
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

        int5 s2c = {n_off, e, 0, 0, 0};
        tot1 *= v_f32_ld_tnsr_b(s2c, s2_packed);
        s2c[0] = n_off + 64;
        tot2 *= v_f32_ld_tnsr_b(s2c, s2_packed);
        s2c[0] = n_off + 128;
        tot3 *= v_f32_ld_tnsr_b(s2c, s2_packed);
        s2c[0] = n_off + 192;
        tot4 *= v_f32_ld_tnsr_b(s2c, s2_packed);

        const float64_pair_t p_lo = {tot1, tot2};
        const float64_pair_t p_hi = {tot3, tot4};
        const bfloat128 out_lo = v_convert_f32_to_bf16_all_b(p_lo);
        const bfloat128 out_hi = v_convert_f32_to_bf16_all_b(p_hi);

        int5 oc = {0, id, 0, 0, 0};
        v_bf16_st_tnsr(oc, out_bf16, out_lo);
        oc[0] = 128;
        v_bf16_st_tnsr(oc, out_bf16, out_hi);

        blk++;
        if (blk == nBlks)
        {
            blk = 0;
            rs++;
            slot++;
            if (slot == topK)
            {
                slot = 0;
                b++;
            }
        }
    }
}
