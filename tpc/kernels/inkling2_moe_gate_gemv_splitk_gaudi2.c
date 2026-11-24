// SPDX-License-Identifier: Apache-2.0
// Multi-core Split-K Exact FP32-Accumulated BF16 MoE Gate GEMV for Inkling 2 Mini on Gaudi2 TPC (B >= 2).
//
// Wave 9 upgrade:
//   - Compact single-stage loops (#pragma clang loop unroll(disable)) eliminate 2-stage peeled tail
//     bloat, shrinking .text from 39 KB down to <= 15 KB (100% resident in Gaudi2's 16 KB L1 I-cache).
//   - B == 1 is dispatched by glue code to the dedicated tiny inkling2_moe_gate_gemv_splitk_b1 ELF.
//   - Parallelizes across S = H / 128 (= 18 for H = 2304) TPC cores, reading w_gate_packed ONCE from
//     HBM across up to 8 tokens in VRF with ZERO VLM spills and ZERO SFU LUTs.
//
// Tensors (TPC dim order, dim0 fastest):
//   x              [H, B] bf16        (PyTorch [B, H], H % 128 == 0, B <= 16)
//   w_gate_packed  [128, H] bf16      (PyTorch [H, 128], row h = [e0,e64, e1,e65, ..., e63,e127])
//   partial_logits [128, S, B] f32    (PyTorch [B, S, 128], S = H / 128)
//
// Index space: [S] (1D; 18 cores for H = 2304).

#include "inkling2_bf16_common.h"

#define TILE_B 8
#define SLICE_H2 64
#ifndef GATE_W_LD_SW
#define GATE_W_LD_SW SW_L0CD
#endif

void main(tensor x, tensor w_gate_packed, tensor partial_logits)
{
    const int B = get_dim_size(x, 1);
    const int5 is_start = get_index_space_offset();
    const int5 is_end = get_index_space_size() + is_start;

    unsigned int x_slm[TILE_B][SLICE_H2];
    #pragma clang loop unroll(disable)
    for (int s = is_start[0]; s < is_end[0]; s++)
    {
        const int h_base = s << 7;
        #pragma clang loop unroll(disable)
        for (int b0 = 0; b0 < B; b0 += TILE_B)
        {
            const int nb = (B - b0 < TILE_B) ? (B - b0) : TILE_B;
            #pragma clang loop unroll(disable)
            for (int j = 0; j < nb; j++)
            {
                const int b = b0 + j;
                int5 pf = {h_base, b, 0, 0, 0};
                prefetch(gen_addr(pf, x));
                pf[0] = h_base + 64;
                prefetch(gen_addr(pf, x));
            }
            #pragma clang loop unroll(disable)
            for (int j = 0; j < nb; j++)
            {
                const int b = b0 + j;
                int5 xc0 = {h_base + 0, b, 0, 0, 0}, xc1 = {h_base + 2, b, 0, 0, 0};
                int5 xc2 = {h_base + 4, b, 0, 0, 0}, xc3 = {h_base + 6, b, 0, 0, 0};
                #pragma clang loop unroll(disable)
                for (int k2 = 0; k2 < SLICE_H2; k2 += 4)
                {
                    const unsigned int v0 = (unsigned int)s_i32_ld_g((__global__ int*)gen_addr(xc0, x));
                    const unsigned int v1 = (unsigned int)s_i32_ld_g((__global__ int*)gen_addr(xc1, x));
                    const unsigned int v2 = (unsigned int)s_i32_ld_g((__global__ int*)gen_addr(xc2, x));
                    const unsigned int v3 = (unsigned int)s_i32_ld_g((__global__ int*)gen_addr(xc3, x));
                    xc0[0] += 8; xc1[0] += 8; xc2[0] += 8; xc3[0] += 8;
                    x_slm[j][k2 + 0] = v0; x_slm[j][k2 + 1] = v1;
                    x_slm[j][k2 + 2] = v2; x_slm[j][k2 + 3] = v3;
                }
            }
            #pragma clang loop unroll(disable)
            for (int j = nb; j < TILE_B; j++)
                #pragma clang loop unroll(disable)
                for (int k2 = 0; k2 < SLICE_H2; k2++)
                    x_slm[j][k2] = 0u;

            float64 a0_0 = 0.0f, a1_0 = 0.0f, a0_1 = 0.0f, a1_1 = 0.0f;
            float64 a0_2 = 0.0f, a1_2 = 0.0f, a0_3 = 0.0f, a1_3 = 0.0f;
            int5 wc0 = {0, h_base + 0, 0, 0, 0}, wc1 = {0, h_base + 1, 0, 0, 0};

            if (nb <= 4)
            {
                #pragma clang loop unroll(disable)
                for (int k2 = 0; k2 < SLICE_H2; k2++)
                {
                    const bfloat128 w0 = v_bf16_ld_tnsr_b(wc0, w_gate_packed, GATE_W_LD_SW, 0, 1, 0);
                    const bfloat128 w1 = v_bf16_ld_tnsr_b(wc1, w_gate_packed, GATE_W_LD_SW, 0, 1, 0);
                    wc0[1] += 2; wc1[1] += 2;
                    const float64_pair_t f0 = v_convert_bf16_to_f32_all_b(w0);
                    const float64_pair_t f1 = v_convert_bf16_to_f32_all_b(w1);
                    const unsigned int p0 = x_slm[0][k2], p1 = x_slm[1][k2];
                    const unsigned int p2 = x_slm[2][k2], p3 = x_slm[3][k2];
                    a0_0 = v_f32_mac_b(f0.v1, u32_bits_to_f32(p0 << 16), a0_0);
                    a1_0 = v_f32_mac_b(f0.v2, u32_bits_to_f32(p0 << 16), a1_0);
                    a0_1 = v_f32_mac_b(f0.v1, u32_bits_to_f32(p1 << 16), a0_1);
                    a1_1 = v_f32_mac_b(f0.v2, u32_bits_to_f32(p1 << 16), a1_1);
                    a0_2 = v_f32_mac_b(f0.v1, u32_bits_to_f32(p2 << 16), a0_2);
                    a1_2 = v_f32_mac_b(f0.v2, u32_bits_to_f32(p2 << 16), a1_2);
                    a0_3 = v_f32_mac_b(f0.v1, u32_bits_to_f32(p3 << 16), a0_3);
                    a1_3 = v_f32_mac_b(f0.v2, u32_bits_to_f32(p3 << 16), a1_3);
                    a0_0 = v_f32_mac_b(f1.v1, u32_bits_to_f32(p0 & 0xFFFF0000u), a0_0);
                    a1_0 = v_f32_mac_b(f1.v2, u32_bits_to_f32(p0 & 0xFFFF0000u), a1_0);
                    a0_1 = v_f32_mac_b(f1.v1, u32_bits_to_f32(p1 & 0xFFFF0000u), a0_1);
                    a1_1 = v_f32_mac_b(f1.v2, u32_bits_to_f32(p1 & 0xFFFF0000u), a1_1);
                    a0_2 = v_f32_mac_b(f1.v1, u32_bits_to_f32(p2 & 0xFFFF0000u), a0_2);
                    a1_2 = v_f32_mac_b(f1.v2, u32_bits_to_f32(p2 & 0xFFFF0000u), a1_2);
                    a0_3 = v_f32_mac_b(f1.v1, u32_bits_to_f32(p3 & 0xFFFF0000u), a0_3);
                    a1_3 = v_f32_mac_b(f1.v2, u32_bits_to_f32(p3 & 0xFFFF0000u), a1_3);
                }
                int5 oc0 = {0, s, b0, 0, 0}, oc1 = {64, s, b0, 0, 0};
                v_f32_st_tnsr(oc0, partial_logits, a0_0); v_f32_st_tnsr(oc1, partial_logits, a1_0);
                if (nb > 1) { oc0[2] = b0 + 1; oc1[2] = b0 + 1; v_f32_st_tnsr(oc0, partial_logits, a0_1); v_f32_st_tnsr(oc1, partial_logits, a1_1); }
                if (nb > 2) { oc0[2] = b0 + 2; oc1[2] = b0 + 2; v_f32_st_tnsr(oc0, partial_logits, a0_2); v_f32_st_tnsr(oc1, partial_logits, a1_2); }
                if (nb > 3) { oc0[2] = b0 + 3; oc1[2] = b0 + 3; v_f32_st_tnsr(oc0, partial_logits, a0_3); v_f32_st_tnsr(oc1, partial_logits, a1_3); }
            }
            else
            {
                float64 a0_4 = 0.0f, a1_4 = 0.0f, a0_5 = 0.0f, a1_5 = 0.0f;
                float64 a0_6 = 0.0f, a1_6 = 0.0f, a0_7 = 0.0f, a1_7 = 0.0f;
                #pragma clang loop unroll(disable)
                for (int k2 = 0; k2 < SLICE_H2; k2++)
                {
                    const bfloat128 w0 = v_bf16_ld_tnsr_b(wc0, w_gate_packed, GATE_W_LD_SW, 0, 1, 0);
                    const bfloat128 w1 = v_bf16_ld_tnsr_b(wc1, w_gate_packed, GATE_W_LD_SW, 0, 1, 0);
                    wc0[1] += 2; wc1[1] += 2;
                    const float64_pair_t f0 = v_convert_bf16_to_f32_all_b(w0);
                    const float64_pair_t f1 = v_convert_bf16_to_f32_all_b(w1);
                    const unsigned int p0 = x_slm[0][k2], p1 = x_slm[1][k2], p2 = x_slm[2][k2], p3 = x_slm[3][k2];
                    const unsigned int p4 = x_slm[4][k2], p5 = x_slm[5][k2], p6 = x_slm[6][k2], p7 = x_slm[7][k2];
                    a0_0 = v_f32_mac_b(f0.v1, u32_bits_to_f32(p0 << 16), a0_0); a1_0 = v_f32_mac_b(f0.v2, u32_bits_to_f32(p0 << 16), a1_0);
                    a0_1 = v_f32_mac_b(f0.v1, u32_bits_to_f32(p1 << 16), a0_1); a1_1 = v_f32_mac_b(f0.v2, u32_bits_to_f32(p1 << 16), a1_1);
                    a0_2 = v_f32_mac_b(f0.v1, u32_bits_to_f32(p2 << 16), a0_2); a1_2 = v_f32_mac_b(f0.v2, u32_bits_to_f32(p2 << 16), a1_2);
                    a0_3 = v_f32_mac_b(f0.v1, u32_bits_to_f32(p3 << 16), a0_3); a1_3 = v_f32_mac_b(f0.v2, u32_bits_to_f32(p3 << 16), a1_3);
                    a0_4 = v_f32_mac_b(f0.v1, u32_bits_to_f32(p4 << 16), a0_4); a1_4 = v_f32_mac_b(f0.v2, u32_bits_to_f32(p4 << 16), a1_4);
                    a0_5 = v_f32_mac_b(f0.v1, u32_bits_to_f32(p5 << 16), a0_5); a1_5 = v_f32_mac_b(f0.v2, u32_bits_to_f32(p5 << 16), a1_5);
                    a0_6 = v_f32_mac_b(f0.v1, u32_bits_to_f32(p6 << 16), a0_6); a1_6 = v_f32_mac_b(f0.v2, u32_bits_to_f32(p6 << 16), a1_6);
                    a0_7 = v_f32_mac_b(f0.v1, u32_bits_to_f32(p7 << 16), a0_7); a1_7 = v_f32_mac_b(f0.v2, u32_bits_to_f32(p7 << 16), a1_7);
                    a0_0 = v_f32_mac_b(f1.v1, u32_bits_to_f32(p0 & 0xFFFF0000u), a0_0); a1_0 = v_f32_mac_b(f1.v2, u32_bits_to_f32(p0 & 0xFFFF0000u), a1_0);
                    a0_1 = v_f32_mac_b(f1.v1, u32_bits_to_f32(p1 & 0xFFFF0000u), a0_1); a1_1 = v_f32_mac_b(f1.v2, u32_bits_to_f32(p1 & 0xFFFF0000u), a1_1);
                    a0_2 = v_f32_mac_b(f1.v1, u32_bits_to_f32(p2 & 0xFFFF0000u), a0_2); a1_2 = v_f32_mac_b(f1.v2, u32_bits_to_f32(p2 & 0xFFFF0000u), a1_2);
                    a0_3 = v_f32_mac_b(f1.v1, u32_bits_to_f32(p3 & 0xFFFF0000u), a0_3); a1_3 = v_f32_mac_b(f1.v2, u32_bits_to_f32(p3 & 0xFFFF0000u), a1_3);
                    a0_4 = v_f32_mac_b(f1.v1, u32_bits_to_f32(p4 & 0xFFFF0000u), a0_4); a1_4 = v_f32_mac_b(f1.v2, u32_bits_to_f32(p4 & 0xFFFF0000u), a1_4);
                    a0_5 = v_f32_mac_b(f1.v1, u32_bits_to_f32(p5 & 0xFFFF0000u), a0_5); a1_5 = v_f32_mac_b(f1.v2, u32_bits_to_f32(p5 & 0xFFFF0000u), a1_5);
                    a0_6 = v_f32_mac_b(f1.v1, u32_bits_to_f32(p6 & 0xFFFF0000u), a0_6); a1_6 = v_f32_mac_b(f1.v2, u32_bits_to_f32(p6 & 0xFFFF0000u), a1_6);
                    a0_7 = v_f32_mac_b(f1.v1, u32_bits_to_f32(p7 & 0xFFFF0000u), a0_7); a1_7 = v_f32_mac_b(f1.v2, u32_bits_to_f32(p7 & 0xFFFF0000u), a1_7);
                }
                int5 oc0 = {0, s, b0, 0, 0}, oc1 = {64, s, b0, 0, 0};
                v_f32_st_tnsr(oc0, partial_logits, a0_0); v_f32_st_tnsr(oc1, partial_logits, a1_0);
                oc0[2] = b0 + 1; oc1[2] = b0 + 1; v_f32_st_tnsr(oc0, partial_logits, a0_1); v_f32_st_tnsr(oc1, partial_logits, a1_1);
                oc0[2] = b0 + 2; oc1[2] = b0 + 2; v_f32_st_tnsr(oc0, partial_logits, a0_2); v_f32_st_tnsr(oc1, partial_logits, a1_2);
                oc0[2] = b0 + 3; oc1[2] = b0 + 3; v_f32_st_tnsr(oc0, partial_logits, a0_3); v_f32_st_tnsr(oc1, partial_logits, a1_3);
                oc0[2] = b0 + 4; oc1[2] = b0 + 4; v_f32_st_tnsr(oc0, partial_logits, a0_4); v_f32_st_tnsr(oc1, partial_logits, a1_4);
                if (nb > 5) { oc0[2] = b0 + 5; oc1[2] = b0 + 5; v_f32_st_tnsr(oc0, partial_logits, a0_5); v_f32_st_tnsr(oc1, partial_logits, a1_5); }
                if (nb > 6) { oc0[2] = b0 + 6; oc1[2] = b0 + 6; v_f32_st_tnsr(oc0, partial_logits, a0_6); v_f32_st_tnsr(oc1, partial_logits, a1_6); }
                if (nb > 7) { oc0[2] = b0 + 7; oc1[2] = b0 + 7; v_f32_st_tnsr(oc0, partial_logits, a0_7); v_f32_st_tnsr(oc1, partial_logits, a1_7); }
            }
        }
    }
}
