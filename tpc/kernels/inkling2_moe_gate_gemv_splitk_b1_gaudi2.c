// SPDX-License-Identifier: Apache-2.0
// Dedicated B=1 Split-K Exact FP32-Accumulated BF16 MoE Gate GEMV for Inkling 2 Mini on Gaudi2 TPC.
//
// Split into its own tiny ELF (~3.5 KB .text) so B=1 decode avoids cold I-cache fetch overhead from
// the batched B=2..16 loops. Dispatched automatically by Inkling2MoeGateGemvSplitkGaudi2 when B == 1.
//
// Tensors (TPC dim order, dim0 fastest):
//   x              [H, 1] bf16        (PyTorch [1, H], H % 128 == 0)
//   w_gate_packed  [128, H] bf16      (PyTorch [H, 128], row h = [e0,e64, e1,e65, ..., e63,e127])
//   partial_logits [128, S, 1] f32    (PyTorch [1, S, 128], S = H / 128)
//
// Index space: [S] (1D; 18 cores for H = 2304).

#include "inkling2_bf16_common.h"

#define SLICE_H 128
#define SLICE_H2 64
#ifndef GATE_W_LD_SW
#define GATE_W_LD_SW SW_L0CD
#endif

void main(tensor x, tensor w_gate_packed, tensor partial_logits)
{
    const int5 is_start = get_index_space_offset();
    const int5 is_end = get_index_space_size() + is_start;
    unsigned int x_slm[SLICE_H2];

    for (int s = is_start[0]; s < is_end[0]; s++)
    {
        const int h_base = s << 7;
        int5 pf = {h_base, 0, 0, 0, 0};
        prefetch(gen_addr(pf, x));
        pf[0] = h_base + 64;
        prefetch(gen_addr(pf, x));

        int5 xc0 = {h_base + 0, 0, 0, 0, 0}, xc1 = {h_base + 2, 0, 0, 0, 0};
        int5 xc2 = {h_base + 4, 0, 0, 0, 0}, xc3 = {h_base + 6, 0, 0, 0, 0};
        for (int k2 = 0; k2 < SLICE_H2; k2 += 4)
        {
            const unsigned int v0 = (unsigned int)s_i32_ld_g((__global__ int*)gen_addr(xc0, x));
            const unsigned int v1 = (unsigned int)s_i32_ld_g((__global__ int*)gen_addr(xc1, x));
            const unsigned int v2 = (unsigned int)s_i32_ld_g((__global__ int*)gen_addr(xc2, x));
            const unsigned int v3 = (unsigned int)s_i32_ld_g((__global__ int*)gen_addr(xc3, x));
            xc0[0] += 8; xc1[0] += 8; xc2[0] += 8; xc3[0] += 8;
            x_slm[k2 + 0] = v0; x_slm[k2 + 1] = v1;
            x_slm[k2 + 2] = v2; x_slm[k2 + 3] = v3;
        }

        float64 lg0_a = 0.0f, lg1_a = 0.0f, lg0_b = 0.0f, lg1_b = 0.0f;
        int5 wc0 = {0, h_base + 0, 0, 0, 0}, wc1 = {0, h_base + 1, 0, 0, 0};
        int5 wc2 = {0, h_base + 2, 0, 0, 0}, wc3 = {0, h_base + 3, 0, 0, 0};

        unsigned int pa0 = x_slm[0], pa1 = x_slm[1];
        bfloat128 wa0 = v_bf16_ld_tnsr_b(wc0, w_gate_packed, GATE_W_LD_SW, 0, 1, 0);
        bfloat128 wa1 = v_bf16_ld_tnsr_b(wc1, w_gate_packed, GATE_W_LD_SW, 0, 1, 0);
        bfloat128 wa2 = v_bf16_ld_tnsr_b(wc2, w_gate_packed, GATE_W_LD_SW, 0, 1, 0);
        bfloat128 wa3 = v_bf16_ld_tnsr_b(wc3, w_gate_packed, GATE_W_LD_SW, 0, 1, 0);
        wc0[1] += 4; wc1[1] += 4; wc2[1] += 4; wc3[1] += 4;

        unsigned int pb0 = x_slm[2], pb1 = x_slm[3];
        bfloat128 wb0 = v_bf16_ld_tnsr_b(wc0, w_gate_packed, GATE_W_LD_SW, 0, 1, 0);
        bfloat128 wb1 = v_bf16_ld_tnsr_b(wc1, w_gate_packed, GATE_W_LD_SW, 0, 1, 0);
        bfloat128 wb2 = v_bf16_ld_tnsr_b(wc2, w_gate_packed, GATE_W_LD_SW, 0, 1, 0);
        bfloat128 wb3 = v_bf16_ld_tnsr_b(wc3, w_gate_packed, GATE_W_LD_SW, 0, 1, 0);

        for (int k4 = 2; k4 < (SLICE_H >> 2); k4 += 2)
        {
            wc0[1] += 4; wc1[1] += 4; wc2[1] += 4; wc3[1] += 4;
            const float xa0 = u32_bits_to_f32(pa0 << 16);
            const float xa1 = u32_bits_to_f32(pa0 & 0xFFFF0000u);
            const float xa2 = u32_bits_to_f32(pa1 << 16);
            const float xa3 = u32_bits_to_f32(pa1 & 0xFFFF0000u);
            const float64_pair_t fa0 = v_convert_bf16_to_f32_all_b(wa0);
            wa0 = v_bf16_ld_tnsr_b(wc0, w_gate_packed, GATE_W_LD_SW, 0, 1, 0);
            const float64_pair_t fa1 = v_convert_bf16_to_f32_all_b(wa1);
            wa1 = v_bf16_ld_tnsr_b(wc1, w_gate_packed, GATE_W_LD_SW, 0, 1, 0);
            const float64_pair_t fa2 = v_convert_bf16_to_f32_all_b(wa2);
            wa2 = v_bf16_ld_tnsr_b(wc2, w_gate_packed, GATE_W_LD_SW, 0, 1, 0);
            const float64_pair_t fa3 = v_convert_bf16_to_f32_all_b(wa3);
            wa3 = v_bf16_ld_tnsr_b(wc3, w_gate_packed, GATE_W_LD_SW, 0, 1, 0);
            pa0 = x_slm[k4 * 2];
            pa1 = x_slm[k4 * 2 + 1];
            lg0_a = v_f32_mac_b(fa0.v1, xa0, lg0_a);
            lg1_a = v_f32_mac_b(fa0.v2, xa0, lg1_a);
            lg0_b = v_f32_mac_b(fa1.v1, xa1, lg0_b);
            lg1_b = v_f32_mac_b(fa1.v2, xa1, lg1_b);
            lg0_a = v_f32_mac_b(fa2.v1, xa2, lg0_a);
            lg1_a = v_f32_mac_b(fa2.v2, xa2, lg1_a);
            lg0_b = v_f32_mac_b(fa3.v1, xa3, lg0_b);
            lg1_b = v_f32_mac_b(fa3.v2, xa3, lg1_b);

            wc0[1] += 4; wc1[1] += 4; wc2[1] += 4; wc3[1] += 4;
            const float xb0 = u32_bits_to_f32(pb0 << 16);
            const float xb1 = u32_bits_to_f32(pb0 & 0xFFFF0000u);
            const float xb2 = u32_bits_to_f32(pb1 << 16);
            const float xb3 = u32_bits_to_f32(pb1 & 0xFFFF0000u);
            const float64_pair_t fb0 = v_convert_bf16_to_f32_all_b(wb0);
            wb0 = v_bf16_ld_tnsr_b(wc0, w_gate_packed, GATE_W_LD_SW, 0, 1, 0);
            const float64_pair_t fb1 = v_convert_bf16_to_f32_all_b(wb1);
            wb1 = v_bf16_ld_tnsr_b(wc1, w_gate_packed, GATE_W_LD_SW, 0, 1, 0);
            const float64_pair_t fb2 = v_convert_bf16_to_f32_all_b(wb2);
            wb2 = v_bf16_ld_tnsr_b(wc2, w_gate_packed, GATE_W_LD_SW, 0, 1, 0);
            const float64_pair_t fb3 = v_convert_bf16_to_f32_all_b(wb3);
            wb3 = v_bf16_ld_tnsr_b(wc3, w_gate_packed, GATE_W_LD_SW, 0, 1, 0);
            pb0 = x_slm[k4 * 2 + 2];
            pb1 = x_slm[k4 * 2 + 3];
            lg0_a = v_f32_mac_b(fb0.v1, xb0, lg0_a);
            lg1_a = v_f32_mac_b(fb0.v2, xb0, lg1_a);
            lg0_b = v_f32_mac_b(fb1.v1, xb1, lg0_b);
            lg1_b = v_f32_mac_b(fb1.v2, xb1, lg1_b);
            lg0_a = v_f32_mac_b(fb2.v1, xb2, lg0_a);
            lg1_a = v_f32_mac_b(fb2.v2, xb2, lg1_a);
            lg0_b = v_f32_mac_b(fb3.v1, xb3, lg0_b);
            lg1_b = v_f32_mac_b(fb3.v2, xb3, lg1_b);
        }

        {
            const float xa0 = u32_bits_to_f32(pa0 << 16);
            const float xa1 = u32_bits_to_f32(pa0 & 0xFFFF0000u);
            const float xa2 = u32_bits_to_f32(pa1 << 16);
            const float xa3 = u32_bits_to_f32(pa1 & 0xFFFF0000u);
            const float64_pair_t fa0 = v_convert_bf16_to_f32_all_b(wa0);
            const float64_pair_t fa1 = v_convert_bf16_to_f32_all_b(wa1);
            const float64_pair_t fa2 = v_convert_bf16_to_f32_all_b(wa2);
            const float64_pair_t fa3 = v_convert_bf16_to_f32_all_b(wa3);
            lg0_a = v_f32_mac_b(fa0.v1, xa0, lg0_a);
            lg1_a = v_f32_mac_b(fa0.v2, xa0, lg1_a);
            lg0_b = v_f32_mac_b(fa1.v1, xa1, lg0_b);
            lg1_b = v_f32_mac_b(fa1.v2, xa1, lg1_b);
            lg0_a = v_f32_mac_b(fa2.v1, xa2, lg0_a);
            lg1_a = v_f32_mac_b(fa2.v2, xa2, lg1_a);
            lg0_b = v_f32_mac_b(fa3.v1, xa3, lg0_b);
            lg1_b = v_f32_mac_b(fa3.v2, xa3, lg1_b);

            const float xb0 = u32_bits_to_f32(pb0 << 16);
            const float xb1 = u32_bits_to_f32(pb0 & 0xFFFF0000u);
            const float xb2 = u32_bits_to_f32(pb1 << 16);
            const float xb3 = u32_bits_to_f32(pb1 & 0xFFFF0000u);
            const float64_pair_t fb0 = v_convert_bf16_to_f32_all_b(wb0);
            const float64_pair_t fb1 = v_convert_bf16_to_f32_all_b(wb1);
            const float64_pair_t fb2 = v_convert_bf16_to_f32_all_b(wb2);
            const float64_pair_t fb3 = v_convert_bf16_to_f32_all_b(wb3);
            lg0_a = v_f32_mac_b(fb0.v1, xb0, lg0_a);
            lg1_a = v_f32_mac_b(fb0.v2, xb0, lg1_a);
            lg0_b = v_f32_mac_b(fb1.v1, xb1, lg0_b);
            lg1_b = v_f32_mac_b(fb1.v2, xb1, lg1_b);
            lg0_a = v_f32_mac_b(fb2.v1, xb2, lg0_a);
            lg1_a = v_f32_mac_b(fb2.v2, xb2, lg1_a);
            lg0_b = v_f32_mac_b(fb3.v1, xb3, lg0_b);
            lg1_b = v_f32_mac_b(fb3.v2, xb3, lg1_b);
        }

        int5 oc0 = {0, s, 0, 0, 0}, oc1 = {64, s, 0, 0, 0};
        v_f32_st_tnsr(oc0, partial_logits, lg0_a + lg0_b);
        v_f32_st_tnsr(oc1, partial_logits, lg1_a + lg1_b);
    }
}
