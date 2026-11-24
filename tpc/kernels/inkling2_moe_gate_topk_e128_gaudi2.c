// SPDX-License-Identifier: Apache-2.0
// Dedicated compact E=128 Inkling2 MoE gate Split-K reduction + routing epilogue on Gaudi2 TPC.
//
// Split into its own tiny ELF (~6 KB .text) with a single non-unrolled sigmoid_nolut loop so
// cold I-cache fetch across 48 decode visits drops ~4x vs the combined ELF. Dispatched
// automatically by Inkling2MoeGateTopkGaudi2 when E == 128.
//
// Tensors (TPC dim order, dim0 fastest):
//   logits       [128, S, B] f32    (PyTorch [B, S, 128], S >= 1)
//   bias         [128] f32          (PyTorch [128])
//   topk_ids     [topK, B] i32      (PyTorch [B, topK], 1 <= topK <= 16)
//   topk_weights [topK, B] f32      (PyTorch [B, topK])
//
// Index space: [B] (1D).

#include "inkling2_bf16_common.h"

static __local__ float64 g_lg[2];
static __local__ float64 g_sig[2];

void main(tensor logits, tensor bias, tensor topk_ids, tensor topk_weights,
          int topK, float route_scale)
{
    const int S = get_dim_size(logits, 1);
    const int5 is_start = get_index_space_offset();
    const int5 is_end = get_index_space_size() + is_start;

    const uint64 lid_u = read_lane_id_4b_b();
    const int64 lane_i32 = (int64)lid_u;
    const int64 lane_hi = lane_i32 + 64;
    const bool64 valid_k = v_i32_cmp_less_b(lane_i32, topK);
    const tpc_xor_masks_t xmasks = make_xor_masks_nolut(lid_u);

    int5 bc0 = {0, 0, 0, 0, 0}, bc1 = {64, 0, 0, 0, 0};
    const float64 bs0 = v_f32_ld_tnsr_b(bc0, bias);
    const float64 bs1 = v_f32_ld_tnsr_b(bc1, bias);

    for (int b = is_start[0]; b < is_end[0]; b++)
    {
        float64 lg0 = 0.0f, lg1 = 0.0f;
        float64 lg0_b = 0.0f, lg1_b = 0.0f;
        int5 lc0a = {0, 0, b, 0, 0}, lc1a = {64, 0, b, 0, 0};
        int5 lc0b = {0, 1, b, 0, 0}, lc1b = {64, 1, b, 0, 0};
        const int S2 = S & ~1;
        for (int s = 0; s < S2; s += 2)
        {
            lg0 += v_f32_ld_tnsr_b(lc0a, logits);
            lg1 += v_f32_ld_tnsr_b(lc1a, logits);
            lg0_b += v_f32_ld_tnsr_b(lc0b, logits);
            lg1_b += v_f32_ld_tnsr_b(lc1b, logits);
            lc0a[1] += 2; lc1a[1] += 2;
            lc0b[1] += 2; lc1b[1] += 2;
        }
        lg0 += lg0_b;
        lg1 += lg1_b;
        if (S & 1)
        {
            lc0a[1] = S2;
            lc1a[1] = S2;
            lg0 += v_f32_ld_tnsr_b(lc0a, logits);
            lg1 += v_f32_ld_tnsr_b(lc1a, logits);
        }

        g_lg[0] = lg0;
        g_lg[1] = lg1;
        #pragma clang loop unroll(disable)
        for (int v = 0; v < 2; v++)
            g_sig[v] = sigmoid_nolut(g_lg[v]);

        const float64 sig0 = g_sig[0];
        const float64 sig1 = g_sig[1];
        float64 sel0 = sig0 + bs0;
        float64 sel1 = sig1 + bs1;

        int64 out_ids_v = 0;
        float64 sel_sig_v = 0.0f;

        #pragma clang loop unroll(disable)
        for (int k = 0; k < topK; k++)
        {
            const bool64 take1 = v_f32_cmp_grt_b(sel1, sel0);
            tpc_argmax_triple_t best = {
                v_f32_mov_vb(sel1, 0, sel0, take1, 0),
                v_i32_mov_vb(lane_hi, 0, lane_i32, take1, 0),
                v_f32_mov_vb(sig1, 0, sig0, take1, 0),
            };
            best = reduce_argmax_sig_nolut(best, xmasks);

            const bool64 m0 = v_i32_cmp_eq_b(lane_i32, best.idx);
            const bool64 m1 = v_i32_cmp_eq_b(lane_hi, best.idx);
            sel0 = v_f32_mov_vb(-3.4e38f, 0, sel0, m0, 0);
            sel1 = v_f32_mov_vb(-3.4e38f, 0, sel1, m1, 0);

            const bool64 is_k = v_i32_cmp_eq_b(lane_i32, k);
            out_ids_v = v_i32_mov_vb(best.idx, 0, out_ids_v, is_k, 0);
            sel_sig_v = v_f32_mov_vb(best.sig, 0, sel_sig_v, is_k, 0);
        }

        const float64 masked_sig = v_f32_mov_vb(0.0f, 0, sel_sig_v, valid_k, 1);
        const float64 sum_sig = v_f32_max_b(reduce_add_fast_nolut(masked_sig, xmasks), 1e-30f);
        const float64 weights_v = masked_sig * (rcp_pos_nolut(sum_sig) * route_scale);

        int5 oc = {0, b, 0, 0, 0};
        v_i32_st_tnsr_partial(oc, topk_ids, out_ids_v, topK - 1, 0, 0, 1, 0);
        v_f32_st_tnsr_partial(oc, topk_weights, weights_v, topK - 1, 0, 0, 1, 0);
    }
}
