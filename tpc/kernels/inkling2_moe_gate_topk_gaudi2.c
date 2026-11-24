// SPDX-License-Identifier: Apache-2.0
// Fused Inkling2 MoE gate Split-K reduction + routing epilogue on Gaudi2 TPC:
//   logits = sum_s partial_logits[b, s, :]    (S = 1 for prefill MME logits, S = H/128 for decode split-K)
//   sig = sigmoid(logits)
//   selection = sig + bias
//   topk_ids = topk(selection, topK, largest=True, sorted=True)  (tie-break by lowest expert id)
//   topk_weights = (sig[topk_ids] / sum(sig[topk_ids])) * route_scale
//
// Uses ONLY ONE SFU LUT (exp_f32=0x100) + zero-LUT rcp_pos_nolut + single-tree 6-step butterfly
// carrying (sel, idx, sig) in parallel (~3x fewer cycles than 3 separate reductions!).
//
// Tensors (TPC dim order, dim0 fastest):
//   logits       [E, S, B] f32    (PyTorch [B, S, E], E in {64, 128, 192, 256}, S >= 1)
//   bias         [E] f32          (PyTorch [E])
//   topk_ids     [topK, B] i32    (PyTorch [B, topK], 1 <= topK <= 16)
//   topk_weights [topK, B] f32    (PyTorch [B, topK])
//
// Index space: [B] (1D).

#include "inkling2_bf16_common.h"

void main(tensor logits, tensor bias, tensor topk_ids, tensor topk_weights,
          int topK, float route_scale)
{
    const int E = get_dim_size(logits, 0);
    const int S = get_dim_size(logits, 1);
    const int nVec = E >> 6;  // 1..4 (E = 64..256)
    const int5 is_start = get_index_space_offset();
    const int5 is_end = get_index_space_size() + is_start;

    const uint64 lid_u = read_lane_id_4b_b();
    const int64 lane_i32 = (int64)lid_u;
    const bool64 valid_k = v_i32_cmp_less_b(lane_i32, topK);
    const tpc_xor_masks_t xmasks = make_xor_masks_nolut(lid_u);

    // General path for E in {64, 192, 256} (E == 128 uses inkling2_moe_gate_topk_e128_gaudi2).
    float64 bs[4];
    for (int v = 0; v < nVec; v++)
    {
        int5 bc = {v << 6, 0, 0, 0, 0};
        bs[v] = v_f32_ld_tnsr_b(bc, bias);
    }

    for (int b = is_start[0]; b < is_end[0]; b++)
    {
        float64 sig[4], sel[4];
        for (int v = 0; v < nVec; v++)
        {
            float64 lg = 0.0f;
            int5 lc = {v << 6, 0, b, 0, 0};
            for (int s = 0; s < S; s++)
            {
                lc[1] = s;
                lg += v_f32_ld_tnsr_b(lc, logits);
            }
            sig[v] = sigmoid_nolut(lg);
            sel[v] = sig[v] + bs[v];
        }

        int64 out_ids_v = 0;
        float64 sel_sig_v = 0.0f;

        for (int k = 0; k < topK; k++)
        {
            tpc_argmax_triple_t best = {sel[0], lane_i32, sig[0]};
            for (int v = 1; v < nVec; v++)
            {
                const bool64 take = v_f32_cmp_grt_b(sel[v], best.val);
                best.val = v_f32_mov_vb(sel[v], 0, best.val, take, 0);
                best.idx = v_i32_mov_vb(lane_i32 + (v << 6), 0, best.idx, take, 0);
                best.sig = v_f32_mov_vb(sig[v], 0, best.sig, take, 0);
            }
            best = reduce_argmax_sig_nolut(best, xmasks);

            for (int v = 0; v < nVec; v++)
                sel[v] = v_f32_mov_vb(-3.4e38f, 0, sel[v], v_i32_cmp_eq_b(lane_i32 + (v << 6), best.idx), 0);

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
