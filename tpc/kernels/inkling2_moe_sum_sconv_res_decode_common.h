// SPDX-License-Identifier: Apache-2.0
// Fused MoE top-k reduction + direct conv_cache static short-conv (W=4) + residual update
// (+ optional PACK_FULL_STATE=1 full 4-stream conv_state output eliminating torch.cat DmaMemcpy)
// for Inkling 2 Mini decode on Gaudi2 TPC.
#pragma once

#include "inkling2_bf16_common.h"

#ifndef PACK_FULL_STATE
#define PACK_FULL_STATE 0
#endif

#if PACK_FULL_STATE
void main(tensor moe_partial, tensor conv_state, tensor load_slots, tensor weight_t,
          tensor partial_res, tensor attn_out, tensor new_kv_state, tensor new_attn_state,
          tensor new_partial, tensor new_full_state,
          float out_scale, int is_first, int stream_offset)
#else
void main(tensor moe_partial, tensor conv_state, tensor load_slots, tensor weight_t,
          tensor partial_res, tensor attn_out,
          tensor new_partial, tensor new_mlp_state,
          float out_scale, int is_first, int stream_offset)
#endif
{
    const int topK = get_dim_size(moe_partial, 1);
#if PACK_FULL_STATE
    const int H = get_dim_size(partial_res, 0);
    const int kvWidth = get_dim_size(new_kv_state, 0);
#endif
    const int5 is_start = get_index_space_offset();
    const int5 is_end = get_index_space_size() + is_start;

    for (int b = is_start[1]; b < is_end[1]; b++)
    {
        int5 lsc = {b, 0, 0, 0, 0};
        const int slot = s_i32_ld_g((__global__ int*)gen_addr(lsc, load_slots));

        for (int blk = is_start[0]; blk < is_end[0]; blk++)
        {
            const int d = blk << 6;
            const int sd = stream_offset + d;

#if PACK_FULL_STATE
            if (d < kvWidth)
            {
                int5 kvc0 = {d, 0, b, 0, 0}, kvc1 = {d, 1, b, 0, 0}, kvc2 = {d, 2, b, 0, 0};
                st_f32_as_bf16(kvc0, new_full_state, ld_bf16_as_f32(kvc0, new_kv_state));
                st_f32_as_bf16(kvc1, new_full_state, ld_bf16_as_f32(kvc1, new_kv_state));
                st_f32_as_bf16(kvc2, new_full_state, ld_bf16_as_f32(kvc2, new_kv_state));
            }
            if (d + H < kvWidth)
            {
                const int d2 = d + H;
                int5 kvc0 = {d2, 0, b, 0, 0}, kvc1 = {d2, 1, b, 0, 0}, kvc2 = {d2, 2, b, 0, 0};
                st_f32_as_bf16(kvc0, new_full_state, ld_bf16_as_f32(kvc0, new_kv_state));
                st_f32_as_bf16(kvc1, new_full_state, ld_bf16_as_f32(kvc1, new_kv_state));
                st_f32_as_bf16(kvc2, new_full_state, ld_bf16_as_f32(kvc2, new_kv_state));
            }
            {
                int5 ac0 = {d, 0, b, 0, 0}, ac1 = {d, 1, b, 0, 0}, ac2 = {d, 2, b, 0, 0};
                int5 fc0 = {kvWidth + d, 0, b, 0, 0};
                int5 fc1 = {kvWidth + d, 1, b, 0, 0};
                int5 fc2 = {kvWidth + d, 2, b, 0, 0};
                st_f32_as_bf16(fc0, new_full_state, ld_bf16_as_f32(ac0, new_attn_state));
                st_f32_as_bf16(fc1, new_full_state, ld_bf16_as_f32(ac1, new_attn_state));
                st_f32_as_bf16(fc2, new_full_state, ld_bf16_as_f32(ac2, new_attn_state));
            }
#endif

            // 1. Reduce MoE partial [topK] in fp32 and round once to bf16 (matches moe.py).
            float64 sum_f = 0.0f;
            int5 pc = {d, 0, b, 0, 0};
            for (int k = 0; k < topK; k++)
            {
                pc[1] = k;
                sum_f += ld_bf16_as_f32(pc, moe_partial);
            }
            const float64 xv = round_f32_via_bf16(sum_f);

            // 2. Direct conv_state read + static short-conv (W=4) + state shift.
            int5 sc0 = {sd, 0, slot, 0, 0};
            int5 sc1 = {sd, 1, slot, 0, 0};
            int5 sc2 = {sd, 2, slot, 0, 0};
            int5 wc0 = {d, 0, 0, 0, 0};
            int5 wc1 = {d, 1, 0, 0, 0};
            int5 wc2 = {d, 2, 0, 0, 0};
            int5 wc3 = {d, 3, 0, 0, 0};

            const float64 s0 = ld_bf16_as_f32(sc0, conv_state);
            const float64 s1 = ld_bf16_as_f32(sc1, conv_state);
            const float64 s2 = ld_bf16_as_f32(sc2, conv_state);

            float64 acc = xv;
            acc = v_f32_mac_b(s0, ld_bf16_as_f32(wc0, weight_t), acc);
            acc = v_f32_mac_b(s1, ld_bf16_as_f32(wc1, weight_t), acc);
            acc = v_f32_mac_b(s2, ld_bf16_as_f32(wc2, weight_t), acc);
            acc = v_f32_mac_b(xv, ld_bf16_as_f32(wc3, weight_t), acc);
            const float64 mlp_out = round_f32_via_bf16(acc * out_scale);

#if PACK_FULL_STATE
            int5 nc0 = {kvWidth + H + d, 0, b, 0, 0};
            int5 nc1 = {kvWidth + H + d, 1, b, 0, 0};
            int5 nc2 = {kvWidth + H + d, 2, b, 0, 0};
            st_f32_as_bf16(nc0, new_full_state, s1);
            st_f32_as_bf16(nc1, new_full_state, s2);
            st_f32_as_bf16(nc2, new_full_state, xv);
#else
            int5 nc0 = {d, 0, b, 0, 0};
            int5 nc1 = {d, 1, b, 0, 0};
            int5 nc2 = {d, 2, b, 0, 0};
            st_f32_as_bf16(nc0, new_mlp_state, s1);
            st_f32_as_bf16(nc1, new_mlp_state, s2);
            st_f32_as_bf16(nc2, new_mlp_state, xv);
#endif

            // 3. Residual block partial update.
            int5 rc = {d, b, 0, 0, 0};
            float64 res = ld_bf16_as_f32(rc, attn_out) + mlp_out;
            if (!is_first)
                res += ld_bf16_as_f32(rc, partial_res);
            st_f32_as_bf16(rc, new_partial, res);
        }
    }
}
