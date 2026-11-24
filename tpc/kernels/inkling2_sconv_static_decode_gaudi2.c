// SPDX-License-Identifier: Apache-2.0
// Fused static short-convolution decode (W=4) reading prev_state directly from conv_state[load_slots[b]]
// at channel offset `stream_offset` for Inkling 2 Mini attn_sconv / mlp_sconv on Gaudi2 TPC.
//
// Tensors (TPC dim order, dim0 fastest):
//   x          [C, B] bf16                     (PyTorch [B, C])
//   conv_state [total_dim, 3, num_slots] bf16  (PyTorch [num_slots, 3, total_dim])
//   load_slots [B] i32                         (PyTorch [B])
//   weight_t   [C, 4] bf16                     (PyTorch [4, C], oldest tap at index 0, newest at index 3)
//   y          [C, B] bf16                     (PyTorch [B, C])
//   new_state  [C, 3, B] bf16                  (PyTorch [B, 3, C])
//
// Index space: [C / 64, B] (2D).

#include "inkling2_bf16_common.h"

void main(tensor x, tensor conv_state, tensor load_slots, tensor weight_t,
          tensor y, tensor new_state, float out_scale, int stream_offset)
{
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
            int5 xc = {d, b, 0, 0, 0};
            int5 sc0 = {sd, 0, slot, 0, 0};
            int5 sc1 = {sd, 1, slot, 0, 0};
            int5 sc2 = {sd, 2, slot, 0, 0};
            int5 nc0 = {d, 0, b, 0, 0};
            int5 nc1 = {d, 1, b, 0, 0};
            int5 nc2 = {d, 2, b, 0, 0};
            int5 wc0 = {d, 0, 0, 0, 0};
            int5 wc1 = {d, 1, 0, 0, 0};
            int5 wc2 = {d, 2, 0, 0, 0};
            int5 wc3 = {d, 3, 0, 0, 0};

            const float64 s0 = ld_bf16_as_f32(sc0, conv_state);
            const float64 s1 = ld_bf16_as_f32(sc1, conv_state);
            const float64 s2 = ld_bf16_as_f32(sc2, conv_state);
            const float64 xv = ld_bf16_as_f32(xc, x);

            float64 acc = xv;
            acc = v_f32_mac_b(s0, ld_bf16_as_f32(wc0, weight_t), acc);
            acc = v_f32_mac_b(s1, ld_bf16_as_f32(wc1, weight_t), acc);
            acc = v_f32_mac_b(s2, ld_bf16_as_f32(wc2, weight_t), acc);
            acc = v_f32_mac_b(xv, ld_bf16_as_f32(wc3, weight_t), acc);

            st_f32_as_bf16(xc, y, acc * out_scale);
            st_f32_as_bf16(nc0, new_state, s1);
            st_f32_as_bf16(nc1, new_state, s2);
            st_f32_as_bf16(nc2, new_state, xv);
        }
    }
}
