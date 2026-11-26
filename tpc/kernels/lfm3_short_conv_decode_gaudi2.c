// SPDX-License-Identifier: Apache-2.0
// Pure functional fused LFM3 gated ShortConv decode kernel on Gaudi2 TPC:
//   B_g, C_g, x = chunk(BCx, 3, dim=-1)
//   bx = bf16(B_g * x)
//   conv_out = sum_{k=0}^{state_len-1} conv_state[load_slot, k] * taps[k] + bx * taps[state_len] (+ bias)
//   new_state[b] = [conv_state[load_slot, 1..state_len-1], bx]  (emitted explicitly, caller scatters to store_idx)
//   out[b] = bf16(C_g * bf16(conv_out))
//
// Zero VLM, zero SFU LUTs, D/64-way parallel per batch item (32 items for D=2048 -> saturates all 24 TPC cores
// even at B=1). Keeping conv_state strictly read-only avoids cross-batch store/load races, L1 read-cache
// coherence issues, and Synapse read-only SRAM caching hazards when prefix caching or slot aliasing occurs.
//
// Tensors (TPC dim order, dim0 fastest):
//   bcx        [3*D, B]              bf16
//   conv_state [D, cache_len, slots] bf16 (read-only)
//   load_idx   [B]                   i32  (out-of-range / negative slots emit zeros)
//   taps       [D, state_len + 1]    f32  (PyTorch [kernel, D])
//   bias       [D]                   f32  (ignored when has_bias == 0)
//   out        [D, B]                bf16
//   new_state  [D, state_len, B]     bf16
//
// Index space: [D / 64, 1, B].
#include "gaudi2_fp8_common.h"

void main(tensor bcx, tensor conv_state, tensor load_idx, tensor taps, tensor bias,
          tensor out, tensor new_state, int has_bias)
{
    const int D = get_dim_size(out, 0);
    const int cache_len = get_dim_size(conv_state, 1);
    const int state_len = get_dim_size(taps, 1) - 1;  // e.g. 6 for L=7
    const int col_off = cache_len - state_len;
    const int num_slots = get_dim_size(conv_state, 2);
    const int5 is_start = get_index_space_offset();
    const int5 is_end = get_index_space_size() + is_start;

    for (int b = is_start[2]; b < is_end[2]; b++)
    {
        int5 ic = {b, 0, 0, 0, 0};
        const int lslot = *(__global__ int*)gen_addr(ic, load_idx);

        // Out-of-bounds / negative slots emit deterministic zeros without reading conv_state.
        if (lslot < 0 || lslot >= num_slots)
        {
            const float64 z = 0.0f;
            for (int g = is_start[0]; g < is_end[0]; g++)
            {
                const int d = g * 64;
                int5 oc = {d, b, 0, 0, 0};
                st_f32_as_bf16(oc, out, z);
                for (int k = 0; k < state_len; k++)
                {
                    int5 sc = {d, k, b, 0, 0};
                    st_f32_as_bf16(sc, new_state, z);
                }
            }
            continue;
        }

        for (int g = is_start[0]; g < is_end[0]; g++)
        {
            const int d = g * 64;
            int5 bc = {d, b, 0, 0, 0};
            int5 cc = {D + d, b, 0, 0, 0};
            int5 xc = {2 * D + d, b, 0, 0, 0};

            const float64 bg = ld_bf16_as_f32(bc, bcx);
            const float64 cg = ld_bf16_as_f32(cc, bcx);
            const float64 xv = ld_bf16_as_f32(xc, bcx);
            const bfloat128 bx_bf16 = v_convert_f32_to_bf16_b(bg * xv, 0, 0, 0, 1, 0);
            const float64 bx = unpack_bf16_lo(bx_bf16);

            float64 acc = 0.0f;
            for (int k = 0; k < state_len; k++)
            {
                int5 lc = {d, col_off + k, lslot, 0, 0};
                int5 tc = {d, k, 0, 0, 0};
                const float64 sv = ld_bf16_as_f32(lc, conv_state);
                acc = v_f32_mac_b(sv, v_f32_ld_tnsr_b(tc, taps), acc);
                if (k > 0)
                {
                    int5 sc = {d, k - 1, b, 0, 0};
                    st_f32_as_bf16(sc, new_state, sv);
                }
            }
            {
                int5 tc = {d, state_len, 0, 0, 0};
                acc = v_f32_mac_b(bx, v_f32_ld_tnsr_b(tc, taps), acc);
            }
            if (has_bias)
            {
                int5 bic = {d, 0, 0, 0, 0};
                acc = acc + v_f32_ld_tnsr_b(bic, bias);
            }
            {
                int5 sc = {d, state_len - 1, b, 0, 0};
                st_bf16_lo_vec(sc, new_state, bx_bf16);
            }
            const float64 y = cg * round_f32_via_bf16(acc);
            int5 oc = {d, b, 0, 0, 0};
            st_f32_as_bf16(oc, out, y);
        }
    }
}
