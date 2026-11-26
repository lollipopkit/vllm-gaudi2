// SPDX-License-Identifier: Apache-2.0
// Fused Qwen3.8 GDN pre-recurrence decode kernel on Gaudi2 TPC:
// Replaces ~10 serial TPC/DMA nodes per GDN layer before gdn_decode:
//   1. Gating: decay = exp(neg_exp_A * softplus(a + dt_bias)), beta = sigmoid(b) (vectorized on w==0)
//   2. 4-tap FP32 causal conv1d + optional bias + in-place conv_state shift (zero torch.cat/DmaMemcpy)
//   3. SiLU activation (VPU-pipelined 4x/2x zero-LUT, rounded through BF16 to match stock F.silu)
//   4. L2-normalization of Q & K (rsqrt(sum(x^2) + 1e-6)) + k.q dot product
//   5. FP32 V output ready directly for gdn_decode_f32_gaudi2
//
// Zero VLM, zero SFU LUTs, 64-way parallel per batch item (16 key heads + 48 value heads) -> saturates
// all 24 TPC cores even at B=1.
//
// Tensors (TPC dim order, dim0 fastest):
//   packed_qkv  [W=10240, B]       bf16  (Q[0..2047], K[2048..4095], V[4096..10239])
//   a_in        [HV=48, B]         bf16
//   b_in        [HV=48, B]         bf16
//   neg_exp_A   [HV=48]            f32
//   dt_bias     [HV=48]            f32
//   conv_state  [W=10240, L, slots] bf16 (updated in place at conv_col_offset..+2, row_offset + b)
//   conv_taps   [W=10240, 4]       f32   (PyTorch [4, W])
//   conv_bias   [W=10240]          f32   (ignored when has_bias == 0)
//   qk_out      [128, 2*H=32, B]   f32   (0..H-1: Q, H..2H-1: K)
//   v_out       [128, HV=48, B]    f32
//   decay_out   [HV=48, B]         f32
//   beta_out    [HV=48, B]         f32
//   kqdot_out   [H=16, B]          f32
//
// Index space: [1, H + HV = 64, B] (dim0 = head_dim, dim1 = head worker, dim2 = batch).
#include "gaudi2_fp8_common.h"

#define HEAD_DIM 128

static inline float64 conv4_bf16_vec(int col, int row, int b, int c_off, int has_bias,
                                     tensor packed_qkv, tensor conv_state, tensor conv_taps, tensor conv_bias)
{
    int5 sc0 = {col, c_off + 0, row, 0, 0};
    int5 sc1 = {col, c_off + 1, row, 0, 0};
    int5 sc2 = {col, c_off + 2, row, 0, 0};
    int5 xc = {col, b, 0, 0, 0};
    int5 tc0 = {col, 0, 0, 0, 0};
    int5 tc1 = {col, 1, 0, 0, 0};
    int5 tc2 = {col, 2, 0, 0, 0};
    int5 tc3 = {col, 3, 0, 0, 0};

    const float64 s0 = ld_bf16_as_f32(sc0, conv_state);
    const float64 s1 = ld_bf16_as_f32(sc1, conv_state);
    const float64 s2 = ld_bf16_as_f32(sc2, conv_state);
    const float64 xv = ld_bf16_as_f32(xc, packed_qkv);

    st_f32_as_bf16(sc0, conv_state, s1);
    st_f32_as_bf16(sc1, conv_state, s2);
    st_f32_as_bf16(sc2, conv_state, xv);

    float64 acc = s0 * v_f32_ld_tnsr_b(tc0, conv_taps);
    acc = v_f32_mac_b(s1, v_f32_ld_tnsr_b(tc1, conv_taps), acc);
    acc = v_f32_mac_b(s2, v_f32_ld_tnsr_b(tc2, conv_taps), acc);
    acc = v_f32_mac_b(xv, v_f32_ld_tnsr_b(tc3, conv_taps), acc);
    if (has_bias)
    {
        int5 bc = {col, 0, 0, 0, 0};
        acc = acc + v_f32_ld_tnsr_b(bc, conv_bias);
    }
    return round_f32_via_bf16(acc);
}

void main(tensor packed_qkv, tensor a_in, tensor b_in, tensor neg_exp_A, tensor dt_bias,
          tensor conv_state, tensor conv_taps, tensor conv_bias,
          tensor qk_out, tensor v_out, tensor decay_out, tensor beta_out, tensor kqdot_out,
          int row_offset, int conv_col_offset, int has_bias)
{
    const int H = get_dim_size(kqdot_out, 0);
    const int HV = get_dim_size(decay_out, 0);
    const int qk_width = 2 * H * HEAD_DIM;
    const int5 is_start = get_index_space_offset();
    const int5 is_end = get_index_space_size() + is_start;
    const tpc_xor_masks_t xmasks = make_xor_masks_nolut(read_lane_id_4b_b());

    for (int b = is_start[2]; b < is_end[2]; b++)
    {
        const int row = row_offset + b;
        for (int w = is_start[1]; w < is_end[1]; w++)
        {
            if (w == 0)
            {
                int5 hc = {0, b, 0, 0, 0};
                int5 pc = {0, 0, 0, 0, 0};
                const float64 av = ld_bf16_as_f32(hc, a_in);
                const float64 bv = ld_bf16_as_f32(hc, b_in);
                const float64 dt = v_f32_ld_tnsr_b(pc, dt_bias);
                const float64 nea = v_f32_ld_tnsr_b(pc, neg_exp_A);
                const float64 sp = softplus_zero_lut(av + dt);
                const float64 decay = exp_f32_nolut(nea * sp);
                const float64 beta = round_f32_via_bf16(sigmoid_zero_lut(bv));
                v_f32_st_tnsr_partial(hc, decay_out, decay, (unsigned short)(HV - 1), 0);
                v_f32_st_tnsr_partial(hc, beta_out, beta, (unsigned short)(HV - 1), 0);
            }

            if (w < H)
            {
                const int kh = w;
                const int q_col = kh * HEAD_DIM;
                const int k_col = (H + kh) * HEAD_DIM;

                const float64 cq0 = conv4_bf16_vec(q_col, row, b, conv_col_offset, has_bias,
                                                   packed_qkv, conv_state, conv_taps, conv_bias);
                const float64 cq1 = conv4_bf16_vec(q_col + 64, row, b, conv_col_offset, has_bias,
                                                   packed_qkv, conv_state, conv_taps, conv_bias);
                const float64 ck0 = conv4_bf16_vec(k_col, row, b, conv_col_offset, has_bias,
                                                   packed_qkv, conv_state, conv_taps, conv_bias);
                const float64 ck1 = conv4_bf16_vec(k_col + 64, row, b, conv_col_offset, has_bias,
                                                   packed_qkv, conv_state, conv_taps, conv_bias);

                const float64_quad_t sv = silu_fast_bf16_x4(cq0, cq1, ck0, ck1);
                const float64 q0 = round_f32_via_bf16(sv.v0);
                const float64 q1 = round_f32_via_bf16(sv.v1);
                const float64 k0 = round_f32_via_bf16(sv.v2);
                const float64 k1 = round_f32_via_bf16(sv.v3);

                const float64 q_sq = v_f32_mac_b(q1, q1, q0 * q0);
                const float64 k_sq = v_f32_mac_b(k1, k1, k0 * k0);
                const float64_pair_t norms = reduce_add_pair_nolut(q_sq, k_sq, xmasks);
                const float64 q_inv = rsqrt_nolut(norms.v1 + 1e-6f);
                const float64 k_inv = rsqrt_nolut(norms.v2 + 1e-6f);

                const float64 qn0 = q0 * q_inv;
                const float64 qn1 = q1 * q_inv;
                const float64 kn0 = k0 * k_inv;
                const float64 kn1 = k1 * k_inv;

                int5 qc0 = {0, kh, b, 0, 0}, qc1 = {64, kh, b, 0, 0};
                int5 kc0 = {0, H + kh, b, 0, 0}, kc1 = {64, H + kh, b, 0, 0};
                v_f32_st_tnsr(qc0, qk_out, qn0);
                v_f32_st_tnsr(qc1, qk_out, qn1);
                v_f32_st_tnsr(kc0, qk_out, kn0);
                v_f32_st_tnsr(kc1, qk_out, kn1);

                const float64 dot_v = reduce_add_fast_nolut(v_f32_mac_b(kn1, qn1, kn0 * qn0), xmasks);
                int5 dc = {kh, b, 0, 0, 0};
                v_f32_st_tnsr_partial(dc, kqdot_out, dot_v, 0, 0);
            }
            else
            {
                const int vh = w - H;
                const int v_col = qk_width + vh * HEAD_DIM;
                const float64 cv0 = conv4_bf16_vec(v_col, row, b, conv_col_offset, has_bias,
                                                   packed_qkv, conv_state, conv_taps, conv_bias);
                const float64 cv1 = conv4_bf16_vec(v_col + 64, row, b, conv_col_offset, has_bias,
                                                   packed_qkv, conv_state, conv_taps, conv_bias);
                const float64_pair_t sv = silu_fast_bf16_x2(cv0, cv1);
                int5 vc0 = {0, vh, b, 0, 0}, vc1 = {64, vh, b, 0, 0};
                v_f32_st_tnsr(vc0, v_out, round_f32_via_bf16(sv.v1));
                v_f32_st_tnsr(vc1, v_out, round_f32_via_bf16(sv.v2));
            }
        }
    }
}
