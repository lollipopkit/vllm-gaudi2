// SPDX-License-Identifier: Apache-2.0
// Single-token Gated DeltaNet decode step on the f32 recurrent state, one HBM pass over S.
//
// Per (batch b, value head h, value index v), with key head h / HEADS_PER_KEY, k/q L2-normalized:
//   sk = sum_i S[i,v] k[i]      sq = sum_i S[i,v] q[i]
//   delta = (value[v] - decay * sk) * beta
//   S'[i,v] = decay * S[i,v] + delta * k[i]
//   out[v] = sum_i S'[i,v] q[i] = decay * sq + delta * (k . q)    (q scaled by `scale`, applied at the end)
// The state is KV-major (value index fastest), so lanes run over v and every term is a vector MAC with a
// broadcast k[i] / q[i]: no cross-lane reductions in the loop. Each member owns one (b, h) and both 64-lane
// halves of V, so every broadcast k[i] / q[i] load serves two vectors (loads, not HBM, bound the loop), and
// keeps its K x V state in VLM between the projection pass and the update pass: S is read from HBM once
// and written once.
// The state is updated in place in the cache pool (rows row_offset .. row_offset + B - 1): a separate
// output tensor would need a second full-state copy back into the pool.
//
// The inputs are what the model graph already has, so no stack / broadcast / transpose node sits between the
// L2 norm and the kernel (each is a ~2 us dependent hop per layer at batch 1): q and k unscaled, the scale
// applied to the output; k . q per key head.
//
// Tensors (TPC dim order, dim0 fastest):
//   pool  [V=128, K=128, HV, slots] f32 (read and written)    qk    [K=128, 2H (q heads, then k heads), B] f32
//   value [V=128, HV, B] f32     decay, beta [HV, B] f32      kqdot [H, B] f32 (unscaled k . q)
//   out   [V=128, HV, B] f32
// Index space: [1, HV, B].

#define DIM 128
#define LANES 64
#define HEADS_PER_KEY 3

__local__ float64 slice0[DIM];  // value columns 0..63
__local__ float64 slice1[DIM];  // value columns 64..127

void main(tensor pool, tensor qk, tensor value, tensor decay_t, tensor beta_t, tensor kqdot_t, tensor out,
          int row_offset, float scale)
{
    const int key_heads = get_dim_size(qk, 1) / 2;
    const int5 is_start = get_index_space_offset();
    const int5 is_end = get_index_space_size() + is_start;

    for (int b = is_start[2]; b < is_end[2]; b++)
    {
        for (int h = is_start[1]; h < is_end[1]; h++)
        {
            const int kh = h / HEADS_PER_KEY;
            int5 cc = {h, b, 0, 0, 0};
            const float64 decay = v_f32_ld_g((__global__ float*)gen_addr(cc, decay_t));
            const float64 beta = v_f32_ld_g((__global__ float*)gen_addr(cc, beta_t));
            cc[0] = kh;
            const float64 kq_dot = v_f32_ld_g((__global__ float*)gen_addr(cc, kqdot_t));

            float64 ka0 = 0.0f, qa0 = 0.0f, kb0 = 0.0f, qb0 = 0.0f;  // half 0: even / odd rows
            float64 ka1 = 0.0f, qa1 = 0.0f, kb1 = 0.0f, qb1 = 0.0f;  // half 1
            int5 s0c = {0, 0, h, row_offset + b, 0};
            int5 s1c = {LANES, 0, h, row_offset + b, 0};
            int5 kc = {0, key_heads + kh, b, 0, 0};
            int5 qc = {0, kh, b, 0, 0};
            // Pass 1: projections of both halves; even/odd accumulators break the MAC dependency chains.
            for (int i = 0; i < DIM; i += 2)
            {
                s0c[1] = i;
                s1c[1] = i;
                const float64 a0 = v_f32_ld_tnsr_b(s0c, pool);
                const float64 a1 = v_f32_ld_tnsr_b(s1c, pool);
                s0c[1] = i + 1;
                s1c[1] = i + 1;
                const float64 b0 = v_f32_ld_tnsr_b(s0c, pool);
                const float64 b1 = v_f32_ld_tnsr_b(s1c, pool);
                kc[0] = i;
                qc[0] = i;
                const float64 k0 = v_f32_ld_g((__global__ float*)gen_addr(kc, qk));
                const float64 q0 = v_f32_ld_g((__global__ float*)gen_addr(qc, qk));
                kc[0] = i + 1;
                qc[0] = i + 1;
                const float64 q1 = v_f32_ld_g((__global__ float*)gen_addr(qc, qk));
                const float64 k1 = v_f32_ld_g((__global__ float*)gen_addr(kc, qk));
                slice0[i] = a0;
                slice1[i] = a1;
                slice0[i + 1] = b0;
                slice1[i + 1] = b1;
                ka0 = v_f32_mac_b(a0, k0, ka0);
                qa0 = v_f32_mac_b(a0, q0, qa0);
                ka1 = v_f32_mac_b(a1, k0, ka1);
                qa1 = v_f32_mac_b(a1, q0, qa1);
                kb0 = v_f32_mac_b(b0, k1, kb0);
                qb0 = v_f32_mac_b(b0, q1, qb0);
                kb1 = v_f32_mac_b(b1, k1, kb1);
                qb1 = v_f32_mac_b(b1, q1, qb1);
            }

            int5 vc = {0, h, b, 0, 0};
            const float64 d0 = (v_f32_ld_tnsr_b(vc, value) - decay * (ka0 + kb0)) * beta;
            v_f32_st_tnsr(vc, out, v_f32_mac_b(d0, kq_dot, decay * (qa0 + qb0)) * scale);
            vc[0] = LANES;
            const float64 d1 = (v_f32_ld_tnsr_b(vc, value) - decay * (ka1 + kb1)) * beta;
            v_f32_st_tnsr(vc, out, v_f32_mac_b(d1, kq_dot, decay * (qa1 + qb1)) * scale);

            // Pass 2: rank-one update from VLM; one k[i] load per row serves both halves. Unrolled for the
            // scheduler to overlap the stores (simulator: -26 % cycles; pass 1 is too large to unroll).
            #pragma loop_unroll(4)
            for (int i = 0; i < DIM; i += 2)
            {
                kc[0] = i;
                const float64 k0 = v_f32_ld_g((__global__ float*)gen_addr(kc, qk));
                kc[0] = i + 1;
                const float64 k1 = v_f32_ld_g((__global__ float*)gen_addr(kc, qk));
                const float64 r00 = v_f32_mac_b(d0, k0, decay * slice0[i]);
                const float64 r01 = v_f32_mac_b(d1, k0, decay * slice1[i]);
                const float64 r10 = v_f32_mac_b(d0, k1, decay * slice0[i + 1]);
                const float64 r11 = v_f32_mac_b(d1, k1, decay * slice1[i + 1]);
                s0c[1] = i;
                s1c[1] = i;
                v_f32_st_tnsr(s0c, pool, r00);
                v_f32_st_tnsr(s1c, pool, r01);
                s0c[1] = i + 1;
                s1c[1] = i + 1;
                v_f32_st_tnsr(s0c, pool, r10);
                v_f32_st_tnsr(s1c, pool, r11);
            }
        }
    }
}
