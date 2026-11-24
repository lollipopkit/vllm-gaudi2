// SPDX-License-Identifier: Apache-2.0
// Gated DeltaNet speculative-decode verification: T = 1 + k tokens per request on the f32 recurrent state,
// keeping the state after every token (the next step resumes from the row of its last accepted token).
//
// Same per-token math as gdn_decode_f32_gaudi2 (KV-major state, lanes over the value index, q pre-scaled):
//   sk = sum_i S[i,v] k[i]      sq = sum_i S[i,v] q[i]
//   delta = (value[v] - decay * sk) * beta
//   S'[i,v] = decay * S[i,v] + delta * k[i]
//   out[v] = decay * sq + delta * (k . q)
// Each member owns one (b, h): it reads the initial K x V state once into VLM, runs the T tokens on VLM and
// stores the state after token t to pool row rows[t + 1, b]. HBM traffic is one state read and T writes.
//
// Tensors (TPC dim order, dim0 fastest):
//   pool  [V=128, K=128, HV, slots] f32 (read and written)    kq    [K=128, 2 (k, q), H, T, B] f32
//   value [V=128, HV, T, B] f32                               coeff [3 (decay, beta, k . q), HV, T, B] f32
//   rows  [1 + T, B] i32: row 0 = initial state row, row 1 + t = row receiving the state after token t
//   out   [V=128, HV, T, B] f32
// Index space: [1, HV, B].

#define DIM 128
#define LANES 64
#define HEADS_PER_KEY 3

__local__ float64 slice0[DIM];  // value columns 0..63
__local__ float64 slice1[DIM];  // value columns 64..127

void main(tensor pool, tensor kq, tensor value, tensor coeff, tensor rows, tensor out)
{
    const int5 is_start = get_index_space_offset();
    const int5 is_end = get_index_space_size() + is_start;
    const int tokens = get_dim_size(value, 2);

    for (int b = is_start[2]; b < is_end[2]; b++)
    {
        for (int h = is_start[1]; h < is_end[1]; h++)
        {
            const int kh = h / HEADS_PER_KEY;
            int5 rc = {0, b, 0, 0, 0};
            const int load_row = s_i32_ld_g((__global__ int*)gen_addr(rc, rows));
            int5 s0c = {0, 0, h, load_row, 0};
            int5 s1c = {LANES, 0, h, load_row, 0};
            for (int i = 0; i < DIM; i++)
            {
                s0c[1] = i;
                s1c[1] = i;
                slice0[i] = v_f32_ld_tnsr_b(s0c, pool);
                slice1[i] = v_f32_ld_tnsr_b(s1c, pool);
            }

            for (int t = 0; t < tokens; t++)
            {
                int5 cc = {0, h, t, b, 0};
                const float64 decay = v_f32_ld_g((__global__ float*)gen_addr(cc, coeff));
                cc[0] = 1;
                const float64 beta = v_f32_ld_g((__global__ float*)gen_addr(cc, coeff));
                cc[0] = 2;
                const float64 kq_dot = v_f32_ld_g((__global__ float*)gen_addr(cc, coeff));

                // Projections; even/odd accumulators break the MAC dependency chains.
                float64 ka0 = 0.0f, qa0 = 0.0f, kb0 = 0.0f, qb0 = 0.0f;
                float64 ka1 = 0.0f, qa1 = 0.0f, kb1 = 0.0f, qb1 = 0.0f;
                int5 kqc = {0, 0, kh, t, b};
                for (int i = 0; i < DIM; i += 2)
                {
                    kqc[0] = i;
                    kqc[1] = 0;
                    const float64 k0 = v_f32_ld_g((__global__ float*)gen_addr(kqc, kq));
                    kqc[1] = 1;
                    const float64 q0 = v_f32_ld_g((__global__ float*)gen_addr(kqc, kq));
                    kqc[0] = i + 1;
                    const float64 q1 = v_f32_ld_g((__global__ float*)gen_addr(kqc, kq));
                    kqc[1] = 0;
                    const float64 k1 = v_f32_ld_g((__global__ float*)gen_addr(kqc, kq));
                    const float64 a0 = slice0[i], a1 = slice1[i];
                    const float64 b0 = slice0[i + 1], b1 = slice1[i + 1];
                    ka0 = v_f32_mac_b(a0, k0, ka0);
                    qa0 = v_f32_mac_b(a0, q0, qa0);
                    ka1 = v_f32_mac_b(a1, k0, ka1);
                    qa1 = v_f32_mac_b(a1, q0, qa1);
                    kb0 = v_f32_mac_b(b0, k1, kb0);
                    qb0 = v_f32_mac_b(b0, q1, qb0);
                    kb1 = v_f32_mac_b(b1, k1, kb1);
                    qb1 = v_f32_mac_b(b1, q1, qb1);
                }

                int5 vc = {0, h, t, b, 0};
                const float64 d0 = (v_f32_ld_tnsr_b(vc, value) - decay * (ka0 + kb0)) * beta;
                v_f32_st_tnsr(vc, out, v_f32_mac_b(d0, kq_dot, decay * (qa0 + qb0)));
                vc[0] = LANES;
                const float64 d1 = (v_f32_ld_tnsr_b(vc, value) - decay * (ka1 + kb1)) * beta;
                v_f32_st_tnsr(vc, out, v_f32_mac_b(d1, kq_dot, decay * (qa1 + qb1)));

                // Rank-one update in VLM, stored to this token's state row.
                rc[0] = t + 1;
                const int store_row = s_i32_ld_g((__global__ int*)gen_addr(rc, rows));
                s0c[3] = store_row;
                s1c[3] = store_row;
                kqc[1] = 0;
                for (int i = 0; i < DIM; i += 2)
                {
                    kqc[0] = i;
                    const float64 k0 = v_f32_ld_g((__global__ float*)gen_addr(kqc, kq));
                    kqc[0] = i + 1;
                    const float64 k1 = v_f32_ld_g((__global__ float*)gen_addr(kqc, kq));
                    const float64 r00 = v_f32_mac_b(d0, k0, decay * slice0[i]);
                    const float64 r01 = v_f32_mac_b(d1, k0, decay * slice1[i]);
                    const float64 r10 = v_f32_mac_b(d0, k1, decay * slice0[i + 1]);
                    const float64 r11 = v_f32_mac_b(d1, k1, decay * slice1[i + 1]);
                    slice0[i] = r00;
                    slice1[i] = r01;
                    slice0[i + 1] = r10;
                    slice1[i + 1] = r11;
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
}
