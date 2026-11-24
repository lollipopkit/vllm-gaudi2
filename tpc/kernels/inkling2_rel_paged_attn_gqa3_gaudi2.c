// SPDX-License-Identifier: Apache-2.0
// Fused GQA-3 Shared-K/V Paged Relative-Bias Decode Attention + Gate-Sink Epilogue for Gaudi2 TPC.
//
// Dispatched automatically by Inkling2RelPagedAttnGateSinkGaudi2 when B >= 3 and groups = H / H_kv == 3
// (e.g., Inkling 2 Mini c=4/c=8 with H=24, H_kv=8 -> 32..64 work items across 24 TPC cores):
//   - Processes all 3 Q heads sharing each kv_h together: 3x fewer key_cache, value_cache, rel_proj_t
//     HBM reads and 3x fewer collect_req_blocks scans!
//   - Reduces all 3 Q heads' dot products simultaneously via reduce_add_triple_nolut (1 butterfly/token).
//   - Non-unrolled 3-head softmax & gate-sink loops keep .text <= 16 KB inside L1 I-cache and VLM <= 8 KB.
//
// Tensors (TPC dim order, dim0 fastest): identical to inkling2_rel_paged_attn_gate_sink_gaudi2.c.
// Index space: [H_kv, B] (2D).

#include "inkling2_bf16_common.h"
#include "inkling2_paged_attn_common.h"

// 22 * 256 B = 5.5 KB VLM (well under 16 KB SFU-safe budget).
static __local__ float64 g_m[3];
static __local__ float64 g_l[3];
static __local__ float64 g_oe[3];
static __local__ float64 g_oo[3];
static __local__ float64 g_scores[3];
static __local__ float64 g_p[3];
static __local__ float64 g_gate_in[2];
static __local__ float64 g_out_tmp[2];

void main(tensor q, tensor r, tensor proj_q, tensor rel_proj_t,
          tensor key_cache, tensor value_cache, tensor k_new, tensor v_new,
          tensor block_list, tensor block_groups, tensor block_usage, tensor block_start_pos,
          tensor positions, tensor out_gated,
          float scale, int sliding_window, int rel_extent)
{
    const int H = get_dim_size(q, 1);
    const int d_rel = get_dim_size(r, 0);
    const int BS = get_dim_size(key_cache, 2);
    const int NB = get_dim_size(block_list, 0);
    const int rel_max = rel_extent - 1;
    const int gate_base = H * (128 + d_rel);
    const int sinks_base = gate_base + (H << 7);
    const int5 is_start = get_index_space_offset();
    const int5 is_end = get_index_space_size() + is_start;
    const uint64 lane_id = read_lane_id_4b_b();
    const tpc_xor_masks_t xmasks = make_xor_masks_nolut(lane_id);

    int5 rp0_c = {0, 0, 0, 0, 0};
    const float64 rel0_v = ld_bf16_as_f32(rp0_c, rel_proj_t);

    #pragma clang loop unroll(disable)
    for (int b = is_start[1]; b < is_end[1]; b++)
    {
        int5 pc = {b, 0, 0, 0, 0};
        const int q_pos = s_i32_ld_g((__global__ int*)gen_addr(pc, positions));
        int5 skc = {sinks_base, b, 0, 0, 0};
        const float64 sinks_row = ld_bf16_as_f32(skc, proj_q);

        const int n_blks = collect_req_blocks(b, q_pos, NB, BS, sliding_window,
                                              block_list, block_groups, block_usage, block_start_pos);

        #pragma clang loop unroll(disable)
        for (int kv_h = is_start[0]; kv_h < is_end[0]; kv_h++)
        {
            const int h0 = kv_h * 3;
            int5 qc0 = {0, h0 + 0, b, 0, 0};
            int5 qc1 = {0, h0 + 1, b, 0, 0};
            int5 qc2 = {0, h0 + 2, b, 0, 0};
            const float64_pair_t qp0 = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(qc0, q));
            const float64_pair_t qp1 = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(qc1, q));
            const float64_pair_t qp2 = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(qc2, q));
            const float64 q0_e = qp0.v1 * scale, q0_o = qp0.v2 * scale;
            const float64 q1_e = qp1.v1 * scale, q1_o = qp1.v2 * scale;
            const float64 q2_e = qp2.v1 * scale, q2_o = qp2.v2 * scale;
            const float64 r0 = ld_bf16_as_f32(qc0, r);
            const float64 r1 = ld_bf16_as_f32(qc1, r);
            const float64 r2 = ld_bf16_as_f32(qc2, r);

            // Seed online softmax directly with shared current token (k_new, v_new) at dist = 0.
            int5 knc = {0, kv_h, b, 0, 0};
            const float64_pair_t knp = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(knc, k_new));
            const float64_pair_t vnp = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(knc, v_new));
            const float64 d0 = v_f32_mac_b(r0, rel0_v, v_f32_mac_b(q0_o, knp.v2, q0_e * knp.v1));
            const float64 d1 = v_f32_mac_b(r1, rel0_v, v_f32_mac_b(q1_o, knp.v2, q1_e * knp.v1));
            const float64 d2 = v_f32_mac_b(r2, rel0_v, v_f32_mac_b(q2_o, knp.v2, q2_e * knp.v1));
            const float64_triple_t m012 = reduce_add_triple_nolut(d0, d1, d2, xmasks);
            g_m[0] = m012.v1; g_m[1] = m012.v2; g_m[2] = m012.v3;
            g_l[0] = 1.0f;    g_l[1] = 1.0f;    g_l[2] = 1.0f;
            g_oe[0] = vnp.v1; g_oe[1] = vnp.v1; g_oe[2] = vnp.v1;
            g_oo[0] = vnp.v2; g_oo[1] = vnp.v2; g_oo[2] = vnp.v2;

            #pragma clang loop unroll(disable)
            for (int mi = 0; mi < n_blks; mi++)
            {
                const int phys = g_my_phys[mi];
                const int s_begin = g_my_sbeg[mi];
                const int s_end = g_my_send[mi];
                const int start_pos = g_my_spos[mi];

                #pragma clang loop unroll(disable)
                for (int cs = s_begin; cs < s_end; cs += 64)
                {
                    const int ce = (cs + 64 < s_end) ? (cs + 64) : s_end;
                    const int cnt = ce - cs;
                    float64 sc0 = -1e30f, sc1 = -1e30f, sc2 = -1e30f;

                    // Phase 1: Shared K + rel_proj_t loads feed all 3 Q heads simultaneously.
                    int5 kc = {0, kv_h, cs, phys, 0};
                    int5 rpc = {0, 0, 0, 0, 0};
                    #pragma clang loop unroll(disable)
                    for (int j = 0; j < cnt; j++)
                    {
                        const int dist = q_pos - (start_pos + cs + j);
                        rpc[1] = (dist < rel_max) ? dist : rel_max;
                        const float64_pair_t kv = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(kc, key_cache));
                        const float64 rv = ld_bf16_as_f32(rpc, rel_proj_t);
                        kc[2] += 1;

                        const float64 t0 = v_f32_mac_b(r0, rv, v_f32_mac_b(q0_o, kv.v2, q0_e * kv.v1));
                        const float64 t1 = v_f32_mac_b(r1, rv, v_f32_mac_b(q1_o, kv.v2, q1_e * kv.v1));
                        const float64 t2 = v_f32_mac_b(r2, rv, v_f32_mac_b(q2_o, kv.v2, q2_e * kv.v1));
                        const float64_triple_t sp = reduce_add_triple_nolut(t0, t1, t2, xmasks);
                        const bool64 mj = v_u32_cmp_eq_b(lane_id, (unsigned int)j);
                        sc0 = v_f32_mov_vb(sp.v1, 0, sc0, mj, 0);
                        sc1 = v_f32_mov_vb(sp.v2, 0, sc1, mj, 0);
                        sc2 = v_f32_mov_vb(sp.v3, 0, sc2, mj, 0);
                    }
                    g_scores[0] = sc0;
                    g_scores[1] = sc1;
                    g_scores[2] = sc2;

                    // Phase 2: Compact 3-head online softmax update (emits exp_f32 once!).
                    const bool64 valid_j = v_u32_cmp_less_b(lane_id, (unsigned int)cnt);
                    #pragma clang loop unroll(disable)
                    for (int g = 0; g < 3; g++)
                    {
                        const float64 sv = g_scores[g];
                        const float64 m_old = g_m[g];
                        const float64 m_chunk = reduce_max_fast_nolut(sv, xmasks);
                        const float64 m_new = v_f32_max_b(m_old, m_chunk);
                        const float64 alpha = exp_f32(v_f32_max_b(m_old - m_new, -80.0f));
                        const float64 pv = v_f32_mov_vb(
                            0.0f, 0, exp_f32(v_f32_max_b(sv - m_new, -80.0f)), valid_j, 1);
                        g_l[g] = g_l[g] * alpha + reduce_add_fast_nolut(pv, xmasks);
                        g_oe[g] *= alpha;
                        g_oo[g] *= alpha;
                        g_m[g] = m_new;
                        g_p[g] = pv;
                    }

                    // Phase 3: Shared V loads accumulate all 3 Q head outputs in registers.
                    const float64 p0 = g_p[0], p1 = g_p[1], p2 = g_p[2];
                    float64 oe0 = g_oe[0], oo0 = g_oo[0];
                    float64 oe1 = g_oe[1], oo1 = g_oo[1];
                    float64 oe2 = g_oe[2], oo2 = g_oo[2];
                    int5 vc = {0, kv_h, cs, phys, 0};
                    #pragma clang loop unroll(disable)
                    for (int j = 0; j < cnt; j++)
                    {
                        const float64_pair_t vv = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(vc, value_cache));
                        vc[2] += 1;
                        const float64 w0 = v_broadcast_element_f32(p0, j);
                        const float64 w1 = v_broadcast_element_f32(p1, j);
                        const float64 w2 = v_broadcast_element_f32(p2, j);
                        oe0 = v_f32_mac_b(vv.v1, w0, oe0);
                        oo0 = v_f32_mac_b(vv.v2, w0, oo0);
                        oe1 = v_f32_mac_b(vv.v1, w1, oe1);
                        oo1 = v_f32_mac_b(vv.v2, w1, oo1);
                        oe2 = v_f32_mac_b(vv.v1, w2, oe2);
                        oo2 = v_f32_mac_b(vv.v2, w2, oo2);
                    }
                    g_oe[0] = oe0; g_oo[0] = oo0;
                    g_oe[1] = oe1; g_oo[1] = oo1;
                    g_oe[2] = oe2; g_oo[2] = oo2;
                }
            }

            // Epilogue: compact single-instance sink + 2-half gate sigmoid for all 3 Q heads.
            #pragma clang loop unroll(disable)
            for (int g = 0; g < 3; g++)
            {
                const int h = h0 + g;
                const float64 m_vec = g_m[g];
                const float64 sink_v = v_broadcast_element_f32(sinks_row, h);
                const float64 e_diff = exp_f32(v_f32_max_b(-v_f32_abs_b(m_vec - sink_v), -80.0f));
                const bool64 m_ge_s = v_f32_cmp_geq_b(m_vec, sink_v);
                const float64 e_m = v_f32_mov_vb(1.0f, 0, e_diff, m_ge_s, 0);
                const float64 e_s = v_f32_mov_vb(e_diff, 0, 1.0f, m_ge_s, 0);
                const float64 norm_scale = e_m * rcp_pos_nolut(v_f32_max_b(g_l[g] * e_m + e_s, 1e-30f));

                int5 gc = {gate_base + (h << 7), b, 0, 0, 0};
                const float64_pair_t gp = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(gc, proj_q));
                g_gate_in[0] = gp.v1;
                g_gate_in[1] = gp.v2;
                g_out_tmp[0] = g_oe[g] * norm_scale;
                g_out_tmp[1] = g_oo[g] * norm_scale;
                #pragma clang loop unroll(disable)
                for (int p = 0; p < 2; p++)
                    g_out_tmp[p] *= sigmoid_nolut(g_gate_in[p]);

                int5 oc = {0, h, b, 0, 0};
                const float64_pair_t out_p = {g_out_tmp[0], g_out_tmp[1]};
                v_bf16_st_tnsr(oc, out_gated, v_convert_f32_to_bf16_all_b(out_p));
            }
        }
    }
}
