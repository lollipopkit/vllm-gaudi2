// SPDX-License-Identifier: Apache-2.0
// Fused Paged Relative-Bias GQA Decode Attention + Gate-Sink Epilogue for Inkling 2 Mini on Gaudi2 TPC.
//
// Wave 8 compact VLIW design:
//   - Direct proj_q [Q_cols, B] input (reads gate at H*(128+d_rel)+h*128 and sinks at H*(256+d_rel)+h ->
//     ZERO gate/sinks .split().contiguous() copies across all 48 decode visits!).
//   - Single-exp_f32 algebraic sink (-abs(m_vec - sink_v)) + non-unrolled 2-half gate sigmoid +
//     compact collect_req_blocks -> >50% smaller .text (~12 KB) to cut cold I-cache fetch overhead!
//   - Direct (k_new, v_new) at dist=0 + 128-wide bf16 loads + 2x-unrolled K-dot & V-acc (0 VRF spills).
//
// Tensors (TPC dim order, dim0 fastest):
//   q               [D=128, H, B] bf16                   (PyTorch [B, H, 128])
//   r               [d_rel, H, B] bf16                   (PyTorch [B, H, d_rel], d_rel <= 64)
//   proj_q          [Q_cols, B] bf16                     (PyTorch [B, Q_cols]; gate/sinks read directly)
//   rel_proj_t      [d_rel, rel_extent] bf16             (PyTorch [rel_extent, d_rel])
//   key_cache       [D=128, H_kv, BS, num_phys] bf16     (PyTorch [num_phys, BS, H_kv, 128])
//   value_cache     [D=128, H_kv, BS, num_phys] bf16     (PyTorch [num_phys, BS, H_kv, 128])
//   k_new           [D=128, H_kv, B] bf16                (PyTorch [B, H_kv, 128], current token at q_pos)
//   v_new           [D=128, H_kv, B] bf16                (PyTorch [B, H_kv, 128], current token at q_pos)
//   block_list      [NB] i32
//   block_groups    [NB] i32
//   block_usage     [NB] i32
//   block_start_pos [NB] i32
//   positions       [B] i32
//   out_gated       [D=128, H, B] bf16                   (PyTorch [B, H, 128])
//
// Index space: [H, B] (2D).

#include "inkling2_bf16_common.h"
#include "inkling2_paged_attn_common.h"

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
    const int H_kv = get_dim_size(key_cache, 1);
    const int BS = get_dim_size(key_cache, 2);
    const int NB = get_dim_size(block_list, 0);
    const int groups = H / H_kv;
    const int rel_max = rel_extent - 1;
    const int gate_base = H * (128 + d_rel);
    const int sinks_base = gate_base + (H << 7);
    const int5 is_start = get_index_space_offset();
    const int5 is_end = get_index_space_size() + is_start;
    const uint64 lane_id = read_lane_id_4b_b();
    const tpc_xor_masks_t xmasks = make_xor_masks_nolut(lane_id);

    int5 rp0_c = {0, 0, 0, 0, 0};
    const float64 rel0_v = ld_bf16_as_f32(rp0_c, rel_proj_t);

    for (int b = is_start[1]; b < is_end[1]; b++)
    {
        int5 pc = {b, 0, 0, 0, 0};
        const int q_pos = s_i32_ld_g((__global__ int*)gen_addr(pc, positions));
        int5 skc = {sinks_base, b, 0, 0, 0};
        const float64 sinks_row = ld_bf16_as_f32(skc, proj_q);

        const int n_blks = collect_req_blocks(b, q_pos, NB, BS, sliding_window,
                                              block_list, block_groups, block_usage, block_start_pos);

        for (int h = is_start[0]; h < is_end[0]; h++)
        {
            const int kv_h = h / groups;
            int5 qc = {0, h, b, 0, 0};
            const float64_pair_t qp = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(qc, q));
            const float64 q_e = qp.v1 * scale;
            const float64 q_o = qp.v2 * scale;
            const float64 rv = ld_bf16_as_f32(qc, r);

            // Seed online softmax directly with current token (k_new, v_new) at dist = 0.
            int5 knc = {0, kv_h, b, 0, 0};
            const float64_pair_t knp = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(knc, k_new));
            const float64_pair_t vnp = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(knc, v_new));
            const float64 d0 = v_f32_mac_b(rv, rel0_v, v_f32_mac_b(q_o, knp.v2, q_e * knp.v1));
            float64 m_vec = reduce_add_fast_nolut(d0, xmasks);
            float64 l_vec = 1.0f;
            float64 o_e = vnp.v1;
            float64 o_o = vnp.v2;

            for (int mi = 0; mi < n_blks; mi++)
            {
                const int phys = g_my_phys[mi];
                const int s_begin = g_my_sbeg[mi];
                const int s_end = g_my_send[mi];
                const int start_pos = g_my_spos[mi];

                for (int cs = s_begin; cs < s_end; cs += 64)
                {
                    const int ce = (cs + 64 < s_end) ? (cs + 64) : s_end;
                    const int cnt = ce - cs;
                    const int cnt2 = cnt & ~1;
                    float64 scores_v = -1e30f;

                    int5 ka_c = {0, kv_h, cs, phys, 0};
                    int5 kb_c = {0, kv_h, cs + 1, phys, 0};
                    int5 rpa = {0, 0, 0, 0, 0}, rpb = {0, 0, 0, 0, 0};

                    #pragma clang loop unroll(disable)
                    for (int j = 0; j < cnt2; j += 2)
                    {
                        const int dist = q_pos - (start_pos + cs + j);
                        rpa[1] = (dist < rel_max) ? dist : rel_max;
                        rpb[1] = (dist - 1 < rel_max) ? (dist - 1) : rel_max;

                        const bfloat128 ka_raw = v_bf16_ld_tnsr_b(ka_c, key_cache);
                        const bfloat128 kb_raw = v_bf16_ld_tnsr_b(kb_c, key_cache);
                        const float64 ra = ld_bf16_as_f32(rpa, rel_proj_t);
                        const float64 rb = ld_bf16_as_f32(rpb, rel_proj_t);
                        ka_c[2] += 2;
                        kb_c[2] += 2;

                        const float64_pair_t ka = v_convert_bf16_to_f32_all_b(ka_raw);
                        const float64_pair_t kb = v_convert_bf16_to_f32_all_b(kb_raw);
                        const float64 da = v_f32_mac_b(rv, ra, v_f32_mac_b(q_o, ka.v2, q_e * ka.v1));
                        const float64 db = v_f32_mac_b(rv, rb, v_f32_mac_b(q_o, kb.v2, q_e * kb.v1));

                        const float64_pair_t sp = reduce_add_pair_nolut(da, db, xmasks);
                        scores_v = v_f32_mov_vb(sp.v1, 0, scores_v, v_u32_cmp_eq_b(lane_id, (unsigned int)j), 0);
                        scores_v = v_f32_mov_vb(sp.v2, 0, scores_v, v_u32_cmp_eq_b(lane_id, (unsigned int)(j + 1)), 0);
                    }
                    if (cnt & 1)
                    {
                        const int j = cnt2;
                        ka_c[2] = cs + j;
                        const int dist = q_pos - (start_pos + cs + j);
                        rpa[1] = (dist < rel_max) ? dist : rel_max;
                        const float64_pair_t ka = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(ka_c, key_cache));
                        const float64 ra = ld_bf16_as_f32(rpa, rel_proj_t);
                        const float64 da = v_f32_mac_b(rv, ra, v_f32_mac_b(q_o, ka.v2, q_e * ka.v1));
                        const float64 sa = reduce_add_fast_nolut(da, xmasks);
                        scores_v = v_f32_mov_vb(sa, 0, scores_v, v_u32_cmp_eq_b(lane_id, (unsigned int)j), 0);
                    }

                    const bool64 valid_j = v_u32_cmp_less_b(lane_id, (unsigned int)cnt);
                    const float64 m_chunk = reduce_max_fast_nolut(scores_v, xmasks);
                    const float64 m_new = v_f32_max_b(m_vec, m_chunk);
                    const float64 alpha = exp_f32(v_f32_max_b(m_vec - m_new, -80.0f));
                    const float64 p_v = v_f32_mov_vb(
                        0.0f, 0, exp_f32(v_f32_max_b(scores_v - m_new, -80.0f)), valid_j, 1);

                    l_vec = l_vec * alpha + reduce_add_fast_nolut(p_v, xmasks);
                    o_e *= alpha;
                    o_o *= alpha;
                    m_vec = m_new;

                    float64 o_eb = 0.0f, o_ob = 0.0f;
                    int5 va_c = {0, kv_h, cs, phys, 0};
                    int5 vb_c = {0, kv_h, cs + 1, phys, 0};
                    #pragma clang loop unroll(disable)
                    for (int j = 0; j < cnt2; j += 2)
                    {
                        const float64 pa = v_broadcast_element_f32(p_v, j);
                        const float64 pb = v_broadcast_element_f32(p_v, j + 1);
                        const bfloat128 va_raw = v_bf16_ld_tnsr_b(va_c, value_cache);
                        const bfloat128 vb_raw = v_bf16_ld_tnsr_b(vb_c, value_cache);
                        va_c[2] += 2;
                        vb_c[2] += 2;
                        const float64_pair_t va = v_convert_bf16_to_f32_all_b(va_raw);
                        const float64_pair_t vb = v_convert_bf16_to_f32_all_b(vb_raw);
                        o_e = v_f32_mac_b(va.v1, pa, o_e);
                        o_o = v_f32_mac_b(va.v2, pa, o_o);
                        o_eb = v_f32_mac_b(vb.v1, pb, o_eb);
                        o_ob = v_f32_mac_b(vb.v2, pb, o_ob);
                    }
                    o_e += o_eb;
                    o_o += o_ob;
                    if (cnt & 1)
                    {
                        const int j = cnt2;
                        va_c[2] = cs + j;
                        const float64 pa = v_broadcast_element_f32(p_v, j);
                        const float64_pair_t va = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(va_c, value_cache));
                        o_e = v_f32_mac_b(va.v1, pa, o_e);
                        o_o = v_f32_mac_b(va.v2, pa, o_o);
                    }
                }
            }

            // Compact single-exp_f32 sink + non-unrolled 2-half gate sigmoid (reads gate directly from proj_q).
            const float64 sink_v = v_broadcast_element_f32(sinks_row, h);
            const float64 e_diff = exp_f32(v_f32_max_b(-v_f32_abs_b(m_vec - sink_v), -80.0f));
            const bool64 m_ge_s = v_f32_cmp_geq_b(m_vec, sink_v);
            const float64 e_m = v_f32_mov_vb(1.0f, 0, e_diff, m_ge_s, 0);
            const float64 e_s = v_f32_mov_vb(e_diff, 0, 1.0f, m_ge_s, 0);
            const float64 norm_scale = e_m * rcp_pos_nolut(v_f32_max_b(l_vec * e_m + e_s, 1e-30f));

            int5 gc = {gate_base + (h << 7), b, 0, 0, 0};
            const float64_pair_t gp = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(gc, proj_q));
            g_gate_in[0] = gp.v1;
            g_gate_in[1] = gp.v2;
            g_out_tmp[0] = o_e * norm_scale;
            g_out_tmp[1] = o_o * norm_scale;
            #pragma clang loop unroll(disable)
            for (int p = 0; p < 2; p++)
                g_out_tmp[p] *= sigmoid_nolut(g_gate_in[p]);

            const float64_pair_t out_p = {g_out_tmp[0], g_out_tmp[1]};
            v_bf16_st_tnsr(qc, out_gated, v_convert_f32_to_bf16_all_b(out_p));
        }
    }
}
