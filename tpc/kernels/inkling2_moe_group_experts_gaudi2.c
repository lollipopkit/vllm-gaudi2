// SPDX-License-Identifier: Apache-2.0
// Zero-sync Compact 4-Bucket Cost-Weighted MoE Expert Grouper for Inkling 2 Mini on Gaudi2 TPC (B >= 2).
//
// Wave 9b upgrades:
//   - Mixed-batch calibrated unified tile cost weights {10, 14, 19, 25} (1.00 : 1.40 : 1.90 : 2.50)
//     with exact round-to-nearest tile boundaries.
//   - Single boundary pass + shared v_b vector store when n_w2_blks == n_w13_blks (100% of Inkling 2
//     Mini layers: 9 == 9), cutting group_experts latency nearly in half vs 2-pass.
//   - O(G) 4-bucket counting sort orders groups by ascending cnt (1 -> 2 -> 3 -> 4) so each TPC core
//     executes a single cnt specialization (minimal I-cache thrashing).
//   - 4x-unrolled e_to_grp init + 4x-unrolled topk_ids loads + compact non-unrolled boundary/pack loops.
//
// Tensors (TPC dim order, dim0 fastest):
//   topk_ids    [topK, B] i32          (PyTorch [B, topK], B * topK <= 128)
//   group_meta  [128, 8] i32           (PyTorch [8, 128]; rows: 0=e, 1=cnt, 2..5=rs0..3,
//                                       6=w13_core_bounds[0..num_cores], 7=w2_core_bounds[0..num_cores])
//   group_count [2] i32                (PyTorch [2]; [G, total_w])
//
// Index space: [1] (single TPC core, ZERO SFU LUTs).

#define MAX_EXPERTS 256
#define MAX_DRAWS 128
#define MAX_CORES 64

void main(tensor topk_ids, tensor group_meta, tensor group_count,
          int num_experts, int n_w13_blks, int n_w2_blks, int num_cores)
{
    const int topK = get_dim_size(topk_ids, 0);
    const int B = get_dim_size(topk_ids, 1);

    int e_to_grp[MAX_EXPERTS];
    int ids_slm[MAX_DRAWS];
    int raw_e[MAX_DRAWS];
    int raw_cnt[MAX_DRAWS];
    int raw_rs[MAX_DRAWS][4];
    int order[MAX_DRAWS];
    int sorted_cnt[MAX_DRAWS];
    int b_out[2][MAX_CORES + 1];

    #pragma clang loop unroll(disable)
    for (int b = 0; b < B; b += 4)
    {
        int5 pf = {0, b, 0, 0, 0};
        prefetch(gen_addr(pf, topk_ids));
    }

    const int E4 = num_experts & ~3;
    #pragma clang loop unroll(disable)
    for (int e = 0; e < E4; e += 4)
    {
        e_to_grp[e + 0] = -1;
        e_to_grp[e + 1] = -1;
        e_to_grp[e + 2] = -1;
        e_to_grp[e + 3] = -1;
    }
    #pragma clang loop unroll(disable)
    for (int e = E4; e < num_experts; e++)
        e_to_grp[e] = -1;

    const int topK4 = topK & ~3;
    int M = 0;
    #pragma clang loop unroll(disable)
    for (int b = 0; b < B; b++)
    {
        int5 c0 = {0, b, 0, 0, 0}, c1 = {1, b, 0, 0, 0}, c2 = {2, b, 0, 0, 0}, c3 = {3, b, 0, 0, 0};
        #pragma clang loop unroll(disable)
        for (int s = 0; s < topK4; s += 4)
        {
            const int v0 = s_i32_ld_g((__global__ int*)gen_addr(c0, topk_ids));
            const int v1 = s_i32_ld_g((__global__ int*)gen_addr(c1, topk_ids));
            const int v2 = s_i32_ld_g((__global__ int*)gen_addr(c2, topk_ids));
            const int v3 = s_i32_ld_g((__global__ int*)gen_addr(c3, topk_ids));
            c0[0] += 4; c1[0] += 4; c2[0] += 4; c3[0] += 4;
            ids_slm[M + 0] = v0; ids_slm[M + 1] = v1;
            ids_slm[M + 2] = v2; ids_slm[M + 3] = v3;
            M += 4;
        }
        #pragma clang loop unroll(disable)
        for (int s = topK4; s < topK; s++)
        {
            c0[0] = s;
            ids_slm[M++] = s_i32_ld_g((__global__ int*)gen_addr(c0, topk_ids));
        }
    }

    int num_groups = 0;
    #pragma clang loop unroll(disable)
    for (int rs = 0; rs < M; rs++)
    {
        int e = ids_slm[rs];
        if (e < 0 || e >= num_experts)
            e = 0;
        int g = e_to_grp[e];
        if (g < 0 || raw_cnt[g] >= 4)
        {
            g = num_groups++;
            e_to_grp[e] = g;
            raw_e[g] = e;
            raw_cnt[g] = 0;
        }
        raw_rs[g][raw_cnt[g]++] = rs;
    }

    // O(G) 4-bucket counting sort by ascending cnt (1, 2, 3, 4).
    int bc[4] = {0, 0, 0, 0};
    #pragma clang loop unroll(disable)
    for (int g = 0; g < num_groups; g++)
        bc[raw_cnt[g] - 1]++;

    int off[4];
    off[0] = 0;
    off[1] = bc[0];
    off[2] = off[1] + bc[1];
    off[3] = off[2] + bc[2];

    const int w_lut[5] = {0, 10, 14, 19, 25};
    int total_w = 0;
    #pragma clang loop unroll(disable)
    for (int g = 0; g < num_groups; g++)
    {
        const int c = raw_cnt[g];
        const int dst = off[c - 1]++;
        order[dst] = g;
        sorted_cnt[dst] = c;
        total_w += w_lut[c];
    }

    // Single boundary pass when n_w2_blks == n_w13_blks (Inkling 2 Mini: 9 == 9).
    const int same_blks = (n_w2_blks == n_w13_blks) ? 1 : 0;
    const int n_passes = same_blks ? 1 : 2;
    const int blks_arr[2] = {n_w13_blks, n_w2_blks};
    const float inv_nc = (num_cores == 24) ? 0.0416666679f : (1.0f / (float)num_cores);

    #pragma clang loop unroll(disable)
    for (int pass = 0; pass < n_passes; pass++)
    {
        const int n_blks = blks_arr[pass];
        const int total_wt = total_w * n_blks;
        const int total_tiles = num_groups * n_blks;
        b_out[pass][0] = 0;

        int q_s = (int)((float)total_wt * inv_nc);
        int r_s = total_wt - q_s * num_cores;
        #pragma clang loop unroll(disable)
        while (r_s < 0)
        {
            q_s--;
            r_s += num_cores;
        }
        #pragma clang loop unroll(disable)
        while (r_s >= num_cores)
        {
            q_s++;
            r_s -= num_cores;
        }

        int g = 0, cw = 0;
        int w = (num_groups > 0) ? w_lut[sorted_cnt[0]] : 1;
        int target = 0, rem_c = 0, prev_b = 0;
        #pragma clang loop unroll(disable)
        for (int c = 1; c < num_cores; c++)
        {
            target += q_s;
            rem_c += r_s;
            if (rem_c >= num_cores)
            {
                rem_c -= num_cores;
                target++;
            }
            #pragma clang loop unroll(disable)
            while (g < num_groups && (cw + w) * n_blks <= target)
            {
                cw += w;
                g++;
                if (g < num_groups)
                    w = w_lut[sorted_cnt[g]];
            }
            int b_val;
            if (g >= num_groups)
                b_val = total_tiles;
            else
            {
                int rem = target - cw * n_blks;
                int blk = 0;
                const int half_w = w >> 1;
                #pragma clang loop unroll(disable)
                while (rem + half_w >= w && blk < n_blks)
                {
                    rem -= w;
                    blk++;
                }
                b_val = g * n_blks + blk;
            }
            if (b_val < prev_b)
                b_val = prev_b;
            if (b_val > total_tiles)
                b_val = total_tiles;
            b_out[pass][c] = b_val;
            prev_b = b_val;
        }
        b_out[pass][num_cores] = total_tiles;
    }

    const int64 lane = (int64)read_lane_id_4b_b();
    const int n_chunks = (num_groups > 64) ? 2 : 1;
    #pragma clang loop unroll(disable)
    for (int ch = 0; ch < n_chunks; ch++)
    {
        const int g_base = ch << 6;
        int g_end = g_base + 64;
        if (g_end > num_groups)
            g_end = num_groups;
        int64 v_e = 0, v_cnt = 0, v_r0 = 0, v_r1 = 0, v_r2 = 0, v_r3 = 0;
        #pragma clang loop unroll(disable)
        for (int g = g_base; g < g_end; g++)
        {
            const int src = order[g];
            const int cnt = raw_cnt[src];
            const int r0 = raw_rs[src][0];
            const int r1 = (cnt > 1) ? raw_rs[src][1] : r0;
            const int r2 = (cnt > 2) ? raw_rs[src][2] : r0;
            const int r3 = (cnt > 3) ? raw_rs[src][3] : r0;
            const bool64 m = v_i32_cmp_eq_b(lane, g - g_base);
            v_e = v_i32_mov_vb(raw_e[src], 0, v_e, m, 0);
            v_cnt = v_i32_mov_vb(cnt, 0, v_cnt, m, 0);
            v_r0 = v_i32_mov_vb(r0, 0, v_r0, m, 0);
            v_r1 = v_i32_mov_vb(r1, 0, v_r1, m, 0);
            v_r2 = v_i32_mov_vb(r2, 0, v_r2, m, 0);
            v_r3 = v_i32_mov_vb(r3, 0, v_r3, m, 0);
        }
        int5 oc = {g_base, 0, 0, 0, 0};
        v_i32_st_tnsr(oc, group_meta, v_e);
        oc[1] = 1; v_i32_st_tnsr(oc, group_meta, v_cnt);
        oc[1] = 2; v_i32_st_tnsr(oc, group_meta, v_r0);
        oc[1] = 3; v_i32_st_tnsr(oc, group_meta, v_r1);
        oc[1] = 4; v_i32_st_tnsr(oc, group_meta, v_r2);
        oc[1] = 5; v_i32_st_tnsr(oc, group_meta, v_r3);
    }

    {
        int64 v_b13 = 0, v_b2 = 0;
        const int c_end = (num_cores + 1 < 64) ? (num_cores + 1) : 64;
        if (same_blks)
        {
            #pragma clang loop unroll(disable)
            for (int c = 0; c < c_end; c++)
                v_b13 = v_i32_mov_vb(b_out[0][c], 0, v_b13, v_i32_cmp_eq_b(lane, c), 0);
            v_b2 = v_b13;
        }
        else
        {
            #pragma clang loop unroll(disable)
            for (int c = 0; c < c_end; c++)
            {
                const bool64 m = v_i32_cmp_eq_b(lane, c);
                v_b13 = v_i32_mov_vb(b_out[0][c], 0, v_b13, m, 0);
                v_b2 = v_i32_mov_vb(b_out[1][c], 0, v_b2, m, 0);
            }
        }
        int5 oc = {0, 6, 0, 0, 0};
        v_i32_st_tnsr(oc, group_meta, v_b13);
        oc[1] = 7;
        v_i32_st_tnsr(oc, group_meta, v_b2);
    }
    if (num_cores >= 64)
    {
        int5 oc = {64, 6, 0, 0, 0};
        v_i32_st_tnsr_partial(oc, group_meta, (int64)b_out[0][64], 0, 0, 0, 1, 0);
        oc[1] = 7;
        v_i32_st_tnsr_partial(oc, group_meta, (int64)b_out[same_blks ? 0 : 1][64], 0, 0, 0, 1, 0);
    }

    int64 v_gc = v_i32_mov_vb(num_groups, 0, (int64)0, v_i32_cmp_eq_b(lane, 0), 0);
    v_gc = v_i32_mov_vb(total_w, 0, v_gc, v_i32_cmp_eq_b(lane, 1), 0);
    int5 cc = {0, 0, 0, 0, 0};
    v_i32_st_tnsr_partial(cc, group_count, v_gc, 1, 0, 0, 1, 0);
}
