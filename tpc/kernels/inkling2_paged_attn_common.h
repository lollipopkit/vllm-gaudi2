// SPDX-License-Identifier: Apache-2.0
// Shared per-request paged KV block collector for Inkling 2 Mini Gaudi2 TPC decode attention kernels.
#pragma once

#define MAX_MATCH_BLKS 512

static __local__ int g_my_phys[MAX_MATCH_BLKS];
static __local__ int g_my_sbeg[MAX_MATCH_BLKS];
static __local__ int g_my_send[MAX_MATCH_BLKS];
static __local__ int g_my_spos[MAX_MATCH_BLKS];

static inline int collect_req_blocks(int b, int q_pos, int NB, int BS, int sliding_window,
                                     tensor block_list, tensor block_groups,
                                     tensor block_usage, tensor block_start_pos)
{
    int cnt = 0;
    int5 pf = {0, 0, 0, 0, 0};
    #pragma clang loop unroll(disable)
    for (int bi = 0; bi < NB; bi += 16)
    {
        pf[0] = bi;
        prefetch(gen_addr(pf, block_groups));
    }
    int5 mc = {0, 0, 0, 0, 0};
    #pragma clang loop unroll(disable)
    for (int bi = 0; bi < NB; bi++)
    {
        mc[0] = bi;
        if (s_i32_ld_g((__global__ int*)gen_addr(mc, block_groups)) != b || cnt >= MAX_MATCH_BLKS)
            continue;
        const int phys = s_i32_ld_g((__global__ int*)gen_addr(mc, block_list));
        int usage = s_i32_ld_g((__global__ int*)gen_addr(mc, block_usage));
        const int start_pos = s_i32_ld_g((__global__ int*)gen_addr(mc, block_start_pos));
        if (phys < 0 || usage <= 0 || start_pos >= q_pos)
            continue;
        if (usage > BS)
            usage = BS;
        int s_end = usage;
        const int max_s = q_pos - start_pos;  // strictly < q_pos (dist=0 comes from k_new/v_new)
        if (max_s < s_end)
            s_end = max_s;
        if (s_end <= 0)
            continue;
        int s_begin = 0;
        if (sliding_window > 0)
        {
            const int min_pos = q_pos - sliding_window + 1;
            if (min_pos > start_pos)
                s_begin = min_pos - start_pos;
            if (s_begin >= s_end)
                continue;
        }
        g_my_phys[cnt] = phys;
        g_my_sbeg[cnt] = s_begin;
        g_my_send[cnt] = s_end;
        g_my_spos[cnt] = start_pos;
        cnt++;
    }
    return cnt;
}
