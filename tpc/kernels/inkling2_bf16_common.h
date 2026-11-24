// SPDX-License-Identifier: Apache-2.0
// Shared natural-order BF16 <-> FP32, FP8 store, and zero-LUT math/reduction helpers for Gaudi2 TPC kernels.
#pragma once

static inline float64 rsqrt_nolut(float64 x)
{
    // IEEE-754 fast inverse sqrt initial estimate + 2 Newton-Raphson iterations ->
    // max rel err ~4.5e-6 (~18 bits, 1700x finer than BF16) with ZERO SFU LUT usage & minimal code size.
    union {
        float64 f;
        uint64 u;
    } cvt;
    cvt.f = x;
    cvt.u = (uint64)0x5f375a86u - v_u32_shr_b(cvt.u, 1);
    float64 y = cvt.f;
    const float64 half_x = 0.5f * x;
    y = y * (1.5f - (half_x * y) * y);
    y = y * (1.5f - (half_x * y) * y);
    return y;
}

static inline float64 rcp_pos_nolut(float64 x)
{
    // IEEE-754 fast reciprocal initial estimate + 3 Newton-Raphson iterations ->
    // full 24-bit FP32 accuracy (< 1 ULP across all normal x > 0) with ZERO SFU LUT usage.
    union {
        float64 f;
        uint64 u;
    } cvt;
    cvt.f = x;
    cvt.u = (uint64)0x7ef311c3u - cvt.u;
    float64 y = cvt.f;
    y = y * (2.0f - x * y);
    y = y * (2.0f - x * y);
    y = y * (2.0f - x * y);
    return y;
}

static inline float64 sigmoid_nolut(float64 x)
{
    // Stable sigmoid using ONLY exp_f32 (0x100) + zero-LUT rcp_pos_nolut (no log_f32).
    const float64 clamped = v_f32_min_b(v_f32_max_b(-x, -80.0f), 80.0f);
    return rcp_pos_nolut(1.0f + exp_f32(clamped));
}

static inline float64 ld_bf16_as_f32(int5 coord, tensor t)
{
    // Hardware 16->32 unpack loads 64 contiguous bf16 elements into the low 16 bits
    // of 64 32-bit lanes in natural order (0..63); shifting left by 16 yields exact fp32.
    const uint64 u = v_u32_ld_tnsr_b(coord, t, SW_UNPACK | SW_UNPCK_16_TO_32, 0, 1, 0);
    union {
        uint64 u;
        float64 f;
    } cvt;
    cvt.u = v_u32_shl_b(u, 16);
    return cvt.f;
}

static inline void st_f32_as_bf16(int5 coord, tensor t, float64 val)
{
    // Convert 64 fp32 lanes to bf16 (rhne) in low 16 bits of each 32-bit lane, then
    // hardware 32->16 pack with partial (size=63 -> 64 elements, offset=0) stores ONLY
    // lanes 0..63 without touching elements 64..127 (safe across multi-core slicing).
    const bfloat128 b = v_convert_f32_to_bf16_b(val, 0, 0, 0, 1, 0);
    v_bf16_st_tnsr_partial(coord, t, b, 63, 0, SW_PACK | SW_PCK_32_TO_16, 1, 0);
}

static inline void st_f32_as_f8(int5 coord, tensor t, float64 val)
{
    // Convert 64 fp32 lanes to f8_143 (rhne) in low 8 bits of each 32-bit lane, then
    // hardware 32->8 pack with partial (size=63 -> 64 bytes, offset=0) stores ONLY
    // lanes 0..63 in natural order without touching neighboring elements.
    const minifloat256 f8 = v_convert_f32_to_f8_b(val, 0, 0, 0, 1, 0);
    v_f8_st_tnsr_partial(coord, t, f8, 63, 0, SW_PACK | SW_PCK_32_TO_8, 1, 0);
}

static inline float64 round_f32_via_bf16(float64 v)
{
    const bfloat128 b = v_convert_f32_to_bf16_b(v, 0, 0, 0, 1, 0);
    union {
        bfloat128 b;
        uint64 u;
        float64 f;
    } cvt;
    cvt.b = b;
    cvt.u = v_u32_shl_b(cvt.u, 16);
    return cvt.f;
}

static inline minifloat bits_to_f8(unsigned int b)
{
    union {
        unsigned char u;
        minifloat f;
    } cvt;
    cvt.u = (unsigned char)b;
    return cvt.f;
}

static inline float u32_bits_to_f32(unsigned int u)
{
    union {
        unsigned int u;
        float f;
    } cvt;
    cvt.u = u;
    return cvt.f;
}

static inline float64 softcapped_situ_vec(float64 g, float64 u, float inv_gcap,
                                          float inv_ucap, float scale_gamma)
{
    const float64 tg = tanh_f32(g * inv_gcap);
    const float64 sg = v_sigmoid_f32(g);
    const float64 tu = tanh_f32(u * inv_ucap);
    return (tg * sg) * (tu * scale_gamma);
}

// ---- Zero-LUT 6-step butterfly masks & multi-vector reductions (fill 4-cycle VPU shadow) ----

#define TPC_SW_MOV_G_ALL \
    (SW_GROUP0_EN | SW_GROUP1_EN | SW_DUAL_GROUP0_EN | SW_DUAL_GROUP1_EN | SW_DUAL_GROUP2_EN | SW_DUAL_GROUP3_EN)
#define TPC_SW_MOV_DG_ALL \
    (SW_WR_LOWER_GROUP0 | SW_WR_UPPER_GROUP0 | SW_WR_LOWER_GROUP1 | SW_WR_UPPER_GROUP1 | \
     SW_WR_LOWER_GROUP2 | SW_WR_UPPER_GROUP2 | SW_WR_LOWER_GROUP3 | SW_WR_UPPER_GROUP3)

typedef struct
{
    uchar256 m1;
    uchar256 m2;
    uchar256 m4;
} tpc_xor_masks_t;

static inline uchar256 u32_byte_to_shuffle_mask(uint64 b)
{
    // Replicate low byte across all 4 bytes of each 32-bit lane for shuffle.f32 / shuffle.i32.
    union {
        uint64 u;
        uchar256 c;
    } cvt;
    const uint64 b2 = b | v_u32_shl_b(b, 8);
    cvt.u = b2 | v_u32_shl_b(b2, 16);
    return cvt.c;
}

static inline tpc_xor_masks_t make_xor_masks_nolut(uint64 lid)
{
    // Gaudi2 shuffle.f32 byte format within each 16-lane dual-group: 0x80 | (group << 5) | local_lane.
    const uint64 g_bits = v_u32_shl_b(v_u32_and_b(v_u32_shr_b(lid, 3), 1u), 5) | (uint64)0x80u;
    const uint64 r_bits = lid & (uint64)7u;
    tpc_xor_masks_t out;
    out.m1 = u32_byte_to_shuffle_mask(g_bits | (r_bits ^ (uint64)1u));
    out.m2 = u32_byte_to_shuffle_mask(g_bits | (r_bits ^ (uint64)2u));
    out.m4 = u32_byte_to_shuffle_mask(g_bits | (r_bits ^ (uint64)4u));
    return out;
}

static inline float64_pair_t reduce_add_pair_nolut(float64 a, float64 b, tpc_xor_masks_t m)
{
    // Interleaved 2-vector sum reduction: executes both trees in one 6-step latency chain.
    a += v_f32_mov_dual_group_all_b(a, 0xFFFFFFFFu, 1, 0, 3, 2, TPC_SW_MOV_DG_ALL, a, 1, 0);
    b += v_f32_mov_dual_group_all_b(b, 0xFFFFFFFFu, 1, 0, 3, 2, TPC_SW_MOV_DG_ALL, b, 1, 0);
    a += v_f32_mov_dual_group_all_b(a, 0xFFFFFFFFu, 2, 3, 0, 1, TPC_SW_MOV_DG_ALL, a, 1, 0);
    b += v_f32_mov_dual_group_all_b(b, 0xFFFFFFFFu, 2, 3, 0, 1, TPC_SW_MOV_DG_ALL, b, 1, 0);
    a += v_f32_mov_group_b(a, 0xFFFFFFFFu, TPC_SW_MOV_G_ALL, a, 1, 0);
    b += v_f32_mov_group_b(b, 0xFFFFFFFFu, TPC_SW_MOV_G_ALL, b, 1, 0);
    a += v_f32_shuffle_b(a, m.m4, 0, a, 1, 0);
    b += v_f32_shuffle_b(b, m.m4, 0, b, 1, 0);
    a += v_f32_shuffle_b(a, m.m2, 0, a, 1, 0);
    b += v_f32_shuffle_b(b, m.m2, 0, b, 1, 0);
    a += v_f32_shuffle_b(a, m.m1, 0, a, 1, 0);
    b += v_f32_shuffle_b(b, m.m1, 0, b, 1, 0);
    float64_pair_t res = {a, b};
    return res;
}

static inline float64 reduce_add_fast_nolut(float64 a, tpc_xor_masks_t m)
{
    a += v_f32_mov_dual_group_all_b(a, 0xFFFFFFFFu, 1, 0, 3, 2, TPC_SW_MOV_DG_ALL, a, 1, 0);
    a += v_f32_mov_dual_group_all_b(a, 0xFFFFFFFFu, 2, 3, 0, 1, TPC_SW_MOV_DG_ALL, a, 1, 0);
    a += v_f32_mov_group_b(a, 0xFFFFFFFFu, TPC_SW_MOV_G_ALL, a, 1, 0);
    a += v_f32_shuffle_b(a, m.m4, 0, a, 1, 0);
    a += v_f32_shuffle_b(a, m.m2, 0, a, 1, 0);
    a += v_f32_shuffle_b(a, m.m1, 0, a, 1, 0);
    return a;
}

static inline float64 reduce_max_fast_nolut(float64 a, tpc_xor_masks_t m)
{
    a = v_f32_max_b(a, v_f32_mov_dual_group_all_b(a, 0xFFFFFFFFu, 1, 0, 3, 2, TPC_SW_MOV_DG_ALL, a, 1, 0));
    a = v_f32_max_b(a, v_f32_mov_dual_group_all_b(a, 0xFFFFFFFFu, 2, 3, 0, 1, TPC_SW_MOV_DG_ALL, a, 1, 0));
    a = v_f32_max_b(a, v_f32_mov_group_b(a, 0xFFFFFFFFu, TPC_SW_MOV_G_ALL, a, 1, 0));
    a = v_f32_max_b(a, v_f32_shuffle_b(a, m.m4, 0, a, 1, 0));
    a = v_f32_max_b(a, v_f32_shuffle_b(a, m.m2, 0, a, 1, 0));
    a = v_f32_max_b(a, v_f32_shuffle_b(a, m.m1, 0, a, 1, 0));
    return a;
}

typedef struct
{
    float64 val;
    int64 idx;
    float64 sig;
} tpc_argmax_triple_t;

static inline tpc_argmax_triple_t argmax_merge_step(tpc_argmax_triple_t a,
                                                    float64 v2, int64 i2, float64 s2)
{
    // Exact lowest-id tie-break: take candidate 2 iff (v2 > a.val) || (v2 == a.val && i2 < a.idx).
    const bool64 gt = v_f32_cmp_grt_b(v2, a.val);
    const bool64 eq = v_f32_cmp_eq_b(v2, a.val);
    const bool64 lt = v_i32_cmp_less_b(i2, a.idx);
    const bool64 take = gt | (eq & lt);
    a.val = v_f32_mov_vb(v2, 0, a.val, take, 0);
    a.idx = v_i32_mov_vb(i2, 0, a.idx, take, 0);
    a.sig = v_f32_mov_vb(s2, 0, a.sig, take, 0);
    return a;
}

static inline tpc_argmax_triple_t reduce_argmax_sig_nolut(tpc_argmax_triple_t a, tpc_xor_masks_t m)
{
    // Single 6-step butterfly carrying (val, idx, sig) in parallel -> 3x fewer cycles than 3 trees!
    {
        const float64 v2 = v_f32_mov_dual_group_all_b(a.val, 0xFFFFFFFFu, 1, 0, 3, 2, TPC_SW_MOV_DG_ALL, a.val, 1, 0);
        const int64 i2 = v_i32_mov_dual_group_all_b(a.idx, 0xFFFFFFFFu, 1, 0, 3, 2, TPC_SW_MOV_DG_ALL, a.idx, 1, 0);
        const float64 s2 = v_f32_mov_dual_group_all_b(a.sig, 0xFFFFFFFFu, 1, 0, 3, 2, TPC_SW_MOV_DG_ALL, a.sig, 1, 0);
        a = argmax_merge_step(a, v2, i2, s2);
    }
    {
        const float64 v2 = v_f32_mov_dual_group_all_b(a.val, 0xFFFFFFFFu, 2, 3, 0, 1, TPC_SW_MOV_DG_ALL, a.val, 1, 0);
        const int64 i2 = v_i32_mov_dual_group_all_b(a.idx, 0xFFFFFFFFu, 2, 3, 0, 1, TPC_SW_MOV_DG_ALL, a.idx, 1, 0);
        const float64 s2 = v_f32_mov_dual_group_all_b(a.sig, 0xFFFFFFFFu, 2, 3, 0, 1, TPC_SW_MOV_DG_ALL, a.sig, 1, 0);
        a = argmax_merge_step(a, v2, i2, s2);
    }
    {
        const float64 v2 = v_f32_mov_group_b(a.val, 0xFFFFFFFFu, TPC_SW_MOV_G_ALL, a.val, 1, 0);
        const int64 i2 = v_i32_mov_group_b(a.idx, 0xFFFFFFFFu, TPC_SW_MOV_G_ALL, a.idx, 1, 0);
        const float64 s2 = v_f32_mov_group_b(a.sig, 0xFFFFFFFFu, TPC_SW_MOV_G_ALL, a.sig, 1, 0);
        a = argmax_merge_step(a, v2, i2, s2);
    }
    {
        const float64 v2 = v_f32_shuffle_b(a.val, m.m4, 0, a.val, 1, 0);
        const int64 i2 = v_i32_shuffle_b(a.idx, m.m4, 0, a.idx, 1, 0);
        const float64 s2 = v_f32_shuffle_b(a.sig, m.m4, 0, a.sig, 1, 0);
        a = argmax_merge_step(a, v2, i2, s2);
    }
    {
        const float64 v2 = v_f32_shuffle_b(a.val, m.m2, 0, a.val, 1, 0);
        const int64 i2 = v_i32_shuffle_b(a.idx, m.m2, 0, a.idx, 1, 0);
        const float64 s2 = v_f32_shuffle_b(a.sig, m.m2, 0, a.sig, 1, 0);
        a = argmax_merge_step(a, v2, i2, s2);
    }
    {
        const float64 v2 = v_f32_shuffle_b(a.val, m.m1, 0, a.val, 1, 0);
        const int64 i2 = v_i32_shuffle_b(a.idx, m.m1, 0, a.idx, 1, 0);
        const float64 s2 = v_f32_shuffle_b(a.sig, m.m1, 0, a.sig, 1, 0);
        a = argmax_merge_step(a, v2, i2, s2);
    }
    return a;
}

typedef struct
{
    float64 v1;
    float64 v2;
    float64 v3;
} float64_triple_t;

static inline float64_triple_t reduce_add_triple_nolut(float64 a, float64 b, float64 c, tpc_xor_masks_t m)
{
    // Interleaved 3-vector sum reduction: executes all 3 GQA-3 head trees in one 6-step latency chain.
    a += v_f32_mov_dual_group_all_b(a, 0xFFFFFFFFu, 1, 0, 3, 2, TPC_SW_MOV_DG_ALL, a, 1, 0);
    b += v_f32_mov_dual_group_all_b(b, 0xFFFFFFFFu, 1, 0, 3, 2, TPC_SW_MOV_DG_ALL, b, 1, 0);
    c += v_f32_mov_dual_group_all_b(c, 0xFFFFFFFFu, 1, 0, 3, 2, TPC_SW_MOV_DG_ALL, c, 1, 0);
    a += v_f32_mov_dual_group_all_b(a, 0xFFFFFFFFu, 2, 3, 0, 1, TPC_SW_MOV_DG_ALL, a, 1, 0);
    b += v_f32_mov_dual_group_all_b(b, 0xFFFFFFFFu, 2, 3, 0, 1, TPC_SW_MOV_DG_ALL, b, 1, 0);
    c += v_f32_mov_dual_group_all_b(c, 0xFFFFFFFFu, 2, 3, 0, 1, TPC_SW_MOV_DG_ALL, c, 1, 0);
    a += v_f32_mov_group_b(a, 0xFFFFFFFFu, TPC_SW_MOV_G_ALL, a, 1, 0);
    b += v_f32_mov_group_b(b, 0xFFFFFFFFu, TPC_SW_MOV_G_ALL, b, 1, 0);
    c += v_f32_mov_group_b(c, 0xFFFFFFFFu, TPC_SW_MOV_G_ALL, c, 1, 0);
    a += v_f32_shuffle_b(a, m.m4, 0, a, 1, 0);
    b += v_f32_shuffle_b(b, m.m4, 0, b, 1, 0);
    c += v_f32_shuffle_b(c, m.m4, 0, c, 1, 0);
    a += v_f32_shuffle_b(a, m.m2, 0, a, 1, 0);
    b += v_f32_shuffle_b(b, m.m2, 0, b, 1, 0);
    c += v_f32_shuffle_b(c, m.m2, 0, c, 1, 0);
    a += v_f32_shuffle_b(a, m.m1, 0, a, 1, 0);
    b += v_f32_shuffle_b(b, m.m1, 0, b, 1, 0);
    c += v_f32_shuffle_b(c, m.m1, 0, c, 1, 0);
    float64_triple_t res = {a, b, c};
    return res;
}
