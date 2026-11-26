// SPDX-License-Identifier: Apache-2.0
// Zero-LUT FP32 exp/log/sigmoid/silu/softplus, VPU-pipelined 2x/4x BF16-precision SiLU,
// and BF16-in-VLM packing helpers for Gaudi2 TPC kernels.
// Using zero SFU LUTs keeps the full 64 KiB VLM available and avoids all LUT cache evictions.
#pragma once

#include "inkling2_bf16_common.h"

typedef struct
{
    float64 v0;
    float64 v1;
    float64 v2;
    float64 v3;
} float64_quad_t;

static inline float64 exp_f32_nolut(float64 x)
{
    // Range-reduced 2^(x * log2(e)) with degree-5 Taylor polynomial on [-0.5, 0.5]:
    // max relative error ~2.4e-6 across [-87, 87], ZERO SFU LUT usage.
    const float64 xc = v_f32_min_b(v_f32_max_b(x, -87.0f), 87.0f);
    const float64 t = xc * 1.4426950408889634f;
    const int64 n = v_convert_f32_to_i32_b(t, 0, 0, 1, 0);  // round-half-to-even
    const float64 nf = v_convert_i32_to_f32_b(n, 0, 0, 1, 0);
    const float64 f = t - nf;

    float64 p = 0.00133335581f;
    p = v_f32_mac_b(p, f, (float64)0.00961812911f);
    p = v_f32_mac_b(p, f, (float64)0.05550410866f);
    p = v_f32_mac_b(p, f, (float64)0.24022650696f);
    p = v_f32_mac_b(p, f, (float64)0.69314718056f);
    p = v_f32_mac_b(p, f, (float64)1.0f);

    union {
        uint64 u;
        float64 f;
    } sc;
    sc.u = v_u32_shl_b((uint64)(n + 127), 23);
    return p * sc.f;
}

static inline float64 log_f32_nolut(float64 x)
{
    // IEEE-754 mantissa extraction x = m * 2^e with m in [sqrt(0.5), sqrt(2)],
    // u = (m - 1) / (m + 1), ln(m) = 2*u*(1 + u^2/3 + u^4/5 + u^6/7).
    // Max absolute error < 2e-7 for positive normal FP32, ZERO SFU LUT usage.
    union {
        float64 f;
        uint64 u;
    } cvt;
    cvt.f = x;
    const uint64 adj = cvt.u + (uint64)(0x3f800000u - 0x3f3504f3u);
    const int64 e = (int64)v_u32_shr_b(adj, 23) - 127;
    cvt.u = (adj & (uint64)0x007fffffu) + (uint64)0x3f3504f3u;
    const float64 m = cvt.f;

    const float64 u = (m - 1.0f) * rcp_pos_nolut(m + 1.0f);
    const float64 u2 = u * u;
    float64 poly = 0.142857143f;                           // 1/7
    poly = v_f32_mac_b(poly, u2, (float64)0.200000000f);   // 1/5
    poly = v_f32_mac_b(poly, u2, (float64)0.333333333f);   // 1/3
    poly = v_f32_mac_b(poly, u2, (float64)1.0f);
    const float64 ln_m = (2.0f * u) * poly;
    const float64 ef = v_convert_i32_to_f32_b(e, 0, 0, 1, 0);
    return v_f32_mac_b(ef, (float64)0.69314718056f, ln_m);
}

static inline float64 sigmoid_zero_lut(float64 x)
{
    return rcp_pos_nolut(1.0f + exp_f32_nolut(-x));
}

static inline float64 softplus_zero_lut(float64 x)
{
    // Match PyTorch softplus threshold (beta=1, threshold=20): x > 20 ? x : log(1 + exp(x)).
    const float64 sp = log_f32_nolut(1.0f + exp_f32_nolut(v_f32_min_b(x, 20.0f)));
    const bool64 gt = v_f32_cmp_grt_b(x, 20.0f);
    return v_f32_mov_vb(x, 0, sp, gt, 0);
}

// BF16-precision zero-LUT SiLU (max rel err < 5e-5, >150x finer than BF16, zero LUTs).
// Degree-4 2^f Taylor polynomial + 2-iteration Newton-Raphson reciprocal using fused negated MAC.
// Inputs < -87.0f flush cleanly to 0.0f so extreme finite negative BF16 / -inf never underflow-multiply.
static inline float64 silu_fast_bf16(float64 x)
{
    const float64 xc = v_f32_min_b(v_f32_max_b(-x, -87.0f), 87.0f);
    const float64 t = xc * 1.4426950408889634f;
    const int64 n = v_convert_f32_to_i32_b(t, 0, 0, 1, 0);
    const float64 nf = v_convert_i32_to_f32_b(n, 0, 0, 1, 0);
    const float64 f = t - nf;

    float64 p = 0.00961812911f;
    p = v_f32_mac_b(p, f, (float64)0.05550410866f);
    p = v_f32_mac_b(p, f, (float64)0.24022650696f);
    p = v_f32_mac_b(p, f, (float64)0.69314718056f);
    p = v_f32_mac_b(p, f, (float64)1.0f);

    union {
        uint64 u;
        float64 f;
    } sc, r;
    sc.u = v_u32_shl_b((uint64)(n + 127), 23);
    const float64 d = v_f32_mac_b(p, sc.f, (float64)1.0f);

    r.f = d;
    r.u = (uint64)0x7ef311c3u - r.u;
    float64 y = r.f;
    y = y * v_f32_mac_b(d, y, (float64)2.0f, SW_NEG);
    y = y * v_f32_mac_b(d, y, (float64)2.0f, SW_NEG);
    const bool64 under = v_f32_cmp_less_b(x, -87.0f);
    return v_f32_mov_vb((float64)0.0f, 0, x * y, under, 0);
}

// 4-vector interleaved BF16-precision zero-LUT SiLU: hides Gaudi2 4-6 cycle VPU latency (~3x faster).
static inline float64_quad_t silu_fast_bf16_x4(float64 x0, float64 x1, float64 x2, float64 x3)
{
    const float64 xc0 = v_f32_min_b(v_f32_max_b(-x0, -87.0f), 87.0f);
    const float64 xc1 = v_f32_min_b(v_f32_max_b(-x1, -87.0f), 87.0f);
    const float64 xc2 = v_f32_min_b(v_f32_max_b(-x2, -87.0f), 87.0f);
    const float64 xc3 = v_f32_min_b(v_f32_max_b(-x3, -87.0f), 87.0f);

    const float64 t0 = xc0 * 1.4426950408889634f;
    const float64 t1 = xc1 * 1.4426950408889634f;
    const float64 t2 = xc2 * 1.4426950408889634f;
    const float64 t3 = xc3 * 1.4426950408889634f;

    const int64 n0 = v_convert_f32_to_i32_b(t0, 0, 0, 1, 0);
    const int64 n1 = v_convert_f32_to_i32_b(t1, 0, 0, 1, 0);
    const int64 n2 = v_convert_f32_to_i32_b(t2, 0, 0, 1, 0);
    const int64 n3 = v_convert_f32_to_i32_b(t3, 0, 0, 1, 0);

    const float64 nf0 = v_convert_i32_to_f32_b(n0, 0, 0, 1, 0);
    const float64 nf1 = v_convert_i32_to_f32_b(n1, 0, 0, 1, 0);
    const float64 nf2 = v_convert_i32_to_f32_b(n2, 0, 0, 1, 0);
    const float64 nf3 = v_convert_i32_to_f32_b(n3, 0, 0, 1, 0);

    const float64 f0 = t0 - nf0;
    const float64 f1 = t1 - nf1;
    const float64 f2 = t2 - nf2;
    const float64 f3 = t3 - nf3;

    const float64 c4 = 0.00961812911f;
    float64 p0 = v_f32_mac_b(c4, f0, (float64)0.05550410866f);
    float64 p1 = v_f32_mac_b(c4, f1, (float64)0.05550410866f);
    float64 p2 = v_f32_mac_b(c4, f2, (float64)0.05550410866f);
    float64 p3 = v_f32_mac_b(c4, f3, (float64)0.05550410866f);

    p0 = v_f32_mac_b(p0, f0, (float64)0.24022650696f);
    p1 = v_f32_mac_b(p1, f1, (float64)0.24022650696f);
    p2 = v_f32_mac_b(p2, f2, (float64)0.24022650696f);
    p3 = v_f32_mac_b(p3, f3, (float64)0.24022650696f);

    p0 = v_f32_mac_b(p0, f0, (float64)0.69314718056f);
    p1 = v_f32_mac_b(p1, f1, (float64)0.69314718056f);
    p2 = v_f32_mac_b(p2, f2, (float64)0.69314718056f);
    p3 = v_f32_mac_b(p3, f3, (float64)0.69314718056f);

    p0 = v_f32_mac_b(p0, f0, (float64)1.0f);
    p1 = v_f32_mac_b(p1, f1, (float64)1.0f);
    p2 = v_f32_mac_b(p2, f2, (float64)1.0f);
    p3 = v_f32_mac_b(p3, f3, (float64)1.0f);

    union {
        uint64 u;
        float64 f;
    } s0, s1, s2, s3;
    s0.u = v_u32_shl_b((uint64)(n0 + 127), 23);
    s1.u = v_u32_shl_b((uint64)(n1 + 127), 23);
    s2.u = v_u32_shl_b((uint64)(n2 + 127), 23);
    s3.u = v_u32_shl_b((uint64)(n3 + 127), 23);

    const float64 d0 = v_f32_mac_b(p0, s0.f, (float64)1.0f);
    const float64 d1 = v_f32_mac_b(p1, s1.f, (float64)1.0f);
    const float64 d2 = v_f32_mac_b(p2, s2.f, (float64)1.0f);
    const float64 d3 = v_f32_mac_b(p3, s3.f, (float64)1.0f);

    s0.f = d0; s1.f = d1; s2.f = d2; s3.f = d3;
    s0.u = (uint64)0x7ef311c3u - s0.u;
    s1.u = (uint64)0x7ef311c3u - s1.u;
    s2.u = (uint64)0x7ef311c3u - s2.u;
    s3.u = (uint64)0x7ef311c3u - s3.u;

    float64 y0 = s0.f, y1 = s1.f, y2 = s2.f, y3 = s3.f;
    const float64 e0 = v_f32_mac_b(d0, y0, (float64)2.0f, SW_NEG);
    const float64 e1 = v_f32_mac_b(d1, y1, (float64)2.0f, SW_NEG);
    const float64 e2 = v_f32_mac_b(d2, y2, (float64)2.0f, SW_NEG);
    const float64 e3 = v_f32_mac_b(d3, y3, (float64)2.0f, SW_NEG);
    y0 = y0 * e0; y1 = y1 * e1; y2 = y2 * e2; y3 = y3 * e3;

    const float64 f_0 = v_f32_mac_b(d0, y0, (float64)2.0f, SW_NEG);
    const float64 f_1 = v_f32_mac_b(d1, y1, (float64)2.0f, SW_NEG);
    const float64 f_2 = v_f32_mac_b(d2, y2, (float64)2.0f, SW_NEG);
    const float64 f_3 = v_f32_mac_b(d3, y3, (float64)2.0f, SW_NEG);
    y0 = y0 * f_0; y1 = y1 * f_1; y2 = y2 * f_2; y3 = y3 * f_3;

    const bool64 u0 = v_f32_cmp_less_b(x0, -87.0f);
    const bool64 u1 = v_f32_cmp_less_b(x1, -87.0f);
    const bool64 u2 = v_f32_cmp_less_b(x2, -87.0f);
    const bool64 u3 = v_f32_cmp_less_b(x3, -87.0f);
    float64_quad_t out = {
        v_f32_mov_vb((float64)0.0f, 0, x0 * y0, u0, 0),
        v_f32_mov_vb((float64)0.0f, 0, x1 * y1, u1, 0),
        v_f32_mov_vb((float64)0.0f, 0, x2 * y2, u2, 0),
        v_f32_mov_vb((float64)0.0f, 0, x3 * y3, u3, 0),
    };
    return out;
}

static inline float64_pair_t silu_fast_bf16_x2(float64 x0, float64 x1)
{
    const float64 xc0 = v_f32_min_b(v_f32_max_b(-x0, -87.0f), 87.0f);
    const float64 xc1 = v_f32_min_b(v_f32_max_b(-x1, -87.0f), 87.0f);
    const float64 t0 = xc0 * 1.4426950408889634f;
    const float64 t1 = xc1 * 1.4426950408889634f;
    const int64 n0 = v_convert_f32_to_i32_b(t0, 0, 0, 1, 0);
    const int64 n1 = v_convert_f32_to_i32_b(t1, 0, 0, 1, 0);
    const float64 nf0 = v_convert_i32_to_f32_b(n0, 0, 0, 1, 0);
    const float64 nf1 = v_convert_i32_to_f32_b(n1, 0, 0, 1, 0);
    const float64 f0 = t0 - nf0;
    const float64 f1 = t1 - nf1;

    const float64 c4 = 0.00961812911f;
    float64 p0 = v_f32_mac_b(c4, f0, (float64)0.05550410866f);
    float64 p1 = v_f32_mac_b(c4, f1, (float64)0.05550410866f);
    p0 = v_f32_mac_b(p0, f0, (float64)0.24022650696f);
    p1 = v_f32_mac_b(p1, f1, (float64)0.24022650696f);
    p0 = v_f32_mac_b(p0, f0, (float64)0.69314718056f);
    p1 = v_f32_mac_b(p1, f1, (float64)0.69314718056f);
    p0 = v_f32_mac_b(p0, f0, (float64)1.0f);
    p1 = v_f32_mac_b(p1, f1, (float64)1.0f);

    union {
        uint64 u;
        float64 f;
    } s0, s1;
    s0.u = v_u32_shl_b((uint64)(n0 + 127), 23);
    s1.u = v_u32_shl_b((uint64)(n1 + 127), 23);
    const float64 d0 = v_f32_mac_b(p0, s0.f, (float64)1.0f);
    const float64 d1 = v_f32_mac_b(p1, s1.f, (float64)1.0f);

    s0.f = d0; s1.f = d1;
    s0.u = (uint64)0x7ef311c3u - s0.u;
    s1.u = (uint64)0x7ef311c3u - s1.u;
    float64 y0 = s0.f, y1 = s1.f;
    y0 = y0 * v_f32_mac_b(d0, y0, (float64)2.0f, SW_NEG);
    y1 = y1 * v_f32_mac_b(d1, y1, (float64)2.0f, SW_NEG);
    y0 = y0 * v_f32_mac_b(d0, y0, (float64)2.0f, SW_NEG);
    y1 = y1 * v_f32_mac_b(d1, y1, (float64)2.0f, SW_NEG);

    const bool64 u0 = v_f32_cmp_less_b(x0, -87.0f);
    const bool64 u1 = v_f32_cmp_less_b(x1, -87.0f);
    float64_pair_t out = {
        v_f32_mov_vb((float64)0.0f, 0, x0 * y0, u0, 0),
        v_f32_mov_vb((float64)0.0f, 0, x1 * y1, u1, 0),
    };
    return out;
}

// Convert two 64-lane FP32 vectors once to BF16 and merge into a single 128-lane bfloat128 slot.
// Even 16-bit lanes hold lo[0..63], odd 16-bit lanes hold hi[0..63], so v_bf16_abs_b/v_bf16_max_b
// process all 128 elements in a single VPU instruction and VLM traffic is halved.
static inline bfloat128 pack_two_f32_to_bf16_vec(float64 lo, float64 hi)
{
    union {
        bfloat128 b;
        uint64 u;
    } c0, c1, out;
    c0.b = v_convert_f32_to_bf16_b(lo, 0, 0, 0, 1, 0);
    c1.b = v_convert_f32_to_bf16_b(hi, 0, 0, 0, 1, 0);
    out.u = (c0.u & (uint64)0x0000ffffu) | v_u32_shl_b(c1.u, 16);
    return out.b;
}

// Merge two already-converted bfloat128 vectors (each holding 64 values in even 16-bit lanes).
static inline bfloat128 merge_two_bf16_lo_vecs(bfloat128 b0, bfloat128 b1)
{
    union {
        bfloat128 b;
        uint64 u;
    } c0, c1, out;
    c0.b = b0;
    c1.b = b1;
    out.u = (c0.u & (uint64)0x0000ffffu) | v_u32_shl_b(c1.u, 16);
    return out.b;
}

// Store 64 BF16 lanes already in the low 16 bits of each 32-bit lane (avoids a second f32->bf16 convert).
static inline void st_bf16_lo_vec(int5 coord, tensor t, bfloat128 b)
{
    v_bf16_st_tnsr_partial(coord, t, b, 63, 0, SW_PACK | SW_PCK_32_TO_16, 1, 0);
}

static inline float64 unpack_bf16_lo(bfloat128 packed)
{
    union {
        bfloat128 b;
        uint64 u;
        float64 f;
    } cvt;
    cvt.b = packed;
    cvt.u = v_u32_shl_b(cvt.u, 16);
    return cvt.f;
}

static inline float64 unpack_bf16_hi(bfloat128 packed)
{
    union {
        bfloat128 b;
        uint64 u;
        float64 f;
    } cvt;
    cvt.b = packed;
    cvt.u = cvt.u & (uint64)0xffff0000u;
    return cvt.f;
}

static inline float64 bf16_vec_max_to_f32(bfloat128 am)
{
    return v_f32_max_b(unpack_bf16_lo(am), unpack_bf16_hi(am));
}
