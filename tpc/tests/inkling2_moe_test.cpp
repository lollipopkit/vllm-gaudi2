// SPDX-License-Identifier: Apache-2.0
// Simulator (or device, TPC_RUNNER=1) test for inkling2_moe_w13_situ_f8_gaudi2 and
// inkling2_moe_w2_f8_gaudi2 against a double-precision reference.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#include "habana/tensor.h"
#include "inkling2_moe_gaudi2.hpp"
#include "tpc_test_core_api.h"

extern "C" tpc_lib_api::GlueCodeReturn InstantiateTpcKernel(tpc_lib_api::HabanaKernelParams* params,
                                                            tpc_lib_api::HabanaKernelInstantiation* instance);

namespace
{
static float F8ToF32(uint8_t b)
{
    const int s = (b >> 7) & 1;
    const int e = (b >> 3) & 0xF;
    const int m = b & 0x7;
    if (e == 0 && m == 0)
        return 0.0f;
    const float v = (e == 0) ? (m / 8.0f) * std::ldexp(1.0f, -6) : (1.0f + m / 8.0f) * std::ldexp(1.0f, e - 7);
    return s ? -v : v;
}

static uint8_t RandF8(std::mt19937& rng)
{
    // Valid finite Gaudi2 f8_143 bytes (magnitude 0x08..0x67 -> ~0.0156..48.0, balanced signs).
    const uint32_t r = rng();
    return static_cast<uint8_t>((r & 0x80u) | (0x08u + (r % 0x60u)));
}

template <class T, int N>
void Describe(tpc_lib_api::Tensor* t, tpc_lib_api::TensorDataType dt, const test::Tensor<T, N>& src)
{
    t->geometry.dims = N;
    t->geometry.dataType = dt;
    for (int i = 0; i < N; i++)
        t->geometry.maxSizes[i] = t->geometry.minSizes[i] = src.Size(i);
}

int Run(int H, int I, int E, int topK, int B)
{
    const int twoI = 2 * I;
    const int nGroups = I / 128;
    const int nOutBlks = H / 256;
    const float gate_cap = 4.0f, up_cap = 25.0f, fp8_max = 240.0f;

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> u(-1.0f, 1.0f);
    std::uniform_int_distribution<int> edist(0, E - 1);

    const uint64_t xSz[] = {(uint64_t)H, (uint64_t)B};
    const uint64_t xI32Sz[] = {(uint64_t)(H / 4), (uint64_t)B};
    const uint64_t xsSz[] = {(uint64_t)B};
    const uint64_t w13Sz[] = {256u, (uint64_t)H, (uint64_t)nGroups, (uint64_t)E};
    const uint64_t s13Sz[] = {(uint64_t)twoI, (uint64_t)E};
    const uint64_t topkSz[] = {(uint64_t)topK, (uint64_t)B};
    const uint64_t actSz[] = {128u, (uint64_t)(nGroups * topK * B)};
    const uint64_t actI32Sz[] = {(uint64_t)(I / 4), (uint64_t)topK, (uint64_t)B};
    const uint64_t actScSz[] = {(uint64_t)(nGroups * topK * B)};
    const uint64_t actSc3Sz[] = {(uint64_t)nGroups, (uint64_t)topK, (uint64_t)B};
    const uint64_t w2Sz[] = {256u, (uint64_t)I, (uint64_t)nOutBlks, (uint64_t)E};
    const uint64_t s2Sz[] = {(uint64_t)H, (uint64_t)E};
    const uint64_t outSz[] = {256u, (uint64_t)(nOutBlks * topK * B)};

    test::Tensor<uint8_t, 2> x_f8(xSz);
    test::Tensor<int32_t, 2> x_i32(xI32Sz);
    test::Tensor<float, 1> x_scale(xsSz);
    test::Tensor<uint8_t, 4> w13_packed(w13Sz);
    test::Tensor<float, 2> s13_packed(s13Sz);
    test::Tensor<int32_t, 2> topk_ids(topkSz);
    test::Tensor<float, 2> topk_weights(topkSz);
    test::Tensor<uint8_t, 2> act_f8(actSz);
    test::Tensor<int32_t, 3> act_i32(actI32Sz);
    test::Tensor<float, 1> act_scale(actScSz);
    test::Tensor<float, 3> act_scale3(actSc3Sz);
    test::Tensor<uint8_t, 4> w2_packed(w2Sz);
    test::Tensor<float, 2> s2_packed(s2Sz);
    test::Tensor<uint16_t, 2> out_bf16(outSz);

    // Reference unpacked weights: w13_ref[e][2*I][H] (interleaved g,u), w2_ref[e][H][I]
    std::vector<uint8_t> w13_ref((size_t)E * twoI * H);
    std::vector<float> s13_ref((size_t)E * twoI);
    std::vector<uint8_t> w2_ref((size_t)E * H * I);
    std::vector<float> s2_ref((size_t)E * H);

    for (int b = 0; b < B; b++)
    {
        x_scale.Data()[b] = 0.01f + 0.02f * std::fabs(u(rng));
        for (int k = 0; k < H; k++)
            x_f8.Data()[b * H + k] = RandF8(rng);
        const int base_e = edist(rng);
        for (int s = 0; s < topK; s++)
        {
            topk_ids.Data()[b * topK + s] = (base_e + s) % E;
            topk_weights.Data()[b * topK + s] = 0.2f + 0.5f * std::fabs(u(rng));
        }
    }

    for (int e = 0; e < E; e++)
    {
        for (int n = 0; n < twoI; n++)
        {
            s13_ref[e * twoI + n] = 0.002f + 0.003f * std::fabs(u(rng));
            for (int k = 0; k < H; k++)
                w13_ref[((size_t)e * twoI + n) * H + k] = RandF8(rng);
        }
        // Pack w13: [E, K, nGroups, 64, 4=(g_even, g_odd, u_even, u_odd)]
        for (int blk = 0; blk < nGroups; blk++)
        {
            for (int j = 0; j < 64; j++)
            {
                const int i_e = blk * 128 + 2 * j;
                const int i_o = i_e + 1;
                s13_packed.Data()[e * twoI + blk * 256 + 0 + j] = s13_ref[e * twoI + 2 * i_e];
                s13_packed.Data()[e * twoI + blk * 256 + 64 + j] = s13_ref[e * twoI + 2 * i_o];
                s13_packed.Data()[e * twoI + blk * 256 + 128 + j] = s13_ref[e * twoI + 2 * i_e + 1];
                s13_packed.Data()[e * twoI + blk * 256 + 192 + j] = s13_ref[e * twoI + 2 * i_o + 1];
                for (int k = 0; k < H; k++)
                {
                    const size_t dst = (((size_t)e * nGroups + blk) * H + k) * 256 + 4 * j;
                    w13_packed.Data()[dst + 0] = w13_ref[((size_t)e * twoI + 2 * i_e) * H + k];
                    w13_packed.Data()[dst + 1] = w13_ref[((size_t)e * twoI + 2 * i_o) * H + k];
                    w13_packed.Data()[dst + 2] = w13_ref[((size_t)e * twoI + 2 * i_e + 1) * H + k];
                    w13_packed.Data()[dst + 3] = w13_ref[((size_t)e * twoI + 2 * i_o + 1) * H + k];
                }
            }
        }
        for (int h = 0; h < H; h++)
        {
            s2_ref[e * H + h] = 0.002f + 0.003f * std::fabs(u(rng));
            for (int i = 0; i < I; i++)
                w2_ref[((size_t)e * H + h) * I + i] = RandF8(rng);
        }
        // Pack w2: [E, nOutBlks, I_reordered(nGroups, even/odd, 64), 256] matching compact [q_even[0..63], q_odd[0..63]].
        for (int blk = 0; blk < nOutBlks; blk++)
        {
            for (int j = 0; j < 64; j++)
            {
                const int h0 = blk * 256 + 2 * j;
                const int h1 = h0 + 1;
                const int h2 = blk * 256 + 128 + 2 * j;
                const int h3 = h2 + 1;
                s2_packed.Data()[e * H + blk * 256 + 0 + j] = s2_ref[e * H + h0];
                s2_packed.Data()[e * H + blk * 256 + 64 + j] = s2_ref[e * H + h1];
                s2_packed.Data()[e * H + blk * 256 + 128 + j] = s2_ref[e * H + h2];
                s2_packed.Data()[e * H + blk * 256 + 192 + j] = s2_ref[e * H + h3];
                for (int g = 0; g < nGroups; g++)
                {
                    for (int eo = 0; eo < 2; eo++)
                    {
                        for (int q = 0; q < 64; q++)
                        {
                            const int orig_i = g * 128 + 2 * q + eo;
                            const int pack_k = g * 128 + eo * 64 + q;
                            const size_t dst = (((size_t)e * nOutBlks + blk) * I + pack_k) * 256 + 4 * j;
                            w2_packed.Data()[dst + 0] = w2_ref[((size_t)e * H + h0) * I + orig_i];
                            w2_packed.Data()[dst + 1] = w2_ref[((size_t)e * H + h1) * I + orig_i];
                            w2_packed.Data()[dst + 2] = w2_ref[((size_t)e * H + h2) * I + orig_i];
                            w2_packed.Data()[dst + 3] = w2_ref[((size_t)e * H + h3) * I + orig_i];
                        }
                    }
                }
            }
        }
    }

    // 1. Simulate w13_situ
    memcpy(x_i32.Data(), x_f8.Data(), (size_t)H * B);
    tpc_lib_api::HabanaKernelParams in1 = {};
    tpc_lib_api::HabanaKernelInstantiation inst1 = {};
    std::vector<tpc_lib_api::Tensor> ins1(6), outs1(2);
    std::vector<tpc_lib_api::TensorAccessPattern> in_ap1(6), out_ap1(2);
    std::vector<char> elf1(1 << 20);
    in1.deviceId = tpc_lib_api::DEVICE_ID_GAUDI2;
    strcpy(in1.guid.name, Inkling2MoeW13SituGaudi2::kGuid);
    in1.inputTensors = ins1.data();
    in1.inputTensorNr = 6;
    in1.outputTensors = outs1.data();
    in1.outputTensorNr = 2;
    Describe(&ins1[0], tpc_lib_api::DATA_I32, x_i32);
    Describe(&ins1[1], tpc_lib_api::DATA_F32, x_scale);
    Describe(&ins1[2], tpc_lib_api::DATA_F8_143, w13_packed);
    Describe(&ins1[3], tpc_lib_api::DATA_F32, s13_packed);
    Describe(&ins1[4], tpc_lib_api::DATA_I32, topk_ids);
    Describe(&ins1[5], tpc_lib_api::DATA_F32, topk_weights);
    Describe(&outs1[0], tpc_lib_api::DATA_F8_143, act_f8);
    Describe(&outs1[1], tpc_lib_api::DATA_F32, act_scale);
    Inkling2MoeW13SituGaudi2::Params p1 = {gate_cap, up_cap, fp8_max};
    in1.nodeParams.nodeParams = &p1;
    in1.nodeParams.nodeParamsSize = sizeof(p1);
    inst1.inputTensorAccessPattern = in_ap1.data();
    inst1.outputTensorAccessPattern = out_ap1.data();
    inst1.kernel.kernelElf = elf1.data();
    inst1.kernel.elfSize = elf1.size();
    if (InstantiateTpcKernel(&in1, &inst1) != tpc_lib_api::GLUE_SUCCESS)
    {
        printf("w13 glue failed\n");
        return 1;
    }
    std::vector<TensorDesc2> descs1 = {x_i32.GetTensorDescriptor(),    x_scale.GetTensorDescriptor(),
                                       w13_packed.GetTensorDescriptor(), s13_packed.GetTensorDescriptor(),
                                       topk_ids.GetTensorDescriptor(), topk_weights.GetTensorDescriptor(),
                                       act_f8.GetTensorDescriptor(),   act_scale.GetTensorDescriptor()};
    VPEStats stats1;
    const unsigned cyc1 = tpc_tests::RunSimulation(in1, inst1, descs1, stats1);

    // 2. Simulate w2
    memcpy(act_i32.Data(), act_f8.Data(), (size_t)I * topK * B);
    memcpy(act_scale3.Data(), act_scale.Data(), sizeof(float) * nGroups * topK * B);
    tpc_lib_api::HabanaKernelParams in2 = {};
    tpc_lib_api::HabanaKernelInstantiation inst2 = {};
    std::vector<tpc_lib_api::Tensor> ins2(5), outs2(1);
    std::vector<tpc_lib_api::TensorAccessPattern> in_ap2(5), out_ap2(1);
    std::vector<char> elf2(1 << 20);
    in2.deviceId = tpc_lib_api::DEVICE_ID_GAUDI2;
    strcpy(in2.guid.name, Inkling2MoeW2Gaudi2::kGuid);
    in2.inputTensors = ins2.data();
    in2.inputTensorNr = 5;
    in2.outputTensors = outs2.data();
    in2.outputTensorNr = 1;
    Describe(&ins2[0], tpc_lib_api::DATA_I32, act_i32);
    Describe(&ins2[1], tpc_lib_api::DATA_F32, act_scale3);
    Describe(&ins2[2], tpc_lib_api::DATA_F8_143, w2_packed);
    Describe(&ins2[3], tpc_lib_api::DATA_F32, s2_packed);
    Describe(&ins2[4], tpc_lib_api::DATA_I32, topk_ids);
    Describe(&outs2[0], tpc_lib_api::DATA_BF16, out_bf16);
    inst2.inputTensorAccessPattern = in_ap2.data();
    inst2.outputTensorAccessPattern = out_ap2.data();
    inst2.kernel.kernelElf = elf2.data();
    inst2.kernel.elfSize = elf2.size();
    if (InstantiateTpcKernel(&in2, &inst2) != tpc_lib_api::GLUE_SUCCESS)
    {
        printf("w2 glue failed\n");
        return 1;
    }
    std::vector<TensorDesc2> descs2 = {act_i32.GetTensorDescriptor(), act_scale3.GetTensorDescriptor(),
                                       w2_packed.GetTensorDescriptor(), s2_packed.GetTensorDescriptor(),
                                       topk_ids.GetTensorDescriptor(), out_bf16.GetTensorDescriptor()};
    VPEStats stats2;
    const unsigned cyc2 = tpc_tests::RunSimulation(in2, inst2, descs2, stats2);

    // 3. Double-precision reference
    double max_err_act = 0, max_ref_act = 0;
    double max_err_out = 0, max_ref_out = 0, max_err_e2e = 0;
    for (int b = 0; b < B; b++)
    {
        for (int slot = 0; slot < topK; slot++)
        {
            const int e = topk_ids.Data()[b * topK + slot];
            const double gamma = topk_weights.Data()[b * topK + slot];
            const double xsc = x_scale.Data()[b];
            std::vector<double> act_ref(I);
            for (int i = 0; i < I; i++)
            {
                double gsum = 0.0, usum = 0.0;
                for (int k = 0; k < H; k++)
                {
                    const double xv = F8ToF32(x_f8.Data()[b * H + k]);
                    gsum += xv * F8ToF32(w13_ref[((size_t)e * twoI + 2 * i) * H + k]);
                    usum += xv * F8ToF32(w13_ref[((size_t)e * twoI + 2 * i + 1) * H + k]);
                }
                const double g = gsum * xsc * s13_ref[e * twoI + 2 * i];
                const double up = usum * xsc * s13_ref[e * twoI + 2 * i + 1];
                const double sig = 1.0 / (1.0 + std::exp(-g));
                const double a = (gate_cap * std::tanh(g / gate_cap) * sig) * (up_cap * std::tanh(up / up_cap)) * gamma;
                act_ref[i] = a;
                const int blk = i / 128;
                const int in_blk = i % 128;
                const int compact_byte = blk * 128 + (in_blk & 1) * 64 + (in_blk >> 1);
                const double got_a =
                    F8ToF32(act_f8.Data()[((size_t)b * topK + slot) * I + compact_byte]) *
                    act_scale.Data()[((size_t)b * topK + slot) * nGroups + blk];
                max_err_act = std::max(max_err_act, std::fabs(got_a - a));
                max_ref_act = std::max(max_ref_act, std::fabs(a));
            }
            for (int h = 0; h < H; h++)
            {
                double sum_q = 0.0, sum_fp = 0.0;
                for (int i = 0; i < I; i++)
                {
                    const int blk = i / 128;
                    const int in_blk = i % 128;
                    const int compact_byte = blk * 128 + (in_blk & 1) * 64 + (in_blk >> 1);
                    const double av =
                        F8ToF32(act_f8.Data()[((size_t)b * topK + slot) * I + compact_byte]) *
                        act_scale.Data()[((size_t)b * topK + slot) * nGroups + blk];
                    const double wv = F8ToF32(w2_ref[((size_t)e * H + h) * I + i]);
                    sum_q += av * wv;
                    sum_fp += act_ref[i] * wv;
                }
                const double ref_q = sum_q * s2_ref[e * H + h];
                const double ref_fp = sum_fp * s2_ref[e * H + h];
                const double got_o = bf16ToFloat(out_bf16.Data()[((size_t)b * topK + slot) * H + h]);
                max_err_out = std::max(max_err_out, std::fabs(got_o - ref_q));
                max_ref_out = std::max(max_ref_out, std::fabs(ref_q));
                max_err_e2e = std::max(max_err_e2e, std::fabs(got_o - ref_fp));
            }
        }
    }

    const double rel_act = max_err_act / std::max(max_ref_act, 1e-12);
    const double rel_out = max_err_out / std::max(max_ref_out, 1e-12);
    const double rel_e2e = max_err_e2e / std::max(max_ref_out, 1e-12);
    // FP8 E4M3 has 3 mantissa bits (~6.25% max single-element quant step); BF16 w2 exact is ~0.4%.
    const bool ok = rel_act < 0.07 && rel_out < 0.008 && rel_e2e < 0.05;
    printf("H=%d I=%d E=%d topK=%d B=%d | w13_cyc=%u w2_cyc=%u | act_rel=%.4f w2_rel=%.4f e2e_rel=%.4f -> %s\n",
           H, I, E, topK, B, cyc1, cyc2, rel_act, rel_out, rel_e2e, ok ? "PASS" : "FAIL");
    if (!ok)
        return 1;

    // 3b. Simulate expert-grouped MoE (group_experts -> w13_grouped_f8 -> w2_grouped_f8) when B*topK <= 128
    if (B * topK <= 128 && (H / 4) <= 640 && (I / 4) <= 384 && E <= 256)
    {
        const int M = B * topK;
        // Build skewed topk_ids covering cnt=1, cnt=2, cnt=3, cnt=4, and cnt>4 (multi-group split!).
        test::Tensor<int32_t, 2> g_ids(topkSz);
        for (int b = 0; b < B; b++)
            for (int s = 0; s < topK; s++)
            {
                const int rs = b * topK + s;
                if (rs < 6)
                    g_ids.Data()[rs] = 0;  // cnt=6 -> split into groups of 4 + 2
                else if (rs < 10)
                    g_ids.Data()[rs] = 1 % E;  // cnt=4
                else if (rs < 13)
                    g_ids.Data()[rs] = 2 % E;  // cnt=3
                else if (rs < 15)
                    g_ids.Data()[rs] = 3 % E;  // cnt=2
                else
                    g_ids.Data()[rs] = (E > 4) ? (4 + ((rs - 15) % (E - 4))) : (rs % E);
            }
        const uint64_t metaSz[] = {128u, 8u};
        const uint64_t cntSz[] = {2u};
        test::Tensor<int32_t, 2> g_meta(metaSz);
        test::Tensor<int32_t, 1> g_cnt(cntSz);
        test::Tensor<uint8_t, 2> g_act_f8(actSz);
        test::Tensor<int32_t, 3> g_act_i32(actI32Sz);
        test::Tensor<float, 1> g_act_sc(actScSz);
        test::Tensor<float, 3> g_act_sc3(actSc3Sz);
        test::Tensor<uint16_t, 2> g_out(outSz);

        // Stage A: group_experts
        tpc_lib_api::HabanaKernelParams ing = {};
        tpc_lib_api::HabanaKernelInstantiation instg = {};
        std::vector<tpc_lib_api::Tensor> insg(1), outsg(2);
        std::vector<tpc_lib_api::TensorAccessPattern> iapg(1), oapg(2);
        std::vector<char> elfg(1 << 20);
        ing.deviceId = tpc_lib_api::DEVICE_ID_GAUDI2;
        ing.maxAvailableTpc = 24;
        strcpy(ing.guid.name, Inkling2MoeGroupExpertsGaudi2::kGuid);
        ing.inputTensors = insg.data(); ing.inputTensorNr = 1;
        ing.outputTensors = outsg.data(); ing.outputTensorNr = 2;
        Describe(&insg[0], tpc_lib_api::DATA_I32, g_ids);
        Describe(&outsg[0], tpc_lib_api::DATA_I32, g_meta);
        Describe(&outsg[1], tpc_lib_api::DATA_I32, g_cnt);
        Inkling2MoeGroupExpertsGaudi2::Params pg = {E, nGroups, nOutBlks};
        ing.nodeParams.nodeParams = &pg; ing.nodeParams.nodeParamsSize = sizeof(pg);
        instg.inputTensorAccessPattern = iapg.data(); instg.outputTensorAccessPattern = oapg.data();
        instg.kernel.kernelElf = elfg.data(); instg.kernel.elfSize = elfg.size();
        if (InstantiateTpcKernel(&ing, &instg) != tpc_lib_api::GLUE_SUCCESS) { printf("group_experts glue failed\n"); return 1; }
        std::vector<TensorDesc2> dg = {g_ids.GetTensorDescriptor(), g_meta.GetTensorDescriptor(), g_cnt.GetTensorDescriptor()};
        VPEStats stg;
        const unsigned cyc_grp = tpc_tests::RunSimulation(ing, instg, dg, stg);

        // Stage B: w13_grouped_f8
        tpc_lib_api::HabanaKernelParams inw1 = {};
        tpc_lib_api::HabanaKernelInstantiation instw1 = {};
        std::vector<tpc_lib_api::Tensor> insw1(7), outsw1(2);
        std::vector<tpc_lib_api::TensorAccessPattern> iapw1(7), oapw1(2);
        std::vector<char> elfw1(1 << 20);
        inw1.deviceId = tpc_lib_api::DEVICE_ID_GAUDI2;
        inw1.maxAvailableTpc = 24;
        strcpy(inw1.guid.name, Inkling2MoeW13GroupedGaudi2::kGuid);
        inw1.inputTensors = insw1.data(); inw1.inputTensorNr = 7;
        inw1.outputTensors = outsw1.data(); inw1.outputTensorNr = 2;
        Describe(&insw1[0], tpc_lib_api::DATA_I32, x_i32);
        Describe(&insw1[1], tpc_lib_api::DATA_F32, x_scale);
        Describe(&insw1[2], tpc_lib_api::DATA_F8_143, w13_packed);
        Describe(&insw1[3], tpc_lib_api::DATA_F32, s13_packed);
        Describe(&insw1[4], tpc_lib_api::DATA_F32, topk_weights);
        Describe(&insw1[5], tpc_lib_api::DATA_I32, g_meta);
        Describe(&insw1[6], tpc_lib_api::DATA_I32, g_cnt);
        Describe(&outsw1[0], tpc_lib_api::DATA_F8_143, g_act_f8);
        Describe(&outsw1[1], tpc_lib_api::DATA_F32, g_act_sc);
        Inkling2MoeW13GroupedGaudi2::Params pw1 = {gate_cap, up_cap, fp8_max};
        inw1.nodeParams.nodeParams = &pw1; inw1.nodeParams.nodeParamsSize = sizeof(pw1);
        instw1.inputTensorAccessPattern = iapw1.data(); instw1.outputTensorAccessPattern = oapw1.data();
        instw1.kernel.kernelElf = elfw1.data(); instw1.kernel.elfSize = elfw1.size();
        if (InstantiateTpcKernel(&inw1, &instw1) != tpc_lib_api::GLUE_SUCCESS) { printf("w13_grouped glue failed\n"); return 1; }
        std::vector<TensorDesc2> dw1 = {x_i32.GetTensorDescriptor(), x_scale.GetTensorDescriptor(),
                                        w13_packed.GetTensorDescriptor(), s13_packed.GetTensorDescriptor(),
                                        topk_weights.GetTensorDescriptor(), g_meta.GetTensorDescriptor(),
                                        g_cnt.GetTensorDescriptor(), g_act_f8.GetTensorDescriptor(),
                                        g_act_sc.GetTensorDescriptor()};
        VPEStats stw1;
        const unsigned cyc_w1g = tpc_tests::RunSimulation(inw1, instw1, dw1, stw1);

        // Stage C: w2_grouped_f8
        memcpy(g_act_i32.Data(), g_act_f8.Data(), (size_t)I * topK * B);
        memcpy(g_act_sc3.Data(), g_act_sc.Data(), sizeof(float) * nGroups * topK * B);
        tpc_lib_api::HabanaKernelParams inw2 = {};
        tpc_lib_api::HabanaKernelInstantiation instw2 = {};
        std::vector<tpc_lib_api::Tensor> insw2(6), outsw2(1);
        std::vector<tpc_lib_api::TensorAccessPattern> iapw2(6), oapw2(1);
        std::vector<char> elfw2(1 << 20);
        inw2.deviceId = tpc_lib_api::DEVICE_ID_GAUDI2;
        inw2.maxAvailableTpc = 24;
        strcpy(inw2.guid.name, Inkling2MoeW2GroupedGaudi2::kGuid);
        inw2.inputTensors = insw2.data(); inw2.inputTensorNr = 6;
        inw2.outputTensors = outsw2.data(); inw2.outputTensorNr = 1;
        Describe(&insw2[0], tpc_lib_api::DATA_I32, g_act_i32);
        Describe(&insw2[1], tpc_lib_api::DATA_F32, g_act_sc3);
        Describe(&insw2[2], tpc_lib_api::DATA_F8_143, w2_packed);
        Describe(&insw2[3], tpc_lib_api::DATA_F32, s2_packed);
        Describe(&insw2[4], tpc_lib_api::DATA_I32, g_meta);
        Describe(&insw2[5], tpc_lib_api::DATA_I32, g_cnt);
        Describe(&outsw2[0], tpc_lib_api::DATA_BF16, g_out);
        instw2.inputTensorAccessPattern = iapw2.data(); instw2.outputTensorAccessPattern = oapw2.data();
        instw2.kernel.kernelElf = elfw2.data(); instw2.kernel.elfSize = elfw2.size();
        if (InstantiateTpcKernel(&inw2, &instw2) != tpc_lib_api::GLUE_SUCCESS) { printf("w2_grouped glue failed\n"); return 1; }
        std::vector<TensorDesc2> dw2 = {g_act_i32.GetTensorDescriptor(), g_act_sc3.GetTensorDescriptor(),
                                        w2_packed.GetTensorDescriptor(), s2_packed.GetTensorDescriptor(),
                                        g_meta.GetTensorDescriptor(), g_cnt.GetTensorDescriptor(),
                                        g_out.GetTensorDescriptor()};
        VPEStats stw2;
        const unsigned cyc_w2g = tpc_tests::RunSimulation(inw2, instw2, dw2, stw2);

        double g_err_act = 0, g_ref_act = 0, g_err_out = 0, g_ref_out = 0;
        for (int b = 0; b < B; b++)
            for (int slot = 0; slot < topK; slot++)
            {
                const int e = g_ids.Data()[b * topK + slot];
                const double gamma = topk_weights.Data()[b * topK + slot];
                const double xsc = x_scale.Data()[b];
                for (int i = 0; i < I; i++)
                {
                    double gsum = 0.0, usum = 0.0;
                    for (int k = 0; k < H; k++)
                    {
                        const double xv = F8ToF32(x_f8.Data()[b * H + k]);
                        gsum += xv * F8ToF32(w13_ref[((size_t)e * twoI + 2 * i) * H + k]);
                        usum += xv * F8ToF32(w13_ref[((size_t)e * twoI + 2 * i + 1) * H + k]);
                    }
                    const double g = gsum * xsc * s13_ref[e * twoI + 2 * i];
                    const double up = usum * xsc * s13_ref[e * twoI + 2 * i + 1];
                    const double sig = 1.0 / (1.0 + std::exp(-g));
                    const double a = (gate_cap * std::tanh(g / gate_cap) * sig) * (up_cap * std::tanh(up / up_cap)) * gamma;
                    const int blk = i / 128, in_blk = i % 128;
                    const int cb = blk * 128 + (in_blk & 1) * 64 + (in_blk >> 1);
                    const double got_a = F8ToF32(g_act_f8.Data()[((size_t)b * topK + slot) * I + cb]) *
                                         g_act_sc.Data()[((size_t)b * topK + slot) * nGroups + blk];
                    g_err_act = std::max(g_err_act, std::fabs(got_a - a));
                    g_ref_act = std::max(g_ref_act, std::fabs(a));
                }
                for (int h = 0; h < H; h++)
                {
                    double sum_q = 0.0;
                    for (int i = 0; i < I; i++)
                    {
                        const int blk = i / 128, in_blk = i % 128;
                        const int cb = blk * 128 + (in_blk & 1) * 64 + (in_blk >> 1);
                        const double av = F8ToF32(g_act_f8.Data()[((size_t)b * topK + slot) * I + cb]) *
                                          g_act_sc.Data()[((size_t)b * topK + slot) * nGroups + blk];
                        sum_q += av * F8ToF32(w2_ref[((size_t)e * H + h) * I + i]);
                    }
                    const double ref_q = sum_q * s2_ref[e * H + h];
                    const double got_o = bf16ToFloat(g_out.Data()[((size_t)b * topK + slot) * H + h]);
                    g_err_out = std::max(g_err_out, std::fabs(got_o - ref_q));
                    g_ref_out = std::max(g_ref_out, std::fabs(ref_q));
                }
            }
        const double r_ga = g_err_act / std::max(g_ref_act, 1e-12);
        const double r_go = g_err_out / std::max(g_ref_out, 1e-12);
        const bool expect_over64 = (E >= 96 && M >= 80);
        const bool ok_grp = (g_cnt.Data()[0] > 0 && g_cnt.Data()[0] <= M &&
                             (!expect_over64 || g_cnt.Data()[0] > 64) && r_ga < 0.07 && r_go < 0.008);
        printf("GROUPED_MOE H=%d I=%d E=%d M=%d G=%d | grp=%u w13g=%u w2g=%u | act=%.4f w2=%.4f -> %s\n",
               H, I, E, M, g_cnt.Data()[0], cyc_grp, cyc_w1g, cyc_w2g, r_ga, r_go, ok_grp ? "PASS" : "FAIL");
        if (!ok_grp)
            return 1;
    }

    // 4. Simulate inkling2_moe_sum_sconv_res_decode_gaudi2 (is_first=0 and is_first=1, direct conv_state + load_slots)
    for (int is_first = 0; is_first <= 1; is_first++)
    {
        const int numSlots = std::max(8, B + 2);
        const int streamOffset = 256;
        const int totalDim = streamOffset + H + 128;
        const uint64_t mpSz[] = {(uint64_t)H, (uint64_t)topK, (uint64_t)B};
        const uint64_t csSz[] = {(uint64_t)totalDim, 3u, (uint64_t)numSlots};
        const uint64_t lsSz[] = {(uint64_t)B};
        const uint64_t stSz[] = {(uint64_t)H, 3u, (uint64_t)B};
        const uint64_t wtSz[] = {(uint64_t)H, 4u};
        const uint64_t hbSz[] = {(uint64_t)H, (uint64_t)B};
        test::Tensor<uint16_t, 3> mp(mpSz), conv_st(csSz), new_st(stSz);
        test::Tensor<int32_t, 1> load_slots(lsSz);
        test::Tensor<uint16_t, 2> wt(wtSz), part_res(hbSz), attn_out(hbSz), new_part(hbSz);
        memcpy(mp.Data(), out_bf16.Data(), sizeof(uint16_t) * H * topK * B);
        std::mt19937 srng(99 + is_first);
        std::normal_distribution<float> snd(0.0f, 0.5f);
        for (size_t i = 0; i < (size_t)totalDim * 3 * numSlots; i++) conv_st.Data()[i] = floatToBf16(snd(srng));
        for (int b = 0; b < B; b++) load_slots.Data()[b] = (numSlots - 1 - b) % numSlots;
        for (size_t i = 0; i < (size_t)H * 4; i++) wt.Data()[i] = floatToBf16(snd(srng));
        for (size_t i = 0; i < (size_t)H * B; i++)
        {
            part_res.Data()[i] = floatToBf16(snd(srng));
            attn_out.Data()[i] = floatToBf16(snd(srng));
        }
        const float outScale = 1.0f / std::sqrt(5.0f);

        tpc_lib_api::HabanaKernelParams in = {};
        tpc_lib_api::HabanaKernelInstantiation inst = {};
        std::vector<tpc_lib_api::Tensor> ins(6), outs(2);
        std::vector<tpc_lib_api::TensorAccessPattern> iap(6), oap(2);
        std::vector<char> elf(1 << 20);
        in.deviceId = tpc_lib_api::DEVICE_ID_GAUDI2;
        strcpy(in.guid.name, Inkling2MoeSumSconvResDecodeGaudi2::kGuid);
        in.inputTensors = ins.data(); in.inputTensorNr = 6;
        in.outputTensors = outs.data(); in.outputTensorNr = 2;
        Describe(&ins[0], tpc_lib_api::DATA_BF16, mp);
        Describe(&ins[1], tpc_lib_api::DATA_BF16, conv_st);
        Describe(&ins[2], tpc_lib_api::DATA_I32, load_slots);
        Describe(&ins[3], tpc_lib_api::DATA_BF16, wt);
        Describe(&ins[4], tpc_lib_api::DATA_BF16, part_res);
        Describe(&ins[5], tpc_lib_api::DATA_BF16, attn_out);
        Describe(&outs[0], tpc_lib_api::DATA_BF16, new_part);
        Describe(&outs[1], tpc_lib_api::DATA_BF16, new_st);
        Inkling2MoeSumSconvResDecodeGaudi2::Params sp = {outScale, is_first, streamOffset};
        in.nodeParams.nodeParams = &sp; in.nodeParams.nodeParamsSize = sizeof(sp);
        inst.inputTensorAccessPattern = iap.data(); inst.outputTensorAccessPattern = oap.data();
        inst.kernel.kernelElf = elf.data(); inst.kernel.elfSize = elf.size();
        if (InstantiateTpcKernel(&in, &inst) != tpc_lib_api::GLUE_SUCCESS) { printf("sum_sconv_res glue failed\n"); return 1; }
        std::vector<TensorDesc2> d = {mp.GetTensorDescriptor(), conv_st.GetTensorDescriptor(),
                                      load_slots.GetTensorDescriptor(), wt.GetTensorDescriptor(),
                                      part_res.GetTensorDescriptor(), attn_out.GetTensorDescriptor(),
                                      new_part.GetTensorDescriptor(), new_st.GetTensorDescriptor()};
        VPEStats st;
        const unsigned cyc_s = tpc_tests::RunSimulation(in, inst, d, st);

        double max_p_err = 0.0, max_p_ref = 0.0, max_s_err = 0.0;
        for (int b = 0; b < B; b++)
        {
            const int slot = load_slots.Data()[b];
            for (int h = 0; h < H; h++)
            {
                float sf = 0.0f;
                for (int k = 0; k < topK; k++) sf += bf16ToFloat(mp.Data()[((size_t)b * topK + k) * H + h]);
                const float x_bf = bf16ToFloat(floatToBf16(sf));
                const float s0 = bf16ToFloat(conv_st.Data()[((size_t)slot * 3 + 0) * totalDim + streamOffset + h]);
                const float s1 = bf16ToFloat(conv_st.Data()[((size_t)slot * 3 + 1) * totalDim + streamOffset + h]);
                const float s2 = bf16ToFloat(conv_st.Data()[((size_t)slot * 3 + 2) * totalDim + streamOffset + h]);
                const float w0 = bf16ToFloat(wt.Data()[0 * H + h]);
                const float w1 = bf16ToFloat(wt.Data()[1 * H + h]);
                const float w2 = bf16ToFloat(wt.Data()[2 * H + h]);
                const float w3 = bf16ToFloat(wt.Data()[3 * H + h]);
                const float conv = (x_bf + s0 * w0 + s1 * w1 + s2 * w2 + x_bf * w3) * outScale;
                const float mlp_bf = bf16ToFloat(floatToBf16(conv));
                const float a_f = bf16ToFloat(attn_out.Data()[(size_t)b * H + h]);
                const float p_f = is_first ? 0.0f : bf16ToFloat(part_res.Data()[(size_t)b * H + h]);
                const float ref_p = bf16ToFloat(floatToBf16(p_f + a_f + mlp_bf));
                const float got_p = bf16ToFloat(new_part.Data()[(size_t)b * H + h]);
                max_p_err = std::max(max_p_err, (double)std::fabs(got_p - ref_p));
                max_p_ref = std::max(max_p_ref, (double)std::fabs(ref_p));
                max_s_err = std::max(max_s_err, (double)std::fabs(bf16ToFloat(new_st.Data()[((size_t)b * 3 + 0) * H + h]) - s1));
                max_s_err = std::max(max_s_err, (double)std::fabs(bf16ToFloat(new_st.Data()[((size_t)b * 3 + 1) * H + h]) - s2));
                max_p_err = std::max(max_p_err, (double)std::fabs(bf16ToFloat(new_st.Data()[((size_t)b * 3 + 2) * H + h]) - x_bf));
            }
        }
        const double rel_p = max_p_err / std::max(max_p_ref, 1e-12);
        // 1 BF16 ULP is 2^-7 = 0.0078125; allow 0.01 across cascaded sum -> sconv -> res BF16 roundings.
        const bool ok_s = (rel_p < 0.01) && (max_s_err == 0.0);
        printf("SUM_SCONV_RES(first=%d) H=%d topK=%d B=%d | cyc=%u | rel=%.5f st_err=%.2e -> %s\n",
               is_first, H, topK, B, cyc_s, rel_p, max_s_err, ok_s ? "PASS" : "FAIL");
        if (!ok_s)
            return 1;
    }
    return 0;
}

int TestGateTopk()
{
    // Exercise both dedicated E=128 ELF and general E in {64, 192} ELF (ties & extremes).
    for (int gE : {64, 128, 192})
    {
        const int gS = 1, gK = 8, gB = 4;
        const float routeScale = 8.0f;
        const uint64_t lgSz[] = {(uint64_t)gE, (uint64_t)gS, (uint64_t)gB};
        const uint64_t bsSz[] = {(uint64_t)gE};
        const uint64_t gtSz[] = {(uint64_t)gK, (uint64_t)gB};
        test::Tensor<float, 3> lg(lgSz);
        test::Tensor<float, 2> gt_wt(gtSz);
        test::Tensor<float, 1> bs(bsSz);
        test::Tensor<int32_t, 2> gt_ids(gtSz);
        std::mt19937 grng(777 + gE);
        std::normal_distribution<float> gnd(0.0f, 3.0f);
        for (int e = 0; e < gE; e++)
            bs.Data()[e] = gnd(grng) * 0.2f;
        for (int b = 0; b < gB; b++)
            for (int e = 0; e < gE; e++)
                lg.Data()[b * gE + e] = gnd(grng);
        // Inject an exact tie and extreme negative/positive logits on b=1.
        const int tie2 = (gE > 64) ? 70 : 22;
        lg.Data()[1 * gE + 5] = 25.0f; bs.Data()[5] = 0.0f;
        lg.Data()[1 * gE + tie2] = 25.0f; bs.Data()[tie2] = 0.0f;
        lg.Data()[1 * gE + 12] = -30.0f; bs.Data()[12] = 5.0f;

        tpc_lib_api::HabanaKernelParams in = {};
        tpc_lib_api::HabanaKernelInstantiation inst = {};
        std::vector<tpc_lib_api::Tensor> ins(2), outs(2);
        std::vector<tpc_lib_api::TensorAccessPattern> iap(2), oap(2);
        std::vector<char> elf(1 << 20);
        in.deviceId = tpc_lib_api::DEVICE_ID_GAUDI2;
        strcpy(in.guid.name, Inkling2MoeGateTopkGaudi2::kGuid);
        in.inputTensors = ins.data(); in.inputTensorNr = 2;
        in.outputTensors = outs.data(); in.outputTensorNr = 2;
        Describe(&ins[0], tpc_lib_api::DATA_F32, lg);
        Describe(&ins[1], tpc_lib_api::DATA_F32, bs);
        Describe(&outs[0], tpc_lib_api::DATA_I32, gt_ids);
        Describe(&outs[1], tpc_lib_api::DATA_F32, gt_wt);
        Inkling2MoeGateTopkGaudi2::Params gp = {gK, routeScale};
        in.nodeParams.nodeParams = &gp; in.nodeParams.nodeParamsSize = sizeof(gp);
        inst.inputTensorAccessPattern = iap.data(); inst.outputTensorAccessPattern = oap.data();
        inst.kernel.kernelElf = elf.data(); inst.kernel.elfSize = elf.size();
        if (InstantiateTpcKernel(&in, &inst) != tpc_lib_api::GLUE_SUCCESS) { printf("gate_topk glue failed\n"); return 1; }
        std::vector<TensorDesc2> d = {lg.GetTensorDescriptor(), bs.GetTensorDescriptor(),
                                      gt_ids.GetTensorDescriptor(), gt_wt.GetTensorDescriptor()};
        VPEStats st;
        const unsigned cyc_g = tpc_tests::RunSimulation(in, inst, d, st);

        int id_mismatches = 0;
        double max_wt_err = 0.0;
        for (int b = 0; b < gB; b++)
        {
            std::vector<std::pair<double, int>> scores(gE);
            for (int e = 0; e < gE; e++)
            {
                const double x = lg.Data()[b * gE + e];
                const double sig = 1.0 / (1.0 + std::exp(-x));
                scores[e] = {sig + bs.Data()[e], e};
            }
            std::stable_sort(scores.begin(), scores.end(), [](const auto& a, const auto& c) {
                return a.first != c.first ? a.first > c.first : a.second < c.second;
            });
            std::vector<double> lps(gK);
            double lp_max = -1e30;
            for (int k = 0; k < gK; k++)
            {
                const int e = scores[k].second;
                if (gt_ids.Data()[b * gK + k] != e)
                    id_mismatches++;
                const double x = lg.Data()[b * gE + e];
                const double lp = (x >= 0.0) ? -std::log1p(std::exp(-x)) : x - std::log1p(std::exp(x));
                lps[k] = lp;
                lp_max = std::max(lp_max, lp);
            }
            double denom = 0.0;
            for (int k = 0; k < gK; k++) denom += std::exp(lps[k] - lp_max);
            for (int k = 0; k < gK; k++)
            {
                const double ref_w = (std::exp(lps[k] - lp_max) / denom) * routeScale;
                max_wt_err = std::max(max_wt_err, std::fabs(gt_wt.Data()[b * gK + k] - ref_w));
            }
        }
        const bool ok_g = (id_mismatches == 0) && (max_wt_err < 1e-4);
        printf("GATE_TOPK E=%d S=%d topK=%d B=%d | cyc=%u | id_err=%d wt_max_err=%.2e -> %s\n",
               gE, gS, gK, gB, cyc_g, id_mismatches, max_wt_err, ok_g ? "PASS" : "FAIL");
        if (!ok_g)
            return 1;
    }
    return 0;
}

int TestGateSplitk()
{
    // Exercise B=1 dedicated ELF, nb<=4 (B=4), 4<nb<=8 (B=6), and multi-tile b0+=8 (B=10).
    for (int gB : {1, 4, 6, 10})
    {
        const int gH = 512, gE = 128, gS = gH / 128, gK = 8;
        const float routeScale = 8.0f;
        const uint64_t gxSz[] = {(uint64_t)gH, (uint64_t)gB};
        const uint64_t gwSz[] = {(uint64_t)gE, (uint64_t)gH};
        const uint64_t gpSz[] = {(uint64_t)gE, (uint64_t)gS, (uint64_t)gB};
        const uint64_t gbSz[] = {(uint64_t)gE};
        const uint64_t gtSz[] = {(uint64_t)gK, (uint64_t)gB};
        test::Tensor<uint16_t, 2> gx(gxSz), gw_pack(gwSz);
        test::Tensor<float, 3> gpart(gpSz);
        test::Tensor<float, 1> gbias(gbSz);
        test::Tensor<int32_t, 2> gt_ids(gtSz);
        test::Tensor<float, 2> gt_wt(gtSz);

        std::mt19937 grng(2026 + gB);
        std::normal_distribution<float> gnd(0.0f, 0.4f);
        std::vector<uint16_t> w_ref((size_t)gE * gH);
        for (int b = 0; b < gB; b++)
            for (int h = 0; h < gH; h++)
                gx.Data()[b * gH + h] = floatToBf16(gnd(grng));
        for (int e = 0; e < gE; e++)
        {
            gbias.Data()[e] = gnd(grng) * 0.2f;
            for (int h = 0; h < gH; h++)
                w_ref[e * gH + h] = floatToBf16(gnd(grng) * 0.2f);
        }
        for (int h = 0; h < gH; h++)
            for (int j = 0; j < 64; j++)
            {
                gw_pack.Data()[h * gE + 2 * j + 0] = w_ref[(j)*gH + h];
                gw_pack.Data()[h * gE + 2 * j + 1] = w_ref[(j + 64) * gH + h];
            }

        // Stage 1: split-K GEMV
        tpc_lib_api::HabanaKernelParams in1 = {};
        tpc_lib_api::HabanaKernelInstantiation inst1 = {};
        std::vector<tpc_lib_api::Tensor> ins1(2), outs1(1);
        std::vector<tpc_lib_api::TensorAccessPattern> iap1(2), oap1(1);
        std::vector<char> elf1(1 << 20);
        in1.deviceId = tpc_lib_api::DEVICE_ID_GAUDI2;
        strcpy(in1.guid.name, Inkling2MoeGateGemvSplitkGaudi2::kGuid);
        in1.inputTensors = ins1.data(); in1.inputTensorNr = 2;
        in1.outputTensors = outs1.data(); in1.outputTensorNr = 1;
        Describe(&ins1[0], tpc_lib_api::DATA_BF16, gx);
        Describe(&ins1[1], tpc_lib_api::DATA_BF16, gw_pack);
        Describe(&outs1[0], tpc_lib_api::DATA_F32, gpart);
        inst1.inputTensorAccessPattern = iap1.data(); inst1.outputTensorAccessPattern = oap1.data();
        inst1.kernel.kernelElf = elf1.data(); inst1.kernel.elfSize = elf1.size();
        if (InstantiateTpcKernel(&in1, &inst1) != tpc_lib_api::GLUE_SUCCESS) { printf("gate_splitk glue failed\n"); return 1; }
        std::vector<TensorDesc2> d1 = {gx.GetTensorDescriptor(), gw_pack.GetTensorDescriptor(), gpart.GetTensorDescriptor()};
        VPEStats st1;
        const unsigned cyc_s1 = tpc_tests::RunSimulation(in1, inst1, d1, st1);

        // Stage 2: reduce + single-tree topk
        tpc_lib_api::HabanaKernelParams in2 = {};
        tpc_lib_api::HabanaKernelInstantiation inst2 = {};
        std::vector<tpc_lib_api::Tensor> ins2(2), outs2(2);
        std::vector<tpc_lib_api::TensorAccessPattern> iap2(2), oap2(2);
        std::vector<char> elf2(1 << 20);
        in2.deviceId = tpc_lib_api::DEVICE_ID_GAUDI2;
        strcpy(in2.guid.name, Inkling2MoeGateTopkGaudi2::kGuid);
        in2.inputTensors = ins2.data(); in2.inputTensorNr = 2;
        in2.outputTensors = outs2.data(); in2.outputTensorNr = 2;
        Describe(&ins2[0], tpc_lib_api::DATA_F32, gpart);
        Describe(&ins2[1], tpc_lib_api::DATA_F32, gbias);
        Describe(&outs2[0], tpc_lib_api::DATA_I32, gt_ids);
        Describe(&outs2[1], tpc_lib_api::DATA_F32, gt_wt);
        Inkling2MoeGateTopkGaudi2::Params gp = {gK, routeScale};
        in2.nodeParams.nodeParams = &gp; in2.nodeParams.nodeParamsSize = sizeof(gp);
        inst2.inputTensorAccessPattern = iap2.data(); inst2.outputTensorAccessPattern = oap2.data();
        inst2.kernel.kernelElf = elf2.data(); inst2.kernel.elfSize = elf2.size();
        if (InstantiateTpcKernel(&in2, &inst2) != tpc_lib_api::GLUE_SUCCESS) { printf("gate_topk(splitk) glue failed\n"); return 1; }
        std::vector<TensorDesc2> d2 = {gpart.GetTensorDescriptor(), gbias.GetTensorDescriptor(),
                                       gt_ids.GetTensorDescriptor(), gt_wt.GetTensorDescriptor()};
        VPEStats st2;
        const unsigned cyc_s2 = tpc_tests::RunSimulation(in2, inst2, d2, st2);

        int id_mismatches = 0;
        double max_wt_err = 0.0;
        for (int b = 0; b < gB; b++)
        {
            std::vector<double> lgs(gE, 0.0);
            std::vector<std::pair<double, int>> scores(gE);
            for (int e = 0; e < gE; e++)
            {
                double acc = 0.0;
                for (int h = 0; h < gH; h++)
                    acc += (double)bf16ToFloat(gx.Data()[b * gH + h]) * bf16ToFloat(w_ref[e * gH + h]);
                lgs[e] = acc;
                scores[e] = {1.0 / (1.0 + std::exp(-acc)) + gbias.Data()[e], e};
            }
            std::stable_sort(scores.begin(), scores.end(), [](const auto& a, const auto& c) {
                return a.first != c.first ? a.first > c.first : a.second < c.second;
            });
            double sum_sig = 0.0;
            for (int k = 0; k < gK; k++)
            {
                const int e = scores[k].second;
                if (gt_ids.Data()[b * gK + k] != e)
                    id_mismatches++;
                sum_sig += 1.0 / (1.0 + std::exp(-lgs[e]));
            }
            for (int k = 0; k < gK; k++)
            {
                const int e = scores[k].second;
                const double ref_w = (1.0 / (1.0 + std::exp(-lgs[e]))) / sum_sig * routeScale;
                max_wt_err = std::max(max_wt_err, std::fabs(gt_wt.Data()[b * gK + k] - ref_w));
            }
        }
        const bool ok_gq = (id_mismatches == 0) && (max_wt_err < 1e-4);
        printf("GATE_SPLITK+TOPK H=%d S=%d E=%d K=%d B=%d | s1_cyc=%u s2_cyc=%u | id_err=%d wt=%.2e -> %s\n",
               gH, gS, gE, gK, gB, cyc_s1, cyc_s2, id_mismatches, max_wt_err, ok_gq ? "PASS" : "FAIL");
        if (!ok_gq)
            return 1;
    }
    return 0;
}
}  // namespace

int main(int argc, char** argv)
{
    if (argc > 1)
    {
        const int H = atoi(argv[1]);
        const int I = argc > 2 ? atoi(argv[2]) : 256;
        const int E = argc > 3 ? atoi(argv[3]) : 6;
        const int topK = argc > 4 ? atoi(argv[4]) : 4;
        const int B = argc > 5 ? atoi(argv[5]) : 5;
        return Run(H, I, E, topK, B);
    }
    int rc = 0;
    rc |= Run(512, 256, 6, 4, 5);    // general path + grouped same_blks (nW13=nW2=2)
    rc |= Run(512, 128, 8, 4, 5);    // grouped nW13=1 != nW2=2 (2-pass boundary partition)
    rc |= Run(256, 128, 96, 8, 10);  // grouped G > 64 (2-chunk group_meta store, M=80)
    rc |= Run(256, 128, 4, 8, 18);   // balanced 24-core w13+w2 path (totalItems=144)
    rc |= TestGateTopk();
    rc |= TestGateSplitk();
    return rc;
}
