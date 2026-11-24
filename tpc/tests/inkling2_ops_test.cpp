// SPDX-License-Identifier: Apache-2.0
// Double-precision simulator tests for Inkling 2 Mini Gaudi2 TPC kernels:
//   - inkling2_sconv_static_decode_gaudi2
//   - inkling2_qkvr_prep_decode_gaudi2 (static/dynamic, tau=0/1)
//   - inkling2_pair_rms_norm_gaudi2
//   - inkling2_blend_rms_norm_gaudi2
//   - inkling2_rel_paged_attn_gate_sink_gaudi2 (local sliding-window and global multi-block)
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

#include "habana/tensor.h"
#include "inkling2_paged_attn_gaudi2.hpp"
#include "inkling2_sconv_norm_gaudi2.hpp"
#include "tpc_test_core_api.h"

extern "C" tpc_lib_api::GlueCodeReturn InstantiateTpcKernel(tpc_lib_api::HabanaKernelParams* params,
                                                            tpc_lib_api::HabanaKernelInstantiation* instance);

namespace
{
template <class T, int N>
void Describe(tpc_lib_api::Tensor* t, tpc_lib_api::TensorDataType dt, const test::Tensor<T, N>& src)
{
    t->geometry.dims = N;
    t->geometry.dataType = dt;
    for (int i = 0; i < N; i++)
        t->geometry.maxSizes[i] = t->geometry.minSizes[i] = src.Size(i);
}

int TestSconvStatic()
{
    const int C = 256, B = 2, numSlots = 6, streamOffset = 128;
    const int totalDim = streamOffset + C + 128;
    const float scale = 1.0f / std::sqrt(5.0f);
    std::mt19937 rng(1);
    std::normal_distribution<float> nd(0.0f, 1.0f);

    const uint64_t xSz[] = {(uint64_t)C, (uint64_t)B};
    const uint64_t csSz[] = {(uint64_t)totalDim, 3, (uint64_t)numSlots};
    const uint64_t lsSz[] = {(uint64_t)B};
    const uint64_t stSz[] = {(uint64_t)C, 3, (uint64_t)B};
    const uint64_t wSz[] = {(uint64_t)C, 4};
    test::Tensor<uint16_t, 2> x(xSz), y(xSz), wT(wSz);
    test::Tensor<uint16_t, 3> convSt(csSz), next(stSz);
    test::Tensor<int32_t, 1> loadSlots(lsSz);
    for (size_t i = 0; i < (size_t)C * B; i++) x.Data()[i] = floatToBf16(nd(rng));
    for (size_t i = 0; i < (size_t)totalDim * 3 * numSlots; i++) convSt.Data()[i] = floatToBf16(nd(rng));
    loadSlots.Data()[0] = 4;
    loadSlots.Data()[1] = 1;
    for (size_t i = 0; i < (size_t)C * 4; i++) wT.Data()[i] = floatToBf16(nd(rng));

    tpc_lib_api::HabanaKernelParams in = {};
    tpc_lib_api::HabanaKernelInstantiation inst = {};
    std::vector<tpc_lib_api::Tensor> ins(4), outs(2);
    std::vector<tpc_lib_api::TensorAccessPattern> inAp(4), outAp(2);
    std::vector<char> elf(1 << 20);
    in.deviceId = tpc_lib_api::DEVICE_ID_GAUDI2;
    strcpy(in.guid.name, Inkling2SconvStaticDecodeGaudi2::kGuid);
    in.inputTensors = ins.data(); in.inputTensorNr = 4;
    in.outputTensors = outs.data(); in.outputTensorNr = 2;
    Describe(&ins[0], tpc_lib_api::DATA_BF16, x);
    Describe(&ins[1], tpc_lib_api::DATA_BF16, convSt);
    Describe(&ins[2], tpc_lib_api::DATA_I32, loadSlots);
    Describe(&ins[3], tpc_lib_api::DATA_BF16, wT);
    Describe(&outs[0], tpc_lib_api::DATA_BF16, y);
    Describe(&outs[1], tpc_lib_api::DATA_BF16, next);
    Inkling2SconvStaticDecodeGaudi2::Params p = {scale, streamOffset};
    in.nodeParams.nodeParams = &p; in.nodeParams.nodeParamsSize = sizeof(p);
    inst.inputTensorAccessPattern = inAp.data(); inst.outputTensorAccessPattern = outAp.data();
    inst.kernel.kernelElf = elf.data(); inst.kernel.elfSize = elf.size();
    if (InstantiateTpcKernel(&in, &inst) != tpc_lib_api::GLUE_SUCCESS) return 1;
    std::vector<TensorDesc2> descs = {x.GetTensorDescriptor(), convSt.GetTensorDescriptor(),
                                      loadSlots.GetTensorDescriptor(), wT.GetTensorDescriptor(),
                                      y.GetTensorDescriptor(), next.GetTensorDescriptor()};
    VPEStats st;
    const unsigned cyc = tpc_tests::RunSimulation(in, inst, descs, st);

    double maxErr = 0, maxRef = 0;
    for (int b = 0; b < B; b++)
    {
        const int slot = loadSlots.Data()[b];
        for (int c = 0; c < C; c++)
        {
            const uint16_t p0b = convSt.Data()[(slot * 3 + 0) * totalDim + streamOffset + c];
            const uint16_t p1b = convSt.Data()[(slot * 3 + 1) * totalDim + streamOffset + c];
            const uint16_t p2b = convSt.Data()[(slot * 3 + 2) * totalDim + streamOffset + c];
            double s0 = bf16ToFloat(p0b), s1 = bf16ToFloat(p1b), s2 = bf16ToFloat(p2b);
            double xv = bf16ToFloat(x.Data()[b * C + c]);
            double ref = (xv + s0 * bf16ToFloat(wT.Data()[0 * C + c]) + s1 * bf16ToFloat(wT.Data()[1 * C + c]) +
                          s2 * bf16ToFloat(wT.Data()[2 * C + c]) + xv * bf16ToFloat(wT.Data()[3 * C + c])) *
                         scale;
            maxErr = std::max(maxErr, std::fabs(bf16ToFloat(y.Data()[b * C + c]) - ref));
            maxRef = std::max(maxRef, std::fabs(ref));
            if (next.Data()[(b * 3 + 0) * C + c] != p1b ||
                next.Data()[(b * 3 + 1) * C + c] != p2b ||
                next.Data()[(b * 3 + 2) * C + c] != x.Data()[b * C + c])
            {
                printf("sconv_static state shift mismatch\n");
                return 1;
            }
        }
    }
    const double rel = maxErr / std::max(maxRef, 1e-12);
    printf("sconv_static | cyc=%u | rel=%.6f -> %s\n", cyc, rel, rel < 0.01 ? "PASS" : "FAIL");
    return rel < 0.01 ? 0 : 1;
}

int TestQkvrPrep(bool hasTau, bool isDynamic, int kvStreams, int H = 6, int Hkv = 2)
{
    const int D = 128, dRel = 16, B = 2, numSlots = 5;
    const int kvSize = Hkv * D;
    const int qSide = H * (D + dRel + D + 1);
    const int qCols = (kvStreams == 1) ? (qSide + 2 * kvSize) : qSide;
    const int kvCols = (kvStreams == 1) ? qCols : kvSize;
    const int kvColOffset = (kvStreams == 1) ? qSide : 0;
    const int totalDim = 2 * kvSize + 256;
    const float convScale = isDynamic ? (1.0f / std::sqrt(9.0f)) : (1.0f / std::sqrt(5.0f));
    const float qkClip = 10.0f, relClip = 2.0f, clipEps = 1e-6f, rmsEps = 1e-6f;
    std::mt19937 rng(100 + (hasTau ? 4 : 0) + (isDynamic ? 2 : 0) + kvStreams);
    std::normal_distribution<float> nd(0.0f, 1.5f);

    const uint64_t pqSz[] = {(uint64_t)qCols, (uint64_t)B};
    const uint64_t qSz[] = {(uint64_t)D, (uint64_t)H, (uint64_t)B};
    const uint64_t rSz[] = {(uint64_t)dRel, (uint64_t)H, (uint64_t)B};
    const uint64_t qsSz[] = {(uint64_t)D, (uint64_t)H};
    const uint64_t rsSz[] = {(uint64_t)dRel, (uint64_t)H};
    const uint64_t tauSz[] = {(uint64_t)B};
    const uint64_t kvSrcSz[] = {(uint64_t)kvCols, (uint64_t)B, (uint64_t)kvStreams};
    const uint64_t csSz[] = {(uint64_t)totalDim, 3, (uint64_t)numSlots};
    const uint64_t lsSz[] = {(uint64_t)B};
    const uint64_t kvwSz[] = {(uint64_t)D, (uint64_t)Hkv, 2, 4};
    const uint64_t kvDynSz[] = {4, (uint64_t)Hkv, (uint64_t)B, 2};
    const uint64_t kvOutSz[] = {(uint64_t)D, (uint64_t)Hkv, (uint64_t)B};
    const uint64_t newKvSz[] = {(uint64_t)D, (uint64_t)Hkv, 2, 3, (uint64_t)B};

    test::Tensor<uint16_t, 2> projQ(pqSz), qsc(qsSz), rsc(rsSz);
    test::Tensor<uint16_t, 3> qout(qSz), rout(rSz), kout(kvOutSz), vout(kvOutSz), kvSrc(kvSrcSz), convSt(csSz);
    test::Tensor<float, 1> tau(tauSz);
    test::Tensor<int32_t, 1> loadSlots(lsSz);
    test::Tensor<uint16_t, 4> kvw(kvwSz), kvdyn(kvDynSz);
    test::Tensor<uint16_t, 5> newKv(newKvSz);

    for (size_t i = 0; i < (size_t)qCols * B; i++) projQ.Data()[i] = floatToBf16(nd(rng));
    for (size_t i = 0; i < (size_t)D * H; i++) qsc.Data()[i] = floatToBf16(0.5f + std::fabs(nd(rng)));
    for (size_t i = 0; i < (size_t)dRel * H; i++) rsc.Data()[i] = floatToBf16(0.5f + std::fabs(nd(rng)));
    for (int b = 0; b < B; b++) tau.Data()[b] = 1.15f + 0.2f * b;
    if (kvStreams == 1)
        memcpy(kvSrc.Data(), projQ.Data(), sizeof(uint16_t) * qCols * B);
    else
        for (size_t i = 0; i < (size_t)kvCols * B * kvStreams; i++) kvSrc.Data()[i] = floatToBf16(nd(rng));
    for (size_t i = 0; i < (size_t)totalDim * 3 * numSlots; i++) convSt.Data()[i] = floatToBf16(nd(rng));
    loadSlots.Data()[0] = 3;
    loadSlots.Data()[1] = 0;
    for (size_t i = 0; i < (size_t)D * Hkv * 2 * 4; i++) kvw.Data()[i] = floatToBf16(nd(rng));
    for (size_t i = 0; i < (size_t)4 * Hkv * B * 2; i++) kvdyn.Data()[i] = floatToBf16(nd(rng) * 0.5f);

    tpc_lib_api::HabanaKernelParams in = {};
    tpc_lib_api::HabanaKernelInstantiation inst = {};
    std::vector<tpc_lib_api::Tensor> ins(9), outs(5);
    std::vector<tpc_lib_api::TensorAccessPattern> inAp(9), outAp(5);
    std::vector<char> elf(1 << 20);
    in.deviceId = tpc_lib_api::DEVICE_ID_GAUDI2;
    strcpy(in.guid.name, Inkling2QkvrPrepDecodeGaudi2::kGuid);
    in.inputTensors = ins.data(); in.inputTensorNr = 9;
    in.outputTensors = outs.data(); in.outputTensorNr = 5;
    Describe(&ins[0], tpc_lib_api::DATA_BF16, projQ);     Describe(&ins[1], tpc_lib_api::DATA_BF16, qsc);
    Describe(&ins[2], tpc_lib_api::DATA_BF16, rsc);       Describe(&ins[3], tpc_lib_api::DATA_F32, tau);
    Describe(&ins[4], tpc_lib_api::DATA_BF16, kvSrc);     Describe(&ins[5], tpc_lib_api::DATA_BF16, convSt);
    Describe(&ins[6], tpc_lib_api::DATA_I32, loadSlots);  Describe(&ins[7], tpc_lib_api::DATA_BF16, kvw);
    Describe(&ins[8], tpc_lib_api::DATA_BF16, isDynamic ? kvdyn : kvw);
    Describe(&outs[0], tpc_lib_api::DATA_BF16, qout);     Describe(&outs[1], tpc_lib_api::DATA_BF16, rout);
    Describe(&outs[2], tpc_lib_api::DATA_BF16, kout);     Describe(&outs[3], tpc_lib_api::DATA_BF16, vout);
    Describe(&outs[4], tpc_lib_api::DATA_BF16, newKv);

    Inkling2QkvrPrepDecodeGaudi2::Params p = {kvColOffset, hasTau ? 1 : 0, isDynamic ? 1 : 0, convScale,
                                              qkClip, relClip, clipEps, rmsEps};
    in.nodeParams.nodeParams = &p; in.nodeParams.nodeParamsSize = sizeof(p);
    inst.inputTensorAccessPattern = inAp.data(); inst.outputTensorAccessPattern = outAp.data();
    inst.kernel.kernelElf = elf.data(); inst.kernel.elfSize = elf.size();
    if (InstantiateTpcKernel(&in, &inst) != tpc_lib_api::GLUE_SUCCESS) return 1;

    std::vector<TensorDesc2> descs = {
        projQ.GetTensorDescriptor(), qsc.GetTensorDescriptor(), rsc.GetTensorDescriptor(),
        tau.GetTensorDescriptor(), kvSrc.GetTensorDescriptor(), convSt.GetTensorDescriptor(),
        loadSlots.GetTensorDescriptor(), kvw.GetTensorDescriptor(),
        (isDynamic ? kvdyn : kvw).GetTensorDescriptor(),
        qout.GetTensorDescriptor(), rout.GetTensorDescriptor(), kout.GetTensorDescriptor(),
        vout.GetTensorDescriptor(), newKv.GetTensorDescriptor()};
    VPEStats st;
    const unsigned cyc = tpc_tests::RunSimulation(in, inst, descs, st);

    double errQ = 0, refQ = 0, errR = 0, refR = 0, errK = 0, refK = 0, errV = 0, refV = 0;
    for (int b = 0; b < B; b++)
    {
        const int slot = loadSlots.Data()[b];
        for (int h = 0; h < H; h++)
        {
            double ssq = 0;
            std::vector<double> qv(D), rv(dRel);
            for (int d = 0; d < D; d++) { qv[d] = bf16ToFloat(projQ.Data()[b * qCols + h * D + d]) * bf16ToFloat(qsc.Data()[h * D + d]); ssq += qv[d] * qv[d]; }
            double qf = std::min((double)qkClip / (std::sqrt(ssq) + clipEps), 1.0);
            for (int d = 0; d < D; d++)
            {
                double ref = qv[d] * qf;
                if (hasTau) ref = bf16ToFloat(floatToBf16((float)ref)) * tau.Data()[b];
                errQ = std::max(errQ, std::fabs(bf16ToFloat(qout.Data()[(b * H + h) * D + d]) - ref));
                refQ = std::max(refQ, std::fabs(ref));
            }
            ssq = 0;
            for (int d = 0; d < dRel; d++) { rv[d] = bf16ToFloat(projQ.Data()[b * qCols + H * D + h * dRel + d]) * bf16ToFloat(rsc.Data()[h * dRel + d]); ssq += rv[d] * rv[d]; }
            double rf = std::min((double)relClip / (std::sqrt(ssq) + clipEps), 1.0);
            for (int d = 0; d < dRel; d++)
            {
                double ref = rv[d] * rf;
                if (hasTau) ref = bf16ToFloat(floatToBf16((float)ref)) * tau.Data()[b];
                errR = std::max(errR, std::fabs(bf16ToFloat(rout.Data()[(b * H + h) * dRel + d]) - ref));
                refR = std::max(refR, std::fabs(ref));
            }
            if (h < Hkv)
            {
                std::vector<double> kc(D), vc(D);
                double ksq = 0, vsq = 0;
                for (int d = 0; d < D; d++)
                {
                    auto eval = [&](int kv) {
                        const uint16_t xb = (kvStreams == 2)
                                                ? kvSrc.Data()[(kv * B + b) * kvCols + h * D + d]
                                                : kvSrc.Data()[b * kvCols + kvColOffset + kv * kvSize + h * D + d];
                        const int ch = kv * kvSize + h * D + d;
                        const uint16_t p0b = convSt.Data()[(slot * 3 + 0) * totalDim + ch];
                        const uint16_t p1b = convSt.Data()[(slot * 3 + 1) * totalDim + ch];
                        const uint16_t p2b = convSt.Data()[(slot * 3 + 2) * totalDim + ch];
                        if (newKv.Data()[(((b * 3 + 0) * 2 + kv) * Hkv + h) * D + d] != p1b ||
                            newKv.Data()[(((b * 3 + 1) * 2 + kv) * Hkv + h) * D + d] != p2b ||
                            newKv.Data()[(((b * 3 + 2) * 2 + kv) * Hkv + h) * D + d] != xb)
                        {
                            printf("qkvr_prep newKv shift mismatch\n");
                            std::exit(1);
                        }
                        double xv = bf16ToFloat(xb), p0 = bf16ToFloat(p0b), p1 = bf16ToFloat(p1b), p2 = bf16ToFloat(p2b);
                        double acc = xv;
                        if (!isDynamic)
                        {
                            acc += p0 * bf16ToFloat(kvw.Data()[((0 * 2 + kv) * Hkv + h) * D + d]) +
                                   p1 * bf16ToFloat(kvw.Data()[((1 * 2 + kv) * Hkv + h) * D + d]) +
                                   p2 * bf16ToFloat(kvw.Data()[((2 * 2 + kv) * Hkv + h) * D + d]) +
                                   xv * bf16ToFloat(kvw.Data()[((3 * 2 + kv) * Hkv + h) * D + d]);
                        }
                        else
                        {
                            double d0 = bf16ToFloat(kvdyn.Data()[((kv * B + b) * Hkv + h) * 4 + 0]);
                            double d1 = bf16ToFloat(kvdyn.Data()[((kv * B + b) * Hkv + h) * 4 + 1]);
                            double d2 = bf16ToFloat(kvdyn.Data()[((kv * B + b) * Hkv + h) * 4 + 2]);
                            double d3 = bf16ToFloat(kvdyn.Data()[((kv * B + b) * Hkv + h) * 4 + 3]);
                            acc += xv * (bf16ToFloat(kvw.Data()[((0 * 2 + kv) * Hkv + h) * D + d]) + d0) +
                                   p2 * (bf16ToFloat(kvw.Data()[((1 * 2 + kv) * Hkv + h) * D + d]) + d1) +
                                   p1 * (bf16ToFloat(kvw.Data()[((2 * 2 + kv) * Hkv + h) * D + d]) + d2) +
                                   p0 * (bf16ToFloat(kvw.Data()[((3 * 2 + kv) * Hkv + h) * D + d]) + d3);
                        }
                        return acc * convScale;
                    };
                    kc[d] = eval(0); vc[d] = eval(1);
                    ksq += kc[d] * kc[d]; vsq += vc[d] * vc[d];
                }
                double kf = std::min((double)qkClip / (std::sqrt(ksq) + clipEps), 1.0);
                double vr = 1.0 / std::sqrt(vsq / D + rmsEps);
                for (int d = 0; d < D; d++)
                {
                    double rk = kc[d] * kf, rv2 = vc[d] * vr;
                    errK = std::max(errK, std::fabs(bf16ToFloat(kout.Data()[(b * Hkv + h) * D + d]) - rk));
                    refK = std::max(refK, std::fabs(rk));
                    errV = std::max(errV, std::fabs(bf16ToFloat(vout.Data()[(b * Hkv + h) * D + d]) - rv2));
                    refV = std::max(refV, std::fabs(rv2));
                }
            }
        }
    }
    const double rQ = errQ / std::max(refQ, 1e-12), rR = errR / std::max(refR, 1e-12);
    const double rK = errK / std::max(refK, 1e-12), rV = errV / std::max(refV, 1e-12);
    const bool ok = rQ < 0.01 && rR < 0.01 && rK < 0.01 && rV < 0.01;
    printf("qkvr_prep(tau=%d,dyn=%d,kvs=%d,H=%d,Hkv=%d) | cyc=%u | Q=%.5f R=%.5f K=%.5f V=%.5f -> %s\n",
           (int)hasTau, (int)isDynamic, kvStreams, H, Hkv, cyc, rQ, rR, rK, rV, ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}

int TestPairAndBlendRms(int D = 256)
{
    const int T = 2, S = 3;
    const float eps = 1e-6f;
    std::mt19937 rng(77 + D);
    std::normal_distribution<float> nd(0.0f, 1.0f);

    const uint64_t dSz[] = {(uint64_t)D, (uint64_t)T, (uint64_t)S};
    const uint64_t pSz[] = {(uint64_t)(2 * D), (uint64_t)T, (uint64_t)S};
    const uint64_t nwSz[] = {(uint64_t)D, (uint64_t)S};

    test::Tensor<float, 3> dest(dSz), msg(dSz);
    test::Tensor<uint16_t, 3> pairOut(pSz), gf(pSz), blendOut(dSz);
    test::Tensor<uint16_t, 2> nw(nwSz);
    for (size_t i = 0; i < (size_t)D * T * S; i++) { dest.Data()[i] = nd(rng); msg.Data()[i] = nd(rng); }
    for (size_t i = 0; i < (size_t)2 * D * T * S; i++) gf.Data()[i] = floatToBf16(nd(rng));
    for (size_t i = 0; i < (size_t)D * S; i++) nw.Data()[i] = floatToBf16(0.5f + nd(rng));

    // 1) pair_rms_norm
    {
        tpc_lib_api::HabanaKernelParams in = {};
        tpc_lib_api::HabanaKernelInstantiation inst = {};
        std::vector<tpc_lib_api::Tensor> ins(2), outs(1);
        std::vector<tpc_lib_api::TensorAccessPattern> inAp(2), outAp(1);
        std::vector<char> elf(1 << 20);
        in.deviceId = tpc_lib_api::DEVICE_ID_GAUDI2;
        strcpy(in.guid.name, Inkling2PairRmsNormGaudi2::kGuid);
        in.inputTensors = ins.data(); in.inputTensorNr = 2;
        in.outputTensors = outs.data(); in.outputTensorNr = 1;
        Describe(&ins[0], tpc_lib_api::DATA_F32, dest);
        Describe(&ins[1], tpc_lib_api::DATA_F32, msg);
        Describe(&outs[0], tpc_lib_api::DATA_BF16, pairOut);
        Inkling2PairRmsNormGaudi2::Params p = {eps};
        in.nodeParams.nodeParams = &p; in.nodeParams.nodeParamsSize = sizeof(p);
        inst.inputTensorAccessPattern = inAp.data(); inst.outputTensorAccessPattern = outAp.data();
        inst.kernel.kernelElf = elf.data(); inst.kernel.elfSize = elf.size();
        if (InstantiateTpcKernel(&in, &inst) != tpc_lib_api::GLUE_SUCCESS) return 1;
        std::vector<TensorDesc2> descs = {dest.GetTensorDescriptor(), msg.GetTensorDescriptor(),
                                          pairOut.GetTensorDescriptor()};
        VPEStats st;
        const unsigned cyc = tpc_tests::RunSimulation(in, inst, descs, st);
        double maxErr = 0, maxRef = 0;
        for (int s = 0; s < S; s++)
            for (int t = 0; t < T; t++)
            {
                double sd = 0, sm = 0;
                for (int d = 0; d < D; d++)
                {
                    double dv = dest.Data()[(s * T + t) * D + d], mv = msg.Data()[(s * T + t) * D + d];
                    sd += dv * dv; sm += mv * mv;
                }
                double rd = 1.0 / std::sqrt(sd / D + eps), rm = 1.0 / std::sqrt(sm / D + eps);
                for (int d = 0; d < D; d++)
                {
                    double r0 = dest.Data()[(s * T + t) * D + d] * rd;
                    double r1 = msg.Data()[(s * T + t) * D + d] * rm;
                    maxErr = std::max(maxErr, std::fabs(bf16ToFloat(pairOut.Data()[(s * T + t) * 2 * D + d]) - r0));
                    maxErr = std::max(maxErr, std::fabs(bf16ToFloat(pairOut.Data()[(s * T + t) * 2 * D + D + d]) - r1));
                    maxRef = std::max({maxRef, std::fabs(r0), std::fabs(r1)});
                }
            }
        const double rel = maxErr / std::max(maxRef, 1e-12);
        printf("pair_rms_norm | cyc=%u | rel=%.6f -> %s\n", cyc, rel, rel < 0.01 ? "PASS" : "FAIL");
        if (rel >= 0.01) return 1;
    }

    // 2) blend_rms_norm
    {
        tpc_lib_api::HabanaKernelParams in = {};
        tpc_lib_api::HabanaKernelInstantiation inst = {};
        std::vector<tpc_lib_api::Tensor> ins(4), outs(1);
        std::vector<tpc_lib_api::TensorAccessPattern> inAp(4), outAp(1);
        std::vector<char> elf(1 << 20);
        in.deviceId = tpc_lib_api::DEVICE_ID_GAUDI2;
        strcpy(in.guid.name, Inkling2BlendRmsNormGaudi2::kGuid);
        in.inputTensors = ins.data(); in.inputTensorNr = 4;
        in.outputTensors = outs.data(); in.outputTensorNr = 1;
        Describe(&ins[0], tpc_lib_api::DATA_F32, dest);
        Describe(&ins[1], tpc_lib_api::DATA_F32, msg);
        Describe(&ins[2], tpc_lib_api::DATA_BF16, gf);
        Describe(&ins[3], tpc_lib_api::DATA_BF16, nw);
        Describe(&outs[0], tpc_lib_api::DATA_BF16, blendOut);
        Inkling2BlendRmsNormGaudi2::Params p = {eps};
        in.nodeParams.nodeParams = &p; in.nodeParams.nodeParamsSize = sizeof(p);
        inst.inputTensorAccessPattern = inAp.data(); inst.outputTensorAccessPattern = outAp.data();
        inst.kernel.kernelElf = elf.data(); inst.kernel.elfSize = elf.size();
        if (InstantiateTpcKernel(&in, &inst) != tpc_lib_api::GLUE_SUCCESS) return 1;
        std::vector<TensorDesc2> descs = {dest.GetTensorDescriptor(), msg.GetTensorDescriptor(),
                                          gf.GetTensorDescriptor(), nw.GetTensorDescriptor(),
                                          blendOut.GetTensorDescriptor()};
        VPEStats st;
        const unsigned cyc = tpc_tests::RunSimulation(in, inst, descs, st);
        double maxErr = 0, maxRef = 0;
        for (int s = 0; s < S; s++)
            for (int t = 0; t < T; t++)
            {
                std::vector<double> x(D);
                double ssq = 0;
                for (int d = 0; d < D; d++)
                {
                    double g = std::tanh(bf16ToFloat(gf.Data()[(s * T + t) * 2 * D + d]));
                    double f = std::tanh(bf16ToFloat(gf.Data()[(s * T + t) * 2 * D + D + d]));
                    x[d] = (1.0 - f) * dest.Data()[(s * T + t) * D + d] + g * msg.Data()[(s * T + t) * D + d];
                    ssq += x[d] * x[d];
                }
                double rstd = 1.0 / std::sqrt(ssq / D + eps);
                for (int d = 0; d < D; d++)
                {
                    double ref = x[d] * rstd * bf16ToFloat(nw.Data()[s * D + d]);
                    maxErr = std::max(maxErr, std::fabs(bf16ToFloat(blendOut.Data()[(s * T + t) * D + d]) - ref));
                    maxRef = std::max(maxRef, std::fabs(ref));
                }
            }
        const double rel = maxErr / std::max(maxRef, 1e-12);
        printf("blend_rms_norm | cyc=%u | rel=%.6f -> %s\n", cyc, rel, rel < 0.01 ? "PASS" : "FAIL");
        if (rel >= 0.01) return 1;
    }

    // 3) blend_rms_norm_f8
    {
        auto f8ToF32 = [](uint8_t b) -> float {
            const int sg = (b >> 7) & 1, ex = (b >> 3) & 0xF, mn = b & 0x7;
            if (ex == 0 && mn == 0) return 0.0f;
            const float v = (ex == 0) ? (mn / 8.0f) * std::ldexp(1.0f, -6) : (1.0f + mn / 8.0f) * std::ldexp(1.0f, ex - 7);
            return sg ? -v : v;
        };
        const float fp8Max = 240.0f;
        const uint64_t scSz[] = {(uint64_t)T, (uint64_t)S};
        test::Tensor<uint16_t, 3> bOut(dSz);
        test::Tensor<uint8_t, 3> f8Out(dSz);
        test::Tensor<float, 2> scOut(scSz);

        tpc_lib_api::HabanaKernelParams in = {};
        tpc_lib_api::HabanaKernelInstantiation inst = {};
        std::vector<tpc_lib_api::Tensor> ins(4), outs(3);
        std::vector<tpc_lib_api::TensorAccessPattern> inAp(4), outAp(3);
        std::vector<char> elf(1 << 20);
        in.deviceId = tpc_lib_api::DEVICE_ID_GAUDI2;
        strcpy(in.guid.name, Inkling2BlendRmsNormF8Gaudi2::kGuid);
        in.inputTensors = ins.data(); in.inputTensorNr = 4;
        in.outputTensors = outs.data(); in.outputTensorNr = 3;
        Describe(&ins[0], tpc_lib_api::DATA_F32, dest);
        Describe(&ins[1], tpc_lib_api::DATA_F32, msg);
        Describe(&ins[2], tpc_lib_api::DATA_BF16, gf);
        Describe(&ins[3], tpc_lib_api::DATA_BF16, nw);
        Describe(&outs[0], tpc_lib_api::DATA_BF16, bOut);
        Describe(&outs[1], tpc_lib_api::DATA_F8_143, f8Out);
        Describe(&outs[2], tpc_lib_api::DATA_F32, scOut);
        Inkling2BlendRmsNormF8Gaudi2::Params p = {eps, fp8Max};
        in.nodeParams.nodeParams = &p; in.nodeParams.nodeParamsSize = sizeof(p);
        inst.inputTensorAccessPattern = inAp.data(); inst.outputTensorAccessPattern = outAp.data();
        inst.kernel.kernelElf = elf.data(); inst.kernel.elfSize = elf.size();
        if (InstantiateTpcKernel(&in, &inst) != tpc_lib_api::GLUE_SUCCESS) return 1;
        std::vector<TensorDesc2> descs = {dest.GetTensorDescriptor(), msg.GetTensorDescriptor(),
                                          gf.GetTensorDescriptor(), nw.GetTensorDescriptor(),
                                          bOut.GetTensorDescriptor(), f8Out.GetTensorDescriptor(),
                                          scOut.GetTensorDescriptor()};
        VPEStats st;
        const unsigned cyc = tpc_tests::RunSimulation(in, inst, descs, st);
        double maxErr = 0, maxF8Err = 0, maxScErr = 0, maxRef = 0;
        for (int s = 0; s < S; s++)
            for (int t = 0; t < T; t++)
            {
                std::vector<double> x(D);
                double ssq = 0;
                for (int d = 0; d < D; d++)
                {
                    double g = std::tanh(bf16ToFloat(gf.Data()[(s * T + t) * 2 * D + d]));
                    double f = std::tanh(bf16ToFloat(gf.Data()[(s * T + t) * 2 * D + D + d]));
                    x[d] = (1.0 - f) * dest.Data()[(s * T + t) * D + d] + g * msg.Data()[(s * T + t) * D + d];
                    ssq += x[d] * x[d];
                }
                double rstd = 1.0 / std::sqrt(ssq / D + eps);
                double amax = 0;
                for (int d = 0; d < D; d++)
                {
                    double ref = x[d] * rstd * bf16ToFloat(nw.Data()[s * D + d]);
                    amax = std::max(amax, std::fabs(ref));
                    maxErr = std::max(maxErr, std::fabs(bf16ToFloat(bOut.Data()[(s * T + t) * D + d]) - ref));
                    maxRef = std::max(maxRef, std::fabs(ref));
                }
                double refSc = std::max(amax / fp8Max, 1e-12);
                double gotSc = scOut.Data()[s * T + t];
                maxScErr = std::max(maxScErr, std::fabs(gotSc - refSc) / refSc);
                for (int d = 0; d < D; d++)
                {
                    double ref = x[d] * rstd * bf16ToFloat(nw.Data()[s * D + d]);
                    double gotF8 = f8ToF32(f8Out.Data()[(s * T + t) * D + d]) * gotSc;
                    maxF8Err = std::max(maxF8Err, std::fabs(gotF8 - ref));
                }
            }
        const double rel = maxErr / std::max(maxRef, 1e-12);
        const double relF8 = maxF8Err / std::max(maxRef, 1e-12);
        const bool ok = (rel < 0.01) && (relF8 < 0.07) && (maxScErr < 1e-3);
        printf("blend_rms_norm_f8 | cyc=%u | rel=%.6f f8=%.4f sc=%.2e -> %s\n", cyc, rel, relF8, maxScErr,
               ok ? "PASS" : "FAIL");
        if (!ok) return 1;
    }
    return 0;
}

int TestRelPagedAttn(int slidingWindow, int B, int H = 6, int Hkv = 2, int BS = 8, int pos0 = 10)
{
    const int D = 128, dRel = 16, relExt = std::max(32, pos0 + 8), numPhys = 6, NB = 5;
    const int gateBase = H * (D + dRel);
    const int sinksBase = gateBase + H * D;
    const int qCols = sinksBase + H + 2 * Hkv * D;
    const float scale = 1.0f / std::sqrt((float)D);
    std::mt19937 rng((slidingWindow ? 42 : 41) + B * 7 + H * 13 + BS);
    std::normal_distribution<float> nd(0.0f, 0.5f);

    const uint64_t qSz[] = {(uint64_t)D, (uint64_t)H, (uint64_t)B};
    const uint64_t rSz[] = {(uint64_t)dRel, (uint64_t)H, (uint64_t)B};
    const uint64_t pqSz[] = {(uint64_t)qCols, (uint64_t)B};
    const uint64_t rpSz[] = {(uint64_t)dRel, (uint64_t)relExt};
    const uint64_t kvSz[] = {(uint64_t)D, (uint64_t)Hkv, (uint64_t)BS, (uint64_t)numPhys};
    const uint64_t kvNewSz[] = {(uint64_t)D, (uint64_t)Hkv, (uint64_t)B};
    const uint64_t nbSz[] = {(uint64_t)NB};
    const uint64_t posSz[] = {(uint64_t)B};

    test::Tensor<uint16_t, 3> q(qSz), r(rSz), out(qSz), kNew(kvNewSz), vNew(kvNewSz);
    test::Tensor<uint16_t, 2> projQ(pqSz), relProjT(rpSz);
    test::Tensor<uint16_t, 4> kCache(kvSz), vCache(kvSz);
    test::Tensor<int32_t, 1> blkList(nbSz), blkGroups(nbSz), blkUsage(nbSz), blkStart(nbSz), positions(posSz);

    for (size_t i = 0; i < (size_t)D * H * B; i++) q.Data()[i] = floatToBf16(nd(rng));
    for (size_t i = 0; i < (size_t)dRel * H * B; i++) r.Data()[i] = floatToBf16(nd(rng));
    for (size_t i = 0; i < (size_t)qCols * B; i++) projQ.Data()[i] = floatToBf16(nd(rng));
    for (size_t i = 0; i < (size_t)dRel * relExt; i++) relProjT.Data()[i] = floatToBf16(nd(rng));
    for (size_t i = 0; i < (size_t)D * Hkv * BS * numPhys; i++) { kCache.Data()[i] = floatToBf16(nd(rng)); vCache.Data()[i] = floatToBf16(nd(rng)); }
    for (size_t i = 0; i < (size_t)D * Hkv * B; i++) { kNew.Data()[i] = floatToBf16(nd(rng)); vNew.Data()[i] = floatToBf16(nd(rng)); }

    if (BS == 8)
    {
        // b=0: pos=10 (2 blocks, usage includes current pos=10 which must be skipped in cache & read from kNew/vNew)
        // b=1: pos=4 (1 block)
        // b=2: pos=0 (edge case: 0 cached tokens, attends ONLY to kNew/vNew!)
        blkList.Data()[0] = 5; blkGroups.Data()[0] = 0; blkUsage.Data()[0] = 3; blkStart.Data()[0] = 8;
        blkList.Data()[1] = 2; blkGroups.Data()[1] = 0; blkUsage.Data()[1] = 8; blkStart.Data()[1] = 0;
        blkList.Data()[2] = (B > 1) ? 1 : -1; blkGroups.Data()[2] = (B > 1) ? 1 : -1; blkUsage.Data()[2] = (B > 1) ? 5 : 0; blkStart.Data()[2] = 0;
        blkList.Data()[3] = (B > 2) ? 3 : -1; blkGroups.Data()[3] = (B > 2) ? 2 : -1; blkUsage.Data()[3] = (B > 2) ? 1 : 0; blkStart.Data()[3] = 0;
        blkList.Data()[4] = -1; blkGroups.Data()[4] = -1; blkUsage.Data()[4] = 0; blkStart.Data()[4] = 0;
        positions.Data()[0] = 10;
        if (B > 1) positions.Data()[1] = 4;
        if (B > 2) positions.Data()[2] = 0;
    }
    else
    {
        // Multi-chunk (cs += 64) inside a single block (e.g. BS=128, pos0=80 -> 64 + 16 cached tokens).
        blkList.Data()[0] = 2; blkGroups.Data()[0] = 0; blkUsage.Data()[0] = pos0 + 1; blkStart.Data()[0] = 0;
        blkList.Data()[1] = (B > 1) ? 4 : -1; blkGroups.Data()[1] = (B > 1) ? 1 : -1; blkUsage.Data()[1] = (B > 1) ? 70 : 0; blkStart.Data()[1] = 0;
        blkList.Data()[2] = (B > 2) ? 1 : -1; blkGroups.Data()[2] = (B > 2) ? 2 : -1; blkUsage.Data()[2] = (B > 2) ? 1 : 0; blkStart.Data()[2] = 0;
        blkList.Data()[3] = -1; blkGroups.Data()[3] = -1; blkUsage.Data()[3] = 0; blkStart.Data()[3] = 0;
        blkList.Data()[4] = -1; blkGroups.Data()[4] = -1; blkUsage.Data()[4] = 0; blkStart.Data()[4] = 0;
        positions.Data()[0] = pos0;
        if (B > 1) positions.Data()[1] = 69;
        if (B > 2) positions.Data()[2] = 0;
    }

    tpc_lib_api::HabanaKernelParams in = {};
    tpc_lib_api::HabanaKernelInstantiation inst = {};
    std::vector<tpc_lib_api::Tensor> ins(13), outs(1);
    std::vector<tpc_lib_api::TensorAccessPattern> inAp(13), outAp(1);
    std::vector<char> elf(1 << 20);
    in.deviceId = tpc_lib_api::DEVICE_ID_GAUDI2;
    strcpy(in.guid.name, Inkling2RelPagedAttnGateSinkGaudi2::kGuid);
    in.inputTensors = ins.data(); in.inputTensorNr = 13;
    in.outputTensors = outs.data(); in.outputTensorNr = 1;
    Describe(&ins[0], tpc_lib_api::DATA_BF16, q); Describe(&ins[1], tpc_lib_api::DATA_BF16, r);
    Describe(&ins[2], tpc_lib_api::DATA_BF16, projQ); Describe(&ins[3], tpc_lib_api::DATA_BF16, relProjT);
    Describe(&ins[4], tpc_lib_api::DATA_BF16, kCache); Describe(&ins[5], tpc_lib_api::DATA_BF16, vCache);
    Describe(&ins[6], tpc_lib_api::DATA_BF16, kNew); Describe(&ins[7], tpc_lib_api::DATA_BF16, vNew);
    Describe(&ins[8], tpc_lib_api::DATA_I32, blkList); Describe(&ins[9], tpc_lib_api::DATA_I32, blkGroups);
    Describe(&ins[10], tpc_lib_api::DATA_I32, blkUsage); Describe(&ins[11], tpc_lib_api::DATA_I32, blkStart);
    Describe(&ins[12], tpc_lib_api::DATA_I32, positions);
    Describe(&outs[0], tpc_lib_api::DATA_BF16, out);
    Inkling2RelPagedAttnGateSinkGaudi2::Params p = {scale, slidingWindow, relExt};
    in.nodeParams.nodeParams = &p; in.nodeParams.nodeParamsSize = sizeof(p);
    inst.inputTensorAccessPattern = inAp.data(); inst.outputTensorAccessPattern = outAp.data();
    inst.kernel.kernelElf = elf.data(); inst.kernel.elfSize = elf.size();
    if (InstantiateTpcKernel(&in, &inst) != tpc_lib_api::GLUE_SUCCESS) return 1;
    std::vector<TensorDesc2> descs = {
        q.GetTensorDescriptor(), r.GetTensorDescriptor(), projQ.GetTensorDescriptor(),
        relProjT.GetTensorDescriptor(), kCache.GetTensorDescriptor(), vCache.GetTensorDescriptor(),
        kNew.GetTensorDescriptor(), vNew.GetTensorDescriptor(), blkList.GetTensorDescriptor(),
        blkGroups.GetTensorDescriptor(), blkUsage.GetTensorDescriptor(), blkStart.GetTensorDescriptor(),
        positions.GetTensorDescriptor(), out.GetTensorDescriptor()};
    VPEStats st;
    const unsigned cyc = tpc_tests::RunSimulation(in, inst, descs, st);

    double maxErr = 0, maxRef = 0;
    const int groups = H / Hkv;
    for (int b = 0; b < B; b++)
        for (int h = 0; h < H; h++)
        {
            const int kvh = h / groups;
            const int qpos = positions.Data()[b];
            std::vector<double> scores;
            std::vector<std::vector<double>> vals;
            // Direct current token (kNew, vNew) at dist = 0
            {
                double sc = 0;
                for (int d = 0; d < D; d++)
                    sc += bf16ToFloat(q.Data()[(b * H + h) * D + d]) *
                          bf16ToFloat(kNew.Data()[(b * Hkv + kvh) * D + d]) * scale;
                for (int d = 0; d < dRel; d++)
                    sc += bf16ToFloat(r.Data()[(b * H + h) * dRel + d]) * bf16ToFloat(relProjT.Data()[0 * dRel + d]);
                scores.push_back(sc);
                std::vector<double> vv(D);
                for (int d = 0; d < D; d++) vv[d] = bf16ToFloat(vNew.Data()[(b * Hkv + kvh) * D + d]);
                vals.push_back(std::move(vv));
            }
            for (int bi = 0; bi < NB; bi++)
            {
                if (blkGroups.Data()[bi] != b || blkList.Data()[bi] < 0) continue;
                int phys = blkList.Data()[bi], usage = blkUsage.Data()[bi], stPos = blkStart.Data()[bi];
                for (int s = 0; s < usage; s++)
                {
                    int dist = qpos - (stPos + s);
                    if (dist <= 0 || (slidingWindow > 0 && dist >= slidingWindow)) continue;
                    int cdist = std::min(dist, relExt - 1);
                    double sc = 0;
                    for (int d = 0; d < D; d++)
                        sc += bf16ToFloat(q.Data()[(b * H + h) * D + d]) *
                              bf16ToFloat(kCache.Data()[((phys * BS + s) * Hkv + kvh) * D + d]) * scale;
                    for (int d = 0; d < dRel; d++)
                        sc += bf16ToFloat(r.Data()[(b * H + h) * dRel + d]) *
                              bf16ToFloat(relProjT.Data()[cdist * dRel + d]);
                    scores.push_back(sc);
                    std::vector<double> vv(D);
                    for (int d = 0; d < D; d++)
                        vv[d] = bf16ToFloat(vCache.Data()[((phys * BS + s) * Hkv + kvh) * D + d]);
                    vals.push_back(std::move(vv));
                }
            }
            double m = -1e300;
            for (double sc : scores) m = std::max(m, sc);
            double l = 0;
            std::vector<double> o(D, 0.0);
            for (size_t i = 0; i < scores.size(); i++)
            {
                double p_i = std::exp(scores[i] - m);
                l += p_i;
                for (int d = 0; d < D; d++) o[d] += p_i * vals[i][d];
            }
            double lse = m + std::log(std::max(l, 1e-30));
            double sk = bf16ToFloat(projQ.Data()[b * qCols + sinksBase + h]);
            double sinkSc = 1.0 / (1.0 + std::exp(-(lse - sk)));
            for (int d = 0; d < D; d++)
            {
                double g = 1.0 / (1.0 + std::exp(-bf16ToFloat(projQ.Data()[b * qCols + gateBase + h * D + d])));
                double ref = (o[d] / std::max(l, 1e-30)) * g * sinkSc;
                double got = bf16ToFloat(out.Data()[(b * H + h) * D + d]);
                maxErr = std::max(maxErr, std::fabs(got - ref));
                maxRef = std::max(maxRef, std::fabs(ref));
            }
        }
    const double rel = maxErr / std::max(maxRef, 1e-12);
    printf("rel_paged_attn(sw=%d, B=%d, H=%d/%d, BS=%d) | cyc=%u | rel=%.6f -> %s\n",
           slidingWindow, B, H, Hkv, BS, cyc, rel, rel < 0.01 ? "PASS" : "FAIL");
    return rel < 0.01 ? 0 : 1;
}
}  // namespace

int main()
{
    int rc = 0;
    rc |= TestSconvStatic();
    rc |= TestQkvrPrep(false, false, 1);
    rc |= TestQkvrPrep(false, false, 2);
    rc |= TestQkvrPrep(false, true, 2);
    rc |= TestQkvrPrep(true, false, 2);
    rc |= TestQkvrPrep(true, true, 2);
    // split_kv = 0 path (H < 2 * Hkv)
    rc |= TestQkvrPrep(true, false, 2, 2, 2);
    rc |= TestQkvrPrep(false, true, 1, 2, 2);
    rc |= TestPairAndBlendRms(256);
    rc |= TestPairAndBlendRms(192);  // D & 64 tail
    rc |= TestRelPagedAttn(0, 1);
    rc |= TestRelPagedAttn(6, 1);
    rc |= TestRelPagedAttn(0, 3);
    rc |= TestRelPagedAttn(6, 3);
    // Non-GQA-3 fallback at B >= 3 (groups = 2 != 3 -> general gate_sink ELF)
    rc |= TestRelPagedAttn(0, 3, 4, 2, 8, 10);
    // Multi-chunk cs += 64 loop (BS = 128, pos0 = 80) for both gate_sink (B=1) and gqa3 (B=3)
    rc |= TestRelPagedAttn(0, 1, 6, 2, 128, 80);
    rc |= TestRelPagedAttn(0, 3, 6, 2, 128, 80);
    return rc;
}
