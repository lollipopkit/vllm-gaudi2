// SPDX-License-Identifier: Apache-2.0
// Double-precision simulator tests for Gaudi2 FP8, Qwen3.8 GDN, and LFM3 ShortConv TPC kernels.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

#include "gaudi2_fp8_ops_gaudi2.hpp"
#include "habana/tensor.h"
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

// Exact IEEE-754 round-half-to-nearest-even (RHNE) FP32 -> BF16 (matches Gaudi2 v_convert_f32_to_bf16_b).
uint16_t FloatToBf16Rhne(float x)
{
    uint32_t u = 0;
    memcpy(&u, &x, sizeof(u));
    if ((u & 0x7fffffffu) > 0x7f800000u)
        return (uint16_t)((u >> 16) | 0x0040u);
    const uint32_t lsb = (u >> 16) & 1u;
    u += 0x7fffu + lsb;
    return (uint16_t)(u >> 16);
}

// Gaudi2 FP8 E4M3 (F8_143: bias=7, max finite=240).
double F8ToF64(uint8_t b)
{
    const int sg = (b >> 7) & 1;
    const int ex = (b >> 3) & 0xF;
    const int mn = b & 0x7;
    if (ex == 0 && mn == 0)
        return 0.0;
    const double v = (ex == 0) ? (mn / 8.0) * std::ldexp(1.0, -6) : (1.0 + mn / 8.0) * std::ldexp(1.0, ex - 7);
    return sg ? -v : v;
}

double RefSigmoid(double x) { return 1.0 / (1.0 + std::exp(-x)); }
double RefSilu(double x) { return x * RefSigmoid(x); }
double RefSoftplus(double x)
{
    if (x > 20.0)
        return x;
    if (x < -20.0)
        return std::exp(x);
    return std::log1p(std::exp(x));
}

int TestAddRmsNormF8(int D, int M, bool wF32, bool outBf16, bool zeroRow0 = false)
{
    const float eps = 1e-6f;
    const float fp8Max = 240.0f;
    std::mt19937 rng(1000 + D + M * 7 + (wF32 ? 13 : 0) + (outBf16 ? 29 : 0) + (zeroRow0 ? 53 : 0));
    std::normal_distribution<float> nd(0.0f, 1.2f);

    const uint64_t matSz[] = {(uint64_t)D, (uint64_t)M};
    const uint64_t wSz[] = {(uint64_t)D};
    const uint64_t scSz[] = {1, (uint64_t)M};

    test::Tensor<uint16_t, 2> x(matSz), res(matSz), outRes(matSz), outBf(matSz);
    test::Tensor<float, 1> wf(wSz);
    test::Tensor<uint16_t, 1> wb(wSz);
    test::Tensor<uint8_t, 2> outF8(matSz);
    test::Tensor<float, 2> outSc(scSz);

    for (int m = 0; m < M; m++)
    {
        for (int d = 0; d < D; d++)
        {
            const float xv = (zeroRow0 && m == 0) ? 0.0f : nd(rng);
            const float rv = (zeroRow0 && m == 0) ? 0.0f : nd(rng);
            x.Data()[m * D + d] = floatToBf16(xv);
            res.Data()[m * D + d] = floatToBf16(rv);
        }
    }
    for (int d = 0; d < D; d++)
    {
        const float wv = 1.0f + 0.3f * nd(rng);
        wf.Data()[d] = wv;
        wb.Data()[d] = floatToBf16(wv);
    }

    tpc_lib_api::HabanaKernelParams in = {};
    tpc_lib_api::HabanaKernelInstantiation inst = {};
    const int nOut = outBf16 ? 4 : 3;
    std::vector<tpc_lib_api::Tensor> ins(3), outs(nOut);
    std::vector<tpc_lib_api::TensorAccessPattern> inAp(3), outAp(nOut);
    std::vector<char> elf(1 << 20);
    in.deviceId = tpc_lib_api::DEVICE_ID_GAUDI2;
    strcpy(in.guid.name, Gaudi2AddRmsNormF8Gaudi2::kGuid);
    in.inputTensors = ins.data();
    in.inputTensorNr = 3;
    in.outputTensors = outs.data();
    in.outputTensorNr = nOut;
    Describe(&ins[0], tpc_lib_api::DATA_BF16, x);
    Describe(&ins[1], tpc_lib_api::DATA_BF16, res);
    if (wF32)
        Describe(&ins[2], tpc_lib_api::DATA_F32, wf);
    else
        Describe(&ins[2], tpc_lib_api::DATA_BF16, wb);
    Describe(&outs[0], tpc_lib_api::DATA_F8_143, outF8);
    Describe(&outs[1], tpc_lib_api::DATA_F32, outSc);
    Describe(&outs[2], tpc_lib_api::DATA_BF16, outRes);
    if (outBf16)
        Describe(&outs[3], tpc_lib_api::DATA_BF16, outBf);

    Gaudi2AddRmsNormF8Gaudi2::Params p = {eps, fp8Max};
    in.nodeParams.nodeParams = &p;
    in.nodeParams.nodeParamsSize = sizeof(p);
    inst.inputTensorAccessPattern = inAp.data();
    inst.outputTensorAccessPattern = outAp.data();
    inst.kernel.kernelElf = elf.data();
    inst.kernel.elfSize = elf.size();
    if (InstantiateTpcKernel(&in, &inst) != tpc_lib_api::GLUE_SUCCESS)
    {
        printf("add_rms_norm_f8 instantiate failed\n");
        return 1;
    }

    std::vector<TensorDesc2> descs = {x.GetTensorDescriptor(), res.GetTensorDescriptor(),
                                      wF32 ? wf.GetTensorDescriptor() : wb.GetTensorDescriptor(),
                                      outF8.GetTensorDescriptor(), outSc.GetTensorDescriptor(),
                                      outRes.GetTensorDescriptor()};
    if (outBf16)
        descs.push_back(outBf.GetTensorDescriptor());
    VPEStats st;
    const unsigned cyc = tpc_tests::RunSimulation(in, inst, descs, st);

    double errRes = 0, refRes = 0, errBf = 0, errF8 = 0, refY = 0, errSc = 0;
    for (int m = 0; m < M; m++)
    {
        std::vector<double> sum(D), y(D);
        double msq = 0;
        for (int d = 0; d < D; d++)
        {
            sum[d] = bf16ToFloat(res.Data()[m * D + d]) + bf16ToFloat(x.Data()[m * D + d]);
            errRes = std::max(errRes, std::fabs(bf16ToFloat(outRes.Data()[m * D + d]) - sum[d]));
            refRes = std::max(refRes, std::fabs(sum[d]));
            msq += sum[d] * sum[d];
        }
        const double rstd = 1.0 / std::sqrt(msq / D + eps);
        double amax = 0;
        for (int d = 0; d < D; d++)
        {
            const double wv = wF32 ? (double)wf.Data()[d] : (double)bf16ToFloat(wb.Data()[d]);
            y[d] = bf16ToFloat(floatToBf16((float)(sum[d] * rstd * wv)));
            amax = std::max(amax, std::fabs(y[d]));
            refY = std::max(refY, std::fabs(y[d]));
            if (outBf16)
                errBf = std::max(errBf, std::fabs(bf16ToFloat(outBf.Data()[m * D + d]) - y[d]));
        }
        const double refSc = (amax + 1e-8) / fp8Max;
        const double gotSc = outSc.Data()[m];
        if (!std::isfinite(gotSc) || gotSc <= 0.0)
        {
            printf("add_rms_norm_f8 non-positive/non-finite scale: %g\n", gotSc);
            return 1;
        }
        errSc = std::max(errSc, std::fabs(gotSc - refSc) / refSc);
        for (int d = 0; d < D; d++)
        {
            const double gotF8 = F8ToF64(outF8.Data()[m * D + d]) * gotSc;
            errF8 = std::max(errF8, std::fabs(gotF8 - y[d]));
        }
    }
    const double rRes = errRes / std::max(refRes, 1e-12);
    const double rBf = errBf / std::max(refY, 1e-12);
    const double rF8 = errF8 / std::max(refY, 1e-12);
    const bool ok = rRes < 0.005 && (!outBf16 || rBf < 0.01) && rF8 < 0.07 && errSc < 1e-2;
    printf("add_rms_norm_f8(D=%d,M=%d,wF32=%d,bf16=%d,z0=%d) | cyc=%u | res=%.5f bf=%.5f f8=%.4f sc=%.2e -> %s\n",
           D, M, (int)wF32, (int)outBf16, (int)zeroRow0, cyc, rRes, rBf, rF8, errSc, ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}

int TestSandwichRmsNormF8(int D, int M)
{
    const float eps = 1e-6f;
    const float fp8Max = 240.0f;
    std::mt19937 rng(1500 + D + M * 17);
    std::normal_distribution<float> nd(0.0f, 1.2f);

    const uint64_t matSz[] = {(uint64_t)D, (uint64_t)M};
    const uint64_t wSz[] = {(uint64_t)D};
    const uint64_t scSz[] = {1, (uint64_t)M};

    test::Tensor<uint16_t, 2> x(matSz), res(matSz), outRes(matSz);
    test::Tensor<uint16_t, 1> pw(wSz), nw(wSz);
    test::Tensor<uint8_t, 2> outF8(matSz);
    test::Tensor<float, 2> outSc(scSz);

    for (size_t i = 0; i < (size_t)D * M; i++)
    {
        x.Data()[i] = floatToBf16(nd(rng));
        res.Data()[i] = floatToBf16(nd(rng));
    }
    for (int d = 0; d < D; d++)
    {
        pw.Data()[d] = floatToBf16(1.0f + 0.3f * nd(rng));
        nw.Data()[d] = floatToBf16(1.0f + 0.3f * nd(rng));
    }

    tpc_lib_api::HabanaKernelParams in = {};
    tpc_lib_api::HabanaKernelInstantiation inst = {};
    std::vector<tpc_lib_api::Tensor> ins(4), outs(3);
    std::vector<tpc_lib_api::TensorAccessPattern> inAp(4), outAp(3);
    std::vector<char> elf(1 << 20);
    in.deviceId = tpc_lib_api::DEVICE_ID_GAUDI2;
    strcpy(in.guid.name, Gaudi2SandwichRmsNormF8Gaudi2::kGuid);
    in.inputTensors = ins.data();
    in.inputTensorNr = 4;
    in.outputTensors = outs.data();
    in.outputTensorNr = 3;
    Describe(&ins[0], tpc_lib_api::DATA_BF16, x);
    Describe(&ins[1], tpc_lib_api::DATA_BF16, res);
    Describe(&ins[2], tpc_lib_api::DATA_BF16, pw);
    Describe(&ins[3], tpc_lib_api::DATA_BF16, nw);
    Describe(&outs[0], tpc_lib_api::DATA_F8_143, outF8);
    Describe(&outs[1], tpc_lib_api::DATA_F32, outSc);
    Describe(&outs[2], tpc_lib_api::DATA_BF16, outRes);

    Gaudi2SandwichRmsNormF8Gaudi2::Params p = {eps, fp8Max};
    in.nodeParams.nodeParams = &p;
    in.nodeParams.nodeParamsSize = sizeof(p);
    inst.inputTensorAccessPattern = inAp.data();
    inst.outputTensorAccessPattern = outAp.data();
    inst.kernel.kernelElf = elf.data();
    inst.kernel.elfSize = elf.size();
    if (InstantiateTpcKernel(&in, &inst) != tpc_lib_api::GLUE_SUCCESS)
    {
        printf("sandwich_rms_norm_f8 instantiate failed\n");
        return 1;
    }

    std::vector<TensorDesc2> descs = {x.GetTensorDescriptor(), res.GetTensorDescriptor(), pw.GetTensorDescriptor(),
                                      nw.GetTensorDescriptor(), outF8.GetTensorDescriptor(),
                                      outSc.GetTensorDescriptor(), outRes.GetTensorDescriptor()};
    VPEStats st;
    const unsigned cyc = tpc_tests::RunSimulation(in, inst, descs, st);

    double errRes = 0, refRes = 0, errF8 = 0, refY = 0, errSc = 0;
    for (int m = 0; m < M; m++)
    {
        std::vector<double> rnew(D), y(D);
        double msq1 = 0;
        for (int d = 0; d < D; d++)
        {
            const double xv = bf16ToFloat(x.Data()[m * D + d]);
            msq1 += xv * xv;
        }
        const double rstd1 = 1.0 / std::sqrt(msq1 / D + eps);
        double msq2 = 0;
        for (int d = 0; d < D; d++)
        {
            const double xp = bf16ToFloat(FloatToBf16Rhne((float)(bf16ToFloat(x.Data()[m * D + d]) * rstd1 *
                                                                  bf16ToFloat(pw.Data()[d]))));
            rnew[d] = bf16ToFloat(FloatToBf16Rhne((float)(bf16ToFloat(res.Data()[m * D + d]) + xp)));
            errRes = std::max(errRes, std::fabs(bf16ToFloat(outRes.Data()[m * D + d]) - rnew[d]));
            refRes = std::max(refRes, std::fabs(rnew[d]));
            msq2 += rnew[d] * rnew[d];
        }
        const double rstd2 = 1.0 / std::sqrt(msq2 / D + eps);
        double amax = 0;
        for (int d = 0; d < D; d++)
        {
            y[d] = bf16ToFloat(FloatToBf16Rhne((float)(rnew[d] * rstd2 * bf16ToFloat(nw.Data()[d]))));
            amax = std::max(amax, std::fabs(y[d]));
            refY = std::max(refY, std::fabs(y[d]));
        }
        const double refSc = (amax + 1e-8) / fp8Max;
        const double gotSc = outSc.Data()[m];
        errSc = std::max(errSc, std::fabs(gotSc - refSc) / refSc);
        for (int d = 0; d < D; d++)
            errF8 = std::max(errF8, std::fabs(F8ToF64(outF8.Data()[m * D + d]) * gotSc - y[d]));
    }
    const double rRes = errRes / std::max(refRes, 1e-12);
    const double rF8 = errF8 / std::max(refY, 1e-12);
    const bool ok = rRes < 0.005 && rF8 < 0.07 && errSc < 1e-2;
    printf("sandwich_rms_norm_f8(D=%d,M=%d) | cyc=%u | res=%.5f f8=%.4f sc=%.2e -> %s\n", D, M, cyc, rRes, rF8, errSc,
           ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}

int TestSiluMulF8(int N, int M)
{
    const float fp8Max = 240.0f;
    std::mt19937 rng(2000 + N + M * 11);
    std::normal_distribution<float> nd(0.0f, 3.0f);

    const uint64_t guSz[] = {(uint64_t)(2 * N), (uint64_t)M};
    const uint64_t outSz[] = {(uint64_t)N, (uint64_t)M};
    const uint64_t scSz[] = {1, (uint64_t)M};

    test::Tensor<uint16_t, 2> gu(guSz);
    test::Tensor<uint8_t, 2> outF8(outSz);
    test::Tensor<float, 2> outSc(scSz);

    for (int m = 0; m < M; m++)
    {
        for (int i = 0; i < 2 * N; i++)
            gu.Data()[m * 2 * N + i] = floatToBf16(nd(rng));
        // Inject extreme gate values to exercise exp clamping in silu_zero_lut.
        gu.Data()[m * 2 * N + 0] = floatToBf16(-35.0f);
        gu.Data()[m * 2 * N + 1] = floatToBf16(35.0f);
    }

    tpc_lib_api::HabanaKernelParams in = {};
    tpc_lib_api::HabanaKernelInstantiation inst = {};
    std::vector<tpc_lib_api::Tensor> ins(1), outs(2);
    std::vector<tpc_lib_api::TensorAccessPattern> inAp(1), outAp(2);
    std::vector<char> elf(1 << 20);
    in.deviceId = tpc_lib_api::DEVICE_ID_GAUDI2;
    strcpy(in.guid.name, Gaudi2SiluMulF8Gaudi2::kGuid);
    in.inputTensors = ins.data();
    in.inputTensorNr = 1;
    in.outputTensors = outs.data();
    in.outputTensorNr = 2;
    Describe(&ins[0], tpc_lib_api::DATA_BF16, gu);
    Describe(&outs[0], tpc_lib_api::DATA_F8_143, outF8);
    Describe(&outs[1], tpc_lib_api::DATA_F32, outSc);

    Gaudi2SiluMulF8Gaudi2::Params p = {fp8Max};
    in.nodeParams.nodeParams = &p;
    in.nodeParams.nodeParamsSize = sizeof(p);
    inst.inputTensorAccessPattern = inAp.data();
    inst.outputTensorAccessPattern = outAp.data();
    inst.kernel.kernelElf = elf.data();
    inst.kernel.elfSize = elf.size();
    if (InstantiateTpcKernel(&in, &inst) != tpc_lib_api::GLUE_SUCCESS)
    {
        printf("silu_mul_f8 instantiate failed\n");
        return 1;
    }

    std::vector<TensorDesc2> descs = {gu.GetTensorDescriptor(), outF8.GetTensorDescriptor(),
                                      outSc.GetTensorDescriptor()};
    VPEStats st;
    const unsigned cyc = tpc_tests::RunSimulation(in, inst, descs, st);

    double errF8 = 0, refY = 0, errSc = 0;
    for (int m = 0; m < M; m++)
    {
        std::vector<double> y(N);
        double amax = 0;
        for (int n = 0; n < N; n++)
        {
            const double g = bf16ToFloat(gu.Data()[m * 2 * N + n]);
            const double u = bf16ToFloat(gu.Data()[m * 2 * N + N + n]);
            y[n] = bf16ToFloat(floatToBf16((float)(RefSilu(g) * u)));
            amax = std::max(amax, std::fabs(y[n]));
            refY = std::max(refY, std::fabs(y[n]));
        }
        const double refSc = (amax + 1e-8) / fp8Max;
        const double gotSc = outSc.Data()[m];
        errSc = std::max(errSc, std::fabs(gotSc - refSc) / refSc);
        for (int n = 0; n < N; n++)
        {
            const double gotF8 = F8ToF64(outF8.Data()[m * N + n]) * gotSc;
            errF8 = std::max(errF8, std::fabs(gotF8 - y[n]));
        }
    }
    const double rF8 = errF8 / std::max(refY, 1e-12);
    const bool ok = rF8 < 0.07 && errSc < 1e-2;
    printf("silu_mul_f8(N=%d,M=%d) | cyc=%u | f8=%.4f sc=%.2e -> %s\n", N, M, cyc, rF8, errSc, ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}

int TestGdnPrepDecode(int H, int HV, int B, int hasBias, int convColOffset, int rowOffset)
{
    const int D = 128;
    const int W = (2 * H + HV) * D;
    const int cacheLen = convColOffset + 3;
    const int slots = rowOffset + B + 2;
    std::mt19937 rng(3000 + H * 3 + HV * 5 + B * 7 + hasBias * 13 + convColOffset * 17);
    std::normal_distribution<float> nd(0.0f, 1.0f);

    const uint64_t qkvSz[] = {(uint64_t)W, (uint64_t)B};
    const uint64_t abSz[] = {(uint64_t)HV, (uint64_t)B};
    const uint64_t hv1Sz[] = {(uint64_t)HV};
    const uint64_t csSz[] = {(uint64_t)W, (uint64_t)cacheLen, (uint64_t)slots};
    const uint64_t tapsSz[] = {(uint64_t)W, 4};
    const uint64_t biasSz[] = {(uint64_t)W};
    const uint64_t qkOutSz[] = {(uint64_t)D, (uint64_t)(2 * H), (uint64_t)B};
    const uint64_t vOutSz[] = {(uint64_t)D, (uint64_t)HV, (uint64_t)B};
    const uint64_t kqdSz[] = {(uint64_t)H, (uint64_t)B};

    test::Tensor<uint16_t, 2> qkv(qkvSz), aIn(abSz), bIn(abSz);
    test::Tensor<float, 1> negExpA(hv1Sz), dtBias(hv1Sz), convBias(biasSz);
    test::Tensor<uint16_t, 3> convState(csSz);
    test::Tensor<float, 2> convTaps(tapsSz), decayOut(abSz), betaOut(abSz), kqdotOut(kqdSz);
    test::Tensor<float, 3> qkOut(qkOutSz), vOut(vOutSz);

    for (size_t i = 0; i < (size_t)W * B; i++) qkv.Data()[i] = floatToBf16(nd(rng));
    for (int b = 0; b < B; b++)
    {
        for (int h = 0; h < HV; h++)
        {
            aIn.Data()[b * HV + h] = floatToBf16(nd(rng) * 3.0f);
            bIn.Data()[b * HV + h] = floatToBf16(nd(rng) * 3.0f);
        }
        // Test softplus branches (>20 and <-20) explicitly on heads 0 and 1.
        aIn.Data()[b * HV + 0] = floatToBf16(25.0f);
        aIn.Data()[b * HV + 1] = floatToBf16(-25.0f);
    }
    for (int h = 0; h < HV; h++)
    {
        negExpA.Data()[h] = -std::exp(nd(rng) * 0.3f);
        dtBias.Data()[h] = nd(rng) * 0.5f;
    }
    for (size_t i = 0; i < (size_t)W * cacheLen * slots; i++) convState.Data()[i] = floatToBf16(nd(rng));
    for (size_t i = 0; i < (size_t)W * 4; i++) convTaps.Data()[i] = nd(rng) * 0.5f;
    for (int i = 0; i < W; i++) convBias.Data()[i] = nd(rng) * 0.2f;

    std::vector<uint16_t> origState(convState.Data(), convState.Data() + (size_t)W * cacheLen * slots);

    tpc_lib_api::HabanaKernelParams in = {};
    tpc_lib_api::HabanaKernelInstantiation inst = {};
    std::vector<tpc_lib_api::Tensor> ins(8), outs(5);
    std::vector<tpc_lib_api::TensorAccessPattern> inAp(8), outAp(5);
    std::vector<char> elf(1 << 20);
    in.deviceId = tpc_lib_api::DEVICE_ID_GAUDI2;
    strcpy(in.guid.name, GdnPrepDecodeGaudi2::kGuid);
    in.inputTensors = ins.data();
    in.inputTensorNr = 8;
    in.outputTensors = outs.data();
    in.outputTensorNr = 5;
    Describe(&ins[0], tpc_lib_api::DATA_BF16, qkv);
    Describe(&ins[1], tpc_lib_api::DATA_BF16, aIn);
    Describe(&ins[2], tpc_lib_api::DATA_BF16, bIn);
    Describe(&ins[3], tpc_lib_api::DATA_F32, negExpA);
    Describe(&ins[4], tpc_lib_api::DATA_F32, dtBias);
    Describe(&ins[5], tpc_lib_api::DATA_BF16, convState);
    Describe(&ins[6], tpc_lib_api::DATA_F32, convTaps);
    Describe(&ins[7], tpc_lib_api::DATA_F32, convBias);
    Describe(&outs[0], tpc_lib_api::DATA_F32, qkOut);
    Describe(&outs[1], tpc_lib_api::DATA_F32, vOut);
    Describe(&outs[2], tpc_lib_api::DATA_F32, decayOut);
    Describe(&outs[3], tpc_lib_api::DATA_F32, betaOut);
    Describe(&outs[4], tpc_lib_api::DATA_F32, kqdotOut);

    GdnPrepDecodeGaudi2::Params p = {rowOffset, convColOffset, hasBias};
    in.nodeParams.nodeParams = &p;
    in.nodeParams.nodeParamsSize = sizeof(p);
    inst.inputTensorAccessPattern = inAp.data();
    inst.outputTensorAccessPattern = outAp.data();
    inst.kernel.kernelElf = elf.data();
    inst.kernel.elfSize = elf.size();
    if (InstantiateTpcKernel(&in, &inst) != tpc_lib_api::GLUE_SUCCESS)
    {
        printf("gdn_prep_decode instantiate failed\n");
        return 1;
    }

    std::vector<TensorDesc2> descs = {
        qkv.GetTensorDescriptor(),      aIn.GetTensorDescriptor(),      bIn.GetTensorDescriptor(),
        negExpA.GetTensorDescriptor(),  dtBias.GetTensorDescriptor(),   convState.GetTensorDescriptor(),
        convTaps.GetTensorDescriptor(), convBias.GetTensorDescriptor(), qkOut.GetTensorDescriptor(),
        vOut.GetTensorDescriptor(),     decayOut.GetTensorDescriptor(), betaOut.GetTensorDescriptor(),
        kqdotOut.GetTensorDescriptor()};
    VPEStats st;
    const unsigned cyc = tpc_tests::RunSimulation(in, inst, descs, st);

    // Check untouched slots/columns and exact shift on updated rows.
    for (int s = 0; s < slots; s++)
    {
        const bool active = (s >= rowOffset && s < rowOffset + B);
        const int b = s - rowOffset;
        for (int c = 0; c < cacheLen; c++)
        {
            for (int w = 0; w < W; w++)
            {
                const size_t idx = ((size_t)s * cacheLen + c) * W + w;
                uint16_t exp = origState[idx];
                if (active && c >= convColOffset && c < convColOffset + 3)
                {
                    if (c == convColOffset + 0)
                        exp = origState[((size_t)s * cacheLen + convColOffset + 1) * W + w];
                    else if (c == convColOffset + 1)
                        exp = origState[((size_t)s * cacheLen + convColOffset + 2) * W + w];
                    else
                        exp = qkv.Data()[(size_t)b * W + w];
                }
                if (convState.Data()[idx] != exp)
                {
                    printf("gdn_prep_decode conv_state mismatch at s=%d c=%d w=%d\n", s, c, w);
                    return 1;
                }
            }
        }
    }

    double errDecay = 0, errBeta = 0, errQK = 0, errV = 0, refV = 0, errDot = 0;
    for (int b = 0; b < B; b++)
    {
        const int row = rowOffset + b;
        for (int h = 0; h < HV; h++)
        {
            const double av = bf16ToFloat(aIn.Data()[b * HV + h]);
            const double bv = bf16ToFloat(bIn.Data()[b * HV + h]);
            const double decRef = std::exp((double)negExpA.Data()[h] * RefSoftplus(av + (double)dtBias.Data()[h]));
            const double betRef = bf16ToFloat(floatToBf16((float)RefSigmoid(bv)));
            errDecay = std::max(errDecay, std::fabs(decayOut.Data()[b * HV + h] - decRef));
            errBeta = std::max(errBeta, std::fabs(betaOut.Data()[b * HV + h] - betRef));
        }

        auto evalConvSilu = [&](int col) -> double {
            const double s0 = bf16ToFloat(origState[((size_t)row * cacheLen + convColOffset + 0) * W + col]);
            const double s1 = bf16ToFloat(origState[((size_t)row * cacheLen + convColOffset + 1) * W + col]);
            const double s2 = bf16ToFloat(origState[((size_t)row * cacheLen + convColOffset + 2) * W + col]);
            const double xv = bf16ToFloat(qkv.Data()[(size_t)b * W + col]);
            double acc = s0 * convTaps.Data()[0 * W + col] + s1 * convTaps.Data()[1 * W + col] +
                         s2 * convTaps.Data()[2 * W + col] + xv * convTaps.Data()[3 * W + col];
            if (hasBias)
                acc += convBias.Data()[col];
            const double cb = bf16ToFloat(floatToBf16((float)acc));
            return bf16ToFloat(floatToBf16((float)RefSilu(cb)));
        };

        for (int kh = 0; kh < H; kh++)
        {
            std::vector<double> q(D), k(D);
            double qsq = 0, ksq = 0;
            for (int d = 0; d < D; d++)
            {
                q[d] = evalConvSilu(kh * D + d);
                k[d] = evalConvSilu((H + kh) * D + d);
                qsq += q[d] * q[d];
                ksq += k[d] * k[d];
            }
            const double qi = 1.0 / std::sqrt(qsq + 1e-6);
            const double ki = 1.0 / std::sqrt(ksq + 1e-6);
            double dotRef = 0;
            for (int d = 0; d < D; d++)
            {
                const double qn = q[d] * qi;
                const double kn = k[d] * ki;
                dotRef += kn * qn;
                errQK = std::max(errQK, std::fabs(qkOut.Data()[((size_t)b * 2 * H + kh) * D + d] - qn));
                errQK = std::max(errQK, std::fabs(qkOut.Data()[((size_t)b * 2 * H + H + kh) * D + d] - kn));
            }
            errDot = std::max(errDot, std::fabs(kqdotOut.Data()[b * H + kh] - dotRef));
        }

        for (int vh = 0; vh < HV; vh++)
        {
            for (int d = 0; d < D; d++)
            {
                const double rv = evalConvSilu((2 * H + vh) * D + d);
                const double gv = vOut.Data()[((size_t)b * HV + vh) * D + d];
                errV = std::max(errV, std::fabs(gv - rv));
                refV = std::max(refV, std::fabs(rv));
            }
        }
    }
    const double rV = errV / std::max(refV, 1e-12);
    const bool ok = errDecay < 1e-5 && errBeta < 1e-2 && errQK < 2e-3 && rV < 1e-2 && errDot < 2e-3;
    printf("gdn_prep_decode(H=%d,HV=%d,B=%d,bias=%d,coff=%d) | cyc=%u | dec=%.2e bet=%.4f qk=%.4f v=%.4f dot=%.4f -> %s\n",
           H, HV, B, hasBias, convColOffset, cyc, errDecay, errBeta, errQK, rV, errDot, ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}

int TestGdnGatedRmsNormF8(int HV, int B)
{
    const int D = 128;
    const float eps = 1e-6f;
    const float fp8Max = 240.0f;
    std::mt19937 rng(4000 + HV * 7 + B * 13);
    std::normal_distribution<float> nd(0.0f, 1.5f);

    const uint64_t xSz[] = {(uint64_t)D, (uint64_t)HV, (uint64_t)B};
    const uint64_t zSz[] = {(uint64_t)(HV * D), (uint64_t)B};
    const uint64_t wSz[] = {(uint64_t)D};
    const uint64_t scSz[] = {1, (uint64_t)B};

    test::Tensor<float, 3> x(xSz);
    test::Tensor<uint16_t, 2> z(zSz);
    test::Tensor<float, 1> w(wSz);
    test::Tensor<uint8_t, 2> outF8(zSz);
    test::Tensor<float, 2> outSc(scSz);

    for (size_t i = 0; i < (size_t)D * HV * B; i++) x.Data()[i] = nd(rng);
    for (size_t i = 0; i < (size_t)D * HV * B; i++) z.Data()[i] = floatToBf16(nd(rng) * 2.0f);
    for (int d = 0; d < D; d++) w.Data()[d] = 0.8f + 0.3f * nd(rng);

    tpc_lib_api::HabanaKernelParams in = {};
    tpc_lib_api::HabanaKernelInstantiation inst = {};
    std::vector<tpc_lib_api::Tensor> ins(3), outs(2);
    std::vector<tpc_lib_api::TensorAccessPattern> inAp(3), outAp(2);
    std::vector<char> elf(1 << 20);
    in.deviceId = tpc_lib_api::DEVICE_ID_GAUDI2;
    strcpy(in.guid.name, GdnGatedRmsNormF8Gaudi2::kGuid);
    in.inputTensors = ins.data();
    in.inputTensorNr = 3;
    in.outputTensors = outs.data();
    in.outputTensorNr = 2;
    Describe(&ins[0], tpc_lib_api::DATA_F32, x);
    Describe(&ins[1], tpc_lib_api::DATA_BF16, z);
    Describe(&ins[2], tpc_lib_api::DATA_F32, w);
    Describe(&outs[0], tpc_lib_api::DATA_F8_143, outF8);
    Describe(&outs[1], tpc_lib_api::DATA_F32, outSc);

    GdnGatedRmsNormF8Gaudi2::Params p = {eps, fp8Max};
    in.nodeParams.nodeParams = &p;
    in.nodeParams.nodeParamsSize = sizeof(p);
    inst.inputTensorAccessPattern = inAp.data();
    inst.outputTensorAccessPattern = outAp.data();
    inst.kernel.kernelElf = elf.data();
    inst.kernel.elfSize = elf.size();
    if (InstantiateTpcKernel(&in, &inst) != tpc_lib_api::GLUE_SUCCESS)
    {
        printf("gdn_gated_rms_norm_f8 instantiate failed\n");
        return 1;
    }

    std::vector<TensorDesc2> descs = {x.GetTensorDescriptor(), z.GetTensorDescriptor(), w.GetTensorDescriptor(),
                                      outF8.GetTensorDescriptor(), outSc.GetTensorDescriptor()};
    VPEStats st;
    const unsigned cyc = tpc_tests::RunSimulation(in, inst, descs, st);

    double errF8 = 0, refY = 0, errSc = 0;
    for (int b = 0; b < B; b++)
    {
        std::vector<double> y(HV * D);
        double amax = 0;
        for (int h = 0; h < HV; h++)
        {
            double ssq = 0;
            for (int d = 0; d < D; d++)
            {
                const double xv = x.Data()[((size_t)b * HV + h) * D + d];
                ssq += xv * xv;
            }
            const double rstd = 1.0 / std::sqrt(ssq / D + eps);
            for (int d = 0; d < D; d++)
            {
                const int idx = h * D + d;
                const double xv = x.Data()[((size_t)b * HV + h) * D + d];
                const double zv = bf16ToFloat(z.Data()[(size_t)b * HV * D + idx]);
                const double val = ((xv * rstd) * w.Data()[d]) * RefSilu(zv);
                y[idx] = bf16ToFloat(floatToBf16((float)val));
                amax = std::max(amax, std::fabs(y[idx]));
                refY = std::max(refY, std::fabs(y[idx]));
            }
        }
        const double refSc = (amax + 1e-8) / fp8Max;
        const double gotSc = outSc.Data()[b];
        errSc = std::max(errSc, std::fabs(gotSc - refSc) / refSc);
        for (int i = 0; i < HV * D; i++)
        {
            const double gotF8 = F8ToF64(outF8.Data()[(size_t)b * HV * D + i]) * gotSc;
            errF8 = std::max(errF8, std::fabs(gotF8 - y[i]));
        }
    }
    const double rF8 = errF8 / std::max(refY, 1e-12);
    const bool ok = rF8 < 0.07 && errSc < 1e-2;
    printf("gdn_gated_rms_norm_f8(HV=%d,B=%d) | cyc=%u | f8=%.4f sc=%.2e -> %s\n", HV, B, cyc, rF8, errSc,
           ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}

int TestLfm3ShortConvDecode(int D, int B, int K, int cacheLen, int hasBias)
{
    const int slots = 6;
    const int stateLen = K - 1;
    const int colOff = cacheLen - stateLen;
    std::mt19937 rng(5000 + D + B * 13 + K * 19 + hasBias * 23);
    std::normal_distribution<float> nd(0.0f, 1.0f);

    const uint64_t bcxSz[] = {(uint64_t)(3 * D), (uint64_t)B};
    const uint64_t csSz[] = {(uint64_t)D, (uint64_t)cacheLen, (uint64_t)slots};
    const uint64_t idxSz[] = {(uint64_t)B};
    const uint64_t tapsSz[] = {(uint64_t)D, (uint64_t)K};
    const uint64_t biasSz[] = {(uint64_t)D};
    const uint64_t outSz[] = {(uint64_t)D, (uint64_t)B};
    const uint64_t newStateSz[] = {(uint64_t)D, (uint64_t)stateLen, (uint64_t)B};

    test::Tensor<uint16_t, 2> bcx(bcxSz), out(outSz);
    test::Tensor<uint16_t, 3> convState(csSz), newState(newStateSz);
    test::Tensor<int32_t, 1> loadIdx(idxSz);
    test::Tensor<float, 2> taps(tapsSz);
    test::Tensor<float, 1> bias(biasSz);

    for (size_t i = 0; i < (size_t)3 * D * B; i++) bcx.Data()[i] = FloatToBf16Rhne(nd(rng));
    for (size_t i = 0; i < (size_t)D * cacheLen * slots; i++) convState.Data()[i] = FloatToBf16Rhne(nd(rng));
    for (size_t i = 0; i < (size_t)D * K; i++) taps.Data()[i] = nd(rng) * 0.4f;
    for (int d = 0; d < D; d++) bias.Data()[d] = nd(rng) * 0.2f;

    loadIdx.Data()[0] = 2;
    if (B > 1)
        loadIdx.Data()[1] = 2;  // duplicate load slot across batch items (prefix-cache / shared-slot safety)
    if (B > 2)
        loadIdx.Data()[2] = -1;  // out-of-range padding slot -> zero out & zero new_state

    std::vector<uint16_t> origState(convState.Data(), convState.Data() + (size_t)D * cacheLen * slots);

    tpc_lib_api::HabanaKernelParams in = {};
    tpc_lib_api::HabanaKernelInstantiation inst = {};
    std::vector<tpc_lib_api::Tensor> ins(5), outs(2);
    std::vector<tpc_lib_api::TensorAccessPattern> inAp(5), outAp(2);
    std::vector<char> elf(1 << 20);
    in.deviceId = tpc_lib_api::DEVICE_ID_GAUDI2;
    strcpy(in.guid.name, Lfm3ShortConvDecodeGaudi2::kGuid);
    in.inputTensors = ins.data();
    in.inputTensorNr = 5;
    in.outputTensors = outs.data();
    in.outputTensorNr = 2;
    Describe(&ins[0], tpc_lib_api::DATA_BF16, bcx);
    Describe(&ins[1], tpc_lib_api::DATA_BF16, convState);
    Describe(&ins[2], tpc_lib_api::DATA_I32, loadIdx);
    Describe(&ins[3], tpc_lib_api::DATA_F32, taps);
    Describe(&ins[4], tpc_lib_api::DATA_F32, bias);
    Describe(&outs[0], tpc_lib_api::DATA_BF16, out);
    Describe(&outs[1], tpc_lib_api::DATA_BF16, newState);

    Lfm3ShortConvDecodeGaudi2::Params p = {hasBias};
    in.nodeParams.nodeParams = &p;
    in.nodeParams.nodeParamsSize = sizeof(p);
    inst.inputTensorAccessPattern = inAp.data();
    inst.outputTensorAccessPattern = outAp.data();
    inst.kernel.kernelElf = elf.data();
    inst.kernel.elfSize = elf.size();
    if (InstantiateTpcKernel(&in, &inst) != tpc_lib_api::GLUE_SUCCESS)
    {
        printf("lfm3_short_conv_decode instantiate failed\n");
        return 1;
    }

    std::vector<TensorDesc2> descs = {bcx.GetTensorDescriptor(),      convState.GetTensorDescriptor(),
                                      loadIdx.GetTensorDescriptor(),  taps.GetTensorDescriptor(),
                                      bias.GetTensorDescriptor(),     out.GetTensorDescriptor(),
                                      newState.GetTensorDescriptor()};
    VPEStats st;
    const unsigned cyc = tpc_tests::RunSimulation(in, inst, descs, st);

    // Verify conv_state input is strictly read-only.
    for (size_t i = 0; i < origState.size(); i++)
    {
        if (convState.Data()[i] != origState[i])
        {
            printf("lfm3_short_conv_decode unexpectedly mutated conv_state at idx=%zu\n", i);
            return 1;
        }
    }

    double maxErr = 0, maxRef = 0;
    for (int b = 0; b < B; b++)
    {
        const int ls = loadIdx.Data()[b];
        if (ls < 0 || ls >= slots)
        {
            for (int d = 0; d < D; d++)
            {
                if (out.Data()[b * D + d] != 0)
                {
                    printf("lfm3_short_conv_decode padding row b=%d d=%d not zero\n", b, d);
                    return 1;
                }
                for (int k = 0; k < stateLen; k++)
                {
                    if (newState.Data()[((size_t)b * stateLen + k) * D + d] != 0)
                    {
                        printf("lfm3_short_conv_decode padding new_state b=%d k=%d d=%d not zero\n", b, k, d);
                        return 1;
                    }
                }
            }
            continue;
        }
        for (int d = 0; d < D; d++)
        {
            const float bg = bf16ToFloat(bcx.Data()[(size_t)b * 3 * D + d]);
            const float cg = bf16ToFloat(bcx.Data()[(size_t)b * 3 * D + D + d]);
            const float xv = bf16ToFloat(bcx.Data()[(size_t)b * 3 * D + 2 * D + d]);
            const uint16_t bxBf16 = FloatToBf16Rhne(bg * xv);
            const double bx = bf16ToFloat(bxBf16);
            double acc = 0.0;
            for (int k = 0; k < stateLen; k++)
            {
                const uint16_t svBf16 = origState[((size_t)ls * cacheLen + colOff + k) * D + d];
                acc += bf16ToFloat(svBf16) * taps.Data()[(size_t)k * D + d];
                if (k > 0)
                {
                    const uint16_t gotNs = newState.Data()[((size_t)b * stateLen + (k - 1)) * D + d];
                    if (gotNs != svBf16)
                    {
                        printf("lfm3_short_conv_decode new_state shift mismatch b=%d k=%d d=%d\n", b, k, d);
                        return 1;
                    }
                }
            }
            acc += bx * taps.Data()[(size_t)stateLen * D + d];
            if (hasBias)
                acc += bias.Data()[d];
            const uint16_t gotLast = newState.Data()[((size_t)b * stateLen + (stateLen - 1)) * D + d];
            if (gotLast != bxBf16)
            {
                printf("lfm3_short_conv_decode new_state bx mismatch b=%d d=%d (got=%u exp=%u)\n", b, d,
                       (unsigned)gotLast, (unsigned)bxBf16);
                return 1;
            }
            const double ref = cg * bf16ToFloat(FloatToBf16Rhne((float)acc));
            const double got = bf16ToFloat(out.Data()[b * D + d]);
            maxErr = std::max(maxErr, std::fabs(got - ref));
            maxRef = std::max(maxRef, std::fabs(ref));
        }
    }

    const double rel = maxErr / std::max(maxRef, 1e-12);
    const bool ok = rel < 0.01;
    printf("lfm3_short_conv_decode(D=%d,B=%d,K=%d,cl=%d,bias=%d) | cyc=%u | rel=%.6f -> %s\n", D, B, K, cacheLen,
           hasBias, cyc, rel, ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
}  // namespace

int main()
{
    int rc = 0;
    rc |= TestAddRmsNormF8(64, 2, false, false);    // min dim, tail-only
    rc |= TestAddRmsNormF8(128, 2, true, true);     // 128-only tail
    rc |= TestAddRmsNormF8(5120, 2, true, false);   // hoist_w=1, F32w, 3-out
    rc |= TestAddRmsNormF8(5120, 1, true, false);   // hoist_w=0, F32w, 3-out
    rc |= TestAddRmsNormF8(5120, 2, true, true);    // hoist_w=1, F32w, 4-out
    rc |= TestAddRmsNormF8(7680, 2, true, false);   // max F32w VLM hoist limit
    rc |= TestAddRmsNormF8(8192, 2, true, false);   // hoist_w=0 (exceeds F32w VLM hoist limit), 3-out
    rc |= TestAddRmsNormF8(12288, 1, false, false); // max RMSNorm dim (48 KiB VLM)
    rc |= TestAddRmsNormF8(2048, 2, false, false);  // hoist_w=1, BF16w, 3-out
    rc |= TestAddRmsNormF8(2048, 1, false, true);   // hoist_w=0, BF16w, 4-out
    rc |= TestAddRmsNormF8(192, 2, false, true, /*zeroRow0=*/true);
    rc |= TestAddRmsNormF8(192, 1, true, true);     // tail + hoist_w=0

    rc |= TestSandwichRmsNormF8(64, 2);
    rc |= TestSandwichRmsNormF8(2048, 2);           // hoist_w=1
    rc |= TestSandwichRmsNormF8(2048, 1);           // hoist_w=0
    rc |= TestSandwichRmsNormF8(7680, 2);           // max sandwich VLM hoist limit
    rc |= TestSandwichRmsNormF8(8192, 2);           // hoist_w=0 (exceeds sandwich hoist limit)
    rc |= TestSandwichRmsNormF8(192, 2);
    rc |= TestSandwichRmsNormF8(192, 1);

    rc |= TestSiluMulF8(64, 2);
    rc |= TestSiluMulF8(7680, 2);
    rc |= TestSiluMulF8(17408, 2);
    rc |= TestSiluMulF8(192, 2);
    rc |= TestSiluMulF8(320, 2);
    rc |= TestSiluMulF8(28672, 1);  // max VLM capacity (56 KiB)

    rc |= TestGdnPrepDecode(16, 48, 2, 0, 0, 1);
    rc |= TestGdnPrepDecode(4, 12, 2, 1, 1, 2);

    rc |= TestGdnGatedRmsNormF8(1, 2);    // min HV
    rc |= TestGdnGatedRmsNormF8(48, 2);
    rc |= TestGdnGatedRmsNormF8(5, 2);    // odd HV tail
    rc |= TestGdnGatedRmsNormF8(128, 1);  // max HV

    rc |= TestLfm3ShortConvDecode(2048, 3, 7, 6, 0);
    rc |= TestLfm3ShortConvDecode(256, 2, 4, 6, 1);
    rc |= TestLfm3ShortConvDecode(64, 2, 2, 1, 1);  // min K=2, min D=64
    return rc;
}
