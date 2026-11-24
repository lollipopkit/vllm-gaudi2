// SPDX-License-Identifier: Apache-2.0
// Simulator test for inkling2_attn_res_norm_bf16_gaudi2 and inkling2_attn_res_f32_gaudi2
// against a double-precision reference (tests n_stack=0, addend, Q=1/3/4, normalize_values).
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

#include "habana/tensor.h"
#include "inkling2_attn_res_gaudi2.hpp"
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

int RunCase(int D, int T, int nStack, int Q, bool hasAddend, bool hasNorm, bool normalizeValues, bool outputF8 = false)
{
    const int stackDim = std::max(nStack, 1);
    const int N = nStack + 1;
    const float eps = 1e-6f;
    const float normEps = 1e-6f;
    const float fp8Max = 240.0f;

    std::mt19937 rng(100 + nStack * 7 + Q * 13 + (hasAddend ? 3 : 0) + (hasNorm ? 5 : 0) + (outputF8 ? 17 : 0));
    std::normal_distribution<float> nd(0.0f, 1.0f);

    const uint64_t stackSz[] = {(uint64_t)D, (uint64_t)T, (uint64_t)stackDim};
    const uint64_t lastSz[] = {(uint64_t)D, (uint64_t)T};
    const uint64_t qSz[] = {(uint64_t)D, (uint64_t)Q};
    const uint64_t outSz[] = {(uint64_t)D, (uint64_t)T, (uint64_t)Q};
    const uint64_t scSz[] = {(uint64_t)T, (uint64_t)Q};

    test::Tensor<uint16_t, 3> stack(stackSz);
    test::Tensor<uint16_t, 2> last(lastSz);
    test::Tensor<uint16_t, 2> addend(lastSz);
    test::Tensor<float, 2> queries(qSz);
    test::Tensor<uint16_t, 2> normW(qSz);
    test::Tensor<uint16_t, 3> outBf16(outSz);
    test::Tensor<uint8_t, 3> outF8(outSz);
    test::Tensor<float, 2> outSc(scSz);
    test::Tensor<float, 3> outF32(outSz);

    for (size_t i = 0; i < (size_t)D * T * stackDim; i++)
        stack.Data()[i] = floatToBf16(nd(rng));
    for (size_t i = 0; i < (size_t)D * T; i++)
    {
        last.Data()[i] = floatToBf16(nd(rng));
        addend.Data()[i] = floatToBf16(nd(rng));
    }
    const float qscale = 1.0f / std::sqrt((float)D);
    for (size_t i = 0; i < (size_t)D * Q; i++)
    {
        queries.Data()[i] = nd(rng) * qscale;
        normW.Data()[i] = floatToBf16(0.5f + 0.5f * std::fabs(nd(rng)));
    }

    tpc_lib_api::HabanaKernelParams in = {};
    tpc_lib_api::HabanaKernelInstantiation inst = {};
    std::vector<tpc_lib_api::Tensor> ins(hasNorm ? 5 : 4), outs(outputF8 ? 3 : 1);
    std::vector<tpc_lib_api::TensorAccessPattern> inAp(ins.size()), outAp(outs.size());
    std::vector<char> elf(1 << 20);
    in.deviceId = tpc_lib_api::DEVICE_ID_GAUDI2;
    strcpy(in.guid.name, outputF8 ? Inkling2AttnResNormF8Gaudi2::kGuid
                                  : (hasNorm ? Inkling2AttnResNormBf16Gaudi2::kGuid : Inkling2AttnResF32Gaudi2::kGuid));
    in.inputTensors = ins.data();
    in.inputTensorNr = ins.size();
    in.outputTensors = outs.data();
    in.outputTensorNr = outs.size();
    Describe(&ins[0], tpc_lib_api::DATA_BF16, stack);
    Describe(&ins[1], tpc_lib_api::DATA_BF16, last);
    Describe(&ins[2], tpc_lib_api::DATA_BF16, addend);
    Describe(&ins[3], tpc_lib_api::DATA_F32, queries);
    if (hasNorm)
    {
        Describe(&ins[4], tpc_lib_api::DATA_BF16, normW);
        Describe(&outs[0], tpc_lib_api::DATA_BF16, outBf16);
        if (outputF8)
        {
            Describe(&outs[1], tpc_lib_api::DATA_F8_143, outF8);
            Describe(&outs[2], tpc_lib_api::DATA_F32, outSc);
        }
    }
    else
    {
        Describe(&outs[0], tpc_lib_api::DATA_F32, outF32);
    }

    Inkling2AttnResNormBf16Gaudi2::Params pNorm = {nStack, hasAddend ? 1 : 0, normalizeValues ? 1 : 0, eps, normEps};
    Inkling2AttnResNormF8Gaudi2::Params pF8 = {nStack, hasAddend ? 1 : 0, normalizeValues ? 1 : 0, eps, normEps,
                                               fp8Max};
    Inkling2AttnResF32Gaudi2::Params pF32 = {nStack, hasAddend ? 1 : 0, normalizeValues ? 1 : 0, eps};
    in.nodeParams.nodeParams = outputF8 ? static_cast<void*>(&pF8)
                                        : (hasNorm ? static_cast<void*>(&pNorm) : static_cast<void*>(&pF32));
    in.nodeParams.nodeParamsSize = outputF8 ? sizeof(pF8) : (hasNorm ? sizeof(pNorm) : sizeof(pF32));
    inst.inputTensorAccessPattern = inAp.data();
    inst.outputTensorAccessPattern = outAp.data();
    inst.kernel.kernelElf = elf.data();
    inst.kernel.elfSize = elf.size();

    if (InstantiateTpcKernel(&in, &inst) != tpc_lib_api::GLUE_SUCCESS)
    {
        printf("glue failed (hasNorm=%d f8=%d)\n", (int)hasNorm, (int)outputF8);
        return 1;
    }

    std::vector<TensorDesc2> descs = {stack.GetTensorDescriptor(), last.GetTensorDescriptor(),
                                      addend.GetTensorDescriptor(), queries.GetTensorDescriptor()};
    if (hasNorm)
    {
        descs.push_back(normW.GetTensorDescriptor());
        descs.push_back(outBf16.GetTensorDescriptor());
        if (outputF8)
        {
            descs.push_back(outF8.GetTensorDescriptor());
            descs.push_back(outSc.GetTensorDescriptor());
        }
    }
    else
    {
        descs.push_back(outF32.GetTensorDescriptor());
    }
    VPEStats stats;
    const unsigned cyc = tpc_tests::RunSimulation(in, inst, descs, stats);

    // Double-precision reference
    double maxErr = 0.0, maxRef = 0.0, maxF8Err = 0.0, maxScErr = 0.0;
    for (int t = 0; t < T; t++)
    {
        std::vector<std::vector<double>> src(N, std::vector<double>(D));
        std::vector<double> invRms(N);
        for (int n = 0; n < N; n++)
        {
            double ssq = 0.0;
            for (int d = 0; d < D; d++)
            {
                double v = (n < nStack)
                               ? (double)bf16ToFloat(stack.Data()[((size_t)n * T + t) * D + d])
                               : (double)bf16ToFloat(last.Data()[(size_t)t * D + d]) +
                                     (hasAddend ? (double)bf16ToFloat(addend.Data()[(size_t)t * D + d]) : 0.0);
                src[n][d] = v;
                ssq += v * v;
            }
            invRms[n] = 1.0 / std::sqrt(ssq / D + eps);
        }
        for (int q = 0; q < Q; q++)
        {
            std::vector<double> logits(N);
            double m = -1e300;
            for (int n = 0; n < N; n++)
            {
                double dot = 0.0;
                for (int d = 0; d < D; d++)
                    dot += src[n][d] * invRms[n] * queries.Data()[(size_t)q * D + d];
                logits[n] = dot;
                m = std::max(m, dot);
            }
            double denom = 0.0;
            std::vector<double> probs(N);
            for (int n = 0; n < N; n++)
            {
                probs[n] = std::exp(logits[n] - m);
                denom += probs[n];
            }
            std::vector<double> mixed(D, 0.0);
            double msq = 0.0;
            for (int d = 0; d < D; d++)
            {
                double acc = 0.0;
                for (int n = 0; n < N; n++)
                {
                    const double val = normalizeValues ? (src[n][d] * invRms[n]) : src[n][d];
                    acc += (probs[n] / denom) * val;
                }
                mixed[d] = acc;
                msq += acc * acc;
            }
            const double rstd = 1.0 / std::sqrt(msq / D + normEps);
            double amax = 0.0;
            for (int d = 0; d < D; d++)
            {
                const double ref = hasNorm ? (mixed[d] * rstd * bf16ToFloat(normW.Data()[(size_t)q * D + d])) : mixed[d];
                amax = std::max(amax, std::fabs(ref));
                const double got = hasNorm
                                       ? (double)bf16ToFloat(outBf16.Data()[((size_t)q * T + t) * D + d])
                                       : (double)outF32.Data()[((size_t)q * T + t) * D + d];
                maxErr = std::max(maxErr, std::fabs(got - ref));
                maxRef = std::max(maxRef, std::fabs(ref));
            }
            if (outputF8)
            {
                const double refSc = std::max(amax / fp8Max, 1e-12);
                const double gotSc = outSc.Data()[(size_t)q * T + t];
                maxScErr = std::max(maxScErr, std::fabs(gotSc - refSc) / refSc);
                for (int d = 0; d < D; d++)
                {
                    const double ref = mixed[d] * rstd * bf16ToFloat(normW.Data()[(size_t)q * D + d]);
                    const double gotF8 = F8ToF32(outF8.Data()[((size_t)q * T + t) * D + d]) * gotSc;
                    maxF8Err = std::max(maxF8Err, std::fabs(gotF8 - ref));
                }
            }
        }
    }
    const double rel = maxErr / std::max(maxRef, 1e-12);
    const double relF8 = maxF8Err / std::max(maxRef, 1e-12);
    const double tol = hasNorm ? 0.015 : 0.002;
    const bool ok = (rel < tol) && (!outputF8 || (relF8 < 0.07 && maxScErr < 1e-3));
    printf("D=%d T=%d nStack=%d Q=%d add=%d norm=%d f8=%d nv=%d | cyc=%u | rel=%.6f f8=%.4f sc=%.2e -> %s\n", D, T,
           nStack, Q, (int)hasAddend, (int)hasNorm, (int)outputF8, (int)normalizeValues, cyc, rel, relF8, maxScErr,
           ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
}  // namespace

int main()
{
    int rc = 0;
    rc |= RunCase(256, 2, 0, 1, false, true, true, false);
    rc |= RunCase(256, 2, 5, 3, true, true, true, false);
    rc |= RunCase(256, 2, 5, 3, true, true, true, true);
    rc |= RunCase(256, 2, 7, 4, false, false, true, false);
    rc |= RunCase(256, 2, 3, 1, true, false, false, false);
    // D & 64 tail path (D = 192 = 3 * 64) across all three variants.
    rc |= RunCase(192, 2, 5, 3, true, true, true, false);
    rc |= RunCase(192, 2, 5, 3, true, true, true, true);
    rc |= RunCase(192, 2, 3, 2, true, false, true, false);
    return rc;
}
