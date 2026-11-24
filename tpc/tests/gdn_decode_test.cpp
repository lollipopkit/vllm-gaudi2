// SPDX-License-Identifier: Apache-2.0
// Simulator (or device, TPC_RUNNER=1) test of gdn_decode_f32_gaudi2 against a double-precision reference.
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "gdn_decode_gaudi2.hpp"
#include "habana/tensor.h"
#include "tpc_test_core_api.h"

extern "C" tpc_lib_api::GlueCodeReturn InstantiateTpcKernel(tpc_lib_api::HabanaKernelParams* params,
                                                            tpc_lib_api::HabanaKernelInstantiation* instance);

namespace
{
constexpr int D = GdnDecodeGaudi2::kDim;

template <int N>
void Describe(tpc_lib_api::Tensor* t, const test::Tensor<float, N>& src)
{
    t->geometry.dims = N;
    t->geometry.dataType = tpc_lib_api::DATA_F32;
    for (int i = 0; i < N; i++)
        t->geometry.maxSizes[i] = t->geometry.minSizes[i] = src.Size(i);
}

int Run(int heads, int batch)
{
    const int keys = heads / GdnDecodeGaudi2::kHeadsPerKey;
    const int offset = 1, slots = batch + 2;  // the batch sits inside a larger pool, as in serving
    const uint64_t state_sz[] = {D, D, (uint64_t)heads, (uint64_t)slots};
    const uint64_t qk_sz[] = {D, 2 * (uint64_t)keys, (uint64_t)batch};
    const uint64_t rows_sz[] = {D, (uint64_t)heads, (uint64_t)batch};
    const uint64_t head_sz[] = {(uint64_t)heads, (uint64_t)batch};
    const uint64_t key_sz[] = {(uint64_t)keys, (uint64_t)batch};
    const float scale = 1.f / std::sqrt((float)D);
    test::Tensor<float, 4> state(state_sz);
    test::Tensor<float, 3> qk(qk_sz), value(rows_sz), out(rows_sz);
    test::Tensor<float, 2> decay(head_sz), beta(head_sz), kqdot(key_sz);

    std::mt19937 rng(1234);
    std::uniform_real_distribution<float> u(-1.f, 1.f), pos(0.f, 1.f);
    for (float* p = state.Data(); p != state.Data() + (size_t)D * D * heads * slots; ++p)
        *p = 0.1f * u(rng);
    for (float* p = value.Data(); p != value.Data() + (size_t)D * heads * batch; ++p)
        *p = u(rng);
    for (int i = 0; i < heads * batch; i++)
    {
        decay.Data()[i] = 0.5f + 0.5f * pos(rng);
        beta.Data()[i] = pos(rng);
    }
    // L2-normalized, unscaled q (heads 0..keys-1) and k (heads keys..2 keys-1).
    for (int b = 0; b < batch; b++)
        for (int hh = 0; hh < 2 * keys; hh++)
        {
            float* v = qk.Data() + ((size_t)b * 2 * keys + hh) * D;
            double n = 0;
            for (int i = 0; i < D; i++)
            {
                v[i] = u(rng);
                n += (double)v[i] * v[i];
            }
            for (int i = 0; i < D; i++)
                v[i] = (float)(v[i] / std::sqrt(n));
        }
    auto q_of = [&](int b, int kh) { return qk.Data() + ((size_t)b * 2 * keys + kh) * D; };
    auto k_of = [&](int b, int kh) { return qk.Data() + ((size_t)b * 2 * keys + keys + kh) * D; };
    for (int b = 0; b < batch; b++)
        for (int kh = 0; kh < keys; kh++)
        {
            float dot = 0;
            for (int i = 0; i < D; i++)
                dot += k_of(b, kh)[i] * q_of(b, kh)[i];
            kqdot.Data()[b * keys + kh] = dot;
        }

    tpc_lib_api::HabanaKernelParams in = {};
    tpc_lib_api::HabanaKernelInstantiation inst = {};
    std::vector<tpc_lib_api::Tensor> ins(6), outs(1);
    std::vector<tpc_lib_api::TensorAccessPattern> in_ap(6), out_ap(1);
    std::vector<char> elf(1 << 20);
    in.deviceId = tpc_lib_api::DEVICE_ID_GAUDI2;
    strcpy(in.guid.name, GdnDecodeGaudi2::kGuid);
    in.inputTensors = ins.data();
    in.inputTensorNr = 6;
    in.outputTensors = outs.data();
    in.outputTensorNr = 1;
    Describe(&ins[0], state);
    Describe(&ins[1], qk);
    Describe(&ins[2], value);
    Describe(&ins[3], decay);
    Describe(&ins[4], beta);
    Describe(&ins[5], kqdot);
    Describe(&outs[0], out);
    GdnDecodeGaudi2::Params params = {offset, scale};
    in.nodeParams.nodeParams = &params;
    in.nodeParams.nodeParamsSize = sizeof(params);
    inst.inputTensorAccessPattern = in_ap.data();
    inst.outputTensorAccessPattern = out_ap.data();
    inst.kernel.kernelElf = elf.data();
    inst.kernel.elfSize = elf.size();
    if (InstantiateTpcKernel(&in, &inst) != tpc_lib_api::GLUE_SUCCESS)
    {
        printf("glue code failed\n");
        return 1;
    }
    std::vector<TensorDesc2> descs = {state.GetTensorDescriptor(), qk.GetTensorDescriptor(),
                                      value.GetTensorDescriptor(), decay.GetTensorDescriptor(),
                                      beta.GetTensorDescriptor(), kqdot.GetTensorDescriptor(),
                                      out.GetTensorDescriptor()};
    const std::vector<float> initial(state.Data(), state.Data() + (size_t)D * D * heads * slots);
    VPEStats stats;
    const unsigned cycles = tpc_tests::RunSimulation(in, inst, descs, stats);
    const float* result = state.Data();  // updated in place

    double max_err_s = 0, max_err_o = 0, max_ref_s = 0, max_ref_o = 0;
    for (int b = 0; b < batch; b++)
        for (int h = 0; h < heads; h++)
        {
            const float* k = k_of(b, h / GdnDecodeGaudi2::kHeadsPerKey);
            const float* q = q_of(b, h / GdnDecodeGaudi2::kHeadsPerKey);
            const double dk = decay.Data()[b * heads + h], bt = beta.Data()[b * heads + h];
            const size_t base = ((size_t)(offset + b) * heads + h) * D * D;  // KV-major: S[i][v] at base + i * D + v
            for (int v = 0; v < D; v++)
            {
                double sk = 0;
                for (int i = 0; i < D; i++)
                    sk += (double)initial[base + (size_t)i * D + v] * k[i];
                const double delta = (value.Data()[((size_t)b * heads + h) * D + v] - dk * sk) * bt;
                double o = 0;
                for (int i = 0; i < D; i++)
                {
                    const size_t at = base + (size_t)i * D + v;
                    const double s_new = dk * initial[at] + delta * k[i];
                    o += s_new * q[i] * scale;
                    max_err_s = std::fmax(max_err_s, std::fabs(s_new - result[at]));
                    max_ref_s = std::fmax(max_ref_s, std::fabs(s_new));
                }
                max_err_o = std::fmax(max_err_o, std::fabs(o - out.Data()[((size_t)b * heads + h) * D + v]));
                max_ref_o = std::fmax(max_ref_o, std::fabs(o));
            }
        }
    const double rel_s = max_err_s / max_ref_s, rel_o = max_err_o / max_ref_o;
    const bool ok = rel_s < 1e-5 && rel_o < 1e-4;
    // Slots outside the batch must be untouched.
    for (int sl : {0, slots - 1})
        for (size_t i = 0; i < (size_t)heads * D * D; i++)
            if (state.Data()[sl * heads * D * D + i] != initial[sl * heads * D * D + i])
            {
                printf("slot %d modified\n", sl);
                return 1;
            }
    printf("heads=%d batch=%d cycles=%u state rel err %.2e out rel err %.2e -> %s\n", heads, batch, cycles, rel_s,
           rel_o, ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
}  // namespace

int main(int argc, char** argv)
{
    const int heads = argc > 1 ? atoi(argv[1]) : 6;
    const int batch = argc > 2 ? atoi(argv[2]) : 2;
    return Run(heads, batch);
}
