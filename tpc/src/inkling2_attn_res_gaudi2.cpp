// SPDX-License-Identifier: Apache-2.0
#include "inkling2_attn_res_gaudi2.hpp"

#include <cstring>
#include <initializer_list>

#include "inkling2_glue_common.hpp"

extern unsigned char _binary___inkling2_attn_res_norm_bf16_gaudi2_o_start;
extern unsigned char _binary___inkling2_attn_res_norm_bf16_gaudi2_o_end;
extern unsigned char _binary___inkling2_attn_res_norm_f8_gaudi2_o_start;
extern unsigned char _binary___inkling2_attn_res_norm_f8_gaudi2_o_end;
extern unsigned char _binary___inkling2_attn_res_f32_gaudi2_o_start;
extern unsigned char _binary___inkling2_attn_res_f32_gaudi2_o_end;

using inkling2_glue::CopyElf;
using inkling2_glue::Is;
using inkling2_glue::IsPosFinite;
using inkling2_glue::Map;

tpc_lib_api::GlueCodeReturn Inkling2AttnResNormBf16Gaudi2::GetKernelName(
    char kernelName[tpc_lib_api::MAX_NODE_NAME])
{
    strcpy(kernelName, kGuid);
    return tpc_lib_api::GLUE_SUCCESS;
}

tpc_lib_api::GlueCodeReturn Inkling2AttnResNormBf16Gaudi2::GetGcDefinitions(
    tpc_lib_api::HabanaKernelParams* in_defs, tpc_lib_api::HabanaKernelInstantiation* out_defs)
{
    enum Input { STACK = 0, LAST, ADDEND, QUERIES, NORM_W, NUM_INPUTS };
    enum Output { OUT = 0, NUM_OUTPUTS };

    if (in_defs->inputTensorNr != NUM_INPUTS)
    {
        in_defs->inputTensorNr = NUM_INPUTS;
        return tpc_lib_api::GLUE_INCOMPATIBLE_INPUT_COUNT;
    }
    if (in_defs->outputTensorNr != NUM_OUTPUTS)
    {
        in_defs->outputTensorNr = NUM_OUTPUTS;
        return tpc_lib_api::GLUE_INCOMPATIBLE_OUTPUT_COUNT;
    }
    if (in_defs->nodeParams.nodeParams == nullptr || in_defs->nodeParams.nodeParamsSize < sizeof(Params))
        return tpc_lib_api::GLUE_INCOMPATIBLE_INPUT_SIZE;

    const auto* params = static_cast<const Params*>(in_defs->nodeParams.nodeParams);
    const uint64_t D = in_defs->inputTensors[LAST].geometry.maxSizes[0];
    const uint64_t T = in_defs->inputTensors[LAST].geometry.maxSizes[1];
    const uint64_t stackDim = in_defs->inputTensors[STACK].geometry.maxSizes[2];
    const uint64_t Q = in_defs->inputTensors[QUERIES].geometry.maxSizes[1];

    if (D == 0 || (D % 64) != 0 || D > 40 * 64 || T == 0 || Q == 0 || Q > 4 ||
        params->nStack < 0 || params->nStack >= 16 || static_cast<uint64_t>(params->nStack) > stackDim ||
        !IsPosFinite(params->eps) || !IsPosFinite(params->normEps))
        return tpc_lib_api::GLUE_INCOMPATIBLE_INPUT_SIZE;

    const uint64_t stackSz[] = {D, T, stackDim};
    const uint64_t lastSz[] = {D, T};
    const uint64_t qSz[] = {D, Q};
    const uint64_t outSz[] = {D, T, Q};

    if (!Is(in_defs->inputTensors[STACK], tpc_lib_api::DATA_BF16, 3, stackSz) ||
        !Is(in_defs->inputTensors[LAST], tpc_lib_api::DATA_BF16, 2, lastSz) ||
        !Is(in_defs->inputTensors[ADDEND], tpc_lib_api::DATA_BF16, 2, lastSz) ||
        !Is(in_defs->inputTensors[QUERIES], tpc_lib_api::DATA_F32, 2, qSz) ||
        !Is(in_defs->inputTensors[NORM_W], tpc_lib_api::DATA_BF16, 2, qSz) ||
        !Is(in_defs->outputTensors[OUT], tpc_lib_api::DATA_BF16, 3, outSz))
        return tpc_lib_api::GLUE_INCOMPATIBLE_INPUT_SIZE;

    out_defs->indexSpaceRank = 2;
    out_defs->indexSpaceGeometry[0] = T;
    out_defs->indexSpaceGeometry[1] = Q;

    for (int i = 0; i < NUM_INPUTS; i++)
        out_defs->inputTensorAccessPattern[i].allRequired = true;

    auto& apOut = out_defs->outputTensorAccessPattern[OUT];
    Map(apOut.mapping[0], 0, 0, 0, static_cast<float>(D - 1));
    Map(apOut.mapping[1], 0, 1, 0, 0);
    Map(apOut.mapping[2], 1, 1, 0, 0);

    const float invD = 1.0f / static_cast<float>(D);
    out_defs->kernel.paramsNr = 6;
    memcpy(&out_defs->kernel.scalarParams[0], &params->nStack, sizeof(int));
    memcpy(&out_defs->kernel.scalarParams[1], &params->hasAddend, sizeof(int));
    memcpy(&out_defs->kernel.scalarParams[2], &params->normalizeValues, sizeof(int));
    memcpy(&out_defs->kernel.scalarParams[3], &params->eps, sizeof(float));
    memcpy(&out_defs->kernel.scalarParams[4], &params->normEps, sizeof(float));
    memcpy(&out_defs->kernel.scalarParams[5], &invD, sizeof(float));

    return CopyElf(out_defs, &_binary___inkling2_attn_res_norm_bf16_gaudi2_o_start,
                   &_binary___inkling2_attn_res_norm_bf16_gaudi2_o_end);
}

tpc_lib_api::GlueCodeReturn Inkling2AttnResNormF8Gaudi2::GetKernelName(
    char kernelName[tpc_lib_api::MAX_NODE_NAME])
{
    strcpy(kernelName, kGuid);
    return tpc_lib_api::GLUE_SUCCESS;
}

tpc_lib_api::GlueCodeReturn Inkling2AttnResNormF8Gaudi2::GetGcDefinitions(
    tpc_lib_api::HabanaKernelParams* in_defs, tpc_lib_api::HabanaKernelInstantiation* out_defs)
{
    enum Input { STACK = 0, LAST, ADDEND, QUERIES, NORM_W, NUM_INPUTS };
    enum Output { OUT_BF16 = 0, OUT_F8, OUT_SCALE, NUM_OUTPUTS };

    if (in_defs->inputTensorNr != NUM_INPUTS)
    {
        in_defs->inputTensorNr = NUM_INPUTS;
        return tpc_lib_api::GLUE_INCOMPATIBLE_INPUT_COUNT;
    }
    if (in_defs->outputTensorNr != NUM_OUTPUTS)
    {
        in_defs->outputTensorNr = NUM_OUTPUTS;
        return tpc_lib_api::GLUE_INCOMPATIBLE_OUTPUT_COUNT;
    }
    if (in_defs->nodeParams.nodeParams == nullptr || in_defs->nodeParams.nodeParamsSize < sizeof(Params))
        return tpc_lib_api::GLUE_INCOMPATIBLE_INPUT_SIZE;

    const auto* params = static_cast<const Params*>(in_defs->nodeParams.nodeParams);
    const uint64_t D = in_defs->inputTensors[LAST].geometry.maxSizes[0];
    const uint64_t T = in_defs->inputTensors[LAST].geometry.maxSizes[1];
    const uint64_t stackDim = in_defs->inputTensors[STACK].geometry.maxSizes[2];
    const uint64_t Q = in_defs->inputTensors[QUERIES].geometry.maxSizes[1];

    if (D == 0 || (D % 64) != 0 || D > 40 * 64 || T == 0 || Q == 0 || Q > 4 ||
        params->nStack < 0 || params->nStack >= 16 || static_cast<uint64_t>(params->nStack) > stackDim ||
        !IsPosFinite(params->eps) || !IsPosFinite(params->normEps) || !IsPosFinite(params->fp8Max))
        return tpc_lib_api::GLUE_INCOMPATIBLE_INPUT_SIZE;

    const uint64_t stackSz[] = {D, T, stackDim};
    const uint64_t lastSz[] = {D, T};
    const uint64_t qSz[] = {D, Q};
    const uint64_t outSz[] = {D, T, Q};
    const uint64_t scSz[] = {T, Q};

    if (!Is(in_defs->inputTensors[STACK], tpc_lib_api::DATA_BF16, 3, stackSz) ||
        !Is(in_defs->inputTensors[LAST], tpc_lib_api::DATA_BF16, 2, lastSz) ||
        !Is(in_defs->inputTensors[ADDEND], tpc_lib_api::DATA_BF16, 2, lastSz) ||
        !Is(in_defs->inputTensors[QUERIES], tpc_lib_api::DATA_F32, 2, qSz) ||
        !Is(in_defs->inputTensors[NORM_W], tpc_lib_api::DATA_BF16, 2, qSz) ||
        !Is(in_defs->outputTensors[OUT_BF16], tpc_lib_api::DATA_BF16, 3, outSz) ||
        !Is(in_defs->outputTensors[OUT_F8], tpc_lib_api::DATA_F8_143, 3, outSz) ||
        !Is(in_defs->outputTensors[OUT_SCALE], tpc_lib_api::DATA_F32, 2, scSz))
        return tpc_lib_api::GLUE_INCOMPATIBLE_INPUT_SIZE;

    out_defs->indexSpaceRank = 2;
    out_defs->indexSpaceGeometry[0] = T;
    out_defs->indexSpaceGeometry[1] = Q;

    for (int i = 0; i < NUM_INPUTS; i++)
        out_defs->inputTensorAccessPattern[i].allRequired = true;

    for (int o : {OUT_BF16, OUT_F8})
    {
        auto& ap = out_defs->outputTensorAccessPattern[o];
        Map(ap.mapping[0], 0, 0, 0, static_cast<float>(D - 1));
        Map(ap.mapping[1], 0, 1, 0, 0);
        Map(ap.mapping[2], 1, 1, 0, 0);
    }
    auto& apSc = out_defs->outputTensorAccessPattern[OUT_SCALE];
    Map(apSc.mapping[0], 0, 1, 0, 0);
    Map(apSc.mapping[1], 1, 1, 0, 0);

    const float invD = 1.0f / static_cast<float>(D);
    out_defs->kernel.paramsNr = 7;
    memcpy(&out_defs->kernel.scalarParams[0], &params->nStack, sizeof(int));
    memcpy(&out_defs->kernel.scalarParams[1], &params->hasAddend, sizeof(int));
    memcpy(&out_defs->kernel.scalarParams[2], &params->normalizeValues, sizeof(int));
    memcpy(&out_defs->kernel.scalarParams[3], &params->eps, sizeof(float));
    memcpy(&out_defs->kernel.scalarParams[4], &params->normEps, sizeof(float));
    memcpy(&out_defs->kernel.scalarParams[5], &invD, sizeof(float));
    memcpy(&out_defs->kernel.scalarParams[6], &params->fp8Max, sizeof(float));

    return CopyElf(out_defs, &_binary___inkling2_attn_res_norm_f8_gaudi2_o_start,
                   &_binary___inkling2_attn_res_norm_f8_gaudi2_o_end);
}

tpc_lib_api::GlueCodeReturn Inkling2AttnResF32Gaudi2::GetKernelName(
    char kernelName[tpc_lib_api::MAX_NODE_NAME])
{
    strcpy(kernelName, kGuid);
    return tpc_lib_api::GLUE_SUCCESS;
}

tpc_lib_api::GlueCodeReturn Inkling2AttnResF32Gaudi2::GetGcDefinitions(
    tpc_lib_api::HabanaKernelParams* in_defs, tpc_lib_api::HabanaKernelInstantiation* out_defs)
{
    enum Input { STACK = 0, LAST, ADDEND, QUERIES, NUM_INPUTS };
    enum Output { OUT = 0, NUM_OUTPUTS };

    if (in_defs->inputTensorNr != NUM_INPUTS)
    {
        in_defs->inputTensorNr = NUM_INPUTS;
        return tpc_lib_api::GLUE_INCOMPATIBLE_INPUT_COUNT;
    }
    if (in_defs->outputTensorNr != NUM_OUTPUTS)
    {
        in_defs->outputTensorNr = NUM_OUTPUTS;
        return tpc_lib_api::GLUE_INCOMPATIBLE_OUTPUT_COUNT;
    }
    if (in_defs->nodeParams.nodeParams == nullptr || in_defs->nodeParams.nodeParamsSize < sizeof(Params))
        return tpc_lib_api::GLUE_INCOMPATIBLE_INPUT_SIZE;

    const auto* params = static_cast<const Params*>(in_defs->nodeParams.nodeParams);
    const uint64_t D = in_defs->inputTensors[LAST].geometry.maxSizes[0];
    const uint64_t T = in_defs->inputTensors[LAST].geometry.maxSizes[1];
    const uint64_t stackDim = in_defs->inputTensors[STACK].geometry.maxSizes[2];
    const uint64_t Q = in_defs->inputTensors[QUERIES].geometry.maxSizes[1];

    if (D == 0 || (D % 64) != 0 || D > 40 * 64 || T == 0 || Q == 0 || Q > 4 ||
        params->nStack < 0 || params->nStack >= 16 || static_cast<uint64_t>(params->nStack) > stackDim ||
        !IsPosFinite(params->eps))
        return tpc_lib_api::GLUE_INCOMPATIBLE_INPUT_SIZE;

    const uint64_t stackSz[] = {D, T, stackDim};
    const uint64_t lastSz[] = {D, T};
    const uint64_t qSz[] = {D, Q};
    const uint64_t outSz[] = {D, T, Q};

    if (!Is(in_defs->inputTensors[STACK], tpc_lib_api::DATA_BF16, 3, stackSz) ||
        !Is(in_defs->inputTensors[LAST], tpc_lib_api::DATA_BF16, 2, lastSz) ||
        !Is(in_defs->inputTensors[ADDEND], tpc_lib_api::DATA_BF16, 2, lastSz) ||
        !Is(in_defs->inputTensors[QUERIES], tpc_lib_api::DATA_F32, 2, qSz) ||
        !Is(in_defs->outputTensors[OUT], tpc_lib_api::DATA_F32, 3, outSz))
        return tpc_lib_api::GLUE_INCOMPATIBLE_INPUT_SIZE;

    out_defs->indexSpaceRank = 2;
    out_defs->indexSpaceGeometry[0] = T;
    out_defs->indexSpaceGeometry[1] = Q;

    for (int i = 0; i < NUM_INPUTS; i++)
        out_defs->inputTensorAccessPattern[i].allRequired = true;

    auto& apOut = out_defs->outputTensorAccessPattern[OUT];
    Map(apOut.mapping[0], 0, 0, 0, static_cast<float>(D - 1));
    Map(apOut.mapping[1], 0, 1, 0, 0);
    Map(apOut.mapping[2], 1, 1, 0, 0);

    const float invD = 1.0f / static_cast<float>(D);
    out_defs->kernel.paramsNr = 5;
    memcpy(&out_defs->kernel.scalarParams[0], &params->nStack, sizeof(int));
    memcpy(&out_defs->kernel.scalarParams[1], &params->hasAddend, sizeof(int));
    memcpy(&out_defs->kernel.scalarParams[2], &params->normalizeValues, sizeof(int));
    memcpy(&out_defs->kernel.scalarParams[3], &params->eps, sizeof(float));
    memcpy(&out_defs->kernel.scalarParams[4], &invD, sizeof(float));

    return CopyElf(out_defs, &_binary___inkling2_attn_res_f32_gaudi2_o_start,
                   &_binary___inkling2_attn_res_f32_gaudi2_o_end);
}
