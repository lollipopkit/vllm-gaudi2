// SPDX-License-Identifier: Apache-2.0
#include "inkling2_sconv_norm_gaudi2.hpp"

#include <cmath>
#include <cstring>
#include <initializer_list>

#include "inkling2_glue_common.hpp"

extern unsigned char _binary___inkling2_sconv_static_decode_gaudi2_o_start;
extern unsigned char _binary___inkling2_sconv_static_decode_gaudi2_o_end;
extern unsigned char _binary___inkling2_qkvr_prep_decode_gaudi2_o_start;
extern unsigned char _binary___inkling2_qkvr_prep_decode_gaudi2_o_end;
extern unsigned char _binary___inkling2_qkvr_prep_decode_dyn_gaudi2_o_start;
extern unsigned char _binary___inkling2_qkvr_prep_decode_dyn_gaudi2_o_end;
extern unsigned char _binary___inkling2_pair_rms_norm_gaudi2_o_start;
extern unsigned char _binary___inkling2_pair_rms_norm_gaudi2_o_end;
extern unsigned char _binary___inkling2_blend_rms_norm_gaudi2_o_start;
extern unsigned char _binary___inkling2_blend_rms_norm_gaudi2_o_end;
extern unsigned char _binary___inkling2_blend_rms_norm_f8_gaudi2_o_start;
extern unsigned char _binary___inkling2_blend_rms_norm_f8_gaudi2_o_end;

using inkling2_glue::CopyElf;
using inkling2_glue::Is;
using inkling2_glue::IsPosFinite;
using inkling2_glue::Map;

tpc_lib_api::GlueCodeReturn Inkling2SconvStaticDecodeGaudi2::GetKernelName(
    char kernelName[tpc_lib_api::MAX_NODE_NAME])
{
    strcpy(kernelName, kGuid);
    return tpc_lib_api::GLUE_SUCCESS;
}

tpc_lib_api::GlueCodeReturn Inkling2SconvStaticDecodeGaudi2::GetGcDefinitions(
    tpc_lib_api::HabanaKernelParams* in_defs, tpc_lib_api::HabanaKernelInstantiation* out_defs)
{
    enum Input { X = 0, CONV_STATE, LOAD_SLOTS, WEIGHT_T, NUM_INPUTS };
    enum Output { Y = 0, NEW_ST, NUM_OUTPUTS };

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
    const uint64_t C = in_defs->inputTensors[X].geometry.maxSizes[0];
    const uint64_t B = in_defs->inputTensors[X].geometry.maxSizes[1];
    const uint64_t totalDim = in_defs->inputTensors[CONV_STATE].geometry.maxSizes[0];
    const uint64_t numSlots = in_defs->inputTensors[CONV_STATE].geometry.maxSizes[2];
    if (C == 0 || (C % 64) != 0 || B == 0 || numSlots == 0 ||
        params->streamOffset < 0 || static_cast<uint64_t>(params->streamOffset) + C > totalDim ||
        !std::isfinite(params->outScale))
        return tpc_lib_api::GLUE_INCOMPATIBLE_INPUT_SIZE;

    const uint64_t xSz[] = {C, B};
    const uint64_t csSz[] = {totalDim, 3, numSlots};
    const uint64_t lsSz[] = {B};
    const uint64_t stSz[] = {C, 3, B};
    const uint64_t wSz[] = {C, 4};

    if (!Is(in_defs->inputTensors[X], tpc_lib_api::DATA_BF16, 2, xSz) ||
        !Is(in_defs->inputTensors[CONV_STATE], tpc_lib_api::DATA_BF16, 3, csSz) ||
        !Is(in_defs->inputTensors[LOAD_SLOTS], tpc_lib_api::DATA_I32, 1, lsSz) ||
        !Is(in_defs->inputTensors[WEIGHT_T], tpc_lib_api::DATA_BF16, 2, wSz) ||
        !Is(in_defs->outputTensors[Y], tpc_lib_api::DATA_BF16, 2, xSz) ||
        !Is(in_defs->outputTensors[NEW_ST], tpc_lib_api::DATA_BF16, 3, stSz))
        return tpc_lib_api::GLUE_INCOMPATIBLE_INPUT_SIZE;

    out_defs->indexSpaceRank = 2;
    out_defs->indexSpaceGeometry[0] = C / 64;
    out_defs->indexSpaceGeometry[1] = B;

    for (int i = 0; i < NUM_INPUTS; i++)
        out_defs->inputTensorAccessPattern[i].allRequired = true;

    auto& apY = out_defs->outputTensorAccessPattern[Y];
    Map(apY.mapping[0], 0, 64, 0, 63);
    Map(apY.mapping[1], 1, 1, 0, 0);

    auto& apNs = out_defs->outputTensorAccessPattern[NEW_ST];
    Map(apNs.mapping[0], 0, 64, 0, 63);
    Map(apNs.mapping[1], 0, 0, 0, 2);
    Map(apNs.mapping[2], 1, 1, 0, 0);

    out_defs->kernel.paramsNr = 2;
    memcpy(&out_defs->kernel.scalarParams[0], &params->outScale, sizeof(float));
    memcpy(&out_defs->kernel.scalarParams[1], &params->streamOffset, sizeof(int));

    return CopyElf(out_defs, &_binary___inkling2_sconv_static_decode_gaudi2_o_start,
                   &_binary___inkling2_sconv_static_decode_gaudi2_o_end);
}

tpc_lib_api::GlueCodeReturn Inkling2QkvrPrepDecodeGaudi2::GetKernelName(
    char kernelName[tpc_lib_api::MAX_NODE_NAME])
{
    strcpy(kernelName, kGuid);
    return tpc_lib_api::GLUE_SUCCESS;
}

tpc_lib_api::GlueCodeReturn Inkling2QkvrPrepDecodeGaudi2::GetGcDefinitions(
    tpc_lib_api::HabanaKernelParams* in_defs, tpc_lib_api::HabanaKernelInstantiation* out_defs)
{
    enum Input { PROJ_Q = 0, Q_SCALE, R_SCALE, TAU, KV_SRC, CONV_STATE, LOAD_SLOTS, KVW_T, KV_DYN, NUM_INPUTS };
    enum Output { Q_OUT = 0, R_OUT, K_OUT, V_OUT, NEW_KV, NUM_OUTPUTS };

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
    const uint64_t qCols = in_defs->inputTensors[PROJ_Q].geometry.maxSizes[0];
    const uint64_t B = in_defs->inputTensors[PROJ_Q].geometry.maxSizes[1];
    const uint64_t D = in_defs->inputTensors[Q_SCALE].geometry.maxSizes[0];
    const uint64_t H = in_defs->inputTensors[Q_SCALE].geometry.maxSizes[1];
    const uint64_t dRel = in_defs->inputTensors[R_SCALE].geometry.maxSizes[0];
    const uint64_t kvCols = in_defs->inputTensors[KV_SRC].geometry.maxSizes[0];
    const uint64_t kvStreams = in_defs->inputTensors[KV_SRC].geometry.maxSizes[2];
    const uint64_t Hkv = in_defs->inputTensors[KVW_T].geometry.maxSizes[1];
    const uint64_t totalDim = in_defs->inputTensors[CONV_STATE].geometry.maxSizes[0];
    const uint64_t numSlots = in_defs->inputTensors[CONV_STATE].geometry.maxSizes[2];
    const uint64_t kvSize = Hkv * D;
    if (D != 128 || H == 0 || B == 0 || dRel == 0 || dRel > 64 || Hkv == 0 || Hkv > H ||
        qCols < H * (D + dRel) || (kvStreams != 1 && kvStreams != 2) || params->kvColOffset < 0 ||
        static_cast<uint64_t>(params->kvColOffset) + (kvStreams == 1 ? 2 * kvSize : kvSize) > kvCols ||
        numSlots == 0 || 2 * kvSize > totalDim ||
        !std::isfinite(params->convScale) || !IsPosFinite(params->qkClip) || !IsPosFinite(params->relClip) ||
        !IsPosFinite(params->clipEps) || !IsPosFinite(params->rmsEps))
        return tpc_lib_api::GLUE_INCOMPATIBLE_INPUT_SIZE;

    const uint64_t pqSz[] = {qCols, B};
    const uint64_t qSz[] = {D, H, B};
    const uint64_t rSz[] = {dRel, H, B};
    const uint64_t qsSz[] = {D, H};
    const uint64_t rsSz[] = {dRel, H};
    const uint64_t tauSz[] = {B};
    const uint64_t kvSrcSz[] = {kvCols, B, kvStreams};
    const uint64_t csSz[] = {totalDim, 3, numSlots};
    const uint64_t lsSz[] = {B};
    const uint64_t kvwSz[] = {D, Hkv, 2, 4};
    const uint64_t kvDynSz[] = {4, Hkv, B, 2};
    const uint64_t kvOutSz[] = {D, Hkv, B};
    const uint64_t newKvSz[] = {D, Hkv, 2, 3, B};

    if (!Is(in_defs->inputTensors[PROJ_Q], tpc_lib_api::DATA_BF16, 2, pqSz) ||
        !Is(in_defs->inputTensors[Q_SCALE], tpc_lib_api::DATA_BF16, 2, qsSz) ||
        !Is(in_defs->inputTensors[R_SCALE], tpc_lib_api::DATA_BF16, 2, rsSz) ||
        !Is(in_defs->inputTensors[TAU], tpc_lib_api::DATA_F32, 1, tauSz) ||
        !Is(in_defs->inputTensors[KV_SRC], tpc_lib_api::DATA_BF16, 3, kvSrcSz) ||
        !Is(in_defs->inputTensors[CONV_STATE], tpc_lib_api::DATA_BF16, 3, csSz) ||
        !Is(in_defs->inputTensors[LOAD_SLOTS], tpc_lib_api::DATA_I32, 1, lsSz) ||
        !Is(in_defs->inputTensors[KVW_T], tpc_lib_api::DATA_BF16, 4, kvwSz) ||
        (params->isDynamic ? !Is(in_defs->inputTensors[KV_DYN], tpc_lib_api::DATA_BF16, 4, kvDynSz)
                           : in_defs->inputTensors[KV_DYN].geometry.dataType != tpc_lib_api::DATA_BF16) ||
        !Is(in_defs->outputTensors[Q_OUT], tpc_lib_api::DATA_BF16, 3, qSz) ||
        !Is(in_defs->outputTensors[R_OUT], tpc_lib_api::DATA_BF16, 3, rSz) ||
        !Is(in_defs->outputTensors[K_OUT], tpc_lib_api::DATA_BF16, 3, kvOutSz) ||
        !Is(in_defs->outputTensors[V_OUT], tpc_lib_api::DATA_BF16, 3, kvOutSz) ||
        !Is(in_defs->outputTensors[NEW_KV], tpc_lib_api::DATA_BF16, 5, newKvSz))
        return tpc_lib_api::GLUE_INCOMPATIBLE_INPUT_SIZE;

    out_defs->indexSpaceRank = 2;
    out_defs->indexSpaceGeometry[0] = H;
    out_defs->indexSpaceGeometry[1] = B;

    for (int i = 0; i < NUM_INPUTS; i++)
        out_defs->inputTensorAccessPattern[i].allRequired = true;

    auto& apQo = out_defs->outputTensorAccessPattern[Q_OUT];
    Map(apQo.mapping[0], 0, 0, 0, 127);
    Map(apQo.mapping[1], 0, 1, 0, 0);
    Map(apQo.mapping[2], 1, 1, 0, 0);

    auto& apRo = out_defs->outputTensorAccessPattern[R_OUT];
    Map(apRo.mapping[0], 0, 0, 0, static_cast<float>(dRel - 1));
    Map(apRo.mapping[1], 0, 1, 0, 0);
    Map(apRo.mapping[2], 1, 1, 0, 0);

    auto& apKo = out_defs->outputTensorAccessPattern[K_OUT];
    Map(apKo.mapping[0], 0, 0, 0, 127);
    Map(apKo.mapping[1], 0, 1, 0, 0);
    Map(apKo.mapping[2], 1, 1, 0, 0);

    auto& apVo = out_defs->outputTensorAccessPattern[V_OUT];
    Map(apVo.mapping[0], 0, 0, 0, 127);
    const float vShift = (H >= 2 * Hkv) ? -static_cast<float>(Hkv) : 0.0f;
    Map(apVo.mapping[1], 0, 1, vShift, vShift);
    Map(apVo.mapping[2], 1, 1, 0, 0);

    auto& apNkv = out_defs->outputTensorAccessPattern[NEW_KV];
    Map(apNkv.mapping[0], 0, 0, 0, 127);
    Map(apNkv.mapping[1], 0, 1, vShift, 0);
    Map(apNkv.mapping[2], 0, 0, 0, 1);
    Map(apNkv.mapping[3], 0, 0, 0, 2);
    Map(apNkv.mapping[4], 1, 1, 0, 0);

    out_defs->kernel.paramsNr = 7;
    memcpy(&out_defs->kernel.scalarParams[0], &params->kvColOffset, sizeof(int));
    memcpy(&out_defs->kernel.scalarParams[1], &params->hasTau, sizeof(int));
    memcpy(&out_defs->kernel.scalarParams[2], &params->convScale, sizeof(float));
    memcpy(&out_defs->kernel.scalarParams[3], &params->qkClip, sizeof(float));
    memcpy(&out_defs->kernel.scalarParams[4], &params->relClip, sizeof(float));
    memcpy(&out_defs->kernel.scalarParams[5], &params->clipEps, sizeof(float));
    memcpy(&out_defs->kernel.scalarParams[6], &params->rmsEps, sizeof(float));

    if (params->isDynamic)
        return CopyElf(out_defs, &_binary___inkling2_qkvr_prep_decode_dyn_gaudi2_o_start,
                       &_binary___inkling2_qkvr_prep_decode_dyn_gaudi2_o_end);
    return CopyElf(out_defs, &_binary___inkling2_qkvr_prep_decode_gaudi2_o_start,
                   &_binary___inkling2_qkvr_prep_decode_gaudi2_o_end);
}

tpc_lib_api::GlueCodeReturn Inkling2PairRmsNormGaudi2::GetKernelName(char kernelName[tpc_lib_api::MAX_NODE_NAME])
{
    strcpy(kernelName, kGuid);
    return tpc_lib_api::GLUE_SUCCESS;
}

tpc_lib_api::GlueCodeReturn Inkling2PairRmsNormGaudi2::GetGcDefinitions(
    tpc_lib_api::HabanaKernelParams* in_defs, tpc_lib_api::HabanaKernelInstantiation* out_defs)
{
    enum Input { DEST = 0, MESSAGE, NUM_INPUTS };
    enum Output { PAIR = 0, NUM_OUTPUTS };

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
    const uint64_t D = in_defs->inputTensors[DEST].geometry.maxSizes[0];
    const uint64_t T = in_defs->inputTensors[DEST].geometry.maxSizes[1];
    const uint64_t S = in_defs->inputTensors[DEST].geometry.maxSizes[2];
    if (D == 0 || (D % 64) != 0 || T == 0 || S == 0 || !IsPosFinite(params->eps))
        return tpc_lib_api::GLUE_INCOMPATIBLE_INPUT_SIZE;

    const uint64_t inSz[] = {D, T, S};
    const uint64_t outSz[] = {2 * D, T, S};
    if (!Is(in_defs->inputTensors[DEST], tpc_lib_api::DATA_F32, 3, inSz) ||
        !Is(in_defs->inputTensors[MESSAGE], tpc_lib_api::DATA_F32, 3, inSz) ||
        !Is(in_defs->outputTensors[PAIR], tpc_lib_api::DATA_BF16, 3, outSz))
        return tpc_lib_api::GLUE_INCOMPATIBLE_INPUT_SIZE;

    out_defs->indexSpaceRank = 2;
    out_defs->indexSpaceGeometry[0] = T;
    out_defs->indexSpaceGeometry[1] = S;

    for (int i = 0; i < NUM_INPUTS; i++)
        out_defs->inputTensorAccessPattern[i].allRequired = true;
    auto& apOut = out_defs->outputTensorAccessPattern[PAIR];
    Map(apOut.mapping[0], 0, 0, 0, static_cast<float>(2 * D - 1));
    Map(apOut.mapping[1], 0, 1, 0, 0);
    Map(apOut.mapping[2], 1, 1, 0, 0);

    const float invD = 1.0f / static_cast<float>(D);
    out_defs->kernel.paramsNr = 2;
    memcpy(&out_defs->kernel.scalarParams[0], &params->eps, sizeof(float));
    memcpy(&out_defs->kernel.scalarParams[1], &invD, sizeof(float));

    return CopyElf(out_defs, &_binary___inkling2_pair_rms_norm_gaudi2_o_start,
                   &_binary___inkling2_pair_rms_norm_gaudi2_o_end);
}

tpc_lib_api::GlueCodeReturn Inkling2BlendRmsNormGaudi2::GetKernelName(char kernelName[tpc_lib_api::MAX_NODE_NAME])
{
    strcpy(kernelName, kGuid);
    return tpc_lib_api::GLUE_SUCCESS;
}

tpc_lib_api::GlueCodeReturn Inkling2BlendRmsNormGaudi2::GetGcDefinitions(
    tpc_lib_api::HabanaKernelParams* in_defs, tpc_lib_api::HabanaKernelInstantiation* out_defs)
{
    enum Input { DEST = 0, MESSAGE, GAIN_FORGET, NORM_W, NUM_INPUTS };
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
    const uint64_t D = in_defs->inputTensors[DEST].geometry.maxSizes[0];
    const uint64_t T = in_defs->inputTensors[DEST].geometry.maxSizes[1];
    const uint64_t S = in_defs->inputTensors[DEST].geometry.maxSizes[2];
    if (D == 0 || (D % 64) != 0 || D > 40 * 64 || T == 0 || S == 0 || !IsPosFinite(params->normEps))
        return tpc_lib_api::GLUE_INCOMPATIBLE_INPUT_SIZE;

    const uint64_t dSz[] = {D, T, S};
    const uint64_t gfSz[] = {2 * D, T, S};
    const uint64_t nwSz[] = {D, S};
    if (!Is(in_defs->inputTensors[DEST], tpc_lib_api::DATA_F32, 3, dSz) ||
        !Is(in_defs->inputTensors[MESSAGE], tpc_lib_api::DATA_F32, 3, dSz) ||
        !Is(in_defs->inputTensors[GAIN_FORGET], tpc_lib_api::DATA_BF16, 3, gfSz) ||
        !Is(in_defs->inputTensors[NORM_W], tpc_lib_api::DATA_BF16, 2, nwSz) ||
        !Is(in_defs->outputTensors[OUT], tpc_lib_api::DATA_BF16, 3, dSz))
        return tpc_lib_api::GLUE_INCOMPATIBLE_INPUT_SIZE;

    out_defs->indexSpaceRank = 2;
    out_defs->indexSpaceGeometry[0] = T;
    out_defs->indexSpaceGeometry[1] = S;

    for (int i = 0; i < NUM_INPUTS; i++)
        out_defs->inputTensorAccessPattern[i].allRequired = true;

    auto& apOut = out_defs->outputTensorAccessPattern[OUT];
    Map(apOut.mapping[0], 0, 0, 0, static_cast<float>(D - 1));
    Map(apOut.mapping[1], 0, 1, 0, 0);
    Map(apOut.mapping[2], 1, 1, 0, 0);

    const float invD = 1.0f / static_cast<float>(D);
    out_defs->kernel.paramsNr = 2;
    memcpy(&out_defs->kernel.scalarParams[0], &params->normEps, sizeof(float));
    memcpy(&out_defs->kernel.scalarParams[1], &invD, sizeof(float));

    return CopyElf(out_defs, &_binary___inkling2_blend_rms_norm_gaudi2_o_start,
                   &_binary___inkling2_blend_rms_norm_gaudi2_o_end);
}

tpc_lib_api::GlueCodeReturn Inkling2BlendRmsNormF8Gaudi2::GetKernelName(char kernelName[tpc_lib_api::MAX_NODE_NAME])
{
    strcpy(kernelName, kGuid);
    return tpc_lib_api::GLUE_SUCCESS;
}

tpc_lib_api::GlueCodeReturn Inkling2BlendRmsNormF8Gaudi2::GetGcDefinitions(
    tpc_lib_api::HabanaKernelParams* in_defs, tpc_lib_api::HabanaKernelInstantiation* out_defs)
{
    enum Input { DEST = 0, MESSAGE, GAIN_FORGET, NORM_W, NUM_INPUTS };
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
    const uint64_t D = in_defs->inputTensors[DEST].geometry.maxSizes[0];
    const uint64_t T = in_defs->inputTensors[DEST].geometry.maxSizes[1];
    const uint64_t S = in_defs->inputTensors[DEST].geometry.maxSizes[2];
    if (D == 0 || (D % 64) != 0 || D > 40 * 64 || T == 0 || S == 0 ||
        !IsPosFinite(params->normEps) || !IsPosFinite(params->fp8Max))
        return tpc_lib_api::GLUE_INCOMPATIBLE_INPUT_SIZE;

    const uint64_t dSz[] = {D, T, S};
    const uint64_t gfSz[] = {2 * D, T, S};
    const uint64_t nwSz[] = {D, S};
    const uint64_t scSz[] = {T, S};
    if (!Is(in_defs->inputTensors[DEST], tpc_lib_api::DATA_F32, 3, dSz) ||
        !Is(in_defs->inputTensors[MESSAGE], tpc_lib_api::DATA_F32, 3, dSz) ||
        !Is(in_defs->inputTensors[GAIN_FORGET], tpc_lib_api::DATA_BF16, 3, gfSz) ||
        !Is(in_defs->inputTensors[NORM_W], tpc_lib_api::DATA_BF16, 2, nwSz) ||
        !Is(in_defs->outputTensors[OUT_BF16], tpc_lib_api::DATA_BF16, 3, dSz) ||
        !Is(in_defs->outputTensors[OUT_F8], tpc_lib_api::DATA_F8_143, 3, dSz) ||
        !Is(in_defs->outputTensors[OUT_SCALE], tpc_lib_api::DATA_F32, 2, scSz))
        return tpc_lib_api::GLUE_INCOMPATIBLE_INPUT_SIZE;

    out_defs->indexSpaceRank = 2;
    out_defs->indexSpaceGeometry[0] = T;
    out_defs->indexSpaceGeometry[1] = S;

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
    out_defs->kernel.paramsNr = 3;
    memcpy(&out_defs->kernel.scalarParams[0], &params->normEps, sizeof(float));
    memcpy(&out_defs->kernel.scalarParams[1], &invD, sizeof(float));
    memcpy(&out_defs->kernel.scalarParams[2], &params->fp8Max, sizeof(float));

    return CopyElf(out_defs, &_binary___inkling2_blend_rms_norm_f8_gaudi2_o_start,
                   &_binary___inkling2_blend_rms_norm_f8_gaudi2_o_end);
}
