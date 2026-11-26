// SPDX-License-Identifier: Apache-2.0
#include "inkling2_moe_gaudi2.hpp"

#include <cmath>
#include <cstring>

#include "inkling2_glue_common.hpp"

extern unsigned char _binary___inkling2_moe_w13_situ_f8_gaudi2_o_start;
extern unsigned char _binary___inkling2_moe_w13_situ_f8_gaudi2_o_end;
extern unsigned char _binary___inkling2_moe_w2_f8_gaudi2_o_start;
extern unsigned char _binary___inkling2_moe_w2_f8_gaudi2_o_end;
extern unsigned char _binary___inkling2_moe_group_experts_gaudi2_o_start;
extern unsigned char _binary___inkling2_moe_group_experts_gaudi2_o_end;
extern unsigned char _binary___inkling2_moe_w13_grouped_f8_gaudi2_o_start;
extern unsigned char _binary___inkling2_moe_w13_grouped_f8_gaudi2_o_end;
extern unsigned char _binary___inkling2_moe_w2_grouped_f8_gaudi2_o_start;
extern unsigned char _binary___inkling2_moe_w2_grouped_f8_gaudi2_o_end;
extern unsigned char _binary___inkling2_moe_gate_topk_gaudi2_o_start;
extern unsigned char _binary___inkling2_moe_gate_topk_gaudi2_o_end;
extern unsigned char _binary___inkling2_moe_gate_topk_e128_gaudi2_o_start;
extern unsigned char _binary___inkling2_moe_gate_topk_e128_gaudi2_o_end;
extern unsigned char _binary___inkling2_moe_gate_gemv_splitk_gaudi2_o_start;
extern unsigned char _binary___inkling2_moe_gate_gemv_splitk_gaudi2_o_end;
extern unsigned char _binary___inkling2_moe_gate_gemv_splitk_b1_gaudi2_o_start;
extern unsigned char _binary___inkling2_moe_gate_gemv_splitk_b1_gaudi2_o_end;
extern unsigned char _binary___inkling2_moe_sum_sconv_res_decode_gaudi2_o_start;
extern unsigned char _binary___inkling2_moe_sum_sconv_res_decode_gaudi2_o_end;
extern unsigned char _binary___inkling2_moe_sum_sconv_pack_decode_gaudi2_o_start;
extern unsigned char _binary___inkling2_moe_sum_sconv_pack_decode_gaudi2_o_end;

using inkling2_glue::CopyElf;
using inkling2_glue::Is;
using inkling2_glue::IsPosFinite;
using inkling2_glue::Map;

tpc_lib_api::GlueCodeReturn Inkling2MoeW13SituGaudi2::GetKernelName(char kernelName[tpc_lib_api::MAX_NODE_NAME])
{
    strcpy(kernelName, kGuid);
    return tpc_lib_api::GLUE_SUCCESS;
}

tpc_lib_api::GlueCodeReturn Inkling2MoeW13SituGaudi2::GetGcDefinitions(
    tpc_lib_api::HabanaKernelParams* in_defs, tpc_lib_api::HabanaKernelInstantiation* out_defs)
{
    enum Input { X_F8 = 0, X_SCALE, W13, S13, TOPK_IDS, TOPK_WEIGHTS, NUM_INPUTS };
    enum Output { ACT_F8 = 0, ACT_SCALE, NUM_OUTPUTS };

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
    const uint64_t K4 = in_defs->inputTensors[X_F8].geometry.maxSizes[0];
    const uint64_t K = K4 * 4;
    const uint64_t B = in_defs->inputTensors[X_F8].geometry.maxSizes[1];
    const uint64_t tileW = in_defs->inputTensors[W13].geometry.maxSizes[0];
    const uint64_t nBlks = in_defs->inputTensors[W13].geometry.maxSizes[2];
    const uint64_t E = in_defs->inputTensors[W13].geometry.maxSizes[3];
    const uint64_t topK = in_defs->inputTensors[TOPK_IDS].geometry.maxSizes[0];

    if (K4 == 0 || K4 > 1024 || B == 0 || tileW != 256 || nBlks == 0 || E == 0 || topK == 0 ||
        !IsPosFinite(params->gateCap) || !IsPosFinite(params->upCap) || !IsPosFinite(params->fp8Max))
        return tpc_lib_api::GLUE_INCOMPATIBLE_INPUT_SIZE;
    const uint64_t twoI = nBlks * 256;
    const uint64_t totalItems = nBlks * topK * B;

    const uint64_t xSz[] = {K4, B};
    const uint64_t xsSz[] = {B};
    const uint64_t w13Sz[] = {256, K, nBlks, E};
    const uint64_t s13Sz[] = {twoI, E};
    const uint64_t topkSz[] = {topK, B};
    const uint64_t actSz[] = {128, totalItems};
    const uint64_t actScSz[] = {totalItems};

    if (!Is(in_defs->inputTensors[X_F8], tpc_lib_api::DATA_I32, 2, xSz) ||
        !Is(in_defs->inputTensors[X_SCALE], tpc_lib_api::DATA_F32, 1, xsSz) ||
        !Is(in_defs->inputTensors[W13], tpc_lib_api::DATA_F8_143, 4, w13Sz) ||
        !Is(in_defs->inputTensors[S13], tpc_lib_api::DATA_F32, 2, s13Sz) ||
        !Is(in_defs->inputTensors[TOPK_IDS], tpc_lib_api::DATA_I32, 2, topkSz) ||
        !Is(in_defs->inputTensors[TOPK_WEIGHTS], tpc_lib_api::DATA_F32, 2, topkSz) ||
        !Is(in_defs->outputTensors[ACT_F8], tpc_lib_api::DATA_F8_143, 2, actSz) ||
        !Is(in_defs->outputTensors[ACT_SCALE], tpc_lib_api::DATA_F32, 1, actScSz))
        return tpc_lib_api::GLUE_INCOMPATIBLE_INPUT_SIZE;

    const int chunkSize = 1;
    const int totalItemsInt = static_cast<int>(totalItems);
    out_defs->indexSpaceRank = 1;
    out_defs->indexSpaceGeometry[0] = totalItems;

    for (int i = 0; i < NUM_INPUTS; i++)
        out_defs->inputTensorAccessPattern[i].allRequired = true;

    auto& apAct = out_defs->outputTensorAccessPattern[ACT_F8];
    Map(apAct.mapping[0], 0, 0, 0, 127);
    Map(apAct.mapping[1], 0, 1, 0, 0);

    auto& apSc = out_defs->outputTensorAccessPattern[ACT_SCALE];
    Map(apSc.mapping[0], 0, 1, 0, 0);

    out_defs->kernel.paramsNr = 5;
    memcpy(&out_defs->kernel.scalarParams[0], &params->gateCap, sizeof(float));
    memcpy(&out_defs->kernel.scalarParams[1], &params->upCap, sizeof(float));
    memcpy(&out_defs->kernel.scalarParams[2], &params->fp8Max, sizeof(float));
    memcpy(&out_defs->kernel.scalarParams[3], &totalItemsInt, sizeof(int));
    memcpy(&out_defs->kernel.scalarParams[4], &chunkSize, sizeof(int));

    return CopyElf(out_defs, &_binary___inkling2_moe_w13_situ_f8_gaudi2_o_start,
                   &_binary___inkling2_moe_w13_situ_f8_gaudi2_o_end);
}

tpc_lib_api::GlueCodeReturn Inkling2MoeW2Gaudi2::GetKernelName(char kernelName[tpc_lib_api::MAX_NODE_NAME])
{
    strcpy(kernelName, kGuid);
    return tpc_lib_api::GLUE_SUCCESS;
}

tpc_lib_api::GlueCodeReturn Inkling2MoeW2Gaudi2::GetGcDefinitions(
    tpc_lib_api::HabanaKernelParams* in_defs, tpc_lib_api::HabanaKernelInstantiation* out_defs)
{
    enum Input { ACT_F8 = 0, ACT_SCALE, W2, S2, TOPK_IDS, NUM_INPUTS };
    enum Output { OUT_BF16 = 0, NUM_OUTPUTS };

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

    const uint64_t I4 = in_defs->inputTensors[ACT_F8].geometry.maxSizes[0];
    const uint64_t I = I4 * 4;
    const uint64_t topK = in_defs->inputTensors[ACT_F8].geometry.maxSizes[1];
    const uint64_t B = in_defs->inputTensors[ACT_F8].geometry.maxSizes[2];
    const uint64_t tileW = in_defs->inputTensors[W2].geometry.maxSizes[0];
    const uint64_t nBlks = in_defs->inputTensors[W2].geometry.maxSizes[2];
    const uint64_t E = in_defs->inputTensors[W2].geometry.maxSizes[3];

    if (I4 == 0 || I4 > 512 || (I % 128) != 0 || topK == 0 || B == 0 || tileW != 256 || nBlks == 0 || E == 0)
        return tpc_lib_api::GLUE_INCOMPATIBLE_INPUT_SIZE;
    const uint64_t nGroups = I / 128;
    if (nGroups > 32)
        return tpc_lib_api::GLUE_INCOMPATIBLE_INPUT_SIZE;
    const uint64_t H = nBlks * 256;
    const uint64_t totalItems = nBlks * topK * B;

    const uint64_t actSz[] = {I4, topK, B};
    const uint64_t actScSz[] = {nGroups, topK, B};
    const uint64_t w2Sz[] = {256, I, nBlks, E};
    const uint64_t s2Sz[] = {H, E};
    const uint64_t topkSz[] = {topK, B};
    const uint64_t outSz[] = {256, totalItems};

    if (!Is(in_defs->inputTensors[ACT_F8], tpc_lib_api::DATA_I32, 3, actSz) ||
        !Is(in_defs->inputTensors[ACT_SCALE], tpc_lib_api::DATA_F32, 3, actScSz) ||
        !Is(in_defs->inputTensors[W2], tpc_lib_api::DATA_F8_143, 4, w2Sz) ||
        !Is(in_defs->inputTensors[S2], tpc_lib_api::DATA_F32, 2, s2Sz) ||
        !Is(in_defs->inputTensors[TOPK_IDS], tpc_lib_api::DATA_I32, 2, topkSz) ||
        !Is(in_defs->outputTensors[OUT_BF16], tpc_lib_api::DATA_BF16, 2, outSz))
        return tpc_lib_api::GLUE_INCOMPATIBLE_INPUT_SIZE;

    const int chunkSize = 1;
    const int totalItemsInt = static_cast<int>(totalItems);
    out_defs->indexSpaceRank = 1;
    out_defs->indexSpaceGeometry[0] = totalItems;

    for (int i = 0; i < NUM_INPUTS; i++)
        out_defs->inputTensorAccessPattern[i].allRequired = true;

    auto& apOut = out_defs->outputTensorAccessPattern[OUT_BF16];
    Map(apOut.mapping[0], 0, 0, 0, 255);
    Map(apOut.mapping[1], 0, 1, 0, 0);

    out_defs->kernel.paramsNr = 2;
    memcpy(&out_defs->kernel.scalarParams[0], &totalItemsInt, sizeof(int));
    memcpy(&out_defs->kernel.scalarParams[1], &chunkSize, sizeof(int));
    return CopyElf(out_defs, &_binary___inkling2_moe_w2_f8_gaudi2_o_start,
                   &_binary___inkling2_moe_w2_f8_gaudi2_o_end);
}

tpc_lib_api::GlueCodeReturn Inkling2MoeGroupExpertsGaudi2::GetKernelName(char kernelName[tpc_lib_api::MAX_NODE_NAME])
{
    strcpy(kernelName, kGuid);
    return tpc_lib_api::GLUE_SUCCESS;
}

tpc_lib_api::GlueCodeReturn Inkling2MoeGroupExpertsGaudi2::GetGcDefinitions(
    tpc_lib_api::HabanaKernelParams* in_defs, tpc_lib_api::HabanaKernelInstantiation* out_defs)
{
    enum Input { TOPK_IDS = 0, NUM_INPUTS };
    enum Output { GROUP_META = 0, GROUP_COUNT, NUM_OUTPUTS };

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
    const uint64_t topK = in_defs->inputTensors[TOPK_IDS].geometry.maxSizes[0];
    const uint64_t B = in_defs->inputTensors[TOPK_IDS].geometry.maxSizes[1];
    const uint64_t M = topK * B;
    if (topK == 0 || B == 0 || M > 128 || params->numExperts <= 0 || params->numExperts > 256 ||
        params->nW13Blks <= 0 || params->nW2Blks <= 0)
        return tpc_lib_api::GLUE_INCOMPATIBLE_INPUT_SIZE;

    const uint64_t tkSz[] = {topK, B};
    const uint64_t metaSz[] = {128, 8};
    const uint64_t cntSz[] = {2};
    if (!Is(in_defs->inputTensors[TOPK_IDS], tpc_lib_api::DATA_I32, 2, tkSz) ||
        !Is(in_defs->outputTensors[GROUP_META], tpc_lib_api::DATA_I32, 2, metaSz) ||
        !Is(in_defs->outputTensors[GROUP_COUNT], tpc_lib_api::DATA_I32, 1, cntSz))
        return tpc_lib_api::GLUE_INCOMPATIBLE_INPUT_SIZE;

    const uint32_t numCoresU = (in_defs->maxAvailableTpc > 0 && in_defs->maxAvailableTpc <= 64)
                                   ? in_defs->maxAvailableTpc
                                   : 24;
    const int numCores = static_cast<int>(numCoresU);

    out_defs->indexSpaceRank = 1;
    out_defs->indexSpaceGeometry[0] = 1;
    out_defs->inputTensorAccessPattern[TOPK_IDS].allRequired = true;

    auto& apMeta = out_defs->outputTensorAccessPattern[GROUP_META];
    Map(apMeta.mapping[0], 0, 0, 0, 127);
    Map(apMeta.mapping[1], 0, 0, 0, 7);
    auto& apCnt = out_defs->outputTensorAccessPattern[GROUP_COUNT];
    Map(apCnt.mapping[0], 0, 0, 0, 1);

    out_defs->kernel.paramsNr = 4;
    memcpy(&out_defs->kernel.scalarParams[0], &params->numExperts, sizeof(int));
    memcpy(&out_defs->kernel.scalarParams[1], &params->nW13Blks, sizeof(int));
    memcpy(&out_defs->kernel.scalarParams[2], &params->nW2Blks, sizeof(int));
    memcpy(&out_defs->kernel.scalarParams[3], &numCores, sizeof(int));
    return CopyElf(out_defs, &_binary___inkling2_moe_group_experts_gaudi2_o_start,
                   &_binary___inkling2_moe_group_experts_gaudi2_o_end);
}

tpc_lib_api::GlueCodeReturn Inkling2MoeW13GroupedGaudi2::GetKernelName(char kernelName[tpc_lib_api::MAX_NODE_NAME])
{
    strcpy(kernelName, kGuid);
    return tpc_lib_api::GLUE_SUCCESS;
}

tpc_lib_api::GlueCodeReturn Inkling2MoeW13GroupedGaudi2::GetGcDefinitions(
    tpc_lib_api::HabanaKernelParams* in_defs, tpc_lib_api::HabanaKernelInstantiation* out_defs)
{
    enum Input { X_F8 = 0, X_SCALE, W13, S13, TOPK_WEIGHTS, GROUP_META, GROUP_COUNT, NUM_INPUTS };
    enum Output { ACT_F8 = 0, ACT_SCALE, NUM_OUTPUTS };

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
    const uint64_t K4 = in_defs->inputTensors[X_F8].geometry.maxSizes[0];
    const uint64_t K = K4 * 4;
    const uint64_t B = in_defs->inputTensors[X_F8].geometry.maxSizes[1];
    const uint64_t tileW = in_defs->inputTensors[W13].geometry.maxSizes[0];
    const uint64_t nBlks = in_defs->inputTensors[W13].geometry.maxSizes[2];
    const uint64_t E = in_defs->inputTensors[W13].geometry.maxSizes[3];
    const uint64_t topK = in_defs->inputTensors[TOPK_WEIGHTS].geometry.maxSizes[0];
    const uint64_t M = topK * B;

    if (K4 == 0 || K4 > 640 || B == 0 || tileW != 256 || nBlks == 0 || E == 0 || topK == 0 || M > 128 ||
        !IsPosFinite(params->gateCap) || !IsPosFinite(params->upCap) || !IsPosFinite(params->fp8Max))
        return tpc_lib_api::GLUE_INCOMPATIBLE_INPUT_SIZE;
    const uint64_t twoI = nBlks * 256;
    const uint64_t totalItems = nBlks * M;

    const uint64_t xSz[] = {K4, B};
    const uint64_t xsSz[] = {B};
    const uint64_t w13Sz[] = {256, K, nBlks, E};
    const uint64_t s13Sz[] = {twoI, E};
    const uint64_t topkSz[] = {topK, B};
    const uint64_t metaSz[] = {128, 8};
    const uint64_t cntSz[] = {2};
    const uint64_t actSz[] = {128, totalItems};
    const uint64_t actScSz[] = {totalItems};

    if (!Is(in_defs->inputTensors[X_F8], tpc_lib_api::DATA_I32, 2, xSz) ||
        !Is(in_defs->inputTensors[X_SCALE], tpc_lib_api::DATA_F32, 1, xsSz) ||
        !Is(in_defs->inputTensors[W13], tpc_lib_api::DATA_F8_143, 4, w13Sz) ||
        !Is(in_defs->inputTensors[S13], tpc_lib_api::DATA_F32, 2, s13Sz) ||
        !Is(in_defs->inputTensors[TOPK_WEIGHTS], tpc_lib_api::DATA_F32, 2, topkSz) ||
        !Is(in_defs->inputTensors[GROUP_META], tpc_lib_api::DATA_I32, 2, metaSz) ||
        !Is(in_defs->inputTensors[GROUP_COUNT], tpc_lib_api::DATA_I32, 1, cntSz) ||
        !Is(in_defs->outputTensors[ACT_F8], tpc_lib_api::DATA_F8_143, 2, actSz) ||
        !Is(in_defs->outputTensors[ACT_SCALE], tpc_lib_api::DATA_F32, 1, actScSz))
        return tpc_lib_api::GLUE_INCOMPATIBLE_INPUT_SIZE;

    const uint32_t numCoresU = (in_defs->maxAvailableTpc > 0 && in_defs->maxAvailableTpc <= 64)
                                   ? in_defs->maxAvailableTpc
                                   : 24;
    const int numCores = static_cast<int>(numCoresU);
    out_defs->indexSpaceRank = 1;
    out_defs->indexSpaceGeometry[0] = numCoresU;
    out_defs->preferredSplitDim = 1;

    for (int i = 0; i < NUM_INPUTS; i++)
    {
        auto& ap = out_defs->inputTensorAccessPattern[i];
        ap.allRequired = true;
        for (unsigned d = 0; d < in_defs->inputTensors[i].geometry.dims; d++)
        {
            ap.mapping[d].indexSpaceDim = 0;
            ap.mapping[d].allRequired = true;
        }
    }
    for (int o = 0; o < NUM_OUTPUTS; o++)
    {
        auto& ap = out_defs->outputTensorAccessPattern[o];
        ap.allRequired = true;
        for (unsigned d = 0; d < in_defs->outputTensors[o].geometry.dims; d++)
        {
            ap.mapping[d].indexSpaceDim = 0;
            ap.mapping[d].allRequired = true;
        }
    }

    out_defs->kernel.paramsNr = 4;
    memcpy(&out_defs->kernel.scalarParams[0], &params->gateCap, sizeof(float));
    memcpy(&out_defs->kernel.scalarParams[1], &params->upCap, sizeof(float));
    memcpy(&out_defs->kernel.scalarParams[2], &params->fp8Max, sizeof(float));
    memcpy(&out_defs->kernel.scalarParams[3], &numCores, sizeof(int));

    return CopyElf(out_defs, &_binary___inkling2_moe_w13_grouped_f8_gaudi2_o_start,
                   &_binary___inkling2_moe_w13_grouped_f8_gaudi2_o_end);
}

tpc_lib_api::GlueCodeReturn Inkling2MoeW2GroupedGaudi2::GetKernelName(char kernelName[tpc_lib_api::MAX_NODE_NAME])
{
    strcpy(kernelName, kGuid);
    return tpc_lib_api::GLUE_SUCCESS;
}

tpc_lib_api::GlueCodeReturn Inkling2MoeW2GroupedGaudi2::GetGcDefinitions(
    tpc_lib_api::HabanaKernelParams* in_defs, tpc_lib_api::HabanaKernelInstantiation* out_defs)
{
    enum Input { ACT_F8 = 0, ACT_SCALE, W2, S2, GROUP_META, GROUP_COUNT, NUM_INPUTS };
    enum Output { OUT_BF16 = 0, NUM_OUTPUTS };

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

    const uint64_t I4 = in_defs->inputTensors[ACT_F8].geometry.maxSizes[0];
    const uint64_t I = I4 * 4;
    const uint64_t topK = in_defs->inputTensors[ACT_F8].geometry.maxSizes[1];
    const uint64_t B = in_defs->inputTensors[ACT_F8].geometry.maxSizes[2];
    const uint64_t tileW = in_defs->inputTensors[W2].geometry.maxSizes[0];
    const uint64_t nBlks = in_defs->inputTensors[W2].geometry.maxSizes[2];
    const uint64_t E = in_defs->inputTensors[W2].geometry.maxSizes[3];
    const uint64_t M = topK * B;

    if (I4 == 0 || I4 > 384 || (I % 128) != 0 || topK == 0 || B == 0 || M > 128 ||
        tileW != 256 || nBlks == 0 || E == 0)
        return tpc_lib_api::GLUE_INCOMPATIBLE_INPUT_SIZE;
    const uint64_t nGroups = I / 128;
    if (nGroups > 32)
        return tpc_lib_api::GLUE_INCOMPATIBLE_INPUT_SIZE;
    const uint64_t H = nBlks * 256;
    const uint64_t totalItems = nBlks * M;

    const uint64_t actSz[] = {I4, topK, B};
    const uint64_t actScSz[] = {nGroups, topK, B};
    const uint64_t w2Sz[] = {256, I, nBlks, E};
    const uint64_t s2Sz[] = {H, E};
    const uint64_t metaSz[] = {128, 8};
    const uint64_t cntSz[] = {2};
    const uint64_t outSz[] = {256, totalItems};

    if (!Is(in_defs->inputTensors[ACT_F8], tpc_lib_api::DATA_I32, 3, actSz) ||
        !Is(in_defs->inputTensors[ACT_SCALE], tpc_lib_api::DATA_F32, 3, actScSz) ||
        !Is(in_defs->inputTensors[W2], tpc_lib_api::DATA_F8_143, 4, w2Sz) ||
        !Is(in_defs->inputTensors[S2], tpc_lib_api::DATA_F32, 2, s2Sz) ||
        !Is(in_defs->inputTensors[GROUP_META], tpc_lib_api::DATA_I32, 2, metaSz) ||
        !Is(in_defs->inputTensors[GROUP_COUNT], tpc_lib_api::DATA_I32, 1, cntSz) ||
        !Is(in_defs->outputTensors[OUT_BF16], tpc_lib_api::DATA_BF16, 2, outSz))
        return tpc_lib_api::GLUE_INCOMPATIBLE_INPUT_SIZE;

    const uint32_t numCoresU = (in_defs->maxAvailableTpc > 0 && in_defs->maxAvailableTpc <= 64)
                                   ? in_defs->maxAvailableTpc
                                   : 24;
    const int numCores = static_cast<int>(numCoresU);
    out_defs->indexSpaceRank = 1;
    out_defs->indexSpaceGeometry[0] = numCoresU;
    out_defs->preferredSplitDim = 1;

    for (int i = 0; i < NUM_INPUTS; i++)
    {
        auto& ap = out_defs->inputTensorAccessPattern[i];
        ap.allRequired = true;
        for (unsigned d = 0; d < in_defs->inputTensors[i].geometry.dims; d++)
        {
            ap.mapping[d].indexSpaceDim = 0;
            ap.mapping[d].allRequired = true;
        }
    }
    auto& apOut = out_defs->outputTensorAccessPattern[OUT_BF16];
    apOut.allRequired = true;
    for (unsigned d = 0; d < 2; d++)
    {
        apOut.mapping[d].indexSpaceDim = 0;
        apOut.mapping[d].allRequired = true;
    }

    out_defs->kernel.paramsNr = 1;
    memcpy(&out_defs->kernel.scalarParams[0], &numCores, sizeof(int));
    return CopyElf(out_defs, &_binary___inkling2_moe_w2_grouped_f8_gaudi2_o_start,
                   &_binary___inkling2_moe_w2_grouped_f8_gaudi2_o_end);
}

tpc_lib_api::GlueCodeReturn Inkling2MoeGateTopkGaudi2::GetKernelName(char kernelName[tpc_lib_api::MAX_NODE_NAME])
{
    strcpy(kernelName, kGuid);
    return tpc_lib_api::GLUE_SUCCESS;
}

tpc_lib_api::GlueCodeReturn Inkling2MoeGateTopkGaudi2::GetGcDefinitions(
    tpc_lib_api::HabanaKernelParams* in_defs, tpc_lib_api::HabanaKernelInstantiation* out_defs)
{
    enum Input { LOGITS = 0, BIAS, NUM_INPUTS };
    enum Output { TOPK_IDS = 0, TOPK_WEIGHTS, NUM_OUTPUTS };

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
    const uint64_t E = in_defs->inputTensors[LOGITS].geometry.maxSizes[0];
    const uint64_t S = in_defs->inputTensors[LOGITS].geometry.maxSizes[1];
    const uint64_t B = in_defs->inputTensors[LOGITS].geometry.maxSizes[2];
    const int topK = params->topK;

    if (E == 0 || (E % 64) != 0 || E > 256 || S == 0 || B == 0 || topK <= 0 || topK > 16 ||
        static_cast<uint64_t>(topK) > E || !std::isfinite(params->routeScale))
        return tpc_lib_api::GLUE_INCOMPATIBLE_INPUT_SIZE;

    const uint64_t lgSz[] = {E, S, B};
    const uint64_t bsSz[] = {E};
    const uint64_t tkSz[] = {static_cast<uint64_t>(topK), B};

    if (!Is(in_defs->inputTensors[LOGITS], tpc_lib_api::DATA_F32, 3, lgSz) ||
        !Is(in_defs->inputTensors[BIAS], tpc_lib_api::DATA_F32, 1, bsSz) ||
        !Is(in_defs->outputTensors[TOPK_IDS], tpc_lib_api::DATA_I32, 2, tkSz) ||
        !Is(in_defs->outputTensors[TOPK_WEIGHTS], tpc_lib_api::DATA_F32, 2, tkSz))
        return tpc_lib_api::GLUE_INCOMPATIBLE_INPUT_SIZE;

    out_defs->indexSpaceRank = 1;
    out_defs->indexSpaceGeometry[0] = B;

    for (int i = 0; i < NUM_INPUTS; i++)
        out_defs->inputTensorAccessPattern[i].allRequired = true;

    for (int o = 0; o < NUM_OUTPUTS; o++)
    {
        auto& ap = out_defs->outputTensorAccessPattern[o];
        Map(ap.mapping[0], 0, 0, 0, static_cast<float>(topK - 1));
        Map(ap.mapping[1], 0, 1, 0, 0);
    }

    out_defs->kernel.paramsNr = 2;
    memcpy(&out_defs->kernel.scalarParams[0], &params->topK, sizeof(int));
    memcpy(&out_defs->kernel.scalarParams[1], &params->routeScale, sizeof(float));

    if (E == 128)
        return CopyElf(out_defs, &_binary___inkling2_moe_gate_topk_e128_gaudi2_o_start,
                       &_binary___inkling2_moe_gate_topk_e128_gaudi2_o_end);
    return CopyElf(out_defs, &_binary___inkling2_moe_gate_topk_gaudi2_o_start,
                   &_binary___inkling2_moe_gate_topk_gaudi2_o_end);
}

tpc_lib_api::GlueCodeReturn Inkling2MoeGateGemvSplitkGaudi2::GetKernelName(
    char kernelName[tpc_lib_api::MAX_NODE_NAME])
{
    strcpy(kernelName, kGuid);
    return tpc_lib_api::GLUE_SUCCESS;
}

tpc_lib_api::GlueCodeReturn Inkling2MoeGateGemvSplitkGaudi2::GetGcDefinitions(
    tpc_lib_api::HabanaKernelParams* in_defs, tpc_lib_api::HabanaKernelInstantiation* out_defs)
{
    enum Input { X = 0, W_GATE, NUM_INPUTS };
    enum Output { PARTIAL_LOGITS = 0, NUM_OUTPUTS };

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

    const uint64_t H = in_defs->inputTensors[X].geometry.maxSizes[0];
    const uint64_t B = in_defs->inputTensors[X].geometry.maxSizes[1];
    if (H == 0 || (H % 128) != 0 || B == 0 || B > 16)
        return tpc_lib_api::GLUE_INCOMPATIBLE_INPUT_SIZE;
    const uint64_t S = H / 128;

    const uint64_t xSz[] = {H, B};
    const uint64_t wSz[] = {128, H};
    const uint64_t pSz[] = {128, S, B};

    if (!Is(in_defs->inputTensors[X], tpc_lib_api::DATA_BF16, 2, xSz) ||
        !Is(in_defs->inputTensors[W_GATE], tpc_lib_api::DATA_BF16, 2, wSz) ||
        !Is(in_defs->outputTensors[PARTIAL_LOGITS], tpc_lib_api::DATA_F32, 3, pSz))
        return tpc_lib_api::GLUE_INCOMPATIBLE_INPUT_SIZE;

    out_defs->indexSpaceRank = 1;
    out_defs->indexSpaceGeometry[0] = S;

    for (int i = 0; i < NUM_INPUTS; i++)
        out_defs->inputTensorAccessPattern[i].allRequired = true;

    auto& apP = out_defs->outputTensorAccessPattern[PARTIAL_LOGITS];
    Map(apP.mapping[0], 0, 0, 0, 127);
    Map(apP.mapping[1], 0, 1, 0, 0);
    Map(apP.mapping[2], 0, 0, 0, static_cast<float>(B - 1));

    out_defs->kernel.paramsNr = 0;
    if (B == 1)
        return CopyElf(out_defs, &_binary___inkling2_moe_gate_gemv_splitk_b1_gaudi2_o_start,
                       &_binary___inkling2_moe_gate_gemv_splitk_b1_gaudi2_o_end);
    return CopyElf(out_defs, &_binary___inkling2_moe_gate_gemv_splitk_gaudi2_o_start,
                   &_binary___inkling2_moe_gate_gemv_splitk_gaudi2_o_end);
}

tpc_lib_api::GlueCodeReturn Inkling2MoeSumSconvResDecodeGaudi2::GetKernelName(
    char kernelName[tpc_lib_api::MAX_NODE_NAME])
{
    strcpy(kernelName, kGuid);
    return tpc_lib_api::GLUE_SUCCESS;
}

tpc_lib_api::GlueCodeReturn Inkling2MoeSumSconvResDecodeGaudi2::GetGcDefinitions(
    tpc_lib_api::HabanaKernelParams* in_defs, tpc_lib_api::HabanaKernelInstantiation* out_defs)
{
    enum Input
    {
        MOE_PARTIAL = 0,
        CONV_STATE,
        LOAD_SLOTS,
        WEIGHT_T,
        PARTIAL_RES,
        ATTN_OUT,
        NEW_KV_STATE,
        NEW_ATTN_STATE
    };
    enum Output { NEW_PARTIAL = 0, NEW_MLP_STATE, NUM_OUTPUTS };

    if (in_defs->inputTensorNr != 6 && in_defs->inputTensorNr != 8)
    {
        in_defs->inputTensorNr = 6;
        return tpc_lib_api::GLUE_INCOMPATIBLE_INPUT_COUNT;
    }
    if (in_defs->outputTensorNr != NUM_OUTPUTS)
    {
        in_defs->outputTensorNr = NUM_OUTPUTS;
        return tpc_lib_api::GLUE_INCOMPATIBLE_OUTPUT_COUNT;
    }
    if (in_defs->nodeParams.nodeParams == nullptr || in_defs->nodeParams.nodeParamsSize < sizeof(Params))
        return tpc_lib_api::GLUE_INCOMPATIBLE_INPUT_SIZE;

    const bool packFullState = (in_defs->inputTensorNr == 8);
    const auto* params = static_cast<const Params*>(in_defs->nodeParams.nodeParams);
    const uint64_t H = in_defs->inputTensors[MOE_PARTIAL].geometry.maxSizes[0];
    const uint64_t topK = in_defs->inputTensors[MOE_PARTIAL].geometry.maxSizes[1];
    const uint64_t B = in_defs->inputTensors[MOE_PARTIAL].geometry.maxSizes[2];
    const uint64_t totalDim = in_defs->inputTensors[CONV_STATE].geometry.maxSizes[0];
    const uint64_t numSlots = in_defs->inputTensors[CONV_STATE].geometry.maxSizes[2];

    if (H == 0 || (H % 64) != 0 || topK == 0 || B == 0 || numSlots == 0 ||
        params->streamOffset < 0 || static_cast<uint64_t>(params->streamOffset) + H > totalDim ||
        !std::isfinite(params->outScale))
        return tpc_lib_api::GLUE_INCOMPATIBLE_INPUT_SIZE;

    uint64_t kvWidth = 0;
    if (packFullState)
    {
        kvWidth = in_defs->inputTensors[NEW_KV_STATE].geometry.maxSizes[0];
        if (kvWidth == 0 || (kvWidth % 64) != 0 || kvWidth > 2 * H || totalDim != kvWidth + 2 * H ||
            static_cast<uint64_t>(params->streamOffset) != kvWidth + H)
            return tpc_lib_api::GLUE_INCOMPATIBLE_INPUT_SIZE;
    }

    const uint64_t mpSz[] = {H, topK, B};
    const uint64_t csSz[] = {totalDim, 3, numSlots};
    const uint64_t lsSz[] = {B};
    const uint64_t stSz[] = {packFullState ? totalDim : H, 3, B};
    const uint64_t kvStSz[] = {kvWidth, 3, B};
    const uint64_t attnStSz[] = {H, 3, B};
    const uint64_t wtSz[] = {H, 4};
    const uint64_t hbSz[] = {H, B};

    if (!Is(in_defs->inputTensors[MOE_PARTIAL], tpc_lib_api::DATA_BF16, 3, mpSz) ||
        !Is(in_defs->inputTensors[CONV_STATE], tpc_lib_api::DATA_BF16, 3, csSz) ||
        !Is(in_defs->inputTensors[LOAD_SLOTS], tpc_lib_api::DATA_I32, 1, lsSz) ||
        !Is(in_defs->inputTensors[WEIGHT_T], tpc_lib_api::DATA_BF16, 2, wtSz) ||
        !Is(in_defs->inputTensors[PARTIAL_RES], tpc_lib_api::DATA_BF16, 2, hbSz) ||
        !Is(in_defs->inputTensors[ATTN_OUT], tpc_lib_api::DATA_BF16, 2, hbSz) ||
        (packFullState && (!Is(in_defs->inputTensors[NEW_KV_STATE], tpc_lib_api::DATA_BF16, 3, kvStSz) ||
                           !Is(in_defs->inputTensors[NEW_ATTN_STATE], tpc_lib_api::DATA_BF16, 3, attnStSz))) ||
        !Is(in_defs->outputTensors[NEW_PARTIAL], tpc_lib_api::DATA_BF16, 2, hbSz) ||
        !Is(in_defs->outputTensors[NEW_MLP_STATE], tpc_lib_api::DATA_BF16, 3, stSz))
        return tpc_lib_api::GLUE_INCOMPATIBLE_INPUT_SIZE;

    out_defs->indexSpaceRank = 2;
    out_defs->indexSpaceGeometry[0] = H / 64;
    out_defs->indexSpaceGeometry[1] = B;

    for (unsigned i = 0; i < in_defs->inputTensorNr; i++)
        out_defs->inputTensorAccessPattern[i].allRequired = true;

    auto& apP = out_defs->outputTensorAccessPattern[NEW_PARTIAL];
    Map(apP.mapping[0], 0, 64, 0, 63);
    Map(apP.mapping[1], 1, 1, 0, 0);

    auto& apS = out_defs->outputTensorAccessPattern[NEW_MLP_STATE];
    Map(apS.mapping[0], 0, 64, 0, packFullState ? static_cast<float>(kvWidth + H + 63) : 63.0f);
    Map(apS.mapping[1], 0, 0, 0, 2);
    Map(apS.mapping[2], 1, 1, 0, 0);

    out_defs->kernel.paramsNr = 3;
    memcpy(&out_defs->kernel.scalarParams[0], &params->outScale, sizeof(float));
    memcpy(&out_defs->kernel.scalarParams[1], &params->isFirst, sizeof(int));
    memcpy(&out_defs->kernel.scalarParams[2], &params->streamOffset, sizeof(int));

    if (packFullState)
        return CopyElf(out_defs, &_binary___inkling2_moe_sum_sconv_pack_decode_gaudi2_o_start,
                       &_binary___inkling2_moe_sum_sconv_pack_decode_gaudi2_o_end);
    return CopyElf(out_defs, &_binary___inkling2_moe_sum_sconv_res_decode_gaudi2_o_start,
                   &_binary___inkling2_moe_sum_sconv_res_decode_gaudi2_o_end);
}
