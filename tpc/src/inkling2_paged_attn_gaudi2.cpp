// SPDX-License-Identifier: Apache-2.0
#include "inkling2_paged_attn_gaudi2.hpp"

#include <cmath>
#include <cstring>

#include "inkling2_glue_common.hpp"

extern unsigned char _binary___inkling2_rel_paged_attn_gate_sink_gaudi2_o_start;
extern unsigned char _binary___inkling2_rel_paged_attn_gate_sink_gaudi2_o_end;
extern unsigned char _binary___inkling2_rel_paged_attn_gqa3_gaudi2_o_start;
extern unsigned char _binary___inkling2_rel_paged_attn_gqa3_gaudi2_o_end;

using inkling2_glue::CopyElf;
using inkling2_glue::Is;
using inkling2_glue::Map;

tpc_lib_api::GlueCodeReturn Inkling2RelPagedAttnGateSinkGaudi2::GetKernelName(
    char kernelName[tpc_lib_api::MAX_NODE_NAME])
{
    strcpy(kernelName, kGuid);
    return tpc_lib_api::GLUE_SUCCESS;
}

tpc_lib_api::GlueCodeReturn Inkling2RelPagedAttnGateSinkGaudi2::GetGcDefinitions(
    tpc_lib_api::HabanaKernelParams* in_defs, tpc_lib_api::HabanaKernelInstantiation* out_defs)
{
    enum Input
    {
        Q = 0,
        R,
        PROJ_Q,
        REL_PROJ_T,
        KEY_CACHE,
        VAL_CACHE,
        K_NEW,
        V_NEW,
        BLK_LIST,
        BLK_GROUPS,
        BLK_USAGE,
        BLK_START_POS,
        POSITIONS,
        NUM_INPUTS
    };
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
    const uint64_t D = in_defs->inputTensors[Q].geometry.maxSizes[0];
    const uint64_t H = in_defs->inputTensors[Q].geometry.maxSizes[1];
    const uint64_t B = in_defs->inputTensors[Q].geometry.maxSizes[2];
    const uint64_t dRel = in_defs->inputTensors[R].geometry.maxSizes[0];
    const uint64_t qCols = in_defs->inputTensors[PROJ_Q].geometry.maxSizes[0];
    const uint64_t relExt = in_defs->inputTensors[REL_PROJ_T].geometry.maxSizes[1];
    const uint64_t Hkv = in_defs->inputTensors[KEY_CACHE].geometry.maxSizes[1];
    const uint64_t BS = in_defs->inputTensors[KEY_CACHE].geometry.maxSizes[2];
    const uint64_t numPhys = in_defs->inputTensors[KEY_CACHE].geometry.maxSizes[3];
    const uint64_t NB = in_defs->inputTensors[BLK_LIST].geometry.maxSizes[0];

    if (D != 128 || H == 0 || H > 64 || B == 0 || dRel == 0 || dRel > 64 ||
        qCols < H * (256 + dRel) + H ||
        Hkv == 0 || (H % Hkv) != 0 || BS == 0 || numPhys == 0 || NB == 0 || NB > 512 ||
        params->relExtent <= 0 || static_cast<uint64_t>(params->relExtent) > relExt ||
        !std::isfinite(params->scale))
        return tpc_lib_api::GLUE_INCOMPATIBLE_INPUT_SIZE;

    const uint64_t qSz[] = {D, H, B};
    const uint64_t rSz[] = {dRel, H, B};
    const uint64_t pqSz[] = {qCols, B};
    const uint64_t rpSz[] = {dRel, relExt};
    const uint64_t kvSz[] = {D, Hkv, BS, numPhys};
    const uint64_t kvNewSz[] = {D, Hkv, B};
    const uint64_t nbSz[] = {NB};
    const uint64_t posSz[] = {B};

    if (!Is(in_defs->inputTensors[Q], tpc_lib_api::DATA_BF16, 3, qSz) ||
        !Is(in_defs->inputTensors[R], tpc_lib_api::DATA_BF16, 3, rSz) ||
        !Is(in_defs->inputTensors[PROJ_Q], tpc_lib_api::DATA_BF16, 2, pqSz) ||
        !Is(in_defs->inputTensors[REL_PROJ_T], tpc_lib_api::DATA_BF16, 2, rpSz) ||
        !Is(in_defs->inputTensors[KEY_CACHE], tpc_lib_api::DATA_BF16, 4, kvSz) ||
        !Is(in_defs->inputTensors[VAL_CACHE], tpc_lib_api::DATA_BF16, 4, kvSz) ||
        !Is(in_defs->inputTensors[K_NEW], tpc_lib_api::DATA_BF16, 3, kvNewSz) ||
        !Is(in_defs->inputTensors[V_NEW], tpc_lib_api::DATA_BF16, 3, kvNewSz) ||
        !Is(in_defs->inputTensors[BLK_LIST], tpc_lib_api::DATA_I32, 1, nbSz) ||
        !Is(in_defs->inputTensors[BLK_GROUPS], tpc_lib_api::DATA_I32, 1, nbSz) ||
        !Is(in_defs->inputTensors[BLK_USAGE], tpc_lib_api::DATA_I32, 1, nbSz) ||
        !Is(in_defs->inputTensors[BLK_START_POS], tpc_lib_api::DATA_I32, 1, nbSz) ||
        !Is(in_defs->inputTensors[POSITIONS], tpc_lib_api::DATA_I32, 1, posSz) ||
        !Is(in_defs->outputTensors[OUT], tpc_lib_api::DATA_BF16, 3, qSz))
        return tpc_lib_api::GLUE_INCOMPATIBLE_INPUT_SIZE;

    const bool useGqa3 = (B >= 3 && (H / Hkv) == 3);

    out_defs->indexSpaceRank = 2;
    out_defs->indexSpaceGeometry[0] = useGqa3 ? Hkv : H;
    out_defs->indexSpaceGeometry[1] = B;

    for (int i = 0; i < NUM_INPUTS; i++)
        out_defs->inputTensorAccessPattern[i].allRequired = true;

    auto& apOut = out_defs->outputTensorAccessPattern[OUT];
    Map(apOut.mapping[0], 0, 0, 0, 127);
    if (useGqa3)
        Map(apOut.mapping[1], 0, 3, 0, 2);
    else
        Map(apOut.mapping[1], 0, 1, 0, 0);
    Map(apOut.mapping[2], 1, 1, 0, 0);

    out_defs->kernel.paramsNr = 3;
    memcpy(&out_defs->kernel.scalarParams[0], &params->scale, sizeof(float));
    memcpy(&out_defs->kernel.scalarParams[1], &params->slidingWindow, sizeof(int));
    memcpy(&out_defs->kernel.scalarParams[2], &params->relExtent, sizeof(int));

    if (useGqa3)
        return CopyElf(out_defs, &_binary___inkling2_rel_paged_attn_gqa3_gaudi2_o_start,
                       &_binary___inkling2_rel_paged_attn_gqa3_gaudi2_o_end);
    return CopyElf(out_defs, &_binary___inkling2_rel_paged_attn_gate_sink_gaudi2_o_start,
                   &_binary___inkling2_rel_paged_attn_gate_sink_gaudi2_o_end);
}
