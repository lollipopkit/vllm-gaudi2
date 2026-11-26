// SPDX-License-Identifier: Apache-2.0
#include "gaudi2_fp8_ops_gaudi2.hpp"

#include <cstring>

#include "inkling2_glue_common.hpp"

extern unsigned char _binary___gaudi2_add_rms_norm_f8_f32w_gaudi2_o_start;
extern unsigned char _binary___gaudi2_add_rms_norm_f8_f32w_gaudi2_o_end;
extern unsigned char _binary___gaudi2_add_rms_norm_f8_bf16_f32w_gaudi2_o_start;
extern unsigned char _binary___gaudi2_add_rms_norm_f8_bf16_f32w_gaudi2_o_end;
extern unsigned char _binary___gaudi2_add_rms_norm_f8_bf16w_gaudi2_o_start;
extern unsigned char _binary___gaudi2_add_rms_norm_f8_bf16w_gaudi2_o_end;
extern unsigned char _binary___gaudi2_add_rms_norm_f8_bf16_bf16w_gaudi2_o_start;
extern unsigned char _binary___gaudi2_add_rms_norm_f8_bf16_bf16w_gaudi2_o_end;
extern unsigned char _binary___gaudi2_sandwich_rms_norm_f8_gaudi2_o_start;
extern unsigned char _binary___gaudi2_sandwich_rms_norm_f8_gaudi2_o_end;
extern unsigned char _binary___gaudi2_silu_mul_f8_gaudi2_o_start;
extern unsigned char _binary___gaudi2_silu_mul_f8_gaudi2_o_end;
extern unsigned char _binary___gdn_prep_decode_gaudi2_o_start;
extern unsigned char _binary___gdn_prep_decode_gaudi2_o_end;
extern unsigned char _binary___gdn_gated_rms_norm_f8_gaudi2_o_start;
extern unsigned char _binary___gdn_gated_rms_norm_f8_gaudi2_o_end;
extern unsigned char _binary___lfm3_short_conv_decode_gaudi2_o_start;
extern unsigned char _binary___lfm3_short_conv_decode_gaudi2_o_end;

using inkling2_glue::CopyElf;
using inkling2_glue::Is;
using inkling2_glue::IsPosFinite;
using inkling2_glue::Map;

namespace
{
// Map a 2D [width, M] tensor onto a rank-2 index space [1, M] with zero duplicated index-space dims,
// matching Habana's FCD-reduction pattern so Synapse GC can SRAM-slice cleanly along M.
inline void MapRow2D(tpc_lib_api::TensorAccessPattern& p, uint64_t width)
{
    Map(p.mapping[0], 0, 0, 0, static_cast<float>(width - 1));
    p.mapping[0].allRequired = true;
    Map(p.mapping[1], 1, 1, 0, 0);
}
}  // namespace

// ============================================================================
// 1. Gaudi2AddRmsNormF8Gaudi2
// ============================================================================

tpc_lib_api::GlueCodeReturn Gaudi2AddRmsNormF8Gaudi2::GetKernelName(char kernelName[tpc_lib_api::MAX_NODE_NAME])
{
    strcpy(kernelName, kGuid);
    return tpc_lib_api::GLUE_SUCCESS;
}

tpc_lib_api::GlueCodeReturn Gaudi2AddRmsNormF8Gaudi2::GetGcDefinitions(
    tpc_lib_api::HabanaKernelParams* in_defs, tpc_lib_api::HabanaKernelInstantiation* out_defs)
{
    enum Input { X = 0, RESIDUAL, WEIGHT, NUM_INPUTS };
    enum Output { OUT_F8 = 0, OUT_SC, OUT_RES, OUT_BF16 };

    if (in_defs->inputTensorNr != NUM_INPUTS)
    {
        in_defs->inputTensorNr = NUM_INPUTS;
        return tpc_lib_api::GLUE_INCOMPATIBLE_INPUT_COUNT;
    }
    if (in_defs->outputTensorNr != 3 && in_defs->outputTensorNr != 4)
    {
        in_defs->outputTensorNr = 3;
        return tpc_lib_api::GLUE_INCOMPATIBLE_OUTPUT_COUNT;
    }
    if (in_defs->nodeParams.nodeParams == nullptr || in_defs->nodeParams.nodeParamsSize < sizeof(Params))
        return tpc_lib_api::GLUE_INCOMPATIBLE_INPUT_SIZE;

    const auto* params = static_cast<const Params*>(in_defs->nodeParams.nodeParams);
    const bool outBf16 = (in_defs->outputTensorNr == 4);
    const auto wType = in_defs->inputTensors[WEIGHT].geometry.dataType;
    const bool wF32 = (wType == tpc_lib_api::DATA_F32);
    if (!wF32 && wType != tpc_lib_api::DATA_BF16)
        return tpc_lib_api::GLUE_INCOMPATIBLE_DATA_TYPE;

    const uint64_t D = in_defs->inputTensors[X].geometry.maxSizes[0];
    const uint64_t M = in_defs->inputTensors[X].geometry.maxSizes[1];
    if (D == 0 || (D % 64) != 0 || D > static_cast<uint64_t>(kMaxDim) || M == 0 ||
        !IsPosFinite(params->eps) || !IsPosFinite(params->fp8Max))
        return tpc_lib_api::GLUE_INCOMPATIBLE_INPUT_SIZE;

    const uint64_t matSz[] = {D, M};
    const uint64_t wSz[] = {D};
    const uint64_t scSz[] = {1, M};
    if (!Is(in_defs->inputTensors[X], tpc_lib_api::DATA_BF16, 2, matSz) ||
        !Is(in_defs->inputTensors[RESIDUAL], tpc_lib_api::DATA_BF16, 2, matSz) ||
        !Is(in_defs->inputTensors[WEIGHT], wType, 1, wSz) ||
        !Is(in_defs->outputTensors[OUT_F8], tpc_lib_api::DATA_F8_143, 2, matSz) ||
        !Is(in_defs->outputTensors[OUT_SC], tpc_lib_api::DATA_F32, 2, scSz) ||
        !Is(in_defs->outputTensors[OUT_RES], tpc_lib_api::DATA_BF16, 2, matSz))
        return tpc_lib_api::GLUE_INCOMPATIBLE_INPUT_SIZE;
    if (outBf16 && !Is(in_defs->outputTensors[OUT_BF16], tpc_lib_api::DATA_BF16, 2, matSz))
        return tpc_lib_api::GLUE_INCOMPATIBLE_INPUT_SIZE;

    out_defs->indexSpaceRank = 2;
    out_defs->indexSpaceGeometry[0] = 1;
    out_defs->indexSpaceGeometry[1] = M;

    MapRow2D(out_defs->inputTensorAccessPattern[X], D);
    MapRow2D(out_defs->inputTensorAccessPattern[RESIDUAL], D);
    out_defs->inputTensorAccessPattern[WEIGHT].allRequired = true;
    MapRow2D(out_defs->outputTensorAccessPattern[OUT_F8], D);
    MapRow2D(out_defs->outputTensorAccessPattern[OUT_SC], 1);
    MapRow2D(out_defs->outputTensorAccessPattern[OUT_RES], D);
    if (outBf16)
        MapRow2D(out_defs->outputTensorAccessPattern[OUT_BF16], D);

    const float invD = 1.0f / static_cast<float>(D);
    out_defs->kernel.paramsNr = 3;
    memcpy(&out_defs->kernel.scalarParams[0], &params->eps, sizeof(float));
    memcpy(&out_defs->kernel.scalarParams[1], &invD, sizeof(float));
    memcpy(&out_defs->kernel.scalarParams[2], &params->fp8Max, sizeof(float));

    if (wF32 && !outBf16)
        return CopyElf(out_defs, &_binary___gaudi2_add_rms_norm_f8_f32w_gaudi2_o_start,
                       &_binary___gaudi2_add_rms_norm_f8_f32w_gaudi2_o_end);
    if (wF32 && outBf16)
        return CopyElf(out_defs, &_binary___gaudi2_add_rms_norm_f8_bf16_f32w_gaudi2_o_start,
                       &_binary___gaudi2_add_rms_norm_f8_bf16_f32w_gaudi2_o_end);
    if (!wF32 && !outBf16)
        return CopyElf(out_defs, &_binary___gaudi2_add_rms_norm_f8_bf16w_gaudi2_o_start,
                       &_binary___gaudi2_add_rms_norm_f8_bf16w_gaudi2_o_end);
    return CopyElf(out_defs, &_binary___gaudi2_add_rms_norm_f8_bf16_bf16w_gaudi2_o_start,
                   &_binary___gaudi2_add_rms_norm_f8_bf16_bf16w_gaudi2_o_end);
}

// ============================================================================
// 1b. Gaudi2SandwichRmsNormF8Gaudi2
// ============================================================================

tpc_lib_api::GlueCodeReturn Gaudi2SandwichRmsNormF8Gaudi2::GetKernelName(char kernelName[tpc_lib_api::MAX_NODE_NAME])
{
    strcpy(kernelName, kGuid);
    return tpc_lib_api::GLUE_SUCCESS;
}

tpc_lib_api::GlueCodeReturn Gaudi2SandwichRmsNormF8Gaudi2::GetGcDefinitions(
    tpc_lib_api::HabanaKernelParams* in_defs, tpc_lib_api::HabanaKernelInstantiation* out_defs)
{
    enum Input { X = 0, RESIDUAL, POST_W, PRE_W, NUM_INPUTS };
    enum Output { OUT_F8 = 0, OUT_SC, OUT_RES, NUM_OUTPUTS };

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
    const uint64_t D = in_defs->inputTensors[X].geometry.maxSizes[0];
    const uint64_t M = in_defs->inputTensors[X].geometry.maxSizes[1];
    if (D == 0 || (D % 64) != 0 || D > static_cast<uint64_t>(kMaxDim) || M == 0 ||
        !IsPosFinite(params->eps) || !IsPosFinite(params->fp8Max))
        return tpc_lib_api::GLUE_INCOMPATIBLE_INPUT_SIZE;

    const uint64_t matSz[] = {D, M};
    const uint64_t wSz[] = {D};
    const uint64_t scSz[] = {1, M};
    if (!Is(in_defs->inputTensors[X], tpc_lib_api::DATA_BF16, 2, matSz) ||
        !Is(in_defs->inputTensors[RESIDUAL], tpc_lib_api::DATA_BF16, 2, matSz) ||
        !Is(in_defs->inputTensors[POST_W], tpc_lib_api::DATA_BF16, 1, wSz) ||
        !Is(in_defs->inputTensors[PRE_W], tpc_lib_api::DATA_BF16, 1, wSz) ||
        !Is(in_defs->outputTensors[OUT_F8], tpc_lib_api::DATA_F8_143, 2, matSz) ||
        !Is(in_defs->outputTensors[OUT_SC], tpc_lib_api::DATA_F32, 2, scSz) ||
        !Is(in_defs->outputTensors[OUT_RES], tpc_lib_api::DATA_BF16, 2, matSz))
        return tpc_lib_api::GLUE_INCOMPATIBLE_INPUT_SIZE;

    out_defs->indexSpaceRank = 2;
    out_defs->indexSpaceGeometry[0] = 1;
    out_defs->indexSpaceGeometry[1] = M;

    MapRow2D(out_defs->inputTensorAccessPattern[X], D);
    MapRow2D(out_defs->inputTensorAccessPattern[RESIDUAL], D);
    out_defs->inputTensorAccessPattern[POST_W].allRequired = true;
    out_defs->inputTensorAccessPattern[PRE_W].allRequired = true;
    MapRow2D(out_defs->outputTensorAccessPattern[OUT_F8], D);
    MapRow2D(out_defs->outputTensorAccessPattern[OUT_SC], 1);
    MapRow2D(out_defs->outputTensorAccessPattern[OUT_RES], D);

    const float invD = 1.0f / static_cast<float>(D);
    out_defs->kernel.paramsNr = 3;
    memcpy(&out_defs->kernel.scalarParams[0], &params->eps, sizeof(float));
    memcpy(&out_defs->kernel.scalarParams[1], &invD, sizeof(float));
    memcpy(&out_defs->kernel.scalarParams[2], &params->fp8Max, sizeof(float));

    return CopyElf(out_defs, &_binary___gaudi2_sandwich_rms_norm_f8_gaudi2_o_start,
                   &_binary___gaudi2_sandwich_rms_norm_f8_gaudi2_o_end);
}

// ============================================================================
// 2. Gaudi2SiluMulF8Gaudi2
// ============================================================================

tpc_lib_api::GlueCodeReturn Gaudi2SiluMulF8Gaudi2::GetKernelName(char kernelName[tpc_lib_api::MAX_NODE_NAME])
{
    strcpy(kernelName, kGuid);
    return tpc_lib_api::GLUE_SUCCESS;
}

tpc_lib_api::GlueCodeReturn Gaudi2SiluMulF8Gaudi2::GetGcDefinitions(
    tpc_lib_api::HabanaKernelParams* in_defs, tpc_lib_api::HabanaKernelInstantiation* out_defs)
{
    enum Input { GATE_UP = 0, NUM_INPUTS };
    enum Output { OUT_F8 = 0, OUT_SC, NUM_OUTPUTS };

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
    const uint64_t N = in_defs->outputTensors[OUT_F8].geometry.maxSizes[0];
    const uint64_t M = in_defs->outputTensors[OUT_F8].geometry.maxSizes[1];
    if (N == 0 || (N % 64) != 0 || N > static_cast<uint64_t>(kMaxDim) || M == 0 || !IsPosFinite(params->fp8Max))
        return tpc_lib_api::GLUE_INCOMPATIBLE_INPUT_SIZE;

    const uint64_t guSz[] = {2 * N, M};
    const uint64_t outSz[] = {N, M};
    const uint64_t scSz[] = {1, M};
    if (!Is(in_defs->inputTensors[GATE_UP], tpc_lib_api::DATA_BF16, 2, guSz) ||
        !Is(in_defs->outputTensors[OUT_F8], tpc_lib_api::DATA_F8_143, 2, outSz) ||
        !Is(in_defs->outputTensors[OUT_SC], tpc_lib_api::DATA_F32, 2, scSz))
        return tpc_lib_api::GLUE_INCOMPATIBLE_INPUT_SIZE;

    out_defs->indexSpaceRank = 2;
    out_defs->indexSpaceGeometry[0] = 1;
    out_defs->indexSpaceGeometry[1] = M;

    MapRow2D(out_defs->inputTensorAccessPattern[GATE_UP], 2 * N);
    MapRow2D(out_defs->outputTensorAccessPattern[OUT_F8], N);
    MapRow2D(out_defs->outputTensorAccessPattern[OUT_SC], 1);

    out_defs->kernel.paramsNr = 1;
    memcpy(&out_defs->kernel.scalarParams[0], &params->fp8Max, sizeof(float));

    return CopyElf(out_defs, &_binary___gaudi2_silu_mul_f8_gaudi2_o_start,
                   &_binary___gaudi2_silu_mul_f8_gaudi2_o_end);
}

// ============================================================================
// 3. GdnPrepDecodeGaudi2
// ============================================================================

tpc_lib_api::GlueCodeReturn GdnPrepDecodeGaudi2::GetKernelName(char kernelName[tpc_lib_api::MAX_NODE_NAME])
{
    strcpy(kernelName, kGuid);
    return tpc_lib_api::GLUE_SUCCESS;
}

tpc_lib_api::GlueCodeReturn GdnPrepDecodeGaudi2::GetGcDefinitions(
    tpc_lib_api::HabanaKernelParams* in_defs, tpc_lib_api::HabanaKernelInstantiation* out_defs)
{
    enum Input { PACKED_QKV = 0, A_IN, B_IN, NEG_EXP_A, DT_BIAS, CONV_STATE, CONV_TAPS, CONV_BIAS, NUM_INPUTS };
    enum Output { QK_OUT = 0, V_OUT, DECAY_OUT, BETA_OUT, KQDOT_OUT, NUM_OUTPUTS };

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
    const uint64_t H = in_defs->outputTensors[KQDOT_OUT].geometry.maxSizes[0];
    const uint64_t B = in_defs->outputTensors[KQDOT_OUT].geometry.maxSizes[1];
    const uint64_t HV = in_defs->outputTensors[DECAY_OUT].geometry.maxSizes[0];
    const uint64_t W = (2 * H + HV) * kHeadDim;
    const uint64_t cacheLen = in_defs->inputTensors[CONV_STATE].geometry.maxSizes[1];
    const uint64_t slots = in_defs->inputTensors[CONV_STATE].geometry.maxSizes[2];

    if (H == 0 || HV == 0 || HV > 64 || B == 0 || cacheLen < 3 ||
        params->rowOffset < 0 || static_cast<uint64_t>(params->rowOffset) + B > slots ||
        params->convColOffset < 0 || static_cast<uint64_t>(params->convColOffset) + 3 > cacheLen)
        return tpc_lib_api::GLUE_INCOMPATIBLE_INPUT_SIZE;

    const uint64_t qkvSz[] = {W, B};
    const uint64_t abSz[] = {HV, B};
    const uint64_t hv1Sz[] = {HV};
    const uint64_t csSz[] = {W, cacheLen, slots};
    const uint64_t tapsSz[] = {W, 4};
    const uint64_t biasSz[] = {W};
    const uint64_t qkOutSz[] = {kHeadDim, 2 * H, B};
    const uint64_t vOutSz[] = {kHeadDim, HV, B};
    const uint64_t kqdSz[] = {H, B};

    if (!Is(in_defs->inputTensors[PACKED_QKV], tpc_lib_api::DATA_BF16, 2, qkvSz) ||
        !Is(in_defs->inputTensors[A_IN], tpc_lib_api::DATA_BF16, 2, abSz) ||
        !Is(in_defs->inputTensors[B_IN], tpc_lib_api::DATA_BF16, 2, abSz) ||
        !Is(in_defs->inputTensors[NEG_EXP_A], tpc_lib_api::DATA_F32, 1, hv1Sz) ||
        !Is(in_defs->inputTensors[DT_BIAS], tpc_lib_api::DATA_F32, 1, hv1Sz) ||
        !Is(in_defs->inputTensors[CONV_STATE], tpc_lib_api::DATA_BF16, 3, csSz) ||
        !Is(in_defs->inputTensors[CONV_TAPS], tpc_lib_api::DATA_F32, 2, tapsSz) ||
        !Is(in_defs->inputTensors[CONV_BIAS], tpc_lib_api::DATA_F32, 1, biasSz) ||
        !Is(in_defs->outputTensors[QK_OUT], tpc_lib_api::DATA_F32, 3, qkOutSz) ||
        !Is(in_defs->outputTensors[V_OUT], tpc_lib_api::DATA_F32, 3, vOutSz) ||
        !Is(in_defs->outputTensors[DECAY_OUT], tpc_lib_api::DATA_F32, 2, abSz) ||
        !Is(in_defs->outputTensors[BETA_OUT], tpc_lib_api::DATA_F32, 2, abSz) ||
        !Is(in_defs->outputTensors[KQDOT_OUT], tpc_lib_api::DATA_F32, 2, kqdSz))
        return tpc_lib_api::GLUE_INCOMPATIBLE_INPUT_SIZE;

    out_defs->indexSpaceRank = 3;
    out_defs->indexSpaceGeometry[0] = 1;
    out_defs->indexSpaceGeometry[1] = H + HV;
    out_defs->indexSpaceGeometry[2] = B;

    // Explicit per-batch access patterns: CONV_STATE is updated in place at slots [rowOffset..rowOffset+B-1]
    // and must map explicitly (matching GdnDecodeGaudi2::MapPool) rather than allRequired=true.
    auto mapIn2 = [&](tpc_lib_api::TensorAccessPattern& p, uint64_t width) {
        Map(p.mapping[0], 0, 0, 0, static_cast<float>(width - 1));
        p.mapping[0].allRequired = true;
        Map(p.mapping[1], 2, 1, 0, 0);
    };
    mapIn2(out_defs->inputTensorAccessPattern[PACKED_QKV], W);
    mapIn2(out_defs->inputTensorAccessPattern[A_IN], HV);
    mapIn2(out_defs->inputTensorAccessPattern[B_IN], HV);
    out_defs->inputTensorAccessPattern[NEG_EXP_A].allRequired = true;
    out_defs->inputTensorAccessPattern[DT_BIAS].allRequired = true;
    {
        auto& apCs = out_defs->inputTensorAccessPattern[CONV_STATE];
        Map(apCs.mapping[0], 0, 0, 0, static_cast<float>(W - 1));
        apCs.mapping[0].allRequired = true;
        Map(apCs.mapping[1], 1, 0, static_cast<float>(params->convColOffset),
            static_cast<float>(params->convColOffset + 2));
        apCs.mapping[1].allRequired = true;
        Map(apCs.mapping[2], 2, 1, static_cast<float>(params->rowOffset),
            static_cast<float>(params->rowOffset));
    }
    out_defs->inputTensorAccessPattern[CONV_TAPS].allRequired = true;
    out_defs->inputTensorAccessPattern[CONV_BIAS].allRequired = true;

    auto mapOut3 = [&](tpc_lib_api::TensorAccessPattern& p, uint64_t d1) {
        Map(p.mapping[0], 0, 0, 0, kHeadDim - 1);
        p.mapping[0].allRequired = true;
        Map(p.mapping[1], 1, 0, 0, static_cast<float>(d1 - 1));
        p.mapping[1].allRequired = true;
        Map(p.mapping[2], 2, 1, 0, 0);
    };
    auto mapOut2 = [&](tpc_lib_api::TensorAccessPattern& p, uint64_t d0) {
        Map(p.mapping[0], 1, 0, 0, static_cast<float>(d0 - 1));
        p.mapping[0].allRequired = true;
        Map(p.mapping[1], 2, 1, 0, 0);
    };
    mapOut3(out_defs->outputTensorAccessPattern[QK_OUT], 2 * H);
    mapOut3(out_defs->outputTensorAccessPattern[V_OUT], HV);
    mapOut2(out_defs->outputTensorAccessPattern[DECAY_OUT], HV);
    mapOut2(out_defs->outputTensorAccessPattern[BETA_OUT], HV);
    mapOut2(out_defs->outputTensorAccessPattern[KQDOT_OUT], H);

    out_defs->kernel.paramsNr = 3;
    out_defs->kernel.scalarParams[0] = params->rowOffset;
    out_defs->kernel.scalarParams[1] = params->convColOffset;
    out_defs->kernel.scalarParams[2] = params->hasBias ? 1 : 0;

    return CopyElf(out_defs, &_binary___gdn_prep_decode_gaudi2_o_start,
                   &_binary___gdn_prep_decode_gaudi2_o_end);
}

// ============================================================================
// 4. GdnGatedRmsNormF8Gaudi2
// ============================================================================

tpc_lib_api::GlueCodeReturn GdnGatedRmsNormF8Gaudi2::GetKernelName(char kernelName[tpc_lib_api::MAX_NODE_NAME])
{
    strcpy(kernelName, kGuid);
    return tpc_lib_api::GLUE_SUCCESS;
}

tpc_lib_api::GlueCodeReturn GdnGatedRmsNormF8Gaudi2::GetGcDefinitions(
    tpc_lib_api::HabanaKernelParams* in_defs, tpc_lib_api::HabanaKernelInstantiation* out_defs)
{
    enum Input { X = 0, Z, WEIGHT, NUM_INPUTS };
    enum Output { OUT_F8 = 0, OUT_SC, NUM_OUTPUTS };

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
    const uint64_t HV = in_defs->inputTensors[X].geometry.maxSizes[1];
    const uint64_t B = in_defs->inputTensors[X].geometry.maxSizes[2];
    if (HV == 0 || HV > static_cast<uint64_t>(kMaxValueHeads) || B == 0 ||
        !IsPosFinite(params->eps) || !IsPosFinite(params->fp8Max))
        return tpc_lib_api::GLUE_INCOMPATIBLE_INPUT_SIZE;

    const uint64_t xSz[] = {kHeadDim, HV, B};
    const uint64_t zSz[] = {HV * kHeadDim, B};
    const uint64_t wSz[] = {kHeadDim};
    const uint64_t scSz[] = {1, B};
    if (!Is(in_defs->inputTensors[X], tpc_lib_api::DATA_F32, 3, xSz) ||
        !Is(in_defs->inputTensors[Z], tpc_lib_api::DATA_BF16, 2, zSz) ||
        !Is(in_defs->inputTensors[WEIGHT], tpc_lib_api::DATA_F32, 1, wSz) ||
        !Is(in_defs->outputTensors[OUT_F8], tpc_lib_api::DATA_F8_143, 2, zSz) ||
        !Is(in_defs->outputTensors[OUT_SC], tpc_lib_api::DATA_F32, 2, scSz))
        return tpc_lib_api::GLUE_INCOMPATIBLE_INPUT_SIZE;

    out_defs->indexSpaceRank = 3;
    out_defs->indexSpaceGeometry[0] = 1;
    out_defs->indexSpaceGeometry[1] = 1;
    out_defs->indexSpaceGeometry[2] = B;

    Map(out_defs->inputTensorAccessPattern[X].mapping[0], 0, 0, 0, kHeadDim - 1);
    out_defs->inputTensorAccessPattern[X].mapping[0].allRequired = true;
    Map(out_defs->inputTensorAccessPattern[X].mapping[1], 1, 0, 0, static_cast<float>(HV - 1));
    out_defs->inputTensorAccessPattern[X].mapping[1].allRequired = true;
    Map(out_defs->inputTensorAccessPattern[X].mapping[2], 2, 1, 0, 0);

    Map(out_defs->inputTensorAccessPattern[Z].mapping[0], 0, 0, 0, static_cast<float>(HV * kHeadDim - 1));
    out_defs->inputTensorAccessPattern[Z].mapping[0].allRequired = true;
    Map(out_defs->inputTensorAccessPattern[Z].mapping[1], 2, 1, 0, 0);
    out_defs->inputTensorAccessPattern[WEIGHT].allRequired = true;

    Map(out_defs->outputTensorAccessPattern[OUT_F8].mapping[0], 0, 0, 0, static_cast<float>(HV * kHeadDim - 1));
    out_defs->outputTensorAccessPattern[OUT_F8].mapping[0].allRequired = true;
    Map(out_defs->outputTensorAccessPattern[OUT_F8].mapping[1], 2, 1, 0, 0);
    Map(out_defs->outputTensorAccessPattern[OUT_SC].mapping[0], 0, 0, 0, 0);
    out_defs->outputTensorAccessPattern[OUT_SC].mapping[0].allRequired = true;
    Map(out_defs->outputTensorAccessPattern[OUT_SC].mapping[1], 2, 1, 0, 0);

    out_defs->kernel.paramsNr = 2;
    memcpy(&out_defs->kernel.scalarParams[0], &params->eps, sizeof(float));
    memcpy(&out_defs->kernel.scalarParams[1], &params->fp8Max, sizeof(float));

    return CopyElf(out_defs, &_binary___gdn_gated_rms_norm_f8_gaudi2_o_start,
                   &_binary___gdn_gated_rms_norm_f8_gaudi2_o_end);
}

// ============================================================================
// 5. Lfm3ShortConvDecodeGaudi2
// ============================================================================

tpc_lib_api::GlueCodeReturn Lfm3ShortConvDecodeGaudi2::GetKernelName(char kernelName[tpc_lib_api::MAX_NODE_NAME])
{
    strcpy(kernelName, kGuid);
    return tpc_lib_api::GLUE_SUCCESS;
}

tpc_lib_api::GlueCodeReturn Lfm3ShortConvDecodeGaudi2::GetGcDefinitions(
    tpc_lib_api::HabanaKernelParams* in_defs, tpc_lib_api::HabanaKernelInstantiation* out_defs)
{
    enum Input { BCX = 0, CONV_STATE, LOAD_IDX, TAPS, BIAS, NUM_INPUTS };
    enum Output { OUT = 0, NEW_STATE, NUM_OUTPUTS };

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
    const uint64_t D = in_defs->outputTensors[OUT].geometry.maxSizes[0];
    const uint64_t B = in_defs->outputTensors[OUT].geometry.maxSizes[1];
    const uint64_t cacheLen = in_defs->inputTensors[CONV_STATE].geometry.maxSizes[1];
    const uint64_t slots = in_defs->inputTensors[CONV_STATE].geometry.maxSizes[2];
    const uint64_t K = in_defs->inputTensors[TAPS].geometry.maxSizes[1];
    if (D == 0 || (D % 64) != 0 || B == 0 || K < 2 || cacheLen < K - 1 || slots == 0)
        return tpc_lib_api::GLUE_INCOMPATIBLE_INPUT_SIZE;

    const uint64_t stateLen = K - 1;
    const uint64_t bcxSz[] = {3 * D, B};
    const uint64_t csSz[] = {D, cacheLen, slots};
    const uint64_t idxSz[] = {B};
    const uint64_t tapsSz[] = {D, K};
    const uint64_t biasSz[] = {D};
    const uint64_t outSz[] = {D, B};
    const uint64_t newStateSz[] = {D, stateLen, B};

    if (!Is(in_defs->inputTensors[BCX], tpc_lib_api::DATA_BF16, 2, bcxSz) ||
        !Is(in_defs->inputTensors[CONV_STATE], tpc_lib_api::DATA_BF16, 3, csSz) ||
        !Is(in_defs->inputTensors[LOAD_IDX], tpc_lib_api::DATA_I32, 1, idxSz) ||
        !Is(in_defs->inputTensors[TAPS], tpc_lib_api::DATA_F32, 2, tapsSz) ||
        !Is(in_defs->inputTensors[BIAS], tpc_lib_api::DATA_F32, 1, biasSz) ||
        !Is(in_defs->outputTensors[OUT], tpc_lib_api::DATA_BF16, 2, outSz) ||
        !Is(in_defs->outputTensors[NEW_STATE], tpc_lib_api::DATA_BF16, 3, newStateSz))
        return tpc_lib_api::GLUE_INCOMPATIBLE_INPUT_SIZE;

    out_defs->indexSpaceRank = 3;
    out_defs->indexSpaceGeometry[0] = D / 64;
    out_defs->indexSpaceGeometry[1] = 1;
    out_defs->indexSpaceGeometry[2] = B;

    for (int i = 0; i < NUM_INPUTS; i++)
        out_defs->inputTensorAccessPattern[i].allRequired = true;

    Map(out_defs->outputTensorAccessPattern[OUT].mapping[0], 0, 64, 0, 63);
    Map(out_defs->outputTensorAccessPattern[OUT].mapping[1], 2, 1, 0, 0);

    auto& apNs = out_defs->outputTensorAccessPattern[NEW_STATE];
    Map(apNs.mapping[0], 0, 64, 0, 63);
    Map(apNs.mapping[1], 1, 0, 0, static_cast<float>(stateLen - 1));
    apNs.mapping[1].allRequired = true;
    Map(apNs.mapping[2], 2, 1, 0, 0);

    out_defs->kernel.paramsNr = 1;
    out_defs->kernel.scalarParams[0] = params->hasBias ? 1 : 0;

    return CopyElf(out_defs, &_binary___lfm3_short_conv_decode_gaudi2_o_start,
                   &_binary___lfm3_short_conv_decode_gaudi2_o_end);
}
