// SPDX-License-Identifier: Apache-2.0
// Single-source X-macro registry for all custom Gaudi2 TPC kernel classes.
#pragma once

#include <cstring>

#include "gdn_decode_gaudi2.hpp"
#include "gdn_verify_gaudi2.hpp"
#include "inkling2_attn_res_gaudi2.hpp"
#include "inkling2_moe_gaudi2.hpp"
#include "inkling2_paged_attn_gaudi2.hpp"
#include "inkling2_sconv_norm_gaudi2.hpp"

#define GAUDI2_TPC_KERNELS(X)                   \
    X(GdnDecodeGaudi2)                          \
    X(GdnVerifyGaudi2)                          \
    X(Inkling2MoeW13SituGaudi2)                 \
    X(Inkling2MoeW2Gaudi2)                      \
    X(Inkling2AttnResNormBf16Gaudi2)            \
    X(Inkling2AttnResNormF8Gaudi2)              \
    X(Inkling2AttnResF32Gaudi2)                 \
    X(Inkling2SconvStaticDecodeGaudi2)          \
    X(Inkling2QkvrPrepDecodeGaudi2)             \
    X(Inkling2PairRmsNormGaudi2)                \
    X(Inkling2BlendRmsNormGaudi2)               \
    X(Inkling2BlendRmsNormF8Gaudi2)             \
    X(Inkling2RelPagedAttnGateSinkGaudi2)       \
    X(Inkling2MoeGateTopkGaudi2)                \
    X(Inkling2MoeSumSconvResDecodeGaudi2)       \
    X(Inkling2MoeGateGemvSplitkGaudi2)          \
    X(Inkling2MoeGroupExpertsGaudi2)            \
    X(Inkling2MoeW13GroupedGaudi2)              \
    X(Inkling2MoeW2GroupedGaudi2)

namespace gaudi2_tpc_registry
{
enum KernelIndex
{
#define GAUDI2_TPC_ENUM(cls) kIdx_##cls,
    GAUDI2_TPC_KERNELS(GAUDI2_TPC_ENUM)
#undef GAUDI2_TPC_ENUM
    kKernelCount
};

inline bool IsOurGuid(const char* guid)
{
    if (guid == nullptr)
        return false;
#define GAUDI2_TPC_MATCH(cls)          \
    if (strcmp(guid, cls::kGuid) == 0) \
        return true;
    GAUDI2_TPC_KERNELS(GAUDI2_TPC_MATCH)
#undef GAUDI2_TPC_MATCH
    return false;
}

inline void FillOurGuids(tpc_lib_api::GuidInfo* guids)
{
    memset(guids, 0, static_cast<size_t>(kKernelCount) * sizeof(guids[0]));
    uint32_t idx = 0;
#define GAUDI2_TPC_FILL(cls) cls().GetKernelName(guids[idx++].name);
    GAUDI2_TPC_KERNELS(GAUDI2_TPC_FILL)
#undef GAUDI2_TPC_FILL
}

inline bool TryInstantiateOurKernel(tpc_lib_api::HabanaKernelParams* params,
                                    tpc_lib_api::HabanaKernelInstantiation* instance,
                                    tpc_lib_api::GlueCodeReturn* outRc)
{
    if (params == nullptr || params->deviceId != tpc_lib_api::DEVICE_ID_GAUDI2)
        return false;
    const char* g = params->guid.name;
#define GAUDI2_TPC_INST(cls)                               \
    if (strcmp(g, cls::kGuid) == 0)                        \
    {                                                      \
        *outRc = cls().GetGcDefinitions(params, instance); \
        return true;                                       \
    }
    GAUDI2_TPC_KERNELS(GAUDI2_TPC_INST)
#undef GAUDI2_TPC_INST
    return false;
}
}  // namespace gaudi2_tpc_registry
