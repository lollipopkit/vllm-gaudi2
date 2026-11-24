// SPDX-License-Identifier: Apache-2.0
// Glue code for Inkling 2 Mini fused Block Attention Residual TPC kernels on Gaudi2:
//   - inkling2_attn_res_norm_bf16_gaudi2: stack/last(+addend) + softmax mix + RMSNorm -> bf16
//   - inkling2_attn_res_norm_f8_gaudi2:   stack/last(+addend) + softmax mix + RMSNorm -> (bf16, f8_143, scale)
//   - inkling2_attn_res_f32_gaudi2:       stack/last(+addend) + softmax mix -> f32
#pragma once

#include "gc_interface.h"
#include "inkling2_params.hpp"
#include "tpc_kernel_lib_interface.h"

class Inkling2AttnResNormBf16Gaudi2
{
public:
    static constexpr const char* kGuid = "inkling2_attn_res_norm_bf16_gaudi2";
    using Params = inkling2_tpc::AttnResNormBf16Params;

    tpc_lib_api::GlueCodeReturn GetKernelName(char kernelName[tpc_lib_api::MAX_NODE_NAME]);
    tpc_lib_api::GlueCodeReturn GetGcDefinitions(tpc_lib_api::HabanaKernelParams* in_defs,
                                                 tpc_lib_api::HabanaKernelInstantiation* out_defs);
};

class Inkling2AttnResNormF8Gaudi2
{
public:
    static constexpr const char* kGuid = "inkling2_attn_res_norm_f8_gaudi2";
    using Params = inkling2_tpc::AttnResNormF8Params;

    tpc_lib_api::GlueCodeReturn GetKernelName(char kernelName[tpc_lib_api::MAX_NODE_NAME]);
    tpc_lib_api::GlueCodeReturn GetGcDefinitions(tpc_lib_api::HabanaKernelParams* in_defs,
                                                 tpc_lib_api::HabanaKernelInstantiation* out_defs);
};

class Inkling2AttnResF32Gaudi2
{
public:
    static constexpr const char* kGuid = "inkling2_attn_res_f32_gaudi2";
    using Params = inkling2_tpc::AttnResF32Params;

    tpc_lib_api::GlueCodeReturn GetKernelName(char kernelName[tpc_lib_api::MAX_NODE_NAME]);
    tpc_lib_api::GlueCodeReturn GetGcDefinitions(tpc_lib_api::HabanaKernelParams* in_defs,
                                                 tpc_lib_api::HabanaKernelInstantiation* out_defs);
};
