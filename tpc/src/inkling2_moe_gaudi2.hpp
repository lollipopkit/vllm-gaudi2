// SPDX-License-Identifier: Apache-2.0
// Glue code for Inkling 2 Mini MoE Gaudi2 TPC kernels:
//   - inkling2_moe_gate_gemv_splitk_gaudi2:     18-core split-K exact FP32-acc BF16 gate GEMV (decode E=128)
//   - inkling2_moe_gate_topk_gaudi2:            fused split-K reduce + single-tree sigmoid/bias/topk/softmax
//   - inkling2_moe_w13_situ_f8_gaudi2:          indirect FP8 w13 GEMV + Softcapped SiTU + compact quant
//   - inkling2_moe_w2_f8_gaudi2:                indirect FP8 w2 GEMV
//   - inkling2_moe_group_experts_gaudi2:        4-bucket cost-weighted expert grouper + core boundary partitioner
//   - inkling2_moe_w13_grouped_f8_gaudi2:       expert-grouped indirect FP8 w13 GEMV + Softcapped SiTU + quant
//   - inkling2_moe_w2_grouped_f8_gaudi2:        expert-grouped indirect FP8 w2 GEMV
//   - inkling2_moe_sum_sconv_res_decode_gaudi2: fused top-k sum + direct conv_cache short-conv + residual add
#pragma once

#include "gc_interface.h"
#include "inkling2_params.hpp"
#include "tpc_kernel_lib_interface.h"

class Inkling2MoeW13SituGaudi2
{
public:
    static constexpr const char* kGuid = "inkling2_moe_w13_situ_f8_gaudi2";
    using Params = inkling2_tpc::MoeW13Params;
    tpc_lib_api::GlueCodeReturn GetKernelName(char kernelName[tpc_lib_api::MAX_NODE_NAME]);
    tpc_lib_api::GlueCodeReturn GetGcDefinitions(tpc_lib_api::HabanaKernelParams* in_defs,
                                                 tpc_lib_api::HabanaKernelInstantiation* out_defs);
};

class Inkling2MoeW2Gaudi2
{
public:
    static constexpr const char* kGuid = "inkling2_moe_w2_f8_gaudi2";
    tpc_lib_api::GlueCodeReturn GetKernelName(char kernelName[tpc_lib_api::MAX_NODE_NAME]);
    tpc_lib_api::GlueCodeReturn GetGcDefinitions(tpc_lib_api::HabanaKernelParams* in_defs,
                                                 tpc_lib_api::HabanaKernelInstantiation* out_defs);
};

class Inkling2MoeGroupExpertsGaudi2
{
public:
    static constexpr const char* kGuid = "inkling2_moe_group_experts_gaudi2";
    using Params = inkling2_tpc::MoeGroupExpertsParams;
    tpc_lib_api::GlueCodeReturn GetKernelName(char kernelName[tpc_lib_api::MAX_NODE_NAME]);
    tpc_lib_api::GlueCodeReturn GetGcDefinitions(tpc_lib_api::HabanaKernelParams* in_defs,
                                                 tpc_lib_api::HabanaKernelInstantiation* out_defs);
};

class Inkling2MoeW13GroupedGaudi2
{
public:
    static constexpr const char* kGuid = "inkling2_moe_w13_grouped_f8_gaudi2";
    using Params = inkling2_tpc::MoeW13Params;
    tpc_lib_api::GlueCodeReturn GetKernelName(char kernelName[tpc_lib_api::MAX_NODE_NAME]);
    tpc_lib_api::GlueCodeReturn GetGcDefinitions(tpc_lib_api::HabanaKernelParams* in_defs,
                                                 tpc_lib_api::HabanaKernelInstantiation* out_defs);
};

class Inkling2MoeW2GroupedGaudi2
{
public:
    static constexpr const char* kGuid = "inkling2_moe_w2_grouped_f8_gaudi2";
    tpc_lib_api::GlueCodeReturn GetKernelName(char kernelName[tpc_lib_api::MAX_NODE_NAME]);
    tpc_lib_api::GlueCodeReturn GetGcDefinitions(tpc_lib_api::HabanaKernelParams* in_defs,
                                                 tpc_lib_api::HabanaKernelInstantiation* out_defs);
};

class Inkling2MoeGateTopkGaudi2
{
public:
    static constexpr const char* kGuid = "inkling2_moe_gate_topk_gaudi2";
    using Params = inkling2_tpc::MoeGateTopkParams;
    tpc_lib_api::GlueCodeReturn GetKernelName(char kernelName[tpc_lib_api::MAX_NODE_NAME]);
    tpc_lib_api::GlueCodeReturn GetGcDefinitions(tpc_lib_api::HabanaKernelParams* in_defs,
                                                 tpc_lib_api::HabanaKernelInstantiation* out_defs);
};

class Inkling2MoeGateGemvSplitkGaudi2
{
public:
    static constexpr const char* kGuid = "inkling2_moe_gate_gemv_splitk_gaudi2";
    tpc_lib_api::GlueCodeReturn GetKernelName(char kernelName[tpc_lib_api::MAX_NODE_NAME]);
    tpc_lib_api::GlueCodeReturn GetGcDefinitions(tpc_lib_api::HabanaKernelParams* in_defs,
                                                 tpc_lib_api::HabanaKernelInstantiation* out_defs);
};

class Inkling2MoeSumSconvResDecodeGaudi2
{
public:
    static constexpr const char* kGuid = "inkling2_moe_sum_sconv_res_decode_gaudi2";
    using Params = inkling2_tpc::MoeSumSconvResParams;
    tpc_lib_api::GlueCodeReturn GetKernelName(char kernelName[tpc_lib_api::MAX_NODE_NAME]);
    tpc_lib_api::GlueCodeReturn GetGcDefinitions(tpc_lib_api::HabanaKernelParams* in_defs,
                                                 tpc_lib_api::HabanaKernelInstantiation* out_defs);
};
