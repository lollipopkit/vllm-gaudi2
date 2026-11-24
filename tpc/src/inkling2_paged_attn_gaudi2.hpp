// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "inkling2_params.hpp"
#include "tpc_kernel_lib_interface.h"

class Inkling2RelPagedAttnGateSinkGaudi2
{
public:
    static constexpr const char* kGuid = "inkling2_rel_paged_attn_gate_sink_gaudi2";
    using Params = inkling2_tpc::RelPagedAttnParams;
    tpc_lib_api::GlueCodeReturn GetKernelName(char kernelName[tpc_lib_api::MAX_NODE_NAME]);
    tpc_lib_api::GlueCodeReturn GetGcDefinitions(tpc_lib_api::HabanaKernelParams* in_defs,
                                                 tpc_lib_api::HabanaKernelInstantiation* out_defs);
};
