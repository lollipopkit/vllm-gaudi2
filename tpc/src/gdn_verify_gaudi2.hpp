// SPDX-License-Identifier: Apache-2.0
// Glue code for the Gated DeltaNet spec-decode verification kernel (gdn_verify_f32_gaudi2, KV-major state).
#pragma once

#include "gc_interface.h"
#include "tpc_kernel_lib_interface.h"

class GdnVerifyGaudi2
{
public:
    static constexpr const char* kGuid = "gdn_verify_f32_gaudi2";
    static constexpr int kDim = 128;        // key / value head dim
    static constexpr int kHeadsPerKey = 3;  // value heads per key head (kernel HEADS_PER_KEY)

    tpc_lib_api::GlueCodeReturn GetKernelName(char kernelName[tpc_lib_api::MAX_NODE_NAME]);
    tpc_lib_api::GlueCodeReturn GetGcDefinitions(tpc_lib_api::HabanaKernelParams* in_defs,
                                                 tpc_lib_api::HabanaKernelInstantiation* out_defs);
};
