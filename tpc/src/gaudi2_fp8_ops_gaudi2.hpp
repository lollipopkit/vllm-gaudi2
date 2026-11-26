// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "gaudi2_fp8_ops_params.hpp"
#include "tpc_kernel_lib_interface.h"

class Gaudi2AddRmsNormF8Gaudi2
{
public:
    static constexpr const char* kGuid = "gaudi2_add_rms_norm_f8_gaudi2";
    static constexpr int kMaxDim = 12288;
    using Params = gaudi2_fp8_params::AddRmsNormF8Params;

    tpc_lib_api::GlueCodeReturn GetKernelName(char kernelName[tpc_lib_api::MAX_NODE_NAME]);
    tpc_lib_api::GlueCodeReturn GetGcDefinitions(tpc_lib_api::HabanaKernelParams* in_defs,
                                                 tpc_lib_api::HabanaKernelInstantiation* out_defs);
};

class Gaudi2SandwichRmsNormF8Gaudi2
{
public:
    static constexpr const char* kGuid = "gaudi2_sandwich_rms_norm_f8_gaudi2";
    static constexpr int kMaxDim = 12288;
    using Params = gaudi2_fp8_params::AddRmsNormF8Params;

    tpc_lib_api::GlueCodeReturn GetKernelName(char kernelName[tpc_lib_api::MAX_NODE_NAME]);
    tpc_lib_api::GlueCodeReturn GetGcDefinitions(tpc_lib_api::HabanaKernelParams* in_defs,
                                                 tpc_lib_api::HabanaKernelInstantiation* out_defs);
};

class Gaudi2SiluMulF8Gaudi2
{
public:
    static constexpr const char* kGuid = "gaudi2_silu_mul_f8_gaudi2";
    static constexpr int kMaxDim = 28672;
    using Params = gaudi2_fp8_params::SiluMulF8Params;

    tpc_lib_api::GlueCodeReturn GetKernelName(char kernelName[tpc_lib_api::MAX_NODE_NAME]);
    tpc_lib_api::GlueCodeReturn GetGcDefinitions(tpc_lib_api::HabanaKernelParams* in_defs,
                                                 tpc_lib_api::HabanaKernelInstantiation* out_defs);
};

class GdnPrepDecodeGaudi2
{
public:
    static constexpr const char* kGuid = "gdn_prep_decode_gaudi2";
    static constexpr int kHeadDim = 128;
    using Params = gaudi2_fp8_params::GdnPrepDecodeParams;

    tpc_lib_api::GlueCodeReturn GetKernelName(char kernelName[tpc_lib_api::MAX_NODE_NAME]);
    tpc_lib_api::GlueCodeReturn GetGcDefinitions(tpc_lib_api::HabanaKernelParams* in_defs,
                                                 tpc_lib_api::HabanaKernelInstantiation* out_defs);
};

class GdnGatedRmsNormF8Gaudi2
{
public:
    static constexpr const char* kGuid = "gdn_gated_rms_norm_f8_gaudi2";
    static constexpr int kHeadDim = 128;
    static constexpr int kMaxValueHeads = 128;
    using Params = gaudi2_fp8_params::GdnGatedRmsNormF8Params;

    tpc_lib_api::GlueCodeReturn GetKernelName(char kernelName[tpc_lib_api::MAX_NODE_NAME]);
    tpc_lib_api::GlueCodeReturn GetGcDefinitions(tpc_lib_api::HabanaKernelParams* in_defs,
                                                 tpc_lib_api::HabanaKernelInstantiation* out_defs);
};

class Lfm3ShortConvDecodeGaudi2
{
public:
    static constexpr const char* kGuid = "lfm3_short_conv_decode_gaudi2";
    using Params = gaudi2_fp8_params::Lfm3ShortConvDecodeParams;

    tpc_lib_api::GlueCodeReturn GetKernelName(char kernelName[tpc_lib_api::MAX_NODE_NAME]);
    tpc_lib_api::GlueCodeReturn GetGcDefinitions(tpc_lib_api::HabanaKernelParams* in_defs,
                                                 tpc_lib_api::HabanaKernelInstantiation* out_defs);
};
