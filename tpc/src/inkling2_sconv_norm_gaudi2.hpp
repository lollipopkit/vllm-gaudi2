// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "inkling2_params.hpp"
#include "tpc_kernel_lib_interface.h"

class Inkling2SconvStaticDecodeGaudi2
{
public:
    static constexpr const char* kGuid = "inkling2_sconv_static_decode_gaudi2";
    using Params = inkling2_tpc::SconvStaticParams;
    tpc_lib_api::GlueCodeReturn GetKernelName(char kernelName[tpc_lib_api::MAX_NODE_NAME]);
    tpc_lib_api::GlueCodeReturn GetGcDefinitions(tpc_lib_api::HabanaKernelParams* in_defs,
                                                 tpc_lib_api::HabanaKernelInstantiation* out_defs);
};

class Inkling2QkvrPrepDecodeGaudi2
{
public:
    static constexpr const char* kGuid = "inkling2_qkvr_prep_decode_gaudi2";
    using Params = inkling2_tpc::QkvrPrepParams;
    tpc_lib_api::GlueCodeReturn GetKernelName(char kernelName[tpc_lib_api::MAX_NODE_NAME]);
    tpc_lib_api::GlueCodeReturn GetGcDefinitions(tpc_lib_api::HabanaKernelParams* in_defs,
                                                 tpc_lib_api::HabanaKernelInstantiation* out_defs);
};

class Inkling2PairRmsNormGaudi2
{
public:
    static constexpr const char* kGuid = "inkling2_pair_rms_norm_gaudi2";
    using Params = inkling2_tpc::PairRmsParams;
    tpc_lib_api::GlueCodeReturn GetKernelName(char kernelName[tpc_lib_api::MAX_NODE_NAME]);
    tpc_lib_api::GlueCodeReturn GetGcDefinitions(tpc_lib_api::HabanaKernelParams* in_defs,
                                                 tpc_lib_api::HabanaKernelInstantiation* out_defs);
};

class Inkling2BlendRmsNormGaudi2
{
public:
    static constexpr const char* kGuid = "inkling2_blend_rms_norm_gaudi2";
    using Params = inkling2_tpc::BlendRmsParams;
    tpc_lib_api::GlueCodeReturn GetKernelName(char kernelName[tpc_lib_api::MAX_NODE_NAME]);
    tpc_lib_api::GlueCodeReturn GetGcDefinitions(tpc_lib_api::HabanaKernelParams* in_defs,
                                                 tpc_lib_api::HabanaKernelInstantiation* out_defs);
};

class Inkling2BlendRmsNormF8Gaudi2
{
public:
    static constexpr const char* kGuid = "inkling2_blend_rms_norm_f8_gaudi2";
    using Params = inkling2_tpc::BlendRmsF8Params;
    tpc_lib_api::GlueCodeReturn GetKernelName(char kernelName[tpc_lib_api::MAX_NODE_NAME]);
    tpc_lib_api::GlueCodeReturn GetGcDefinitions(tpc_lib_api::HabanaKernelParams* in_defs,
                                                 tpc_lib_api::HabanaKernelInstantiation* out_defs);
};
