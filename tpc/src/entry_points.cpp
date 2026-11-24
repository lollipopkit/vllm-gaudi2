// SPDX-License-Identifier: Apache-2.0
// Perf-lib entry points of libgaudi2_tpc_kernels.so (load with GC_KERNEL_PATH).
#include "inkling2_kernel_registry.hpp"

extern "C" {

tpc_lib_api::GlueCodeReturn GetKernelGuids(_IN_ tpc_lib_api::DeviceId deviceId, _INOUT_ uint32_t* kernelCount,
                                           _OUT_ tpc_lib_api::GuidInfo* guids)
{
    if (kernelCount == nullptr)
        return tpc_lib_api::GLUE_FAILED;
    const uint32_t count = deviceId == tpc_lib_api::DEVICE_ID_GAUDI2
                               ? static_cast<uint32_t>(gaudi2_tpc_registry::kKernelCount)
                               : 0;
    if (guids != nullptr)
    {
        if (*kernelCount < count)
        {
            *kernelCount = count;
            return tpc_lib_api::GLUE_FAILED;
        }
        if (count > 0)
            gaudi2_tpc_registry::FillOurGuids(guids);
    }
    *kernelCount = count;
    return tpc_lib_api::GLUE_SUCCESS;
}

tpc_lib_api::GlueCodeReturn InstantiateTpcKernel(_IN_ tpc_lib_api::HabanaKernelParams* params,
                                                 _OUT_ tpc_lib_api::HabanaKernelInstantiation* instance)
{
    tpc_lib_api::GlueCodeReturn rc = tpc_lib_api::GLUE_NODE_NOT_FOUND;
    if (gaudi2_tpc_registry::TryInstantiateOurKernel(params, instance, &rc))
        return rc;
    return tpc_lib_api::GLUE_NODE_NOT_FOUND;
}

tpc_lib_api::GlueCodeReturn GetShapeInference(tpc_lib_api::DeviceId, tpc_lib_api::ShapeInferenceParams*,
                                              tpc_lib_api::ShapeInferenceOutput*)
{
    return tpc_lib_api::GLUE_SUCCESS;  // static shapes only
}

}  // extern "C"
