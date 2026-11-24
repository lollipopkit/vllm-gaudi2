// SPDX-License-Identifier: Apache-2.0
// libgaudi2_tpc_proxy.so: our kernels plus every kernel of the stock perf library behind one file, so
// GC_KERNEL_PATH can stay a single path (the recipe cache rejects a ':'-separated list). The stock library
// (GAUDI2_TPC_STOCK_LIB, default /usr/lib/habanalabs/libtpc_kernels.so) is dlopen'ed; the guid, instantiate,
// layout, and shape-inference entry points merge both, every other C entry point jumps straight into it.
#include <dlfcn.h>

#include <cstdio>
#include <cstdlib>

#include "inkling2_kernel_registry.hpp"

using tpc_lib_api::GlueCodeReturn;

namespace
{
void* g_stock;

template <class F>
F Stock(const char* name)
{
    void* sym = dlsym(g_stock, name);
    if (sym == nullptr)
    {
        fprintf(stderr, "gaudi2_tpc_proxy: stock library lacks %s\n", name);
        abort();
    }
    return reinterpret_cast<F>(sym);
}
}  // namespace

// Pure forwards: a tail jump through a pointer resolved at load, so no signature is needed.
// GetSupportedDataLayouts, GetSupportedLayouts, and GetSupportedLayoutsV2 are wrapped below so
// stock kernels (e.g. resize_fwd_f32 for F.interpolate in ViT) retain their required NCHW/NHWC
// layouts while our custom kernels report a wildcard ('x'...) layout-agnostic entry.
#define FORWARDS(X)                                                                                         \
    X(GetClangGitVersion) X(GetKernelISAs) X(GetKernelNames) X(GetKernelSupportingDynamicShapeNames)        \
    X(GetLibVersion) X(GetShapeInferenceFunction) X(GetShapeInferenceLibVersion) X(GetShapeInferenceUniqueId) \
    X(GetSuggestedManipulation) X(GetSuggestedTensorManipulation) X(GetSuggestedTensorManipulationV2)       \
    X(GetVersion) X(HabanaKernel) X(HabanaKernelV2)

#define DECLARE(name) extern "C" __attribute__((visibility("hidden"))) void* g_fwd_##name;
#define DEFINE(name) void* g_fwd_##name;
#define TRAMPOLINE(name)                                                                                    \
    asm(".pushsection .text\n.globl " #name "\n.type " #name ", @function\n" #name ":\n\tjmp *g_fwd_" #name \
        "(%rip)\n.size " #name ", .-" #name "\n.popsection\n");
#define RESOLVE(name)                                                                                       \
    if (!(g_fwd_##name = dlsym(g_stock, #name))) {                                                          \
        fprintf(stderr, "gaudi2_tpc_proxy: stock library lacks %s\n", #name);                                \
        abort();                                                                                            \
    }

FORWARDS(DECLARE)
extern "C" {
FORWARDS(DEFINE)
}
FORWARDS(TRAMPOLINE)

__attribute__((constructor)) static void LoadStock()
{
    const char* path = getenv("GAUDI2_TPC_STOCK_LIB");
    if (path == nullptr)
        path = "/usr/lib/habanalabs/libtpc_kernels.so";
    if ((g_stock = dlopen(path, RTLD_NOW | RTLD_LOCAL)) == nullptr) {
        fprintf(stderr, "gaudi2_tpc_proxy: %s\n", dlerror());
        abort();
    }
    FORWARDS(RESOLVE)
}

extern "C" {

GlueCodeReturn GetKernelGuids(_IN_ tpc_lib_api::DeviceId deviceId, _INOUT_ uint32_t* kernelCount,
                              _OUT_ tpc_lib_api::GuidInfo* guids)
{
    if (kernelCount == nullptr)
        return tpc_lib_api::GLUE_FAILED;
    auto stock = Stock<tpc_lib_api::pfnGetKernelGuids>("GetKernelGuids");
    const uint32_t ours = deviceId == tpc_lib_api::DEVICE_ID_GAUDI2
                              ? static_cast<uint32_t>(gaudi2_tpc_registry::kKernelCount)
                              : 0;
    uint32_t n = 0;
    GlueCodeReturn rc = stock(deviceId, &n, nullptr);
    if (rc != tpc_lib_api::GLUE_SUCCESS)
        return rc;
    if (guids != nullptr) {
        if (*kernelCount < n + ours) {
            *kernelCount = n + ours;
            return tpc_lib_api::GLUE_FAILED;
        }
        if ((rc = stock(deviceId, &n, guids)) != tpc_lib_api::GLUE_SUCCESS)
            return rc;
        if (ours > 0)
            gaudi2_tpc_registry::FillOurGuids(&guids[n]);
    }
    *kernelCount = n + ours;
    return tpc_lib_api::GLUE_SUCCESS;
}

GlueCodeReturn InstantiateTpcKernel(_IN_ tpc_lib_api::HabanaKernelParams* params,
                                    _OUT_ tpc_lib_api::HabanaKernelInstantiation* instance)
{
    GlueCodeReturn rc = tpc_lib_api::GLUE_NODE_NOT_FOUND;
    if (gaudi2_tpc_registry::TryInstantiateOurKernel(params, instance, &rc))
        return rc;
    return Stock<tpc_lib_api::pfnInstantiateTpcKernel>("InstantiateTpcKernel")(params, instance);
}

GlueCodeReturn GetSupportedDataLayouts(_IN_ const tpc_lib_api::HabanaKernelParams* params,
                                       _OUT_ tpc_lib_api::NodeDataLayouts* supportedLayouts,
                                       _INOUT_ uint32_t* layoutCount)
{
    if (params != nullptr && gaudi2_tpc_registry::IsOurGuid(params->guid.name)) {
        if (layoutCount == nullptr)
            return tpc_lib_api::GLUE_FAILED;
        if (supportedLayouts != nullptr) {
            if (*layoutCount < 1) {
                *layoutCount = 1;
                return tpc_lib_api::GLUE_FAILED;
            }
            auto fillWildcard = [](tpc_lib_api::TensorDataLayout& dst, uint32_t dims) {
                memset(dst.layout, 0, sizeof(dst.layout));
                const uint32_t n = (dims > 0 && dims < tpc_lib_api::MAX_TENSOR_DIM) ? dims : 5u;
                memset(dst.layout, 'x', n);
            };
            auto& out = supportedLayouts[0];
            if (out.inputs != nullptr) {
                for (uint32_t i = 0; i < out.inputTensorNr; i++) {
                    const uint32_t dims =
                        (params->inputTensors != nullptr && i < params->inputTensorNr)
                            ? params->inputTensors[i].geometry.dims
                            : 5u;
                    fillWildcard(out.inputs[i], dims);
                }
            }
            if (out.outputs != nullptr) {
                for (uint32_t i = 0; i < out.outputTensorNr; i++) {
                    const uint32_t dims =
                        (params->outputTensors != nullptr && i < params->outputTensorNr)
                            ? params->outputTensors[i].geometry.dims
                            : 5u;
                    fillWildcard(out.outputs[i], dims);
                }
            }
            if (out.shapeTensors != nullptr) {
                for (uint32_t i = 0; i < out.shapeTensorNr; i++)
                    fillWildcard(out.shapeTensors[i], 5u);
            }
        }
        *layoutCount = 1;
        return tpc_lib_api::GLUE_SUCCESS;
    }
    return Stock<tpc_lib_api::pfnGetSupportedDataLayout>("GetSupportedDataLayouts")(
        params, supportedLayouts, layoutCount);
}

gcapi::GlueCodeReturn_t GetSupportedLayouts(_IN_ gcapi::HabanaKernelParams_t* params,
                                            _OUT_ gcapi::NodeDataLayouts* supportedLayouts,
                                            _INOUT_ unsigned* layoutCount)
{
    using PfnV1 = gcapi::GlueCodeReturn_t (*)(gcapi::HabanaKernelParams_t*, gcapi::NodeDataLayouts*, unsigned*);
    if (params != nullptr && gaudi2_tpc_registry::IsOurGuid(params->nodeName)) {
        if (layoutCount == nullptr)
            return gcapi::GLUE_FAILED;
        if (supportedLayouts != nullptr) {
            if (*layoutCount < 1) {
                *layoutCount = 1;
                return gcapi::GLUE_FAILED;
            }
            for (unsigned i = 0; i < gcapi::MAX_TENSOR_NR; i++) {
                memset(supportedLayouts[0].inputs[i].layout, 'x', gcapi::MAX_TENSOR_DIM);
                memset(supportedLayouts[0].outputs[i].layout, 'x', gcapi::MAX_TENSOR_DIM);
            }
        }
        *layoutCount = 1;
        return gcapi::GLUE_SUCCESS;
    }
    return Stock<PfnV1>("GetSupportedLayouts")(params, supportedLayouts, layoutCount);
}

gcapi::GlueCodeReturn_t GetSupportedLayoutsV2(_IN_ gcapi::HabanaKernelParamsV2_t* params,
                                              _OUT_ gcapi::NodeDataLayoutsV2* supportedLayouts,
                                              _INOUT_ unsigned* layoutCount)
{
    if (params != nullptr && gaudi2_tpc_registry::IsOurGuid(params->nodeName)) {
        if (layoutCount == nullptr)
            return gcapi::GLUE_FAILED;
        if (supportedLayouts != nullptr) {
            if (*layoutCount < 1) {
                *layoutCount = 1;
                return gcapi::GLUE_FAILED;
            }
            for (unsigned i = 0; i < gcapi::MAX_TENSOR_NR; i++) {
                memset(supportedLayouts[0].inputs[i].layout, 'x', gcapi::MAX_TENSOR_DIM);
                memset(supportedLayouts[0].outputs[i].layout, 'x', gcapi::MAX_TENSOR_DIM);
                memset(supportedLayouts[0].shapeTensors[i].layout, 'x', gcapi::MAX_TENSOR_DIM);
            }
        }
        *layoutCount = 1;
        return gcapi::GLUE_SUCCESS;
    }
    return Stock<gcapi::pfnGetSupportedDataLayoutV2>("GetSupportedLayoutsV2")(
        params, supportedLayouts, layoutCount);
}

GlueCodeReturn GetShapeInference(tpc_lib_api::DeviceId deviceId, tpc_lib_api::ShapeInferenceParams* in,
                                 tpc_lib_api::ShapeInferenceOutput* out)
{
    if (in != nullptr) {
        const char* guid = in->pGuid != nullptr ? in->pGuid->name : in->guid.name;
        if (gaudi2_tpc_registry::IsOurGuid(guid))
            return tpc_lib_api::GLUE_SUCCESS;  // static shapes only
    }
    return Stock<tpc_lib_api::pfnGetShapeInference>("GetShapeInference")(deviceId, in, out);
}

}  // extern "C"
