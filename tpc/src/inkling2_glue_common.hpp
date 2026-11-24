// SPDX-License-Identifier: Apache-2.0
// Shared validation, tensor-access-pattern, and ELF-copy helpers for Inkling 2 Mini Gaudi2 TPC glue code.
#pragma once

#include <cmath>
#include <cstring>

#include "tpc_kernel_lib_interface.h"

namespace inkling2_glue
{
inline bool Is(const tpc_lib_api::Tensor& t, tpc_lib_api::TensorDataType dt, unsigned dims, const uint64_t* sizes)
{
    if (t.geometry.dataType != dt || t.geometry.dims != dims)
        return false;
    for (unsigned d = 0; d < dims; d++)
        if (t.geometry.maxSizes[d] != sizes[d])
            return false;
    return true;
}

inline void Map(tpc_lib_api::DimIndexSpaceMapping& m, unsigned isDim, float a, float startB, float endB)
{
    m.indexSpaceDim = isDim;
    m.a = a;
    m.start_b = startB;
    m.end_b = endB;
}

inline bool IsPosFinite(float v)
{
    return std::isfinite(v) && v > 0.0f;
}

inline tpc_lib_api::GlueCodeReturn CopyElf(tpc_lib_api::HabanaKernelInstantiation* out_defs,
                                           const unsigned char* start, const unsigned char* end)
{
    const unsigned isaSize = static_cast<unsigned>(end - start);
    const unsigned given = out_defs->kernel.elfSize;
    out_defs->kernel.elfSize = isaSize;
    if (out_defs->kernel.kernelElf == nullptr || given < isaSize)
        return tpc_lib_api::GLUE_INSUFFICIENT_ELF_BUFFER;
    memcpy(out_defs->kernel.kernelElf, start, isaSize);
    return tpc_lib_api::GLUE_SUCCESS;
}
}  // namespace inkling2_glue
