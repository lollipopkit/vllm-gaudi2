// SPDX-License-Identifier: Apache-2.0
// Shared parameter POD structs between Gaudi2 FP8/GDN/LFM3 TPC glue code and PT2 custom ops.
#pragma once

namespace gaudi2_fp8_params
{
struct AddRmsNormF8Params
{
    float eps;
    float fp8Max;
};

struct SiluMulF8Params
{
    float fp8Max;
};

struct GdnPrepDecodeParams
{
    int rowOffset;
    int convColOffset;
    int hasBias;
};

struct GdnGatedRmsNormF8Params
{
    float eps;
    float fp8Max;
};

struct Lfm3ShortConvDecodeParams
{
    int hasBias;
};
}  // namespace gaudi2_fp8_params
