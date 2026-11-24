// SPDX-License-Identifier: Apache-2.0
// Shared ABI-identical parameter structs for Inkling 2 Mini Gaudi2 TPC glue code and PT2 custom ops.
#pragma once

namespace inkling2_tpc
{
struct MoeW13Params
{
    float gateCap;
    float upCap;
    float fp8Max;
};

struct MoeGroupExpertsParams
{
    int numExperts;
    int nW13Blks;
    int nW2Blks;
};

struct MoeGateTopkParams
{
    int topK;
    float routeScale;
};

struct MoeSumSconvResParams
{
    float outScale;
    int isFirst;
    int streamOffset;
};

struct AttnResNormBf16Params
{
    int nStack;
    int hasAddend;
    int normalizeValues;
    float eps;
    float normEps;
};

struct AttnResNormF8Params
{
    int nStack;
    int hasAddend;
    int normalizeValues;
    float eps;
    float normEps;
    float fp8Max;
};

struct AttnResF32Params
{
    int nStack;
    int hasAddend;
    int normalizeValues;
    float eps;
};

struct SconvStaticParams
{
    float outScale;
    int streamOffset;
};

struct QkvrPrepParams
{
    int kvColOffset;
    int hasTau;
    int isDynamic;
    float convScale;
    float qkClip;
    float relClip;
    float clipEps;
    float rmsEps;
};

struct PairRmsParams
{
    float eps;
};

struct BlendRmsParams
{
    float normEps;
};

struct BlendRmsF8Params
{
    float normEps;
    float fp8Max;
};

struct RelPagedAttnParams
{
    float scale;
    int slidingWindow;
    int relExtent;
};
}  // namespace inkling2_tpc
