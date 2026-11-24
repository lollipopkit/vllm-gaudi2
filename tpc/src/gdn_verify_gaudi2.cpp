// SPDX-License-Identifier: Apache-2.0
#include "gdn_verify_gaudi2.hpp"

#include <cstring>

extern unsigned char _binary___gdn_verify_f32_gaudi2_o_start;
extern unsigned char _binary___gdn_verify_f32_gaudi2_o_end;

namespace
{
enum Input { POOL = 0, KQ, VALUE, COEFF, ROWS, NUM_INPUTS };
enum Output { OUT = 0, NUM_OUTPUTS };

bool Is(const tpc_lib_api::Tensor& t, tpc_lib_api::TensorDataType type, unsigned dims, const uint64_t* sizes)
{
    if (t.geometry.dataType != type || t.geometry.dims != dims)
        return false;
    for (unsigned d = 0; d < dims; d++)
        if (t.geometry.maxSizes[d] != sizes[d])
            return false;
    return true;
}

// [V, HV, T, B]: member (h, b) owns all value columns and tokens of its head.
void MapRows(tpc_lib_api::TensorAccessPattern& p, uint64_t tokens)
{
    p.mapping[0] = {0, 0, 0, GdnVerifyGaudi2::kDim - 1};
    p.mapping[1] = {1, 1, 0, 0};
    p.mapping[2] = {0, 0, 0, static_cast<float>(tokens - 1)};
    p.mapping[3] = {2, 1, 0, 0};
}
}  // namespace

tpc_lib_api::GlueCodeReturn GdnVerifyGaudi2::GetKernelName(char kernelName[tpc_lib_api::MAX_NODE_NAME])
{
    strcpy(kernelName, kGuid);
    return tpc_lib_api::GLUE_SUCCESS;
}

tpc_lib_api::GlueCodeReturn GdnVerifyGaudi2::GetGcDefinitions(tpc_lib_api::HabanaKernelParams* in_defs,
                                                              tpc_lib_api::HabanaKernelInstantiation* out_defs)
{
    if (in_defs->inputTensorNr != NUM_INPUTS)
    {
        in_defs->inputTensorNr = NUM_INPUTS;
        return tpc_lib_api::GLUE_INCOMPATIBLE_INPUT_COUNT;
    }
    if (in_defs->outputTensorNr != NUM_OUTPUTS)
    {
        in_defs->outputTensorNr = NUM_OUTPUTS;
        return tpc_lib_api::GLUE_INCOMPATIBLE_OUTPUT_COUNT;
    }
    const uint64_t* p = in_defs->inputTensors[POOL].geometry.maxSizes;
    const uint64_t* v = in_defs->inputTensors[VALUE].geometry.maxSizes;
    const uint64_t heads = p[2], slots = p[3], tokens = v[2], batch = v[3];
    if (heads % kHeadsPerKey != 0 || tokens == 0)
        return tpc_lib_api::GLUE_INCOMPATIBLE_INPUT_SIZE;
    const uint64_t pool[] = {kDim, kDim, heads, slots};
    const uint64_t kq[] = {kDim, 2, heads / kHeadsPerKey, tokens, batch};
    const uint64_t rows4[] = {kDim, heads, tokens, batch};
    const uint64_t coeff[] = {3, heads, tokens, batch};  // decay, beta, k . q
    const uint64_t rows[] = {tokens + 1, batch};
    const auto f32 = tpc_lib_api::DATA_F32;
    if (!Is(in_defs->inputTensors[POOL], f32, 4, pool) || !Is(in_defs->inputTensors[KQ], f32, 5, kq) ||
        !Is(in_defs->inputTensors[VALUE], f32, 4, rows4) || !Is(in_defs->inputTensors[COEFF], f32, 4, coeff) ||
        !Is(in_defs->inputTensors[ROWS], tpc_lib_api::DATA_I32, 2, rows) ||
        !Is(in_defs->outputTensors[OUT], f32, 4, rows4))
        return tpc_lib_api::GLUE_INCOMPATIBLE_INPUT_SIZE;

    out_defs->indexSpaceRank = 3;
    out_defs->indexSpaceGeometry[0] = 1;
    out_defs->indexSpaceGeometry[1] = heads;
    out_defs->indexSpaceGeometry[2] = batch;

    // Pool rows come from the rows tensor at run time; kq / coeff / rows are small.
    out_defs->inputTensorAccessPattern[POOL].allRequired = true;
    out_defs->inputTensorAccessPattern[KQ].allRequired = true;
    MapRows(out_defs->inputTensorAccessPattern[VALUE], tokens);
    out_defs->inputTensorAccessPattern[COEFF].allRequired = true;
    out_defs->inputTensorAccessPattern[ROWS].allRequired = true;
    MapRows(out_defs->outputTensorAccessPattern[OUT], tokens);

    const unsigned isaSize = &_binary___gdn_verify_f32_gaudi2_o_end - &_binary___gdn_verify_f32_gaudi2_o_start;
    const unsigned given = out_defs->kernel.elfSize;
    out_defs->kernel.elfSize = isaSize;
    if (given < isaSize)
        return tpc_lib_api::GLUE_INSUFFICIENT_ELF_BUFFER;
    memcpy(out_defs->kernel.kernelElf, &_binary___gdn_verify_f32_gaudi2_o_start, isaSize);
    return tpc_lib_api::GLUE_SUCCESS;
}
