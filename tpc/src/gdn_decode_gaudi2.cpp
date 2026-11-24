// SPDX-License-Identifier: Apache-2.0
#include "gdn_decode_gaudi2.hpp"

#include <cstring>
#include <initializer_list>

extern unsigned char _binary___gdn_decode_f32_gaudi2_o_start;
extern unsigned char _binary___gdn_decode_f32_gaudi2_o_end;

namespace
{
enum Input { POOL = 0, QK, VALUE, DECAY, BETA, KQDOT, NUM_INPUTS };
enum Output { OUT = 0, NUM_OUTPUTS };

using tpc_lib_api::DimIndexSpaceMapping;

void Map(DimIndexSpaceMapping& m, unsigned isDim, float a, float startB, float endB)
{
    m.indexSpaceDim = isDim;
    m.a = a;
    m.start_b = startB;
    m.end_b = endB;
}

// [V, K, HV, slots]: every member reads the full K extent of its kColumns value columns, in slot offset + b.
void MapPool(tpc_lib_api::TensorAccessPattern& p, int offset)
{
    Map(p.mapping[0], 0, GdnDecodeGaudi2::kColumns, 0, GdnDecodeGaudi2::kColumns - 1);
    Map(p.mapping[1], 0, 0, 0, GdnDecodeGaudi2::kDim - 1);
    Map(p.mapping[2], 1, 1, 0, 0);
    Map(p.mapping[3], 2, 1, offset, offset);
}

// [V, HV, B]
void MapRows(tpc_lib_api::TensorAccessPattern& p)
{
    Map(p.mapping[0], 0, GdnDecodeGaudi2::kColumns, 0, GdnDecodeGaudi2::kColumns - 1);
    Map(p.mapping[1], 1, 1, 0, 0);
    Map(p.mapping[2], 2, 1, 0, 0);
}

bool Is(const tpc_lib_api::Tensor& t, unsigned dims, const uint64_t* sizes)
{
    if (t.geometry.dataType != tpc_lib_api::DATA_F32 || t.geometry.dims != dims)
        return false;
    for (unsigned d = 0; d < dims; d++)
        if (t.geometry.maxSizes[d] != sizes[d])
            return false;
    return true;
}
}  // namespace

tpc_lib_api::GlueCodeReturn GdnDecodeGaudi2::GetKernelName(char kernelName[tpc_lib_api::MAX_NODE_NAME])
{
    strcpy(kernelName, kGuid);
    return tpc_lib_api::GLUE_SUCCESS;
}

tpc_lib_api::GlueCodeReturn GdnDecodeGaudi2::GetGcDefinitions(tpc_lib_api::HabanaKernelParams* in_defs,
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
    if (in_defs->nodeParams.nodeParams == nullptr)
        return tpc_lib_api::GLUE_INCOMPATIBLE_INPUT_SIZE;
    const auto* params = static_cast<const Params*>(in_defs->nodeParams.nodeParams);
    const int offset = params->rowOffset;
    const uint64_t* p = in_defs->inputTensors[POOL].geometry.maxSizes;
    const uint64_t* v = in_defs->inputTensors[VALUE].geometry.maxSizes;
    const uint64_t heads = p[2], slots = p[3], batch = v[2], keys = heads / kHeadsPerKey;
    if (heads % kHeadsPerKey != 0 || offset < 0 || offset + batch > slots)
        return tpc_lib_api::GLUE_INCOMPATIBLE_INPUT_SIZE;
    const uint64_t pool[] = {kDim, kDim, heads, slots};
    const uint64_t qk[] = {kDim, 2 * keys, batch};
    const uint64_t rows[] = {kDim, heads, batch};
    const uint64_t perHead[] = {heads, batch};
    const uint64_t perKey[] = {keys, batch};
    if (!Is(in_defs->inputTensors[POOL], 4, pool) || !Is(in_defs->inputTensors[QK], 3, qk) ||
        !Is(in_defs->inputTensors[VALUE], 3, rows) || !Is(in_defs->inputTensors[DECAY], 2, perHead) ||
        !Is(in_defs->inputTensors[BETA], 2, perHead) || !Is(in_defs->inputTensors[KQDOT], 2, perKey) ||
        !Is(in_defs->outputTensors[OUT], 3, rows))
        return tpc_lib_api::GLUE_INCOMPATIBLE_INPUT_SIZE;

    out_defs->indexSpaceRank = 3;
    out_defs->indexSpaceGeometry[0] = kDim / kColumns;
    out_defs->indexSpaceGeometry[1] = heads;
    out_defs->indexSpaceGeometry[2] = batch;

    MapPool(out_defs->inputTensorAccessPattern[POOL], offset);
    // Small per-batch tensors; key head = value head / 3.
    for (int t : {QK, DECAY, BETA, KQDOT})
        out_defs->inputTensorAccessPattern[t].allRequired = true;
    MapRows(out_defs->inputTensorAccessPattern[VALUE]);
    MapRows(out_defs->outputTensorAccessPattern[OUT]);

    out_defs->kernel.paramsNr = 2;
    out_defs->kernel.scalarParams[0] = offset;
    memcpy(&out_defs->kernel.scalarParams[1], &params->scale, sizeof(float));

    const unsigned isaSize = &_binary___gdn_decode_f32_gaudi2_o_end - &_binary___gdn_decode_f32_gaudi2_o_start;
    const unsigned given = out_defs->kernel.elfSize;
    out_defs->kernel.elfSize = isaSize;
    if (given < isaSize)
        return tpc_lib_api::GLUE_INSUFFICIENT_ELF_BUFFER;
    memcpy(out_defs->kernel.kernelElf, &_binary___gdn_decode_f32_gaudi2_o_start, isaSize);
    return tpc_lib_api::GLUE_SUCCESS;
}
