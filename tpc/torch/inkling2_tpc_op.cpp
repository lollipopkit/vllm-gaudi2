// SPDX-License-Identifier: Apache-2.0
// PT2 custom ops running Inkling 2 Mini custom Gaudi2 TPC kernels inside torch.compile(backend="hpu_backend").
// Requires libgaudi2_tpc_proxy.so (or libgaudi2_tpc_kernels.so) on GC_KERNEL_PATH.
#include <torch/extension.h>

#include <tuple>

#include "hpu_custom_op_pt2.h"
#include "inkling2_params.hpp"

namespace
{
// Keep all ops in the "custom_op" namespace so torch.compile(backend="hpu_backend")
// fuses them inside compiled graphs without eager fallbacks.
constexpr const char* kW13Schema = "custom_op::inkling2_moe_w13_situ_f8";
constexpr const char* kW13Guid = "inkling2_moe_w13_situ_f8_gaudi2";
constexpr const char* kW2Schema = "custom_op::inkling2_moe_w2_f8";
constexpr const char* kW2Guid = "inkling2_moe_w2_f8_gaudi2";
constexpr const char* kGroupExpertsSchema = "custom_op::inkling2_moe_group_experts";
constexpr const char* kGroupExpertsGuid = "inkling2_moe_group_experts_gaudi2";
constexpr const char* kW13GroupedSchema = "custom_op::inkling2_moe_w13_grouped_f8";
constexpr const char* kW13GroupedGuid = "inkling2_moe_w13_grouped_f8_gaudi2";
constexpr const char* kW2GroupedSchema = "custom_op::inkling2_moe_w2_grouped_f8";
constexpr const char* kW2GroupedGuid = "inkling2_moe_w2_grouped_f8_gaudi2";
constexpr const char* kAttnResNormSchema = "custom_op::inkling2_attn_res_norm_bf16";
constexpr const char* kAttnResNormGuid = "inkling2_attn_res_norm_bf16_gaudi2";
constexpr const char* kAttnResNormF8Schema = "custom_op::inkling2_attn_res_norm_f8";
constexpr const char* kAttnResNormF8Guid = "inkling2_attn_res_norm_f8_gaudi2";
constexpr const char* kAttnResF32Schema = "custom_op::inkling2_attn_res_f32";
constexpr const char* kAttnResF32Guid = "inkling2_attn_res_f32_gaudi2";
constexpr const char* kSconvStaticSchema = "custom_op::inkling2_sconv_static_decode";
constexpr const char* kSconvStaticGuid = "inkling2_sconv_static_decode_gaudi2";
constexpr const char* kQkvrPrepSchema = "custom_op::inkling2_qkvr_prep_decode";
constexpr const char* kQkvrPrepGuid = "inkling2_qkvr_prep_decode_gaudi2";
constexpr const char* kPairRmsSchema = "custom_op::inkling2_pair_rms_norm";
constexpr const char* kPairRmsGuid = "inkling2_pair_rms_norm_gaudi2";
constexpr const char* kBlendRmsSchema = "custom_op::inkling2_blend_rms_norm";
constexpr const char* kBlendRmsGuid = "inkling2_blend_rms_norm_gaudi2";
constexpr const char* kBlendRmsF8Schema = "custom_op::inkling2_blend_rms_norm_f8";
constexpr const char* kBlendRmsF8Guid = "inkling2_blend_rms_norm_f8_gaudi2";
constexpr const char* kPagedAttnSchema = "custom_op::inkling2_rel_paged_attn_gate_sink";
constexpr const char* kPagedAttnGuid = "inkling2_rel_paged_attn_gate_sink_gaudi2";
constexpr const char* kMoeGateTopkSchema = "custom_op::inkling2_moe_gate_topk";
constexpr const char* kMoeGateTopkGuid = "inkling2_moe_gate_topk_gaudi2";
constexpr const char* kMoeGateGemvSplitkSchema = "custom_op::inkling2_moe_gate_gemv_splitk";
constexpr const char* kMoeGateGemvSplitkGuid = "inkling2_moe_gate_gemv_splitk_gaudi2";
constexpr const char* kMoeSumSconvResSchema = "custom_op::inkling2_moe_sum_sconv_res_decode";
constexpr const char* kMoeSumSconvResGuid = "inkling2_moe_sum_sconv_res_decode_gaudi2";

using W13Params = inkling2_tpc::MoeW13Params;
using GroupExpertsParams = inkling2_tpc::MoeGroupExpertsParams;
struct EmptyParams { int unused; };
using MoeGateTopkParams = inkling2_tpc::MoeGateTopkParams;
using MoeSumSconvResParams = inkling2_tpc::MoeSumSconvResParams;
using AttnResNormParams = inkling2_tpc::AttnResNormBf16Params;
using AttnResNormF8Params = inkling2_tpc::AttnResNormF8Params;
using AttnResF32Params = inkling2_tpc::AttnResF32Params;
using SconvStaticParams = inkling2_tpc::SconvStaticParams;
using QkvrPrepParams = inkling2_tpc::QkvrPrepParams;
using PairRmsParams = inkling2_tpc::PairRmsParams;
using BlendRmsParams = inkling2_tpc::BlendRmsParams;
using BlendRmsF8Params = inkling2_tpc::BlendRmsF8Params;
using PagedAttnParams = inkling2_tpc::RelPagedAttnParams;

// ---- Output meta & param fillers ----

habana::PartialOutputMetaDataVector W13OutputMeta(const at::Stack& inputs)
{
    const auto w13 = inputs[2].toTensor();
    const auto ids = inputs[4].toTensor();
    const int64_t total = ids.size(0) * ids.size(1) * w13.size(1);
    return {{c10::ScalarType::Float8_e4m3fn, {total, 128}}, {c10::ScalarType::Float, {total}}};
}

std::shared_ptr<void> FillW13Params(const at::Stack& inputs, size_t& size)
{
    HPU_PARAMS_STUB(W13Params);
    params->gateCap = static_cast<float>(inputs[6].toDouble());
    params->upCap = static_cast<float>(inputs[7].toDouble());
    params->fp8Max = static_cast<float>(inputs[8].toDouble());
    return params;
}

habana::PartialOutputMetaDataVector W2OutputMeta(const at::Stack& inputs)
{
    const auto w2 = inputs[2].toTensor();
    const auto ids = inputs[4].toTensor();
    const int64_t total = ids.size(0) * ids.size(1) * w2.size(1);
    return {{c10::ScalarType::BFloat16, {total, 256}}};
}

habana::PartialOutputMetaDataVector GroupExpertsOutputMeta(const at::Stack&)
{
    return {{c10::ScalarType::Int, {8, 128}}, {c10::ScalarType::Int, {2}}};
}

std::shared_ptr<void> FillGroupExpertsParams(const at::Stack& inputs, size_t& size)
{
    HPU_PARAMS_STUB(GroupExpertsParams);
    params->numExperts = static_cast<int>(inputs[1].toInt());
    params->nW13Blks = static_cast<int>(inputs[2].toInt());
    params->nW2Blks = static_cast<int>(inputs[3].toInt());
    return params;
}

habana::PartialOutputMetaDataVector W13GroupedOutputMeta(const at::Stack& inputs)
{
    const auto w13 = inputs[2].toTensor();
    const auto tw = inputs[4].toTensor();
    const int64_t total = tw.size(0) * tw.size(1) * w13.size(1);
    return {{c10::ScalarType::Float8_e4m3fn, {total, 128}}, {c10::ScalarType::Float, {total}}};
}

std::shared_ptr<void> FillW13GroupedParams(const at::Stack& inputs, size_t& size)
{
    HPU_PARAMS_STUB(W13Params);
    params->gateCap = static_cast<float>(inputs[7].toDouble());
    params->upCap = static_cast<float>(inputs[8].toDouble());
    params->fp8Max = static_cast<float>(inputs[9].toDouble());
    return params;
}

habana::PartialOutputMetaDataVector W2GroupedOutputMeta(const at::Stack& inputs)
{
    const auto act = inputs[0].toTensor();
    const auto w2 = inputs[2].toTensor();
    const int64_t total = act.size(0) * act.size(1) * w2.size(1);
    return {{c10::ScalarType::BFloat16, {total, 256}}};
}

std::shared_ptr<void> FillNoParams(const at::Stack&, size_t& size)
{
    HPU_PARAMS_STUB(EmptyParams);
    params->unused = 0;
    return params;
}

habana::PartialOutputMetaDataVector AttnResNormOutputMeta(const at::Stack& inputs)
{
    const auto last = inputs[1].toTensor();     // [T, D]
    const auto queries = inputs[3].toTensor();  // [Q, D]
    return {{c10::ScalarType::BFloat16, {queries.size(0), last.size(0), last.size(1)}}};
}

std::shared_ptr<void> FillAttnResNormParams(const at::Stack& inputs, size_t& size)
{
    HPU_PARAMS_STUB(AttnResNormParams);
    params->nStack = static_cast<int>(inputs[5].toInt());
    params->hasAddend = static_cast<int>(inputs[6].toInt());
    params->normalizeValues = static_cast<int>(inputs[7].toInt());
    params->eps = static_cast<float>(inputs[8].toDouble());
    params->normEps = static_cast<float>(inputs[9].toDouble());
    return params;
}

habana::PartialOutputMetaDataVector AttnResNormF8OutputMeta(const at::Stack& inputs)
{
    const auto last = inputs[1].toTensor();     // [T, D]
    const auto queries = inputs[3].toTensor();  // [Q, D]
    return {{c10::ScalarType::BFloat16, {queries.size(0), last.size(0), last.size(1)}},
            {c10::ScalarType::Float8_e4m3fn, {queries.size(0), last.size(0), last.size(1)}},
            {c10::ScalarType::Float, {queries.size(0), last.size(0)}}};
}

std::shared_ptr<void> FillAttnResNormF8Params(const at::Stack& inputs, size_t& size)
{
    HPU_PARAMS_STUB(AttnResNormF8Params);
    params->nStack = static_cast<int>(inputs[5].toInt());
    params->hasAddend = static_cast<int>(inputs[6].toInt());
    params->normalizeValues = static_cast<int>(inputs[7].toInt());
    params->eps = static_cast<float>(inputs[8].toDouble());
    params->normEps = static_cast<float>(inputs[9].toDouble());
    params->fp8Max = static_cast<float>(inputs[10].toDouble());
    return params;
}

habana::PartialOutputMetaDataVector AttnResF32OutputMeta(const at::Stack& inputs)
{
    const auto last = inputs[1].toTensor();     // [T, D]
    const auto queries = inputs[3].toTensor();  // [Q, D]
    return {{c10::ScalarType::Float, {queries.size(0), last.size(0), last.size(1)}}};
}

std::shared_ptr<void> FillAttnResF32Params(const at::Stack& inputs, size_t& size)
{
    HPU_PARAMS_STUB(AttnResF32Params);
    params->nStack = static_cast<int>(inputs[4].toInt());
    params->hasAddend = static_cast<int>(inputs[5].toInt());
    params->normalizeValues = static_cast<int>(inputs[6].toInt());
    params->eps = static_cast<float>(inputs[7].toDouble());
    return params;
}

habana::PartialOutputMetaDataVector SconvStaticOutputMeta(const at::Stack& inputs)
{
    const auto x = inputs[0].toTensor();  // [B, C]
    return {{c10::ScalarType::BFloat16, {x.size(0), x.size(1)}},
            {c10::ScalarType::BFloat16, {x.size(0), 3, x.size(1)}}};
}

std::shared_ptr<void> FillSconvStaticParams(const at::Stack& inputs, size_t& size)
{
    HPU_PARAMS_STUB(SconvStaticParams);
    params->outScale = static_cast<float>(inputs[4].toDouble());
    params->streamOffset = static_cast<int>(inputs[5].toInt());
    return params;
}

habana::PartialOutputMetaDataVector QkvrPrepOutputMeta(const at::Stack& inputs)
{
    const auto proj_q = inputs[0].toTensor();   // [B, Q_cols]
    const auto q_scale = inputs[1].toTensor();  // [H, 128]
    const auto r_scale = inputs[2].toTensor();  // [H, d_rel]
    const auto kvw_t = inputs[7].toTensor();    // [4, 2, H_kv, 128]
    const int64_t b = proj_q.size(0), h = q_scale.size(0), d = q_scale.size(1);
    const int64_t d_rel = r_scale.size(1), hkv = kvw_t.size(2);
    return {{c10::ScalarType::BFloat16, {b, h, d}},
            {c10::ScalarType::BFloat16, {b, h, d_rel}},
            {c10::ScalarType::BFloat16, {b, hkv, d}},
            {c10::ScalarType::BFloat16, {b, hkv, d}},
            {c10::ScalarType::BFloat16, {b, 3, 2, hkv, d}}};
}

std::shared_ptr<void> FillQkvrPrepParams(const at::Stack& inputs, size_t& size)
{
    HPU_PARAMS_STUB(QkvrPrepParams);
    params->kvColOffset = static_cast<int>(inputs[9].toInt());
    params->hasTau = static_cast<int>(inputs[10].toInt());
    params->isDynamic = static_cast<int>(inputs[11].toInt());
    params->convScale = static_cast<float>(inputs[12].toDouble());
    params->qkClip = static_cast<float>(inputs[13].toDouble());
    params->relClip = static_cast<float>(inputs[14].toDouble());
    params->clipEps = static_cast<float>(inputs[15].toDouble());
    params->rmsEps = static_cast<float>(inputs[16].toDouble());
    return params;
}

habana::PartialOutputMetaDataVector PairRmsOutputMeta(const at::Stack& inputs)
{
    const auto dest = inputs[0].toTensor();  // [S, T, D]
    return {{c10::ScalarType::BFloat16, {dest.size(0), dest.size(1), 2 * dest.size(2)}}};
}

std::shared_ptr<void> FillPairRmsParams(const at::Stack& inputs, size_t& size)
{
    HPU_PARAMS_STUB(PairRmsParams);
    params->eps = static_cast<float>(inputs[2].toDouble());
    return params;
}

habana::PartialOutputMetaDataVector BlendRmsOutputMeta(const at::Stack& inputs)
{
    const auto dest = inputs[0].toTensor();  // [S, T, D]
    return {{c10::ScalarType::BFloat16, {dest.size(0), dest.size(1), dest.size(2)}}};
}

std::shared_ptr<void> FillBlendRmsParams(const at::Stack& inputs, size_t& size)
{
    HPU_PARAMS_STUB(BlendRmsParams);
    params->normEps = static_cast<float>(inputs[4].toDouble());
    return params;
}

habana::PartialOutputMetaDataVector BlendRmsF8OutputMeta(const at::Stack& inputs)
{
    const auto dest = inputs[0].toTensor();  // [S, T, D]
    return {{c10::ScalarType::BFloat16, {dest.size(0), dest.size(1), dest.size(2)}},
            {c10::ScalarType::Float8_e4m3fn, {dest.size(0), dest.size(1), dest.size(2)}},
            {c10::ScalarType::Float, {dest.size(0), dest.size(1)}}};
}

std::shared_ptr<void> FillBlendRmsF8Params(const at::Stack& inputs, size_t& size)
{
    HPU_PARAMS_STUB(BlendRmsF8Params);
    params->normEps = static_cast<float>(inputs[4].toDouble());
    params->fp8Max = static_cast<float>(inputs[5].toDouble());
    return params;
}

habana::PartialOutputMetaDataVector PagedAttnOutputMeta(const at::Stack& inputs)
{
    const auto q = inputs[0].toTensor();  // [B, H, 128]
    return {{c10::ScalarType::BFloat16, {q.size(0), q.size(1), q.size(2)}}};
}

std::shared_ptr<void> FillPagedAttnParams(const at::Stack& inputs, size_t& size)
{
    HPU_PARAMS_STUB(PagedAttnParams);
    params->scale = static_cast<float>(inputs[13].toDouble());
    params->slidingWindow = static_cast<int>(inputs[14].toInt());
    params->relExtent = static_cast<int>(inputs[15].toInt());
    return params;
}

habana::PartialOutputMetaDataVector MoeGateTopkOutputMeta(const at::Stack& inputs)
{
    const auto logits = inputs[0].toTensor();  // [B, S, E]
    const int64_t b = logits.size(0);
    const int64_t topK = inputs[2].toInt();
    return {{c10::ScalarType::Int, {b, topK}}, {c10::ScalarType::Float, {b, topK}}};
}

std::shared_ptr<void> FillMoeGateTopkParams(const at::Stack& inputs, size_t& size)
{
    HPU_PARAMS_STUB(MoeGateTopkParams);
    params->topK = static_cast<int>(inputs[2].toInt());
    params->routeScale = static_cast<float>(inputs[3].toDouble());
    return params;
}

habana::PartialOutputMetaDataVector MoeGateGemvSplitkOutputMeta(const at::Stack& inputs)
{
    const auto x = inputs[0].toTensor();  // [B, H]
    const int64_t b = x.size(0), s = x.size(1) / 128;
    return {{c10::ScalarType::Float, {b, s, 128}}};
}

habana::PartialOutputMetaDataVector MoeSumSconvResOutputMeta(const at::Stack& inputs)
{
    const auto partial_res = inputs[4].toTensor();  // [B, H]
    return {{c10::ScalarType::BFloat16, {partial_res.size(0), partial_res.size(1)}},
            {c10::ScalarType::BFloat16, {partial_res.size(0), 3, partial_res.size(1)}}};
}

std::shared_ptr<void> FillMoeSumSconvResParams(const at::Stack& inputs, size_t& size)
{
    HPU_PARAMS_STUB(MoeSumSconvResParams);
    params->outScale = static_cast<float>(inputs[6].toDouble());
    params->isFirst = static_cast<int>(inputs[7].toInt());
    params->streamOffset = static_cast<int>(inputs[8].toInt());
    return params;
}

bool Register()
{
    habana::custom_op::registerUserCustomOp(kW13Schema, kW13Guid, W13OutputMeta, FillW13Params);
    habana::custom_op::registerUserCustomOp(kW2Schema, kW2Guid, W2OutputMeta, FillNoParams);
    habana::custom_op::registerUserCustomOp(kGroupExpertsSchema, kGroupExpertsGuid, GroupExpertsOutputMeta,
                                            FillGroupExpertsParams);
    habana::custom_op::registerUserCustomOp(kW13GroupedSchema, kW13GroupedGuid, W13GroupedOutputMeta,
                                            FillW13GroupedParams);
    habana::custom_op::registerUserCustomOp(kW2GroupedSchema, kW2GroupedGuid, W2GroupedOutputMeta, FillNoParams);
    habana::custom_op::registerUserCustomOp(kAttnResNormSchema, kAttnResNormGuid, AttnResNormOutputMeta,
                                            FillAttnResNormParams);
    habana::custom_op::registerUserCustomOp(kAttnResNormF8Schema, kAttnResNormF8Guid, AttnResNormF8OutputMeta,
                                            FillAttnResNormF8Params);
    habana::custom_op::registerUserCustomOp(kAttnResF32Schema, kAttnResF32Guid, AttnResF32OutputMeta,
                                            FillAttnResF32Params);
    habana::custom_op::registerUserCustomOp(kSconvStaticSchema, kSconvStaticGuid, SconvStaticOutputMeta,
                                            FillSconvStaticParams);
    habana::custom_op::registerUserCustomOp(kQkvrPrepSchema, kQkvrPrepGuid, QkvrPrepOutputMeta, FillQkvrPrepParams);
    habana::custom_op::registerUserCustomOp(kPairRmsSchema, kPairRmsGuid, PairRmsOutputMeta, FillPairRmsParams);
    habana::custom_op::registerUserCustomOp(kBlendRmsSchema, kBlendRmsGuid, BlendRmsOutputMeta, FillBlendRmsParams);
    habana::custom_op::registerUserCustomOp(kBlendRmsF8Schema, kBlendRmsF8Guid, BlendRmsF8OutputMeta,
                                            FillBlendRmsF8Params);
    habana::custom_op::registerUserCustomOp(kPagedAttnSchema, kPagedAttnGuid, PagedAttnOutputMeta,
                                            FillPagedAttnParams);
    habana::custom_op::registerUserCustomOp(kMoeGateTopkSchema, kMoeGateTopkGuid, MoeGateTopkOutputMeta,
                                            FillMoeGateTopkParams);
    habana::custom_op::registerUserCustomOp(kMoeGateGemvSplitkSchema, kMoeGateGemvSplitkGuid,
                                            MoeGateGemvSplitkOutputMeta, FillNoParams);
    habana::custom_op::registerUserCustomOp(kMoeSumSconvResSchema, kMoeSumSconvResGuid, MoeSumSconvResOutputMeta,
                                            FillMoeSumSconvResParams);
    return true;
}

// ---- HPU implementations & Meta kernels ----

std::tuple<at::Tensor, at::Tensor> MoeW13SituF8(at::Tensor x_i32, at::Tensor x_scale, at::Tensor w13_packed,
                                                at::Tensor s13_packed, at::Tensor topk_ids, at::Tensor topk_weights,
                                                double gate_cap, double up_cap, double fp8_max)
{
    auto op = habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor(kW13Schema);
    auto outs = op.execute({x_i32, x_scale, w13_packed, s13_packed, topk_ids, topk_weights, gate_cap, up_cap, fp8_max});
    return {outs[0], outs[1]};
}

std::tuple<at::Tensor, at::Tensor> MoeW13SituF8Meta(at::Tensor, at::Tensor, at::Tensor w13_packed, at::Tensor,
                                                    at::Tensor topk_ids, at::Tensor, double, double, double)
{
    const auto total = topk_ids.sym_size(0) * topk_ids.sym_size(1) * w13_packed.sym_size(1);
    return {w13_packed.new_empty_symint({total, 128}, w13_packed.options().dtype(c10::ScalarType::Float8_e4m3fn)),
            w13_packed.new_empty_symint({total}, w13_packed.options().dtype(c10::ScalarType::Float))};
}

at::Tensor MoeW2F8(at::Tensor act_i32, at::Tensor act_scale, at::Tensor w2_packed, at::Tensor s2_packed,
                   at::Tensor topk_ids)
{
    auto op = habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor(kW2Schema);
    return op.execute({act_i32, act_scale, w2_packed, s2_packed, topk_ids})[0];
}

at::Tensor MoeW2F8Meta(at::Tensor, at::Tensor, at::Tensor w2_packed, at::Tensor, at::Tensor topk_ids)
{
    const auto total = topk_ids.sym_size(0) * topk_ids.sym_size(1) * w2_packed.sym_size(1);
    return w2_packed.new_empty_symint({total, 256}, w2_packed.options().dtype(c10::ScalarType::BFloat16));
}

std::tuple<at::Tensor, at::Tensor> MoeGroupExperts(at::Tensor topk_ids, int64_t num_experts, int64_t n_w13_blks,
                                                   int64_t n_w2_blks)
{
    auto op = habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor(kGroupExpertsSchema);
    auto outs = op.execute({topk_ids, num_experts, n_w13_blks, n_w2_blks});
    return {outs[0], outs[1]};
}

std::tuple<at::Tensor, at::Tensor> MoeGroupExpertsMeta(at::Tensor topk_ids, int64_t, int64_t, int64_t)
{
    auto opts = topk_ids.options().dtype(c10::ScalarType::Int);
    return {topk_ids.new_empty({8, 128}, opts), topk_ids.new_empty({2}, opts)};
}

std::tuple<at::Tensor, at::Tensor> MoeW13GroupedF8(at::Tensor x_i32, at::Tensor x_scale, at::Tensor w13_packed,
                                                   at::Tensor s13_packed, at::Tensor topk_weights,
                                                   at::Tensor group_meta, at::Tensor group_count,
                                                   double gate_cap, double up_cap, double fp8_max)
{
    auto op = habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor(kW13GroupedSchema);
    auto outs = op.execute({x_i32, x_scale, w13_packed, s13_packed, topk_weights, group_meta, group_count,
                            gate_cap, up_cap, fp8_max});
    return {outs[0], outs[1]};
}

std::tuple<at::Tensor, at::Tensor> MoeW13GroupedF8Meta(at::Tensor, at::Tensor, at::Tensor w13_packed, at::Tensor,
                                                       at::Tensor topk_weights, at::Tensor, at::Tensor,
                                                       double, double, double)
{
    const auto total = topk_weights.sym_size(0) * topk_weights.sym_size(1) * w13_packed.sym_size(1);
    return {w13_packed.new_empty_symint({total, 128}, w13_packed.options().dtype(c10::ScalarType::Float8_e4m3fn)),
            w13_packed.new_empty_symint({total}, w13_packed.options().dtype(c10::ScalarType::Float))};
}

at::Tensor MoeW2GroupedF8(at::Tensor act_i32, at::Tensor act_scale, at::Tensor w2_packed, at::Tensor s2_packed,
                          at::Tensor group_meta, at::Tensor group_count)
{
    auto op = habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor(kW2GroupedSchema);
    return op.execute({act_i32, act_scale, w2_packed, s2_packed, group_meta, group_count})[0];
}

at::Tensor MoeW2GroupedF8Meta(at::Tensor act_i32, at::Tensor, at::Tensor w2_packed, at::Tensor,
                              at::Tensor, at::Tensor)
{
    const auto total = act_i32.sym_size(0) * act_i32.sym_size(1) * w2_packed.sym_size(1);
    return w2_packed.new_empty_symint({total, 256}, w2_packed.options().dtype(c10::ScalarType::BFloat16));
}

at::Tensor AttnResNormBf16(at::Tensor stack, at::Tensor last, at::Tensor addend, at::Tensor queries,
                           at::Tensor norm_weight, int64_t n_stack, int64_t has_addend, int64_t normalize_values,
                           double eps, double norm_eps)
{
    auto op = habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor(kAttnResNormSchema);
    return op.execute({stack, last, addend, queries, norm_weight, n_stack, has_addend, normalize_values, eps,
                       norm_eps})[0];
}

at::Tensor AttnResNormBf16Meta(at::Tensor, at::Tensor last, at::Tensor, at::Tensor queries, at::Tensor, int64_t,
                               int64_t, int64_t, double, double)
{
    return last.new_empty_symint({queries.sym_size(0), last.sym_size(0), last.sym_size(1)},
                                 last.options().dtype(c10::ScalarType::BFloat16));
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> AttnResNormF8(
    at::Tensor stack, at::Tensor last, at::Tensor addend, at::Tensor queries, at::Tensor norm_weight,
    int64_t n_stack, int64_t has_addend, int64_t normalize_values, double eps, double norm_eps, double fp8_max)
{
    auto op = habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor(kAttnResNormF8Schema);
    auto outs = op.execute({stack, last, addend, queries, norm_weight, n_stack, has_addend, normalize_values, eps,
                            norm_eps, fp8_max});
    return {outs[0], outs[1], outs[2]};
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> AttnResNormF8Meta(
    at::Tensor, at::Tensor last, at::Tensor, at::Tensor queries, at::Tensor, int64_t, int64_t, int64_t, double,
    double, double)
{
    const auto q = queries.sym_size(0), t = last.sym_size(0), d = last.sym_size(1);
    return {last.new_empty_symint({q, t, d}, last.options().dtype(c10::ScalarType::BFloat16)),
            last.new_empty_symint({q, t, d}, last.options().dtype(c10::ScalarType::Float8_e4m3fn)),
            last.new_empty_symint({q, t}, last.options().dtype(c10::ScalarType::Float))};
}

at::Tensor AttnResF32(at::Tensor stack, at::Tensor last, at::Tensor addend, at::Tensor queries, int64_t n_stack,
                      int64_t has_addend, int64_t normalize_values, double eps)
{
    auto op = habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor(kAttnResF32Schema);
    return op.execute({stack, last, addend, queries, n_stack, has_addend, normalize_values, eps})[0];
}

at::Tensor AttnResF32Meta(at::Tensor, at::Tensor last, at::Tensor, at::Tensor queries, int64_t, int64_t, int64_t,
                          double)
{
    return last.new_empty_symint({queries.sym_size(0), last.sym_size(0), last.sym_size(1)},
                                 last.options().dtype(c10::ScalarType::Float));
}

std::tuple<at::Tensor, at::Tensor> SconvStaticDecode(at::Tensor x, at::Tensor conv_state, at::Tensor load_slots,
                                                     at::Tensor weight_t, double out_scale, int64_t stream_offset)
{
    auto op = habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor(kSconvStaticSchema);
    auto outs = op.execute({x, conv_state, load_slots, weight_t, out_scale, stream_offset});
    return {outs[0], outs[1]};
}

std::tuple<at::Tensor, at::Tensor> SconvStaticDecodeMeta(at::Tensor x, at::Tensor, at::Tensor, at::Tensor, double,
                                                         int64_t)
{
    auto opts = x.options().dtype(c10::ScalarType::BFloat16);
    return {x.new_empty_symint({x.sym_size(0), x.sym_size(1)}, opts),
            x.new_empty_symint({x.sym_size(0), 3, x.sym_size(1)}, opts)};
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor> QkvrPrepDecode(
    at::Tensor proj_q, at::Tensor q_scale, at::Tensor r_scale, at::Tensor tau, at::Tensor kv_src,
    at::Tensor conv_state, at::Tensor load_slots, at::Tensor kvw_t, at::Tensor kv_dyn,
    int64_t kv_col_offset, int64_t has_tau, int64_t is_dynamic, double conv_scale,
    double qk_clip, double rel_clip, double clip_eps, double rms_eps)
{
    auto op = habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor(kQkvrPrepSchema);
    auto outs = op.execute({proj_q, q_scale, r_scale, tau, kv_src, conv_state, load_slots, kvw_t, kv_dyn,
                            kv_col_offset, has_tau, is_dynamic, conv_scale, qk_clip, rel_clip, clip_eps, rms_eps});
    return {outs[0], outs[1], outs[2], outs[3], outs[4]};
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor> QkvrPrepDecodeMeta(
    at::Tensor proj_q, at::Tensor q_scale, at::Tensor r_scale, at::Tensor, at::Tensor, at::Tensor, at::Tensor,
    at::Tensor kvw_t, at::Tensor, int64_t, int64_t, int64_t, double, double, double, double, double)
{
    auto opts = proj_q.options().dtype(c10::ScalarType::BFloat16);
    const auto b = proj_q.sym_size(0), h = q_scale.sym_size(0), d = q_scale.sym_size(1);
    const auto d_rel = r_scale.sym_size(1), hkv = kvw_t.sym_size(2);
    return {proj_q.new_empty_symint({b, h, d}, opts),
            proj_q.new_empty_symint({b, h, d_rel}, opts),
            proj_q.new_empty_symint({b, hkv, d}, opts),
            proj_q.new_empty_symint({b, hkv, d}, opts),
            proj_q.new_empty_symint({b, 3, 2, hkv, d}, opts)};
}

at::Tensor PairRmsNorm(at::Tensor dest, at::Tensor message, double eps)
{
    auto op = habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor(kPairRmsSchema);
    return op.execute({dest, message, eps})[0];
}

at::Tensor PairRmsNormMeta(at::Tensor dest, at::Tensor, double)
{
    return dest.new_empty_symint({dest.sym_size(0), dest.sym_size(1), dest.sym_size(2) * 2},
                                 dest.options().dtype(c10::ScalarType::BFloat16));
}

at::Tensor BlendRmsNorm(at::Tensor dest, at::Tensor message, at::Tensor gain_forget, at::Tensor norm_weight,
                        double norm_eps)
{
    auto op = habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor(kBlendRmsSchema);
    return op.execute({dest, message, gain_forget, norm_weight, norm_eps})[0];
}

at::Tensor BlendRmsNormMeta(at::Tensor dest, at::Tensor, at::Tensor, at::Tensor, double)
{
    return dest.new_empty_symint({dest.sym_size(0), dest.sym_size(1), dest.sym_size(2)},
                                 dest.options().dtype(c10::ScalarType::BFloat16));
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> BlendRmsNormF8(
    at::Tensor dest, at::Tensor message, at::Tensor gain_forget, at::Tensor norm_weight,
    double norm_eps, double fp8_max)
{
    auto op = habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor(kBlendRmsF8Schema);
    auto outs = op.execute({dest, message, gain_forget, norm_weight, norm_eps, fp8_max});
    return {outs[0], outs[1], outs[2]};
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> BlendRmsNormF8Meta(
    at::Tensor dest, at::Tensor, at::Tensor, at::Tensor, double, double)
{
    const auto s = dest.sym_size(0), t = dest.sym_size(1), d = dest.sym_size(2);
    return {dest.new_empty_symint({s, t, d}, dest.options().dtype(c10::ScalarType::BFloat16)),
            dest.new_empty_symint({s, t, d}, dest.options().dtype(c10::ScalarType::Float8_e4m3fn)),
            dest.new_empty_symint({s, t}, dest.options().dtype(c10::ScalarType::Float))};
}

at::Tensor RelPagedAttnGateSink(at::Tensor q, at::Tensor r, at::Tensor proj_q, at::Tensor rel_proj_t,
                                at::Tensor key_cache, at::Tensor value_cache, at::Tensor k_new, at::Tensor v_new,
                                at::Tensor block_list, at::Tensor block_groups, at::Tensor block_usage,
                                at::Tensor block_start_pos, at::Tensor positions,
                                double scale, int64_t sliding_window, int64_t rel_extent)
{
    auto op = habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor(kPagedAttnSchema);
    return op.execute({q, r, proj_q, rel_proj_t, key_cache, value_cache, k_new, v_new, block_list, block_groups,
                       block_usage, block_start_pos, positions, scale, sliding_window, rel_extent})[0];
}

at::Tensor RelPagedAttnGateSinkMeta(at::Tensor q, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor,
                                    at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor,
                                    at::Tensor, double, int64_t, int64_t)
{
    return q.new_empty_symint({q.sym_size(0), q.sym_size(1), q.sym_size(2)},
                              q.options().dtype(c10::ScalarType::BFloat16));
}

std::tuple<at::Tensor, at::Tensor> MoeGateTopk(at::Tensor logits, at::Tensor bias, int64_t top_k, double route_scale)
{
    auto op = habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor(kMoeGateTopkSchema);
    auto outs = op.execute({logits, bias, top_k, route_scale});
    return {outs[0], outs[1]};
}

std::tuple<at::Tensor, at::Tensor> MoeGateTopkMeta(at::Tensor logits, at::Tensor, int64_t top_k, double)
{
    const auto b = logits.sym_size(0);
    return {logits.new_empty_symint({b, top_k}, logits.options().dtype(c10::ScalarType::Int)),
            logits.new_empty_symint({b, top_k}, logits.options().dtype(c10::ScalarType::Float))};
}

at::Tensor MoeGateGemvSplitk(at::Tensor x, at::Tensor w_gate_packed)
{
    auto op = habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor(kMoeGateGemvSplitkSchema);
    return op.execute({x, w_gate_packed})[0];
}

at::Tensor MoeGateGemvSplitkMeta(at::Tensor x, at::Tensor)
{
    const auto b = x.sym_size(0), s = x.sym_size(1) / 128;
    return x.new_empty_symint({b, s, 128}, x.options().dtype(c10::ScalarType::Float));
}

std::tuple<at::Tensor, at::Tensor> MoeSumSconvResDecode(at::Tensor moe_partial, at::Tensor conv_state,
                                                        at::Tensor load_slots, at::Tensor weight_t,
                                                        at::Tensor partial_res, at::Tensor attn_out,
                                                        double out_scale, int64_t is_first, int64_t stream_offset)
{
    auto op = habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor(kMoeSumSconvResSchema);
    auto outs = op.execute(
        {moe_partial, conv_state, load_slots, weight_t, partial_res, attn_out, out_scale, is_first, stream_offset});
    return {outs[0], outs[1]};
}

std::tuple<at::Tensor, at::Tensor> MoeSumSconvResDecodeMeta(at::Tensor, at::Tensor, at::Tensor, at::Tensor,
                                                            at::Tensor partial_res, at::Tensor, double, int64_t,
                                                            int64_t)
{
    auto opts = partial_res.options().dtype(c10::ScalarType::BFloat16);
    return {partial_res.new_empty_symint({partial_res.sym_size(0), partial_res.sym_size(1)}, opts),
            partial_res.new_empty_symint({partial_res.sym_size(0), 3, partial_res.sym_size(1)}, opts)};
}
}  // namespace

TORCH_LIBRARY_FRAGMENT(custom_op, m)
{
    m.def("inkling2_moe_w13_situ_f8(Tensor x_i32, Tensor x_scale, Tensor w13_packed, Tensor s13_packed, "
          "Tensor topk_ids, Tensor topk_weights, float gate_cap, float up_cap, float fp8_max) -> (Tensor, Tensor)");
    m.def("inkling2_moe_w2_f8(Tensor act_i32, Tensor act_scale, Tensor w2_packed, Tensor s2_packed, "
          "Tensor topk_ids) -> Tensor");
    m.def("inkling2_moe_group_experts(Tensor topk_ids, int num_experts, int n_w13_blks, int n_w2_blks) -> "
          "(Tensor, Tensor)");
    m.def("inkling2_moe_w13_grouped_f8(Tensor x_i32, Tensor x_scale, Tensor w13_packed, Tensor s13_packed, "
          "Tensor topk_weights, Tensor group_meta, Tensor group_count, float gate_cap, float up_cap, "
          "float fp8_max) -> (Tensor, Tensor)");
    m.def("inkling2_moe_w2_grouped_f8(Tensor act_i32, Tensor act_scale, Tensor w2_packed, Tensor s2_packed, "
          "Tensor group_meta, Tensor group_count) -> Tensor");
    m.def("inkling2_attn_res_norm_bf16(Tensor stack, Tensor last, Tensor addend, Tensor queries, Tensor norm_weight, "
          "int n_stack, int has_addend, int normalize_values, float eps, float norm_eps) -> Tensor");
    m.def("inkling2_attn_res_norm_f8(Tensor stack, Tensor last, Tensor addend, Tensor queries, Tensor norm_weight, "
          "int n_stack, int has_addend, int normalize_values, float eps, float norm_eps, float fp8_max) -> "
          "(Tensor, Tensor, Tensor)");
    m.def("inkling2_attn_res_f32(Tensor stack, Tensor last, Tensor addend, Tensor queries, "
          "int n_stack, int has_addend, int normalize_values, float eps) -> Tensor");
    m.def("inkling2_sconv_static_decode(Tensor x, Tensor conv_state, Tensor load_slots, Tensor weight_t, "
          "float out_scale, int stream_offset) -> (Tensor, Tensor)");
    m.def("inkling2_qkvr_prep_decode(Tensor proj_q, Tensor q_scale, Tensor r_scale, Tensor tau, "
          "Tensor kv_src, Tensor conv_state, Tensor load_slots, Tensor kvw_t, Tensor kv_dyn, "
          "int kv_col_offset, int has_tau, int is_dynamic, float conv_scale, float qk_clip, float rel_clip, "
          "float clip_eps, float rms_eps) -> (Tensor, Tensor, Tensor, Tensor, Tensor)");
    m.def("inkling2_pair_rms_norm(Tensor dest, Tensor message, float eps) -> Tensor");
    m.def("inkling2_blend_rms_norm(Tensor dest, Tensor message, Tensor gain_forget, Tensor norm_weight, "
          "float norm_eps) -> Tensor");
    m.def("inkling2_blend_rms_norm_f8(Tensor dest, Tensor message, Tensor gain_forget, Tensor norm_weight, "
          "float norm_eps, float fp8_max) -> (Tensor, Tensor, Tensor)");
    m.def("inkling2_rel_paged_attn_gate_sink(Tensor q, Tensor r, Tensor proj_q, Tensor rel_proj_t, "
          "Tensor key_cache, Tensor value_cache, Tensor k_new, Tensor v_new, Tensor block_list, Tensor block_groups, "
          "Tensor block_usage, Tensor block_start_pos, Tensor positions, float scale, int sliding_window, "
          "int rel_extent) -> Tensor");
    m.def("inkling2_moe_gate_topk(Tensor logits, Tensor bias, int top_k, float route_scale) -> (Tensor, Tensor)");
    m.def("inkling2_moe_gate_gemv_splitk(Tensor x, Tensor w_gate_packed) -> Tensor");
    m.def("inkling2_moe_sum_sconv_res_decode(Tensor moe_partial, Tensor conv_state, Tensor load_slots, "
          "Tensor weight_t, Tensor partial_res, Tensor attn_out, float out_scale, int is_first, "
          "int stream_offset) -> (Tensor, Tensor)");
}
TORCH_LIBRARY_IMPL(custom_op, HPU, m)
{
    m.impl("inkling2_moe_w13_situ_f8", MoeW13SituF8);
    m.impl("inkling2_moe_w2_f8", MoeW2F8);
    m.impl("inkling2_moe_group_experts", MoeGroupExperts);
    m.impl("inkling2_moe_w13_grouped_f8", MoeW13GroupedF8);
    m.impl("inkling2_moe_w2_grouped_f8", MoeW2GroupedF8);
    m.impl("inkling2_attn_res_norm_bf16", AttnResNormBf16);
    m.impl("inkling2_attn_res_norm_f8", AttnResNormF8);
    m.impl("inkling2_attn_res_f32", AttnResF32);
    m.impl("inkling2_sconv_static_decode", SconvStaticDecode);
    m.impl("inkling2_qkvr_prep_decode", QkvrPrepDecode);
    m.impl("inkling2_pair_rms_norm", PairRmsNorm);
    m.impl("inkling2_blend_rms_norm", BlendRmsNorm);
    m.impl("inkling2_blend_rms_norm_f8", BlendRmsNormF8);
    m.impl("inkling2_rel_paged_attn_gate_sink", RelPagedAttnGateSink);
    m.impl("inkling2_moe_gate_topk", MoeGateTopk);
    m.impl("inkling2_moe_gate_gemv_splitk", MoeGateGemvSplitk);
    m.impl("inkling2_moe_sum_sconv_res_decode", MoeSumSconvResDecode);
}
TORCH_LIBRARY_IMPL(custom_op, Meta, m)
{
    m.impl("inkling2_moe_w13_situ_f8", MoeW13SituF8Meta);
    m.impl("inkling2_moe_w2_f8", MoeW2F8Meta);
    m.impl("inkling2_moe_group_experts", MoeGroupExpertsMeta);
    m.impl("inkling2_moe_w13_grouped_f8", MoeW13GroupedF8Meta);
    m.impl("inkling2_moe_w2_grouped_f8", MoeW2GroupedF8Meta);
    m.impl("inkling2_attn_res_norm_bf16", AttnResNormBf16Meta);
    m.impl("inkling2_attn_res_norm_f8", AttnResNormF8Meta);
    m.impl("inkling2_attn_res_f32", AttnResF32Meta);
    m.impl("inkling2_sconv_static_decode", SconvStaticDecodeMeta);
    m.impl("inkling2_qkvr_prep_decode", QkvrPrepDecodeMeta);
    m.impl("inkling2_pair_rms_norm", PairRmsNormMeta);
    m.impl("inkling2_blend_rms_norm", BlendRmsNormMeta);
    m.impl("inkling2_blend_rms_norm_f8", BlendRmsNormF8Meta);
    m.impl("inkling2_rel_paged_attn_gate_sink", RelPagedAttnGateSinkMeta);
    m.impl("inkling2_moe_gate_topk", MoeGateTopkMeta);
    m.impl("inkling2_moe_gate_gemv_splitk", MoeGateGemvSplitkMeta);
    m.impl("inkling2_moe_sum_sconv_res_decode", MoeSumSconvResDecodeMeta);
}

static const bool kRegistered = Register();
