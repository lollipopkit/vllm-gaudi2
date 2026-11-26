// SPDX-License-Identifier: Apache-2.0
// PT2 custom ops running Gaudi2 FP8, Qwen3.8 GDN, and LFM3 ShortConv TPC kernels inside
// torch.compile(backend="hpu_backend"). Requires libgaudi2_tpc_proxy.so (or libgaudi2_tpc_kernels.so)
// on GC_KERNEL_PATH.
#include <torch/extension.h>

#include <cmath>
#include <tuple>

#include "gaudi2_fp8_ops_params.hpp"
#include "hpu_custom_op_pt2.h"

namespace
{
// Keep all ops in the "custom_op" namespace so torch.compile(backend="hpu_backend")
// fuses them inside compiled graphs without eager fallbacks.
constexpr const char* kAddRmsNormF8Schema = "custom_op::gaudi2_add_rms_norm_f8";
constexpr const char* kAddRmsNormF8Bf16Schema = "custom_op::gaudi2_add_rms_norm_f8_bf16";
constexpr const char* kAddRmsNormF8Guid = "gaudi2_add_rms_norm_f8_gaudi2";

constexpr const char* kSandwichRmsNormF8Schema = "custom_op::gaudi2_sandwich_rms_norm_f8";
constexpr const char* kSandwichRmsNormF8Guid = "gaudi2_sandwich_rms_norm_f8_gaudi2";

constexpr const char* kSiluMulF8Schema = "custom_op::gaudi2_silu_mul_f8";
constexpr const char* kSiluMulF8Guid = "gaudi2_silu_mul_f8_gaudi2";

constexpr const char* kGdnPrepSchema = "custom_op::gdn_prep_decode";
constexpr const char* kGdnPrepGuid = "gdn_prep_decode_gaudi2";

constexpr const char* kGdnGatedRmsF8Schema = "custom_op::gdn_gated_rms_norm_f8";
constexpr const char* kGdnGatedRmsF8Guid = "gdn_gated_rms_norm_f8_gaudi2";

constexpr const char* kLfm3ShortConvSchema = "custom_op::lfm3_short_conv_decode";
constexpr const char* kLfm3ShortConvGuid = "lfm3_short_conv_decode_gaudi2";

constexpr int64_t kMaxRmsNormDim = 12288;
constexpr int64_t kMaxSiluMulDim = 28672;
constexpr int64_t kMaxGdnValueHeads = 128;

using gaudi2_fp8_params::AddRmsNormF8Params;
using gaudi2_fp8_params::GdnGatedRmsNormF8Params;
using gaudi2_fp8_params::GdnPrepDecodeParams;
using gaudi2_fp8_params::Lfm3ShortConvDecodeParams;
using gaudi2_fp8_params::SiluMulF8Params;

inline bool IsPosFinite(double v)
{
    return std::isfinite(v) && v > 0.0;
}

// ---- Output meta & param fillers ----

habana::PartialOutputMetaDataVector AddRmsNormF8OutputMeta(const at::Stack& inputs)
{
    const auto x = inputs[0].toTensor();  // [M, D]
    return {{c10::ScalarType::Float8_e4m3fn, {x.size(0), x.size(1)}},
            {c10::ScalarType::Float, {x.size(0), 1}},
            {c10::ScalarType::BFloat16, {x.size(0), x.size(1)}}};
}

habana::PartialOutputMetaDataVector AddRmsNormF8Bf16OutputMeta(const at::Stack& inputs)
{
    const auto x = inputs[0].toTensor();  // [M, D]
    return {{c10::ScalarType::Float8_e4m3fn, {x.size(0), x.size(1)}},
            {c10::ScalarType::Float, {x.size(0), 1}},
            {c10::ScalarType::BFloat16, {x.size(0), x.size(1)}},
            {c10::ScalarType::BFloat16, {x.size(0), x.size(1)}}};
}

std::shared_ptr<void> FillAddRmsNormF8Params(const at::Stack& inputs, size_t& size)
{
    HPU_PARAMS_STUB(AddRmsNormF8Params);
    params->eps = static_cast<float>(inputs[3].toDouble());
    params->fp8Max = static_cast<float>(inputs[4].toDouble());
    return params;
}

std::shared_ptr<void> FillSandwichRmsNormF8Params(const at::Stack& inputs, size_t& size)
{
    HPU_PARAMS_STUB(AddRmsNormF8Params);
    params->eps = static_cast<float>(inputs[4].toDouble());
    params->fp8Max = static_cast<float>(inputs[5].toDouble());
    return params;
}

habana::PartialOutputMetaDataVector SiluMulF8OutputMeta(const at::Stack& inputs)
{
    const auto gu = inputs[0].toTensor();  // [M, 2*N]
    return {{c10::ScalarType::Float8_e4m3fn, {gu.size(0), gu.size(1) / 2}},
            {c10::ScalarType::Float, {gu.size(0), 1}}};
}

std::shared_ptr<void> FillSiluMulF8Params(const at::Stack& inputs, size_t& size)
{
    HPU_PARAMS_STUB(SiluMulF8Params);
    params->fp8Max = static_cast<float>(inputs[1].toDouble());
    return params;
}

habana::PartialOutputMetaDataVector GdnPrepOutputMeta(const at::Stack& inputs)
{
    const auto qkv = inputs[0].toTensor();   // [B, (2H + HV) * 128]
    const auto a_in = inputs[1].toTensor();  // [B, HV]
    const int64_t b = qkv.size(0);
    const int64_t hv = a_in.size(1);
    const int64_t total_heads = qkv.size(1) / 128;
    TORCH_CHECK(qkv.size(1) % 128 == 0 && total_heads > hv && ((total_heads - hv) % 2) == 0,
                "packed_qkv width must equal (2*H + HV) * 128 with H > 0");
    const int64_t h = (total_heads - hv) / 2;
    return {{c10::ScalarType::Float, {b, 2 * h, 128}},
            {c10::ScalarType::Float, {b, hv, 128}},
            {c10::ScalarType::Float, {b, hv}},
            {c10::ScalarType::Float, {b, hv}},
            {c10::ScalarType::Float, {b, h}}};
}

std::shared_ptr<void> FillGdnPrepParams(const at::Stack& inputs, size_t& size)
{
    HPU_PARAMS_STUB(GdnPrepDecodeParams);
    params->rowOffset = static_cast<int>(inputs[8].toInt());
    params->convColOffset = static_cast<int>(inputs[9].toInt());
    params->hasBias = static_cast<int>(inputs[10].toInt());
    return params;
}

habana::PartialOutputMetaDataVector GdnGatedRmsF8OutputMeta(const at::Stack& inputs)
{
    const auto z = inputs[1].toTensor();  // [B, HV*128]
    return {{c10::ScalarType::Float8_e4m3fn, {z.size(0), z.size(1)}},
            {c10::ScalarType::Float, {z.size(0), 1}}};
}

std::shared_ptr<void> FillGdnGatedRmsF8Params(const at::Stack& inputs, size_t& size)
{
    HPU_PARAMS_STUB(GdnGatedRmsNormF8Params);
    params->eps = static_cast<float>(inputs[3].toDouble());
    params->fp8Max = static_cast<float>(inputs[4].toDouble());
    return params;
}

habana::PartialOutputMetaDataVector Lfm3ShortConvOutputMeta(const at::Stack& inputs)
{
    const auto bcx = inputs[0].toTensor();   // [B, 3*D]
    const auto taps = inputs[3].toTensor();  // [K, D]
    TORCH_CHECK(taps.dim() == 2 && taps.size(0) >= 2, "taps must be [K, D] with K >= 2");
    return {{c10::ScalarType::BFloat16, {bcx.size(0), taps.size(1)}},
            {c10::ScalarType::BFloat16, {bcx.size(0), taps.size(0) - 1, taps.size(1)}}};
}

std::shared_ptr<void> FillLfm3ShortConvParams(const at::Stack& inputs, size_t& size)
{
    HPU_PARAMS_STUB(Lfm3ShortConvDecodeParams);
    params->hasBias = static_cast<int>(inputs[5].toInt());
    return params;
}

bool Register()
{
    habana::custom_op::registerUserCustomOp(kAddRmsNormF8Schema, kAddRmsNormF8Guid, AddRmsNormF8OutputMeta,
                                            FillAddRmsNormF8Params);
    habana::custom_op::registerUserCustomOp(kAddRmsNormF8Bf16Schema, kAddRmsNormF8Guid, AddRmsNormF8Bf16OutputMeta,
                                            FillAddRmsNormF8Params);
    habana::custom_op::registerUserCustomOp(kSandwichRmsNormF8Schema, kSandwichRmsNormF8Guid, AddRmsNormF8OutputMeta,
                                            FillSandwichRmsNormF8Params);
    habana::custom_op::registerUserCustomOp(kSiluMulF8Schema, kSiluMulF8Guid, SiluMulF8OutputMeta,
                                            FillSiluMulF8Params);
    habana::custom_op::registerUserCustomOp(kGdnPrepSchema, kGdnPrepGuid, GdnPrepOutputMeta, FillGdnPrepParams);
    habana::custom_op::registerUserCustomOp(kGdnGatedRmsF8Schema, kGdnGatedRmsF8Guid, GdnGatedRmsF8OutputMeta,
                                            FillGdnGatedRmsF8Params);
    habana::custom_op::registerUserCustomOp(kLfm3ShortConvSchema, kLfm3ShortConvGuid, Lfm3ShortConvOutputMeta,
                                            FillLfm3ShortConvParams);
    return true;
}

void CheckAddRmsNormInputs(const at::Tensor& x, const at::Tensor& residual, const at::Tensor& weight, double eps,
                           double fp8_max)
{
    TORCH_CHECK(x.dim() == 2 && residual.dim() == 2 && weight.dim() == 1, "expected 2D x/residual and 1D weight");
    TORCH_CHECK(x.is_contiguous() && residual.is_contiguous() && weight.is_contiguous(),
                "x, residual, and weight must be contiguous");
    TORCH_CHECK(x.scalar_type() == c10::ScalarType::BFloat16 && residual.scalar_type() == c10::ScalarType::BFloat16,
                "x and residual must be bfloat16");
    TORCH_CHECK(weight.scalar_type() == c10::ScalarType::Float || weight.scalar_type() == c10::ScalarType::BFloat16,
                "weight must be float32 or bfloat16");
    TORCH_CHECK(x.sizes() == residual.sizes() && weight.size(0) == x.size(1),
                "x, residual, and weight shapes must match");
    TORCH_CHECK(x.size(0) > 0 && x.size(1) > 0 && (x.size(1) % 64) == 0 && x.size(1) <= kMaxRmsNormDim,
                "expected M > 0 and D in (0, 12288] divisible by 64");
    TORCH_CHECK(IsPosFinite(eps) && IsPosFinite(fp8_max), "eps and fp8_max must be positive finite");
}

// ---- HPU implementations & symbolic Meta kernels ----

std::tuple<at::Tensor, at::Tensor, at::Tensor> AddRmsNormF8(at::Tensor x, at::Tensor residual, at::Tensor weight,
                                                            double eps, double fp8_max)
{
    CheckAddRmsNormInputs(x, residual, weight, eps, fp8_max);
    auto op = habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor(kAddRmsNormF8Schema);
    auto outs = op.execute({x, residual, weight, eps, fp8_max});
    return {outs[0], outs[1], outs[2]};
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> AddRmsNormF8Meta(at::Tensor x, at::Tensor, at::Tensor, double, double)
{
    const auto m = x.sym_size(0), d = x.sym_size(1);
    return {x.new_empty_symint({m, d}, x.options().dtype(c10::ScalarType::Float8_e4m3fn)),
            x.new_empty_symint({m, 1}, x.options().dtype(c10::ScalarType::Float)),
            x.new_empty_symint({m, d}, x.options().dtype(c10::ScalarType::BFloat16))};
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> AddRmsNormF8Bf16(at::Tensor x, at::Tensor residual,
                                                                            at::Tensor weight, double eps,
                                                                            double fp8_max)
{
    CheckAddRmsNormInputs(x, residual, weight, eps, fp8_max);
    auto op = habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor(kAddRmsNormF8Bf16Schema);
    auto outs = op.execute({x, residual, weight, eps, fp8_max});
    return {outs[0], outs[1], outs[2], outs[3]};
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> AddRmsNormF8Bf16Meta(at::Tensor x, at::Tensor, at::Tensor,
                                                                                double, double)
{
    const auto m = x.sym_size(0), d = x.sym_size(1);
    return {x.new_empty_symint({m, d}, x.options().dtype(c10::ScalarType::Float8_e4m3fn)),
            x.new_empty_symint({m, 1}, x.options().dtype(c10::ScalarType::Float)),
            x.new_empty_symint({m, d}, x.options().dtype(c10::ScalarType::BFloat16)),
            x.new_empty_symint({m, d}, x.options().dtype(c10::ScalarType::BFloat16))};
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> SandwichRmsNormF8(at::Tensor x, at::Tensor residual, at::Tensor post_w,
                                                                 at::Tensor pre_w, double eps, double fp8_max)
{
    TORCH_CHECK(x.dim() == 2 && residual.dim() == 2 && post_w.dim() == 1 && pre_w.dim() == 1,
                "expected 2D x/residual and 1D post_w/pre_w");
    TORCH_CHECK(x.is_contiguous() && residual.is_contiguous() && post_w.is_contiguous() && pre_w.is_contiguous(),
                "x, residual, post_w, pre_w must be contiguous");
    TORCH_CHECK(x.scalar_type() == c10::ScalarType::BFloat16 && residual.scalar_type() == c10::ScalarType::BFloat16 &&
                    post_w.scalar_type() == c10::ScalarType::BFloat16 &&
                    pre_w.scalar_type() == c10::ScalarType::BFloat16,
                "x, residual, post_w, pre_w must be bfloat16");
    TORCH_CHECK(x.sizes() == residual.sizes() && post_w.size(0) == x.size(1) && pre_w.size(0) == x.size(1),
                "x, residual, post_w, pre_w shapes must match");
    TORCH_CHECK(x.size(0) > 0 && x.size(1) > 0 && (x.size(1) % 64) == 0 && x.size(1) <= kMaxRmsNormDim,
                "expected M > 0 and D in (0, 12288] divisible by 64");
    TORCH_CHECK(IsPosFinite(eps) && IsPosFinite(fp8_max), "eps and fp8_max must be positive finite");
    auto op = habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor(kSandwichRmsNormF8Schema);
    auto outs = op.execute({x, residual, post_w, pre_w, eps, fp8_max});
    return {outs[0], outs[1], outs[2]};
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> SandwichRmsNormF8Meta(at::Tensor x, at::Tensor, at::Tensor, at::Tensor,
                                                                     double, double)
{
    const auto m = x.sym_size(0), d = x.sym_size(1);
    return {x.new_empty_symint({m, d}, x.options().dtype(c10::ScalarType::Float8_e4m3fn)),
            x.new_empty_symint({m, 1}, x.options().dtype(c10::ScalarType::Float)),
            x.new_empty_symint({m, d}, x.options().dtype(c10::ScalarType::BFloat16))};
}

std::tuple<at::Tensor, at::Tensor> SiluMulF8(at::Tensor gate_up, double fp8_max)
{
    TORCH_CHECK(gate_up.dim() == 2 && gate_up.is_contiguous() && gate_up.scalar_type() == c10::ScalarType::BFloat16,
                "gate_up must be contiguous 2D bfloat16 [M, 2*N]");
    TORCH_CHECK(gate_up.size(0) > 0 && gate_up.size(1) > 0 && (gate_up.size(1) % 128) == 0 &&
                    (gate_up.size(1) / 2) <= kMaxSiluMulDim,
                "expected M > 0 and N = gate_up.size(1)/2 in (0, 28672] divisible by 64");
    TORCH_CHECK(IsPosFinite(fp8_max), "fp8_max must be positive finite");
    auto op = habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor(kSiluMulF8Schema);
    auto outs = op.execute({gate_up, fp8_max});
    return {outs[0], outs[1]};
}

std::tuple<at::Tensor, at::Tensor> SiluMulF8Meta(at::Tensor gate_up, double)
{
    const auto m = gate_up.sym_size(0), n = gate_up.sym_size(1) / 2;
    return {gate_up.new_empty_symint({m, n}, gate_up.options().dtype(c10::ScalarType::Float8_e4m3fn)),
            gate_up.new_empty_symint({m, 1}, gate_up.options().dtype(c10::ScalarType::Float))};
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor> GdnPrepDecode(
    at::Tensor packed_qkv, at::Tensor a_in, at::Tensor b_in, at::Tensor neg_exp_A, at::Tensor dt_bias,
    at::Tensor conv_state, at::Tensor conv_taps, at::Tensor conv_bias, int64_t row_offset, int64_t conv_col_offset,
    int64_t has_bias)
{
    TORCH_CHECK(packed_qkv.dim() == 2 && packed_qkv.is_contiguous() &&
                    packed_qkv.scalar_type() == c10::ScalarType::BFloat16,
                "packed_qkv must be contiguous [B, W] bfloat16");
    TORCH_CHECK(a_in.dim() == 2 && b_in.dim() == 2 && a_in.is_contiguous() && b_in.is_contiguous() &&
                    a_in.scalar_type() == c10::ScalarType::BFloat16 && b_in.scalar_type() == c10::ScalarType::BFloat16,
                "a_in and b_in must be contiguous [B, HV] bfloat16");
    TORCH_CHECK(conv_state.dim() == 3 && conv_state.is_contiguous() &&
                    conv_state.scalar_type() == c10::ScalarType::BFloat16,
                "conv_state must be contiguous [slots, cache_len, W] bfloat16");
    TORCH_CHECK(neg_exp_A.dim() == 1 && dt_bias.dim() == 1 && conv_taps.dim() == 2 && conv_bias.dim() == 1 &&
                    neg_exp_A.is_contiguous() && dt_bias.is_contiguous() && conv_taps.is_contiguous() &&
                    conv_bias.is_contiguous() && neg_exp_A.scalar_type() == c10::ScalarType::Float &&
                    dt_bias.scalar_type() == c10::ScalarType::Float &&
                    conv_taps.scalar_type() == c10::ScalarType::Float &&
                    conv_bias.scalar_type() == c10::ScalarType::Float,
                "neg_exp_A, dt_bias, conv_taps, conv_bias must be contiguous float32");
    const int64_t b = packed_qkv.size(0);
    const int64_t w = packed_qkv.size(1);
    const int64_t hv = a_in.size(1);
    TORCH_CHECK(b > 0 && hv > 0 && hv <= 64 && a_in.size(0) == b && b_in.sizes() == a_in.sizes(),
                "invalid B or HV for gdn_prep_decode");
    TORCH_CHECK(w % 128 == 0 && (w / 128) > hv && (((w / 128) - hv) % 2) == 0,
                "packed_qkv width must equal (2*H + HV) * 128 with H > 0");
    TORCH_CHECK(neg_exp_A.size(0) == hv && dt_bias.size(0) == hv && conv_state.size(2) == w &&
                    conv_taps.size(0) == 4 && conv_taps.size(1) == w && conv_bias.size(0) == w,
                "parameter shapes do not match packed_qkv/HV");
    TORCH_CHECK(conv_state.size(1) >= 3 && row_offset >= 0 && row_offset + b <= conv_state.size(0) &&
                    conv_col_offset >= 0 && conv_col_offset + 3 <= conv_state.size(1),
                "conv_state row_offset/conv_col_offset out of bounds");
    auto op = habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor(kGdnPrepSchema);
    auto outs = op.execute({packed_qkv, a_in, b_in, neg_exp_A, dt_bias, conv_state, conv_taps, conv_bias, row_offset,
                            conv_col_offset, has_bias});
    return {outs[0], outs[1], outs[2], outs[3], outs[4]};
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor> GdnPrepDecodeMeta(
    at::Tensor packed_qkv, at::Tensor a_in, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor,
    int64_t, int64_t, int64_t)
{
    const auto b = packed_qkv.sym_size(0);
    const auto hv = a_in.sym_size(1);
    const auto h = (packed_qkv.sym_size(1) / 128 - hv) / 2;
    auto opts = packed_qkv.options().dtype(c10::ScalarType::Float);
    return {packed_qkv.new_empty_symint({b, h * 2, 128}, opts),
            packed_qkv.new_empty_symint({b, hv, 128}, opts),
            packed_qkv.new_empty_symint({b, hv}, opts),
            packed_qkv.new_empty_symint({b, hv}, opts),
            packed_qkv.new_empty_symint({b, h}, opts)};
}

std::tuple<at::Tensor, at::Tensor> GdnGatedRmsNormF8(at::Tensor x, at::Tensor z, at::Tensor weight, double eps,
                                                     double fp8_max)
{
    TORCH_CHECK(x.dim() == 3 && x.is_contiguous() && x.size(2) == 128 && x.scalar_type() == c10::ScalarType::Float,
                "x must be contiguous [B, HV, 128] float32");
    TORCH_CHECK(z.dim() == 2 && z.is_contiguous() && z.scalar_type() == c10::ScalarType::BFloat16,
                "z must be contiguous [B, HV*128] bfloat16");
    TORCH_CHECK(weight.dim() == 1 && weight.is_contiguous() && weight.size(0) == 128 &&
                    weight.scalar_type() == c10::ScalarType::Float,
                "weight must be contiguous [128] float32");
    TORCH_CHECK(x.size(0) > 0 && x.size(1) > 0 && x.size(1) <= kMaxGdnValueHeads && z.size(0) == x.size(0) &&
                    z.size(1) == x.size(1) * 128,
                "x and z shapes must match with B > 0 and HV in (0, 128]");
    TORCH_CHECK(IsPosFinite(eps) && IsPosFinite(fp8_max), "eps and fp8_max must be positive finite");
    auto op = habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor(kGdnGatedRmsF8Schema);
    auto outs = op.execute({x, z, weight, eps, fp8_max});
    return {outs[0], outs[1]};
}

std::tuple<at::Tensor, at::Tensor> GdnGatedRmsNormF8Meta(at::Tensor, at::Tensor z, at::Tensor, double, double)
{
    const auto b = z.sym_size(0), d = z.sym_size(1);
    return {z.new_empty_symint({b, d}, z.options().dtype(c10::ScalarType::Float8_e4m3fn)),
            z.new_empty_symint({b, 1}, z.options().dtype(c10::ScalarType::Float))};
}

std::tuple<at::Tensor, at::Tensor> Lfm3ShortConvDecode(at::Tensor bcx, at::Tensor conv_state, at::Tensor load_idx,
                                                       at::Tensor taps, at::Tensor bias, int64_t has_bias)
{
    TORCH_CHECK(bcx.dim() == 2 && bcx.is_contiguous() && bcx.scalar_type() == c10::ScalarType::BFloat16,
                "bcx must be contiguous [B, 3*D] bfloat16");
    TORCH_CHECK(conv_state.dim() == 3 && conv_state.is_contiguous() &&
                    conv_state.scalar_type() == c10::ScalarType::BFloat16,
                "conv_state must be contiguous [slots, cache_len, D] bfloat16");
    TORCH_CHECK(load_idx.dim() == 1 && load_idx.is_contiguous() && load_idx.scalar_type() == c10::ScalarType::Int,
                "load_idx must be contiguous 1D int32");
    TORCH_CHECK(taps.dim() == 2 && bias.dim() == 1 && taps.is_contiguous() && bias.is_contiguous() &&
                    taps.scalar_type() == c10::ScalarType::Float && bias.scalar_type() == c10::ScalarType::Float,
                "taps and bias must be contiguous float32");
    const int64_t b = bcx.size(0);
    const int64_t d = taps.size(1);
    const int64_t k = taps.size(0);
    TORCH_CHECK(b > 0 && d > 0 && (d % 64) == 0 && k >= 2 && conv_state.size(0) > 0 &&
                    conv_state.size(1) >= k - 1 && conv_state.size(2) == d && bcx.size(1) == 3 * d &&
                    load_idx.size(0) == b && bias.size(0) == d,
                "lfm3_short_conv_decode input shapes are incompatible");
    auto op = habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor(kLfm3ShortConvSchema);
    auto outs = op.execute({bcx, conv_state, load_idx, taps, bias, has_bias});
    return {outs[0], outs[1]};
}

std::tuple<at::Tensor, at::Tensor> Lfm3ShortConvDecodeMeta(at::Tensor bcx, at::Tensor, at::Tensor, at::Tensor taps,
                                                           at::Tensor, int64_t)
{
    const auto b = bcx.sym_size(0);
    const auto d = taps.sym_size(1);
    const auto state_len = taps.sym_size(0) - 1;
    auto opts = bcx.options().dtype(c10::ScalarType::BFloat16);
    return {bcx.new_empty_symint({b, d}, opts), bcx.new_empty_symint({b, state_len, d}, opts)};
}
}  // namespace

TORCH_LIBRARY_FRAGMENT(custom_op, m)
{
    m.def("gaudi2_add_rms_norm_f8(Tensor x, Tensor residual, Tensor weight, float eps, float fp8_max) -> "
          "(Tensor, Tensor, Tensor)");
    m.def("gaudi2_add_rms_norm_f8_bf16(Tensor x, Tensor residual, Tensor weight, float eps, float fp8_max) -> "
          "(Tensor, Tensor, Tensor, Tensor)");
    m.def("gaudi2_sandwich_rms_norm_f8(Tensor x, Tensor residual, Tensor post_w, Tensor pre_w, float eps, "
          "float fp8_max) -> (Tensor, Tensor, Tensor)");
    m.def("gaudi2_silu_mul_f8(Tensor gate_up, float fp8_max) -> (Tensor, Tensor)");
    m.def("gdn_prep_decode(Tensor packed_qkv, Tensor a_in, Tensor b_in, Tensor neg_exp_A, Tensor dt_bias, "
          "Tensor conv_state, Tensor conv_taps, Tensor conv_bias, int row_offset, int conv_col_offset, "
          "int has_bias) -> (Tensor, Tensor, Tensor, Tensor, Tensor)");
    m.def("gdn_gated_rms_norm_f8(Tensor x, Tensor z, Tensor weight, float eps, float fp8_max) -> (Tensor, Tensor)");
    m.def("lfm3_short_conv_decode(Tensor bcx, Tensor conv_state, Tensor load_idx, Tensor taps, Tensor bias, "
          "int has_bias) -> (Tensor, Tensor)");
}
TORCH_LIBRARY_IMPL(custom_op, HPU, m)
{
    m.impl("gaudi2_add_rms_norm_f8", AddRmsNormF8);
    m.impl("gaudi2_add_rms_norm_f8_bf16", AddRmsNormF8Bf16);
    m.impl("gaudi2_sandwich_rms_norm_f8", SandwichRmsNormF8);
    m.impl("gaudi2_silu_mul_f8", SiluMulF8);
    m.impl("gdn_prep_decode", GdnPrepDecode);
    m.impl("gdn_gated_rms_norm_f8", GdnGatedRmsNormF8);
    m.impl("lfm3_short_conv_decode", Lfm3ShortConvDecode);
}
TORCH_LIBRARY_IMPL(custom_op, Meta, m)
{
    m.impl("gaudi2_add_rms_norm_f8", AddRmsNormF8Meta);
    m.impl("gaudi2_add_rms_norm_f8_bf16", AddRmsNormF8Bf16Meta);
    m.impl("gaudi2_sandwich_rms_norm_f8", SandwichRmsNormF8Meta);
    m.impl("gaudi2_silu_mul_f8", SiluMulF8Meta);
    m.impl("gdn_prep_decode", GdnPrepDecodeMeta);
    m.impl("gdn_gated_rms_norm_f8", GdnGatedRmsNormF8Meta);
    m.impl("lfm3_short_conv_decode", Lfm3ShortConvDecodeMeta);
}

static const bool kRegistered = Register();
