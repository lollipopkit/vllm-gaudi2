// SPDX-License-Identifier: Apache-2.0
// torch.ops.custom_op.gdn_decode / gdn_verify: PT2 custom ops running the gdn_decode_f32_gaudi2 and
// gdn_verify_f32_gaudi2 TPC kernels (libgaudi2_tpc_kernels.so must be on GC_KERNEL_PATH).
#include <torch/extension.h>

#include "hpu_custom_op_pt2.h"

namespace
{
// The HPU torch.compile backend keeps only the "custom_op" namespace inside fused graphs (any other
// namespace falls back to eager and recompiles a one-node recipe on every call).
constexpr const char* kSchema = "custom_op::gdn_decode";
constexpr const char* kGuid = "gdn_decode_f32_gaudi2";
constexpr const char* kVerifySchema = "custom_op::gdn_verify";
constexpr const char* kVerifyGuid = "gdn_verify_f32_gaudi2";

struct Params
{
    int rowOffset;  // GdnDecodeGaudi2::Params
    float scale;
};

habana::PartialOutputMetaDataVector OutputMeta(const at::Stack& inputs)
{
    return {{c10::ScalarType::Float, inputs[2].toTensor().sizes().vec()}};
}

std::shared_ptr<void> FillParams(const at::Stack& inputs, size_t& size)
{
    HPU_PARAMS_STUB(Params);
    params->rowOffset = static_cast<int>(inputs[6].toInt());
    params->scale = static_cast<float>(inputs[7].toDouble());
    return params;
}

std::shared_ptr<void> FillNoParams(const at::Stack&, size_t& size)
{
    HPU_PARAMS_STUB(Params);  // the verify kernel has no scalar params
    return params;
}

bool Register()
{
    habana::custom_op::registerUserCustomOp(kSchema, kGuid, OutputMeta, FillParams);
    habana::custom_op::registerUserCustomOp(kVerifySchema, kVerifyGuid, OutputMeta, FillNoParams);
    return true;
}

at::Tensor GdnDecode(at::Tensor pool, at::Tensor qk, at::Tensor value, at::Tensor decay, at::Tensor beta,
                     at::Tensor kqdot, int64_t row_offset, double scale)
{
    for (const auto* t : {&pool, &qk, &value, &decay, &beta, &kqdot})
        TORCH_CHECK(t->scalar_type() == c10::ScalarType::Float, "gdn_decode expects float32 tensors");
    TORCH_CHECK(pool.dim() == 4 && pool.size(2) == 128 && pool.size(3) == 128 && pool.is_contiguous(),
                "pool must be a contiguous [slots, HV, 128, 128] tensor");
    TORCH_CHECK(row_offset >= 0 && row_offset + value.size(0) <= pool.size(0), "batch rows outside the pool");
    TORCH_CHECK(value.dim() == 3 && qk.dim() == 3 && qk.size(2) == 128, "value [B, HV, 128], qk [B, 2H, 128]");
    TORCH_CHECK(decay.dim() == 2 && beta.dim() == 2 && kqdot.dim() == 2, "decay, beta [B, HV]; kqdot [B, H]");
    auto op = habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor(kSchema);
    return op.execute({pool, qk, value, decay, beta, kqdot, row_offset, scale})[0];
}

// Meta kernels use the symbolic sizes: value.sizes() would specialize every compiled graph on the batch size.
at::Tensor GdnDecodeMeta(at::Tensor, at::Tensor, at::Tensor value, at::Tensor, at::Tensor, at::Tensor, int64_t,
                         double)
{
    return value.new_empty_symint(value.sym_sizes());
}

at::Tensor GdnVerify(at::Tensor pool, at::Tensor kq, at::Tensor value, at::Tensor coeff, at::Tensor rows)
{
    for (const auto* t : {&pool, &kq, &value, &coeff})
        TORCH_CHECK(t->scalar_type() == c10::ScalarType::Float, "gdn_verify expects float32 tensors");
    TORCH_CHECK(rows.scalar_type() == c10::ScalarType::Int, "rows must be int32");
    TORCH_CHECK(pool.dim() == 4 && pool.size(2) == 128 && pool.size(3) == 128 && pool.is_contiguous(),
                "pool must be a contiguous [slots, HV, 128, 128] tensor");
    TORCH_CHECK(kq.dim() == 5 && kq.size(3) == 2 && kq.size(4) == 128, "kq must be [B, T, H, 2, 128]");
    TORCH_CHECK(value.dim() == 4 && coeff.dim() == 4 && coeff.size(3) == 3,
                "value [B, T, HV, 128], coeff [B, T, HV, 3]");
    TORCH_CHECK(rows.dim() == 2 && rows.size(0) == value.size(0) && rows.size(1) == value.size(1) + 1,
                "rows must be [B, 1 + T]");
    auto op = habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor(kVerifySchema);
    return op.execute({pool, kq, value, coeff, rows})[0];
}

at::Tensor GdnVerifyMeta(at::Tensor, at::Tensor, at::Tensor value, at::Tensor, at::Tensor)
{
    return value.new_empty_symint(value.sym_sizes());
}
}  // namespace

TORCH_LIBRARY_FRAGMENT(custom_op, m)
{
    // KV-major f32 state pool [slots, HV, K, V], updated in place in slots row_offset .. row_offset + B - 1
    // (the TPC kernel writes its input; the schema stays functional so graphs keep the op as is).
    // qk [B, 2H, K] (L2-normalized q heads, then k heads; q unscaled); value [B, HV, V]; decay, beta [B, HV];
    // kqdot [B, H] (k . q, unscaled); the output is multiplied by scale. Returns out [B, HV, V].
    m.def("gdn_decode(Tensor pool, Tensor qk, Tensor value, Tensor decay, Tensor beta, Tensor kqdot, int row_offset, "
          "float scale) -> Tensor");
    // Spec-decode verification of T = 1 + k tokens per request on the same pool: initial state from row
    // rows[b, 0], the state after token t written to row rows[b, 1 + t]. kq [B, T, H, 2, K]; value
    // [B, T, HV, V]; coeff [B, T, HV, 3]; rows [B, 1 + T] int32. Returns out [B, T, HV, V].
    m.def("gdn_verify(Tensor pool, Tensor kq, Tensor value, Tensor coeff, Tensor rows) -> Tensor");
}
TORCH_LIBRARY_IMPL(custom_op, HPU, m)
{
    m.impl("gdn_decode", GdnDecode);
    m.impl("gdn_verify", GdnVerify);
}
TORCH_LIBRARY_IMPL(custom_op, Meta, m)
{
    m.impl("gdn_decode", GdnDecodeMeta);
    m.impl("gdn_verify", GdnVerifyMeta);
}

static const bool kRegistered = Register();
