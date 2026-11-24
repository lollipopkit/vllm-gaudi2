import torch
from torch.nn.parameter import Parameter

import vllm.model_executor.model_loader.utils as hpu_utils
import vllm.model_executor.model_loader.base_loader as base_loader
from vllm.model_executor.layers.quantization.base_config import QuantizeMethodBase
from vllm.model_executor.layers.attention import (Attention, MLAAttention)
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead, UnquantizedEmbeddingMethod
from vllm.model_executor.model_loader.reload import set_torchao_reload_attrs

import vllm_gaudi.extension.ops as hpu_ops
from vllm_gaudi import envs


class HPUFp8LMHeadMethod(QuantizeMethodBase):
    """lm_head as a per-channel FP8 GEMM with dynamic per-token activation scales.

    The bf16 head is the largest single weight read of a decode step (vocab x hidden); FP8 halves it
    (or quarters it relative to an fp32 head_dtype cast, since fp8_gemm_v2 outputs fp32 directly).
    Does NOT inherit from UnquantizedEmbeddingMethod so LogitsProcessor._apply_head never bypasses
    `apply()` to cast raw unscaled FP8 weight bytes directly.
    """

    supports_out_dtype = True

    def create_weights(self, layer: torch.nn.Module, *weight_args, **extra_weight_attrs) -> None:
        raise NotImplementedError("HPUFp8LMHeadMethod is installed post-load by _quantize_lm_heads.")

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
        out_dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        out = hpu_ops.apply_fp8_linear_hpu(
            x.view(-1, x.shape[-1]),
            layer.weight,
            layer.weight_scale_inv,
            bias=bias,
            out_dtype=out_dtype,
        )
        return out.view(*x.shape[:-1], out.shape[-1])  # explicit width: x can have 0 rows (drafter warmup)


_LM_HEAD_QUANT_ROWS = 16384


def _quantize_lm_heads(model) -> None:
    for _, module in model.named_modules():
        # A head shared with a drafter is seen twice; the dtype check skips the second visit.
        if (isinstance(module, ParallelLMHead) and type(module.quant_method) is UnquantizedEmbeddingMethod
                and module.weight.dtype == torch.bfloat16):
            weight = module.weight.data
            qweight = torch.empty_like(weight, dtype=torch.float8_e4m3fn)
            scale = torch.empty(weight.shape[0], dtype=torch.float32, device=weight.device)
            # Row chunks keep the abs/amax temporaries small (the whole head is 2.5 GB in bf16).
            for i in range(0, weight.shape[0], _LM_HEAD_QUANT_ROWS):
                q, s = hpu_ops.dynamic_quant(weight[i:i + _LM_HEAD_QUANT_ROWS], use_cguid=False)
                qweight[i:i + _LM_HEAD_QUANT_ROWS] = q
                scale[i:i + _LM_HEAD_QUANT_ROWS] = s.squeeze(-1)
            module.weight = Parameter(qweight, requires_grad=False)
            module.weight_scale_inv = Parameter(scale, requires_grad=False)
            module.quant_method = HPUFp8LMHeadMethod()
            del weight


def hpu_process_weights_after_loading(model, model_config, target_device):
    """Gaudi override: accept device strings (e.g., "hpu")."""
    target_device = torch.device(target_device)
    for _, module in model.named_modules():
        quant_method = getattr(module, "quant_method", None)
        if isinstance(quant_method, QuantizeMethodBase):
            #with device_loading_context(module, target_device):
            quant_method.process_weights_after_loading(module)

    # Initialize post-load attention weights for both Attention and MLA.
    for _, module in model.named_modules():
        if isinstance(module, (Attention, MLAAttention)) and hasattr(module, "process_weights_after_loading"):
            #with device_loading_context(module, target_device):
            module.process_weights_after_loading(model_config.dtype)

    # Run model-level post-load hooks (idempotent) so non-persistent buffers and
    # load-time FP8 projections are also initialized under `--load-format dummy`.
    if hasattr(model, "process_weights_after_loading"):
        model.process_weights_after_loading()

    if envs.VLLM_HPU_FP8_LM_HEAD:
        _quantize_lm_heads(model)

    if model_config.quantization == "torchao":
        set_torchao_reload_attrs(model, model_config)


hpu_utils.process_weights_after_loading = hpu_process_weights_after_loading
base_loader.process_weights_after_loading = hpu_process_weights_after_loading
