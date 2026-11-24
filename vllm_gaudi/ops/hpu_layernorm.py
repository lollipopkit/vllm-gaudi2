from typing import Optional, Union
import torch
import torch.nn.functional as F
from vllm.model_executor.layers.layernorm import \
    RMSNorm, GemmaRMSNorm, RMSNormGated

# FusedRMSNorm beats the GC-fused native graph from prefill sizes on ([98304, 128]: 0.33 vs 0.75 ms) but not at
# decode sizes ([1536, 128]: 0.17 vs 0.12 ms; bench/gated_norm_bench.py).
_GATED_FUSED_MIN_ROWS = 4096


@RMSNorm.register_oot
class HPURMSNorm(RMSNorm):

    def forward_oot(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        from vllm_gaudi.extension.kernels import rms_norm
        HPUFusedRMSNorm = rms_norm()
        if residual is not None:
            orig_shape = x.shape
            residual = residual + x.reshape(residual.shape)
            # Note: HPUFusedRMSNorm requires 3D tensors as inputs
            x = HPUFusedRMSNorm.apply(residual, self.weight, self.variance_epsilon)
            return x.reshape(orig_shape), residual

        x = HPUFusedRMSNorm.apply(x, self.weight, self.variance_epsilon)
        return x


@GemmaRMSNorm.register_oot
class HPUGemmaRMSNorm(GemmaRMSNorm):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gemma_weight = None

    def prepare_hpu_constants(self):
        # (1 + w) once after loading in FP32 to match GemmaRMSNorm.forward_native.
        self.gemma_weight = self.weight.float() + 1.0

    def forward_oot(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        from vllm_gaudi.extension.kernels import rms_norm
        HPUFusedRMSNorm = rms_norm()
        # Match GemmaRMSNorm.forward_native: add the offset in FP32.
        # BF16 addition rounds trained weights and can be folded differently
        # by the HPU compiler when this norm feeds a projection.
        gemma_weight = getattr(self, "gemma_weight", None)
        if gemma_weight is None:
            gemma_weight = self.weight.float() + 1.0
        orig_dtype = x.dtype
        if residual is not None:
            orig_shape = x.shape
            # Normalize the FP32 sum before rounding the returned residual,
            # as in the upstream fused_add_rms_norm implementation.
            residual_float = residual.float() + x.reshape(residual.shape).float()
            x = HPUFusedRMSNorm.apply(residual_float, gemma_weight, self.variance_epsilon)
            return x.to(orig_dtype).reshape(orig_shape), residual_float.to(residual.dtype)

        x = HPUFusedRMSNorm.apply(x.float(), gemma_weight, self.variance_epsilon)
        return x.to(orig_dtype)


@RMSNormGated.register_oot
class HPURMSNormGated(RMSNormGated):

    def forward_oot(self, x: torch.Tensor, z: Optional[torch.Tensor] = None) -> torch.Tensor:
        if (z is None or self.group_size is not None or not self.norm_before_gate or x.dim() != 2
                or x.shape[0] < _GATED_FUSED_MIN_ROWS):
            return self.forward_native(x, z)
        from vllm_gaudi.extension.kernels import rms_norm
        out = rms_norm().apply(x.float(), self.weight.float(), self.eps)
        act_fn = F.sigmoid if self.activation == "sigmoid" else F.silu
        return (out * act_fn(z.float())).to(x.dtype)
