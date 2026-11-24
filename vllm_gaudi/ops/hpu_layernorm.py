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
        # (1 + w) once after loading: per call it is a constant + add node pair before every norm.
        self.gemma_weight = self.weight + 1.0

    def forward_oot(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        from vllm_gaudi.extension.kernels import rms_norm
        HPUFusedRMSNorm = rms_norm()
        # GemmaRMSNorm uses (1 + w) instead of w
        gemma_weight = self.gemma_weight if self.gemma_weight is not None else self.weight + 1.0
        if residual is not None:
            orig_shape = x.shape
            residual = residual + x.reshape(residual.shape)
            # Note: HPUFusedRMSNorm requires 3D tensors as inputs
            x = HPUFusedRMSNorm.apply(residual, gemma_weight, self.variance_epsilon)
            return x.reshape(orig_shape), residual

        x = HPUFusedRMSNorm.apply(x, gemma_weight, self.variance_epsilon)
        return x


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
