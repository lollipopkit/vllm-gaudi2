# SPDX-License-Identifier: Apache-2.0
"""LFM3 on HPU: decode layer groups (models/layer_groups.py, VLLM_HPU_LFM3_COMPILE_LAYER_GROUP_SIZE).

LFM3 decode is host-bound (docs/lfm3.md: ~2.4 ms of device work in a ~10 ms step at batch 1); every decoder layer
is a compiled region with its own dispatch and Dynamo cache lookup. Groups of layers run as one region per group
in decode steps; prefill keeps the per-layer regions.
"""
from itertools import islice

import torch
from vllm.distributed import get_pp_group
from vllm.model_executor.layers.mamba.short_conv import ShortConv
from vllm_gaudi.models.lfm3_base import Lfm3Attention, Lfm3ForCausalLM, Lfm3Model
from vllm.sequence import IntermediateTensors

from vllm_gaudi.models.layer_groups import LayerGroupModel, use_layer_groups


def configure_hpu_lfm3_gated_fp8(model: torch.nn.Module) -> int:
    """Keep LFM3 gated output projections on abs/amax FP8 scale when CGUID dynamic quant is on.

    On Gaudi2, fusing a multiplicative gate producer into the CGUID scale op + FP8 GEMM can
    corrupt outputs (see qwen3_next.configure_hpu_qwen3_gated_fp8); opt Lfm3Attention.o_proj
    (sigmoid gate) and ShortConv.out_proj (C * conv_out gate) out of CGUID scale.
    """
    from vllm_gaudi.extension import ops
    from vllm_gaudi.ops.hpu_compressed_tensors import HPUCompressedTensorsLinearMethod
    from vllm_gaudi.ops.hpu_fp8 import Fp8LinearMethod

    if not ops.is_hpu_gaudi2:
        return 0
    count = 0
    for mod in model.modules():
        proj = None
        if isinstance(mod, Lfm3Attention):
            proj = mod.o_proj
        elif isinstance(mod, ShortConv):
            proj = mod.out_proj
        if proj is not None and isinstance(
                getattr(proj, "quant_method", None),
            (Fp8LinearMethod, HPUCompressedTensorsLinearMethod),
        ):
            proj._hpu_avoid_cguid_dynamic_quant = True
            count += 1
    return count


class HpuLfm3Model(LayerGroupModel, Lfm3Model):
    """Lfm3Model.forward (lfm3_base.py) with the decode layer-group branch; the per-layer path is the base model's."""
    layer_group_size_env = "VLLM_HPU_LFM3_COMPILE_LAYER_GROUP_SIZE"

    def hpu_layer_group_final_norm(self) -> torch.nn.Module | None:
        return self.embedding_norm if get_pp_group().is_last_rank else None

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        if get_pp_group().is_first_rank:
            hidden_states = inputs_embeds if inputs_embeds is not None else self.embed_input_ids(input_ids)
            residual = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]

        grouped_final_norm = False
        if use_layer_groups(self, hidden_states.shape[0]):
            for group in self._hpu_compiled_layer_groups:
                hidden_states, residual, *captures = group(positions=positions,
                                                           hidden_states=hidden_states,
                                                           residual=residual)
                assert not captures, "LFM3 has no aux hidden-state layers"
            grouped_final_norm = self._hpu_layer_groups_include_final_norm
        else:
            for layer in islice(self.layers, self.start_layer, self.end_layer):
                hidden_states, residual = layer(positions, hidden_states, residual)

        if not get_pp_group().is_last_rank:
            return IntermediateTensors({"hidden_states": hidden_states, "residual": residual})
        if not grouped_final_norm:
            hidden_states, _ = self.embedding_norm(hidden_states, residual)
        return hidden_states


class HpuLfm3ForCausalLM(Lfm3ForCausalLM):

    def __init__(self, *, vllm_config, prefix: str = "") -> None:
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        self.model.__class__ = HpuLfm3Model
        configure_hpu_lfm3_gated_fp8(self.model)
