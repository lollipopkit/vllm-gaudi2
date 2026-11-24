# SPDX-License-Identifier: Apache-2.0
"""Decode layer groups: several decoder layers (and optionally the final norm) in one torch.compile region.

Fewer compiled regions per decode step means fewer Dynamo guard checks and graph launches, which dominate host time
for small batches. Used by HpuQwen3NextModel and HpuLfm3Model; the model runner builds the groups
(HPUModelRunner._maybe_compile_layer_groups) before its per-layer compilation, so the groups capture the original
layers. Derived from 1CatAI/1Cat-vLLM-Gaudi (Apache-2.0) at e2266e0.
"""
from itertools import islice
import types

import torch
from vllm.forward_context import get_forward_context


class DecoderLayerGroup(torch.nn.Module):
    """Decoder layers called as layer(positions=, hidden_states=, residual=) -> (hidden_states, residual)."""

    def __init__(self,
                 layers: tuple[torch.nn.Module, ...],
                 final_norm: torch.nn.Module | None = None,
                 capture_after: tuple[int, ...] = ()):
        super().__init__()
        # The model's ModuleList stays the sole owner of the layers so state-dict and KV-cache
        # layer names are unchanged.
        object.__setattr__(self, "_layers", layers)
        object.__setattr__(self, "_final_norm", final_norm)
        # Positions (within the group) of layers whose output, hidden_states + residual, is an aux hidden state
        # (speculative decoding drafters read them); returned as extra outputs in that order.
        object.__setattr__(self, "_capture_after", capture_after)

    def forward(self, *, positions: torch.Tensor, hidden_states: torch.Tensor, residual: torch.Tensor | None = None):
        captures = []
        for i, layer in enumerate(self._layers):
            hidden_states, residual = layer(positions=positions, hidden_states=hidden_states, residual=residual)
            if i in self._capture_after:
                captures.append(hidden_states + residual if residual is not None else hidden_states)
        if self._final_norm is not None:
            hidden_states, residual = self._final_norm(hidden_states, residual)
        return hidden_states, residual, *captures


def compile_layer_groups(model: torch.nn.Module,
                         group_size: int,
                         compile_fn,
                         final_norm: torch.nn.Module | None,
                         layer_kinds: tuple | None = None,
                         first_group_null_residual: bool = False) -> int:
    """Attach compiled decode layer groups of model.layers[start_layer:end_layer] to ``model``; must run before
    per-layer compilation. Aux hidden-state layers (model.aux_hidden_state_layers, if any) are captured.
    layer_kinds (one per layer of that range) separates the code objects of groups whose layers trace
    differently, e.g. LFM3's conv / attention layers, which share one layer class."""
    layers = tuple(islice(model.layers, model.start_layer, model.end_layer))
    kinds = layer_kinds or (None, ) * len(layers)
    aux = set(getattr(model, "aux_hidden_state_layers", ()))  # i + 1 marks the output of layer i
    groups = [
        DecoderLayerGroup(layers[i:i + group_size],
                          capture_after=tuple(j for j in range(len(layers[i:i + group_size]))
                                              if model.start_layer + i + j + 1 in aux))
        for i in range(0, len(layers), group_size)
    ]
    if groups and final_norm is not None:
        object.__setattr__(groups[-1], "_final_norm", final_norm)
    # One forward code object per group structure: Dynamo keys its cache on the code object, so groups that
    # trace to different graphs (aux captures, final norm, size, layer kinds, first-rank null residual) never share
    # cache entries and the guards need not tell them apart.
    variants: dict[tuple, type] = {}
    for i, g in enumerate(groups):
        key = (
            len(g._layers),
            kinds[i * group_size:(i + 1) * group_size],
            g._capture_after,
            g._final_norm is not None,
            first_group_null_residual and i == 0,
        )
        if key not in variants:
            fwd = DecoderLayerGroup.forward
            code = fwd.__code__.replace(co_name=f"forward_v{len(variants)}")
            variants[key] = type(
                f"DecoderLayerGroupV{len(variants)}", (DecoderLayerGroup, ),
                {"forward": types.FunctionType(code, fwd.__globals__, code.co_name, fwd.__defaults__, fwd.__closure__)})
        g.__class__ = variants[key]
    object.__setattr__(model, "_hpu_compiled_layer_groups", tuple(compile_fn(g) for g in groups))
    object.__setattr__(model, "_hpu_layer_groups_include_final_norm", final_norm is not None)
    return len(groups)


class LayerGroupModel:
    """Mixin for decoder models whose decode steps can run layer groups; HPUModelRunner._maybe_compile_layer_groups
    builds them when the model's group-size variable is > 1. The model's forward runs the groups when
    use_layer_groups() says so."""
    # config attribute (vllm_gaudi features) holding the group size; must be set by the model
    layer_group_size_env: str
    # TP > 1 is unvalidated for every model so far
    layer_groups_tp1_only = True

    def hpu_layer_group_final_norm(self) -> torch.nn.Module | None:
        """The final norm to fuse into the last group (None on a non-last PP rank)."""
        raise NotImplementedError

    def hpu_layer_kinds(self) -> tuple | None:
        """One kind per layer of the model's range (config.layer_types): groups whose layers trace differently
        must not share a code object, since the groups' module guards are skipped (static_modules)."""
        kinds = getattr(self.config, "layer_types", None)
        return tuple(kinds[self.start_layer:self.end_layer]) if kinds else None


def use_layer_groups(model: torch.nn.Module, batch_size: int, max_batch_size: int | None = None) -> bool:
    """Whether this step runs the layer groups: decode steps only, up to max_batch_size (None: any)."""
    if getattr(model, "_hpu_compiled_layer_groups", None) is None:
        return False
    metadata = get_forward_context().attn_metadata
    if isinstance(metadata, dict):
        metadata = next(iter(metadata.values()), None)
    return (metadata is not None and not bool(getattr(metadata, "is_prompt", False))
            and (max_batch_size is None or batch_size <= max_batch_size))
