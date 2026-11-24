# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Looped Inkling2 model (`inkling2_loop`) on Gaudi2 HPU."""

from __future__ import annotations

import os
import types
from collections.abc import Callable

import torch
from torch import nn

from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.logger import init_logger

from ..configs import Inkling2ModelConfig
from ..quant_config import Inkling2FP8QuantConfig
from .attention import (
    DecodeAttnContext,
    build_decode_attn_context,
    compute_decode_attn_context_tensors,
)
from .model import (
    Inkling2HPUDecoderLayer,
    Inkling2HPUForConditionalGeneration,
    Inkling2HPUModel,
)
from .ops import (
    GAUDI2_FP8_MAX,
    attn_res,
    blend_rms_norm,
    blend_rms_norm_f8,
    compute_log_scaling_tau,
    pair_rms_norm,
)
from .tpc_ops import has_tpc_blend, has_tpc_paged_attn

logger = init_logger(__name__)
LOOP_STREAMS = ("attn", "k", "v", "mlp")


class Inkling2HPUMessageController(nn.Module):
    """Per-stream low-rank gated message controller (weights alias stacked buffers)."""

    def __init__(
        self, down: torch.Tensor, gain: torch.Tensor, forget: torch.Tensor
    ) -> None:
        super().__init__()
        self.down_weight = nn.Parameter(down, requires_grad=False)
        self.gain_weight = nn.Parameter(gain, requires_grad=False)
        self.forget_weight = nn.Parameter(forget, requires_grad=False)


class Inkling2HPULoopDecoderLayer(Inkling2HPUDecoderLayer):
    """A loop core layer with per-pass caches and stacked message controllers."""

    def __init__(
        self,
        config: Inkling2ModelConfig,
        layer_id: int,
        quant_config: Inkling2FP8QuantConfig | None,
        prefix: str,
    ) -> None:
        super().__init__(
            config, layer_id, quant_config, prefix, num_passes=config.num_loop_passes
        )
        assert config.loop_config is not None
        rank, width = config.loop_config["gate_rank"], config.hidden_size
        streams = len(LOOP_STREAMS)
        self.register_buffer(
            "controller_down",
            torch.empty(streams, rank, 2 * width, dtype=torch.bfloat16),
            persistent=False,
        )
        self.register_buffer(
            "controller_gain_forget",
            torch.empty(streams, 2 * width, rank, dtype=torch.bfloat16),
            persistent=False,
        )
        self.controllers = nn.ModuleDict(
            {
                stream: Inkling2HPUMessageController(
                    self.controller_down[i],
                    self.controller_gain_forget[i, :width],
                    self.controller_gain_forget[i, width:],
                )
                for i, stream in enumerate(LOOP_STREAMS)
            }
        )

    def process_weights_after_loading(self) -> None:
        # Explicitly stack per-stream controller parameters into contiguous buffers
        # (PyTorch `.to(device)` breaks initial view aliasing between buffers/params).
        width = self.controller_gain_forget.shape[1] // 2
        for i, stream in enumerate(LOOP_STREAMS):
            controller = self.controllers[stream]
            self.controller_down[i].copy_(controller.down_weight)
            self.controller_gain_forget[i, :width].copy_(controller.gain_weight)
            self.controller_gain_forget[i, width:].copy_(controller.forget_weight)

    def _controller_gain_forget(
        self,
        streams: slice,
        dest: torch.Tensor,
        message: torch.Tensor,
    ) -> torch.Tensor:
        pair = pair_rms_norm(dest, message)  # [S, T, 2D] bf16
        down = self.controller_down[streams]  # [S, r, 2D] bf16
        up = self.controller_gain_forget[streams]  # [S, 2D, r] bf16
        hidden = torch.bmm(pair, down.transpose(1, 2))  # [S, T, r] bf16
        if dest.device.type == "hpu" and has_tpc_blend():
            return torch.bmm(hidden, up.transpose(1, 2))  # [S, T, 2D] bf16 (fp32 MME acc)
        return torch.bmm(hidden.float(), up.transpose(1, 2).float())  # [S, T, 2D] fp32

    def blend(
        self,
        streams: slice,
        dest: torch.Tensor,
        message: torch.Tensor,
        norm_weight: torch.Tensor,
        norm_eps: float,
    ) -> torch.Tensor:
        """Blend fp32 `(S, T, D)` `dest` and `message` and apply RMSNorm -> bf16 `(S, T, D)`."""
        gain_forget = self._controller_gain_forget(streams, dest, message)
        return blend_rms_norm(dest, message, gain_forget, norm_weight, norm_eps)

    def blend_f8(
        self,
        streams: slice,
        dest: torch.Tensor,
        message: torch.Tensor,
        norm_weight: torch.Tensor,
        norm_eps: float,
        *,
        fp8_max: float = GAUDI2_FP8_MAX,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        """Blend + RMSNorm + fused FP8 E4M3 quant -> `(bf16, fp8 | None, scale | None)`."""
        gain_forget = self._controller_gain_forget(streams, dest, message)
        return blend_rms_norm_f8(dest, message, gain_forget, norm_weight, norm_eps, fp8_max)


class Inkling2HPUDecodePrologue(nn.Module):
    """Compiled decode prologue: computes `tau` and `DecodeAttnContext` in 1 HPU recipe."""

    def __init__(
        self,
        *,
        n_floor: int | None,
        alpha: float,
        sliding_window: int,
        rel_extent: int,
    ) -> None:
        super().__init__()
        self._n_floor = n_floor
        self._alpha = alpha
        self._sliding_window = sliding_window
        self._rel_extent = rel_extent

    def forward(
        self,
        positions: torch.Tensor,
        block_groups: torch.Tensor,
        block_usage: torch.Tensor,
        batch_size: int,
        block_size: int,
        block_logical_idx: torch.Tensor | None = None,
        block_list: torch.Tensor | None = None,
        slot_mapping: torch.Tensor | None = None,
        kv_num_slots: int = 0,
        load_indices: torch.Tensor | None = None,
        store_indices: torch.Tensor | None = None,
        conv_num_slots: int = 0,
        tpc_only: bool = False,
    ) -> tuple[torch.Tensor | None, DecodeAttnContext]:
        tau = compute_log_scaling_tau(positions, self._n_floor, self._alpha)
        decode_ctx = compute_decode_attn_context_tensors(
            block_groups,
            block_usage,
            positions,
            batch_size,
            block_size=block_size,
            sliding_window=self._sliding_window,
            rel_extent=self._rel_extent,
            block_logical_idx=block_logical_idx,
            block_list=block_list,
            slot_mapping=slot_mapping,
            kv_num_slots=kv_num_slots,
            load_indices=load_indices,
            store_indices=store_indices,
            conv_num_slots=conv_num_slots,
            tpc_only=tpc_only,
        )
        return tau, decode_ctx


class Inkling2HPULoopDecodeGroup(nn.Module):
    """Static decode-only compiled region for `G` consecutive loop visits.

    All per-layer modules, queries, norms, and pass indices are captured as
    non-submodule static attributes (via `object.__setattr__`) so state_dict
    keys remain exclusively under `model.layers.*` and zero Python lists are
    passed across the `torch.compile` boundary. Each group receives its own
    distinct `forward` code object so Dynamo cache entries never collide under
    `VLLM_FULL_WARMUP=1` (`skip_guard_eval_unsafe`).
    """

    def __init__(
        self,
        *,
        layers: tuple[Inkling2HPUDecoderLayer, ...],
        first_flags: tuple[bool, ...],
        pass_idx: int,
        mlp_queries: tuple[torch.Tensor, ...],
        next_queries: tuple[torch.Tensor, ...],
        next_norm_weights: tuple[torch.Tensor | None, ...],
        next_norm_eps: tuple[float, ...],
        message_queries: tuple[torch.Tensor, ...] | None,
        prelude_entry_query: torch.Tensor | None = None,
        prelude_num_blocks: int = 0,
        need_next_f8: tuple[bool, ...] | None = None,
    ) -> None:
        super().__init__()
        object.__setattr__(self, "_layers", layers)
        object.__setattr__(self, "_first_flags", first_flags)
        object.__setattr__(self, "_pass_idx", pass_idx)
        object.__setattr__(self, "_mlp_queries", mlp_queries)
        object.__setattr__(self, "_next_queries", next_queries)
        object.__setattr__(self, "_next_norm_weights", next_norm_weights)
        object.__setattr__(self, "_next_norm_eps", next_norm_eps)
        object.__setattr__(self, "_message_queries", message_queries)
        object.__setattr__(self, "_prelude_entry_query", prelude_entry_query)
        object.__setattr__(self, "_prelude_num_blocks", prelude_num_blocks)
        object.__setattr__(self, "_need_next_f8", need_next_f8)

    def forward(
        self,
        positions: torch.Tensor,
        completed_blocks: torch.Tensor | None,
        partial: torch.Tensor,
        inputs: torch.Tensor | None,
        tau: torch.Tensor | None,
        decode_ctx: DecodeAttnContext,
        bank: torch.Tensor | None = None,
        batch_size: int = 1,
        prelude_completed: torch.Tensor | None = None,
        prelude_partial: torch.Tensor | None = None,
        inputs_f8: torch.Tensor | None = None,
        inputs_sc: torch.Tensor | None = None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
    ]:
        # Fold pass-boundary bank + prelude_dests into the compiled group (zero eager ops).
        if self._prelude_entry_query is not None:
            assert completed_blocks is not None
            assert prelude_completed is not None and prelude_partial is not None
            bank = torch.cat(
                [completed_blocks[self._prelude_num_blocks + 1 :], partial.unsqueeze(0)],
                dim=0,
            )
            cur_completed = prelude_completed
            cur_partial = prelude_partial
            cur_inputs = attn_res(
                cur_completed,
                cur_partial,
                None,
                self._prelude_entry_query,
                None,
                0.0,
                True,
            )
            cur_inputs_f8: torch.Tensor | None = None
            cur_inputs_sc: torch.Tensor | None = None
        else:
            cur_completed = (
                partial.new_empty((0, partial.shape[0], partial.shape[1]))
                if completed_blocks is None
                else completed_blocks
            )
            cur_partial = partial
            cur_inputs = inputs
            cur_inputs_f8 = inputs_f8
            cur_inputs_sc = inputs_sc

        pass_idx = self._pass_idx
        msg_queries = self._message_queries
        need_f8_flags = self._need_next_f8
        for idx, layer in enumerate(self._layers):
            first = self._first_flags[idx]
            if need_f8_flags is not None:
                need_f8 = need_f8_flags[idx]
                res = layer(
                    positions,
                    cur_completed,
                    cur_partial,
                    cur_inputs,
                    self._mlp_queries[idx],
                    self._next_queries[idx],
                    self._next_norm_weights[idx],
                    self._next_norm_eps[idx],
                    tau,
                    first=first,
                    pass_idx=pass_idx,
                    batch_size=batch_size,
                    decode_ctx=decode_ctx,
                    bank=bank,
                    message_query=None if msg_queries is None else msg_queries[idx],
                    inputs_f8=cur_inputs_f8,
                    inputs_sc=cur_inputs_sc,
                    return_next_f8=need_f8,
                )
                if need_f8:
                    new_partial, cur_inputs, cur_inputs_f8, cur_inputs_sc = res
                else:
                    new_partial, cur_inputs = res
                    cur_inputs_f8 = cur_inputs_sc = None
            else:
                new_partial, cur_inputs = layer(
                    positions,
                    cur_completed,
                    cur_partial,
                    cur_inputs,
                    self._mlp_queries[idx],
                    self._next_queries[idx],
                    self._next_norm_weights[idx],
                    self._next_norm_eps[idx],
                    tau,
                    first=first,
                    pass_idx=pass_idx,
                    batch_size=batch_size,
                    decode_ctx=decode_ctx,
                    bank=bank,
                    message_query=None if msg_queries is None else msg_queries[idx],
                )
                cur_inputs_f8 = cur_inputs_sc = None
            if first:
                cur_completed = (
                    torch.cat([cur_completed, cur_partial.unsqueeze(0)], dim=0)
                    if cur_completed.shape[0] > 0
                    else cur_partial.unsqueeze(0)
                )
            cur_partial = new_partial
        assert cur_inputs is not None
        return cur_completed, cur_partial, cur_inputs, bank, cur_inputs_f8, cur_inputs_sc


class Inkling2HPULoopModel(Inkling2HPUModel):
    def __init__(
        self,
        config: Inkling2ModelConfig,
        quant_config: Inkling2FP8QuantConfig | None,
        prefix: str,
    ) -> None:
        config.check_loop_options()
        super().__init__(config, quant_config, prefix)
        self.loop_core = config.loop_core_layers
        self.hidden_size = config.hidden_size
        shape = (2 * len(self.loop_core), config.hidden_size)
        self.middle_cycle_cross_pass_queries = nn.Parameter(
            torch.zeros(shape, dtype=torch.float32), requires_grad=False
        )
        self.middle_cycle_cross_pass_kv_queries = nn.Parameter(
            torch.zeros(shape, dtype=torch.float32), requires_grad=False
        )
        self.register_buffer(
            "message_queries",
            torch.zeros(
                len(self.loop_core),
                len(LOOP_STREAMS),
                config.hidden_size,
                dtype=torch.float32,
            ),
            persistent=False,
        )
        self._visits: tuple[tuple[int, int], ...] = tuple(
            [(i, 0) for i in range(self.loop_core.start)]
            + [(i, p) for p in range(config.num_loop_passes) for i in self.loop_core]
            + [(i, 0) for i in range(self.loop_core.stop, len(self.layers))]
        )

    def compile_hpu_decode_groups(
        self, compile_fn: Callable[[nn.Module], nn.Module]
    ) -> int:
        """Build and compile decode-only multi-layer groups before per-layer compilation.

        Called by `HPUModelRunner._maybe_compile_layer_groups` on TP=1 so decode
        steps run 8 fused regions instead of 48 while prefill keeps per-layer regions.
        """
        if getattr(self, "_hpu_decode_groups", None) is not None:
            return len(self._hpu_decode_groups)
        g = int(os.environ.get("INKLING2_COMPILE_GROUP_SIZE", "6"))
        core = self.loop_core
        if g <= 1:
            return 0
        if get_tensor_model_parallel_world_size() != 1:
            logger.warning(
                "INKLING2_COMPILE_GROUP_SIZE=%d requires TP=1; using per-layer compilation.",
                g,
            )
            return 0
        if core.start % g != 0 or core.stop % g != 0 or len(self.layers) % g != 0:
            logger.warning(
                "INKLING2_COMPILE_GROUP_SIZE=%d does not divide core=(%d,%d) and num_layers=%d; "
                "using per-layer compilation.",
                g,
                core.start,
                core.stop,
                len(self.layers),
            )
            return 0

        visits = self._visits
        total_visits = len(visits)
        groups: list[tuple[int, int, nn.Module]] = []
        for g_idx, v_start in enumerate(range(0, total_visits, g)):
            first_i, pass_idx = visits[v_start]
            looped = pass_idx > 0
            slice_visits = visits[v_start : v_start + g]
            assert all(p == pass_idx for _, p in slice_visits)
            grp_layers = tuple(self.layers[i] for i, _ in slice_visits)
            first_flags = tuple((i % self.block_size) == 0 for i, _ in slice_visits)
            mlp_qs = tuple(self.attn_res.mlp_queries[i] for i, _ in slice_visits)
            next_qs: list[torch.Tensor] = []
            next_nws: list[torch.Tensor | None] = []
            next_eps: list[float] = []
            for offset, (i, _) in enumerate(slice_visits):
                v_idx = v_start + offset
                if v_idx + 1 == total_visits:
                    next_qs.append(self.attn_res.output_query)
                    next_nws.append(self.norm.weight)
                    next_eps.append(self.norm.eps)
                else:
                    next_i, next_pass = visits[v_idx + 1]
                    next_qs.append(self.attn_res.attn_queries[next_i - 1])
                    if next_pass:
                        next_nws.append(None)
                        next_eps.append(0.0)
                    else:
                        nnorm = self.layers[next_i].attn_norm
                        next_nws.append(nnorm.weight)
                        next_eps.append(nnorm.eps)
            msg_qs = (
                tuple(self.message_queries[i - core.start] for i, _ in slice_visits)
                if looped
                else None
            )
            is_loop_entry = first_i == core.start and looped
            prelude_blocks = (core.start + self.block_size - 1) // self.block_size
            need_next_f8 = tuple(
                nw is not None and (v_start + offset + 1 < total_visits)
                for offset, nw in enumerate(next_nws)
            )
            grp = Inkling2HPULoopDecodeGroup(
                layers=grp_layers,
                first_flags=first_flags,
                pass_idx=pass_idx,
                mlp_queries=mlp_qs,
                next_queries=tuple(next_qs),
                next_norm_weights=tuple(next_nws),
                next_norm_eps=tuple(next_eps),
                message_queries=msg_qs,
                prelude_entry_query=(
                    self.attn_res.attn_queries[core.start - 1] if is_loop_entry else None
                ),
                prelude_num_blocks=prelude_blocks if is_loop_entry else 0,
                need_next_f8=need_next_f8,
            )
            # Distinct forward code object per visit group: Dynamo keys its cache
            # on the code object, preventing cache collisions under skip_guard_eval_unsafe.
            fwd = Inkling2HPULoopDecodeGroup.forward
            code = fwd.__code__.replace(co_name=f"inkling2_decode_g{g_idx}")
            cls = type(
                f"Inkling2HPULoopDecodeGroupV{g_idx}",
                (Inkling2HPULoopDecodeGroup,),
                {
                    "forward": types.FunctionType(
                        code, fwd.__globals__, code.co_name, fwd.__defaults__, fwd.__closure__
                    )
                },
            )
            grp.__class__ = cls
            groups.append((first_i, pass_idx, compile_fn(grp)))

        prologue = Inkling2HPUDecodePrologue(
            n_floor=self.config.log_scaling_n_floor,
            alpha=self.config.log_scaling_alpha,
            sliding_window=self.config.sliding_window_size,
            rel_extent=self.config.rel_extent,
        )
        object.__setattr__(self, "_hpu_decode_prologue", compile_fn(prologue))
        object.__setattr__(self, "_hpu_compiled_embed_norm", compile_fn(self.embed_norm))
        object.__setattr__(self, "_hpu_decode_groups", tuple(groups))
        return len(groups)

    def process_weights_after_loading(self) -> None:
        queries = self.middle_cycle_cross_pass_queries.view(-1, 2, self.hidden_size)
        kv_queries = self.middle_cycle_cross_pass_kv_queries.view(-1, 2, self.hidden_size)
        stacked = torch.stack(
            (queries[:, 0], kv_queries[:, 0], kv_queries[:, 1], queries[:, 1]), dim=1
        )
        self.message_queries.copy_(stacked * (self.hidden_size**-0.5))

    def _make_layer(
        self,
        config: Inkling2ModelConfig,
        layer_id: int,
        quant_config: Inkling2FP8QuantConfig | None,
        prefix: str,
    ) -> Inkling2HPUDecoderLayer:
        if layer_id in config.loop_core_layers:
            return Inkling2HPULoopDecoderLayer(config, layer_id, quant_config, prefix)
        return super()._make_layer(config, layer_id, quant_config, prefix)

    def forward(
        self,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor,
        batch_size: int = 1,
    ) -> torch.Tensor:
        from vllm.forward_context import get_forward_context

        num_tokens, hidden = inputs_embeds.shape
        core = self.loop_core
        attn_md = get_forward_context().attn_metadata
        if isinstance(attn_md, dict):
            attn_md = next(iter(attn_md.values()), None)
        is_decode = attn_md is not None and not getattr(attn_md, "is_prompt", True)
        eff_bs = num_tokens if is_decode else batch_size

        # Decode fast path: compiled prologue + static compiled layer groups (zero eager ops).
        decode_groups = getattr(self, "_hpu_decode_groups", None)
        decode_prologue = getattr(self, "_hpu_decode_prologue", None)
        if (
            decode_groups is not None
            and decode_prologue is not None
            and is_decode
            and getattr(attn_md, "block_groups", None) is not None
            and getattr(attn_md, "block_usage", None) is not None
        ):
            if not getattr(self, "_hpu_slot_meta_ready", False):
                self._init_hpu_slot_meta()
            block_size = int(getattr(attn_md, "block_size", 128))
            block_list = getattr(attn_md, "block_list", None)
            tpc_only = (
                positions.device.type == "hpu"
                and inputs_embeds.dtype == torch.bfloat16
                and has_tpc_paged_attn()
                and getattr(self, "_hpu_kv_bf16", False)
                and getattr(self, "_hpu_tpc_pa_shapes_ok", False)
                and block_list is not None
                and 0 < block_list.shape[0] <= 512
            )
            tau, decode_ctx = decode_prologue(
                positions,
                attn_md.block_groups,
                attn_md.block_usage,
                eff_bs,
                block_size,
                getattr(attn_md, "block_logical_idx", None),
                block_list,
                getattr(attn_md, "slot_mapping", None),
                getattr(self, "_hpu_kv_num_slots", 0),
                getattr(attn_md, "load_indices_tensor", None),
                getattr(attn_md, "store_indices_tensor", None),
                getattr(self, "_hpu_conv_num_slots", 0),
                tpc_only,
            )
            return self._forward_decode_grouped(
                positions, inputs_embeds, tau, eff_bs, decode_ctx, decode_groups
            )

        tau = self._log_scaling_tau(positions)
        decode_ctx = build_decode_attn_context(
            attn_md,
            positions,
            eff_bs,
            sliding_window=self.config.sliding_window_size,
            rel_extent=self.config.rel_extent,
        )

        empty_blocks = inputs_embeds.new_empty((0, num_tokens, hidden))
        blocks: list[torch.Tensor] = []
        completed = empty_blocks
        partial = inputs_embeds
        visits = self._visits

        inputs: torch.Tensor | None = None
        inputs_f8: torch.Tensor | None = None
        inputs_sc: torch.Tensor | None = None
        bank: torch.Tensor | None = None
        prelude_blocks_list: list[torch.Tensor] | None = None
        prelude_completed: torch.Tensor | None = None
        prelude_partial: torch.Tensor | None = None
        prelude_dests: torch.Tensor | None = None

        for visit, (i, pass_idx) in enumerate(visits):
            looped = pass_idx > 0
            if i == core.start and not looped:
                prelude_blocks_list = list(blocks)
                prelude_completed = completed
                prelude_partial = partial
            elif i == core.start:
                assert prelude_blocks_list is not None and prelude_partial is not None
                assert prelude_completed is not None
                bank = torch.stack([*blocks[len(prelude_blocks_list) + 1 :], partial], dim=0)
                blocks = list(prelude_blocks_list)
                completed = prelude_completed
                partial = prelude_partial
                if prelude_dests is None:
                    query = self.attn_res.attn_queries[i - 1]
                    prelude_dests = attn_res(
                        completed,
                        prelude_partial,
                        None,
                        query,
                        None,
                        0.0,
                        True,
                    )
                inputs = prelude_dests
                inputs_f8 = inputs_sc = None

            if visit + 1 == len(visits):
                next_query, next_norm = self.attn_res.output_query, self.norm
                need_f8 = False
            else:
                next_i, next_pass = visits[visit + 1]
                next_query = self.attn_res.attn_queries[next_i - 1]
                next_norm = None if next_pass else self.layers[next_i].attn_norm
                need_f8 = next_norm is not None

            first = i % self.block_size == 0
            layer = self.layers[i]
            res = layer(
                positions,
                completed,
                partial,
                inputs,
                self.attn_res.mlp_queries[i],
                next_query,
                None if next_norm is None else next_norm.weight,
                0.0 if next_norm is None else next_norm.eps,
                tau,
                first=first,
                pass_idx=pass_idx,
                batch_size=batch_size,
                decode_ctx=decode_ctx,
                bank=bank if looped else None,
                message_query=self.message_queries[i - core.start] if looped else None,
                inputs_f8=inputs_f8,
                inputs_sc=inputs_sc,
                return_next_f8=need_f8,
            )
            if need_f8:
                new_partial, inputs, inputs_f8, inputs_sc = res
            else:
                new_partial, inputs = res
                inputs_f8 = inputs_sc = None
            if first:
                blocks.append(partial)
                completed = torch.stack(blocks, dim=0)
            partial = new_partial

        assert inputs is not None
        return inputs[0]

    def _init_hpu_slot_meta(self) -> None:
        """Cache static KV/conv cache sizes and integer cache-group indices once after KV binding."""
        kv0 = self.layers[0].attn.kv_cache
        cs0 = self.layers[0].conv_state._state_buffer()
        if (
            not isinstance(kv0, (list, tuple))
            or len(kv0) < 2
            or kv0[0].numel() == 0
            or cs0.numel() == 0
        ):
            return
        kv_num_slots = int(kv0[0].shape[0]) if kv0[0].ndim == 3 else 0
        kv_bf16 = kv0[0].dtype == torch.bfloat16 and kv0[1].dtype == torch.bfloat16
        tpc_pa_shapes_ok = all(
            layer.attn.head_dim == 128
            and layer.attn.num_heads <= 64
            and layer.attn.d_rel <= 64
            for layer in self.layers
        )
        conv_num_slots = int(cs0.shape[0])
        for layer in self.layers:
            for cs in layer.conv_states:
                cg = getattr(cs, "cache_group_idx", None)
                if cg is not None and getattr(cs, "_cache_group_int", None) is None:
                    object.__setattr__(cs, "_cache_group_int", int(cg.item()))
        object.__setattr__(self, "_hpu_kv_num_slots", kv_num_slots)
        object.__setattr__(self, "_hpu_kv_bf16", kv_bf16)
        object.__setattr__(self, "_hpu_tpc_pa_shapes_ok", tpc_pa_shapes_ok)
        object.__setattr__(self, "_hpu_conv_num_slots", conv_num_slots)
        object.__setattr__(self, "_hpu_slot_meta_ready", True)

    def _forward_decode_grouped(
        self,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor,
        tau: torch.Tensor | None,
        batch_size: int,
        decode_ctx: DecodeAttnContext,
        decode_groups: tuple[tuple[int, int, nn.Module], ...],
    ) -> torch.Tensor:
        core = self.loop_core
        completed: torch.Tensor | None = None
        partial = inputs_embeds
        inputs: torch.Tensor | None = None
        inputs_f8: torch.Tensor | None = None
        inputs_sc: torch.Tensor | None = None
        bank: torch.Tensor | None = None
        prelude_completed: torch.Tensor | None = None
        prelude_partial: torch.Tensor | None = None

        for first_i, pass_idx, group in decode_groups:
            looped = pass_idx > 0
            if first_i == core.start and not looped:
                prelude_completed = completed
                prelude_partial = partial
            is_loop_entry = first_i == core.start and looped
            completed, partial, inputs, bank, inputs_f8, inputs_sc = group(
                positions,
                completed,
                partial,
                inputs,
                tau,
                decode_ctx,
                bank if (looped and not is_loop_entry) else None,
                batch_size,
                prelude_completed if is_loop_entry else None,
                prelude_partial if is_loop_entry else None,
                None if is_loop_entry else inputs_f8,
                None if is_loop_entry else inputs_sc,
            )

        assert inputs is not None
        return inputs[0]


class Inkling2HPULoopForConditionalGeneration(Inkling2HPUForConditionalGeneration):
    """Architecture class for `inkling2_loop` checkpoints on Gaudi2 HPU."""
