# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inkling2 base model (HPU / Intel Gaudi2)."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Callable, Iterable, Sequence

import torch
from torch import nn

from vllm.config import VllmConfig
from vllm.distributed import (
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_reduce,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.interfaces import (
    MultiModalEmbeddings,
    SupportsMultiModal,
)
from vllm.model_executor.models.utils import _merge_multimodal_embeddings
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.sequence import IntermediateTensors

from ..configs import Inkling2MMConfig, Inkling2ModelConfig
from ..processing import (
    IMAGE_PLACEHOLDER,
    Inkling2DummyInputsBuilder,
    Inkling2MultiModalProcessor,
    Inkling2ProcessingInfo,
)
from ..quant_config import GAUDI2_FP8_MAX, Inkling2FP8QuantConfig
from .attention import (
    DecodeAttnContext,
    Inkling2HPUAttention,
    build_decode_attn_context,
)
from .moe import _MOE_TPC_MAX_TOKENS, Inkling2HPUMoE
from .ops import attn_res, attn_res_f8, compute_log_scaling_tau, rms_norm
from .sconv import (
    ATTN_STREAM,
    MLP_STREAM,
    Inkling2HPUConvState,
    Inkling2HPUShortConv,
)
from .tpc_ops import (
    ensure_tpc_ops_loaded,
    has_tpc_moe_post,
    has_tpc_qkvr_prep,
    has_tpc_sconv,
)
from .vision import Inkling2HPUVisionEncoder


class _LayerList(nn.ModuleDict):
    """ModuleDict with integer indexing so parameter names stay `*.N.*`
    without triggering `nn.ModuleList` regional compilation on containers."""

    def __init__(self, modules: Sequence[nn.Module]) -> None:
        super().__init__({str(i): m for i, m in enumerate(modules)})
        self._items: tuple[nn.Module, ...] = tuple(modules)

    def __getitem__(self, idx: int | str | slice):  # type: ignore[override]
        if isinstance(idx, (int, slice)):
            return self._items[idx]
        return super().__getitem__(idx)

    def __iter__(self):  # type: ignore[override]
        return iter(self._items)

    def __len__(self) -> int:
        return len(self._items)


class Inkling2HPURMSNorm(nn.Module):
    """RMSNorm; `num_inputs > 1` stacks per-input norm weights."""

    def __init__(
        self,
        hidden_size: int,
        eps: float,
        out_dtype: torch.dtype | None = torch.bfloat16,
        num_inputs: int = 1,
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(
            torch.ones(num_inputs, hidden_size, dtype=torch.bfloat16),
            requires_grad=False,
        )
        self.weight.weight_loader = self._load  # type: ignore[attr-defined]
        self.eps = float(eps)
        self.out_dtype = out_dtype

    @staticmethod
    def _load(param: nn.Parameter, loaded: torch.Tensor, shard_id: int = 0) -> None:
        param.data[shard_id].copy_(loaded)

    def forward(self, x: torch.Tensor, index: int = 0) -> torch.Tensor:
        return rms_norm(x, self.weight[index], self.eps, self.out_dtype)


class Inkling2HPUAttnRes(nn.Module):
    """Pseudo-queries of the block attention residuals (fp32), pre-scaled by D^-0.5."""

    def __init__(self, config: Inkling2ModelConfig) -> None:
        super().__init__()
        hidden = config.hidden_size
        local_ids = set(config.local_layer_ids)
        self.attn_queries = nn.ParameterList(
            nn.Parameter(
                torch.zeros(
                    (1, hidden) if i in local_ids else (3, hidden),
                    dtype=torch.float32,
                ),
                requires_grad=False,
            )
            for i in range(1, config.num_hidden_layers)
        )
        self.mlp_queries = nn.ParameterList(
            nn.Parameter(torch.zeros(1, hidden, dtype=torch.float32), False)
            for _ in range(config.num_hidden_layers)
        )
        self.output_query = nn.Parameter(
            torch.zeros(1, hidden, dtype=torch.float32), requires_grad=False
        )
        for param in self.parameters():
            param.weight_loader = self._load_query  # type: ignore[attr-defined]
        self.scale = hidden**-0.5

    def _load_query(self, param: nn.Parameter, weight: torch.Tensor) -> None:
        param.data.copy_(weight.view(param.shape)).mul_(self.scale)


class Inkling2HPUDecoderLayer(nn.Module):
    def __init__(
        self,
        config: Inkling2ModelConfig,
        layer_id: int,
        quant_config: Inkling2FP8QuantConfig | None,
        prefix: str,
        num_passes: int = 1,
    ) -> None:
        super().__init__()
        self.is_local = layer_id in config.local_layer_ids
        tp_size = get_tensor_model_parallel_world_size()
        num_kv_heads = (
            config.swa_num_key_value_heads
            if self.is_local
            else config.num_key_value_heads
        )
        head_dim = config.swa_head_dim if self.is_local else config.head_dim
        kv_size = (num_kv_heads // tp_size) * head_dim

        self.conv_states = _LayerList(
            [
                Inkling2HPUConvState(
                    kv_size=kv_size,
                    hidden_size=config.hidden_size,
                    kernel_size=config.sconv_kernel_size,
                    prefix=f"{prefix}.conv_state" + (f".pass{p}" if p else ""),
                )
                for p in range(num_passes)
            ]
        )
        self.conv_state: Inkling2HPUConvState = self.conv_states[0]  # type: ignore[assignment]
        self.attn = Inkling2HPUAttention(
            config,
            self.is_local,
            self.conv_state,
            prefix=f"{prefix}.attn",
            pass_conv_states=tuple(self.conv_states[1:]),  # type: ignore[arg-type]
        )
        eps = config.rms_norm_eps
        self.num_attn_inputs = 1 if self.is_local else 3
        self.attn_norm = Inkling2HPURMSNorm(
            config.hidden_size, eps, num_inputs=self.num_attn_inputs
        )
        self.mlp_norm = Inkling2HPURMSNorm(config.hidden_size, eps)
        fp8_max = quant_config.fp8_max if quant_config is not None else GAUDI2_FP8_MAX
        self.mlp = Inkling2HPUMoE(config, fp8_max=fp8_max)
        self.attn_sconv, self.mlp_sconv = (
            Inkling2HPUShortConv(
                config.hidden_size, config.sconv_kernel_size, self.conv_state, stream
            )
            for stream in (ATTN_STREAM, MLP_STREAM)
        )
        self.normalize_values = config.attn_res_normalize_values

    def attention(
        self,
        positions: torch.Tensor,
        inputs: Sequence[torch.Tensor],
        tau: torch.Tensor | None,
        pass_idx: int = 0,
        batch_size: int = 1,
        decode_ctx: DecodeAttnContext | None = None,
        kv_conv_state: tuple[
            torch.Tensor | None,
            torch.Tensor | None,
            bool,
            torch.Tensor | None,
            torch.Tensor | None,
        ]
        | None = None,
        direct_conv_cache: tuple[torch.Tensor, torch.Tensor] | None = None,
        x_f8: torch.Tensor | None = None,
        x_sc: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.is_local and len(inputs) == 1:
            return self.attn(
                positions,
                inputs[0],
                pass_idx=pass_idx,
                batch_size=batch_size,
                decode_ctx=decode_ctx,
                kv_conv_state=kv_conv_state,
                direct_conv_cache=direct_conv_cache,
                x_f8=None if x_f8 is None else x_f8[0],
                x_sc=None if x_sc is None else x_sc[0],
            )
        return self.attn(
            positions,
            inputs[0],
            inputs[1],
            inputs[2],
            None if self.is_local else tau,
            pass_idx=pass_idx,
            batch_size=batch_size,
            decode_ctx=decode_ctx,
            kv_conv_state=kv_conv_state,
            direct_conv_cache=direct_conv_cache,
            x_f8=x_f8,
            x_sc=x_sc,
        )

    def forward(
        self,
        positions: torch.Tensor,
        completed_blocks: torch.Tensor,
        partial: torch.Tensor,
        inputs: torch.Tensor | None,
        mlp_query: torch.Tensor,
        next_query: torch.Tensor,
        next_norm_weight: torch.Tensor | None,
        next_norm_eps: float,
        tau: torch.Tensor | None,
        *,
        first: bool,
        pass_idx: int = 0,
        batch_size: int = 1,
        decode_ctx: DecodeAttnContext | None = None,
        bank: torch.Tensor | None = None,
        message_query: torch.Tensor | None = None,
        inputs_f8: torch.Tensor | None = None,
        inputs_sc: torch.Tensor | None = None,
        return_next_f8: bool = False,
    ) -> (
        tuple[torch.Tensor, torch.Tensor]
        | tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]
    ):
        """Run one decoder layer visit (compiled as a single fused HPU cluster).

        On Gaudi2 TPC decode, reads all 4 short-conv streams directly from `state_buf`
        inside the fused TPC kernels (zero `gather_fwd_bf16` / `split` ops) and writes
        them back with one contiguous `index_copy_` at layer exit.
        """
        from vllm.forward_context import get_forward_context

        owner = self.conv_states[pass_idx]
        tp_size = get_tensor_model_parallel_world_size()
        attn_metadata = get_forward_context().attn_metadata

        can_direct_conv = (
            partial.device.type == "hpu"
            and partial.dtype == torch.bfloat16
            and tp_size == 1
            and self.attn.head_dim == 128
            and self.attn.d_rel <= 64
            and self.attn.k_sconv.kernel_size == 4
            and not self.attn_sconv.dynamic
            and self.attn_sconv.kernel_size == 4
            and not self.mlp_sconv.dynamic
            and self.mlp_sconv.kernel_size == 4
            and (owner.hidden_size % 64) == 0
            and has_tpc_qkvr_prep()
            and has_tpc_sconv()
        )
        direct_refs = (
            owner.get_decode_cache_refs(attn_metadata, decode_ctx=decode_ctx)
            if can_direct_conv
            else None
        )

        if direct_refs is not None:
            conv_state_buf, load_slots_i32, safe_store = direct_refs
            prev_k = prev_v = prev_attn = prev_mlp = None
            conv_is_prompt = False
            pad_mask = seq_lens = None
        else:
            conv_state_buf = load_slots_i32 = None
            prev_streams, safe_store, conv_is_prompt, pad_mask, seq_lens = owner.load_streams(
                attn_metadata, batch_size=batch_size, decode_ctx=decode_ctx
            )
            prev_k = None if prev_streams is None else prev_streams[0]
            prev_v = None if prev_streams is None else prev_streams[1]
            prev_attn = None if prev_streams is None else prev_streams[2]
            prev_mlp = None if prev_streams is None else prev_streams[3]

        messages: torch.Tensor | None = None
        attn_inputs: Sequence[torch.Tensor]
        attn_f8: torch.Tensor | None = None
        attn_sc: torch.Tensor | None = None
        if bank is not None:
            assert inputs is not None and message_query is not None
            messages = attn_res(
                bank[:-1],
                bank[-1],
                None,
                message_query,
                None,
                0.0,
                True,
            )
            dests, norm_weight = inputs, self.attn_norm.weight
            if self.is_local:
                dests = dests.expand(3, -1, -1)
                norm_weight = norm_weight.expand(3, -1)
            attn_inputs, attn_f8, attn_sc = self.blend_f8(  # type: ignore[attr-defined]
                slice(0, 3),
                dests,
                messages[:3],
                norm_weight,
                self.attn_norm.eps,
                fp8_max=self.attn.fp8_max,
            )
        elif inputs is None:
            attn_inputs = [
                self.attn_norm(partial, j) for j in range(self.num_attn_inputs)
            ]
        else:
            attn_inputs = inputs  # type: ignore[assignment]
            attn_f8 = inputs_f8
            attn_sc = inputs_sc

        new_kv_state: torch.Tensor | None = None
        new_k_state: torch.Tensor | None = None
        new_v_state: torch.Tensor | None = None
        if direct_refs is not None:
            assert conv_state_buf is not None and load_slots_i32 is not None
            attn_res_tuple = self.attention(
                positions,
                attn_inputs,
                tau,
                pass_idx=pass_idx,
                batch_size=batch_size,
                decode_ctx=decode_ctx,
                direct_conv_cache=(conv_state_buf, load_slots_i32),
                x_f8=attn_f8,
                x_sc=attn_sc,
            )
            assert isinstance(attn_res_tuple, tuple) and len(attn_res_tuple) == 2
            attn_partial, new_kv_state = attn_res_tuple
            attn_out, new_attn_state = self.attn_sconv.apply_with_state(
                attn_partial,
                None,
                is_prompt=False,
                batch_size=batch_size,
                conv_state=conv_state_buf,
                load_slots=load_slots_i32,
                stream_offset=2 * owner.kv_size,
            )
        else:
            attn_res_tuple = self.attention(
                positions,
                attn_inputs,
                tau,
                pass_idx=pass_idx,
                batch_size=batch_size,
                decode_ctx=decode_ctx,
                kv_conv_state=(prev_k, prev_v, conv_is_prompt, pad_mask, seq_lens),
                x_f8=attn_f8,
                x_sc=attn_sc,
            )
            assert isinstance(attn_res_tuple, tuple) and len(attn_res_tuple) == 3
            attn_partial, new_k_state, new_v_state = attn_res_tuple
            if tp_size > 1:
                attn_partial = tensor_model_parallel_all_reduce(attn_partial)
            attn_out, new_attn_state = self.attn_sconv.apply_with_state(
                attn_partial,
                prev_attn,
                is_prompt=conv_is_prompt,
                batch_size=batch_size,
                pad_mask=pad_mask,
                seq_lens=seq_lens,
            )

        if first:
            all_blocks = (
                torch.cat([completed_blocks, partial.unsqueeze(0)], dim=0)
                if completed_blocks.shape[0] > 0
                else partial.unsqueeze(0)
            )
            mlp_last = attn_out
            mlp_addend = None
        else:
            all_blocks = completed_blocks
            mlp_last = partial
            mlp_addend = attn_out

        mlp_f8: torch.Tensor | None = None
        mlp_sc: torch.Tensor | None = None
        if messages is not None:
            mlp_dest = attn_res(
                all_blocks,
                mlp_last,
                mlp_addend,
                mlp_query,
                None,
                0.0,
                self.normalize_values,
            )
            mlp_in, mlp_f8, mlp_sc = self.blend_f8(  # type: ignore[attr-defined]
                slice(3, 4),
                mlp_dest,
                messages[3:],
                self.mlp_norm.weight,
                self.mlp_norm.eps,
                fp8_max=self.mlp.experts.fp8_max,
            )
        else:
            mlp_in, mlp_f8, mlp_sc = attn_res_f8(
                all_blocks,
                mlp_last,
                mlp_addend,
                mlp_query,
                self.mlp_norm.weight,
                self.mlp_norm.eps,
                self.normalize_values,
                fp8_max=self.mlp.experts.fp8_max,
            )

        mlp_x_in = mlp_in[0]
        mlp_x_f8 = None if mlp_f8 is None else mlp_f8[0]
        mlp_x_sc = None if mlp_sc is None else mlp_sc[0]
        use_tpc_moe_post = (
            not conv_is_prompt
            and (direct_refs is not None or prev_mlp is not None)
            and tp_size == 1
            and mlp_x_in.device.type == "hpu"
            and mlp_x_in.dtype == torch.bfloat16
            and (mlp_x_in.shape[-1] % 64) == 0
            and mlp_x_in.shape[0] <= _MOE_TPC_MAX_TOKENS
            and getattr(self.mlp.experts, "_use_tpc_packed", False)
            and not self.mlp_sconv.dynamic
            and self.mlp_sconv.kernel_size == 4
            and has_tpc_sconv()
            and has_tpc_moe_post()
        )
        if use_tpc_moe_post:
            moe_partial = self.mlp(
                mlp_x_in, return_partial=True, x_q=mlp_x_f8, x_sc=mlp_x_sc
            )
            w_t = (
                self.mlp_sconv._weight_t
                if getattr(self.mlp_sconv, "_weight_t_ready", False)
                else self.mlp_sconv.conv_weight.t().contiguous()
            )
            if direct_refs is not None:
                assert conv_state_buf is not None and load_slots_i32 is not None
                mlp_cs, mlp_ls, mlp_off = (
                    conv_state_buf,
                    load_slots_i32,
                    2 * owner.kv_size + owner.hidden_size,
                )
            else:
                assert prev_mlp is not None
                mlp_cs = prev_mlp.to(torch.bfloat16).contiguous()
                mlp_ls = torch.arange(mlp_x_in.shape[0], device=mlp_x_in.device, dtype=torch.int32)
                mlp_off = 0
            new_partial, new_mlp_state = torch.ops.custom_op.inkling2_moe_sum_sconv_res_decode(
                moe_partial.contiguous(),
                mlp_cs,
                mlp_ls,
                w_t,
                partial.contiguous(),
                attn_out.contiguous(),
                float((self.mlp_sconv.kernel_size + 1) ** -0.5),
                1 if first else 0,
                mlp_off,
            )
            if direct_refs is not None:
                assert new_kv_state is not None
                owner.store_streams(safe_store, (new_kv_state, new_attn_state, new_mlp_state))
            else:
                assert new_k_state is not None and new_v_state is not None
                owner.store_streams(
                    safe_store, (new_k_state, new_v_state, new_attn_state, new_mlp_state)
                )
        else:
            if direct_refs is not None:
                assert conv_state_buf is not None and load_slots_i32 is not None and new_kv_state is not None
                mlp_out, new_mlp_state = self.mlp_sconv.apply_with_state(
                    self.mlp(mlp_x_in, x_q=mlp_x_f8, x_sc=mlp_x_sc),
                    None,
                    is_prompt=False,
                    batch_size=batch_size,
                    conv_state=conv_state_buf,
                    load_slots=load_slots_i32,
                    stream_offset=2 * owner.kv_size + owner.hidden_size,
                )
                owner.store_streams(safe_store, (new_kv_state, new_attn_state, new_mlp_state))
            else:
                assert new_k_state is not None and new_v_state is not None
                mlp_out, new_mlp_state = self.mlp_sconv.apply_with_state(
                    self.mlp(mlp_x_in, x_q=mlp_x_f8, x_sc=mlp_x_sc),
                    prev_mlp,
                    is_prompt=conv_is_prompt,
                    batch_size=batch_size,
                    pad_mask=pad_mask,
                    seq_lens=seq_lens,
                )
                owner.store_streams(
                    safe_store, (new_k_state, new_v_state, new_attn_state, new_mlp_state)
                )
            if first:
                new_partial = (attn_out.float() + mlp_out.float()).to(partial.dtype)
            else:
                new_partial = (
                    partial.float() + attn_out.float() + mlp_out.float()
                ).to(partial.dtype)

        if return_next_f8:
            if next_norm_weight is not None:
                next_inputs, next_f8, next_sc = attn_res_f8(
                    all_blocks,
                    new_partial,
                    None,
                    next_query,
                    next_norm_weight,
                    next_norm_eps,
                    self.normalize_values,
                    fp8_max=self.attn.fp8_max,
                )
                return new_partial, next_inputs, next_f8, next_sc
            next_inputs = attn_res(
                all_blocks,
                new_partial,
                None,
                next_query,
                None,
                next_norm_eps,
                self.normalize_values,
            )
            return new_partial, next_inputs, None, None

        next_inputs = attn_res(
            all_blocks,
            new_partial,
            None,
            next_query,
            next_norm_weight,
            next_norm_eps,
            self.normalize_values,
        )
        return new_partial, next_inputs


class Inkling2HPUModel(nn.Module):
    def __init__(
        self,
        config: Inkling2ModelConfig,
        quant_config: Inkling2FP8QuantConfig | None,
        prefix: str,
    ) -> None:
        super().__init__()
        self.config = config
        self.block_size = config.n_layers_per_attn_res_block
        self.normalize_values = config.attn_res_normalize_values
        self.embed_tokens = VocabParallelEmbedding(
            config.padded_vocab_size, config.hidden_size
        )
        self.embed_norm = Inkling2HPURMSNorm(
            config.hidden_size, config.rms_norm_eps, out_dtype=None
        )
        self.layers = nn.ModuleList(
            [
                self._make_layer(config, i, quant_config, f"{prefix}.layers.{i}")
                for i in range(config.num_hidden_layers)
            ]
        )
        self.attn_res = Inkling2HPUAttnRes(config)
        self.norm = Inkling2HPURMSNorm(config.hidden_size, config.rms_norm_eps)

    def _make_layer(
        self,
        config: Inkling2ModelConfig,
        layer_id: int,
        quant_config: Inkling2FP8QuantConfig | None,
        prefix: str,
    ) -> Inkling2HPUDecoderLayer:
        return Inkling2HPUDecoderLayer(config, layer_id, quant_config, prefix)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        # Use compiled embed_norm for 2D bucketed prompt/decode inputs; keep 1D
        # variable-length multimodal inputs eager to avoid per-length recompiles.
        if input_ids.ndim > 1:
            norm = getattr(self, "_hpu_compiled_embed_norm", self.embed_norm)
            return norm(self.embed_tokens(input_ids))
        return self.embed_norm(self.embed_tokens(input_ids))

    def _log_scaling_tau(self, positions: torch.Tensor) -> torch.Tensor | None:
        return compute_log_scaling_tau(
            positions, self.config.log_scaling_n_floor, self.config.log_scaling_alpha
        )

    def forward(
        self,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor,
        batch_size: int = 1,
    ) -> torch.Tensor:
        from vllm.forward_context import get_forward_context

        num_tokens, hidden = inputs_embeds.shape
        num_layers = len(self.layers)
        blocks: list[torch.Tensor] = []
        partial = inputs_embeds
        tau = self._log_scaling_tau(positions)
        attn_md = get_forward_context().attn_metadata
        if isinstance(attn_md, dict):
            attn_md = next(iter(attn_md.values()), None)
        decode_bs = num_tokens if (attn_md is not None and not getattr(attn_md, "is_prompt", True)) else batch_size
        decode_ctx = build_decode_attn_context(
            attn_md,
            positions,
            decode_bs,
            sliding_window=self.config.sliding_window_size,
            rel_extent=self.config.rel_extent,
        )

        empty_blocks = inputs_embeds.new_empty((0, num_tokens, hidden))
        inputs: torch.Tensor | None = None
        inputs_f8: torch.Tensor | None = None
        inputs_sc: torch.Tensor | None = None
        for i, layer in enumerate(self.layers):
            first = i % self.block_size == 0
            completed = torch.stack(blocks, dim=0) if blocks else empty_blocks
            need_f8 = i + 1 < num_layers
            if not need_f8:
                next_query = self.attn_res.output_query
                next_norm = self.norm
            else:
                next_query = self.attn_res.attn_queries[i]
                next_norm = self.layers[i + 1].attn_norm
            res = layer(
                positions,
                completed,
                partial,
                inputs,
                self.attn_res.mlp_queries[i],
                next_query,
                next_norm.weight,
                next_norm.eps,
                tau,
                first=first,
                pass_idx=0,
                batch_size=batch_size,
                decode_ctx=decode_ctx,
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
            partial = new_partial

        assert inputs is not None
        return inputs[0]


@MULTIMODAL_REGISTRY.register_processor(
    Inkling2MultiModalProcessor,
    info=Inkling2ProcessingInfo,
    dummy_inputs=Inkling2DummyInputsBuilder,
)
class Inkling2HPUForConditionalGeneration(nn.Module, SupportsMultiModal):
    _renames = (
        ("model.llm.unembed.", "lm_head."),
        ("model.llm.embed.", "model.embed_tokens."),
        ("model.llm.", "model."),
        (".q_l2_clip.scale", ".q_scale"),
        (".rel_l2_clip.scale", ".r_scale"),
        (".rel_logits_proj.proj", ".rel_logits_proj"),
    )
    _shards = {
        ".attn.wq_du.": (".attn.qkvr.", 0),
        ".attn.wr_du.": (".attn.qkvr.", 1),
        ".attn.gate_attn.": (".attn.qkvr.", 2),
        ".attn.attn_sink_proj.": (".attn.qkvr.", 3),
        ".attn.wk_dv.": (".attn.qkvr.", 4),
        ".attn.wv_dv.": (".attn.qkvr.", 5),
        ".q_input_norm.": (".attn_norm.", 0),
        ".k_input_norm.": (".attn_norm.", 1),
        ".v_input_norm.": (".attn_norm.", 2),
    }

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config: Inkling2MMConfig = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        assert quant_config is None or isinstance(quant_config, Inkling2FP8QuantConfig)
        assert vllm_config.parallel_config.pipeline_parallel_size == 1
        self.config = config
        self.text_config = text_config = config.text_config
        self.visual = (
            Inkling2HPUVisionEncoder(config.vision_config, text_config.hidden_size)
            if config.vision_config is not None
            and vllm_config.model_config.multimodal_config is not None
            and vllm_config.model_config.multimodal_config.get_limit_per_prompt("image")
            else None
        )
        model_cls: type[Inkling2HPUModel] = Inkling2HPUModel
        if text_config.loop_config is not None:
            from .loop import Inkling2HPULoopModel

            model_cls = Inkling2HPULoopModel
        self.model = model_cls(text_config, quant_config, f"{prefix}model")
        self.lm_head = ParallelLMHead(
            text_config.padded_vocab_size, text_config.hidden_size
        )
        self.logits_processor = LogitsProcessor(
            text_config.padded_vocab_size,
            org_vocab_size=text_config.vocab_size,
            soft_cap=text_config.final_logit_softcapping,
        )

    def get_language_model(self) -> nn.Module:
        return self

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        if modality.startswith("image"):
            return IMAGE_PLACEHOLDER
        raise ValueError("Only image modality is supported")

    def embed_multimodal(self, **kwargs: object) -> MultiModalEmbeddings:
        patches = kwargs.get("vision_pixels")
        grid_thw = kwargs.get("vision_grid_thw")
        if patches is None or grid_thw is None:
            return ()
        assert isinstance(patches, torch.Tensor)
        assert isinstance(grid_thw, torch.Tensor)
        assert self.visual is not None
        embeds = self.visual(patches, grid_thw)
        return embeds.split(grid_thw.prod(-1).tolist())

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: MultiModalEmbeddings | None = None,
        *,
        is_multimodal: torch.Tensor | None = None,
    ) -> torch.Tensor:
        inputs_embeds = self.model.embed_input_ids(input_ids)
        if not multimodal_embeddings:
            return inputs_embeds
        assert is_multimodal is not None
        return _merge_multimodal_embeddings(
            inputs_embeds=inputs_embeds,
            multimodal_embeddings=multimodal_embeddings,
            is_multimodal=is_multimodal,
        )

    def process_weights_after_loading(self) -> None:
        ensure_tpc_ops_loaded()
        if hasattr(self.model, "process_weights_after_loading"):
            self.model.process_weights_after_loading()
        for layer in self.model.layers:
            if hasattr(layer, "process_weights_after_loading"):
                layer.process_weights_after_loading()
            if hasattr(layer, "attn") and hasattr(layer.attn, "process_weights_after_loading"):
                layer.attn.process_weights_after_loading()
            for sconv_name in ("attn_sconv", "mlp_sconv"):
                sconv = getattr(layer, sconv_name, None)
                if sconv is not None and hasattr(sconv, "process_weights_after_loading"):
                    sconv.process_weights_after_loading()
            if hasattr(layer, "mlp"):
                if hasattr(layer.mlp, "gate") and hasattr(
                    layer.mlp.gate, "process_weights_after_loading"
                ):
                    layer.mlp.gate.process_weights_after_loading()
                if hasattr(layer.mlp, "experts") and hasattr(
                    layer.mlp.experts, "process_weights_after_loading"
                ):
                    layer.mlp.experts.process_weights_after_loading()

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor:
        del intermediate_tensors, kwargs
        orig_shape = positions.shape
        if inputs_embeds is None:
            assert input_ids is not None
            inputs_embeds = self.model.embed_input_ids(input_ids)
        hidden_size = inputs_embeds.shape[-1]
        batch_size = orig_shape[0] if len(orig_shape) > 1 else 1
        out = self.model(
            positions.reshape(-1),
            inputs_embeds.reshape(-1, hidden_size),
            batch_size=batch_size,
        )
        if len(orig_shape) > 1:
            return out.view(*orig_shape, hidden_size)
        return out

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        return self.logits_processor(self.lm_head, hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        params = dict(self.named_parameters())
        num_layers = self.text_config.num_hidden_layers
        layer_limits = {
            "layers": num_layers,
            "attn_queries": num_layers - 1,
            "mlp_queries": num_layers,
        }
        loaded: set[str] = set()
        loaded_shards: set[tuple[str, int]] = set()
        visual_weights: list[tuple[str, torch.Tensor]] = []
        for name, weight in weights:
            if name.startswith("model.visual."):
                if self.visual is not None:
                    visual_weights.append((name.removeprefix("model.visual."), weight))
                continue
            for old, new in self._renames:
                name = name.replace(old, new)
            shard_id = None
            for old, (new, shard) in self._shards.items():
                if old in name:
                    name = name.replace(old, new)
                    shard_id = shard
                    break
            if name not in params:
                m = re.search(
                    r"\.(layers|attn_queries|mlp_queries)\.(\d+)(?:\.|$)", name
                )
                if m is not None and int(m.group(2)) >= layer_limits[m.group(1)]:
                    continue
                raise ValueError(f"Unexpected checkpoint tensor: {name}")
            param = params[name]
            if param.dtype != weight.dtype:
                raise ValueError(
                    f"{name}: checkpoint dtype {weight.dtype} != {param.dtype}. "
                    "Make sure the checkpoint was converted with tools/convert_inkling2_fp8.py."
                )
            loader: Callable[..., None] = getattr(
                param, "weight_loader", default_weight_loader
            )
            if shard_id is None:
                loader(param, weight)
                loaded.add(name)
            else:
                loader(param, weight, shard_id)
                loaded_shards.add((name, shard_id))
        num_shards = Counter(new for new, _ in self._shards.values())
        for name, count in Counter(name for name, _ in loaded_shards).items():
            if ".attn_norm." in name:
                expected = params[name].shape[0]
            else:
                fused = next(f for f in num_shards if f in name)
                expected = num_shards[fused]
            if count == expected:
                loaded.add(name)
        if self.visual is not None:
            loaded |= {f"visual.{n}" for n in self.visual.load_weights(visual_weights)}
        if missing := params.keys() - loaded:
            raise ValueError(f"Missing checkpoint tensors: {sorted(missing)}")
        self.process_weights_after_loading()
        return loaded
