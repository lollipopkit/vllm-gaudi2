# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""LFM3 (LiquidAI): LFM2's gated short convolutions plus full- and sliding-window
attention layers with per-type RoPE, QK RMSNorm and a sigmoid output gate per
head, and sandwich RMSNorms around every operator and feed-forward block."""

from collections.abc import Iterable
from itertools import islice

import torch
import torch.nn as nn

from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, ModelConfig, VllmConfig
from vllm.distributed import get_pp_group, get_tensor_model_parallel_world_size
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.mamba.mamba_utils import (
    MambaStateCopyFunc,
    MambaStateCopyFuncCalculator,
    MambaStateDtypeCalculator,
    MambaStateShapeCalculator,
)
from vllm.model_executor.layers.mamba.short_conv import ShortConv
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.sequence import IntermediateTensors

from vllm.model_executor.models.interfaces import HasInnerState, IsHybrid, SupportsLoRA, SupportsPP, SupportsQuant
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    PPMissingLayer,
    WeightsMapper,
    extract_layer_index,
    make_empty_intermediate_tensors_factory,
    make_layers,
    maybe_prefix,
)


class Lfm3MLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        self.w13 = MergedColumnParallelLinear(
            input_size=hidden_size,
            output_sizes=[intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.w13",
        )
        self.w2 = RowParallelLinear(
            input_size=intermediate_size,
            output_size=hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.w2",
        )
        self.act_fn = SiluAndMul()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up, _ = self.w13(x)
        x, _ = self.w2(self.act_fn(gate_up))
        return x


def zero_tail_rotary(rope: nn.Module, rope_angles: int) -> None:
    """LFM3's partial RoPE: the frequency progression of the full head dimension,
    with every frequency past `rope_angles` set to zero, so dims (i, i + head_dim / 2)
    stay paired as in the full rotation and the tail pairs are left unrotated.
    Recomputes the cache of a full-dim RotaryEmbedding in place."""
    inv_freq = rope._compute_inv_freq(rope.base)
    inv_freq[rope_angles:] = 0.0
    t = torch.arange(rope.max_position_embeddings, dtype=torch.float)
    freqs = torch.outer(t, inv_freq)
    cache = torch.cat((freqs.cos(), freqs.sin()), dim=-1)
    rope.cos_sin_cache = cache.to(rope.cos_sin_cache.device, rope.cos_sin_cache.dtype)
    if hasattr(rope, "cos_sin_cache_bf16"):
        rope.cos_sin_cache_bf16 = None


class Lfm3Attention(nn.Module):
    """Attention of a `full_attention` or `sliding_attention` layer."""

    def __init__(
        self,
        config,
        layer_idx: int,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = config.num_attention_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = config.num_key_value_heads
        if self.total_num_kv_heads >= tp_size:
            assert self.total_num_kv_heads % tp_size == 0
        else:
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = config.head_dim
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5

        self.qkv_proj = QKVParallelLinear(
            hidden_size=self.hidden_size,
            head_size=self.head_dim,
            total_num_heads=self.total_num_heads,
            total_num_kv_heads=self.total_num_kv_heads,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )
        self.o_proj = RowParallelLinear(
            input_size=self.total_num_heads * self.head_dim,
            output_size=self.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )
        # One sigmoid gate per head; sharded like the heads.
        self.gate_proj = ColumnParallelLinear(
            input_size=self.hidden_size,
            output_size=self.total_num_heads,
            bias=False,
            gather_output=False,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_proj",
        )
        self.q_norm = RMSNorm(self.head_dim, eps=config.norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=config.norm_eps)

        layer_type = config.layer_types[layer_idx]
        rope = config.rope_parameters[layer_type]
        factor = rope.get("partial_rotary_factor", 1.0)
        self.rotary_emb = get_rope(
            self.head_dim,
            max_position=config.max_position_embeddings,
            rope_parameters={"rope_type": rope["rope_type"], "rope_theta": rope["rope_theta"]},
            is_neox_style=True,
        )
        if factor < 1.0:  # get_rope shares one instance per parameter set; idempotent
            zero_tail_rotary(self.rotary_emb, int(factor * self.head_dim) // 2)
        sliding_window = (
            config.sliding_window if layer_type == "sliding_attention" else None
        )
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            per_layer_sliding_window=sliding_window,
            prefix=f"{prefix}.attn",
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        # Shape-agnostic: the HPU runner feeds [batch, seq, hidden], others [tokens, hidden].
        lead = hidden_states.shape[:-1]
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q = self.q_norm(q.view(*lead, self.num_heads, self.head_dim))
        k = self.k_norm(k.view(*lead, self.num_kv_heads, self.head_dim))
        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q.view(*lead, -1), k.view(*lead, -1), v)
        gate, _ = self.gate_proj(hidden_states)
        attn_output = attn_output.view(*lead, self.num_heads, self.head_dim) * (
            torch.sigmoid(gate).unsqueeze(-1)
        )
        output, _ = self.o_proj(attn_output.view(*lead, -1))
        return output


class Lfm3DecoderLayer(nn.Module):
    def __init__(
        self,
        config,
        layer_idx: int,
        model_config: ModelConfig | None = None,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.is_attention_layer = config.layer_types[layer_idx] != "conv"
        if self.is_attention_layer:
            self.self_attn = Lfm3Attention(
                config,
                layer_idx,
                cache_config=cache_config,
                quant_config=quant_config,
                prefix=f"{prefix}.self_attn",
            )
        else:
            self.short_conv = ShortConv(
                config=config,
                dim=config.hidden_size,
                layer_idx=layer_idx,
                model_config=model_config,
                cache_config=cache_config,
                quant_config=quant_config,
                prefix=f"{prefix}.short_conv",
            )
        self.feed_forward = Lfm3MLP(
            config.hidden_size,
            config.intermediate_size,
            quant_config=quant_config,
            prefix=f"{prefix}.feed_forward",
        )
        self.operator_norm = RMSNorm(config.hidden_size, eps=config.norm_eps)
        self.ffn_norm = RMSNorm(config.hidden_size, eps=config.norm_eps)
        self.use_sandwich_norm = config.use_sandwich_norm
        if self.use_sandwich_norm:
            self.post_operator_norm = RMSNorm(config.hidden_size, eps=config.norm_eps)
            self.post_ffn_norm = RMSNorm(config.hidden_size, eps=config.norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            residual = hidden_states
            hidden_states = self.operator_norm(hidden_states)
        else:
            hidden_states, residual = self.operator_norm(hidden_states, residual)
        if self.is_attention_layer:
            hidden_states = self.self_attn(positions, hidden_states)
        elif getattr(self.short_conv, "_returns_output", False):
            hidden_states = self.short_conv(hidden_states)
        else:
            output = torch.empty_like(hidden_states)
            self.short_conv(hidden_states, output)
            hidden_states = output
        if self.use_sandwich_norm:
            hidden_states = self.post_operator_norm(hidden_states)
        hidden_states, residual = self.ffn_norm(hidden_states, residual)
        hidden_states = self.feed_forward(hidden_states)
        if self.use_sandwich_norm:
            hidden_states = self.post_ffn_norm(hidden_states)
        return hidden_states, residual


@support_torch_compile
class Lfm3Model(nn.Module):
    # HF names the short-conv container `.conv` and its depthwise conv `.conv.conv`;
    # vLLM's ShortConv holds the latter as `.conv` too, so the container becomes
    # `.short_conv` (str.replace consumes the first `.conv.` and leaves the inner one).
    # Stacked keys end with a dot so that applying the mapper twice (an outer multimodal
    # mapper, then this one) is a no-op: `.w1` would otherwise match inside `.w13`.
    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_substr={".conv.": ".short_conv."},
        orig_to_new_stacked={
            ".q_proj.": (".qkv_proj.", "q"),
            ".k_proj.": (".qkv_proj.", "k"),
            ".v_proj.": (".qkv_proj.", "v"),
            ".w1.": (".w13.", 0),
            ".w3.": (".w13.", 1),
        },
    )

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        model_config = vllm_config.model_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config
        self.config = config
        self.vocab_size = config.vocab_size

        self.embed_tokens = VocabParallelEmbedding(
            self.vocab_size, config.hidden_size, org_num_embeddings=config.vocab_size
        )

        def get_layer(prefix: str):
            return Lfm3DecoderLayer(
                config,
                extract_layer_index(prefix),
                model_config,
                cache_config,
                quant_config=quant_config,
                prefix=prefix,
            )

        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers, get_layer, prefix=f"{prefix}.layers"
        )
        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states", "residual"], config.hidden_size
        )
        if get_pp_group().is_last_rank:
            self.embedding_norm = RMSNorm(config.hidden_size, eps=config.norm_eps)
        else:
            self.embedding_norm = PPMissingLayer()

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.embed_input_ids(input_ids)
            residual = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]

        for layer in islice(self.layers, self.start_layer, self.end_layer):
            hidden_states, residual = layer(positions, hidden_states, residual)
        if not get_pp_group().is_last_rank:
            return IntermediateTensors(
                {"hidden_states": hidden_states, "residual": residual}
            )
        hidden_states, _ = self.embedding_norm(hidden_states, residual)
        return hidden_states

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)


class Lfm3ForCausalLM(
    nn.Module, HasInnerState, SupportsLoRA, SupportsPP, IsHybrid, SupportsQuant
):
    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "w13": ["w1", "w3"],
        "in_proj": ["in_proj"],
    }
    hf_to_vllm_mapper = Lfm3Model.hf_to_vllm_mapper
    embedding_modules = {
        "embed_tokens": "input_embeddings",
        "lm_head": "output_embeddings",
    }

    @classmethod
    def get_mamba_state_dtype_from_config(
        cls, vllm_config: "VllmConfig"
    ) -> tuple[torch.dtype, ...]:
        return MambaStateDtypeCalculator.short_conv_state_dtype(
            vllm_config.model_config.dtype,
            vllm_config.cache_config.mamba_cache_dtype,
        )

    @classmethod
    def get_mamba_state_shape_from_config(
        cls, vllm_config: "VllmConfig"
    ) -> tuple[tuple[int, int]]:
        hf_config = vllm_config.model_config.hf_config
        return MambaStateShapeCalculator.short_conv_state_shape(
            tp_world_size=vllm_config.parallel_config.tensor_parallel_size,
            intermediate_size=hf_config.hidden_size,
            conv_kernel=hf_config.conv_L_cache,
        )

    @classmethod
    def get_mamba_state_copy_func(cls) -> tuple[MambaStateCopyFunc]:
        return MambaStateCopyFuncCalculator.short_conv_state_copy_func()

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        super().__init__()
        self.config = config
        self.model = Lfm3Model(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )
        if get_pp_group().is_last_rank:
            self.lm_head = ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                prefix=maybe_prefix(prefix, "lm_head"),
            )
            if config.tie_word_embeddings:
                self.lm_head = self.lm_head.tie_weights(self.model.embed_tokens)
        else:
            self.lm_head = PPMissingLayer()
        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        return self.model(input_ids, positions, intermediate_tensors, inputs_embeds)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.logits_processor(self.lm_head, hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        # A tied checkpoint may still carry lm_head.weight; the module shares the embedding.
        skip = ["lm_head."] if self.config.tie_word_embeddings else None
        loader = AutoWeightsLoader(self, ignore_unexpected_prefixes=skip)
        return loader.load_weights(weights)
