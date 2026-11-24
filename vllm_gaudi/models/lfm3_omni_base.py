# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""LFM3-Omni (LiquidAI), text output: the LFM3-VL model plus a speech encoder. 16 kHz audio
becomes 128-bin log-mel frames; three stride-2 convolutions turn 8 frames into one vector,
a 17-layer LFM2 stack (gated causal convs and causal sliding-window attention) encodes them
and a GELU projector maps each to one text token. The speech output head (depth decoder,
codebook embeddings, codec) is not loaded: generation is text only."""

import itertools
from collections.abc import Mapping, Sequence
from types import SimpleNamespace
from typing import Annotated, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BatchFeature

from vllm.config.multimodal import MultiModalDummyOptions
from vllm.inputs import MultiModalDataDict
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding.common import ApplyRotaryEmb
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import MultiModalFieldConfig, MultiModalKwargsItems
from vllm.multimodal.parse import MultiModalDataItems
from vllm.multimodal.processing import PromptReplacement, PromptUpdate, PromptUpdateDetails
from vllm.multimodal.processing.processor import (
    MultiModalProcessingResult,
    PlaceholderFeaturesInfo,
)
from vllm.utils.tensor_schema import TensorSchema, TensorShape

from vllm.model_executor.models.qwen2_5_omni_thinker import merge_interleaved_embeddings
from vllm.model_executor.models.qwen3_vl import Qwen3VLMultiModalDataParser
from vllm.model_executor.models.utils import WeightsMapper, maybe_prefix

from .lfm3_vl_base import (
    Lfm3VlDummyInputsBuilder,
    Lfm3VlForConditionalGeneration,
    Lfm3VlMultiModalProcessor,
    Lfm3VlProcessingInfo,
    Lfm3VlVisionMLP,
)

# mel frames per audio token (three stride-2 convolutions)
_AUDIO_STRIDE = 8


def _num_audio_tokens(num_frames: int) -> int:
    return -(num_frames // -_AUDIO_STRIDE)


def _audio_token_bucket(num_tokens: int) -> int:
    """Padded encoder length: steps of 32 tokens (2.56 s) up to 512, of 128 above; 42 shapes up to 300 s."""
    step = 32 if num_tokens <= 512 else 128
    return step * -(-num_tokens // step)


# Longest accepted clip (3750 audio tokens); also bounded by max_model_len. The model card states no clip limit,
# only the 32K context (input + output).
_MAX_AUDIO_SECONDS = 300


class Lfm3OmniAudioInputs(TensorSchema):
    """
    Dimensions:
        - na: number of clips
        - nmb: number of mel bins
        - nf: number of mel frames (padded to the longest clip of a request; clips of
          different requests batched together keep their own lengths)
    """

    type: Literal["audio_features"] = "audio_features"
    input_features: Annotated[
        torch.Tensor | list[torch.Tensor], TensorShape("na", "nmb", "nf", dynamic_dims={"nf"})
    ]
    feature_attention_mask: Annotated[
        torch.Tensor | list[torch.Tensor], TensorShape("na", "nf", dynamic_dims={"nf"})
    ]


def _ffn_dim(cfg) -> int:
    """Lfm2MLP's feed-forward width: 2/3 of intermediate_size, scaled, rounded up."""
    dim = cfg.intermediate_size
    if cfg.block_auto_adjust_ff_dim:
        dim = int(2 * dim / 3)
        if cfg.block_ffn_dim_multiplier is not None:
            dim = int(cfg.block_ffn_dim_multiplier * dim)
        dim = cfg.block_multiple_of * -(-dim // cfg.block_multiple_of)
    return dim


class Lfm3OmniDownsampler(nn.Module):
    """Three stride-2 convolutions over the (mel, time) plane: 128 mel bins x 8 frames -> one vector."""

    def __init__(self, config):
        super().__init__()
        ch = config.conv_channels
        conv = {"kernel_size": 3, "stride": 2, "padding": 1}
        self.conv_in = nn.Conv2d(1, ch, **conv)
        self.conv_mid = nn.Conv2d(ch, ch, groups=ch, **conv)
        self.lin_mid = nn.Conv2d(ch, ch, kernel_size=1)
        self.conv_end = nn.Conv2d(ch, ch, groups=ch, **conv)
        self.lin_end = nn.Conv2d(ch, ch, kernel_size=1)
        self.lin_out = nn.Conv1d(config.num_mel_bins * ch // 8, config.hidden_size, kernel_size=1)

    def forward(self, features: torch.Tensor, num_frames: int) -> torch.Tensor:
        """[1, mel, frames] zero-padded past num_frames -> [frames / 8, hidden]. Stage outputs past the clip are
        zeroed as in the reference (the next stride-2 conv reads one position past them)."""

        def mask(x: torch.Tensor, length: int) -> torch.Tensor:
            # built on the host: the clip length must not become a graph constant
            valid = (torch.arange(x.shape[-1]) < length).to(device=x.device, dtype=x.dtype)
            return x * valid

        length = (num_frames + 1) // 2
        x = mask(F.silu(self.conv_in(features.unsqueeze(1))), length)
        length = (length + 1) // 2
        x = mask(F.silu(self.lin_mid(self.conv_mid(x))), length)
        x = F.silu(self.lin_end(self.conv_end(x)))
        return self.lin_out(x.flatten(1, 2)).transpose(1, 2)[0]  # [tokens, hidden]


class Lfm3OmniEncoderConv(nn.Module):
    """LFM2 gated short convolution, causal, over one clip."""

    def __init__(self, cfg, quant_config, prefix: str):
        super().__init__()
        hidden = cfg.hidden_size
        self.in_proj = ReplicatedLinear(
            hidden, 3 * hidden, bias=False, quant_config=quant_config, prefix=f"{prefix}.in_proj"
        )
        self.conv = nn.Conv1d(
            hidden, hidden, kernel_size=cfg.conv_L_cache, groups=hidden,
            padding=cfg.conv_L_cache - 1, bias=cfg.conv_bias,
        )
        self.out_proj = ReplicatedLinear(
            hidden, hidden, bias=False, quant_config=quant_config, prefix=f"{prefix}.out_proj"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bcx, _ = self.in_proj(x)
        b, c, x = bcx.t().chunk(3, dim=0)
        y = self.conv((b * x).unsqueeze(0).to(self.conv.weight.dtype))[0, :, : x.shape[-1]]
        out, _ = self.out_proj((c * y.to(c.dtype)).t())
        return out


class Lfm3OmniEncoderAttention(nn.Module):
    """LFM2 attention (QK RMSNorm, RoPE), causal within a sliding window, over one clip."""

    def __init__(self, cfg, window: int, quant_config, prefix: str):
        super().__init__()
        self.head_dim = cfg.hidden_size // cfg.num_attention_heads
        self.window = window
        self.qkv_proj = QKVParallelLinear(
            hidden_size=cfg.hidden_size,
            head_size=self.head_dim,
            total_num_heads=cfg.num_attention_heads,
            total_num_kv_heads=cfg.num_key_value_heads,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )
        self.num_heads = self.qkv_proj.num_heads
        self.num_kv_heads = self.qkv_proj.num_kv_heads
        self.out_proj = RowParallelLinear(
            cfg.num_attention_heads * self.head_dim,
            cfg.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.out_proj",
        )
        self.q_layernorm = RMSNorm(self.head_dim, eps=cfg.norm_eps)
        self.k_layernorm = RMSNorm(self.head_dim, eps=cfg.norm_eps)
        self.apply_rotary_emb = ApplyRotaryEmb(enforce_enable=True)

    def forward(self, x: torch.Tensor, cos, sin, mask: torch.Tensor) -> torch.Tensor:
        n = x.shape[0]
        qkv, _ = self.qkv_proj(x)
        q_size, kv_size = self.num_heads * self.head_dim, self.num_kv_heads * self.head_dim
        q, k, v = qkv.split([q_size, kv_size, kv_size], dim=-1)
        q = self.apply_rotary_emb(self.q_layernorm(q.view(n, self.num_heads, self.head_dim))[None], cos, sin)
        k = self.apply_rotary_emb(self.k_layernorm(k.view(n, self.num_kv_heads, self.head_dim))[None], cos, sin)
        v = v.view(1, n, self.num_kv_heads, self.head_dim)
        out = F.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
            attn_mask=mask, enable_gqa=self.num_heads != self.num_kv_heads,
        )
        out, _ = self.out_proj(out.transpose(1, 2).reshape(n, self.num_heads * self.head_dim))
        return out


class Lfm3OmniEncoderLayer(nn.Module):
    def __init__(self, cfg, window: int, layer_idx: int, quant_config, prefix: str):
        super().__init__()
        self.is_attention_layer = cfg.layer_types[layer_idx] == "full_attention"
        if self.is_attention_layer:
            self.self_attn = Lfm3OmniEncoderAttention(cfg, window, quant_config, f"{prefix}.self_attn")
        else:
            self.conv = Lfm3OmniEncoderConv(cfg, quant_config, f"{prefix}.conv")
        mlp_cfg = SimpleNamespace(hidden_size=cfg.hidden_size, intermediate_size=_ffn_dim(cfg))
        self.feed_forward = Lfm3VlVisionMLP(mlp_cfg, quant_config, f"{prefix}.feed_forward")
        self.operator_norm = RMSNorm(cfg.hidden_size, eps=cfg.norm_eps)
        self.ffn_norm = RMSNorm(cfg.hidden_size, eps=cfg.norm_eps)

    def forward(self, x, cos, sin, mask):
        normed = self.operator_norm(x)
        x = x + (self.self_attn(normed, cos, sin, mask) if self.is_attention_layer else self.conv(normed))
        return x + self.feed_forward(self.ffn_norm(x))


class Lfm3OmniAudioEncoder(nn.Module):
    def __init__(self, config, quant_config: QuantizationConfig | None = None, prefix: str = ""):
        super().__init__()
        cfg = config.text_config
        self.window = config.sliding_window
        self.downsample = Lfm3OmniDownsampler(config)
        self.lfm = nn.Module()
        self.lfm.layers = nn.ModuleList(
            [
                Lfm3OmniEncoderLayer(cfg, self.window, i, quant_config, f"{prefix}.lfm.layers.{i}")
                for i in range(cfg.num_hidden_layers)
            ]
        )
        self.lfm.embedding_norm = RMSNorm(cfg.hidden_size, eps=cfg.norm_eps)
        head_dim = cfg.hidden_size // cfg.num_attention_heads
        theta = cfg.rope_parameters["rope_theta"]
        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, features: torch.Tensor, num_frames: int) -> torch.Tensor:
        """[mel, >= num_frames] -> [bucket tokens, hidden], padded to _audio_token_bucket (causal layers: the
        padding never reaches a valid token); the first _num_audio_tokens(num_frames) rows are the clip."""
        frames = _AUDIO_STRIDE * _audio_token_bucket(_num_audio_tokens(num_frames))
        clip = features[:, :num_frames]
        if frames > num_frames:
            clip = F.pad(clip, (0, frames - num_frames))
        x = self.downsample(clip[None].to(self.downsample.conv_in.weight.dtype), num_frames)
        n = x.shape[0]
        pos = torch.arange(n, device=x.device, dtype=torch.float32)
        freqs = pos[:, None] * self.inv_freq
        cos, sin = freqs.cos(), freqs.sin()
        dist = pos[:, None] - pos[None, :]  # query - key
        mask = (dist >= 0) & (dist < self.window)
        for layer in self.lfm.layers:
            x = layer(x, cos, sin, mask)
        return self.lfm.embedding_norm(x)


class Lfm3OmniAudioProjector(nn.Module):
    def __init__(self, config, quant_config=None, prefix: str = ""):
        super().__init__()
        in_dim = config.audio_encoder_config.hidden_size
        out_dim = config.text_config.hidden_size
        self.layer_norm = nn.LayerNorm(in_dim)
        self.linear_1 = ReplicatedLinear(in_dim, out_dim, bias=True, quant_config=quant_config,
                                         prefix=f"{prefix}.linear_1")
        self.linear_2 = ReplicatedLinear(out_dim, out_dim, bias=True, quant_config=quant_config,
                                         prefix=f"{prefix}.linear_2")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, _ = self.linear_1(self.layer_norm(x))
        x, _ = self.linear_2(F.gelu(x))
        return x


class Lfm3OmniProcessingInfo(Lfm3VlProcessingInfo):
    def get_feature_extractor(self, **kwargs: object):
        fe = self.get_hf_processor(**kwargs).feature_extractor
        assert fe.encoder_stride == _AUDIO_STRIDE, fe.encoder_stride
        return fe

    def get_data_parser(self):
        return Qwen3VLMultiModalDataParser(
            self.get_hf_config().vision_config.spatial_merge_size,
            video_needs_metadata=True,
            target_sr=self.get_feature_extractor().sampling_rate,
            target_channels=1,
            expected_hidden_size=self._get_expected_hidden_size(),
            allow_missing_mm_embeddings=self.allow_missing_mm_embeddings,
        )

    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        # mm_processor_kwargs {"use_audio_in_video": true} pairs the i-th video with the i-th audio clip
        return {"image": None, "video": None, "audio": None}

    def get_max_audio_samples(self, seq_len: int) -> int:
        """Longest clip in samples: _MAX_AUDIO_SECONDS, shortened to fit seq_len audio tokens."""
        fe = self.get_feature_extractor()
        per_token = fe.encoder_stride * fe.hop_length
        samples = min(_MAX_AUDIO_SECONDS * fe.sampling_rate, seq_len * per_token)
        while self.get_num_audio_tokens(samples) > seq_len:  # the STFT window adds up to one token
            samples -= per_token
        return samples

    def get_num_audio_tokens(self, num_samples: int) -> int:
        fe = self.get_feature_extractor()
        return fe.num_encoder_tokens(fe.num_frames(num_samples))

    def get_mm_max_tokens_per_item(self, seq_len: int, mm_counts: Mapping[str, int]) -> Mapping[str, int]:
        max_audio = self.get_num_audio_tokens(self.get_max_audio_samples(seq_len))
        return {
            "image": self.get_max_image_tokens(),
            "video": self.get_max_video_tokens(seq_len, mm_counts),
            "audio": max_audio,
        }


class Lfm3OmniDummyInputsBuilder(Lfm3VlDummyInputsBuilder):
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        proc = self.info.get_hf_processor()
        video = f"{proc.vision_start_token}{proc.video_token}{proc.vision_end_token}"  # as the chat template
        return (
            proc.image_token * mm_counts.get("image", 0)
            + video * mm_counts.get("video", 0)
            + proc.audio_token * mm_counts.get("audio", 0)
        )

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: MultiModalDummyOptions,
    ) -> MultiModalDataDict:
        data = super().get_dummy_mm_data(seq_len, mm_counts, mm_options)
        sr = self.info.get_feature_extractor().sampling_rate
        data["audio"] = self._get_dummy_audios(
            length=self.info.get_max_audio_samples(seq_len),
            num_audios=mm_counts.get("audio", 0),
            overrides=mm_options.get("audio"),
        )
        return data


class Lfm3OmniMultiModalProcessor(Lfm3VlMultiModalProcessor):
    def _get_video_token_ids(self) -> tuple[int, int, int]:
        # the Omni config carries no vision token ids; the processor's tokens are the same
        proc = self.info.get_hf_processor()
        tokenizer = self.info.get_tokenizer()
        tokens = (proc.vision_start_token, proc.video_token, proc.vision_end_token)
        ids = tuple(tokenizer.convert_tokens_to_ids(t) for t in tokens)
        if any(i is None or i == tokenizer.unk_token_id for i in ids):
            raise ValueError(f"the tokenizer lacks the video tokens {tokens}")
        return ids

    def _finalize_video_repl(self, repl: PromptUpdateDetails) -> PromptUpdateDetails:
        # with vision markers the whole expansion is wrapped once more (Lfm3OmniProcessor._video_expansion)
        if not self.info.get_hf_processor().vision_markers:
            return repl
        start, _, end = self._get_video_token_ids()
        inner = repl.is_embed

        def is_embed(full: list[int]) -> torch.Tensor:  # the markers are text; the inner mask stays as it was
            mid = torch.ones(len(full) - 2, dtype=torch.bool) if inner is None else inner(full[1:-1])
            edge = torch.zeros(1, dtype=torch.bool)
            return torch.cat([edge, mid.to(torch.bool), edge])

        return PromptUpdateDetails(full=[start, *repl.full, end], is_embed=is_embed)

    def _extra_video_outputs(self, metadata, video_kwargs: Mapping[str, object]) -> dict[str, object]:
        # first-frame time of each temporal patch: with use_audio_in_video the soundtrack's tokens are split there
        indices = self.info._get_video_frame_indices(
            metadata, video_kwargs.get("do_sample_frames"), video_kwargs.get("fps"), video_kwargs.get("num_frames")
        )
        temporal = self.info.get_video_processor().temporal_patch_size
        fps = 24 if metadata["fps"] is None else round(metadata["fps"], 6)  # as the timestamps (Lfm3VlProcessingInfo)
        return {"frame_group_starts": torch.tensor([i / fps for i in indices[::temporal]], dtype=torch.float64)}

    def _apply_hf_processor_main(self, mm_items: MultiModalDataItems, hf_kwargs: Mapping[str, object]) -> BatchFeature:
        # use_audio_in_video is expanded here (_get_prompt_updates): the HF calls get the videos alone
        hf_kwargs = dict(hf_kwargs)
        use_audio_in_video = bool(hf_kwargs.pop("use_audio_in_video", False))
        if isinstance(videos_kwargs := hf_kwargs.get("videos_kwargs"), Mapping):
            videos_kwargs = dict(videos_kwargs)
            use_audio_in_video = bool(videos_kwargs.pop("use_audio_in_video", use_audio_in_video))
            hf_kwargs["videos_kwargs"] = videos_kwargs
        outputs = super()._apply_hf_processor_main(mm_items, hf_kwargs)
        if "video_grid_thw" in outputs:
            outputs["use_audio_in_video"] = torch.tensor(use_audio_in_video)
        return outputs

    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        num_videos = len(hf_inputs.get("video_grid_thw", ()))
        return dict(
            super()._get_mm_fields_config(hf_inputs, hf_processor_mm_kwargs),
            input_features=MultiModalFieldConfig.batched("audio"),
            feature_attention_mask=MultiModalFieldConfig.batched("audio"),
            frame_group_starts=MultiModalFieldConfig.batched("video", keep_on_cpu=True),
            use_audio_in_video=MultiModalFieldConfig.shared("video", num_videos),
        )

    def _soundtrack_video_repl(self, video_item, audio_item, hf_processor_mm_kwargs) -> PromptUpdateDetails:
        """A video with its soundtrack (Lfm3OmniProcessor._video_expansion): each frame group's video tokens followed
        by the soundtrack's encoder tokens up to the next group's first frame, without audio markers."""
        if self.info.ctx.get_mm_config().get_video_pruning_spec() is not None:
            raise ValueError("use_audio_in_video does not support video pruning")
        proc = self.info.get_hf_processor(**hf_processor_mm_kwargs)
        fe = self.info.get_feature_extractor(**hf_processor_mm_kwargs)
        tokenizer = self.info.get_tokenizer()
        start, video, end = self._get_video_token_ids()
        grid = video_item["video_grid_thw"].data.tolist()
        per_group = grid[1] * grid[2] // proc.video_processor.merge_size**2
        timestamps = list(video_item["timestamps"].data)
        starts = video_item["frame_group_starts"].data.tolist()
        assert len(timestamps) == len(starts) == grid[0], (len(timestamps), len(starts), grid)
        num_audio = fe.num_encoder_tokens(int(audio_item["feature_attention_mask"].data.sum()))
        max_audio = self.info.get_mm_max_tokens_per_item(self.info.ctx.model_config.max_model_len, {})["audio"]
        if num_audio > max_audio:
            raise ValueError(f"the soundtrack needs {num_audio} tokens, more than the {max_audio} allowed "
                             f"(at most {_MAX_AUDIO_SECONDS} s and max_model_len)")
        seconds_per_token = fe.hop_length * fe.encoder_stride / fe.sampling_rate
        bounds = [0] + [min(num_audio, round(t / seconds_per_token)) for t in starts[1:]] + [num_audio]
        bounds = list(itertools.accumulate(bounds, max))
        full: list[int] = []
        for i, timestamp in enumerate(timestamps):
            full += tokenizer.encode(f"<{float(timestamp):.1f} seconds>", add_special_tokens=False)
            full += [start] + [video] * per_group + [end] + [proc.audio_token_id] * (bounds[i + 1] - bounds[i])
        if proc.vision_markers:
            full = [start, *full, end]
        return PromptUpdateDetails.select_token_id(full, video)

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptUpdate]:
        updates = list(super()._get_prompt_updates(mm_items, hf_processor_mm_kwargs, out_mm_kwargs))
        audio_token_id = self.info.get_hf_processor(**hf_processor_mm_kwargs).audio_token_id

        fe = self.info.get_feature_extractor(**hf_processor_mm_kwargs)
        max_tokens = self.info.get_mm_max_tokens_per_item(self.info.ctx.model_config.max_model_len, {})["audio"]

        def audio_replacement(item_idx: int) -> list[int]:
            mask = out_mm_kwargs["audio"][item_idx]["feature_attention_mask"].data
            num_tokens = fe.num_encoder_tokens(int(mask.sum()))
            if num_tokens > max_tokens:
                raise ValueError(
                    f"Audio clip {item_idx} needs {num_tokens} tokens, more than the {max_tokens} allowed "
                    f"(at most {_MAX_AUDIO_SECONDS} s and max_model_len)"
                )
            return [audio_token_id] * num_tokens

        updates.append(PromptReplacement(modality="audio", target=[audio_token_id], replacement=audio_replacement))

        video_update = next((u for u in updates if u.modality == "video"), None)
        if video_update is not None:
            plain = video_update.replacement

            def video_replacement(item_idx: int):
                item = out_mm_kwargs["video"][item_idx]
                if not bool(item["use_audio_in_video"].data):
                    return plain(item_idx)
                if item_idx >= len(out_mm_kwargs.get("audio", [])):
                    raise ValueError("use_audio_in_video needs one audio clip (the soundtrack) per video")
                return self._soundtrack_video_repl(item, out_mm_kwargs["audio"][item_idx], hf_processor_mm_kwargs)

            updates[updates.index(video_update)] = PromptReplacement(
                modality="video", target=video_update.target, replacement=video_replacement
            )
        return updates

    def _maybe_apply_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        mm_res: MultiModalProcessingResult,
    ) -> tuple[list[int], Mapping[str, list[PlaceholderFeaturesInfo]]]:
        """Videos with their soundtracks (use_audio_in_video): the soundtrack's tokens sit inside the video's span,
        so its own audio span leaves the prompt and its placeholder is the video's, embedding at the audio tokens.
        Detected from the applied video updates, so cached items (no kwargs) are handled as well."""
        updates = mm_res.prompt_updates
        if not updates.get("video") or not updates.get("audio"):
            return super()._maybe_apply_prompt_updates(mm_items, mm_res)
        proc = self.info.get_hf_processor()
        audio_id = proc.audio_token_id
        without_audio = {k: v for k, v in updates.items() if k != "audio"}
        video_items = mm_res.kwargs.get("video", [])
        if all(item is not None for item in video_items):
            if not any(bool(item["use_audio_in_video"].data) for item in video_items):
                return super()._maybe_apply_prompt_updates(mm_items, mm_res)
        else:  # cached items carry no kwargs: the applied video updates tell
            _, trial = self._apply_prompt_updates(mm_res.prompt_ids, without_audio)
            if not any(audio_id in p.tokens for p in trial["video"]):
                return super()._maybe_apply_prompt_updates(mm_items, mm_res)

        counts = mm_items.get_all_counts()
        if counts.get("audio", 0) != counts.get("video", 0):
            raise ValueError("use_audio_in_video needs one audio clip (the soundtrack) per video")
        self._validate_mm_kwargs(mm_res.kwargs, counts)
        self._validate_mm_updates(updates, counts)
        start, video_id, end = self._get_video_token_ids()
        span = [start, audio_id, end] if proc.audio_markers else [audio_id]
        prompt_ids, i, removed = [], 0, 0
        ids = mm_res.prompt_ids
        while i < len(ids):
            if ids[i : i + len(span)] == span:
                i += len(span)
                removed += 1
            else:
                prompt_ids.append(ids[i])
                i += 1
        if removed != counts["audio"]:
            raise ValueError(f"expected {counts['audio']} soundtrack placeholders in the prompt, found {removed}")
        prompt_ids, placeholders = self._apply_prompt_updates(prompt_ids, without_audio)
        videos = placeholders["video"]
        placeholders = dict(placeholders)
        placeholders["audio"] = [
            PlaceholderFeaturesInfo("audio", v.item_idx, v.start_idx, v.tokens, torch.tensor(v.tokens) == audio_id)
            for v in videos
        ]
        self._validate_mm_placeholders(placeholders, counts)
        return prompt_ids, placeholders


@MULTIMODAL_REGISTRY.register_processor(
    Lfm3OmniMultiModalProcessor,
    info=Lfm3OmniProcessingInfo,
    dummy_inputs=Lfm3OmniDummyInputsBuilder,
)
class Lfm3OmniForConditionalGeneration(Lfm3VlForConditionalGeneration):
    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
            **Lfm3VlForConditionalGeneration.hf_to_vllm_mapper.orig_to_new_prefix,
            "model.audio_encoder.": "audio_encoder.",
            "model.audio_projector.": "audio_projector.",
            # the speech output head; text generation does not use it
            "model.depth_decoder.": None,
            "model.codebook_embedding.": None,
        },
        orig_to_new_stacked=Lfm3VlForConditionalGeneration.hf_to_vllm_mapper.orig_to_new_stacked,
    )

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        if modality.startswith("image"):
            return "<|image|>"
        if modality.startswith("video"):
            return "<|vision_start|><|video|><|vision_end|>"
        if modality.startswith("audio"):
            return "<|audio|>"
        raise ValueError("Only image, video and audio modalities are supported")

    def __init__(self, *, vllm_config, prefix: str = "") -> None:
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        config = vllm_config.model_config.hf_config
        with self._mark_tower_model(vllm_config, "audio"):
            self.audio_encoder = Lfm3OmniAudioEncoder(
                config.audio_encoder_config, None, maybe_prefix(prefix, "audio_encoder")
            )
            self.audio_projector = Lfm3OmniAudioProjector(
                config, None, maybe_prefix(prefix, "audio_projector")
            )

    def _process_audio_input(self, audio: Lfm3OmniAudioInputs) -> list[torch.Tensor]:
        # iterating a stacked tensor or a list both give one item per clip; one host sync for all lengths
        lengths = [int(n) for n in torch.stack([m.sum() for m in audio["feature_attention_mask"]]).tolist()]
        return [
            self.audio_projector(self.audio_encoder(features, n))[: _num_audio_tokens(n)]
            for features, n in zip(audio["input_features"], lengths)
        ]

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings=None,
        *,
        is_multimodal: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # a video with its soundtrack (use_audio_in_video) interleaves video and audio positions while the
        # embeddings come per item, so with both modalities present they are scattered by modality (tagged at the
        # encoder gather) instead of in item order; LFM3's span keeps markers and timestamps between the runs
        modalities = {getattr(e, "modality", None) for e in multimodal_embeddings or ()}
        if is_multimodal is None or not {"video", "audio"} <= modalities:
            return super().embed_input_ids(input_ids, multimodal_embeddings, is_multimodal=is_multimodal)
        ids, mm = input_ids.reshape(-1), is_multimodal.reshape(-1)  # the HPU runner may pass [B, S] ids, a flat mask
        inputs_embeds = self._embed_text_input_ids(ids, self.get_language_model().embed_input_ids, is_multimodal=mm)
        merge_interleaved_embeddings(
            inputs_embeds, multimodal_embeddings, mm & (ids == self.config.video_token_id),
            mm & (ids == self.config.audio_token_id), mm
        )
        return inputs_embeds.view(*input_ids.shape, inputs_embeds.shape[-1])

    def _embed_modality(self, key: str, kwargs: dict[str, object]) -> Sequence[torch.Tensor]:
        if key == "input_features":
            return self._process_audio_input(
                Lfm3OmniAudioInputs(
                    input_features=kwargs["input_features"],
                    feature_attention_mask=kwargs["feature_attention_mask"],
                )
            )
        return super()._embed_modality(key, kwargs)

