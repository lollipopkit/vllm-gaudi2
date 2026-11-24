# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""LFM3-VL (LiquidAI): the LFM3 text model behind a vision encoder of gated depthwise
2-D convolutions and bidirectional attention over 16x16 patches (axial 2-D RoPE),
merged 2x2 by a GELU projector. Image and video inputs are Qwen2-VL style: flattened
patches grouped by merge block plus a (t, h, w) grid per item; video prompts carry
Qwen3-VL style per-frame timestamps."""

from collections.abc import Iterable, Mapping, Sequence
from typing import Annotated, Any, Literal

import numpy as np
import torch
import torch.nn as nn
from transformers import BatchFeature
from transformers.models.qwen2_vl.image_processing_qwen2_vl import (
    smart_resize as image_smart_resize,
)
from transformers.models.qwen3_vl.video_processing_qwen3_vl import (
    smart_resize as video_smart_resize,
)
from transformers.video_processing_utils import BaseVideoProcessor
from transformers.video_utils import VideoMetadata

from vllm.config import VllmConfig
from vllm.config.multimodal import MultiModalDummyOptions
from vllm.inputs import MultiModalDataDict
from vllm.model_executor.layers.activation import SiluAndMul, get_act_fn
from vllm.model_executor.layers.attention import MMEncoderAttention
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.mamba.mamba_utils import (
    MambaStateCopyFunc,
    MambaStateCopyFuncCalculator,
    MambaStateDtypeCalculator,
    MambaStateShapeCalculator,
)
from vllm.model_executor.layers.rotary_embedding.common import ApplyRotaryEmb
from vllm.model_executor.models.interfaces import IsHybrid, MultiModalEmbeddings, SupportsMultiModal, SupportsPP
from vllm.model_executor.models.qwen3_vl import (
    Qwen3VLDummyInputsBuilder,
    Qwen3VLMultiModalProcessor,
    Qwen3VLProcessingInfo,
    _replace_video_token_placeholders,
)
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    WeightsMapper,
    init_vllm_registered_model,
    maybe_prefix,
)
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import MultiModalKwargsItems
from vllm.multimodal.parse import ImageSize, MultiModalDataItems
from vllm.multimodal.processing import PromptReplacement, PromptUpdate, PromptUpdateDetails
from vllm.multimodal.video_prune.evs import compute_retained_tokens_count
from vllm.multimodal.video_prune.vidcom2 import (
    compute_retained_tokens_count as vidcom2_compute_retained_tokens_count,
)
from vllm.sequence import IntermediateTensors
from vllm.utils.collection_utils import is_list_of
from vllm.utils.math_utils import round_up
from vllm.utils.tensor_schema import TensorSchema, TensorShape


class Lfm3VlImagePixelInputs(TensorSchema):
    """
    Dimensions:
        - np: number of patches over all images
        - cps: in_channels * temporal_patch_size * patch_size ** 2
        - ni: number of images
    """

    type: Literal["pixel_values"] = "pixel_values"
    pixel_values: Annotated[torch.Tensor, TensorShape("np", "cps")]
    image_grid_thw: Annotated[torch.Tensor, TensorShape("ni", 3)]


class Lfm3VlVideoPixelInputs(TensorSchema):
    """
    Dimensions:
        - np: number of patches over all videos
        - cps: in_channels * temporal_patch_size * patch_size ** 2
        - nv: number of videos
    """

    type: Literal["pixel_values_videos"] = "pixel_values_videos"
    pixel_values_videos: Annotated[torch.Tensor, TensorShape("np", "cps")]
    video_grid_thw: Annotated[torch.Tensor, TensorShape("nv", 3)]


class Lfm3VlVisionMLP(nn.Module):
    def __init__(self, config, quant_config=None, prefix: str = ""):
        super().__init__()
        self.w13 = MergedColumnParallelLinear(
            config.hidden_size,
            [config.intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.w13",
        )
        self.w2 = RowParallelLinear(
            config.intermediate_size,
            config.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.w2",
        )
        self.act_fn = SiluAndMul()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up, _ = self.w13(x)
        x, _ = self.w2(self.act_fn(gate_up))
        return x


class Lfm3VlVisionAttention(nn.Module):
    """Bidirectional grouped-query attention over the patches of one frame."""

    def __init__(self, config, quant_config=None, prefix: str = ""):
        super().__init__()
        self.total_num_heads = config.num_attention_heads
        self.total_num_kv_heads = config.num_key_value_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.qkv_proj = QKVParallelLinear(
            hidden_size=config.hidden_size,
            head_size=self.head_dim,
            total_num_heads=self.total_num_heads,
            total_num_kv_heads=self.total_num_kv_heads,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )
        self.num_heads = self.qkv_proj.num_heads
        self.num_kv_heads = self.qkv_proj.num_kv_heads
        self.out_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            config.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.out_proj",
        )
        self.q_layernorm = RMSNorm(self.head_dim, eps=config.norm_eps)
        self.k_layernorm = RMSNorm(self.head_dim, eps=config.norm_eps)
        self.attn = MMEncoderAttention(
            num_heads=self.num_heads,
            head_size=self.head_dim,
            scale=self.head_dim**-0.5,
            num_kv_heads=self.num_kv_heads,
            prefix=f"{prefix}.attn",
        )
        self.apply_rotary_emb = ApplyRotaryEmb(enforce_enable=True)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int | None,
    ) -> torch.Tensor:
        n = x.shape[0]
        qkv, _ = self.qkv_proj(x)
        q_size, kv_size = self.num_heads * self.head_dim, self.num_kv_heads * self.head_dim
        q, k, v = qkv.split([q_size, kv_size, kv_size], dim=-1)
        q = self.q_layernorm(q.view(n, self.num_heads, self.head_dim)).unsqueeze(0)
        k = self.k_layernorm(k.view(n, self.num_kv_heads, self.head_dim)).unsqueeze(0)
        v = v.view(1, n, self.num_kv_heads, self.head_dim)
        q = self.apply_rotary_emb(q, cos, sin)
        k = self.apply_rotary_emb(k, cos, sin)
        out = self.attn(query=q, key=k, value=v, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen)
        out, _ = self.out_proj(out.reshape(n, self.num_heads * self.head_dim))
        return out


class Lfm3VlVisionConv2d(nn.Module):
    """Gated depthwise 2-D convolution over the patch grid of each frame (patches arrive
    in merge-block order and are convolved on the raster grid)."""

    def __init__(self, config, quant_config=None, prefix: str = ""):
        super().__init__()
        hidden = config.hidden_size
        self.merge = config.spatial_merge_size
        # Replicated: the depthwise conv weights are per channel and not sharded.
        self.in_proj = ReplicatedLinear(
            hidden, 3 * hidden, bias=False, quant_config=quant_config, prefix=f"{prefix}.in_proj"
        )
        self.spatial_conv = nn.Conv2d(
            hidden,
            hidden,
            kernel_size=config.conv_kernel_size,
            padding=config.conv_kernel_size // 2,
            groups=hidden,
            bias=False,
        )
        self.out_proj = ReplicatedLinear(
            hidden, hidden, bias=False, quant_config=quant_config, prefix=f"{prefix}.out_proj"
        )

    def forward(self, x: torch.Tensor, frame_shapes: list[tuple[int, int]]) -> torch.Tensor:
        bcx, _ = self.in_proj(x)
        b, c, x = bcx.chunk(3, dim=-1)
        gated = b * x
        m, ch = self.merge, gated.shape[-1]
        outs = []
        offset = 0
        # Explicit sizes, no -1: after a resolution change the HPU backend recompiles with symbolic
        # shapes and infers -1 wrongly (a [C, h/m, m, w/m, m] -> [1, -1, h, w] reshape gave [1, 0, h, w]).
        for h, w in frame_shapes:
            frame = gated[offset : offset + h * w]
            offset += h * w
            # merge-block order -> [C, h, w] raster grid
            grid = frame.view(h // m, w // m, m, m, ch).permute(4, 0, 2, 1, 3).reshape(1, ch, h, w)
            grid = self.spatial_conv(grid.to(self.spatial_conv.weight.dtype)).to(frame.dtype)
            outs.append(grid.view(ch, h // m, m, w // m, m).permute(1, 3, 2, 4, 0).reshape(h * w, ch))
        out, _ = self.out_proj(c * torch.cat(outs, dim=0))
        return out


class Lfm3VlVisionLayer(nn.Module):
    def __init__(self, config, layer_idx: int, quant_config=None, prefix: str = ""):
        super().__init__()
        self.is_attention_layer = config.layer_types[layer_idx] == "full_attention"
        if self.is_attention_layer:
            self.self_attn = Lfm3VlVisionAttention(config, quant_config, f"{prefix}.self_attn")
        else:
            self.conv = Lfm3VlVisionConv2d(config, quant_config, f"{prefix}.conv")
        self.feed_forward = Lfm3VlVisionMLP(config, quant_config, f"{prefix}.feed_forward")
        self.operator_norm = RMSNorm(config.hidden_size, eps=config.norm_eps)
        self.ffn_norm = RMSNorm(config.hidden_size, eps=config.norm_eps)

    def forward(self, x, cos, sin, cu_seqlens, max_seqlen, frame_shapes):
        normed = self.operator_norm(x)
        if self.is_attention_layer:
            x = x + self.self_attn(normed, cos, sin, cu_seqlens, max_seqlen)
        else:
            x = x + self.conv(normed, frame_shapes)
        return x + self.feed_forward(self.ffn_norm(x))


class Lfm3VlVisionModel(nn.Module):
    layer_cls = Lfm3VlVisionLayer  # platform plugins may substitute the layer implementation

    def __init__(self, config, quant_config=None, prefix: str = ""):
        super().__init__()
        self.config = config
        self.merge = config.spatial_merge_size
        self.patch_size = config.patch_size
        patch_dim = config.in_channels * config.temporal_patch_size * config.patch_size**2
        self.patchifier = ReplicatedLinear(
            patch_dim, config.hidden_size, bias=False, prefix=f"{prefix}.patchifier"
        )
        self.embedding_norm = RMSNorm(config.hidden_size, eps=config.norm_eps)
        self.layers = nn.ModuleList(
            [
                self.layer_cls(config, i, quant_config, f"{prefix}.layers.{i}")
                for i in range(config.num_hidden_layers)
            ]
        )
        # Axial RoPE: the same frequencies for the h and w axes, each over a quarter of
        # the head dim; (h, w) angles concatenated and then duplicated for rotate_half.
        head_dim = config.hidden_size // config.num_attention_heads
        axis_dim = head_dim // 2
        theta = config.rope_parameters["rope_theta"]
        inv_freq = 1.0 / (theta ** (torch.arange(0, axis_dim, 2, dtype=torch.float32) / axis_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @property
    def dtype(self) -> torch.dtype:
        return self.patchifier.weight.dtype

    def rotary_cos_sin(self, grid_thw: list[list[int]], device) -> tuple[torch.Tensor, torch.Tensor]:
        pos = []
        for t, h, w in grid_thw:
            hpos = torch.arange(h, device=device).unsqueeze(1).expand(-1, w)
            wpos = torch.arange(w, device=device).unsqueeze(0).expand(h, -1)
            block = (h // self.merge, self.merge, w // self.merge, self.merge)
            hpos = hpos.reshape(block).permute(0, 2, 1, 3).flatten()
            wpos = wpos.reshape(block).permute(0, 2, 1, 3).flatten()
            pos.append(torch.stack([hpos, wpos], dim=-1).repeat(t, 1))
        pos = torch.cat(pos, dim=0).float()  # [N, 2]
        freqs = (pos[..., None] * self.inv_freq).flatten(1)  # [N, head_dim / 2]: h angles, then w
        return freqs.cos(), freqs.sin()

    def forward(self, pixel_values: torch.Tensor, grid_thw: list[list[int]]) -> torch.Tensor:
        x, _ = self.patchifier(pixel_values.to(self.dtype))
        x = self.embedding_norm(x)
        cos, sin = self.rotary_cos_sin(grid_thw, x.device)
        frame_shapes = [(h, w) for t, h, w in grid_thw for _ in range(t)]
        lengths = [h * w for h, w in frame_shapes]
        cu_seqlens = torch.tensor([0] + lengths, device=x.device, dtype=torch.int32).cumsum(0, dtype=torch.int32)
        max_seqlen = max(lengths)
        for layer in self.layers:
            x = layer(x, cos, sin, cu_seqlens, max_seqlen, frame_shapes)
        return x


class Lfm3VlMultiModalProjector(nn.Module):
    """Merges each block of downsample_factor ** 2 vision tokens into one text token."""

    def __init__(self, config, quant_config=None, prefix: str = ""):
        super().__init__()
        vision_hidden = config.vision_config.hidden_size
        self.factor = config.downsample_factor
        self.layer_norm = nn.LayerNorm(vision_hidden, eps=config.projector_layer_norm_eps)
        self.linear_1 = ColumnParallelLinear(
            vision_hidden * self.factor**2,
            config.projector_hidden_size,
            bias=config.projector_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.linear_1",
        )
        self.act = get_act_fn(config.projector_hidden_act)
        self.linear_2 = RowParallelLinear(
            config.projector_hidden_size,
            config.text_config.hidden_size,
            bias=config.projector_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.linear_2",
        )

    def forward(self, image_features: torch.Tensor) -> torch.Tensor:
        x = self.layer_norm(image_features)
        x = x.reshape(x.shape[0] // self.factor**2, self.linear_1.input_size)
        x, _ = self.linear_1(x)
        x, _ = self.linear_2(self.act(x))
        return x


class Lfm3VlProcessingInfo(Qwen3VLProcessingInfo):
    def get_hf_config(self):
        return self.ctx.get_hf_config()

    def get_hf_processor(self, **kwargs: object):
        return self.ctx.get_hf_processor(**kwargs)

    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        return {"image": None, "video": None}

    def get_supported_mm_processor_kwargs(self) -> dict[str, set[str]]:
        supported = super(Qwen3VLProcessingInfo, self).get_supported_mm_processor_kwargs()
        if "videos_kwargs" in supported:
            supported["videos_kwargs"].update({"min_pixels", "max_pixels"})
        return supported

    def _get_vision_info(
        self,
        *,
        image_width: int,
        image_height: int,
        num_frames: int = 2,
        do_resize: bool = True,
        image_processor,
        mm_kwargs: Mapping[str, object],
        modality: Literal["image", "video"] | None = None,
    ) -> tuple[ImageSize, int]:
        # LFM3's HF video processor is a BaseVideoProcessor subclass, not Qwen3VLVideoProcessor.
        is_video = isinstance(image_processor, BaseVideoProcessor)

        hf_config = self.get_hf_config()
        vision_config = hf_config.vision_config
        patch_size = vision_config.patch_size
        merge_size = vision_config.spatial_merge_size
        temporal_patch_size = vision_config.temporal_patch_size

        if modality is None:
            modality = "video" if is_video else "image"

        merged_mm_kwargs = self._merge_and_resolve_mm_processor_kwargs(mm_kwargs)
        scope = "videos_kwargs" if modality == "video" else "images_kwargs"
        scoped_mm_kwargs = merged_mm_kwargs.get(scope, {})
        size = self._get_vision_size(
            scoped_mm_kwargs,
            default_size=image_processor.size,
        )

        if do_resize:
            if is_video:
                smart_resize = video_smart_resize
                extra_kwargs = {
                    "num_frames": num_frames,
                    "temporal_factor": temporal_patch_size,
                }
            else:
                smart_resize = image_smart_resize
                extra_kwargs = {}

            resized_height, resized_width = smart_resize(
                height=image_height,
                width=image_width,
                factor=patch_size * merge_size,
                min_pixels=size["shortest_edge"],
                max_pixels=size["longest_edge"],
                **extra_kwargs,
            )
            preprocessed_size = ImageSize(width=resized_width, height=resized_height)
        else:
            preprocessed_size = ImageSize(width=image_width, height=image_height)

        padded_num_frames = round_up(num_frames, temporal_patch_size)
        grid_t = max(padded_num_frames // temporal_patch_size, 1)
        grid_h = preprocessed_size.height // patch_size
        grid_w = preprocessed_size.width // patch_size

        num_patches = grid_t * grid_h * grid_w
        num_vision_tokens = num_patches // (merge_size**2)

        return preprocessed_size, num_vision_tokens

    def _calculate_timestamps(
        self, indices: list[int] | torch.Tensor, video_fps: float | None, merge_size: int
    ) -> list[float]:
        # as LFM3's replace_video_token: decoder float noise (30.000000000000004) rounded off, 24 fps if unknown
        fps = 24 if video_fps is None else round(video_fps, 6)
        return super()._calculate_timestamps(indices, fps, merge_size)

    def _get_video_second_idx(
        self,
        metadata: dict[str, Any],
        do_sample_frames: bool | None = None,
        sampled_fps: float | None = None,
        sampled_num_frames: int | None = None,
    ) -> list[float]:
        temporal_patch_size = self.get_video_processor().temporal_patch_size
        indices = self._get_video_frame_indices(
            metadata, do_sample_frames, sampled_fps, sampled_num_frames
        )
        return self._calculate_timestamps(indices, metadata["fps"], temporal_patch_size)

    def _get_video_frame_indices(
        self,
        metadata: dict[str, Any],
        do_sample_frames: bool | None = None,
        sampled_fps: float | None = None,
        sampled_num_frames: int | None = None,
    ) -> list[int]:
        """Indices of the frames the HF processor keeps, in the original clip."""
        video_processor = self.get_video_processor()
        indices = metadata["frames_indices"]
        if do_sample_frames is None:
            do_sample_frames = metadata.get("do_sample_frames", False)

        if do_sample_frames:
            total_num_frames = metadata["total_num_frames"]
            if sampled_num_frames is not None:
                num_frames = sampled_num_frames
            else:
                sampled_fps = sampled_fps if sampled_fps else video_processor.fps
                num_frames = int(total_num_frames / metadata["fps"] * sampled_fps)

            num_frames = min(
                min(
                    max(num_frames, video_processor.min_frames),
                    video_processor.max_frames,
                ),
                total_num_frames,
            )
            indices = (
                np.linspace(0, total_num_frames - 1, num_frames)
                .round()
                .astype(int)
                .tolist()
            )
        return indices


class Lfm3VlDummyInputsBuilder(Qwen3VLDummyInputsBuilder):
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        proc = self.info.get_hf_processor()
        video = f"{proc.vision_start_token}{proc.video_token}{proc.vision_end_token}"
        return proc.image_token * mm_counts.get("image", 0) + video * mm_counts.get(
            "video", 0
        )

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: MultiModalDummyOptions,
    ) -> MultiModalDataDict:
        if not mm_counts.get("video", 0):
            target_image_width, target_image_height = (
                self.info.get_image_size_with_most_features()
            )
            return {
                "image": self._get_dummy_images(
                    width=target_image_width,
                    height=target_image_height,
                    num_images=mm_counts.get("image", 0),
                    overrides=mm_options.get("image"),
                ),
                "video": [],
            }
        return super().get_dummy_mm_data(seq_len, mm_counts, mm_options)


class Lfm3VlMultiModalProcessor(Qwen3VLMultiModalProcessor):
    # LFM3's HF processor rejects video placeholders that arrive without video tensors.
    hf_accepts_unfilled_video_placeholders = False

    def _get_video_token_ids(self) -> tuple[int, int, int]:
        """(vision start, video, vision end) token ids of the video placeholder."""
        hf_config = self.info.get_hf_config()
        return (
            hf_config.vision_start_token_id,
            hf_config.video_token_id,
            hf_config.vision_end_token_id,
        )

    def _finalize_video_repl(self, repl: PromptUpdateDetails) -> PromptUpdateDetails:
        """Hook for subclasses (e.g. Lfm3Omni) that wrap a video's expansion further."""
        return repl

    def _extra_video_outputs(
        self, metadata: VideoMetadata, video_kwargs: Mapping[str, object]
    ) -> dict[str, Any]:
        """Hook: further per-video outputs (batched by the subclass's field config)."""
        return {}

    def _apply_hf_processor_main(
        self,
        mm_items: MultiModalDataItems,
        hf_kwargs: Mapping[str, object],
    ) -> BatchFeature:
        hf_data, hf_kwargs, passthrough_data = self._get_hf_mm_inputs(
            mm_items, hf_kwargs
        )

        if not hf_data:
            return self._finalize_hf_mm_data(hf_data, hf_kwargs, passthrough_data)

        merged_mm_kwargs = self.info._merge_and_resolve_mm_processor_kwargs(hf_kwargs)

        if "images_kwargs" in merged_mm_kwargs:
            image_scoped_kwargs = dict(merged_mm_kwargs["images_kwargs"])
            if image_scoped_kwargs.keys() & {"size", "min_pixels", "max_pixels"}:
                default_image_size = self.info.get_image_processor().size
                image_scoped_kwargs["size"] = self.info._get_vision_size(
                    image_scoped_kwargs,
                    default_size=default_image_size,
                )
            merged_mm_kwargs["images_kwargs"] = image_scoped_kwargs

        if "videos_kwargs" in merged_mm_kwargs:
            video_scoped_kwargs = dict(merged_mm_kwargs["videos_kwargs"])
            if video_scoped_kwargs.keys() & {"size", "min_pixels", "max_pixels"}:
                default_video_size = self.info.get_video_processor().size
                video_scoped_kwargs["size"] = self.info._get_vision_size(
                    video_scoped_kwargs,
                    default_size=default_video_size,
                )
            video_scoped_kwargs.pop("min_pixels", None)
            video_scoped_kwargs.pop("max_pixels", None)
            merged_mm_kwargs["videos_kwargs"] = video_scoped_kwargs

        video_input_ids_lst: list[list[int]] = []
        if videos := hf_data.pop("videos", []):
            assert isinstance(videos, list)
            self._get_per_video_kwargs(hf_kwargs, len(videos))
            video_grid_thw_lst = []
            pixel_values_videos_lst = []
            timestamps_per_video = []
            extra_video_outputs: dict[str, list[Any]] = {}

            hf_config = self.info.get_hf_config()
            tokenizer = self.info.get_tokenizer()
            merge_size = hf_config.vision_config.spatial_merge_size
            pruning_spec = self.info.ctx.get_mm_config().get_video_pruning_spec()
            vision_start_token_id, video_token_id, vision_end_token_id = (
                self._get_video_token_ids()
            )

            for item_idx, item in enumerate(videos):
                video_array, metadata = item

                video_mm_kwargs = dict(merged_mm_kwargs)
                video_scoped_kwargs = dict(video_mm_kwargs.get("videos_kwargs", {}))
                video_mm_kwargs["videos_kwargs"] = video_scoped_kwargs

                sampled_fps = video_scoped_kwargs.get("fps")
                if is_list_of(sampled_fps, float):
                    video_scoped_kwargs["fps"] = sampled_fps[item_idx]
                sampled_num_frames = video_scoped_kwargs.get("num_frames")
                if is_list_of(sampled_num_frames, int):
                    video_scoped_kwargs["num_frames"] = sampled_num_frames[item_idx]
                if "do_sample_frames" not in video_scoped_kwargs:
                    video_scoped_kwargs["do_sample_frames"] = metadata.get(
                        "do_sample_frames", False
                    )

                metadata = VideoMetadata(
                    **{k: metadata[k] for k in metadata if k != "do_sample_frames"}
                )

                timestamps = self.info._get_video_second_idx(
                    metadata=metadata,
                    do_sample_frames=video_scoped_kwargs["do_sample_frames"],
                    sampled_fps=video_scoped_kwargs.get("fps"),
                    sampled_num_frames=video_scoped_kwargs.get("num_frames"),
                )
                timestamps_per_video.append(timestamps)

                video_mm_data = {
                    "videos": [[video_array]],
                    "video_metadata": [[metadata]],
                }
                if (
                    "num_frames" in video_scoped_kwargs
                    and "fps" not in video_scoped_kwargs
                ):
                    video_scoped_kwargs["fps"] = None

                video_hf_processor = self.info.get_hf_processor(**video_mm_kwargs)
                video_outputs = self.info.ctx.call_hf_processor(
                    video_hf_processor,
                    dict(
                        text="<|vision_start|>"
                        f"{video_hf_processor.video_token}<|vision_end|>",
                        **video_mm_data,
                    ),
                    video_mm_kwargs,
                )
                video_outputs.pop("input_ids", None)

                video_grid_thw = video_outputs["video_grid_thw"]
                num_frames = int(video_grid_thw[0, 0])
                tokens_per_frame_base = int(video_grid_thw[0, 1:].prod()) // (
                    merge_size**2
                )

                if pruning_spec is not None:
                    method, prune_q = pruning_spec
                    count_fn = (
                        vidcom2_compute_retained_tokens_count
                        if method == "vidcom2"
                        else compute_retained_tokens_count
                    )
                    num_tokens = count_fn(
                        tokens_per_frame=tokens_per_frame_base,
                        num_frames=num_frames,
                        q=prune_q,
                    )
                    tokens_per_frame = [num_tokens] + [0] * (num_frames - 1)
                    select_token_id = False
                else:
                    tokens_per_frame = [tokens_per_frame_base] * num_frames
                    select_token_id = True

                video_repl = self._finalize_video_repl(
                    Qwen3VLMultiModalProcessor.get_video_repl(
                        tokens_per_frame=tokens_per_frame,
                        timestamps=timestamps,
                        tokenizer=tokenizer,
                        vision_start_token_id=vision_start_token_id,
                        vision_end_token_id=vision_end_token_id,
                        video_token_id=video_token_id,
                        select_token_id=select_token_id,
                    )
                )
                video_input_ids_lst.append(list(video_repl.full))

                video_grid_thw_lst.append(video_outputs["video_grid_thw"])
                pixel_values_videos_lst.append(video_outputs["pixel_values_videos"])
                for key, value in self._extra_video_outputs(
                    metadata, video_scoped_kwargs
                ).items():
                    extra_video_outputs.setdefault(key, []).append(value)

            video_outputs = dict(
                pixel_values_videos=torch.cat(pixel_values_videos_lst),
                video_grid_thw=torch.cat(video_grid_thw_lst),
                timestamps=timestamps_per_video,
                **extra_video_outputs,
            )
        else:
            video_outputs = dict()

        non_video_mm_kwargs = dict(merged_mm_kwargs)
        if "videos_kwargs" in non_video_mm_kwargs:
            non_video_scoped_kwargs = dict(non_video_mm_kwargs["videos_kwargs"])
            non_video_scoped_kwargs.pop("fps", None)
            non_video_scoped_kwargs.pop("num_frames", None)
            for key in self._get_per_video_kwargs({}):
                non_video_scoped_kwargs[key] = None
            non_video_mm_kwargs["videos_kwargs"] = non_video_scoped_kwargs

        strip_videos = (
            video_input_ids_lst and not self.hf_accepts_unfilled_video_placeholders
        )
        if strip_videos and hf_data.keys() <= {"text"}:
            processed_data = BatchFeature()
        else:
            if strip_videos:
                hf_data = dict(
                    hf_data,
                    text=self._get_hf_mm_text(
                        {**mm_items.get_all_counts(), "video": 0}
                    ),
                )
            processed_data = self.info.ctx.call_hf_processor(
                self.info.get_hf_processor(**non_video_mm_kwargs),
                hf_data,
                non_video_mm_kwargs,
            )

        if strip_videos:
            processed_data.pop("input_ids", None)
        elif video_input_ids_lst:
            start_id, video_id, end_id = self._get_video_token_ids()
            if self._expands_only_video_token(self.info.get_hf_processor()):
                video_target = [video_id]
            else:
                video_target = [start_id, video_id, end_id]
            input_ids = processed_data.pop("input_ids")
            if not isinstance(input_ids, list):
                input_ids = input_ids.tolist()
            (prompt_ids,) = input_ids
            expanded_ids = _replace_video_token_placeholders(
                prompt_ids, video_target, video_input_ids_lst
            )
            processed_data["input_ids"] = [expanded_ids]

        processed_data.update(video_outputs)
        return self._finalize_hf_mm_data(
            hf_data, hf_kwargs, passthrough_data, processed_data
        )

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, Any],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptUpdate]:
        hf_processor = self.info.get_hf_processor(**hf_processor_mm_kwargs)
        image_processor = self.info.get_image_processor(**hf_processor_mm_kwargs)
        tokenizer = self.info.get_tokenizer()
        merge_length = image_processor.merge_size**2

        def get_image_replacement(item_idx: int):
            out_item = out_mm_kwargs["image"][item_idx]
            grid_thw = out_item["image_grid_thw"].data
            assert isinstance(grid_thw, torch.Tensor)
            num_tokens = int(grid_thw.prod()) // merge_length
            return [hf_processor.image_token_id] * num_tokens

        image_update = PromptReplacement(
            modality="image",
            target=[hf_processor.image_token_id],
            replacement=get_image_replacement,
        )
        if "video" not in self.info.get_supported_mm_limits():
            return [image_update]

        vision_start_token_id, video_token_id, vision_end_token_id = (
            self._get_video_token_ids()
        )

        def get_video_replacement(item_idx: int):
            out_item = out_mm_kwargs["video"][item_idx]
            grid_thw = out_item["video_grid_thw"].data
            assert isinstance(grid_thw, torch.Tensor)

            timestamps = out_item["timestamps"].data
            assert isinstance(timestamps, (list, torch.Tensor))
            assert len(timestamps) == grid_thw[0], (
                f"The timestamps length({len(timestamps)}) should be equal "
                f"video length ({grid_thw[0]})."
            )

            num_frames = int(grid_thw[0])
            tokens_per_frame_base = int(grid_thw[1:].prod()) // merge_length

            pruning_spec = self.info.ctx.get_mm_config().get_video_pruning_spec()
            if pruning_spec is not None:
                method, prune_q = pruning_spec
                count_fn = (
                    vidcom2_compute_retained_tokens_count
                    if method == "vidcom2"
                    else compute_retained_tokens_count
                )
                num_tokens = count_fn(
                    tokens_per_frame=tokens_per_frame_base,
                    num_frames=num_frames,
                    q=prune_q,
                )
                tokens_per_frame = [num_tokens] + [0] * (num_frames - 1)
                select_token_id = False
            else:
                tokens_per_frame = [tokens_per_frame_base] * num_frames
                select_token_id = True

            return self._finalize_video_repl(
                Qwen3VLMultiModalProcessor.get_video_repl(
                    tokens_per_frame=tokens_per_frame,
                    timestamps=timestamps,
                    tokenizer=tokenizer,
                    vision_start_token_id=vision_start_token_id,
                    vision_end_token_id=vision_end_token_id,
                    video_token_id=video_token_id,
                    select_token_id=select_token_id,
                )
            )

        if self._expands_only_video_token(hf_processor):
            video_target = [video_token_id]
        else:
            video_target = [
                vision_start_token_id,
                video_token_id,
                vision_end_token_id,
            ]

        return [
            image_update,
            PromptReplacement(
                modality="video",
                target=video_target,
                replacement=get_video_replacement,
            ),
        ]


@MULTIMODAL_REGISTRY.register_processor(
    Lfm3VlMultiModalProcessor,
    info=Lfm3VlProcessingInfo,
    dummy_inputs=Lfm3VlDummyInputsBuilder,
)
class Lfm3VlForConditionalGeneration(nn.Module, SupportsMultiModal, SupportsPP, IsHybrid):
    vision_tower_cls = Lfm3VlVisionModel  # platform plugins may substitute the tower implementation

    merge_by_field_config = True
    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
            "lm_head.": "language_model.lm_head.",
            "model.language_model.": "language_model.model.",
            "model.vision_tower.": "vision_tower.",
            "model.multi_modal_projector.": "multi_modal_projector.",
        },
        # Also applied to the language model's weights, whose own mapper (same keys)
        # runs again inside it; the trailing dots keep the second pass a no-op.
        orig_to_new_stacked={
            ".q_proj.": (".qkv_proj.", "q"),
            ".k_proj.": (".qkv_proj.", "k"),
            ".v_proj.": (".qkv_proj.", "v"),
            ".w1.": (".w13.", 0),
            ".w3.": (".w13.", 1),
        },
    )

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        if modality.startswith("image"):
            return "<|image|>"
        if modality.startswith("video"):
            return "<|vision_start|><|video|><|vision_end|>"
        raise ValueError("Only image and video modalities are supported")

    @classmethod
    def get_mamba_state_dtype_from_config(cls, vllm_config: "VllmConfig") -> tuple[torch.dtype, ...]:
        return MambaStateDtypeCalculator.short_conv_state_dtype(
            vllm_config.model_config.dtype,
            vllm_config.cache_config.mamba_cache_dtype,
        )

    @classmethod
    def get_mamba_state_shape_from_config(cls, vllm_config: "VllmConfig") -> tuple[tuple[int, int]]:
        text_config = vllm_config.model_config.hf_config.text_config
        return MambaStateShapeCalculator.short_conv_state_shape(
            tp_world_size=vllm_config.parallel_config.tensor_parallel_size,
            intermediate_size=text_config.hidden_size,
            conv_kernel=text_config.conv_L_cache,
        )

    @classmethod
    def get_mamba_state_copy_func(cls) -> tuple[MambaStateCopyFunc]:
        return MambaStateCopyFuncCalculator.short_conv_state_copy_func()

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        self.config = config
        self.vllm_config = vllm_config
        self.multimodal_config = vllm_config.model_config.multimodal_config
        with self._mark_tower_model(vllm_config, {"image", "video"}):
            self.vision_tower = self.vision_tower_cls(
                config.vision_config, quant_config, maybe_prefix(prefix, "vision_tower")
            )
            self.multi_modal_projector = Lfm3VlMultiModalProjector(
                config, quant_config, maybe_prefix(prefix, "multi_modal_projector")
            )
        with self._mark_language_model(vllm_config):
            self.language_model = init_vllm_registered_model(
                vllm_config=vllm_config,
                hf_config=config.text_config,
                prefix=maybe_prefix(prefix, "language_model"),
                architectures=["Lfm3ForCausalLM"],
            )
        self.make_empty_intermediate_tensors = (
            self.language_model.make_empty_intermediate_tensors
        )

    def _parse_and_validate_image_input(self, **kwargs: object) -> Lfm3VlImagePixelInputs | None:
        pixel_values = kwargs.pop("pixel_values", None)
        image_grid_thw = kwargs.pop("image_grid_thw", None)
        if pixel_values is None:
            return None
        return Lfm3VlImagePixelInputs(pixel_values=pixel_values, image_grid_thw=image_grid_thw)

    def _parse_and_validate_video_input(self, **kwargs: object) -> Lfm3VlVideoPixelInputs | None:
        pixel_values_videos = kwargs.pop("pixel_values_videos", None)
        video_grid_thw = kwargs.pop("video_grid_thw", None)
        if pixel_values_videos is None:
            return None
        return Lfm3VlVideoPixelInputs(
            pixel_values_videos=pixel_values_videos, video_grid_thw=video_grid_thw
        )

    def _embed_vision(self, pixel_values: torch.Tensor, grid_thw: list[list[int]]) -> torch.Tensor:
        """Text-space embeddings of every merge block of the given images or videos."""
        return self.multi_modal_projector(self.vision_tower(pixel_values, grid_thw))

    def _process_vision_input(
        self, pixel_values: torch.Tensor, grid_thw: torch.Tensor
    ) -> tuple[torch.Tensor, ...]:
        grid = grid_thw.tolist()
        merge = self.vision_tower.merge**2
        return self._embed_vision(pixel_values, grid).split([t * h * w // merge for t, h, w in grid])

    def _process_image_input(self, image_input: Lfm3VlImagePixelInputs) -> tuple[torch.Tensor, ...]:
        return self._process_vision_input(image_input["pixel_values"], image_input["image_grid_thw"])

    def _process_video_input(self, video_input: Lfm3VlVideoPixelInputs) -> tuple[torch.Tensor, ...]:
        return self._process_vision_input(
            video_input["pixel_values_videos"], video_input["video_grid_thw"]
        )

    def _embed_modality(self, key: str, kwargs: dict[str, object]) -> Sequence[torch.Tensor]:
        """Embeddings of the items whose leading kwarg is `key`; other keys give none."""
        if key == "pixel_values":
            return self._process_image_input(self._parse_and_validate_image_input(**kwargs))
        if key == "pixel_values_videos":
            return self._process_video_input(self._parse_and_validate_video_input(**kwargs))
        return ()

    def embed_multimodal(self, **kwargs: object) -> MultiModalEmbeddings:
        embeds: list[torch.Tensor] = []
        # kwargs keep the order of the modalities in the prompt
        for key in list(kwargs):
            embeds += self._embed_modality(key, kwargs)
        return embeds

    def get_language_model(self) -> torch.nn.Module:
        return self.language_model

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor | IntermediateTensors:
        if intermediate_tensors is not None:
            inputs_embeds = None
        return self.language_model(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
        )

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        return self.language_model.compute_logits(hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        skip = ["language_model.lm_head."] if self.config.text_config.tie_word_embeddings else None
        loader = AutoWeightsLoader(self, ignore_unexpected_prefixes=skip)
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)
