# SPDX-License-Identifier: Apache-2.0
"""LFM3-VL / LFM3-Omni vision tower on HPU: the frames of an encoder batch run in fixed bucket shapes.

Upstream runs the tower on the concatenated frames: its attention splits them by cu_seqlens (a host sync, then one
FusedSDPA call per frame) and its depthwise conv loops over the frames, so every new combination of frame sizes is
a new graph for every layer, compiled while serving (MMStar at native resolution: 277 grids, docs/lfm3.md).

Here the frames are grouped by padded length and each group runs as [rows, length] tokens: rows a power of two,
length from LENGTH_BUCKETS, frames at the start of their row in the upstream (merge-block) order. The attention is
one FusedSDPA call with a key-padding mask (its valid_sequence_lengths is causal-only); the depthwise conv gathers
every token's neighbours through
host-built indices, an extra zero row standing for positions outside the frame. All indices come from grid_thw on
the host, so no shape depends on device data. Padding never reaches a frame token: keys past a row's valid length
are masked, conv neighbours are frame tokens or the zero row, and the rest is per token. Padding slots hold zero
pixels and rows without a frame attend over their full (zero) length, so they stay finite.
"""
import bisect
import functools
import os
from dataclasses import dataclass

import numpy as np
import torch

from vllm_gaudi.models.lfm3_omni_base import Lfm3OmniForConditionalGeneration, _AUDIO_STRIDE, _audio_token_bucket
from vllm_gaudi.models.lfm3_vl_base import (Lfm3VlForConditionalGeneration, Lfm3VlVisionLayer, Lfm3VlVisionModel)
from vllm_gaudi.utils import async_h2d_copy

# Patches per frame (a multiple of merge**2): steps of at most 1.5x; longer frames round up to multiples of 2048.
LENGTH_BUCKETS = [64, 128, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096, 6144, 8192]
MAX_ROWS = 64  # frames per pass; a larger group runs in several passes
MAX_MASK_ELEMENTS = 1 << 28  # rows * length**2 of the attention mask (512 MiB in bf16)


def length_bucket(n: int) -> int:
    i = bisect.bisect_left(LENGTH_BUCKETS, n)
    return LENGTH_BUCKETS[i] if i < len(LENGTH_BUCKETS) else -(-n // 2048) * 2048


def rows_bucket(n: int) -> int:
    return 1 << (n - 1).bit_length()


def max_rows(length: int) -> int:
    """Frames per pass at this length: a power of two within MAX_ROWS and the attention-mask budget."""
    fit = max(1, MAX_MASK_ELEMENTS // (length * length))
    return min(MAX_ROWS, 1 << (fit.bit_length() - 1))


@functools.lru_cache(maxsize=1024)
def frame_geometry(h: int, w: int, merge: int, kernel: int) -> tuple[np.ndarray, np.ndarray]:
    """For the h x w patches of a frame in merge-block order: their (row, col) [h * w, 2], and the merge-block
    index of each conv neighbour [kernel**2, h * w] (row-major offsets like the conv weight; -1 outside)."""
    br, bc, ir, ic = np.meshgrid(np.arange(h // merge),
                                 np.arange(w // merge),
                                 np.arange(merge),
                                 np.arange(merge),
                                 indexing="ij")
    rows = (br * merge + ir).reshape(-1)
    cols = (bc * merge + ic).reshape(-1)
    order = np.empty((h, w), dtype=np.int64)  # raster position -> merge-block index
    order[rows, cols] = np.arange(h * w)
    pad = kernel // 2
    neighbours = np.full((kernel * kernel, h * w), -1, dtype=np.int64)
    for j, (dr, dc) in enumerate((dr, dc) for dr in range(-pad, pad + 1) for dc in range(-pad, pad + 1)):
        r, c = rows + dr, cols + dc
        inside = (r >= 0) & (r < h) & (c >= 0) & (c < w)
        neighbours[j, inside] = order[r[inside], c[inside]]
    return np.stack([rows, cols], axis=1), neighbours


@dataclass
class PackedGroup:
    rows: int
    length: int
    source: torch.Tensor  # [rows * length] pixel row per slot; num_patches selects the zero row
    positions: torch.Tensor  # [rows * length, 2] float (row, col) for the axial RoPE
    valid: torch.Tensor  # [rows] int32 valid length per row
    neighbours: torch.Tensor  # [kernel**2, rows * length] slot of each conv neighbour; rows * length = zero row
    keep: torch.Tensor  # slots holding frame tokens
    target: torch.Tensor  # their positions in the concatenated frames
    keep_merged: torch.Tensor  # the same for merge blocks (merge**2 consecutive tokens, block-aligned in a row)
    target_merged: torch.Tensor


def plan_groups(frames: list[tuple[int, int]], merge: int, kernel: int, device: torch.device) -> list[PackedGroup]:
    """Packed groups for frames given as (h, w) patch grids, concatenated in this order. Lengths and row starts
    are multiples of merge**2, so a frame's merge blocks stay whole in the packed layout."""
    offsets = np.concatenate([[0], np.cumsum([h * w for h, w in frames])])
    num_patches = int(offsets[-1])
    by_length: dict[int, list[int]] = {}
    for f, (h, w) in enumerate(frames):
        by_length.setdefault(length_bucket(h * w), []).append(f)
    groups = []
    for length, ids in sorted(by_length.items()):
        step = max_rows(length)
        for start in range(0, len(ids), step):
            chunk = ids[start:start + step]
            rows = rows_bucket(len(chunk))
            slots = rows * length
            source = np.full(slots, num_patches, dtype=np.int64)
            positions = np.zeros((slots, 2), dtype=np.float32)
            neighbours = np.full((kernel * kernel, slots), slots, dtype=np.int64)
            valid = np.full(rows, length, dtype=np.int32)  # rows without a frame: finite zeros
            keep, target = [], []
            for r, f in enumerate(chunk):
                h, w = frames[f]
                n = h * w
                base = r * length
                pos, nbr = frame_geometry(h, w, merge, kernel)
                source[base:base + n] = offsets[f] + np.arange(n)
                positions[base:base + n] = pos
                neighbours[:, base:base + n] = np.where(nbr >= 0, nbr + base, slots)
                valid[r] = n
                keep.append(base + np.arange(n))
                target.append(offsets[f] + np.arange(n))

            def put(a):  # non-blocking like the runner's host-to-device copies (utils.async_h2d_copy)
                return async_h2d_copy(torch.from_numpy(a), device=device)

            keep_slots, target_slots = np.concatenate(keep), np.concatenate(target)
            block = merge * merge
            groups.append(
                PackedGroup(rows, length, put(source), put(positions), put(valid), put(neighbours), put(keep_slots),
                            put(target_slots), put(keep_slots[::block] // block), put(target_slots[::block] // block)))
    return groups


class HpuLfm3VlVisionLayer(Lfm3VlVisionLayer):
    """Lfm3VlVisionLayer on the packed layout (same parameters; the submodules' forwards are not used)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.is_attention_layer:
            from vllm_gaudi.extension.utils import ModuleFusedSDPA
            import vllm_gaudi.extension.kernels as kernels
            self.fsdpa = ModuleFusedSDPA(kernels.fsdpa())

    def forward(self, x, cos, sin, mask, neighbours):
        normed = self.operator_norm(x)
        if self.is_attention_layer:
            x = x + self._attention(normed, cos, sin, mask)
        else:
            x = x + self._conv(normed, neighbours)
        return x + self.feed_forward(self.ffn_norm(x))

    def _attention(self, x, cos, sin, mask):
        from vllm_gaudi.extension.runtime import get_config

        attn = self.self_attn
        n, rows = x.shape[0], mask.shape[0]
        length = n // rows
        heads, kv_heads, dim = attn.num_heads, attn.num_kv_heads, attn.head_dim
        qkv, _ = attn.qkv_proj(x)
        q, k, v = qkv.split([heads * dim, kv_heads * dim, kv_heads * dim], dim=-1)
        q = attn.apply_rotary_emb(attn.q_layernorm(q.view(n, heads, dim)).unsqueeze(0), cos, sin)
        k = attn.apply_rotary_emb(attn.k_layernorm(k.view(n, kv_heads, dim)).unsqueeze(0), cos, sin)
        # [1, n, h, d] -> [rows, h, length, d]; explicit sizes (no -1, see lfm3_vl_base.py)
        q = q.view(rows, length, heads, dim).transpose(1, 2)
        k = k.view(rows, length, kv_heads, dim).transpose(1, 2)
        v = v.view(rows, length, kv_heads, dim).transpose(1, 2)
        # GQA expanded here: FusedSDPA's own GQA path reshapes the mask per head group, which a mask shared by all
        # heads does not have
        if heads != kv_heads:
            k = k.repeat_interleave(heads // kv_heads, dim=1)
            v = v.repeat_interleave(heads // kv_heads, dim=1)
        # the same softmax rule as HpuMMEncoderAttention (ops/hpu_mm_encoder_attention.py)
        out = self.fsdpa(q,
                         k,
                         v,
                         mask,
                         dropout_p=0.0,
                         is_causal=False,
                         scale=dim**-0.5,
                         softmax_mode='fp32' if get_config().fp32_softmax else 'fast',
                         recompute_mode=True,
                         valid_sequence_lengths=None)
        out, _ = attn.out_proj(out.transpose(1, 2).reshape(n, heads * dim))
        return out

    def _conv(self, x, neighbours):
        conv = self.conv
        bcx, _ = conv.in_proj(x)
        b, c, x = bcx.chunk(3, dim=-1)
        channels = x.shape[-1]
        # fp32 like the conv's internal accumulation
        gated = torch.cat([(b * x).float(), x.new_zeros(1, channels, dtype=torch.float32)])
        weight = conv.spatial_conv.weight.float().view(channels, neighbours.shape[0])
        acc = gated.index_select(0, neighbours[0]) * weight[:, 0]
        for j in range(1, neighbours.shape[0]):
            acc = acc + gated.index_select(0, neighbours[j]) * weight[:, j]
        # upstream: conv in the weight dtype, then the activation dtype
        out, _ = conv.out_proj(c * acc.to(conv.spatial_conv.weight.dtype).to(c.dtype))
        return out


class HpuLfm3VlVisionModel(Lfm3VlVisionModel):
    layer_cls = HpuLfm3VlVisionLayer

    def run_groups(self, pixel_values: torch.Tensor, grid_thw: list[list[int]]):
        """(group, packed tower output [rows * length, hidden]) for each packed group of the frames."""
        frames = [(h, w) for t, h, w in grid_thw for _ in range(t)]
        assert pixel_values.shape[0] == sum(h * w for h, w in frames), "pixel rows do not match grid_thw"
        kernel = self.config.conv_kernel_size
        pixels = torch.cat([pixel_values, pixel_values.new_zeros(1, pixel_values.shape[-1])])
        for g in plan_groups(frames, self.merge, kernel, pixel_values.device):
            x, _ = self.patchifier(pixels.index_select(0, g.source).to(self.dtype))
            x = self.embedding_norm(x)
            freqs = (g.positions[..., None] * self.inv_freq).flatten(1)  # h angles, then w (upstream order)
            cos, sin = freqs.cos(), freqs.sin()
            # additive key-padding mask [rows, 1, length, length] (a broadcast view), shared by the attention layers
            keep = torch.arange(g.length, device=x.device) < g.valid.unsqueeze(1)
            mask = torch.zeros(g.rows, 1, 1, g.length, dtype=x.dtype, device=x.device).masked_fill(
                ~keep.view(g.rows, 1, 1, g.length),
                torch.finfo(x.dtype).min).expand(g.rows, 1, g.length, g.length)
            for layer in self.layers:
                x = layer(x, cos, sin, mask, g.neighbours)
            yield g, x

    def forward(self, pixel_values: torch.Tensor, grid_thw: list[list[int]]) -> torch.Tensor:
        """Per-patch features in the concatenated-frame order, as upstream."""
        out = pixel_values.new_empty(pixel_values.shape[0], self.config.hidden_size, dtype=self.dtype)
        for g, x in self.run_groups(pixel_values, grid_thw):
            out.index_copy_(0, g.target, x.index_select(0, g.keep))
        return out

    def warmup_shapes(self, max_patches: int, max_frame_patches: int) -> list[tuple[int, int]]:
        """Every (rows, length) a pass can take when an encoder batch holds at most max_patches patches and a
        frame at most max_frame_patches."""
        step = self.merge**2
        shapes = []
        shortest = step  # the shortest frame of the current length bucket
        while shortest <= max_frame_patches:
            length = length_bucket(shortest)
            most = rows_bucket(max(1, max_patches // shortest))
            rows = 1
            while rows <= min(most, max_rows(length)):
                shapes.append((rows, length))
                rows *= 2
            shortest = length + step
        return shapes


class _PackedVision:
    """Image and video embeddings with the projector on the packed layout too, so that its shapes are the tower's
    buckets, and the model-runner warmup hook for them (HPUModelRunner.warmup_multimodal_graphs)."""

    def _embed_vision(self, pixel_values: torch.Tensor, grid_thw: list[list[int]]) -> torch.Tensor:
        tower, projector = self.vision_tower, self.multi_modal_projector
        block = tower.merge**2
        assert projector.factor == tower.merge, "the projector must merge the tower's merge blocks"
        out = None
        for g, x in tower.run_groups(pixel_values, grid_thw):
            embeds = projector(x)  # [rows * length / block, text hidden]
            if out is None:
                out = embeds.new_empty(pixel_values.shape[0] // block, embeds.shape[-1])
            out.index_copy_(0, g.target_merged, embeds.index_select(0, g.keep_merged))
        return out

    @torch.inference_mode()
    def warmup_vision_tower(self, encoder_budget: int, max_item_tokens: int) -> int:
        """Compile every packed shape an encoder batch within the budget (in text tokens) can produce, frames of
        merge x length/merge patches filling a bucket exactly."""
        tower = self.vision_tower
        block = tower.merge**2
        shapes = tower.warmup_shapes(encoder_budget * block, max_item_tokens * block)
        for rows, length in shapes:
            pixels = torch.zeros(rows * length,
                                 tower.patchifier.input_size,
                                 dtype=tower.dtype,
                                 device=tower.inv_freq.device)
            self._embed_vision(pixels, [[1, tower.merge, length // tower.merge]] * rows)
        return len(shapes)


class HpuLfm3VlForConditionalGeneration(_PackedVision, Lfm3VlForConditionalGeneration):
    vision_tower_cls = HpuLfm3VlVisionModel


class HpuLfm3OmniForConditionalGeneration(_PackedVision, Lfm3OmniForConditionalGeneration):
    vision_tower_cls = HpuLfm3VlVisionModel

    @torch.inference_mode()
    def warmup_audio_encoder(self, max_audio_tokens: int) -> int:
        enc = getattr(self, "audio_encoder", None)
        if enc is None or max_audio_tokens <= 0 or os.environ.get("VLLM_HPU_OMNI_AUDIO_WARMUP", "1") == "0":
            return 0
        mel = int(self.config.audio_encoder_config.num_mel_bins)
        dev = enc.inv_freq.device
        dt = enc.downsample.conv_in.weight.dtype
        cap = _audio_token_bucket(max_audio_tokens)
        buckets = sorted(
            {_audio_token_bucket(t)
             for t in list(range(32, min(cap, 512) + 1, 32)) + list(range(640, cap + 1, 128))})
        for toks in buckets:
            frames = toks * _AUDIO_STRIDE
            self.audio_projector(enc(torch.zeros(mel, frames, dtype=dt, device=dev), frames))
        return len(buckets)
