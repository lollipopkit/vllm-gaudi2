# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inkling2 PE-Lang-style ViT vision encoder on Gaudi2 HPU."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from vllm.model_executor.model_loader.weight_utils import default_weight_loader

_CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
_CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
_PROCESSOR_CLIP_STD = (0.26862954, 0.2613026, 0.2757771)


def _apply_2d_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    meta: torch.Tensor,
    inv_freq: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply interleaved 2D RoPE to `q, k` of shape `[T, H, D]`.

    `meta` is `[T, 4]` int32 `(y, x, height, width)` with `y = x = -1` on CLS rows.
    First half of pairs rotate by `x + 1`, second half by `y + 1` (CLS -> 0).
    """
    t, h, d = q.shape

    pos_x = (meta[:, 1].float() + 1.0).unsqueeze(-1) * inv_freq.unsqueeze(0)  # [T, quarter]
    pos_y = (meta[:, 0].float() + 1.0).unsqueeze(-1) * inv_freq.unsqueeze(0)  # [T, quarter]
    angles = torch.cat([pos_x, pos_y], dim=-1)  # [T, D // 2]
    cos = torch.cos(angles).unsqueeze(1)  # [T, 1, D // 2]
    sin = torch.sin(angles).unsqueeze(1)

    def _rot(x: torch.Tensor) -> torch.Tensor:
        xf = x.float()
        even = xf[..., 0::2]
        odd = xf[..., 1::2]
        out_even = even * cos - odd * sin
        out_odd = odd * cos + even * sin
        return torch.stack((out_even, out_odd), dim=-1).reshape(t, h, d).to(x.dtype)

    return _rot(q), _rot(k)


class Inkling2HPUVisionAttention(nn.Module):
    def __init__(self, width: int, num_heads: int) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = width // num_heads
        self.qkv_proj = nn.Linear(width, 3 * width, bias=True, dtype=torch.bfloat16)
        self.out_proj = nn.Linear(width, width, bias=True, dtype=torch.bfloat16)
        inv_freq = torch.tensor(
            [10000.0 ** (-dim / self.head_dim) for dim in range(0, self.head_dim, 4)],
            dtype=torch.float32,
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(
        self,
        hidden: torch.Tensor,
        meta: torch.Tensor,
        seqlens: list[int],
    ) -> torch.Tensor:
        qkv = self.qkv_proj(hidden).view(-1, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(1)
        q, k = _apply_2d_rope(q, k, meta, self.inv_freq)

        # Run bidirectional SDPA per image to avoid cross-image attention leakage.
        outs: list[torch.Tensor] = []
        offset = 0
        for length in seqlens:
            end = offset + length
            qi = q[offset:end].transpose(0, 1).unsqueeze(0)  # [1, H, L, D]
            ki = k[offset:end].transpose(0, 1).unsqueeze(0)
            vi = v[offset:end].transpose(0, 1).unsqueeze(0)
            oi = F.scaled_dot_product_attention(qi, ki, vi, is_causal=False)
            outs.append(oi.squeeze(0).transpose(0, 1).reshape(length, -1))
            offset = end
        return self.out_proj(torch.cat(outs, dim=0))


class Inkling2HPUVisionMLP(nn.Module):
    def __init__(self, width: int, intermediate_size: int) -> None:
        super().__init__()
        self.c_fc = nn.Linear(width, intermediate_size, bias=True, dtype=torch.bfloat16)
        self.c_proj = nn.Linear(intermediate_size, width, bias=True, dtype=torch.bfloat16)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.c_proj(F.gelu(self.c_fc(hidden)))


class Inkling2HPUVisionLayerScale(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.gamma = nn.Parameter(torch.empty(width, dtype=torch.bfloat16))


class Inkling2HPUVisionLayer(nn.Module):
    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__()
        width, eps = config["hidden_size"], config["layer_norm_eps"]
        self.ln_1 = nn.LayerNorm(width, eps=eps, dtype=torch.bfloat16)
        self.self_attn = Inkling2HPUVisionAttention(width, config["num_attention_heads"])
        self.ln_2 = nn.LayerNorm(width, eps=eps, dtype=torch.bfloat16)
        self.mlp = Inkling2HPUVisionMLP(width, int(width * config["mlp_ratio"]))
        self.ls_1 = Inkling2HPUVisionLayerScale(width)
        self.ls_2 = Inkling2HPUVisionLayerScale(width)

    def forward(
        self,
        hidden: torch.Tensor,
        meta: torch.Tensor,
        seqlens: list[int],
    ) -> torch.Tensor:
        attn = self.self_attn(self.ln_1(hidden), meta, seqlens)
        hidden = (hidden.float() + attn.float() * self.ls_1.gamma.float()).to(hidden.dtype)
        mlp = self.mlp(self.ln_2(hidden))
        return (hidden.float() + mlp.float() * self.ls_2.gamma.float()).to(hidden.dtype)


class Inkling2HPUVisionEmbeddings(nn.Module):
    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__()
        width, grid = config["hidden_size"], config["posemb_grid_size"]
        self.patch_embedding = nn.Linear(
            3 * config["native_patch_size"] ** 2, width, bias=False, dtype=torch.bfloat16
        )
        self.position_embedding = nn.Parameter(
            torch.empty(grid * grid, width, dtype=torch.bfloat16)
        )
        self.class_embedding = nn.Parameter(torch.empty(width, dtype=torch.bfloat16))
        self.cls_position_embedding = nn.Parameter(
            torch.empty(width, dtype=torch.bfloat16)
        )


class Inkling2HPUVisionBackbone(nn.Module):
    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__()
        self.embeddings = Inkling2HPUVisionEmbeddings(config)
        self.ln_pre = nn.LayerNorm(
            config["hidden_size"], eps=config["layer_norm_eps"], dtype=torch.bfloat16
        )
        self.layers = nn.ModuleList(
            Inkling2HPUVisionLayer(config) for _ in range(config["num_hidden_layers"])
        )


class Inkling2HPUVisionEncoder(nn.Module):
    """Exact HPU implementation of the Inkling2 ViT + 2x2 merge projector."""

    def __init__(self, config: Mapping[str, Any], text_hidden_size: int) -> None:
        super().__init__()
        self.patch_size = config["native_patch_size"]
        self.merge_size = config["adapter_merge_size"]
        self.posemb_grid_size = config["posemb_grid_size"]
        self.text_hidden_size = text_hidden_size
        self.backbone = Inkling2HPUVisionBackbone(config)
        self.proj_in = nn.Linear(
            self.merge_size**2 * config["hidden_size"],
            config["projection_hidden_dim"],
            bias=config["projector_bias"],
            dtype=torch.bfloat16,
        )
        self.proj_out = nn.Linear(
            config["projection_hidden_dim"],
            text_hidden_size,
            bias=config["projector_bias"],
            dtype=torch.bfloat16,
        )

    def _normalize_pixels(self, pixels: torch.Tensor) -> torch.Tensor:
        mean = torch.tensor(_CLIP_MEAN, dtype=torch.float32, device=pixels.device)
        p_std = torch.tensor(_PROCESSOR_CLIP_STD, dtype=torch.float32, device=pixels.device)
        std = torch.tensor(_CLIP_STD, dtype=torch.float32, device=pixels.device)
        val = ((pixels.float() / 255.0 - mean) / p_std).to(torch.bfloat16).float()
        return (2.0 * (val * std + mean) - 1.0).to(torch.bfloat16)

    def forward(self, pixels: torch.Tensor, grid_thw: torch.Tensor) -> torch.Tensor:
        m, p = self.merge_size, self.patch_size
        side = m * p
        assert pixels.shape[1:] == (side, side, 3)
        grid = grid_thw.tolist()
        assert all(t == 1 for t, _, _ in grid)
        assert sum(h * w for _, h, w in grid) == pixels.shape[0]

        norm_pixels = self._normalize_pixels(pixels)  # [N, m*p, m*p, 3]
        embeddings = self.backbone.embeddings
        g = self.posemb_grid_size
        pos_grid = (
            embeddings.position_embedding.float()
            .view(g, g, -1)
            .permute(2, 0, 1)
            .unsqueeze(0)
        )  # [1, D, G, G]
        cls_tok = (
            embeddings.class_embedding.float() + embeddings.cls_position_embedding.float()
        ).to(torch.bfloat16).unsqueeze(0)  # [1, D]

        hidden_parts: list[torch.Tensor] = []
        meta_parts: list[torch.Tensor] = []
        seqlens: list[int] = []
        pool_indices: list[torch.Tensor] = []

        patch_cursor = 0
        tok_cursor = 0
        for _, h, w in grid:
            height, width = h * m, w * m
            img_cells = norm_pixels[patch_cursor : patch_cursor + h * w]  # [h*w, m*p, m*p, 3]
            patch_cursor += h * w
            # Rearrange [h, w, m, p, m, p, 3] -> [h*m*w*m, p*p*3] in row-major native patch order.
            native = (
                img_cells.view(h, w, m, p, m, p, 3)
                .permute(0, 2, 1, 4, 3, 5, 6)
                .reshape(height * width, p * p * 3)
            )
            patch_emb = embeddings.patch_embedding(native)  # [H*W, D]
            pos_resized = (
                F.interpolate(pos_grid, size=(height, width), mode="bilinear", align_corners=False)
                .squeeze(0)
                .permute(1, 2, 0)
                .reshape(height * width, -1)
                .to(torch.bfloat16)
            )
            img_hidden = torch.cat([cls_tok, (patch_emb + pos_resized).to(torch.bfloat16)], dim=0)
            hidden_parts.append(img_hidden)

            # Metadata [H*W + 1, 4]: CLS=(-1,-1,H,W), patches=(y,x,H,W)
            y, x = np.divmod(np.arange(height * width, dtype=np.int32), width)
            m_arr = np.empty((height * width + 1, 4), dtype=np.int32)
            m_arr[0, :2] = -1
            m_arr[1:, 0], m_arr[1:, 1] = y, x
            m_arr[:, 2], m_arr[:, 3] = height, width
            meta_parts.append(torch.from_numpy(m_arr).to(pixels.device))

            # 2x2 merge order [h, w, m, m] -> flat token indices after CLS.
            merged = (y // m) * (w * m * m) + (x // m) * (m * m) + (y % m) * m + (x % m)
            pool = np.empty(height * width, dtype=np.int64)
            pool[merged] = tok_cursor + 1 + np.arange(height * width, dtype=np.int64)
            pool_indices.append(torch.from_numpy(pool).to(pixels.device))

            seqlens.append(height * width + 1)
            tok_cursor += height * width + 1

        hidden = self.backbone.ln_pre(torch.cat(hidden_parts, dim=0))
        meta = torch.cat(meta_parts, dim=0)
        for layer in self.backbone.layers:
            hidden = layer(hidden, meta, seqlens)

        pool_idx = torch.cat(pool_indices, dim=0)
        pooled = hidden.index_select(0, pool_idx).view(-1, (m * m) * hidden.shape[-1])
        return self.proj_out(F.gelu(self.proj_in(pooled)))

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        params = dict(self.named_parameters())
        loaded: set[str] = set()
        qkv_shards: Counter[str] = Counter()
        for name, weight in weights:
            for shard, proj in enumerate(("q_proj", "k_proj", "v_proj")):
                if f".self_attn.{proj}." in name:
                    name = name.replace(proj, "qkv_proj")
                    param = params[name]
                    size = param.shape[0] // 3
                    default_weight_loader(
                        param.data[shard * size : (shard + 1) * size], weight
                    )
                    qkv_shards[name] += 1
                    break
            else:
                if name not in params:
                    raise ValueError(f"Unexpected vision checkpoint tensor: {name}")
                default_weight_loader(params[name], weight)
                loaded.add(name)
        return loaded | {name for name, count in qkv_shards.items() if count == 3}
