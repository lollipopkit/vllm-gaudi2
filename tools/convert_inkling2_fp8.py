#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Stream-convert thinkingmachines/Inkling-2-Mini (NVFP4 + MXFP8) to Gaudi2 FP8 E4M3.

Only routed expert weights (`model.llm.layers.N.mlp.experts.w{13,2}_weight`)
are quantized in the upstream checkpoint:
  - NVFP4 layers (0..19): uint8 packed E2M1 nibbles + E4M3FN block scales (group=16)
    + per-expert float32 `weight_amax` and scalar `weight_outer_scale_denominator`.
  - MXFP8 layers (20..23): E4M3FN values + UE8M0 uint8 block scales (group=32).

All other tensors (attention, sconv, loop controllers, vision encoder, embed/unembed)
are copied unchanged. Expert weights are dequantized on CPU in expert chunks and
re-quantized to Gaudi2 native FP8 E4M3 (range [-240, 240]) with per-channel float32
scales, keeping peak host RAM around 4-6 GB for Inkling 2 Mini.
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from collections import defaultdict
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

GAUDI2_FP8_MAX = 240.0

# IEEE E2M1 lookup table (sign in bit 3, exponent bits 2:1, mantissa bit 0).
_E2M1_LUT = torch.tensor(
    [
        0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
        -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
    ],
    dtype=torch.float32,
)

_MAX_SHARD_BYTES = 4_800_000_000  # ~4.8 GB per output shard


def dequantize_nvfp4(
    weight: torch.Tensor,
    block_scale: torch.Tensor,
    amax: torch.Tensor,
    outer_denom: torch.Tensor,
) -> torch.Tensor:
    """Dequantize NVFP4 tensor `[E, N, K//2]` uint8 -> `[E, N, K]` float32."""
    assert weight.dtype == torch.uint8
    assert block_scale.dtype == torch.float8_e4m3fn
    e, n, half_k = weight.shape
    k = half_k * 2
    assert block_scale.shape == (e, n, k // 16), (block_scale.shape, (e, n, k // 16))

    low = (weight & 0x0F).to(torch.int64)
    high = ((weight >> 4) & 0x0F).to(torch.int64)
    unpacked = torch.empty((e, n, k), dtype=torch.float32)
    unpacked[..., 0::2] = _E2M1_LUT[low]
    unpacked[..., 1::2] = _E2M1_LUT[high]

    # Global scale per expert: amax / outer_scale_denominator (ModelOpt NVFP4).
    global_scale = (amax.float() / outer_denom.float().view(1)).view(e, 1, 1, 1)
    scales = block_scale.float().unsqueeze(-1) * global_scale
    return (unpacked.view(e, n, k // 16, 16) * scales).view(e, n, k)


def dequantize_mxfp8(
    weight: torch.Tensor,
    block_scale: torch.Tensor,
) -> torch.Tensor:
    """Dequantize MXFP8 tensor `[E, N, K]` float8_e4m3fn + UE8M0 scales -> float32."""
    assert weight.dtype == torch.float8_e4m3fn
    assert block_scale.dtype == torch.uint8
    e, n, k = weight.shape
    assert block_scale.shape == (e, n, k // 32), (block_scale.shape, (e, n, k // 32))

    exp = block_scale.to(torch.int32)
    # UE8M0: 2^(exp - 127); exp == 0 clamps to 2^-127, matching upstream mxfp8.py.
    scale_bits = (exp << 23).view(torch.float32)
    scales = torch.where(exp == 0, torch.full_like(scale_bits, 2.0**-127), scale_bits)
    return (weight.float().view(e, n, k // 32, 32) * scales.unsqueeze(-1)).view(e, n, k)


def quantize_gaudi2_fp8_per_channel(
    w_fp32: torch.Tensor,
    fp8_max: float = GAUDI2_FP8_MAX,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize `[E, N, K]` float32 -> (`[E, N, K]` float8_e4m3fn, `[E, N]` float32 scale).

    Values are scaled into `[-fp8_max, fp8_max]` (`240.0` for Gaudi2) so raw
    bytes can be loaded onto HPU without overflow or runtime rescaling.
    """
    amax = w_fp32.abs().amax(dim=-1)
    scale = (amax / fp8_max).clamp(min=1e-12)
    q = (w_fp32 / scale.unsqueeze(-1)).clamp(-fp8_max, fp8_max).to(torch.float8_e4m3fn)
    return q, scale.to(torch.float32)


def convert_expert_tensor(
    weight: torch.Tensor,
    block_scale: torch.Tensor,
    amax: torch.Tensor | None,
    outer_denom: torch.Tensor | None,
    *,
    chunk_experts: int = 16,
    fp8_max: float = GAUDI2_FP8_MAX,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Convert one expert weight tensor in chunks to bound peak RAM.

    Returns `(fp8_weight, per_channel_scale, mean_rel_rmse)`.
    """
    is_nvfp4 = amax is not None
    e, n, k_stored = weight.shape
    k = k_stored * 2 if is_nvfp4 else k_stored
    out_w = torch.empty((e, n, k), dtype=torch.float8_e4m3fn)
    out_s = torch.empty((e, n), dtype=torch.float32)
    sq_err = 0.0
    sq_ref = 0.0

    for start in range(0, e, chunk_experts):
        end = min(start + chunk_experts, e)
        if is_nvfp4:
            assert outer_denom is not None
            w_f32 = dequantize_nvfp4(
                weight[start:end],
                block_scale[start:end],
                amax[start:end],
                outer_denom,
            )
        else:
            w_f32 = dequantize_mxfp8(weight[start:end], block_scale[start:end])

        q, s = quantize_gaudi2_fp8_per_channel(w_f32, fp8_max=fp8_max)
        out_w[start:end] = q
        out_s[start:end] = s

        recon = q.float() * s.unsqueeze(-1)
        diff = recon - w_f32
        sq_err += float(diff.square().sum().item())
        sq_ref += float(w_f32.square().sum().item())

    rel_rmse = (sq_err / max(sq_ref, 1e-30)) ** 0.5
    return out_w, out_s, rel_rmse


class _ShardWriter:
    def __init__(self, out_dir: Path, max_shard_bytes: int = _MAX_SHARD_BYTES) -> None:
        self.out_dir = out_dir
        self.max_shard_bytes = max_shard_bytes
        self._buf: dict[str, torch.Tensor] = {}
        self._buf_bytes = 0
        self._shards: list[tuple[str, list[str]]] = []
        self.total_bytes = 0

    def add(self, name: str, tensor: torch.Tensor) -> None:
        nbytes = tensor.numel() * tensor.element_size()
        if self._buf and self._buf_bytes + nbytes > self.max_shard_bytes:
            self._flush()
        self._buf[name] = tensor.contiguous()
        self._buf_bytes += nbytes
        self.total_bytes += nbytes

    def _flush(self) -> None:
        if not self._buf:
            return
        idx = len(self._shards) + 1
        tmp_name = f"shard_{idx:05d}.safetensors"
        save_file(self._buf, str(self.out_dir / tmp_name))
        self._shards.append((tmp_name, list(self._buf.keys())))
        self._buf.clear()
        self._buf_bytes = 0

    def finalize(self) -> dict[str, str]:
        self._flush()
        total = len(self._shards)
        weight_map: dict[str, str] = {}
        for i, (tmp_name, keys) in enumerate(self._shards, 1):
            final_name = f"model-{i:05d}-of-{total:05d}.safetensors"
            (self.out_dir / tmp_name).rename(self.out_dir / final_name)
            for k in keys:
                weight_map[k] = final_name
        index = {
            "metadata": {"total_size": self.total_bytes},
            "weight_map": weight_map,
        }
        (self.out_dir / "model.safetensors.index.json").write_text(
            json.dumps(index, indent=2, sort_keys=True) + "\n"
        )
        return weight_map


def _load_tensor(src_dir: Path, weight_map: dict[str, str], name: str) -> torch.Tensor:
    shard_path = src_dir / weight_map[name]
    with safe_open(str(shard_path), framework="pt", device="cpu") as f:
        return f.get_tensor(name)


def convert_checkpoint(
    src_dir: Path,
    dst_dir: Path,
    *,
    chunk_experts: int = 16,
    fp8_max: float = GAUDI2_FP8_MAX,
) -> None:
    t0 = time.monotonic()
    dst_dir.mkdir(parents=True, exist_ok=True)
    index_path = src_dir / "model.safetensors.index.json"
    if not index_path.exists():
        raise FileNotFoundError(f"Missing {index_path}")
    weight_map: dict[str, str] = json.loads(index_path.read_text())["weight_map"]

    # Identify expert groups by prefix `...mlp.experts.w{13,2}`.
    expert_groups: dict[str, set[str]] = defaultdict(set)
    plain_tensors: list[str] = []
    for name in sorted(weight_map):
        if ".mlp.experts.w13_" in name or ".mlp.experts.w2_" in name:
            for suffix in (
                "_weight_outer_scale_denominator",
                "_weight_scale",
                "_weight_amax",
                "_weight",
            ):
                if name.endswith(suffix):
                    prefix = name[: -len(suffix)]
                    expert_groups[prefix].add(suffix)
                    break
            else:
                plain_tensors.append(name)
        else:
            plain_tensors.append(name)

    writer = _ShardWriter(dst_dir)

    # 1. Copy all non-expert tensors grouped by source shard to minimize file opens.
    by_shard: dict[str, list[str]] = defaultdict(list)
    for name in plain_tensors:
        by_shard[weight_map[name]].append(name)

    for shard_name, keys in sorted(by_shard.items()):
        with safe_open(str(src_dir / shard_name), framework="pt", device="cpu") as f:
            for k in keys:
                writer.add(k, f.get_tensor(k))

    # 2. Convert each expert projection (NVFP4 or MXFP8 -> Gaudi2 FP8 per-channel).
    max_rmse = 0.0
    for idx, (prefix, suffixes) in enumerate(sorted(expert_groups.items()), 1):
        w = _load_tensor(src_dir, weight_map, f"{prefix}_weight")
        s = _load_tensor(src_dir, weight_map, f"{prefix}_weight_scale")
        is_nvfp4 = "_weight_amax" in suffixes
        amax = _load_tensor(src_dir, weight_map, f"{prefix}_weight_amax") if is_nvfp4 else None
        denom = (
            _load_tensor(src_dir, weight_map, f"{prefix}_weight_outer_scale_denominator")
            if is_nvfp4
            else None
        )
        q, scale, rel_rmse = convert_expert_tensor(
            w, s, amax, denom, chunk_experts=chunk_experts, fp8_max=fp8_max
        )
        max_rmse = max(max_rmse, rel_rmse)
        writer.add(f"{prefix}_weight", q)
        writer.add(f"{prefix}_weight_scale", scale)
        fmt = "NVFP4" if is_nvfp4 else "MXFP8"
        print(
            f"[{idx:02d}/{len(expert_groups)}] {prefix} ({fmt} -> FP8@{fp8_max:g}) "
            f"shape={tuple(q.shape)} rel_rmse={rel_rmse:.4%}"
        )

    writer.finalize()

    # 3. Copy metadata/tokenizer files and update config.json quantization_config.
    for item in src_dir.iterdir():
        if item.name.endswith(".safetensors") or item.name in (
            "model.safetensors.index.json",
            "hf_quant_config.json",
            "config.json",
        ):
            continue
        dst = dst_dir / item.name
        if item.is_file():
            shutil.copy2(item, dst)
        elif item.is_dir() and not item.name.startswith("."):
            shutil.copytree(item, dst, dirs_exist_ok=True)

    cfg = json.loads((src_dir / "config.json").read_text())
    cfg["quantization_config"] = {
        "quant_method": "inkling2_fp8",
        "fp8_max": fp8_max,
        "scheme": "per_channel",
        "converted_from": "nvfp4_mxfp8",
    }
    (dst_dir / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")

    elapsed = time.monotonic() - t0
    print(
        f"Done in {elapsed:.1f}s. Output: {dst_dir} "
        f"({writer.total_bytes / 1e9:.2f} GB, max_rel_rmse={max_rmse:.4%})"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("src", type=Path, help="Source Inkling-2-Mini checkpoint directory")
    parser.add_argument("dst", type=Path, help="Destination Gaudi2 FP8 checkpoint directory")
    parser.add_argument("--chunk-experts", type=int, default=16)
    parser.add_argument("--fp8-max", type=float, default=GAUDI2_FP8_MAX)
    args = parser.parse_args()
    convert_checkpoint(
        args.src,
        args.dst,
        chunk_experts=args.chunk_experts,
        fp8_max=args.fp8_max,
    )


if __name__ == "__main__":
    main()
