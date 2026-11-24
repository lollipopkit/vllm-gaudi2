# SPDX-License-Identifier: Apache-2.0
"""Unit tests for tools/convert_inkling2_fp8.py."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from convert_inkling2_fp8 import (
    GAUDI2_FP8_MAX,
    _E2M1_LUT,
    convert_checkpoint,
    convert_expert_tensor,
    dequantize_mxfp8,
    dequantize_nvfp4,
    quantize_gaudi2_fp8_per_channel,
)


def test_dequantize_nvfp4_exact() -> None:
    e, n, k = 2, 4, 32
    # Pack known E2M1 codes: low nibble = 2 (1.0), high nibble = 11 (-1.5)
    packed = torch.full((e, n, k // 2), (11 << 4) | 2, dtype=torch.uint8)
    block_scale = torch.tensor(
        [[[2.0, 0.5]] * n, [[4.0, 1.0]] * n], dtype=torch.float8_e4m3fn
    )
    amax = torch.tensor([6.0, 12.0], dtype=torch.float32)
    denom = torch.tensor([3.0], dtype=torch.float32)

    out = dequantize_nvfp4(packed, block_scale, amax, denom)
    assert out.shape == (e, n, k)
    # Expert 0: global_scale = 2.0, group 0 scale = 2.0 -> total 4.0
    assert torch.allclose(out[0, :, 0:16:2], torch.full((n, 8), 4.0))
    assert torch.allclose(out[0, :, 1:16:2], torch.full((n, 8), -6.0))
    # Expert 1: global_scale = 4.0, group 1 scale = 1.0 -> total 4.0
    assert torch.allclose(out[1, :, 16:32:2], torch.full((n, 8), 4.0))
    assert torch.allclose(out[1, :, 17:32:2], torch.full((n, 8), -6.0))


def test_dequantize_mxfp8_exact() -> None:
    e, n, k = 2, 3, 64
    w = torch.full((e, n, k), 1.5, dtype=torch.float8_e4m3fn)
    # exp = 127 -> 2^0 = 1.0, exp = 129 -> 2^2 = 4.0, exp = 0 -> 2^-127
    exp = torch.tensor(
        [[[127, 129]] * n, [[126, 0]] * n], dtype=torch.uint8
    )
    out = dequantize_mxfp8(w, exp)
    assert torch.allclose(out[0, :, :32], torch.full((n, 32), 1.5))
    assert torch.allclose(out[0, :, 32:], torch.full((n, 32), 6.0))
    assert torch.allclose(out[1, :, :32], torch.full((n, 32), 0.75))
    assert torch.allclose(out[1, :, 32:], torch.full((n, 32), 1.5 * 2.0**-127))


def test_quantize_gaudi2_fp8_bounds_and_accuracy() -> None:
    torch.manual_seed(0)
    w = torch.randn(4, 16, 128) * 3.5
    q, scale = quantize_gaudi2_fp8_per_channel(w, fp8_max=GAUDI2_FP8_MAX)
    assert q.dtype == torch.float8_e4m3fn
    assert scale.dtype == torch.float32
    assert q.float().abs().max().item() <= GAUDI2_FP8_MAX
    recon = q.float() * scale.unsqueeze(-1)
    rel_rmse = (recon - w).norm() / w.norm()
    assert rel_rmse < 0.03, f"Unexpectedly high FP8 rel_rmse: {rel_rmse}"


def test_convert_checkpoint_end_to_end(tmp_path: Path) -> None:
    src = tmp_path / "src"
    dst = tmp_path / "dst"
    src.mkdir()

    e, n, k = 4, 8, 32
    tensors = {
        "model.llm.embed.weight": torch.randn(16, 32, dtype=torch.bfloat16),
        "model.llm.layers.0.mlp.experts.w13_weight": torch.randint(
            0, 256, (e, n, k // 2), dtype=torch.uint8
        ),
        "model.llm.layers.0.mlp.experts.w13_weight_scale": torch.ones(
            (e, n, k // 16), dtype=torch.float8_e4m3fn
        ),
        "model.llm.layers.0.mlp.experts.w13_weight_amax": torch.full(
            (e,), 2.0, dtype=torch.float32
        ),
        "model.llm.layers.0.mlp.experts.w13_weight_outer_scale_denominator": torch.tensor(
            [1.0], dtype=torch.float32
        ),
        "model.llm.layers.20.mlp.experts.w2_weight": torch.randn(
            (e, n, k), dtype=torch.float32
        ).clamp(-448, 448).to(torch.float8_e4m3fn),
        "model.llm.layers.20.mlp.experts.w2_weight_scale": torch.full(
            (e, n, k // 32), 127, dtype=torch.uint8
        ),
    }
    shard = "model-00001-of-00001.safetensors"
    save_file(tensors, str(src / shard))
    (src / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {k: shard for k in tensors}})
    )
    (src / "config.json").write_text(
        json.dumps({"architectures": ["Inkling2LoopForConditionalGeneration"]})
    )
    (src / "hf_quant_config.json").write_text("{}")

    convert_checkpoint(src, dst, chunk_experts=2)

    assert not (dst / "hf_quant_config.json").exists()
    cfg = json.loads((dst / "config.json").read_text())
    assert cfg["quantization_config"]["quant_method"] == "inkling2_fp8"
    assert cfg["quantization_config"]["fp8_max"] == GAUDI2_FP8_MAX

    idx = json.loads((dst / "model.safetensors.index.json").read_text())
    wm = idx["weight_map"]
    assert "model.llm.layers.0.mlp.experts.w13_weight_amax" not in wm
    assert "model.llm.layers.0.mlp.experts.w13_weight_outer_scale_denominator" not in wm
    assert "model.llm.layers.0.mlp.experts.w13_weight" in wm
    assert "model.llm.layers.0.mlp.experts.w13_weight_scale" in wm
    assert "model.llm.layers.20.mlp.experts.w2_weight" in wm
    assert "model.llm.layers.20.mlp.experts.w2_weight_scale" in wm

    with safe_open(str(dst / wm["model.llm.layers.0.mlp.experts.w13_weight"]), framework="pt") as f:
        w0 = f.get_tensor("model.llm.layers.0.mlp.experts.w13_weight")
        s0 = f.get_tensor("model.llm.layers.0.mlp.experts.w13_weight_scale")
        assert w0.shape == (e, n, k)
        assert w0.dtype == torch.float8_e4m3fn
        assert s0.shape == (e, n)
        assert s0.dtype == torch.float32
        assert w0.float().abs().max().item() <= GAUDI2_FP8_MAX


if __name__ == "__main__":
    import tempfile

    test_dequantize_nvfp4_exact()
    test_dequantize_mxfp8_exact()
    test_quantize_gaudi2_fp8_bounds_and_accuracy()
    with tempfile.TemporaryDirectory() as d:
        test_convert_checkpoint_end_to_end(Path(d))
    print("All tests passed.")
