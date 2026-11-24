# SPDX-License-Identifier: Apache-2.0
"""HpuLfm3VlVisionModel (packed frames, bucket shapes) against the upstream Lfm3VlVisionModel."""
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm_gaudi.models.lfm3_vl_base import Lfm3VlMultiModalProjector, Lfm3VlVisionModel
from vllm_gaudi.models.lfm3_vl import (HpuLfm3VlVisionModel, _PackedVision, frame_geometry, length_bucket, plan_groups)

# (t, h, w) patch grids: different sizes, two length buckets, a two-frame item
GRIDS = [[1, 4, 6], [1, 8, 8], [1, 2, 2], [2, 6, 10], [1, 16, 12]]


def _config():
    return SimpleNamespace(hidden_size=64,
                           num_attention_heads=4,
                           num_key_value_heads=2,
                           norm_eps=1e-5,
                           conv_kernel_size=3,
                           spatial_merge_size=2,
                           patch_size=16,
                           in_channels=3,
                           temporal_patch_size=2,
                           num_hidden_layers=4,
                           layer_types=["conv", "full_attention", "conv", "full_attention"],
                           intermediate_size=128,
                           rope_parameters={"rope_theta": 10000.0})


def test_frame_geometry_matches_conv2d() -> None:
    """The neighbour indices reproduce a zero-padded 3x3 depthwise conv on the raster grid."""
    h, w, merge, channels = 6, 10, 2, 5
    torch.manual_seed(0)
    x = torch.randn(h * w, channels)  # merge-block order
    weight = torch.randn(channels, 1, 3, 3)
    grid = x.view(h // merge, w // merge, merge, merge, channels).permute(4, 0, 2, 1, 3).reshape(1, channels, h, w)
    ref = torch.nn.functional.conv2d(grid, weight, padding=1, groups=channels)
    ref = ref.view(channels, h // merge, merge, w // merge, merge).permute(1, 3, 2, 4, 0).reshape(h * w, channels)
    pos, nbr = frame_geometry(h, w, merge, 3)
    ext = torch.cat([x, torch.zeros(1, channels)])
    idx = torch.from_numpy(np.where(nbr >= 0, nbr, h * w))
    out = sum(ext[idx[j]] * weight.view(channels, 9)[:, j] for j in range(9))
    torch.testing.assert_close(out, ref, atol=1e-5, rtol=1e-5)
    assert pos.shape == (h * w, 2) and pos[:, 0].max() == h - 1 and pos[:, 1].max() == w - 1


def test_plan_covers_every_patch() -> None:
    frames = [(h, w) for t, h, w in GRIDS for _ in range(t)]
    groups = plan_groups(frames, 2, 3, torch.device("cpu"))
    target = torch.cat([g.target for g in groups]).sort().values
    assert torch.equal(target, torch.arange(sum(h * w for h, w in frames)))
    for g in groups:
        assert g.rows & (g.rows - 1) == 0 and g.length == length_bucket(int(g.valid.max()))


@pytest.mark.skipif(not (hasattr(torch, "hpu") and torch.hpu.is_available()), reason="needs an HPU")
def test_packed_tower_matches_upstream(default_vllm_config: None, dist_init, monkeypatch) -> None:
    import vllm_gaudi.extension.runtime as runtime
    real_get_config = runtime.get_config

    class _Config:  # LFM3's encoder softmax rule without a model config behind the feature detection

        def __getattr__(self, name):
            return True if name == "fp32_softmax" else getattr(real_get_config(), name)

    monkeypatch.setattr(runtime, "get_config", lambda: _Config())
    torch.manual_seed(0)
    cfg = _config()
    with torch.device("hpu"):
        ref = Lfm3VlVisionModel(cfg)
        packed = HpuLfm3VlVisionModel(cfg)
    for name, p in ref.named_parameters():  # vLLM linears start uninitialized
        noise = torch.randn(p.shape)
        p.data.copy_(1.0 + 0.1 * noise if "norm" in name else 0.05 * noise)
    ref = ref.to(torch.bfloat16)
    packed.load_state_dict(ref.state_dict())
    packed = packed.to(torch.bfloat16)
    num_patches = sum(t * h * w for t, h, w in GRIDS)
    pixels = torch.randn(num_patches, 3 * 2 * 16 * 16).to("hpu")
    with torch.no_grad():
        expected = ref(pixels, GRIDS).float().cpu()
        got = packed(pixels, GRIDS).float().cpu()
    assert got.shape == expected.shape
    rel = ((got - expected).norm() / expected.norm()).item()
    assert rel < 2e-2, rel

    # the projector on the packed layout, as _PackedVision._embed_vision runs it, against the projector on
    # the packed tower's per-patch output
    model_cfg = SimpleNamespace(vision_config=cfg,
                                downsample_factor=2,
                                projector_layer_norm_eps=1e-5,
                                projector_hidden_size=96,
                                projector_hidden_act="gelu",
                                projector_bias=True,
                                text_config=SimpleNamespace(hidden_size=48))
    with torch.device("hpu"):
        projector = Lfm3VlMultiModalProjector(model_cfg)
    for p in projector.parameters():
        p.data.copy_(0.05 * torch.randn(p.shape))
    projector = projector.to(torch.bfloat16)
    model = SimpleNamespace(vision_tower=packed, multi_modal_projector=projector)
    with torch.no_grad():
        expected = projector(packed(pixels, GRIDS)).float().cpu()
        got = _PackedVision._embed_vision(model, pixels, GRIDS).float().cpu()
    torch.testing.assert_close(got, expected, atol=1e-2, rtol=1e-2)


def test_warmup_shapes_cover_reachable_passes() -> None:
    """Every pass an encoder batch within the budget can produce is among the warmup shapes."""
    tower = SimpleNamespace(merge=2)
    max_patches, max_frame = 4096, 1024
    shapes = set(HpuLfm3VlVisionModel.warmup_shapes(tower, max_patches, max_frame))
    rng = np.random.default_rng(0)
    for _ in range(200):
        frames, total = [], 0
        while True:
            h, w = 2 * int(rng.integers(1, 17)), 2 * int(rng.integers(1, 17))
            if h * w > max_frame or total + h * w > max_patches:
                break
            frames.append((h, w))
            total += h * w
        for g in plan_groups(frames, 2, 3, torch.device("cpu")):
            assert (g.rows, g.length) in shapes
