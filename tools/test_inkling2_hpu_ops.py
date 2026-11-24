# SPDX-License-Identifier: Apache-2.0
"""Unit tests for Inkling2 HPU math primitives, short conv, attention, MoE, and vision."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import torch

_REPO_ROOT = Path(__file__).resolve().parent.parent
_HPU_DIR = (
    _REPO_ROOT / "vllm_gaudi" / "models" / "inkling2" / "hpu"
    if (_REPO_ROOT / "vllm_gaudi" / "models" / "inkling2" / "hpu").is_dir()
    else _REPO_ROOT / "vllm-gaudi" / "vllm_gaudi" / "models" / "inkling2" / "hpu"
)


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


import logging

import pytest

# Only create minimal `vllm.logger` stub at import time when real `vllm` is absent
# (e.g. CPU-only laptops). When running alongside real `vllm` in a shared pytest
# session, isolate `vllm*` inside a module-scoped fixture during test execution so
# collection and sibling test modules never see stubbed `vllm` modules.
for _pkg in ("inkling2_pkg", "inkling2_pkg.hpu"):
    if _pkg not in sys.modules:
        sys.modules[_pkg] = types.ModuleType(_pkg)

_temp_vllm_stubs: list[str] = []
try:
    import vllm.logger  # noqa: F401
except ImportError:
    for _pkg in ("vllm", "vllm.logger"):
        if _pkg not in sys.modules:
            sys.modules[_pkg] = types.ModuleType(_pkg)
            _temp_vllm_stubs.append(_pkg)
    sys.modules["vllm.logger"].init_logger = logging.getLogger

try:
    tpc_ops = _load_module("inkling2_pkg.hpu.tpc_ops", _HPU_DIR / "tpc_ops.py")
    sys.modules["inkling2_pkg.hpu"].tpc_ops = tpc_ops
    ops = _load_module("inkling2_pkg.hpu.ops", _HPU_DIR / "ops.py")
    sys.modules["inkling2_pkg.hpu"].ops = ops
finally:
    for _pkg in _temp_vllm_stubs:
        sys.modules.pop(_pkg, None)


@pytest.fixture(scope="module", autouse=True)
def _isolate_vllm_sys_modules():
    saved = {
        k: sys.modules.pop(k)
        for k in list(sys.modules)
        if k == "vllm" or k.startswith("vllm.")
    }
    for _pkg in ("vllm", "vllm.logger"):
        sys.modules[_pkg] = types.ModuleType(_pkg)
    sys.modules["vllm.logger"].init_logger = logging.getLogger
    try:
        yield
    finally:
        for k in [k for k in list(sys.modules) if k == "vllm" or k.startswith("vllm.") or k.startswith("inkling2_")]:
            sys.modules.pop(k, None)
        sys.modules.update(saved)


def test_softcapped_situ_matches_reference() -> None:
    torch.manual_seed(0)
    gate_up = torch.randn(4, 16, dtype=torch.bfloat16)
    gamma = torch.rand(4, 1, dtype=torch.float32)
    out = ops.softcapped_situ(gate_up, gamma, gate_cap=4.0, up_cap=25.0)

    g = gate_up[:, 0::2].float()
    u = gate_up[:, 1::2].float()
    ref = ((4.0 * torch.tanh(g / 4.0) * torch.sigmoid(g)) * (25.0 * torch.tanh(u / 25.0)) * gamma).to(
        torch.bfloat16
    )
    assert torch.equal(out, ref)


def test_gate_sink_matches_reference() -> None:
    torch.manual_seed(1)
    t, h, d = 5, 4, 8
    attn_out = torch.randn(t, h, d, dtype=torch.float32)
    gate = torch.randn(t, h * d, dtype=torch.bfloat16)
    lse = torch.randn(t, h, dtype=torch.float32)
    sinks = torch.randn(t, h, dtype=torch.bfloat16)
    out = ops.gate_sink(attn_out, gate, lse, sinks)

    ref = (
        attn_out
        * torch.sigmoid(gate.view(t, h, d).float())
        * torch.sigmoid(lse - sinks.float()).unsqueeze(-1)
    ).reshape(t, h * d).to(torch.bfloat16)
    assert torch.allclose(out.float(), ref.float(), atol=1e-2)


def test_attn_res_and_blend() -> None:
    torch.manual_seed(2)
    n_stack, t, d = 2, 3, 16
    stack = torch.randn(n_stack, t, d, dtype=torch.bfloat16)
    last = torch.randn(t, d, dtype=torch.bfloat16)
    addend = torch.randn(t, d, dtype=torch.bfloat16)
    queries = torch.randn(3, d, dtype=torch.float32) * (d**-0.5)
    norm_w = torch.ones(3, d, dtype=torch.bfloat16)

    mixed_fp32 = ops.attn_res(stack, last, addend, queries, None, 0.0, True)
    assert mixed_fp32.shape == (3, t, d) and mixed_fp32.dtype == torch.float32

    mixed_bf16 = ops.attn_res(stack, last, addend, queries, norm_w, 1e-6, True)
    assert mixed_bf16.shape == (3, t, d) and mixed_bf16.dtype == torch.bfloat16

    dest = torch.randn(3, t, d, dtype=torch.float32)
    msg = torch.randn(3, t, d, dtype=torch.float32)
    pair = ops.pair_rms_norm(dest, msg)
    assert pair.shape == (3, t, 2 * d) and pair.dtype == torch.bfloat16

    gf = torch.randn(3, t, 2 * d, dtype=torch.float32)
    blended = ops.blend_rms_norm(dest, msg, gf, norm_w, 1e-6)
    assert blended.shape == (3, t, d) and blended.dtype == torch.bfloat16


def test_sconv_prefill_equals_step_by_step_decode() -> None:
    # Stub minimal vllm imports so sconv helpers can be tested standalone.
    for mod_name in (
        "vllm",
        "vllm.config",
        "vllm.distributed",
        "vllm.forward_context",
        "vllm.model_executor",
        "vllm.model_executor.layers",
        "vllm.model_executor.layers.attention_layer_base",
        "vllm.model_executor.utils",
        "vllm.v1",
        "vllm.v1.attention",
        "vllm.v1.attention.backend",
        "vllm.v1.attention.backends",
        "vllm.v1.attention.backends.registry",
        "vllm.v1.attention.selector",
        "vllm.v1.kv_cache_interface",
    ):
        if mod_name not in sys.modules:
            sys.modules[mod_name] = types.ModuleType(mod_name)
    sys.modules["vllm.config"].VllmConfig = object
    sys.modules["vllm.config"].get_current_vllm_config = lambda: None
    sys.modules["vllm.distributed"].get_tensor_model_parallel_rank = lambda: 0
    sys.modules["vllm.distributed"].get_tensor_model_parallel_world_size = lambda: 1
    sys.modules["vllm.forward_context"].get_forward_context = lambda: None
    sys.modules["vllm.model_executor.layers.attention_layer_base"].AttentionLayerBase = object
    sys.modules["vllm.model_executor.utils"].set_weight_attrs = lambda *a, **kw: None
    sys.modules["vllm.v1.attention.backend"].AttentionBackend = object
    sys.modules["vllm.v1.attention.backends.registry"].MambaAttentionBackendEnum = types.SimpleNamespace(
        SHORT_CONV="short_conv"
    )
    sys.modules["vllm.v1.attention.selector"].get_mamba_attn_backend = lambda *a, **kw: object
    sys.modules["vllm.v1.kv_cache_interface"].KVCacheSpec = object
    sys.modules["vllm.v1.kv_cache_interface"].MambaSpec = object

    sconv = _load_module("inkling2_pkg.hpu.sconv", _HPU_DIR / "sconv.py")
    sys.modules["inkling2_hpu_sconv"] = sconv

    torch.manual_seed(3)
    seq_len, c, w, num_heads = 9, 16, 4, 2
    x = torch.randn(seq_len, c, dtype=torch.bfloat16)

    # 1. Static conv: prefill vs decode step-by-step
    sw_static = torch.randn(c, w, dtype=torch.bfloat16)
    y_prefill, state_prefill = sconv._causal_conv_prefill_single(
        x, None, sw_static, None, num_heads, w
    )
    state = torch.zeros(1, w - 1, c, dtype=torch.bfloat16)
    ys = []
    for t in range(seq_len):
        yt, state = sconv._causal_conv_decode_batch(
            x[t : t + 1], state, sw_static, None, num_heads, w
        )
        ys.append(yt)
    y_decode = torch.cat(ys, dim=0)
    assert torch.allclose(y_prefill.float(), y_decode.float(), atol=1e-2)
    assert torch.equal(state_prefill, state[0])

    # 2. Dynamic conv: prefill vs decode step-by-step
    sw_dyn = torch.randn(w, c, dtype=torch.bfloat16)
    dyn = torch.randn(seq_len, num_heads * w, dtype=torch.bfloat16)
    y_prefill_d, state_prefill_d = sconv._causal_conv_prefill_single(
        x, None, sw_dyn, dyn, num_heads, w
    )
    state_d = torch.zeros(1, w - 1, c, dtype=torch.bfloat16)
    ys_d = []
    for t in range(seq_len):
        yt, state_d = sconv._causal_conv_decode_batch(
            x[t : t + 1], state_d, sw_dyn, dyn[t : t + 1], num_heads, w
        )
        ys_d.append(yt)
    y_decode_d = torch.cat(ys_d, dim=0)
    assert torch.allclose(y_prefill_d.float(), y_decode_d.float(), atol=1e-2)
    assert torch.equal(state_prefill_d, state_d[0])


def test_vision_encoder_forward_shapes() -> None:
    for mod_name in ("vllm.model_executor.model_loader", "vllm.model_executor.model_loader.weight_utils"):
        if mod_name not in sys.modules:
            sys.modules[mod_name] = types.ModuleType(mod_name)
    sys.modules["vllm.model_executor.model_loader.weight_utils"].default_weight_loader = (
        lambda p, w: p.data.copy_(w)
    )
    vision = _load_module("inkling2_hpu_vision", _HPU_DIR / "vision.py")

    cfg = {
        "native_patch_size": 4,
        "adapter_merge_size": 2,
        "posemb_grid_size": 4,
        "hidden_size": 32,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "mlp_ratio": 2.0,
        "layer_norm_eps": 1e-5,
        "projection_hidden_dim": 64,
        "projector_bias": False,
    }
    enc = vision.Inkling2HPUVisionEncoder(cfg, text_hidden_size=48)
    for p in enc.parameters():
        torch.nn.init.normal_(p, std=0.02)

    # Two images: 2x3 cells and 1x2 cells => 8 cells total, each 8x8x3
    grid_thw = torch.tensor([[1, 2, 3], [1, 1, 2]], dtype=torch.int64)
    pixels = torch.randint(0, 255, (8, 8, 8, 3), dtype=torch.uint8).to(torch.bfloat16)
    out = enc(pixels, grid_thw)
    assert out.shape == (8, 48)
    assert out.dtype == torch.bfloat16
    assert torch.isfinite(out).all()


def test_fp8_bmm_matches_reference() -> None:
    torch.manual_seed(4)
    r, m, k, n = 4, 2, 16, 12
    x = torch.randn(r, m, k, dtype=torch.bfloat16)
    w_raw = torch.randn(r, n, k, dtype=torch.float32)
    scale = (w_raw.abs().amax(dim=-1) / ops.GAUDI2_FP8_MAX).clamp(min=1e-12)
    w_fp8 = (w_raw / scale.unsqueeze(-1)).clamp(-ops.GAUDI2_FP8_MAX, ops.GAUDI2_FP8_MAX).to(
        torch.float8_e4m3fn
    )
    out = ops.fp8_bmm(x, w_fp8, scale)
    ref = torch.bmm(x.float(), (w_fp8.float() * scale.unsqueeze(-1)).transpose(1, 2)).to(
        torch.bfloat16
    )
    assert torch.allclose(out.float(), ref.float(), atol=1e-2)


def test_sconv_batched_prefill_with_padding() -> None:
    sconv = sys.modules.get("inkling2_pkg.hpu.sconv") or _load_module(
        "inkling2_pkg.hpu.sconv", _HPU_DIR / "sconv.py"
    )
    sys.modules["inkling2_hpu_sconv"] = sconv
    torch.manual_seed(5)
    b, s, c, w, h = 2, 10, 16, 4, 2
    seq_lens = torch.tensor([6, 9], dtype=torch.long)
    x = torch.randn(b, s, c, dtype=torch.bfloat16)
    sw = torch.randn(c, w, dtype=torch.bfloat16)
    y_batched, st_batched = sconv._causal_conv_prefill_batch(
        x, None, sw, None, h, w, seq_lens=seq_lens
    )
    for idx in range(b):
        L = int(seq_lens[idx].item())
        y_ref, st_ref = sconv._causal_conv_prefill_single(x[idx, :L], None, sw, None, h, w)
        assert torch.allclose(y_batched[idx, :L].float(), y_ref.float(), atol=1e-2)
        assert torch.equal(st_batched[idx], st_ref)


def test_decode_attention_matches_sdpa_reference() -> None:
    # Stub minimal modules needed to import attention.py standalone.
    for mod_name in (
        "vllm.model_executor.layers.linear",
        "vllm.utils",
        "vllm.utils.torch_utils",
        "vllm.v1.attention.selector",
        "vllm.v1.kv_cache_interface",
        "inkling2_pkg",
        "inkling2_pkg.configs",
        "inkling2_pkg.hpu",
    ):
        if mod_name not in sys.modules:
            sys.modules[mod_name] = types.ModuleType(mod_name)
    sys.modules["vllm.model_executor.layers.linear"].MergedColumnParallelLinear = object
    sys.modules["vllm.model_executor.layers.linear"].RowParallelLinear = object
    sys.modules["vllm.utils.torch_utils"].kv_cache_dtype_str_to_dtype = lambda *a, **kw: torch.bfloat16
    sys.modules["vllm.v1.attention.selector"].get_attn_backend = lambda *a, **kw: object
    sys.modules["vllm.v1.kv_cache_interface"].FullAttentionSpec = object
    sys.modules["inkling2_pkg.configs"].Inkling2ModelConfig = object
    sys.modules["inkling2_pkg.hpu"].ops = ops
    sys.modules["inkling2_pkg.hpu"].sconv = sys.modules["inkling2_hpu_sconv"]
    sys.modules["inkling2_pkg.hpu.ops"] = ops
    sys.modules["inkling2_pkg.hpu.sconv"] = sys.modules["inkling2_hpu_sconv"]

    spec = importlib.util.spec_from_file_location(
        "inkling2_pkg.hpu.attention", _HPU_DIR / "attention.py"
    )
    assert spec is not None and spec.loader is not None
    attn_mod = importlib.util.module_from_spec(spec)
    sys.modules["inkling2_pkg.hpu.attention"] = attn_mod
    spec.loader.exec_module(attn_mod)

    torch.manual_seed(6)
    b, h, h_kv, d, d_rel, bs = 2, 4, 2, 8, 4, 8
    rel_extent, sw = 32, 8
    # Request 0 has 11 tokens (blocks 2, 5 -> usages 8, 3); Request 1 has 5 tokens (block 1 -> usage 5)
    num_phys = 8
    k_cache = torch.randn(num_phys * bs, h_kv, d, dtype=torch.bfloat16)
    v_cache = torch.randn(num_phys * bs, h_kv, d, dtype=torch.bfloat16)
    block_list = torch.tensor([2, 5, 1, -1], dtype=torch.long)
    block_groups = torch.tensor([0, 0, 1, -1], dtype=torch.long)
    block_usage = torch.tensor([8, 3, 5, 0], dtype=torch.long)
    positions = torch.tensor([10, 4], dtype=torch.long)
    q = torch.randn(b, h, d, dtype=torch.bfloat16)
    r = torch.randn(b, h, d_rel, dtype=torch.bfloat16)

    for is_local, extent, win in ((False, rel_extent, None), (True, sw, sw)):
        proj = torch.randn(d_rel, extent, dtype=torch.bfloat16)
        dummy = types.SimpleNamespace(
            is_local=is_local,
            num_heads=h,
            num_kv_heads=h_kv,
            head_dim=d,
            d_rel=d_rel,
            scaling=d**-0.5,
            sliding_window=win,
            rel_extent=extent,
            rel_logits_proj=proj,
            rel_logits_proj_t=proj.t().contiguous(),
        )
        md = types.SimpleNamespace(
            is_prompt=False,
            block_size=bs,
            block_list=block_list,
            block_groups=block_groups,
            block_usage=block_usage,
        )
        dctx = attn_mod.build_decode_attn_context(
            md, positions, b, sliding_window=sw, rel_extent=rel_extent
        )
        out_vec, lse_vec = attn_mod.Inkling2HPUAttention._decode_attention(
            dummy, q, r, positions, (k_cache, v_cache), md, decode_ctx=dctx
        )

        # Reference per-request via _sdpa_with_rel_bias_and_lse
        rel_full = torch.matmul(r.float(), proj.float()).to(q.dtype)
        k4 = k_cache.view(num_phys, bs, h_kv, d)
        v4 = v_cache.view(num_phys, bs, h_kv, d)
        req_blocks = ([2, 5], [1])
        req_lens = (11, 5)
        for req_i, (blks, seq_len) in enumerate(zip(req_blocks, req_lens)):
            k_req = k4[blks].reshape(-1, h_kv, d)[:seq_len]
            v_req = v4[blks].reshape(-1, h_kv, d)[:seq_len]
            k_pos = torch.arange(seq_len, dtype=torch.long)
            o_ref, lse_ref = attn_mod._sdpa_with_rel_bias_and_lse(
                q[req_i : req_i + 1],
                k_req,
                v_req,
                rel_full[req_i : req_i + 1],
                positions[req_i : req_i + 1],
                k_pos,
                scale=d**-0.5,
                sliding_window=win,
            )
            assert torch.allclose(out_vec[req_i].float(), o_ref[0].float(), atol=2e-2)
            assert torch.allclose(lse_vec[req_i].float(), lse_ref[0].float(), atol=2e-2)


def test_moe_dense_chunk_matches_gather_chunk() -> None:
    sys.modules["vllm.distributed"].tensor_model_parallel_all_reduce = lambda t: t
    if "vllm.logger" not in sys.modules:
        import logging

        logger_mod = types.ModuleType("vllm.logger")
        logger_mod.init_logger = logging.getLogger
        sys.modules["vllm.logger"] = logger_mod
    _load_module("inkling2_pkg.hpu.tpc_ops", _HPU_DIR / "tpc_ops.py")
    spec = importlib.util.spec_from_file_location("inkling2_pkg.hpu.moe", _HPU_DIR / "moe.py")
    assert spec is not None and spec.loader is not None
    moe_mod = importlib.util.module_from_spec(spec)
    sys.modules["inkling2_pkg.hpu.moe"] = moe_mod
    spec.loader.exec_module(moe_mod)

    torch.manual_seed(7)
    cfg = types.SimpleNamespace(
        n_routed_experts=16,
        n_shared_experts=0,
        hidden_size=16,
        intermediate_size=8,
        num_experts_per_tok=4,
        route_scale=2.5,
        use_gate_bias=True,
        mlp_gate_softcap=4.0,
        mlp_up_softcap=25.0,
    )
    experts = moe_mod.Inkling2HPUFP8Experts(cfg)
    for w, s in (
        (experts.w13_weight, experts.w13_weight_scale),
        (experts.w2_weight, experts.w2_weight_scale),
    ):
        raw = torch.randn(w.shape, dtype=torch.float32) * 0.1
        sc = (raw.abs().amax(dim=-1) / ops.GAUDI2_FP8_MAX).clamp(min=1e-12)
        w.data.copy_((raw / sc.unsqueeze(-1)).clamp(-ops.GAUDI2_FP8_MAX, ops.GAUDI2_FP8_MAX).to(torch.float8_e4m3fn))
        s.data.copy_(sc)

    m = 10
    x = torch.randn(m, cfg.hidden_size, dtype=torch.bfloat16)
    ids = torch.argsort(torch.randn(m, cfg.n_routed_experts), dim=-1)[:, : cfg.num_experts_per_tok].long()
    weights = torch.softmax(torch.randn(m, cfg.num_experts_per_tok), dim=-1)

    y_gather = experts._run_gather_chunk(x, ids, weights)
    y_dense = experts._run_dense_chunk(x, ids, weights)
    assert torch.allclose(y_gather.float(), y_dense.float(), atol=2e-2)

    # EP=2 rank slicing (scatter_add_ path when multiple non-local ids clamp to 0)
    experts_r0 = moe_mod.Inkling2HPUFP8Experts(cfg)
    experts_r1 = moe_mod.Inkling2HPUFP8Experts(cfg)
    half = cfg.n_routed_experts // 2
    for rank, ex in ((0, experts_r0), (1, experts_r1)):
        ex.ep_size = 2
        ex.ep_rank = rank
        ex.num_experts = half
        ex.expert_start = rank * half
        ex.expert_end = (rank + 1) * half
        ex.w13_weight = torch.nn.Parameter(experts.w13_weight[ex.expert_start : ex.expert_end].clone())
        ex.w13_weight_scale = torch.nn.Parameter(
            experts.w13_weight_scale[ex.expert_start : ex.expert_end].clone()
        )
        ex.w2_weight = torch.nn.Parameter(experts.w2_weight[ex.expert_start : ex.expert_end].clone())
        ex.w2_weight_scale = torch.nn.Parameter(
            experts.w2_weight_scale[ex.expert_start : ex.expert_end].clone()
        )
    y_ep_dense = experts_r0._run_dense_chunk(x, ids, weights).float() + experts_r1._run_dense_chunk(
        x, ids, weights
    ).float()
    y_ep_gather = experts_r0._run_gather_chunk(x, ids, weights).float() + experts_r1._run_gather_chunk(
        x, ids, weights
    ).float()
    assert torch.allclose(y_ep_dense, y_gather.float(), atol=2e-2)
    assert torch.allclose(y_ep_gather, y_gather.float(), atol=2e-2)


def test_decode_attention_out_of_order_blocks_bf16_usage_and_rel_fallback() -> None:
    attn_mod = sys.modules["inkling2_pkg.hpu.attention"]
    torch.manual_seed(8)
    b, h, h_kv, d, d_rel, bs = 2, 4, 2, 8, 4, 8
    rel_extent, sw = 32, 8
    num_phys = 8
    k_cache = torch.randn(num_phys * bs, h_kv, d, dtype=torch.bfloat16)
    v_cache = torch.randn(num_phys * bs, h_kv, d, dtype=torch.bfloat16)
    # Request 0 has 2 blocks (logical 0 -> phys 5, logical 1 -> phys 2), intentionally out of order
    # in block_list (phys 2 appears BEFORE phys 5 in block_list, mimicking sorted/contiguous PA).
    block_list = torch.tensor([2, 1, 5, -1], dtype=torch.long)
    block_groups = torch.tensor([0, 1, 0, -1], dtype=torch.long)
    block_logical_idx = torch.tensor([1, 0, 0, 0], dtype=torch.long)
    # Match real HPUModelRunner dtype: block_usage is bfloat16 on HPU.
    block_usage = torch.tensor([3, 5, 8, 0], dtype=torch.bfloat16)
    positions = torch.tensor([10, 4], dtype=torch.long)
    q = torch.randn(b, h, d, dtype=torch.bfloat16)
    r = torch.randn(b, h, d_rel, dtype=torch.bfloat16)

    for is_local, extent, win in ((False, rel_extent, None), (True, sw, sw)):
        proj = torch.randn(d_rel, extent, dtype=torch.bfloat16)
        dummy = types.SimpleNamespace(
            is_local=is_local,
            num_heads=h,
            num_kv_heads=h_kv,
            head_dim=d,
            d_rel=d_rel,
            scaling=d**-0.5,
            sliding_window=win,
            rel_extent=extent,
            rel_logits_proj=proj,
            rel_logits_proj_t=torch.zeros_like(proj.t().contiguous()),
            _rel_proj_t_ready=False,  # exercises uninitialized dummy-load fallback
        )
        md = types.SimpleNamespace(
            is_prompt=False,
            block_size=bs,
            block_list=block_list,
            block_groups=block_groups,
            block_usage=block_usage,
            block_logical_idx=block_logical_idx,
        )
        dctx = attn_mod.build_decode_attn_context(
            md, positions, b, sliding_window=sw, rel_extent=rel_extent
        )
        out_vec, lse_vec = attn_mod.Inkling2HPUAttention._decode_attention(
            dummy, q, r, positions, (k_cache, v_cache), md, decode_ctx=dctx
        )

        rel_full = torch.matmul(r.float(), proj.float()).to(q.dtype)
        k4 = k_cache.view(num_phys, bs, h_kv, d)
        v4 = v_cache.view(num_phys, bs, h_kv, d)
        # Logical order for req 0 is [phys 5 (0..7), phys 2 (8..10)]; req 1 is [phys 1 (0..4)].
        for req_i, (blks, seq_len) in enumerate((([5, 2], 11), ([1], 5))):
            k_req = k4[blks].reshape(-1, h_kv, d)[:seq_len]
            v_req = v4[blks].reshape(-1, h_kv, d)[:seq_len]
            k_pos = torch.arange(seq_len, dtype=torch.long)
            o_ref, lse_ref = attn_mod._sdpa_with_rel_bias_and_lse(
                q[req_i : req_i + 1],
                k_req,
                v_req,
                rel_full[req_i : req_i + 1],
                positions[req_i : req_i + 1],
                k_pos,
                scale=d**-0.5,
                sliding_window=win,
            )
            assert torch.allclose(out_vec[req_i].float(), o_ref[0].float(), atol=2e-2)
            assert torch.allclose(lse_vec[req_i].float(), lse_ref[0].float(), atol=2e-2)


def test_sconv_load_store_streams_nan_guard_and_empty_buf() -> None:
    sconv = sys.modules.get("inkling2_hpu_sconv") or _load_module(
        "inkling2_hpu_sconv", _HPU_DIR / "sconv.py"
    )
    cs = sconv.Inkling2HPUConvState.__new__(sconv.Inkling2HPUConvState)
    torch.nn.Module.__init__(cs)
    cs.prefix = "layer0.conv"
    cs.stream_sizes = (4, 4, 8, 8)
    cs.total_dim = 24
    cs.kernel_size = 4
    cs.cache_group_idx = None

    # Empty state buffer (profile run) -> no-op
    cs.kv_cache = [torch.empty(0, dtype=torch.bfloat16)]
    prev, safe_store, is_prompt, _, _ = cs.load_streams(types.SimpleNamespace(is_prompt=True))
    assert prev is None and safe_store is None and is_prompt is True

    # Non-empty buffer filled with NaNs: has_initial_states_p=[False, True] must zero NaN slot
    buf = torch.full((4, 3, 24), float("nan"), dtype=torch.bfloat16)
    buf[2].fill_(1.5)
    cs.kv_cache = [buf]
    md = types.SimpleNamespace(
        is_prompt=True,
        load_indices_tensor=torch.tensor([1, 2], dtype=torch.long),
        store_indices_tensor=torch.tensor([0, 3], dtype=torch.long),
        has_initial_states_p=torch.tensor([False, True]),
        seq_lens_tensor=torch.tensor([5, 7], dtype=torch.long),
        padding_mask_flat=None,
    )
    prev, safe_store, is_prompt, _, seq_lens = cs.load_streams(md, batch_size=2)
    assert is_prompt is True and prev is not None and safe_store is not None
    assert seq_lens is not None and seq_lens.tolist() == [5, 7]
    for part in prev:
        assert torch.isfinite(part).all()
        assert (part[0] == 0).all()
        assert (part[1] == 1.5).all()

    parts = [torch.full((2, 3, s), float(i + 1), dtype=torch.bfloat16) for i, s in enumerate(cs.stream_sizes)]
    cs.store_streams(safe_store, parts)
    assert (buf[0, :, :4] == 1.0).all() and (buf[3, :, 16:] == 4.0).all()

    # Verify MambaSpec block_size covers max_model_len in non-align mode
    captured: list[dict] = []
    orig_mamba_spec = sconv.MambaSpec
    sconv.MambaSpec = lambda **kw: captured.append(kw) or kw
    try:
        cs._dtype = torch.bfloat16
        cfg_none = types.SimpleNamespace(
            cache_config=types.SimpleNamespace(
                block_size=128, mamba_block_size=None, mamba_cache_mode="none"
            ),
            model_config=types.SimpleNamespace(max_model_len=2000),
        )
        cs.get_kv_cache_spec(cfg_none)
        assert captured[-1]["block_size"] == 2048
        cfg_align = types.SimpleNamespace(
            cache_config=types.SimpleNamespace(
                block_size=128, mamba_block_size=None, mamba_cache_mode="align"
            ),
            model_config=types.SimpleNamespace(max_model_len=2000),
        )
        cs.get_kv_cache_spec(cfg_align)
        assert captured[-1]["block_size"] == 128
    finally:
        sconv.MambaSpec = orig_mamba_spec


def test_loop_decode_group_matches_ungrouped_with_nondivisible_core_start() -> None:
    # Exercise prelude_num_blocks ceil when core.start % block_size != 0.
    torch.manual_seed(9)
    for mod_name in (
        "inkling2_pkg.quant_config",
        "inkling2_pkg.hpu.model",
        "vllm.logger",
        "vllm.forward_context",
    ):
        if mod_name not in sys.modules:
            sys.modules[mod_name] = types.ModuleType(mod_name)
    sys.modules["inkling2_pkg"].__path__ = []
    sys.modules["inkling2_pkg.hpu"].__path__ = []
    sys.modules["inkling2_pkg.quant_config"].Inkling2FP8QuantConfig = object
    sys.modules["vllm.logger"].init_logger = lambda name: types.SimpleNamespace(warning=lambda *a, **k: None)
    sys.modules["vllm.forward_context"].get_forward_context = lambda: types.SimpleNamespace(attn_metadata=None)
    sys.modules["inkling2_pkg.hpu.model"].Inkling2HPUDecoderLayer = object
    sys.modules["inkling2_pkg.hpu.model"].Inkling2HPUModel = torch.nn.Module
    sys.modules["inkling2_pkg.hpu.model"].Inkling2HPUForConditionalGeneration = torch.nn.Module

    spec = importlib.util.spec_from_file_location("inkling2_pkg.hpu.loop", _HPU_DIR / "loop.py")
    assert spec is not None and spec.loader is not None
    loop_mod = importlib.util.module_from_spec(spec)
    sys.modules["inkling2_pkg.hpu.loop"] = loop_mod
    spec.loader.exec_module(loop_mod)

    # core = range(2, 6), block_size = 3 -> prelude covers layers 0,1 (1 block: ceil(2/3) = 1),
    # total layers = 8, num_loop_passes = 2 -> visits: (0,0),(1,0), (2,0)..(5,0), (2,1)..(5,1), (6,0),(7,0)
    # group size = 2 divides core=(2,6) and num_layers=8.
    d, t = 8, 2
    block_size = 3
    core = range(2, 6)

    class _DummyLayer(torch.nn.Module):
        def __init__(self, idx: int) -> None:
            super().__init__()
            self.w = torch.randn(d, d, dtype=torch.float32) * 0.1
            self.idx = idx

        def forward(
            self,
            positions,
            completed_blocks,
            partial,
            inputs,
            mlp_q,
            next_q,
            next_nw,
            next_eps,
            tau,
            *,
            first=False,
            pass_idx=0,
            batch_size=1,
            decode_ctx=None,
            bank=None,
            message_query=None,
        ):
            delta = torch.matmul(inputs[0].float(), self.w).to(partial.dtype)
            if bank is not None and message_query is not None:
                delta = delta + bank.mean(dim=0) * 0.05
            new_partial = delta if first else partial + delta
            if first:
                completed_blocks = (
                    torch.cat([completed_blocks, partial.unsqueeze(0)], dim=0)
                    if completed_blocks.shape[0] > 0
                    else partial.unsqueeze(0)
                )
            nxt = ops.attn_res(completed_blocks, new_partial, None, next_q, next_nw, next_eps, True)
            return new_partial, nxt

    layers = [_DummyLayer(i) for i in range(8)]
    visits = (
        [(0, 0), (1, 0)]
        + [(i, 0) for i in core]
        + [(i, 1) for i in core]
        + [(6, 0), (7, 0)]
    )
    attn_qs = [torch.randn(3, d) * 0.1 for _ in range(7)]
    mlp_qs = [torch.randn(3, d) * 0.1 for _ in range(8)]
    out_q = torch.randn(3, d) * 0.1
    msg_qs = [torch.randn(3, d) * 0.1 for _ in core]
    norm_w = torch.ones(3, d, dtype=torch.bfloat16)

    # Run reference ungrouped loop
    x0 = torch.randn(t, d, dtype=torch.bfloat16)
    completed = x0.new_empty((0, t, d))
    partial = x0
    cur_in = x0.unsqueeze(0).expand(3, -1, -1).contiguous()
    prelude_completed = None
    prelude_partial = None
    bank = None
    prelude_num_blocks = (core.start + block_size - 1) // block_size
    assert prelude_num_blocks == 1  # ceil(2/3), whereas floor 2//3 == 0 would drop layer 0's block

    for v_idx, (i, p) in enumerate(visits):
        if i == core.start and p == 0:
            prelude_completed = completed
            prelude_partial = partial
        elif i == core.start and p > 0:
            bank = torch.cat([completed[prelude_num_blocks + 1 :], partial.unsqueeze(0)], dim=0)
            completed = prelude_completed
            partial = prelude_partial
            cur_in = ops.attn_res(completed, partial, None, attn_qs[core.start - 1], None, 0.0, True)
        first = (i % block_size) == 0
        if v_idx + 1 == len(visits):
            nq, nw, ne = out_q, norm_w, 1e-6
        else:
            ni, np_ = visits[v_idx + 1]
            nq = attn_qs[ni - 1]
            nw, ne = (None, 0.0) if np_ else (norm_w, 1e-6)
        new_p, cur_in = layers[i](
            None,
            completed,
            partial,
            cur_in,
            mlp_qs[i],
            nq,
            nw,
            ne,
            None,
            first=first,
            pass_idx=p,
            bank=bank if p > 0 else None,
            message_query=msg_qs[i - core.start] if p > 0 else None,
        )
        if first:
            completed = (
                torch.cat([completed, partial.unsqueeze(0)], dim=0)
                if completed.shape[0] > 0
                else partial.unsqueeze(0)
            )
        partial = new_p
    ref_out = cur_in[0]

    # Run grouped via Inkling2HPULoopDecodeGroup (g=2)
    g = 2
    groups = []
    for v_start in range(0, len(visits), g):
        sl = visits[v_start : v_start + g]
        fi, pi = sl[0]
        looped = pi > 0
        nqs, nws, nes = [], [], []
        for off, (i, _) in enumerate(sl):
            vi = v_start + off
            if vi + 1 == len(visits):
                nqs.append(out_q)
                nws.append(norm_w)
                nes.append(1e-6)
            else:
                ni, np_ = visits[vi + 1]
                nqs.append(attn_qs[ni - 1])
                nws.append(None if np_ else norm_w)
                nes.append(0.0 if np_ else 1e-6)
        is_entry = fi == core.start and looped
        grp = loop_mod.Inkling2HPULoopDecodeGroup(
            layers=tuple(layers[i] for i, _ in sl),
            first_flags=tuple((i % block_size) == 0 for i, _ in sl),
            pass_idx=pi,
            mlp_queries=tuple(mlp_qs[i] for i, _ in sl),
            next_queries=tuple(nqs),
            next_norm_weights=tuple(nws),
            next_norm_eps=tuple(nes),
            message_queries=tuple(msg_qs[i - core.start] for i, _ in sl) if looped else None,
            prelude_entry_query=attn_qs[core.start - 1] if is_entry else None,
            prelude_num_blocks=prelude_num_blocks if is_entry else 0,
        )
        groups.append((fi, pi, grp))

    g_completed = None
    g_partial = x0
    g_in = x0.unsqueeze(0).expand(3, -1, -1).contiguous()
    g_bank = None
    g_pre_c = None
    g_pre_p = None
    for fi, pi, grp in groups:
        if fi == core.start and pi == 0:
            g_pre_c, g_pre_p = g_completed, g_partial
        g_completed, g_partial, g_in, g_bank, _, _ = grp(
            None,
            g_completed,
            g_partial,
            g_in,
            None,
            None,
            g_bank if pi > 0 else None,
            1,
            g_pre_c if (fi == core.start and pi > 0) else None,
            g_pre_p if (fi == core.start and pi > 0) else None,
        )
    assert torch.allclose(g_in[0].float(), ref_out.float(), atol=1e-5)


def test_compute_decode_attn_context_tpc_only_and_slots() -> None:
    attn_mod = sys.modules["inkling2_pkg.hpu.attention"]
    # Out-of-order logical indices: block 0 is logical 1 (start 128), block 2 is logical 0 (start 0).
    block_list = torch.tensor([5, 1, 2, -1], dtype=torch.long)
    block_groups = torch.tensor([0, 1, 0, -1], dtype=torch.long)
    block_usage = torch.tensor([3, 5, 128, 0], dtype=torch.bfloat16)
    block_logical_idx = torch.tensor([1, 0, 0, 0], dtype=torch.long)
    positions = torch.tensor([130, 4], dtype=torch.long)
    slot_mapping = torch.tensor([259, 133], dtype=torch.long)
    load_indices = torch.tensor([[1, 2], [3, 0]], dtype=torch.long)
    store_indices = torch.tensor([[2, 3], [0, 1]], dtype=torch.long)

    ctx = attn_mod.compute_decode_attn_context_tensors(
        block_groups,
        block_usage,
        positions,
        2,
        block_size=128,
        sliding_window=128,
        rel_extent=128,
        block_logical_idx=block_logical_idx,
        block_list=block_list,
        slot_mapping=slot_mapping,
        kv_num_slots=256,
        load_indices=load_indices,
        store_indices=store_indices,
        conv_num_slots=4,
        tpc_only=True,
    )
    assert ctx.safe_grp is None
    assert ctx.valid_slot_global is None and ctx.valid_slot_local is None
    assert ctx.clamped_global is None and ctx.clamped_local is None
    assert ctx.grp_mask is None and ctx.grp_w_t is None
    assert ctx.block_start_pos_i32.tolist() == [128, 0, 0, 0]
    assert ctx.block_list_i32 is not None and ctx.block_list_i32.tolist() == [5, 1, 2, -1]
    assert ctx.block_usage_i32.tolist() == [3, 5, 128, 0]
    assert ctx.kv_slots is not None and ctx.kv_slots.tolist() == [3, 133]
    assert ctx.conv_load_slots is not None and ctx.conv_load_slots.dtype == torch.int32
    assert ctx.conv_load_slots.tolist() == [[1, 2], [3, 0]]
    assert ctx.conv_store_slots is not None and ctx.conv_store_slots.tolist() == [[2, 3], [0, 1]]


def test_sconv_hoisted_2d_cache_group_idx() -> None:
    sconv = sys.modules["inkling2_pkg.hpu.sconv"]
    attn_mod = sys.modules["inkling2_pkg.hpu.attention"]
    cs = sconv.Inkling2HPUConvState.__new__(sconv.Inkling2HPUConvState)
    torch.nn.Module.__init__(cs)
    cs.prefix = "layer1.conv"
    cs.stream_sizes = (4, 4, 8, 8)
    cs.total_dim = 24
    cs.kernel_size = 4
    cs.cache_group_idx = torch.tensor(1, dtype=torch.long)
    cs._cache_group_int = 1

    buf = torch.arange(4 * 3 * 24, dtype=torch.bfloat16).view(4, 3, 24)
    cs.kv_cache = [buf]
    dctx = attn_mod.DecodeAttnContext(
        safe_grp=None,
        valid_slot_global=None,
        valid_slot_local=None,
        clamped_global=None,
        clamped_local=None,
        grp_mask=None,
        grp_w_t=None,
        neg_large=None,
        block_groups_i32=torch.zeros(1, dtype=torch.int32),
        block_usage_i32=torch.zeros(1, dtype=torch.int32),
        block_start_pos_i32=torch.zeros(1, dtype=torch.int32),
        positions_i32=torch.zeros(2, dtype=torch.int32),
        conv_load_slots=torch.tensor([[0, 1], [2, 3]], dtype=torch.int32),
        conv_store_slots=torch.tensor([[1, 0], [3, 2]], dtype=torch.int64),
        conv_num_slots=4,
    )
    md = types.SimpleNamespace(is_prompt=False)
    prev, safe_store, is_prompt, _, _ = cs.load_streams(md, batch_size=2, decode_ctx=dctx)
    assert is_prompt is False
    assert prev is not None and safe_store is not None
    assert safe_store.tolist() == [3, 2]
    assert torch.equal(torch.cat(prev, dim=-1), buf[torch.tensor([2, 3])])

    refs = cs.get_decode_cache_refs(md, decode_ctx=dctx)
    assert refs is not None
    state_buf, load_i32, store_idx = refs
    assert state_buf is buf
    assert load_i32.dtype == torch.int32 and load_i32.tolist() == [2, 3]
    assert store_idx.tolist() == [3, 2]


def test_moe_gate_forward_matches_reference() -> None:
    moe_mod = sys.modules["inkling2_pkg.hpu.moe"]
    torch.manual_seed(11)
    cfg = types.SimpleNamespace(
        n_routed_experts=128,
        hidden_size=128,
        num_experts_per_tok=8,
        route_scale=8.0,
        use_gate_bias=True,
    )
    gate = moe_mod.Inkling2HPUGate(cfg)
    gate.weight.data.copy_(torch.randn(128, 128, dtype=torch.bfloat16) * 0.2)
    gate.bias.data.copy_(torch.randn(128, dtype=torch.float32) * 0.1)
    w_orig = gate.weight.data.clone()
    gate.process_weights_after_loading()
    assert gate._weight_f32_ready and gate._weight_packed_ready
    assert gate.weight_packed_bf16.shape == (128, 128)

    x = torch.randn(4, 128, dtype=torch.bfloat16)
    ids, weights = gate(x)
    logits = torch.nn.functional.linear(x.float(), w_orig.float())
    _, ref_ids = torch.topk(torch.sigmoid(logits) + gate.bias.unsqueeze(0), 8, dim=-1, largest=True, sorted=True)
    ref_lp = torch.nn.functional.logsigmoid(logits.gather(-1, ref_ids))
    ref_w = torch.softmax(ref_lp, dim=-1) * 8.0
    assert torch.equal(ids, ref_ids)
    assert torch.allclose(weights, ref_w, atol=1e-5)


def test_loop_decode_group_need_next_f8() -> None:
    loop_mod = sys.modules["inkling2_pkg.hpu.loop"]
    torch.manual_seed(12)
    d, t = 8, 2
    seen_f8: list[tuple[bool, bool]] = []

    class _F8DummyLayer(torch.nn.Module):
        def forward(
            self,
            positions,
            completed_blocks,
            partial,
            inputs,
            mlp_q,
            next_q,
            next_nw,
            next_eps,
            tau,
            *,
            first=False,
            pass_idx=0,
            batch_size=1,
            decode_ctx=None,
            bank=None,
            message_query=None,
            inputs_f8=None,
            inputs_sc=None,
            return_next_f8=False,
        ):
            seen_f8.append((inputs_f8 is not None and inputs_sc is not None, return_next_f8))
            nxt_bf16 = inputs.clone()
            if return_next_f8:
                f8 = nxt_bf16.clamp(-240.0, 240.0).to(torch.float8_e4m3fn)
                sc = torch.ones((nxt_bf16.shape[0], nxt_bf16.shape[1]), dtype=torch.float32)
                return partial, nxt_bf16, f8, sc
            return partial, nxt_bf16

    layers = (_F8DummyLayer(), _F8DummyLayer())
    q = torch.randn(3, d)
    nw = torch.ones(3, d, dtype=torch.bfloat16)
    grp = loop_mod.Inkling2HPULoopDecodeGroup(
        layers=layers,
        first_flags=(False, False),
        pass_idx=0,
        mlp_queries=(q, q),
        next_queries=(q, q),
        next_norm_weights=(nw, nw),
        next_norm_eps=(1e-6, 1e-6),
        message_queries=None,
        need_next_f8=(True, True),
    )
    x0 = torch.randn(t, d, dtype=torch.bfloat16)
    inp = x0.unsqueeze(0).expand(3, -1, -1).contiguous()
    _, _, out_bf16, _, out_f8, out_sc = grp(None, None, x0, inp, None, None, None, t)
    assert seen_f8 == [(False, True), (True, True)]
    assert out_f8 is not None and out_f8.dtype == torch.float8_e4m3fn and out_f8.shape == out_bf16.shape
    assert out_sc is not None and out_sc.dtype == torch.float32 and out_sc.shape == (3, t)
