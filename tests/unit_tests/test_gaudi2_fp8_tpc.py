# SPDX-License-Identifier: Apache-2.0
"""Unit and live-HPU tests for Gaudi2 FP8/GDN/ShortConv TPC custom ops and fusion wiring."""

from unittest import mock

import pytest
import torch
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.fusion.quant_activation import QuantizedActivation, get_input_quant_key
from vllm.model_executor.layers.layernorm import GemmaRMSNorm, RMSNorm
from vllm.model_executor.layers.quantization.utils.quant_utils import kFp8DynamicTokenSym

from vllm_gaudi.extension import ops
from vllm_gaudi.ops import hpu_gaudi2_fp8_tpc as fp8_tpc
from vllm_gaudi.ops.hpu_fp8 import Fp8LinearMethod
from vllm_gaudi.ops.hpu_row_parallel_linear import HPURowParallelLinear


def test_norm_fusion_eligibility():
    with set_current_vllm_config(VllmConfig()):
        rms_ok = RMSNorm(2048, eps=1e-6)
        rms_ok.weight.data = rms_ok.weight.data.to(torch.bfloat16)
        rms_fp16 = RMSNorm(2048, eps=1e-6)
        rms_fp16.weight.data = rms_fp16.weight.data.to(torch.float16)
        gemma_ok = GemmaRMSNorm(5120, eps=1e-6)
        rms_post = RMSNorm(2048, eps=1e-6)
        rms_post.weight.data = rms_post.weight.data.to(torch.bfloat16)
        rms_bad_eps = RMSNorm(2048, eps=1e-5)
        rms_bad_eps.weight.data = rms_bad_eps.weight.data.to(torch.bfloat16)

    assert fp8_tpc.can_fuse_add_rms_norm_f8(rms_ok, 2048)
    assert fp8_tpc.can_fuse_add_rms_norm_f8(gemma_ok, 5120)
    assert not fp8_tpc.can_fuse_add_rms_norm_f8(rms_fp16, 2048)
    assert not fp8_tpc.can_fuse_add_rms_norm_f8(rms_ok, 2000)  # not % 64
    assert not fp8_tpc.can_fuse_add_rms_norm_f8(rms_ok, fp8_tpc.MAX_RMS_NORM_F8_DIM + 64)

    assert fp8_tpc.can_fuse_sandwich_rms_norm_f8(rms_post, rms_ok, 2048)
    assert not fp8_tpc.can_fuse_sandwich_rms_norm_f8(rms_bad_eps, rms_ok, 2048)
    assert not fp8_tpc.can_fuse_sandwich_rms_norm_f8(gemma_ok, rms_ok, 2048)


def test_gc_kernel_path_ready_checks_guids(tmp_path, monkeypatch):
    stale = tmp_path / "libgaudi2_tpc_proxy.so"
    stale.write_bytes(b"gdn_decode_f32_gaudi2\x00")
    monkeypatch.setenv("GC_KERNEL_PATH", str(stale))
    assert not fp8_tpc._gc_kernel_path_ready()

    fresh = tmp_path / "libgaudi2_tpc_kernels.so"
    fresh.write_bytes(b"".join(fp8_tpc._REQUIRED_GUIDS))
    monkeypatch.setenv("GC_KERNEL_PATH", f"{stale}:{fresh}")
    assert fp8_tpc._gc_kernel_path_ready()


def test_apply_block_fp8_linear_accepts_prequant_activation(monkeypatch):
    captured = {}

    def fake_prequant(x_fp8, x_scale, weight, weight_scale, bias=None, trans_B=True, out_dtype=torch.bfloat16):
        captured["shape"] = tuple(x_fp8.shape)
        captured["scale_shape"] = tuple(x_scale.shape)
        captured["trans_B"] = trans_B
        captured["out_dtype"] = out_dtype
        return torch.zeros(x_fp8.shape[0], weight.shape[0], dtype=out_dtype)

    monkeypatch.setattr(ops, "apply_prequant_fp8_linear_hpu", fake_prequant)

    class _DummyLayer:
        weight = torch.zeros(64, 128, dtype=torch.float8_e4m3fn)
        weight_scale_inv = torch.ones(64, dtype=torch.float32)

    qa = QuantizedActivation(
        data=torch.zeros(4, 128, dtype=torch.float8_e4m3fn),
        scale=torch.ones(4, 1, dtype=torch.float32),
        orig_dtype=torch.bfloat16,
        orig_shape=torch.Size([2, 2, 128]),
        quant_key=kFp8DynamicTokenSym,
    )
    out = ops.apply_block_fp8_linear_hpu(qa, _DummyLayer(), [128, 128], force_channel_fp8=True)
    assert out.shape == (2, 2, 64)
    assert out.dtype == torch.bfloat16
    assert captured == {"shape": (4, 128), "scale_shape": (4, 1), "trans_B": True, "out_dtype": torch.bfloat16}


def test_silu_mul_support_predicate_and_row_gate(monkeypatch):
    with set_current_vllm_config(VllmConfig()):
        act = SiluAndMul()
    linear = mock.Mock()
    x_cpu = torch.zeros(2, 256, dtype=torch.bfloat16)
    assert not fp8_tpc._silu_mul_f8_supported(act, x_cpu, linear)

    monkeypatch.setattr(fp8_tpc, "_MIN_ROWS", 1)
    monkeypatch.setattr(fp8_tpc, "_MAX_ROWS", 64)
    assert fp8_tpc.within_fp8_tpc_max_rows(torch.empty(64, 2048))
    assert not fp8_tpc.within_fp8_tpc_max_rows(torch.empty(128, 2048))
    assert fp8_tpc.within_fp8_tpc_max_rows(torch.empty(32, 1, 2048))
    assert not fp8_tpc.within_fp8_tpc_max_rows(torch.empty(1, 128, 2048))
    # Zero-row, 1D, and zero-width inputs are always rejected.
    assert not fp8_tpc.within_fp8_tpc_max_rows(torch.empty(0, 2048))
    assert not fp8_tpc.within_fp8_tpc_max_rows(torch.empty(2048))
    assert not fp8_tpc.within_fp8_tpc_max_rows(torch.empty(4, 0))
    monkeypatch.setattr(fp8_tpc, "_MIN_ROWS", 16)
    assert not fp8_tpc.within_fp8_tpc_max_rows(torch.empty(8, 2048))
    assert fp8_tpc.within_fp8_tpc_max_rows(torch.empty(16, 2048))
    monkeypatch.setattr(fp8_tpc, "_MAX_ROWS", 0)
    assert fp8_tpc.within_fp8_tpc_max_rows(torch.empty(2048, 2048))
    assert not fp8_tpc.within_fp8_tpc_max_rows(torch.empty(0, 2048))

    import vllm_gaudi.envs as gaudi_envs

    monkeypatch.delenv("VLLM_HPU_GAUDI2_FP8_TPC_MIN_ROWS", raising=False)
    monkeypatch.delenv("VLLM_HPU_CGUID_DYNAMIC_QUANT", raising=False)
    monkeypatch.delenv("VLLM_HPU_FLASHINFER_GDN", raising=False)
    assert gaudi_envs.VLLM_HPU_GAUDI2_FP8_TPC_MIN_ROWS == 16
    monkeypatch.setenv("VLLM_HPU_FLASHINFER_GDN", "1")
    assert gaudi_envs.VLLM_HPU_GAUDI2_FP8_TPC_MIN_ROWS == 1
    monkeypatch.setenv("VLLM_HPU_CGUID_DYNAMIC_QUANT", "0")
    assert gaudi_envs.VLLM_HPU_GAUDI2_FP8_TPC_MIN_ROWS == 16
    monkeypatch.setenv("VLLM_HPU_GAUDI2_FP8_TPC_MIN_ROWS", "   ")
    assert gaudi_envs.VLLM_HPU_GAUDI2_FP8_TPC_MIN_ROWS == 16
    monkeypatch.setenv("VLLM_HPU_GAUDI2_FP8_TPC_MIN_ROWS", "0")
    assert gaudi_envs.VLLM_HPU_GAUDI2_FP8_TPC_MIN_ROWS == 1


def test_expose_dynamic_fp8_key_guards(monkeypatch):
    monkeypatch.setattr(ops, "is_hpu_gaudi2", True)

    method = Fp8LinearMethod.__new__(Fp8LinearMethod)
    method.block_quant = False

    class _Layer(torch.nn.Module):
        def __init__(self, input_is_parallel=True, tp_size=1, input_scale=None, dtype=torch.float8_e4m3fn):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(64, 128, dtype=dtype), requires_grad=False)
            self.input_is_parallel = input_is_parallel
            self.tp_size = tp_size
            self.input_scale = input_scale

    ok_layer = _Layer(input_is_parallel=True, tp_size=1)
    method._maybe_expose_dynamic_fp8_key(ok_layer)
    assert get_input_quant_key(ok_layer) == kFp8DynamicTokenSym

    # input_is_parallel=False must NOT expose key even when tp_size == 1 (upstream calls split_tensor_along_last_dim).
    non_par = _Layer(input_is_parallel=False, tp_size=1)
    method._maybe_expose_dynamic_fp8_key(non_par)
    assert get_input_quant_key(non_par) is None

    # Static input_scale must NOT expose dynamic per-token key.
    static_layer = _Layer(input_is_parallel=True, input_scale=torch.ones(1))
    method._maybe_expose_dynamic_fp8_key(static_layer)
    assert get_input_quant_key(static_layer) is None


def test_hpu_row_parallel_linear_accepts_quantized_activation(monkeypatch):
    layer = HPURowParallelLinear.__new__(HPURowParallelLinear)
    torch.nn.Module.__init__(layer)
    layer.input_is_parallel = True
    layer.tp_size = 2
    layer.tp_rank = 0
    layer.reduce_results = True
    layer.num_chunks = 2
    layer.chunk_threshold = 2
    layer.output_size_per_partition = 16
    layer.bias = None
    layer.skip_bias_add = False
    layer.return_bias = True

    seen_chunks = []

    class _FakeQuantMethod:
        def apply(self, _layer, x, bias=None):
            assert isinstance(x, QuantizedActivation)
            seen_chunks.append((tuple(x.data.shape), tuple(x.scale.shape), tuple(x.orig_shape)))
            return torch.ones(*x.orig_shape[:-1], _layer.output_size_per_partition, dtype=x.orig_dtype)

    layer.quant_method = _FakeQuantMethod()

    class _DummyHandle:
        def wait(self):
            return None

    monkeypatch.setattr("vllm_gaudi.ops.hpu_row_parallel_linear.get_tp_group", lambda: mock.Mock(device_group=None))
    monkeypatch.setattr(torch.distributed, "all_reduce", lambda tensor, group=None, async_op=False: _DummyHandle())

    # Chunked 3D decode path ([batch=4, seq=1, hidden=64])
    qa_3d = QuantizedActivation(
        data=torch.zeros(4, 64, dtype=torch.float8_e4m3fn),
        scale=torch.ones(4, 1, dtype=torch.float32),
        orig_dtype=torch.bfloat16,
        orig_shape=torch.Size([4, 1, 64]),
        quant_key=kFp8DynamicTokenSym,
    )
    out, bias = layer.forward(qa_3d)
    assert out.shape == (4, 1, 16)
    assert bias is None
    assert seen_chunks == [((2, 64), (2, 1), (2, 1, 64)), ((2, 64), (2, 1), (2, 1, 64))]

    # Unchunked 2D path (below chunk_threshold)
    seen_chunks.clear()
    layer.tp_size = 1
    qa_2d = QuantizedActivation(
        data=torch.zeros(3, 64, dtype=torch.float8_e4m3fn),
        scale=torch.ones(3, 1, dtype=torch.float32),
        orig_dtype=torch.bfloat16,
        orig_shape=torch.Size([3, 64]),
        quant_key=kFp8DynamicTokenSym,
    )
    out2, _ = layer.forward(qa_2d)
    assert out2.shape == (3, 16)
    assert seen_chunks == [((3, 64), (3, 1), (3, 64))]


def test_hpu_short_conv_multi_group_slot_selection_and_lfm3_layer_kinds():
    from vllm_gaudi.models.lfm3 import HpuLfm3Model
    from vllm_gaudi.ops.hpu_short_conv import HPUShortConv

    sc = HPUShortConv.__new__(HPUShortConv)
    sc.cache_group_idx = torch.tensor(2, dtype=torch.int64)
    sc._decode_group_int = 1
    indices = torch.tensor([[10, 11], [20, 21], [30, 31]], dtype=torch.int64)
    # Prompt path (`is_decode=False`) must always read `cache_group_idx` (row 2), never `_decode_group_int`.
    assert torch.equal(sc._select_cache_group(indices, is_decode=False), indices[2])
    assert torch.equal(sc._select_cache_group(indices, is_decode=True), indices[1])
    sc._decode_group_int = None
    assert torch.equal(sc._select_cache_group(indices, is_decode=True), indices[2])

    class _DummyFF:
        w2 = torch.nn.Linear(16, 16, bias=False)

    class _DummyLayer(torch.nn.Module):
        def __init__(self, is_attn: bool):
            super().__init__()
            self.is_attention_layer = is_attn
            self.feed_forward = _DummyFF()
            self.short_conv = None if is_attn else sc

    m = HpuLfm3Model.__new__(HpuLfm3Model)
    torch.nn.Module.__init__(m)
    m.layers = torch.nn.ModuleList([_DummyLayer(False), _DummyLayer(True), _DummyLayer(False), _DummyLayer(True)])
    m.start_layer = 0
    m.end_layer = 4
    for lt in (["conv", "sliding_attention", "conv", "sliding_attention"], None):
        m.config = mock.Mock(layer_types=lt)
        kinds = m.hpu_layer_kinds()
        assert kinds[0] != kinds[2], f"ShortConv layers at different indices must produce distinct kinds (lt={lt})"
        assert kinds[1] == kinds[3], f"Identical attention layers may still share group kind (lt={lt})"

    # Without compiled layer groups, `_ensure_short_conv_group_ints` must leave `_decode_group_int` unset.
    m._ensure_short_conv_group_ints()
    assert sc._decode_group_int is None


@pytest.mark.skipif(not fp8_tpc.ensure_gaudi2_fp8_tpc_loaded(), reason="Gaudi2 FP8 TPC ops not loaded")
def test_hpu_live_ops_equivalence_and_symbolic_meta():
    dev = torch.device("hpu")
    fp8_max = float(ops.FP8_MAX)

    # 1. Sandwich RMSNorm + FP8 quant equivalence
    torch.manual_seed(11)
    x = torch.randn(4, 2048, dtype=torch.bfloat16, device=dev)
    res = torch.randn(4, 2048, dtype=torch.bfloat16, device=dev)
    pw = (torch.randn(2048, dtype=torch.bfloat16, device=dev) * 0.2 + 1.0)
    nw = (torch.randn(2048, dtype=torch.bfloat16, device=dev) * 0.2 + 1.0)
    fn_sw = torch.compile(
        lambda a, b, c, d: torch.ops.custom_op.gaudi2_sandwich_rms_norm_f8(a, b, c, d, 1e-6, fp8_max),
        backend="hpu_backend",
    )
    qf, sc, r_out = fn_sw(x, res, pw, nw)
    x_f = x.float()
    xp = ((x_f * torch.rsqrt(x_f.pow(2).mean(-1, keepdim=True) + 1e-6)) * pw.float()).bfloat16()
    r_ref = res + xp
    rf = r_ref.float()
    y_ref = ((rf * torch.rsqrt(rf.pow(2).mean(-1, keepdim=True) + 1e-6)) * nw.float()).bfloat16()
    torch.hpu.synchronize()
    assert (r_out.float() - r_ref.float()).abs().max().item() < 0.05
    dq = qf.float() * sc
    rel = (dq - y_ref.float()).abs().max().item() / y_ref.float().abs().max().item()
    assert rel < 0.08

    # 2. Pure-functional 2-output lfm3_short_conv_decode equivalence (read-only conv_state)
    D, B, K, slots = 256, 3, 7, 6
    state_len = K - 1
    bcx = torch.randn(B, 3 * D, dtype=torch.bfloat16, device=dev)
    cs = torch.randn(slots, state_len, D, dtype=torch.bfloat16, device=dev)
    cs_orig = cs.clone()
    lidx = torch.tensor([2, 2, 4], dtype=torch.int32, device=dev)
    taps = torch.randn(K, D, dtype=torch.float32, device=dev) * 0.3
    bias = torch.zeros(D, dtype=torch.float32, device=dev)
    fn_sc = torch.compile(
        lambda a, b, c, d, e: torch.ops.custom_op.lfm3_short_conv_decode(a, b, c, d, e, 0),
        backend="hpu_backend",
    )
    y_sc, ns_sc = fn_sc(bcx, cs, lidx, taps, bias)
    torch.hpu.synchronize()
    assert torch.equal(cs, cs_orig)
    b_g, c_g, x_g = bcx.chunk(3, dim=-1)
    bx_ref = (b_g * x_g).to(torch.bfloat16)
    init_st = cs[lidx.long()]
    acc = sum(init_st[:, k, :].float() * taps[k] for k in range(state_len)) + bx_ref.float() * taps[state_len]
    y_ref_sc = c_g * acc.to(torch.bfloat16)
    ns_ref = torch.cat((init_st[:, 1:, :], bx_ref.unsqueeze(1)), dim=1)
    assert torch.equal(ns_sc, ns_ref)
    assert (y_sc.float() - y_ref_sc.float()).abs().max().item() < 0.05

    # 3. Symbolic dynamic batch tracing for all 7 ops (1 compile across B=16 and B=24).
    torch._dynamo.reset()
    compile_calls = [0]

    def _counting_backend(gm, example_inputs):
        compile_calls[0] += 1
        return gm

    @torch.compile(backend=_counting_backend, dynamic=True)
    def _trace_all(a, b, w, gu, qkv, ain, bin_, nea, dtb, cs_gdn, ctaps, cbias, gx, gz, gw, bcx_in, cs_sc, lidx_in,
                   staps, sbias):
        assert fp8_tpc.within_fp8_tpc_max_rows(a)
        o1 = torch.ops.custom_op.gaudi2_add_rms_norm_f8(a, b, w, 1e-6, fp8_max)
        o2 = torch.ops.custom_op.gaudi2_add_rms_norm_f8_bf16(a, b, w, 1e-6, fp8_max)
        o3 = torch.ops.custom_op.gaudi2_sandwich_rms_norm_f8(a, b, w, w, 1e-6, fp8_max)
        o4 = torch.ops.custom_op.gaudi2_silu_mul_f8(gu, fp8_max)
        o5 = torch.ops.custom_op.gdn_prep_decode(qkv, ain, bin_, nea, dtb, cs_gdn, ctaps, cbias, 0, 0, 0)
        o6 = torch.ops.custom_op.gdn_gated_rms_norm_f8(gx, gz, gw, 1e-6, fp8_max)
        o7 = torch.ops.custom_op.lfm3_short_conv_decode(bcx_in, cs_sc, lidx_in, staps, sbias, 0)
        return o1[0], o2[3], o3[0], o4[0], o5[0], o6[0], o7[0], o7[1]

    H_k, H_v = 4, 12
    W_gdn = (2 * H_k + H_v) * 128
    nea = torch.ones(H_v, dtype=torch.float32, device=dev) * -0.5
    dtb = torch.zeros(H_v, dtype=torch.float32, device=dev)
    cs_gdn = torch.zeros(30, 3, W_gdn, dtype=torch.bfloat16, device=dev)
    ctaps = torch.ones(4, W_gdn, dtype=torch.float32, device=dev) * 0.25
    cbias = torch.zeros(W_gdn, dtype=torch.float32, device=dev)
    gw = torch.ones(128, dtype=torch.float32, device=dev)
    cs_sc = torch.zeros(30, 6, 256, dtype=torch.bfloat16, device=dev)
    staps = torch.ones(7, 256, dtype=torch.float32, device=dev) * 0.1
    sbias = torch.zeros(256, dtype=torch.float32, device=dev)
    w256 = torch.ones(256, dtype=torch.bfloat16, device=dev)

    for bs in (16, 24):
        _trace_all(
            torch.randn(bs, 256, dtype=torch.bfloat16, device=dev),
            torch.randn(bs, 256, dtype=torch.bfloat16, device=dev),
            w256,
            torch.randn(bs, 512, dtype=torch.bfloat16, device=dev),
            torch.randn(bs, W_gdn, dtype=torch.bfloat16, device=dev),
            torch.randn(bs, H_v, dtype=torch.bfloat16, device=dev),
            torch.randn(bs, H_v, dtype=torch.bfloat16, device=dev),
            nea, dtb, cs_gdn, ctaps, cbias,
            torch.randn(bs, H_v, 128, dtype=torch.float32, device=dev),
            torch.randn(bs, H_v * 128, dtype=torch.bfloat16, device=dev),
            gw,
            torch.randn(bs, 768, dtype=torch.bfloat16, device=dev),
            cs_sc,
            torch.arange(bs, dtype=torch.int32, device=dev),
            staps, sbias,
        )
    assert compile_calls[0] == 1, f"expected 1 symbolic compile across batch sizes, got {compile_calls[0]}"
