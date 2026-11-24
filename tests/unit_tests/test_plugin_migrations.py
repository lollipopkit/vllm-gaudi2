# SPDX-License-Identifier: Apache-2.0
"""Unit tests verifying vLLM plugin registrations and patches migrated into vllm_gaudi."""

from __future__ import annotations

from types import SimpleNamespace
from typing import get_args
from unittest.mock import patch

import pytest
import torch

from vllm_gaudi import envs as gaudi_envs
from vllm_gaudi.models import register_model
from vllm_gaudi.patches import (
    _patch_inkling2_registrations,
    _patch_logits_processor_apply_head,
    _patch_v1_dflash_candidate_heads,
    apply_plugin_registrations,
)
from vllm_gaudi.platform import HpuPlatform


def test_inkling2_config_and_quant_registration_idempotent() -> None:
    from vllm.model_executor.layers.quantization import (
        QUANTIZATION_METHODS,
        QuantizationMethods,
        _CUSTOMIZED_METHOD_TO_QUANT_CONFIG,
        get_quantization_config,
    )
    from vllm.transformers_utils.config import _CONFIG_REGISTRY

    from vllm_gaudi.models.inkling2.configs import (
        Inkling2LoopMMConfig,
        Inkling2MMConfig,
        Inkling2ModelConfig,
    )
    from vllm_gaudi.models.inkling2.quant_config import Inkling2FP8QuantConfig

    # Exercise first-time registration + dedup branch cleanly.
    _CUSTOMIZED_METHOD_TO_QUANT_CONFIG.pop("inkling2_fp8", None)
    while "inkling2_fp8" in QUANTIZATION_METHODS:
        QUANTIZATION_METHODS.remove("inkling2_fp8")
    if "inkling2_fp8" not in HpuPlatform.supported_quantization:
        HpuPlatform.supported_quantization.append("inkling2_fp8")

    apply_plugin_registrations()
    _patch_inkling2_registrations()  # idempotent second call

    assert _CONFIG_REGISTRY["inkling2"] is Inkling2MMConfig
    assert _CONFIG_REGISTRY["inkling2_loop"] is Inkling2LoopMMConfig
    assert _CONFIG_REGISTRY["inkling2_model"] is Inkling2ModelConfig
    assert _CONFIG_REGISTRY["inkling2_loop_text"] is Inkling2ModelConfig

    assert get_quantization_config("inkling2_fp8") is Inkling2FP8QuantConfig
    assert QUANTIZATION_METHODS.count("inkling2_fp8") == 1
    assert HpuPlatform.supported_quantization.count("inkling2_fp8") == 1
    # Custom method (not in upstream QuantizationMethods Literal) so
    # ModelConfig._verify_quantization probes it ahead of built-in overrides.
    assert "inkling2_fp8" not in get_args(QuantizationMethods)

    # Verify override_quantization_method branches.
    ink_cfg = SimpleNamespace(architectures=["Inkling2LoopForConditionalGeneration"])
    other_cfg = SimpleNamespace(architectures=["LlamaForCausalLM"])
    assert Inkling2FP8QuantConfig.override_quantization_method({"quant_method": "inkling2_fp8"}, None, ink_cfg) == "inkling2_fp8"
    assert Inkling2FP8QuantConfig.override_quantization_method({"quant_method": "inkling2_fp8"}, None, other_cfg) is None
    with pytest.raises(ValueError, match="convert_inkling2_fp8.py"):
        Inkling2FP8QuantConfig.override_quantization_method(
            {"quant_method": "modelopt", "quant_algo": "NVFP4"}, None, ink_cfg
        )


def test_logits_processor_apply_head_honors_supports_out_dtype() -> None:
    from vllm.model_executor.layers.logits_processor import LogitsProcessor

    _patch_logits_processor_apply_head()
    _patch_logits_processor_apply_head()  # idempotent

    seen: dict[str, object] = {}

    class _DummyQuantMethod:
        supports_out_dtype = True

        def apply(self, layer, x, bias=None, out_dtype=None):
            seen["out_dtype"] = out_dtype
            seen["bias"] = bias
            res = x.to(out_dtype or x.dtype)
            return res + bias if bias is not None else res

    bias = torch.ones(4, dtype=torch.float32)
    x = torch.randn(2, 4, dtype=torch.bfloat16)
    dummy_lm_head = SimpleNamespace(quant_method=_DummyQuantMethod())

    # Different head_dtype + non-None bias -> forwards out_dtype & bias to apply().
    out = LogitsProcessor._apply_head(SimpleNamespace(head_dtype=torch.float32), dummy_lm_head, x, bias)
    assert out.dtype == torch.float32
    assert seen["out_dtype"] == torch.float32
    assert seen["bias"] is bias

    # Same dtype or head_dtype=None -> delegates to upstream path (no out_dtype kwarg).
    for head_dtype in (torch.bfloat16, None):
        seen.clear()
        out_same = LogitsProcessor._apply_head(SimpleNamespace(head_dtype=head_dtype), dummy_lm_head, x, None)
        assert out_same.dtype == torch.bfloat16
        assert seen["out_dtype"] is None

    # Quant method without supports_out_dtype -> falls back to upstream error for quantized lm_head.
    class _UnsupportedQuantMethod:
        supports_out_dtype = False

        def apply(self, layer, x, bias=None):
            return x

    with pytest.raises(ValueError, match="unquantized lm_head"):
        LogitsProcessor._apply_head(
            SimpleNamespace(head_dtype=torch.float32),
            SimpleNamespace(quant_method=_UnsupportedQuantMethod()),
            x,
            None,
        )


def test_v1_dflash_candidate_heads_patch() -> None:
    from vllm.config.vllm import VllmConfig

    _patch_v1_dflash_candidate_heads()
    _patch_v1_dflash_candidate_heads()  # idempotent

    # On vLLM >= 0.34, V1 feature gating moved to HpuPlatform._unsupported_engine_features.
    spec_ok = SimpleNamespace(
        method="dflash",
        custom_method=None,
        speculator_config=object(),
        enable_adaptive_verification=False,
        draft_lm_head_quantization=None,
        draft_model_config=SimpleNamespace(
            architectures=["DFlash2DraftModel"],
            hf_config=SimpleNamespace(layer_types=["sliding_attention"]),
        ),
    )
    cfg_ok = SimpleNamespace(
        model_config=None,
        parallel_config=SimpleNamespace(prefill_context_parallel_size=1, enable_batch_sharded_sampling=False),
        speculative_config=spec_ok,
    )
    assert "DFlash candidate-head drafts" not in HpuPlatform._unsupported_engine_features(cfg_ok)

    spec_beam = SimpleNamespace(**{
        **spec_ok.__dict__,
        "draft_model_config": SimpleNamespace(
            architectures=["DFlash2BeamDraftModel"],
            hf_config=SimpleNamespace(layer_types=["sliding_attention"]),
        ),
    })
    cfg_beam = SimpleNamespace(
        model_config=None,
        parallel_config=SimpleNamespace(prefill_context_parallel_size=1, enable_batch_sharded_sampling=False),
        speculative_config=spec_beam,
    )
    assert "DFlash candidate-head drafts" in HpuPlatform._unsupported_engine_features(cfg_beam)

    # Also verify legacy VllmConfig._get_v1_model_runner_unsupported_features hook when present.
    if hasattr(VllmConfig, "_get_v1_model_runner_unsupported_features"):
        dummy_cfg = SimpleNamespace(
            parallel_config=SimpleNamespace(
                prefill_context_parallel_size=1,
                pipeline_parallel_size=1,
                enable_batch_sharded_sampling=False,
            ),
            scheduler_config=SimpleNamespace(async_scheduling=False),
            speculative_config=None,
            model_config=None,
            _dflash_needs_multi_kv_group=lambda: True,
            _is_dflash_candidate_draft=lambda: True,
        )
        with patch("vllm.platforms.current_platform", HpuPlatform):
            unsupported = VllmConfig._get_v1_model_runner_unsupported_features(dummy_cfg)
        assert "DFlash candidate-head drafts" not in unsupported
        assert "mixed sliding/full dflash drafts" in unsupported
        with patch("vllm.platforms.current_platform", SimpleNamespace()):
            unsupported = VllmConfig._get_v1_model_runner_unsupported_features(dummy_cfg)
        assert "DFlash candidate-head drafts" in unsupported
        with patch("vllm.platforms.current_platform", SimpleNamespace(supports_v1_dflash_candidate_heads=lambda: False)):
            unsupported = VllmConfig._get_v1_model_runner_unsupported_features(dummy_cfg)
        assert "DFlash candidate-head drafts" in unsupported


def test_model_registry_lfm3_and_inkling2() -> None:
    from vllm.model_executor.models.registry import ModelRegistry

    orig_val = gaudi_envs.VLLM_HPU_LFM3_PACKED_TOWER
    try:
        with patch.object(gaudi_envs, "VLLM_HPU_LFM3_PACKED_TOWER", True):
            register_model()
            expected_packed = {
                "Lfm3ForCausalLM": ("vllm_gaudi.models.lfm3", "HpuLfm3ForCausalLM"),
                "Lfm3VlForConditionalGeneration": ("vllm_gaudi.models.lfm3_vl", "HpuLfm3VlForConditionalGeneration"),
                "Lfm3OmniForConditionalGeneration": (
                    "vllm_gaudi.models.lfm3_vl",
                    "HpuLfm3OmniForConditionalGeneration",
                ),
                "Inkling2ForConditionalGeneration": (
                    "vllm_gaudi.models.inkling2",
                    "Inkling2ForConditionalGeneration",
                ),
                "Inkling2LoopForConditionalGeneration": (
                    "vllm_gaudi.models.inkling2",
                    "Inkling2LoopForConditionalGeneration",
                ),
            }
            for arch, (mod, cls) in expected_packed.items():
                entry = ModelRegistry.models[arch]
                assert (entry.module_name, entry.class_name) == (mod, cls), (arch, entry)

        with patch.object(gaudi_envs, "VLLM_HPU_LFM3_PACKED_TOWER", False):
            register_model()
            expected_unpacked = {
                "Lfm3VlForConditionalGeneration": (
                    "vllm_gaudi.models.lfm3_vl_base",
                    "Lfm3VlForConditionalGeneration",
                ),
                "Lfm3OmniForConditionalGeneration": (
                    "vllm_gaudi.models.lfm3_omni_base",
                    "Lfm3OmniForConditionalGeneration",
                ),
            }
            for arch, (mod, cls) in expected_unpacked.items():
                entry = ModelRegistry.models[arch]
                assert (entry.module_name, entry.class_name) == (mod, cls), (arch, entry)
    finally:
        with patch.object(gaudi_envs, "VLLM_HPU_LFM3_PACKED_TOWER", orig_val):
            register_model()


def test_lfm3_multimodal_processors_independent_of_upstream_qwen3_vl() -> None:
    from vllm.model_executor.models.qwen3_vl import Qwen3VLProcessingInfo

    from vllm_gaudi.models.lfm3_omni_base import (
        Lfm3OmniDummyInputsBuilder,
        Lfm3OmniMultiModalProcessor,
        Lfm3OmniProcessingInfo,
    )
    from vllm_gaudi.models.lfm3_vl_base import (
        Lfm3VlDummyInputsBuilder,
        Lfm3VlMultiModalProcessor,
        Lfm3VlProcessingInfo,
    )

    # Confirm upstream Qwen3VLProcessingInfo is pristine while LFM3 subclasses carry the hooks.
    assert not hasattr(Qwen3VLProcessingInfo, "_get_video_frame_indices")
    assert hasattr(Lfm3VlProcessingInfo, "_get_video_frame_indices")
    assert hasattr(Lfm3OmniProcessingInfo, "_get_video_frame_indices")
    assert hasattr(Lfm3VlMultiModalProcessor, "_get_video_token_ids")
    assert hasattr(Lfm3VlMultiModalProcessor, "_finalize_video_repl")
    assert hasattr(Lfm3VlMultiModalProcessor, "_extra_video_outputs")
    assert Lfm3VlMultiModalProcessor.hf_accepts_unfilled_video_placeholders is False
    assert issubclass(Lfm3OmniMultiModalProcessor, Lfm3VlMultiModalProcessor)
    assert issubclass(Lfm3OmniDummyInputsBuilder, Lfm3VlDummyInputsBuilder)

    # Behavioral check on _get_video_frame_indices (presampled + sampled).
    fake_info = SimpleNamespace(
        get_video_processor=lambda: SimpleNamespace(fps=2.0, min_frames=2, max_frames=8),
    )
    meta = {"frames_indices": [0, 3, 6], "total_num_frames": 10, "fps": 5.0}
    assert Lfm3VlProcessingInfo._get_video_frame_indices(fake_info, meta, do_sample_frames=False) == [0, 3, 6]
    assert Lfm3VlProcessingInfo._get_video_frame_indices(
        fake_info, meta, do_sample_frames=True, sampled_num_frames=4
    ) == [0, 3, 6, 9]

    # Behavioral check on get_dummy_mm_data with video=0 (never touches video processor).
    fake_builder = SimpleNamespace(
        info=SimpleNamespace(get_image_size_with_most_features=lambda: (64, 64)),
        _get_dummy_images=lambda **kw: ["img"] * kw["num_images"],
    )
    dummy = Lfm3VlDummyInputsBuilder.get_dummy_mm_data(fake_builder, 1024, {"image": 2, "video": 0}, {})
    assert dummy == {"image": ["img", "img"], "video": []}
