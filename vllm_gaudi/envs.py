# SPDX-License-Identifier: Apache-2.0

import os
from typing import TYPE_CHECKING, Any, Callable, Optional

if TYPE_CHECKING:
    VLLM_USE_HPU_CONTIGUOUS_CACHE_FETCH: bool = True
    VLLM_HPU_FORCE_CHANNEL_FP8: bool = True
    VLLM_HPU_HETERO_KV_LAYOUT: bool = False
    VLLM_HPU_MULTI_MODEL_CONFIG: Optional[str] = None
    VLLM_NIXL_ABORT_REQUEST_TIMEOUT: float = 300.0
    VLLM_NIXL_SIDE_CHANNEL_HOST: str = "localhost"
    VLLM_NIXL_SIDE_CHANNEL_PORT: int = 5600
    VLLM_HPU_NIXL_JOINT_KV: bool = False
    VLLM_HPU_NIXL_STAGING_SLOTS: int = 0
    VLLM_MINIMAX_M3_MOE_TOKEN_TILE: int = 512
    VLLM_MINIMAX_M3_MOE_DECODE_GATHER: bool = True
    VLLM_MINIMAX_M3_MOE_GATHER_MAX_TOKENS: int = 16
    VLLM_MM_WARMUP_OUTSIDE_COMPILE_ONLY: bool = False
    VLLM_HPU_MOE_GATHER: bool = False
    VLLM_HPU_MOE_GATHER_RATIO: float = 0.4
    VLLM_HPU_MOE_GATHER_VERIFY: bool = False
    VLLM_COMPACT_GDN: bool = False
    VLLM_HPU_FLASHINFER_GDN: bool = False
    VLLM_HPU_FLASHINFER_GDN_FUSED_DECODE: bool = False
    VLLM_HPU_FLASHINFER_GDN_PREFILL: bool = False
    VLLM_HPU_GDN_DIRECT_STATE: bool = True
    VLLM_HPU_CGUID_DYNAMIC_QUANT: bool = False
    VLLM_HPU_CGUID_DYNAMIC_QUANT_MAX_ROWS: int = 32
    VLLM_HPU_FP8_LM_HEAD: bool = False
    VLLM_HPU_SAMPLER_TOPK_CAP: int = 64
    VLLM_HPU_GDN_PREFILL_TUNING: str = ""
    VLLM_HPU_GDN_TPC_DECODE: bool = False
    VLLM_HPU_GAUDI2_FP8_TPC: bool = True
    VLLM_HPU_GAUDI2_FP8_TPC_TORCH_LIB: str = ""
    VLLM_HPU_GAUDI2_FP8_TPC_MIN_ROWS: int = 1
    VLLM_HPU_GAUDI2_FP8_TPC_MAX_ROWS: int = 64
    VLLM_HPU_TPC_ADD_RMS_NORM_F8: bool = True
    VLLM_HPU_TPC_SANDWICH_RMS_NORM_F8: bool = True
    VLLM_HPU_TPC_SILU_MUL_F8: bool = True
    VLLM_HPU_TPC_GDN_PREP: bool = True
    VLLM_HPU_TPC_GDN_GATED_NORM_F8: bool = True
    VLLM_HPU_TPC_LFM3_SHORT_CONV: bool = True
    VLLM_HPU_LFM3_PACKED_TOWER: bool = True
    HPU_KIMI_SITU_IMPL: str = "exact"
    HPU_KIMI_VIT_DP: bool = False
    HPU_KIMI_PREFILL_MERGE: bool = False
    HPU_KIMI_KDA_PREFILL_CHUNK: str = "64,16"
    HPU_KIMI_KDA_INTRA_SUB_CHUNK: str = "0"
    VLLM_HPU_TWO_STAGE_TOPK: bool = True
    VLLM_HPU_MIN_PREFILL_CHUNK: str = ""


def _bool_env(name: str, default: str) -> bool:
    return os.environ.get(name, default).strip().lower() in ("1", "true")

# The begin-* and end* here are used by the documentation generator
# to extract the used env vars.
# begin-env-vars-definition
environment_variables: dict[str, Callable[[], Any]] = {
    # Contiguous cache fetching to avoid using costly gather operation on
    # Gaudi3. This is only applicable to HPU contiguous cache. If set to true,
    # contiguous cache fetch will be used.
    "VLLM_USE_HPU_CONTIGUOUS_CACHE_FETCH":
    lambda: os.environ.get("VLLM_CONTIGUOUS_PA", "true").lower() in ("1", "true"),

    # Convert block fp8 to channel fp8 for HPU
    # If `QUANT_CONFIG` is set, this will be forced to false.
    "VLLM_HPU_FORCE_CHANNEL_FP8":
    lambda: os.environ.get("VLLM_HPU_FORCE_CHANNEL_FP8", "true").lower() in
    ("1", "true") and os.environ.get("QUANT_CONFIG", None) is None,

    # Enable prefill side kv_layout and block_size for heterogeneous run.
    "VLLM_HPU_HETERO_KV_LAYOUT":
    lambda: os.environ.get("VLLM_HPU_HETERO_KV_LAYOUT", "false").lower() in ("0", "false"),

    # Path to a YAML config file describing the multi-model setup
    # (model names, weights, tensor-parallel config, etc.).
    # When unset, multi-model mode is disabled.
    "VLLM_HPU_MULTI_MODEL_CONFIG":
    lambda: os.environ.get("VLLM_HPU_MULTI_MODEL_CONFIG", None),

    # Timeout in seconds for NIXL abort request handling
    "VLLM_NIXL_ABORT_REQUEST_TIMEOUT":
    lambda: float(os.environ.get("VLLM_NIXL_ABORT_REQUEST_TIMEOUT", "300")),

    # NIXL side channel host and port for KV transfer
    "VLLM_NIXL_SIDE_CHANNEL_HOST":
    lambda: os.environ.get("VLLM_NIXL_SIDE_CHANNEL_HOST", "localhost"),
    "VLLM_NIXL_SIDE_CHANNEL_PORT":
    lambda: int(os.environ.get("VLLM_NIXL_SIDE_CHANNEL_PORT", "5600")),

    # Enable joint-KV staging so an HPU prefill instance can serve a GPU
    # (FLASH_ATTN, blocks-first) decode instance over NIXL. When the remote
    # decode registers ONE joint [K|V] region per layer (block_len covering
    # both K and V, V at block_len // 2), the HPU must advertise the same
    # region model. Leave false for HPU<->HPU disagg (separate K/V regions).
    "VLLM_HPU_NIXL_JOINT_KV":
    lambda: os.environ.get("VLLM_HPU_NIXL_JOINT_KV", "false").lower() in ("1", "true"),

    # Number of joint-KV staging slots for heterogeneous (HPU prefill ->
    # GPU decode) NIXL transfer. Each slot holds one on-save block for all
    # layers laid out blocks-first [2, n_kv_heads, block_size_on_save,
    # head_size] so the GPU can read K and V from one contiguous region
    # (V at block_len // 2). 0 (default) auto-sizes to the full concurrent
    # workload: max_num_seqs * ceil(max_model_len / block_size_on_save), which
    # guarantees every schedulable request gets a reservation. Registration
    # fails fast if that pool does not fit device memory. Only set a smaller
    # value if you are certain peak concurrent transfer demand stays under it.
    "VLLM_HPU_NIXL_STAGING_SLOTS":
    lambda: int(os.environ.get("VLLM_HPU_NIXL_STAGING_SLOTS", "0")),

    # MiniMax-M3 SwiGLU-OAI expert execution tuning.
    "VLLM_MINIMAX_M3_MOE_TOKEN_TILE":
    lambda: int(os.environ.get("VLLM_MINIMAX_M3_MOE_TOKEN_TILE", "512")),
    "VLLM_MINIMAX_M3_MOE_DECODE_GATHER":
    lambda: os.environ.get("VLLM_MINIMAX_M3_MOE_DECODE_GATHER", "1").lower() in ("1", "true"),
    "VLLM_MINIMAX_M3_MOE_GATHER_MAX_TOKENS":
    lambda: int(os.environ.get("VLLM_MINIMAX_M3_MOE_GATHER_MAX_TOKENS", "16")),

    # Run multimodal warmup outside PT_COMPILE_ONLY_MODE for models with
    # data-dependent output shapes that must be materialized during warmup.
    "VLLM_MM_WARMUP_OUTSIDE_COMPILE_ONLY":
    lambda: os.environ.get("VLLM_MM_WARMUP_OUTSIDE_COMPILE_ONLY", "false").strip().lower() in ("1", "true"),

    # EXPERIMENTAL custom gathered-expert FP8 MoE combine (silu only).
    # Off by default; falls back to the stock Habana fused op when disabled or
    # when the number of gathered experts (min(local_experts, tokens*top_k))
    # would exceed this fraction of this rank's local_experts. Parameterizing on
    # the ratio to local_experts (rather than an absolute count) keeps the gate
    # sane across EP configs, which change local_experts.
    #
    # Default rationale (0.4): this optimization has only been observed to help
    # the Qwen 3.5 MoE family. A crossover sweep across Qwen3.5 35B/122B/397B at
    # several EP levels shows the custom-gather win/loss cutoff tracks
    # g/local_experts most closely, and it lands around this ratio.
    "VLLM_HPU_MOE_GATHER":
    lambda: os.environ.get("VLLM_HPU_MOE_GATHER", "0").lower() in ("1", "true"),
    "VLLM_HPU_MOE_GATHER_RATIO":
    lambda: float(os.environ.get("VLLM_HPU_MOE_GATHER_RATIO", "0.4")),
    "VLLM_HPU_MOE_GATHER_VERIFY":
    lambda: os.environ.get("VLLM_HPU_MOE_GATHER_VERIFY", "0").lower() in ("1", "true"),

    # Use the compact recurrent-state (conv/ssm) layout for gated delta net
    # models. The model runner auto-detects and sets this during init, so read
    # it lazily rather than caching it at import time.
    "VLLM_COMPACT_GDN":
    lambda: os.environ.get("VLLM_COMPACT_GDN", "0").strip().lower() in ("1", "true"),

    # Qwen3.5/3.8 GDN tactics derived from 1CatAI/1Cat-vLLM-Gaudi.
    # Parent switch for the FlashInfer-Gaudi GDN decode adapter: packed
    # single-token recurrence on the contiguous group-major compact state.
    "VLLM_HPU_FLASHINFER_GDN":
    lambda: _bool_env("VLLM_HPU_FLASHINFER_GDN", "false"),
    # Fused gating + width-4 conv + recurrence recipe for Qwen3.8 TP1 decode
    # batches 1/2/4/8/16/32. Follows the parent switch unless set.
    "VLLM_HPU_FLASHINFER_GDN_FUSED_DECODE":
    lambda: _bool_env("VLLM_HPU_FLASHINFER_GDN_FUSED_DECODE", os.environ.get("VLLM_HPU_FLASHINFER_GDN", "false")),
    # Shape-gated FlashQLA graph tactic for single-sequence Qwen3.8 TP1 GDN
    # prefill. Independent of the parent switch.
    "VLLM_HPU_FLASHINFER_GDN_PREFILL":
    lambda: _bool_env("VLLM_HPU_FLASHINFER_GDN_PREFILL", "false"),
    # Mark full decode batches whose requests own compact base slots
    # 0..B-1 as direct-state so GDN layers can use contiguous views instead of
    # index_select/index_copy. Requires VLLM_HPU_FLASHINFER_GDN, compact GDN
    # and prefix caching off.
    "VLLM_HPU_GDN_DIRECT_STATE":
    lambda: _bool_env("VLLM_HPU_GDN_DIRECT_STATE", "true"),
    # Use the calculate_scale_for_cast CGUID for decode-sized per-token dynamic
    # FP8 activation scales. Follows VLLM_HPU_FLASHINFER_GDN unless set.
    "VLLM_HPU_CGUID_DYNAMIC_QUANT":
    lambda: _bool_env("VLLM_HPU_CGUID_DYNAMIC_QUANT", os.environ.get("VLLM_HPU_FLASHINFER_GDN", "false")),
    # Largest flattened row count that uses the CGUID; prefill keeps the
    # abs/amax path.
    "VLLM_HPU_CGUID_DYNAMIC_QUANT_MAX_ROWS":
    lambda: int(os.environ.get("VLLM_HPU_CGUID_DYNAMIC_QUANT_MAX_ROWS", "32")),
    # Quantize an unquantized bf16 lm_head to per-channel FP8 after loading.
    "VLLM_HPU_FP8_LM_HEAD":
    lambda: _bool_env("VLLM_HPU_FP8_LM_HEAD", "false"),
    # Top-k/top-p sampling over the top-N logits instead of sorting the vocabulary, used when every
    # request's top_k <= N; 0 disables (native full sort).
    "VLLM_HPU_SAMPLER_TOPK_CAP":
    lambda: int(os.getenv("VLLM_HPU_SAMPLER_TOPK_CAP", "64")),
    # JSON object merged into the promoted GDN prefill tactic's tuning, e.g. '{"compute_dtype": "bfloat16"}'
    # (A/B of tactic knobs without editing the manifest).
    "VLLM_HPU_GDN_PREFILL_TUNING":
    lambda: os.getenv("VLLM_HPU_GDN_PREFILL_TUNING", ""),
    # GDN decode recurrence on the custom single-pass TPC kernel (tpc/); the SSM state pool becomes KV-major.
    # Needs the fused direct-state decode and the kernel library on GC_KERNEL_PATH.
    "VLLM_HPU_GDN_TPC_DECODE":
    lambda: _bool_env("VLLM_HPU_GDN_TPC_DECODE", "false"),
    # Custom Gaudi2 FP8/GDN/ShortConv TPC kernels (tpc/): fused Add+RMSNorm+FP8, SandwichRMSNorm+FP8,
    # SiluAndMul+FP8, GDN prep/gated-norm, and LFM3 ShortConv decode. Activates only on Gaudi2 when the
    # custom kernel library is on GC_KERNEL_PATH and tpc/torch/gaudi2_fp8_tpc*.so is built.
    "VLLM_HPU_GAUDI2_FP8_TPC":
    lambda: _bool_env("VLLM_HPU_GAUDI2_FP8_TPC", "true"),
    # Optional explicit path to tpc/torch/gaudi2_fp8_tpc*.so.
    "VLLM_HPU_GAUDI2_FP8_TPC_TORCH_LIB":
    lambda: os.environ.get("VLLM_HPU_GAUDI2_FP8_TPC_TORCH_LIB", ""),
    # Smallest flattened token row count using 1-core-per-row fused FP8 norm/SwiGLU TPC kernels.
    # Defaults to 1 when CGUID dynamic quant is enabled (beats CGUID from B=1) and 16 otherwise
    # (when native abs/amax fuses across all 24 TPC cores at small B<=8, while 1-core-per-row wins at B>=16).
    "VLLM_HPU_GAUDI2_FP8_TPC_MIN_ROWS":
    lambda: max(
        1,
        int(
            os.environ.get("VLLM_HPU_GAUDI2_FP8_TPC_MIN_ROWS", "").strip()
            or ("1" if _bool_env("VLLM_HPU_CGUID_DYNAMIC_QUANT", os.environ.get("VLLM_HPU_FLASHINFER_GDN", "false"))
                else "16")
        ),
    ),
    # Largest flattened token row count using custom fused FP8 norm/SwiGLU TPC kernels (0 = unlimited).
    # Prefill buckets (>=128 rows) stay on Synapse's N-sliced SRAM W13+SiluAndMul bundle.
    "VLLM_HPU_GAUDI2_FP8_TPC_MAX_ROWS":
    lambda: int(os.environ.get("VLLM_HPU_GAUDI2_FP8_TPC_MAX_ROWS", "").strip() or "64"),
    # Per-kernel toggles for custom Gaudi2 FP8/GDN/ShortConv TPC ops (require VLLM_HPU_GAUDI2_FP8_TPC=1;
    # VLLM_HPU_TPC_GDN_GATED_NORM_F8 additionally requires VLLM_HPU_TPC_GDN_PREP=1 so gdn_decode outputs FP32).
    "VLLM_HPU_TPC_ADD_RMS_NORM_F8":
    lambda: _bool_env("VLLM_HPU_TPC_ADD_RMS_NORM_F8", "true"),
    "VLLM_HPU_TPC_SANDWICH_RMS_NORM_F8":
    lambda: _bool_env("VLLM_HPU_TPC_SANDWICH_RMS_NORM_F8", "true"),
    "VLLM_HPU_TPC_SILU_MUL_F8":
    lambda: _bool_env("VLLM_HPU_TPC_SILU_MUL_F8", "true"),
    "VLLM_HPU_TPC_GDN_PREP":
    lambda: _bool_env("VLLM_HPU_TPC_GDN_PREP", "true"),
    "VLLM_HPU_TPC_GDN_GATED_NORM_F8":
    lambda: _bool_env("VLLM_HPU_TPC_GDN_GATED_NORM_F8", "true"),
    "VLLM_HPU_TPC_LFM3_SHORT_CONV":
    lambda: _bool_env("VLLM_HPU_TPC_LFM3_SHORT_CONV", "true"),
    # LFM3-VL / LFM3-Omni vision tower on packed bucket shapes (vllm_gaudi/models/lfm3_vl.py); false keeps the
    # upstream tower (one graph per combination of frame sizes). TODO: remove with the fallback in models/__init__.py.
    "VLLM_HPU_LFM3_PACKED_TOWER":
    lambda: _bool_env("VLLM_HPU_LFM3_PACKED_TOWER", "true"),

    # Kimi-K3 routed experts use the SiTU activation (beta*tanh(gate/beta)*sigmoid(gate)*up, soft-clipped up), which
    # the Habana fused MoE op does not have. "exact" computes the experts unfused with the real activation (MXFP4
    # weights dequantized per expert chunk); "silu" maps SiTU to silu in the fused kernel (fast, APPROXIMATE).
    "HPU_KIMI_SITU_IMPL":
    lambda: os.environ.get("HPU_KIMI_SITU_IMPL", "exact").strip().lower(),

    # Kimi-K3 video: shard the video chunks of the vision tower across the TP ranks
    "HPU_KIMI_VIT_DP":
    lambda: os.environ.get("HPU_KIMI_VIT_DP", "0").strip().lower() in ("1", "true", "on", "yes"),

    # Kimi-K3 KDA prefill "<chunk_size>,<vec_chunks>" (see kda_chunked_steps_vec)
    "HPU_KIMI_KDA_PREFILL_CHUNK":
    lambda: os.environ.get("HPU_KIMI_KDA_PREFILL_CHUNK", "64,16"),

    # Kimi-K3 KDA prefill: rows per block of the blockwise intra-chunk A / Aqk ("0" = full pairwise decay)
    "HPU_KIMI_KDA_INTRA_SUB_CHUNK":
    lambda: os.environ.get("HPU_KIMI_KDA_INTRA_SUB_CHUNK", "0"),

    # Kimi-K3 (no prefix caching): let several fresh prompts share one padded prefill batch
    "HPU_KIMI_PREFILL_MERGE":
    lambda: os.environ.get("HPU_KIMI_PREFILL_MERGE", "0").strip().lower() in ("1", "true"),

    # Grouped top-k router: top-k of wide expert rows (>= 512 experts) as a top-k per 128 experts, then over the winners
    "VLLM_HPU_TWO_STAGE_TOPK":
    lambda: os.environ.get("VLLM_HPU_TWO_STAGE_TOPK", "1").strip().lower() in ("1", "true"),

    # Async scheduler: smallest first chunk of a new prefill that may join a step already carrying another prefill
    # (integer, empty or 0 = off)
    "VLLM_HPU_MIN_PREFILL_CHUNK":
    lambda: os.environ.get("VLLM_HPU_MIN_PREFILL_CHUNK", ""),
}

# end-env-vars-definition


def __getattr__(name: str):
    # lazy evaluation of environment variables
    if name in environment_variables:
        return environment_variables[name]()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return list(environment_variables.keys())
