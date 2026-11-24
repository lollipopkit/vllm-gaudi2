# vllm-gaudi2

Optimized vLLM hardware plugin for **Intel Gaudi2 (`HL-225`, 96 GiB HBM2e, 2.45 TB/s)**.

Targets **Gaudi2 only** (not tested on Gaudi1 or Gaudi3). The Python package name remains `vllm_gaudi` for vLLM plugin discovery and upstream compatibility.

> **Testbed & 350 W Power Cap**: All benchmarks below were measured on a single `HL-225` (PCIe Gen3 x16) **hard-capped at 350 W** (stock TDP: 600 W; compute-bound stages such as prefill and high-concurrency decode run power-limited), Ubuntu 24.04, SynapseAI 1.24.2, PyTorch 2.13+hpu, vLLM `v0.34.0`.

## Benchmarks (1× HL-225, **350 W cap**)

| Model | Precision | Performance | Quality / Key Optimizations |
|---|---|---|---|
| **Inkling 2 Mini** (26.4B Looped MoE + 23-layer ViT) | FP8 (`inkling2_fp8`) | `133.1 tok/s` (`7.51 ms`, `c=1`) · `316.1` (`c=4`) · `488.2` (`c=8`) · `431.2` (`c=8` diverse) | `25.55 GiB` HBM, 17 custom TPC-C kernels (`2.87×` at `c=1`, `2.65×` at `c=8` vs. stock TPC), 6-layer decode groups, native FP8→FP32 `lm_head` |
| **Qwen3.8-27B** (48 GDN + 16 GQA) | FP8 + FP8 KV | `62.9–63.7 tok/s` (`15.7–15.9 ms`, `b=1`) · `240–245` out / `2,211–2,248` total (`c=8`) · `322.7` out / `2,970` total (`c=32`) | GSM8K `95.3%`, NLL `+0.0012` vs BF16, custom GDN TPC kernels, FlashQLA prefill, MTP / DFlash2 |
| **LiquidAI LFM3** (`1.7B` / `VL-2B` / `Omni-2B`) | FP8 (`e4m3fn`) | `180.5 tok/s` (`b=1`) · `1,257` (`b=8`) · `1,380` out / `12,478` total (`c=32`, vendor sampling) | GSM8K `76.7%`, MMStar `74.4%`, LibriSpeech WER `2.62%`, `HPUShortConv`, packed vision tower |

### Comparison vs. Upstream `vllm-project/vllm-gaudi`

Measured on the same single `HL-225` (`350 W` cap) with `vllm bench serve` (`2048` input / `256` output tokens) on **Qwen3.8-27B-FP8**:

| Metric | Upstream `vllm-gaudi` | `vllm-gaudi2` | Speedup |
|---|---:|---:|---:|
| `c=1` decode speed (`1 / TPOT`) | `33.5 tok/s` (`29.84 ms`) | `62.9–63.7 tok/s` (`15.7–15.9 ms`) | **1.88–1.90×** |
| `c=1` serve output throughput | `31.19 tok/s` | `57.7 tok/s` | **1.85×** |
| `c=8` serve output / total throughput | `113.8` / `1,048 tok/s` (`58.48 ms` TPOT) | `240–245` / `2,211–2,248 tok/s` (`24.8–25.5 ms` TPOT) | **2.11–2.15×** |
| `c=32` serve output / total throughput | `145.0` / `1,335 tok/s` (`166.4 ms` TPOT) | `322.7` / `2,970 tok/s` | **2.22×** |
| **Inkling 2 Mini** / **LiquidAI LFM3** | Not supported | Supported (FP8 + custom TPC / HPU backends) | — |

> The upstream `vllm-gaudi` baseline was recorded on the same hardware prior to the PyTorch 2.13 / `v0.34.0` upgrade (PyTorch 2.9, default prefix caching enabled); `vllm-gaudi2` runs with `--no-enable-prefix-caching` and `--kv-cache-dtype fp8_inc` as shown in Quick Start.

## Key Features

- **19 Custom Gaudi2 TPC-C Kernels ([`tpc/`](tpc/))**: `gdn_decode` / `gdn_verify` for Qwen3.8 plus 17 Inkling2 decode kernels (MoE gate/grouping/FP8 GEMV, paged relative-bias attention, Block AttnRes, short-conv), bundled with stock `libtpc_kernels.so` inside single-file `libgaudi2_tpc_proxy.so` (preserves Synapse recipe cache), PyTorch C++ ops (`tpc/torch/`), and simulator test suites (`tpc/tests/`).
- **Out-of-Tree Model Plugins (`vllm_gaudi/models/`)**: Full HPU backends for Inkling 2 Mini (`inkling2/`), LiquidAI LFM3 / VL / Omni (`lfm3*.py`), and DFlash2 (`qwen3_dflash2.py`), registered through vLLM plugin entry points and guarded runtime hooks (`vllm_gaudi/patches.py`).
- **Decode & Runtime Optimizations**: Multi-layer compiled decode groups (`layer_groups.py`), FlashQLA GDN prefill (`flashinfer_gaudi/`), native FP8→FP32 `lm_head` (`VLLM_HPU_FP8_LM_HEAD=1`), `int32` paged-attention indices, symbolic `fp8_gemm_v2` meta shapes, and zero-redundant-DMA decode (`13 → 4` H2D DMAs/step).
- **Offline Weight Converter ([`tools/convert_inkling2_fp8.py`](tools/convert_inkling2_fp8.py))**: Streams Inkling 2 Mini NVFP4/MXFP8 routed experts to per-channel Gaudi2 FP8 (`max=240.0`).

## Installation

Install `vllm` at tag `v0.34.0` while pinning your installed Habana PyTorch packages, then install this plugin:

```bash
# 0. Pin existing Habana PyTorch packages so pip does not overwrite them
PINS=$(mktemp)
pip list --format=freeze | grep -iE '^(torch|habana|neural)' > "$PINS"

# 1. vLLM v0.34.0 (empty target)
git clone --branch v0.34.0 --depth 1 https://github.com/vllm-project/vllm.git
cd vllm
pip install -r <(sed '/^torch/d' requirements/build/cuda.txt) -c "$PINS"
VLLM_TARGET_DEVICE=empty pip install --no-build-isolation -e . -c "$PINS"
cd ..

# 2. This plugin
git clone https://github.com/lollipopkit/vllm-gaudi2.git
cd vllm-gaudi2
pip install --no-build-isolation -e . -c "$PINS"
rm -f "$PINS"

# 3. Optional: build custom Gaudi2 TPC-C kernels & PyTorch custom ops (requires habanatools)
cmake -S tpc -B tpc/build && make -C tpc/build -j
(cd tpc/torch && python setup.py build_ext --inplace)
```

## Quick Start

```bash
# Qwen3.8-27B-FP8 (compact GDN + direct-state TPC decode + FP8 KV)
VLLM_HPU_FLASHINFER_GDN=1 VLLM_HPU_FP8_LM_HEAD=1 \
VLLM_HPU_GDN_TPC_DECODE=1 VLLM_HPU_QWEN3_COMPILE_LAYER_GROUP_SIZE=4 \
GC_KERNEL_PATH=$PWD/tpc/build/libgaudi2_tpc_proxy.so \
vllm serve /models/Qwen3.8-27B-FP8 --kv-cache-dtype fp8_inc --no-enable-prefix-caching --port 8000

# LiquidAI LFM3-1.7B / VL-2B / Omni-2B
VLLM_HPU_LFM3_COMPILE_LAYER_GROUP_SIZE=4 \
vllm serve /models/LFM3-1.7B-FP8 --trust-remote-code --no-enable-prefix-caching --port 8001

# Thinking Machines Inkling 2 Mini (convert NVFP4/MXFP8 -> Gaudi2 FP8 once)
python tools/convert_inkling2_fp8.py /models/Inkling-2-Mini /models/Inkling-2-Mini-FP8
VLLM_HPU_FP8_LM_HEAD=1 INKLING2_FP8_ATTN=1 INKLING2_COMPILE_GROUP_SIZE=6 \
GC_KERNEL_PATH=$PWD/tpc/build/libgaudi2_tpc_proxy.so \
vllm serve /models/Inkling-2-Mini-FP8 --no-enable-prefix-caching --port 8002
```

## Key Environment Variables

| Variable | Default | Purpose |
|---|---|---|
| `GC_KERNEL_PATH` | stock | Point to `tpc/build/libgaudi2_tpc_proxy.so` to enable custom TPC kernels with recipe caching |
| `VLLM_HPU_FP8_LM_HEAD` | `0` | Quantize `lm_head` to per-channel FP8 post-load (supports direct FP8→FP32 logits) |
| `VLLM_HPU_FLASHINFER_GDN` | `0` | Enable FlashInfer-Gaudi GDN decode and FlashQLA prefill for Qwen3.8 |
| `VLLM_HPU_GDN_TPC_DECODE` | `0` | Enable custom `gdn_decode` / `gdn_verify` TPC kernels for Qwen3.8 |
| `VLLM_HPU_QWEN3_COMPILE_LAYER_GROUP_SIZE` | `1` | Static decode layer group size for Qwen3.8 (recommended: `4`) |
| `VLLM_HPU_LFM3_COMPILE_LAYER_GROUP_SIZE` | `1` | Static decode layer group size for LFM3 (recommended: `4`) |
| `INKLING2_FP8_ATTN` | `1` | Quantize Inkling2 attention `qkvr` / `wo_ud` projections to FP8 at load time |
| `INKLING2_COMPILE_GROUP_SIZE` | `6` | Static decode layer group size for Inkling2 (8 groups across 48 visits) |
| `VLLM_FULL_WARMUP` | `0` | Warm all buckets then disable Dynamo guard checks during decode |

See [`docs/configuration/env_variables.md`](docs/configuration/env_variables.md) for all upstream `vllm_gaudi` flags.

## Tests

```bash
# Python unit tests (set VLLM_USE_FAKE_HPU=1 to run CPU-only tests without a Gaudi2 card)
VLLM_USE_FAKE_HPU=1 pytest -q \
  tests/unit_tests/test_plugin_migrations.py \
  tests/unit_tests/vendored_v1/test_imports.py \
  tests/unit_tests/spec_decode/test_hpu_dflash.py \
  tools/test_convert_inkling2_fp8.py \
  tools/test_inkling2_hpu_ops.py

# HPU vision tower test (requires Gaudi2)
pytest -q tests/unit_tests/test_hpu_lfm3_vl_tower.py

# Gaudi2 TPC simulator suites (requires habanatools)
LD_LIBRARY_PATH=/usr/lib/habanatools tpc/build/gdn_decode_test 6 2
LD_LIBRARY_PATH=/usr/lib/habanatools tpc/build/inkling2_attn_res_test
LD_LIBRARY_PATH=/usr/lib/habanatools tpc/build/inkling2_ops_test
LD_LIBRARY_PATH=/usr/lib/habanatools tpc/build/inkling2_moe_test
```

## License

Licensed under [Apache-2.0](LICENSE). See [`NOTICE`](NOTICE) for third-party attributions (`vllm-project/vllm-gaudi`, `1Cat-vLLM-Gaudi`, and Habana Labs BSD-2-Clause TPC test helpers).
