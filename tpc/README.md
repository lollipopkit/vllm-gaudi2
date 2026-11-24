# Gaudi2 Custom TPC-C Kernels

19 custom Gaudi2 TPC-C kernels packaged into `libgaudi2_tpc_proxy.so` (wraps `/usr/lib/habanalabs/libtpc_kernels.so` with full layout forwarding behind a single `GC_KERNEL_PATH` file so Synapse recipe caching works):

- **Qwen3.8 GDN (2 GUIDs)**: `gdn_decode_f32_gaudi2`, `gdn_verify_f32_gaudi2` (single-pass recurrent state update on KV-major `[B, HV, K, V]` pool).
- **Inkling 2 Mini (17 GUIDs)**: split-K BF16 MoE gate GEMV + top-k, 4-bucket expert grouping + FP8 `w13`/`w2` GEMV, fused MoE sum + short-conv epilogue, `qkvr_prep`, `sconv_static`, paged relative-bias attention + gate-sink (with GQA-3 shared-K/V fast path), Block AttnRes (`f32`/`bf16`/`f8`), and loop blend/pair RMSNorm kernels.

## Build & Test

Requires `habanatools` (`/usr/bin/tpc-clang`, `/usr/lib/habanatools`):

```bash
cmake -S tpc -B tpc/build && make -C tpc/build -j
(cd tpc/torch && python setup.py build_ext --inplace)

# Run simulator tests (pass TPC_RUNNER=1 to run on Gaudi2 hardware)
LD_LIBRARY_PATH=/usr/lib/habanatools tpc/build/gdn_decode_test 6 2
LD_LIBRARY_PATH=/usr/lib/habanatools tpc/build/inkling2_attn_res_test
LD_LIBRARY_PATH=/usr/lib/habanatools tpc/build/inkling2_ops_test
LD_LIBRARY_PATH=/usr/lib/habanatools tpc/build/inkling2_moe_test
```

Use `GC_KERNEL_PATH=<repo>/tpc/build/libgaudi2_tpc_proxy.so` at runtime.

`tests/habana/` contains Habana Labs BSD-2-Clause test helpers from `HabanaAI/Habana_Custom_Kernel`.
