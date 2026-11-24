# Gaudi2 custom TPC kernels

`gdn_decode_f32_gaudi2`: single-token Gated DeltaNet decode step on the f32 recurrent state (Qwen3.8 shapes),
reading and writing the state once. The state is KV-major (`[B, HV, K, V]`, value index fastest) so the
projections are lane-parallel MACs instead of cross-lane reductions.

Build (needs `habanatools`: `/usr/bin/tpc-clang`, `/usr/lib/habanatools`):

    cmake -S tpc -B tpc/build && make -C tpc/build -j
    LD_LIBRARY_PATH=/usr/lib/habanatools tpc/build/gdn_decode_test 6 2   # simulator; TPC_RUNNER=1 for the device
    (cd tpc/torch && python setup.py build_ext --inplace)                # torch.ops.custom_op.gdn_decode

Run with `GC_KERNEL_PATH=<repo>/tpc/build/libgaudi2_tpc_kernels.so:/usr/lib/habanalabs/libtpc_kernels.so`.
`tests/habana/` holds Habana's BSD-licensed test helpers from HabanaAI/Habana_Custom_Kernel.
