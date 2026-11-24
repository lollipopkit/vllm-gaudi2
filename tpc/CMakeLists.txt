# SPDX-License-Identifier: Apache-2.0
# Custom Gaudi2 TPC kernels: libgaudi2_tpc_kernels.so (GC_KERNEL_PATH) and simulator tests.
# Needs habanatools (tpc-clang, tpc test core) under /usr/bin and /usr/lib/habanatools.
cmake_minimum_required(VERSION 3.16)
project(gaudi2_tpc_kernels CXX)

set(TPC_CLANG /usr/bin/tpc-clang CACHE FILEPATH "TPC-C compiler")
set(TPC_OBJDUMP /usr/bin/tpc-llvm-objdump CACHE FILEPATH "TPC disassembler")
set(HABANATOOLS /usr/lib/habanatools CACHE PATH "habanatools libraries and headers")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Werror -fno-strict-aliasing")
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(TPC_KERNEL_HEADERS
    ${CMAKE_CURRENT_SOURCE_DIR}/kernels/inkling2_bf16_common.h
    ${CMAKE_CURRENT_SOURCE_DIR}/kernels/inkling2_attn_res_common.h
    ${CMAKE_CURRENT_SOURCE_DIR}/kernels/inkling2_blend_rms_norm_common.h
    ${CMAKE_CURRENT_SOURCE_DIR}/kernels/inkling2_paged_attn_common.h
    ${CMAKE_CURRENT_SOURCE_DIR}/kernels/inkling2_qkvr_prep_decode_common.h)

# TPC-C source -> TPC ELF -> x86 object exposing _binary___<name>_o_{start,end} for the glue code.
function(add_tpc_kernel name)
  set(src ${CMAKE_CURRENT_SOURCE_DIR}/kernels/${name}.c)
  add_custom_command(
    OUTPUT ${name}_x86.o ${name}.tpcasm
    DEPENDS ${src} ${TPC_KERNEL_HEADERS}
    COMMAND ${TPC_CLANG} -Wall -Werror -march=gaudi2 -O2 -I${HABANATOOLS}/include
            -I${CMAKE_CURRENT_SOURCE_DIR}/kernels ${src} -c -o ${name}.o
    COMMAND objcopy -I binary -O elf64-x86-64 -B i386:x86-64 ./${name}.o ${name}_x86.o
    COMMAND ${TPC_OBJDUMP} --triple tpc -d -j .text --no-show-raw-insn --no-leading-addr --mcpu=gaudi2
            ${name}.o > ${name}.tpcasm
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
  set(TPC_OBJECTS ${TPC_OBJECTS} ${CMAKE_CURRENT_BINARY_DIR}/${name}_x86.o PARENT_SCOPE)
endfunction()

add_tpc_kernel(gdn_decode_f32_gaudi2)
add_tpc_kernel(gdn_verify_f32_gaudi2)
add_tpc_kernel(inkling2_moe_w13_situ_f8_gaudi2)
add_tpc_kernel(inkling2_moe_w2_f8_gaudi2)
add_tpc_kernel(inkling2_moe_group_experts_gaudi2)
add_tpc_kernel(inkling2_moe_w13_grouped_f8_gaudi2)
add_tpc_kernel(inkling2_moe_w2_grouped_f8_gaudi2)
add_tpc_kernel(inkling2_moe_gate_topk_gaudi2)
add_tpc_kernel(inkling2_moe_gate_topk_e128_gaudi2)
add_tpc_kernel(inkling2_moe_gate_gemv_splitk_gaudi2)
add_tpc_kernel(inkling2_moe_gate_gemv_splitk_b1_gaudi2)
add_tpc_kernel(inkling2_moe_sum_sconv_res_decode_gaudi2)
add_tpc_kernel(inkling2_attn_res_norm_bf16_gaudi2)
add_tpc_kernel(inkling2_attn_res_norm_f8_gaudi2)
add_tpc_kernel(inkling2_attn_res_f32_gaudi2)
add_tpc_kernel(inkling2_sconv_static_decode_gaudi2)
add_tpc_kernel(inkling2_qkvr_prep_decode_gaudi2)
add_tpc_kernel(inkling2_qkvr_prep_decode_dyn_gaudi2)
add_tpc_kernel(inkling2_pair_rms_norm_gaudi2)
add_tpc_kernel(inkling2_blend_rms_norm_gaudi2)
add_tpc_kernel(inkling2_blend_rms_norm_f8_gaudi2)
add_tpc_kernel(inkling2_rel_paged_attn_gate_sink_gaudi2)
add_tpc_kernel(inkling2_rel_paged_attn_gqa3_gaudi2)

set(GLUE_SOURCES
    src/gdn_decode_gaudi2.cpp
    src/gdn_verify_gaudi2.cpp
    src/inkling2_moe_gaudi2.cpp
    src/inkling2_attn_res_gaudi2.cpp
    src/inkling2_sconv_norm_gaudi2.cpp
    src/inkling2_paged_attn_gaudi2.cpp)

add_custom_target(tpc_elfs DEPENDS ${TPC_OBJECTS})

add_library(gaudi2_tpc_kernels SHARED src/entry_points.cpp ${GLUE_SOURCES} ${TPC_OBJECTS})
add_dependencies(gaudi2_tpc_kernels tpc_elfs)
target_include_directories(gaudi2_tpc_kernels PUBLIC src ${HABANATOOLS}/include)
target_link_options(gaudi2_tpc_kernels PRIVATE -Wl,-z,noexecstack)

# Single-file GC_KERNEL_PATH: our kernels + the stock perf library (forwarded), so the recipe cache works.
add_library(gaudi2_tpc_proxy SHARED src/proxy_entry_points.cpp ${GLUE_SOURCES} ${TPC_OBJECTS})
add_dependencies(gaudi2_tpc_proxy tpc_elfs)
target_include_directories(gaudi2_tpc_proxy PRIVATE src ${HABANATOOLS}/include)
target_link_libraries(gaudi2_tpc_proxy dl)
target_link_options(gaudi2_tpc_proxy PRIVATE -Wl,-z,noexecstack)
set_source_files_properties(${TPC_OBJECTS} PROPERTIES EXTERNAL_OBJECT TRUE GENERATED TRUE)

function(add_tpc_sim_test target src)
  add_executable(${target} ${src} tests/habana/float16.cpp)
  target_include_directories(${target} PRIVATE tests)
  target_compile_options(${target} PRIVATE -Wno-error)  # vendored habana test helpers
  target_link_libraries(${target} gaudi2_tpc_kernels ${HABANATOOLS}/libtpc_tests_core_ext.so
                        ${HABANATOOLS}/libtpcsim_shared.so ${HABANATOOLS}/libTpcElfReader.so)
  set_target_properties(${target} PROPERTIES BUILD_RPATH ${HABANATOOLS})
endfunction()

add_tpc_sim_test(gdn_decode_test tests/gdn_decode_test.cpp)
add_tpc_sim_test(inkling2_moe_test tests/inkling2_moe_test.cpp)
add_tpc_sim_test(inkling2_attn_res_test tests/inkling2_attn_res_test.cpp)
add_tpc_sim_test(inkling2_ops_test tests/inkling2_ops_test.cpp)
