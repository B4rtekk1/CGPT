cmake_minimum_required(VERSION 4.3)
project(CGPT VERSION 1.0 LANGUAGES CXX CUDA)

include(CTest)
enable_testing()
find_package(CUDAToolkit REQUIRED)

# On Windows, keep CUPTI's runtime DLL next to the test executables. Linux
# loads libcupti.so through the CUDA toolkit/loader and needs no copy step.
if (WIN32)
    file(GLOB CGPT_CUPTI_RUNTIME_DLLS
            "${CUDAToolkit_CUPTI_INCLUDE_DIR}/../lib64/cupti64_*.dll")
    if (NOT CGPT_CUPTI_RUNTIME_DLLS)
        message(FATAL_ERROR "Could not locate the CUPTI runtime DLL")
    endif()
    list(GET CGPT_CUPTI_RUNTIME_DLLS 0 CGPT_CUPTI_RUNTIME_DLL)
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
# Keep native cubins for the original Ampere target and Hopper/H100.  The
# runtime launch heuristic selects suitable block geometry for the detected
# device; compiling both targets keeps that selection free of JIT surprises.
set(CMAKE_CUDA_ARCHITECTURES "86-real;90a-real" CACHE STRING "CUDA architectures")

set(CGPT_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)

add_library(cgpt_core STATIC
        include/data/dataset_loader.h
        src/data/dataset_loader.cpp
        src/ops/embedding.cu
        src/ops/cross_entropy.cu
        src/ops/cut_cross_entropy.cu
        src/ops/softmax.cu
        src/ops/linear.cu
        src/ops/backward/linear_backward.cu
        src/ops/backward/rmsnorm_backward.cu
        src/ops/backward/swiglu_backward.cu
        src/ops/backward/attention_backward.cu
        src/ops/backward/embedding_backward.cu
        src/ops/backward/rope_backward.cu
        src/optim/adamw.cu
        src/ops/rmsnorm.cu
        src/ops/swiglu.cu
        src/ops/rope.cu
        src/core/tensor.cu
        src/core/kv_cache.cu
        src/core/transformer_block.cu
        src/core/transformer_model.cu
        src/core/weight_initialization.cu
        src/core/generation.cu
        src/core/cuda_graph.cu
        src/core/huggingface_export.cpp
        src/ops/attention.cu)

# Cross entropy is dominated by exp/log latency over a very wide vocabulary.
# Keep the relaxed intrinsics scoped to this numerically-stable kernel; the
# max-subtracted softmax still bounds exp's input to <= 0.
set_source_files_properties(src/ops/cross_entropy.cu src/ops/softmax.cu PROPERTIES
        COMPILE_OPTIONS "$<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>")

target_include_directories(cgpt_core PUBLIC
        ${CGPT_INCLUDE_DIR}
        ${CGPT_INCLUDE_DIR}/core)
target_link_libraries(cgpt_core PUBLIC CUDA::cudart CUDA::cublasLt)
if (MSVC)
    target_compile_options(cgpt_core PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/Zc:preprocessor>)
endif()
set_target_properties(cgpt_core PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(cgpt_core PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)

add_library(cgpt_tokenizer STATIC
        include/tokenizer/bpe_tokenizer.h
        include/utils/progress_bar.h
        src/tokenizer/bpe_tokenizer.cpp
        src/utils/progress_bar.cpp)
target_include_directories(cgpt_tokenizer PUBLIC ${CGPT_INCLUDE_DIR})
target_link_libraries(cgpt_core PUBLIC cgpt_tokenizer)

if (MSVC AND CMAKE_SIZEOF_VOID_P EQUAL 8)
    set_source_files_properties(src/tokenizer/bpe_tokenizer.cpp PROPERTIES
            COMPILE_OPTIONS "/arch:AVX2")
endif()

function(add_cgpt_test name source dependency)
    add_executable(${name} ${source} tests/cuda_kernel_profiler.cpp)
    target_link_libraries(${name} PRIVATE ${dependency} CUDA::cudart CUDA::cupti)
    if (WIN32)
        add_custom_command(TARGET ${name} POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                        "${CGPT_CUPTI_RUNTIME_DLL}" $<TARGET_FILE_DIR:${name}>)
    endif()
    add_test(NAME ${name} COMMAND ${name})
    set_target_properties(${name} PROPERTIES
            CUDA_SEPARABLE_COMPILATION ON
            CUDA_RESOLVE_DEVICE_SYMBOLS ON)
endfunction()

add_cgpt_test(rmsnorm_test tests/rmsnorm_test.cu cgpt_core)
add_cgpt_test(linear_test tests/linear_test.cu cgpt_core)
add_cgpt_test(linear_backward_test tests/linear_backward_test.cu cgpt_core)
add_cgpt_test(swiglu_test tests/swiglu_test.cu cgpt_core)
add_cgpt_test(rope_test tests/rope_test.cu cgpt_core)
add_cgpt_test(tensor_test tests/tensor_test.cu cgpt_core)
add_cgpt_test(device_buffer_test tests/device_buffer_test.cu cgpt_core)
add_cgpt_test(dtype_test tests/dtype_test.cu cgpt_core)
add_cgpt_test(flash_attention_test tests/flash_attention_test.cu cgpt_core)
add_cgpt_test(embedding_test tests/embedding_test.cu cgpt_core)
add_cgpt_test(cross_entropy_test tests/cross_entropy_test.cu cgpt_core)
add_cgpt_test(cut_cross_entropy_test tests/cut_cross_entropy_test.cu cgpt_core)
add_cgpt_test(softmax_test tests/softmax_test.cu cgpt_core)
add_cgpt_test(cuda_graph_test tests/cuda_graph_test.cu cgpt_core)
add_cgpt_test(cuda_benchmark_test tests/cuda_benchmark_test.cu cgpt_core)
add_cgpt_test(kv_cache_test tests/kv_cache_test.cu cgpt_core)
add_cgpt_test(transformer_block_test tests/transformer_block_test.cu cgpt_core)
add_cgpt_test(transformer_model_test tests/transformer_model_test.cu cgpt_core)
add_cgpt_test(adamw_test tests/adamw_test.cu cgpt_core)
add_cgpt_test(dataset_loader_test tests/dataset_loader_test.cpp cgpt_core)
add_cgpt_test(bpe_tokenizer_test tests/bpe_tokenizer_test.cpp cgpt_tokenizer)
add_cgpt_test(bpe_tokenizer_100mb_test tests/bpe_tokenizer_100mb_test.cpp cgpt_tokenizer)

add_executable(run_all_tests tests/run_all_tests.cpp)
add_dependencies(run_all_tests
        rmsnorm_test linear_test linear_backward_test swiglu_test rope_test tensor_test device_buffer_test dtype_test
        flash_attention_test embedding_test cross_entropy_test cut_cross_entropy_test softmax_test cuda_graph_test cuda_benchmark_test kv_cache_test transformer_block_test transformer_model_test adamw_test
        dataset_loader_test bpe_tokenizer_test bpe_tokenizer_100mb_test)
if (WIN32)
    add_custom_target(run_all
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${CGPT_CUPTI_RUNTIME_DLL}" $<TARGET_FILE_DIR:run_all_tests>)
else()
    add_custom_target(run_all)
endif()

set(CGPT_RUN_ALL_ARGUMENTS
                rmsnorm_test $<TARGET_FILE:rmsnorm_test>
                linear_test $<TARGET_FILE:linear_test>
                linear_backward_test $<TARGET_FILE:linear_backward_test>
                swiglu_test $<TARGET_FILE:swiglu_test>
                rope_test $<TARGET_FILE:rope_test>
                tensor_test $<TARGET_FILE:tensor_test>
                device_buffer_test $<TARGET_FILE:device_buffer_test>
                dtype_test $<TARGET_FILE:dtype_test>
                flash_attention_test $<TARGET_FILE:flash_attention_test>
                embedding_test $<TARGET_FILE:embedding_test>
                cross_entropy_test $<TARGET_FILE:cross_entropy_test>
                cut_cross_entropy_test $<TARGET_FILE:cut_cross_entropy_test>
                softmax_test $<TARGET_FILE:softmax_test>
                cuda_graph_test $<TARGET_FILE:cuda_graph_test>
                cuda_benchmark_test $<TARGET_FILE:cuda_benchmark_test>
                kv_cache_test $<TARGET_FILE:kv_cache_test>
                transformer_block_test $<TARGET_FILE:transformer_block_test>
                transformer_model_test $<TARGET_FILE:transformer_model_test>
                adamw_test $<TARGET_FILE:adamw_test>
                dataset_loader_test $<TARGET_FILE:dataset_loader_test>
                bpe_tokenizer_test $<TARGET_FILE:bpe_tokenizer_test>
        bpe_tokenizer_100mb_test $<TARGET_FILE:bpe_tokenizer_100mb_test>)
set_property(TARGET run_all APPEND PROPERTY COMMAND
        $<TARGET_FILE:run_all_tests> ${CGPT_RUN_ALL_ARGUMENTS})
add_dependencies(run_all run_all_tests)
set_property(TARGET run_all PROPERTY USES_TERMINAL TRUE)

find_package(Python3 COMPONENTS Interpreter QUIET)
if (Python3_Interpreter_FOUND)
    add_custom_target(profile_kernels
            COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/scripts/profile_kernels.py
                    --build-dir ${CMAKE_BINARY_DIR}
                    --output ${CMAKE_CURRENT_SOURCE_DIR}/kernel_profile.md
            DEPENDS run_all_tests
            USES_TERMINAL
            VERBATIM
            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
            COMMENT "Running cross-platform CUDA kernel profiler")
endif()

set_tests_properties(
        rmsnorm_test linear_test linear_backward_test swiglu_test rope_test tensor_test device_buffer_test dtype_test flash_attention_test embedding_test cross_entropy_test softmax_test cuda_graph_test cuda_benchmark_test kv_cache_test transformer_block_test transformer_model_test adamw_test dataset_loader_test bpe_tokenizer_test
        PROPERTIES TIMEOUT 30)

# The cross-entropy benchmark processes a GPT-2-sized vocabulary and needs
# slightly longer than the standard test timeout on supported GPUs.
set_tests_properties(cross_entropy_test softmax_test PROPERTIES TIMEOUT 60)

set_tests_properties(bpe_tokenizer_100mb_test PROPERTIES
        TIMEOUT 900
        SKIP_RETURN_CODE 77
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

add_executable(cgpt_train train/train.cpp)
target_link_libraries(cgpt_train PRIVATE cgpt_core cgpt_tokenizer CUDA::cudart)
set_target_properties(cgpt_train PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON)
