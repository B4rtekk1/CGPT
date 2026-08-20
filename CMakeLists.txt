cmake_minimum_required(VERSION 4.3)
project(CGPT VERSION 1.0 LANGUAGES CXX CUDA)
include(CTest)
enable_testing()
find_package(CUDAToolkit REQUIRED)
find_package(OpenMP)
if (WIN32)
    file(GLOB CGPT_CUPTI_RUNTIME_DLLS "${CUDAToolkit_CUPTI_INCLUDE_DIR}/../lib64/cupti64_*.dll")
    if (NOT CGPT_CUPTI_RUNTIME_DLLS)
        message(FATAL_ERROR "Could not locate the CUPTI runtime DLL")
    endif()
    list(GET CGPT_CUPTI_RUNTIME_DLLS 0 CGPT_CUPTI_RUNTIME_DLL)
endif()
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES "86-real;90a-real" CACHE STRING "CUDA architectures" FORCE)
set(CGPT_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)
add_library(cgpt_core STATIC
        include/data/dataset_loader.h src/data/dataset_loader.cpp
        src/ops/embedding.cu src/ops/cross_entropy.cu src/ops/cut_cross_entropy.cu src/ops/softmax.cu src/ops/linear.cu
        src/ops/backward/linear_backward.cu src/ops/backward/rmsnorm_backward.cu src/ops/backward/swiglu_backward.cu
        src/ops/backward/attention_backward.cu src/ops/backward/embedding_backward.cu src/ops/backward/rope_backward.cu
        src/optim/adamw.cu src/ops/rmsnorm.cu src/ops/swiglu.cu src/ops/rope.cu src/core/tensor.cu src/core/kv_cache.cu
        src/core/transformer_block.cu src/core/transformer_model.cu src/core/weight_initialization.cu src/core/generation.cu
        src/core/cuda_graph.cu src/core/huggingface_export.cpp src/ops/attention.cu
        src/ops/cpu/swiglu_cpu.cpp src/ops/cpu/softmax_cpu.cpp)
set_source_files_properties(src/ops/cross_entropy.cu src/ops/softmax.cu PROPERTIES COMPILE_OPTIONS "$<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>")
target_include_directories(cgpt_core PUBLIC ${CGPT_INCLUDE_DIR} ${CGPT_INCLUDE_DIR}/core)
target_link_libraries(cgpt_core PUBLIC CUDA::cudart CUDA::cublasLt)
if (OpenMP_CXX_FOUND)
    target_link_libraries(cgpt_core PUBLIC OpenMP::OpenMP_CXX)
endif()
if (MSVC)
    target_compile_options(cgpt_core PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/arch:AVX2>)
else()
    target_compile_options(cgpt_core PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-mavx2>
        $<$<COMPILE_LANGUAGE:CXX>:-mfma>
        $<$<COMPILE_LANGUAGE:CXX>:-mf16c>)
endif()
if (MSVC)
    target_compile_options(cgpt_core PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/Zc:preprocessor>)
endif()
set_target_properties(cgpt_core PROPERTIES CUDA_SEPARABLE_COMPILATION ON CUDA_RESOLVE_DEVICE_SYMBOLS ON)
add_library(cgpt_tokenizer STATIC include/tokenizer/bpe_tokenizer.h include/utils/progress_bar.h src/tokenizer/bpe_tokenizer.cpp src/utils/progress_bar.cpp)
target_include_directories(cgpt_tokenizer PUBLIC ${CGPT_INCLUDE_DIR})
target_link_libraries(cgpt_core PUBLIC cgpt_tokenizer)
if (MSVC AND CMAKE_SIZEOF_VOID_P EQUAL 8)
    set_source_files_properties(src/tokenizer/bpe_tokenizer.cpp PROPERTIES COMPILE_OPTIONS "/arch:AVX2")
endif()
function(add_cgpt_test name source dependency)
    add_executable(${name} ${source} tests/cuda_kernel_profiler.cpp)
    target_link_libraries(${name} PRIVATE ${dependency} CUDA::cudart CUDA::cupti)
    if (WIN32)
        add_custom_command(TARGET ${name} POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy_if_different "${CGPT_CUPTI_RUNTIME_DLL}" $<TARGET_FILE_DIR:${name}>)
    endif()
    add_test(NAME ${name} COMMAND ${name})
    set_target_properties(${name} PROPERTIES CUDA_SEPARABLE_COMPILATION ON CUDA_RESOLVE_DEVICE_SYMBOLS ON)
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
add_cgpt_test(fp16_numerical_stability_test tests/fp16_numerical_stability_test.cu cgpt_core)
add_cgpt_test(dataset_loader_test tests/dataset_loader_test.cpp cgpt_core)
add_cgpt_test(bpe_tokenizer_test tests/bpe_tokenizer_test.cpp cgpt_tokenizer)
add_cgpt_test(bpe_tokenizer_100mb_test tests/bpe_tokenizer_100mb_test.cpp cgpt_tokenizer)
add_executable(run_all_tests tests/run_all_tests.cpp)
add_dependencies(run_all_tests rmsnorm_test linear_test linear_backward_test swiglu_test rope_test tensor_test device_buffer_test dtype_test flash_attention_test embedding_test cross_entropy_test cut_cross_entropy_test softmax_test cuda_graph_test cuda_benchmark_test kv_cache_test transformer_block_test transformer_model_test adamw_test fp16_numerical_stability_test dataset_loader_test bpe_tokenizer_test bpe_tokenizer_100mb_test)
add_executable(cgpt_train train/train.cpp)
target_link_libraries(cgpt_train PRIVATE cgpt_core cgpt_tokenizer CUDA::cudart)
set_target_properties(cgpt_train PROPERTIES CUDA_SEPARABLE_COMPILATION ON CUDA_RESOLVE_DEVICE_SYMBOLS ON)
add_executable(cgpt_cli src/cli.cpp)
target_link_libraries(cgpt_cli PRIVATE cgpt_core cgpt_tokenizer CUDA::cudart)
set_target_properties(cgpt_cli PROPERTIES CUDA_SEPARABLE_COMPILATION ON CUDA_RESOLVE_DEVICE_SYMBOLS ON)

add_custom_target(run_all
        COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure
        DEPENDS run_all_tests
        USES_TERMINAL)

set_tests_properties(
        rmsnorm_test linear_test linear_backward_test swiglu_test rope_test tensor_test device_buffer_test dtype_test
        flash_attention_test embedding_test cross_entropy_test cut_cross_entropy_test softmax_test cuda_graph_test
        cuda_benchmark_test kv_cache_test transformer_block_test transformer_model_test adamw_test
        fp16_numerical_stability_test dataset_loader_test bpe_tokenizer_test
        PROPERTIES TIMEOUT 30)
set_tests_properties(cross_entropy_test softmax_test PROPERTIES TIMEOUT 60)
set_tests_properties(bpe_tokenizer_100mb_test PROPERTIES TIMEOUT 900 SKIP_RETURN_CODE 77
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
