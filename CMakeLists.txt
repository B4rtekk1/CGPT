cmake_minimum_required(VERSION 4.3)
project(CGPT VERSION 1.0 LANGUAGES CXX CUDA)

include(CTest)
enable_testing()
find_package(CUDAToolkit REQUIRED)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES 86-real CACHE STRING "CUDA architectures")

set(CGPT_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)

add_library(cgpt_core STATIC
        src/ops/linear.cu
        src/ops/rmsnorm.cu
        src/ops/swiglu.cu
        src/ops/rope.cu
        src/tensor.cu
        src/ops/flash_attention.cu)

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

if (MSVC AND CMAKE_SIZEOF_VOID_P EQUAL 8)
    set_source_files_properties(src/tokenizer/bpe_tokenizer.cpp PROPERTIES
            COMPILE_OPTIONS "/arch:AVX2")
endif()

add_executable(CGPT src/main.cu)
target_link_libraries(CGPT PRIVATE cgpt_core cgpt_tokenizer)

function(add_cgpt_test name source dependency)
    add_executable(${name} ${source})
    target_link_libraries(${name} PRIVATE ${dependency})
    add_test(NAME ${name} COMMAND ${name})
    set_target_properties(${name} PROPERTIES
            CUDA_SEPARABLE_COMPILATION ON
            CUDA_RESOLVE_DEVICE_SYMBOLS ON)
endfunction()

add_cgpt_test(rmsnorm_test tests/rmsnorm_test.cu cgpt_core)
add_cgpt_test(linear_test tests/linear_test.cu cgpt_core)
add_cgpt_test(swiglu_test tests/swiglu_test.cu cgpt_core)
add_cgpt_test(rope_test tests/rope_test.cu cgpt_core)
add_cgpt_test(tensor_test tests/tensor_test.cu cgpt_core)
add_cgpt_test(device_buffer_test tests/device_buffer_test.cu cgpt_core)
add_cgpt_test(gpu_buffer_test tests/gpu_buffer_test.cu cgpt_core)
add_cgpt_test(dtype_test tests/dtype_test.cu cgpt_core)
add_cgpt_test(flash_attention_test tests/flash_attention_test.cu cgpt_core)
add_cgpt_test(bpe_tokenizer_test tests/bpe_tokenizer_test.cpp cgpt_tokenizer)
add_cgpt_test(bpe_tokenizer_100mb_test tests/bpe_tokenizer_100mb_test.cpp cgpt_tokenizer)

set_tests_properties(
        rmsnorm_test linear_test swiglu_test rope_test tensor_test device_buffer_test gpu_buffer_test dtype_test flash_attention_test bpe_tokenizer_test
        PROPERTIES TIMEOUT 30)

set_tests_properties(bpe_tokenizer_100mb_test PROPERTIES
        TIMEOUT 900
        SKIP_RETURN_CODE 77
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
