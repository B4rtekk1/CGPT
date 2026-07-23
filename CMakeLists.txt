cmake_minimum_required(VERSION 4.3)
project(CGPT LANGUAGES CXX CUDA)

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
        src/ops/rmsnorm.cu
        src/tensor.cu)

target_include_directories(cgpt_core PUBLIC
        ${CGPT_INCLUDE_DIR}
        ${CGPT_INCLUDE_DIR}/core)
target_link_libraries(cgpt_core PUBLIC CUDA::cudart CUDA::cublasLt)
set_target_properties(cgpt_core PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

add_executable(CGPT src/main.cu
        include/core/cublas_context.h)
target_link_libraries(CGPT PRIVATE cgpt_core)

function(add_cgpt_test name source)
    add_executable(${name} ${source})
    target_link_libraries(${name} PRIVATE cgpt_core)
    add_test(NAME ${name} COMMAND ${name})
    set_target_properties(${name} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endfunction()

add_cgpt_test(rmsnorm_test tests/rmsnorm_test.cu)
add_cgpt_test(tensor_test tests/tensor_test.cu)
add_cgpt_test(device_buffer_test tests/device_buffer_test.cu)
add_cgpt_test(gpu_buffer_test tests/gpu_buffer_test.cu)
add_cgpt_test(dtype_test tests/dtype_test.cu)

set_tests_properties(
        rmsnorm_test tensor_test device_buffer_test gpu_buffer_test dtype_test
        PROPERTIES TIMEOUT 30)
