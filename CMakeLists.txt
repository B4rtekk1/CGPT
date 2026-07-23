cmake_minimum_required(VERSION 4.3)
project(CGPT CUDA)

enable_testing()

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_ARCHITECTURES 86-real)

add_executable(CGPT src/main.cu include/core/cuda_check.h
        include/core/gpu_buffer.h
        include/ops/rmsnorm.h
        src/ops/rmsnorm.cu
        include/core/tensor.h
        src/tensor.cu
        include/core/device_guard.h
        include/core/device_buffer.h
        include/core/device_buffer.h
        include/core/dtype.h)

target_include_directories(CGPT PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include)

set_target_properties(CGPT PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

add_executable(rmsnorm_test
        tests/rmsnorm_test.cu
        src/ops/rmsnorm.cu
        include/ops/rmsnorm.h
        include/core/tensor.h
        include/core/gpu_buffer.h
        include/core/cuda_check.h)

target_include_directories(rmsnorm_test PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include)

set_target_properties(rmsnorm_test PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

add_test(NAME rmsnorm_test COMMAND rmsnorm_test)

add_executable(tensor_test
        tests/tensor_test.cu
        src/tensor.cu
        include/core/tensor.h
        include/core/gpu_buffer.h
        include/core/cuda_check.h)

target_include_directories(tensor_test PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include)

set_target_properties(tensor_test PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

add_test(NAME tensor_test COMMAND tensor_test)

add_executable(device_buffer_test
        tests/device_buffer_test.cu
        include/core/device_buffer.h
        include/core/cuda_check.h)

target_include_directories(device_buffer_test PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include)

set_target_properties(device_buffer_test PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

add_test(NAME device_buffer_test COMMAND device_buffer_test)

add_executable(dtype_test
        tests/dtype_test.cu
        include/core/dtype.h)

target_include_directories(dtype_test PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include)

set_target_properties(dtype_test PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

add_test(NAME dtype_test COMMAND dtype_test)
