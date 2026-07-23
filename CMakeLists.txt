cmake_minimum_required(VERSION 4.3)
project(CGPT CUDA)

enable_testing()

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_ARCHITECTURES 86-real)

add_executable(CGPT src/main.cu include/cuda_check.h
        include/gpu_buffer.h
        include/ops/rmsnorm.h
        src/ops/rmsnorm.cu
        include/tensor.h
        src/tensor.cu
        include/device_guard.h)

target_include_directories(CGPT PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include)

set_target_properties(CGPT PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

add_executable(rmsnorm_test
        tests/rmsnorm_test.cu
        src/ops/rmsnorm.cu
        include/ops/rmsnorm.h
        include/tensor.h
        include/gpu_buffer.h
        include/cuda_check.h)

target_include_directories(rmsnorm_test PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include)

set_target_properties(rmsnorm_test PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

add_test(NAME rmsnorm_test COMMAND rmsnorm_test)

add_executable(tensor_test
        tests/tensor_test.cu
        src/tensor.cu
        include/tensor.h
        include/gpu_buffer.h
        include/cuda_check.h)

target_include_directories(tensor_test PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include)

set_target_properties(tensor_test PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

add_test(NAME tensor_test COMMAND tensor_test)
