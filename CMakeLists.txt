cmake_minimum_required(VERSION 4.3)
project(CGPT CUDA)

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_ARCHITECTURES 86-real)

add_executable(CGPT main.cu src/common/cuda_check.h
        include/gpu_buffer.h)

set_target_properties(CGPT PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)