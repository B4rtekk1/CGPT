#pragma once

#include <cuda_runtime_api.h>
#include <cublasLt.h>
#include <cstdio>
#include <cstdlib>

/// Helper function to print an error message and exit the program when a CUDA runtime API call fails.
[[noreturn]] inline void cuda_check_fail(
    cudaError_t error,
    const char* expression,
    const char* file,
    int line
    ) {
    std::fprintf(
        stderr,
        "CUDA error\n"
        "  expression: %s\n"
        "  message:    %s\n"
        "  location:   %s:%d\n",
        expression,
        cudaGetErrorString(error),
        file,
        line
    );

    std::exit(EXIT_FAILURE);
}

/// Macro to check the return value of a CUDA runtime API call.
#define CUDA_CHECK(expr)                                              \
    do {                                                              \
        const cudaError_t _cuda_error = (expr);                       \
        if (_cuda_error != cudaSuccess) {                             \
            cuda_check_fail(_cuda_error, #expr, __FILE__, __LINE__);  \
        }                                                             \
    } while (false)

/// Macro to check the return value of a cuBLAS API call.
#define CUBLAS_CHECK(expr)                                             \
    do {                                                               \
        const cublasStatus_t _cublas_status = (expr);                  \
        if (_cublas_status != CUBLAS_STATUS_SUCCESS) {                 \
            std::fprintf(stderr, "cuBLAS error %d at %s:%d (%s)\n",    \
                static_cast<int>(_cublas_status),                      \
                __FILE__, __LINE__, #expr);                            \
            std::exit(EXIT_FAILURE);                                   \
        }                                                              \
    } while (false)