#pragma once

#include <cuda_runtime_api.h>
#include <cstdio>
#include <cstdlib>

/**
 * @brief Reports a failed CUDA Runtime API call and terminates the process.
 *
 * The diagnostic includes the failed expression, CUDA's human-readable error
 * message, and the source location at which the check was performed.
 *
 * @param error CUDA error code returned by the failed call.
 * @param expression String representation of the checked expression.
 * @param file Source file containing the check.
 * @param line Source line containing the check.
 * @note This function never returns.
 */
[[noreturn]] inline void cuda_check_fail(
    cudaError_t error,
    const char* expression,
    const char* file,
    const int line
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

/**
 * @brief Checks a CUDA Runtime API result and terminates on failure.
 * @param expr CUDA Runtime API expression returning `cudaError_t`.
 */
#define CUDA_CHECK(expr)                                              \
    do {                                                              \
        const cudaError_t _cuda_error = (expr);                       \
        if (_cuda_error != cudaSuccess) {                             \
            cuda_check_fail(_cuda_error, #expr, __FILE__, __LINE__);  \
        }                                                             \
    } while (false)

/**
 * @brief Checks a cuBLAS API result and terminates on failure.
 * @param expr cuBLAS API expression returning `cublasStatus_t`.
 */
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