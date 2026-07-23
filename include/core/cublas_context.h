#pragma once

#include "cuda_check.h"
#include <cublas_v2.h>

#include <cstdio>
#include <cstdlib>

#include "device_guard.h"

inline void cublas_check_fail(
    cublasStatus_t status,
    const char* expression,
    const char* file,
    int line
    ) {
    std::fprintf(
        stderr,
        "cuBLASLt error %d\n"
        "  expression: %s\n"
        "  location:   %s:%d\n",
        static_cast<int>(status),
        expression,
        file,
        line
    );

    std::exit(EXIT_FAILURE);
}

#define CUBLAS_CHECK(expr)                                             \
    do {                                                               \
        const cublasStatus_t _cublas_status = (expr);                  \
        if (_cublas_status != CUBLAS_STATUS_SUCCESS) {                 \
            cublas_check_fail(                                         \
                _cublas_status, #expr, __FILE__, __LINE__              \
            );                                                         \
        }                                                              \
    } while (false)

class CublasLtContext {
public:
    explicit CublasLtContext(int device_index = 0)
        : device_index_(device_index) {
        DeviceGuard guard(device_index_);
        CUBLAS_CHECK(cublasLtCreate(&handle_));
    }

    ~CublasLtContext() {
        if (handle_ != nullptr) {
            cublasLtDestroy(handle_);
        }
    }

    CublasLtContext(const CublasLtContext&) = delete;
    CublasLtContext& operator=(const CublasLtContext&) = delete;

    [[nodiscard]] cublasLtHandle_t handle() const noexcept {
        return handle_;
    }

    [[nodiscard]] int device_index() const noexcept {
        return device_index_;
    }

private:
    cublasLtHandle_t handle_ = nullptr;
    int device_index_ = 0;
};
