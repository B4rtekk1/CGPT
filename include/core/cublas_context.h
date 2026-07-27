#pragma once

#include "cuda_check.h"
#include <cublasLt.h>
#include <cublas_v2.h>

#include "device_guard.h"

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
