#pragma once

#include "cuda_check.h"
#include <cublasLt.h>
#include <cublas_v2.h>

#include "device_guard.h"

/**
 * @brief RAII wrapper around a cuBLASLt library context.
 *
 * The context is created on the selected CUDA device and destroyed when the
 * wrapper goes out of scope. Copying is disabled because a cuBLASLt handle is
 * an owned resource.
 */
class CublasLtContext {
public:
    /**
     * @brief Creates a cuBLASLt context on a CUDA device.
     * @param device_index CUDA device on which the handle is created.
     */
    explicit CublasLtContext(int device_index = 0)
        : device_index_(device_index) {
        DeviceGuard guard(device_index_);
        CUBLAS_CHECK(cublasLtCreate(&handle_));
    }

    /** @brief Destroys the cuBLASLt context, if it was created. */
    ~CublasLtContext() {
        if (handle_ != nullptr) {
            cublasLtDestroy(handle_);
        }
    }

    /** @brief Copy construction is disabled because the handle is owned. */
    CublasLtContext(const CublasLtContext&) = delete;

    /** @brief Copy assignment is disabled because the handle is owned. */
    CublasLtContext& operator=(const CublasLtContext&) = delete;

    /**
     * @brief Returns the underlying cuBLASLt handle.
     * @return Native cuBLASLt handle managed by this context.
     */
    [[nodiscard]] cublasLtHandle_t handle() const noexcept {
        return handle_;
    }

    /**
     * @brief Returns the CUDA device associated with this context.
     * @return CUDA device index passed to the constructor.
     */
    [[nodiscard]] int device_index() const noexcept {
        return device_index_;
    }

private:
    cublasLtHandle_t handle_ = nullptr;
    int device_index_ = 0;
};