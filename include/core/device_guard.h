#pragma once

#include "cuda_check.h"
#include <stdexcept>

/**
 * @brief Temporarily switches the active CUDA device within a scope.
 *
 * The previously active device is restored when guard is destroyed. The target device is validated before switching and
 * constructing the guard doesn't change the active device when target is already selected.
 */
class DeviceGuard {
public:
    /**
     * @brief Selects a CUDA device for the lifetime of the guard.
     * @param target_device Non-negative CUDA device index to select.
     * @throws std::invalid_argument If the device index is negative or outside the range of available CUDA devices.
     */
    explicit DeviceGuard(const int target_device) {
        if (target_device < 0) {
            throw std::invalid_argument("target_device must be non-negative");
        }

        int device_count = 0;
        CUDA_CHECK(cudaGetDeviceCount(&device_count));
        if (target_device >= device_count) {
            throw std::invalid_argument("target_device is out of range");
        }

        CUDA_CHECK(cudaGetDevice(&previous_device_));

        if (previous_device_ != target_device) {
            CUDA_CHECK(cudaSetDevice(target_device));
            changed_ = true;
        }
    }

    /**
     * @brief Restores the CUDA device that was active before construction.
     */
    ~DeviceGuard() {
        if (changed_) {
            cudaSetDevice(previous_device_);
        }
    }

private:
    int previous_device_ = 0;
    bool changed_ = false;
};
