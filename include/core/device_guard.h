#pragma once

#include "cuda_check.h"
#include <stdexcept>

class DeviceGuard {
public:
    explicit DeviceGuard(int target_device) {
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

    ~DeviceGuard() {
        if (changed_) {
            cudaSetDevice(previous_device_);
        }
    }

private:
    int previous_device_ = 0;
    bool changed_ = false;
};
