#pragma once

#include "cuda_check.h"

#include <utility>

class DeviceBuffer {
public:
    DeviceBuffer() = default;

    explicit DeviceBuffer(std::size_t bytes) {
        allocate(bytes);
    }

    ~DeviceBuffer() {
        release();
    }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    DeviceBuffer(DeviceBuffer&& other) noexcept
            : data_(std::exchange(other.data_, nullptr)),
              bytes_(std::exchange(other.bytes_, 0)) {}

    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
        if (this != &other) {
            release();
            data_ = std::exchange(other.data_, nullptr);
            bytes_ = std::exchange(other.bytes_, 0);
        }
        return *this;
    }

    void allocate(std::size_t bytes) {
        if (bytes == 0) {
            release();
            return;
        }

        // Keep the current allocation alive until the replacement succeeds.
        void* new_data = nullptr;
        CUDA_CHECK(cudaMalloc(&new_data, bytes));
        release();
        data_ = new_data;
        bytes_ = bytes;
    }

    void release() noexcept {
        if (data_ != nullptr) {
            CUDA_CHECK(cudaFree(data_));
            data_ = nullptr;
            bytes_ = 0;
        }
    }

    [[nodiscard]] void* data() noexcept { return data_; }
    [[nodiscard]] const void* data() const noexcept { return data_; }
    [[nodiscard]] std::size_t bytes() const noexcept { return bytes_; }
    [[nodiscard]] bool empty() const noexcept { return data_ == nullptr; }

private:
    void* data_ = nullptr;
    std::size_t bytes_ = 0;

};
