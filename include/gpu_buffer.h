#pragma once

#include <complex.h>
#include <cuda_runtime_api.h>
#include <utility>

#include "../src/common/cuda_check.h"

template <typename T>
class GpuBuffer {
public:
    GpuBuffer() = default;

    explicit GpuBuffer(std::size_t count) {
        allocate(count);
    }

    ~GpuBuffer() {
        release();
    }

    GpuBuffer(const GpuBuffer&) = delete;
    GpuBuffer& operator=(const GpuBuffer&) = delete;

    GpuBuffer(GpuBuffer&& other) noexcept
        : data_(std::exchange(other.data_, nullptr)),
          count_(std::exchange(other.count_, 0)) {}

    GpuBuffer& operator=(GpuBuffer&& other) noexcept {
        if (this != &other) {
            release();

            data_ = std::exchange(other.data_, nullptr);
            count_ = std::exchange(other.count_, 0);
        }
        return *this;
    }

    void allocate(std::size_t count) {
        release();

        if (count == 0) {
            return;
        }

        CUDA_CHECK(cudaMalloc(&data_, count * sizeof(T)));
        count_ = count;
    }

    void release() noexcept {
        if (data_ != nullptr) {
            cudaFree(data_);
            data_ = nullptr;
            count_ = 0;
        }
    }

    void copy_from_host(const T* source, std::size_t count) {
        if (count > count_) {
            throw std::out_of_range("GpuBuffer: source is larger than the buffer size");
        }

        CUDA_CHECK(cudaMemcpy(data_, source, count * sizeof(T), cudaMemcpyHostToDevice));
    }

    void copy_to_host(T* destination, std::size_t count) const {
        if (count > count_) {
            throw std::out_of_range("GpuBuffer: destination is larger than the buffer size");
        }

        CUDA_CHECK(cudaMemcpy(destination, data_, count * sizeof(T), cudaMemcpyDeviceToHost));
    }

    [[nodiscard]] T* data() noexcept {
        return data_;
    }

    [[nodiscard]] const T* data() const noexcept {
        return data_;
    }

    [[nodiscard]] std::size_t size() const noexcept {
        return count_;
    }

    [[nodiscard]] std::size_t empty() const noexcept {
        return data_ == nullptr;
    }

private:
    T* data_ = nullptr;
    std::size_t count_ = 0;
};