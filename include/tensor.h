#pragma once

#include <algorithm>
#include <cstddef>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>
#include <cuda_runtime.h>

#include "gpu_buffer.h"

enum class DeviceType {
    CPU,
    CUDA
};

class Tensor {
public:
    explicit Tensor(
        std::vector<std::size_t> shape,
        DeviceType device_type = DeviceType::CUDA
    )
        : shape_(std::move(shape)),
          device_type_(device_type),
          storage_(device_type == DeviceType::CUDA ? element_count(shape_) : 0),
          host_storage_(device_type == DeviceType::CPU ? element_count(shape_) : 0) {}

    [[nodiscard]] float* data() noexcept {
        return device_type_ == DeviceType::CUDA
            ? storage_.data()
            : host_storage_.data();
    }

    [[nodiscard]] const float* data() const noexcept {
        return device_type_ == DeviceType::CUDA
            ? storage_.data()
            : host_storage_.data();
    }

    [[nodiscard]] const std::vector<std::size_t>& shape() const noexcept {
        return shape_;
    }

    [[nodiscard]] std::size_t numel() const noexcept {
        return device_type_ == DeviceType::CUDA
            ? storage_.size()
            : host_storage_.size();
    }

    [[nodiscard]] std::size_t dim() const noexcept {
        return shape_.size();
    }

    [[nodiscard]] DeviceType device_type() const noexcept {
        return device_type_;
    }

    [[nodiscard]] DeviceType deviceType() const noexcept {
        return device_type();
    }

    [[nodiscard]] std::size_t size(std::size_t axis) const {
        if (axis >= shape_.size()) {
            throw std::out_of_range("Tensor axis out of range");
        }
        return shape_[axis];
    }

    [[nodiscard]] static Tensor empty(
        std::vector<std::size_t> shape,
        DeviceType device_type = DeviceType::CUDA
        );

    [[nodiscard]] static Tensor zeros(
        std::vector<std::size_t> shape,
        DeviceType device_type = DeviceType::CUDA,
        cudaStream_t stream = nullptr
        );

    [[nodiscard]] static Tensor ones(
        std::vector<std::size_t> shape,
        DeviceType device_type = DeviceType::CUDA,
        cudaStream_t stream = nullptr
        );

    [[nodiscard]] static Tensor full(
        std::vector<std::size_t> shape,
        float value,
        DeviceType device_type = DeviceType::CUDA,
        cudaStream_t stream = nullptr
        );

    [[nodiscard]] static Tensor eye(
        std::size_t rows,
        std::size_t columns,
        DeviceType device_type = DeviceType::CUDA,
        cudaStream_t stream = nullptr
        );

    void copy_from_host(std::span<const float> source) {
        if (source.size() != numel()) {
            throw std::invalid_argument("Tensor: invalid input size");
        }

        if (device_type_ == DeviceType::CUDA) {
            storage_.copy_from_host(source.data(), source.size());
        } else {
            std::copy(source.begin(), source.end(), host_storage_.begin());
        }
    }

    void copy_to_host(std::span<float> destination) const {
        if (destination.size() != numel()) {
            throw std::invalid_argument("Tensor: invalid output size");
        }

        if (device_type_ == DeviceType::CUDA) {
            storage_.copy_to_host(destination.data(), destination.size());
        } else {
            std::copy(host_storage_.begin(), host_storage_.end(), destination.begin());
        }
    }

private:
    static std::size_t element_count(const std::vector<std::size_t>& shape) {
        if (shape.empty()) {
            throw std::invalid_argument("Tensor shape cannot be empty");
        }

        std::size_t count = 1;

        for (const std::size_t dimension : shape) {
            if (dimension == 0) {
                throw std::invalid_argument("Tensor dimension cannot be zero");
            }

            count *= dimension;
        }
        return count;
    }

    std::vector<std::size_t> shape_;
    DeviceType device_type_;
    GpuBuffer<float> storage_;
    std::vector<float> host_storage_;
};
