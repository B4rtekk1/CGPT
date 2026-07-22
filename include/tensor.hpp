#pragma once

#include <cstddef>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

#include "gpu_buffer.h"

class Tensor {
public:
    explicit Tensor(std::vector<std::size_t> shape)
        : shape_(std::move(shape)),
          storage_(element_count(shape_)) {}

    [[nodiscard]] float* data() noexcept {
        return storage_.data();
    }

    [[nodiscard]] const float* data() const noexcept {
        return storage_.data();
    }

    [[nodiscard]] const std::vector<std::size_t>& shape() const noexcept {
        return shape_;
    }

    [[nodiscard]] std::size_t numel() const noexcept {
        return storage_.size();
    }

    [[nodiscard]] std::size_t dim() const noexcept {
        return storage_.size();
    }

    [[nodiscard]] std::size_t size(std::size_t axis) const {
        if (axis >= shape_.size()) {
            throw std::out_of_range("Tensor axis out of range");
        }
        return shape_[axis];
    }

    void copy_from_host(std::span<const float> source) {
        if (source.size() != numel()) {
            throw std::invalid_argument("Tensor: invalid input size");
        }

        storage_.copy_from_host(source.data(), source.size());
    }

    void copy_to_host(std::span<float> destination) const {
        if (destination.size() != numel()) {
            throw std::invalid_argument("Tensor: invalid output size");
        }

        storage_.copy_to_host(destination.data(), destination.size());
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
    GpuBuffer<float> storage_;
};
