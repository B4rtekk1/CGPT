#pragma once

#include <limits>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>
#include <cuda_runtime.h>

#include "dtype.h"
#include "device_buffer.h"

enum class DeviceType {
    CPU,
    CUDA
};

class Tensor {
public:
    explicit Tensor(
        std::vector<std::size_t> shape,
        const DeviceType device_type = DeviceType::CUDA,
        const Dtype dtype = Dtype::F32
    )
        : shape_(std::move(shape)),
          device_type_(validate_device_type(device_type)),
          dtype_(validate_storage_dtype(dtype)),
          element_count_(element_count(shape_)),
          storage_(device_type == DeviceType::CUDA
                       ? dtype_bytes(element_count_, dtype_)
                       : 0),
          host_storage_(device_type == DeviceType::CPU
                            ? dtype_bytes(element_count_, dtype_)
                            : 0) {}

    explicit Tensor(
        std::vector<std::size_t> shape,
        Dtype dtype,
        DeviceType device_type = DeviceType::CUDA
    ) : Tensor(std::move(shape), device_type, dtype) {}

    Tensor(const Tensor& other)
        : shape_(other.shape_),
          device_type_(other.device_type_),
          dtype_(other.dtype_),
          element_count_(other.element_count_),
          storage_(other.device_type_ == DeviceType::CUDA ? other.nbytes() : 0),
          host_storage_(other.device_type_ == DeviceType::CPU ? other.nbytes() : 0) {
        if (device_type_ == DeviceType::CUDA) {
            CUDA_CHECK(cudaMemcpy(
                storage_.data(), other.storage_.data(),
                other.nbytes(), cudaMemcpyDeviceToDevice));
        } else {
            host_storage_ = other.host_storage_;
        }
    }

    Tensor& operator=(const Tensor& other) {
        if (this == &other) {
            return *this;
        }

        Tensor copy(other);
        swap(copy);
        return *this;
    }

    Tensor(Tensor&& other) noexcept
        : shape_(std::move(other.shape_)),
          device_type_(other.device_type_),
          dtype_(other.dtype_),
          element_count_(std::exchange(other.element_count_, 0)),
          storage_(std::move(other.storage_)),
          host_storage_(std::move(other.host_storage_)) {}

    Tensor& operator=(Tensor&& other) noexcept {
        if (this != &other) {
            shape_ = std::move(other.shape_);
            device_type_ = other.device_type_;
            dtype_ = other.dtype_;
            element_count_ = std::exchange(other.element_count_, 0);
            storage_ = std::move(other.storage_);
            host_storage_ = std::move(other.host_storage_);
        }
        return *this;
    }

    [[nodiscard]] float* data() {
        require_f32_data();
        return static_cast<float*>(raw_data());
    }

    [[nodiscard]] const float* data() const {
        require_f32_data();
        return static_cast<const float*>(raw_data());
    }

    [[nodiscard]] void* raw_data() noexcept {
        return device_type_ == DeviceType::CUDA
            ? storage_.data()
            : static_cast<void*>(host_storage_.data());
    }

    [[nodiscard]] const void* raw_data() const noexcept {
        return device_type_ == DeviceType::CUDA
            ? storage_.data()
            : static_cast<const void*>(host_storage_.data());
    }

    [[nodiscard]] const std::vector<std::size_t>& shape() const noexcept {
        return shape_;
    }

    [[nodiscard]] std::size_t numel() const noexcept {
        return element_count_;
    }

    [[nodiscard]] std::size_t nbytes() const noexcept {
        return element_count_ * dtype_size(dtype_);
    }

    [[nodiscard]] std::size_t dim() const noexcept {
        return shape_.size();
    }

    [[nodiscard]] DeviceType device_type() const noexcept {
        return device_type_;
    }

    [[nodiscard]] Dtype dtype() const noexcept {
        return dtype_;
    }

    [[nodiscard]] std::size_t size(std::size_t axis) const {
        if (axis >= shape_.size()) {
            throw std::out_of_range("Tensor axis out of range");
        }
        return shape_[axis];
    }

    /**
     * @brief Changes tensor metadata without moving or reallocating storage.
     *
     * The new shape must contain exactly the same number of elements as the
     * current shape. Because Tensor storage is contiguous, every valid reshape
     * remains contiguous as well.
     *
     * @param new_shape Requested tensor shape.
     * @throws std::invalid_argument If the shape is empty, contains a zero,
     *         overflows size_t, or changes the number of elements.
     */
    void reshape(std::vector<std::size_t> new_shape) {
        const std::size_t new_element_count = element_count(new_shape);
        if (new_element_count != element_count_) {
            throw std::invalid_argument(
                "Tensor::reshape cannot change the number of elements");
        }
        shape_ = std::move(new_shape);
    }

    [[nodiscard]] static Tensor empty(
        std::vector<std::size_t> shape,
        DeviceType device_type = DeviceType::CUDA,
        Dtype dtype = Dtype::F32
        );

    [[nodiscard]] static Tensor zeros(
        std::vector<std::size_t> shape,
        DeviceType device_type = DeviceType::CUDA,
        cudaStream_t stream = nullptr,
        Dtype dtype = Dtype::F32
        );

    [[nodiscard]] static Tensor ones(
        std::vector<std::size_t> shape,
        DeviceType device_type = DeviceType::CUDA,
        cudaStream_t stream = nullptr,
        Dtype dtype = Dtype::F32
        );

    [[nodiscard]] static Tensor full(
        std::vector<std::size_t> shape,
        float value,
        DeviceType device_type = DeviceType::CUDA,
        cudaStream_t stream = nullptr,
        Dtype dtype = Dtype::F32
        );

    [[nodiscard]] static Tensor eye(
        std::size_t rows,
        std::size_t columns,
        DeviceType device_type = DeviceType::CUDA,
        cudaStream_t stream = nullptr,
        Dtype dtype = Dtype::F32
        );

    void copy_from_host(std::span<const float> source);
    void copy_to_host(std::span<float> destination) const;

private:
    void swap(Tensor& other) noexcept {
        using std::swap;
        swap(shape_, other.shape_);
        swap(device_type_, other.device_type_);
        swap(dtype_, other.dtype_);
        swap(element_count_, other.element_count_);
        swap(storage_, other.storage_);
        swap(host_storage_, other.host_storage_);
    }

    static DeviceType validate_device_type(DeviceType device_type) {
        if (device_type != DeviceType::CPU && device_type != DeviceType::CUDA) {
            throw std::invalid_argument("Tensor: invalid device type");
        }
        return device_type;
    }

    static Dtype validate_storage_dtype(Dtype dtype) {
        if (dtype != Dtype::F16 && dtype != Dtype::BF16 && dtype != Dtype::F32) {
            throw std::invalid_argument("Tensor storage supports only floating-point dtypes");
        }
        return dtype;
    }

    void require_f32_data() const {
        if (dtype_ != Dtype::F32) {
            throw std::logic_error("Tensor::data<float> requires F32; use raw_data for other dtypes");
        }
    }

    static std::size_t element_count(const std::vector<std::size_t>& shape) {
        if (shape.empty()) {
            throw std::invalid_argument("Tensor shape cannot be empty");
        }

        std::size_t count = 1;

        for (const std::size_t dimension : shape) {
            if (dimension == 0) {
                throw std::invalid_argument("Tensor dimension cannot be zero");
            }

            if (count > std::numeric_limits<std::size_t>::max() / dimension) {
                throw std::invalid_argument("Tensor shape size overflows");
            }

            count *= dimension;
        }
        return count;
    }

    std::vector<std::size_t> shape_;
    DeviceType device_type_;
    Dtype dtype_;
    std::size_t element_count_ = 0;
    DeviceBuffer storage_;
    std::vector<std::uint8_t> host_storage_;
};
