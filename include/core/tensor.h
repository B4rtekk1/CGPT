#pragma once

#include <limits>
#include <cstring>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>
#include <cuda_runtime.h>

#include "dtype.h"
#include "device_buffer.h"

/** @brief Memory location used by a tensor. */
enum class DeviceType {
    /** @brief Host CPU memory. */
    CPU,
    /** @brief CUDA device memory. */
    CUDA
};

/**
 * @brief Owning contiguous tensor supporting CPU and CUDA storage.
 *
 * CUDA storage is managed through `DeviceBuffer`, while CPU storage is held
 * in a byte vector. The tensor shape, data type, and device type are stored as
 * metadata alongside the allocation.
 */
class Tensor {
public:
    /** Creates a non-owning view over contiguous CUDA memory. */
    [[nodiscard]] static Tensor device_view(
        std::vector<std::size_t> shape, void* data, Dtype dtype) {
        return Tensor(std::move(shape), DeviceType::CUDA, dtype,
                      DeviceBuffer::BorrowedTag{}, data);
    }

    /**
     * @brief Constructs a tensor with the requested shape and storage type.
     * @param shape Tensor dimensions. Every dimension must be non-zero.
     * @param device_type Memory location for the tensor.
     * @param dtype Floating-point storage data type.
     * @throws std::invalid_argument If the device, data type, or shape is invalid.
     */
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

    /** @brief Constructs a tensor with the data type argument before the device. */
    explicit Tensor(
        std::vector<std::size_t> shape,
        Dtype dtype,
        DeviceType device_type = DeviceType::CUDA
    ) : Tensor(std::move(shape), device_type, dtype) {}

private:
    Tensor(std::vector<std::size_t> shape, DeviceType device_type, Dtype dtype,
           DeviceBuffer::BorrowedTag, void* data)
        : shape_(std::move(shape)),
          device_type_(validate_device_type(device_type)),
          dtype_(validate_storage_dtype(dtype)),
          element_count_(element_count(shape_)),
          storage_(data, dtype_bytes(element_count_, dtype_), DeviceBuffer::BorrowedTag{}) {}

public:

    /** @brief Creates a deep copy of another tensor. */
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

    /** @brief Replaces this tensor with a deep copy of another tensor. */
    Tensor& operator=(const Tensor& other) {
        if (this == &other) {
            return *this;
        }

        Tensor copy(other);
        swap(copy);
        return *this;
    }

    /** @brief Transfers tensor storage from another tensor. */
    Tensor(Tensor&& other) noexcept
        : shape_(std::move(other.shape_)),
          device_type_(other.device_type_),
          dtype_(other.dtype_),
          element_count_(std::exchange(other.element_count_, 0)),
          storage_(std::move(other.storage_)),
          host_storage_(std::move(other.host_storage_)) {}

    /** @brief Transfers tensor storage from another tensor. */
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

    /**
     * @brief Returns a mutable F32 data pointer.
     * @throws std::logic_error If the tensor data type is not F32.
     */
    [[nodiscard]] float* data() {
        require_f32_data();
        return static_cast<float*>(raw_data());
    }

    /**
     * @brief Returns a const F32 data pointer.
     * @throws std::logic_error If the tensor data type is not F32.
     */
    [[nodiscard]] const float* data() const {
        require_f32_data();
        return static_cast<const float*>(raw_data());
    }

    /** @brief Returns a mutable pointer to the raw storage. */
    [[nodiscard]] void* raw_data() noexcept {
        return device_type_ == DeviceType::CUDA
            ? storage_.data()
            : static_cast<void*>(host_storage_.data());
    }

    /** @brief Returns a const pointer to the raw storage. */
    [[nodiscard]] const void* raw_data() const noexcept {
        return device_type_ == DeviceType::CUDA
            ? storage_.data()
            : static_cast<const void*>(host_storage_.data());
    }

    /** @brief Returns the tensor shape. */
    [[nodiscard]] const std::vector<std::size_t>& shape() const noexcept {
        return shape_;
    }

    /** @brief Returns the total number of tensor elements. */
    [[nodiscard]] std::size_t numel() const noexcept {
        return element_count_;
    }

    /** @brief Returns the storage size in bytes. */
    [[nodiscard]] std::size_t nbytes() const noexcept {
        return element_count_ * dtype_size(dtype_);
    }

    /** @brief Returns the number of tensor dimensions. */
    [[nodiscard]] std::size_t dim() const noexcept {
        return shape_.size();
    }

    /** @brief Returns the tensor memory location. */
    [[nodiscard]] DeviceType device_type() const noexcept {
        return device_type_;
    }

    /** @brief Returns the tensor storage data type. */
    [[nodiscard]] Dtype dtype() const noexcept {
        return dtype_;
    }

    /**
     * @brief Returns the size of one tensor axis.
     * @param axis Zero-based axis index.
     * @throws std::out_of_range If @p axis is outside the tensor rank.
     */
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

    /** @brief Creates an uninitialized tensor. */
    [[nodiscard]] static Tensor empty(
        std::vector<std::size_t> shape,
        DeviceType device_type = DeviceType::CUDA,
        Dtype dtype = Dtype::F32
        );

    /** @brief Creates a zero-initialized tensor. */
    [[nodiscard]] static Tensor zeros(
        std::vector<std::size_t> shape,
        DeviceType device_type = DeviceType::CUDA,
        cudaStream_t stream = nullptr,
        Dtype dtype = Dtype::F32
        );

    /** @brief Creates a one-initialized tensor. */
    [[nodiscard]] static Tensor ones(
        std::vector<std::size_t> shape,
        DeviceType device_type = DeviceType::CUDA,
        cudaStream_t stream = nullptr,
        Dtype dtype = Dtype::F32
        );

    /** @brief Creates a tensor initialized to one scalar value. */
    [[nodiscard]] static Tensor full(
        std::vector<std::size_t> shape,
        float value,
        DeviceType device_type = DeviceType::CUDA,
        cudaStream_t stream = nullptr,
        Dtype dtype = Dtype::F32
        );

    /** @brief Creates a two-dimensional identity matrix. */
    [[nodiscard]] static Tensor eye(
        std::size_t rows,
        std::size_t columns,
        DeviceType device_type = DeviceType::CUDA,
        cudaStream_t stream = nullptr,
        Dtype dtype = Dtype::F32
        );

    /**
     * @brief Copies F32 values from host memory into the tensor.
     * @param source Host values whose size must match `numel()`.
     */
    void copy_from_host(std::span<const float> source);

    /**
     * @brief Copies F32 tensor values into host memory.
     * @param destination Destination span whose size must match `numel()`.
     */
    void copy_to_host(std::span<float> destination) const;

    /** Copies raw tensor bytes from host memory into this tensor. */
    void copy_raw_from_host(std::span<const std::uint8_t> source) {
        if (source.size() != nbytes())
            throw std::invalid_argument("Tensor::copy_raw_from_host size mismatch");
        if (device_type_ == DeviceType::CUDA) {
            CUDA_CHECK(cudaMemcpy(storage_.data(), source.data(), source.size(), cudaMemcpyHostToDevice));
        } else {
            std::memcpy(host_storage_.data(), source.data(), source.size());
        }
    }

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
