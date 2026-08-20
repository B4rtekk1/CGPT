#pragma once

#include "cuda_check.h"

#include <utility>

/**
 * @brief RAII owner for a raw allocation in CUDA device memory.
 *
 * `DeviceBuffer` owns a byte-sized allocation obtained with `cudaMalloc` and
 * releases it with `cudaFree`. The class is non-copyable and movable, so each
 * allocation has a single owner. Reallocating the buffer preserves the
 * current allocation until the replacement allocation succeeds.
 */
class DeviceBuffer {
public:
    struct BorrowedTag {};
    /** @brief Constructs an empty device buffer. */
    DeviceBuffer() = default;

    /**
     * @brief Constructs a buffer with the requested capacity.
     * @param bytes Number of bytes to allocate in device memory.
     */
    explicit DeviceBuffer(std::size_t bytes) {
        allocate(bytes);
    }

    DeviceBuffer(void* data, std::size_t bytes, BorrowedTag)
        : data_(data), bytes_(bytes), owns_(false) {}

    /** @brief Releases the owned device allocation, if any. */
    ~DeviceBuffer() {
        release();
    }

    /** @brief Copy construction is disabled because the buffer owns a resource. */
    DeviceBuffer(const DeviceBuffer&) = delete;

    /** @brief Copy assignment is disabled because the buffer owns a resource. */
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    /**
     * @brief Transfers ownership from another device buffer.
     * @param other Buffer whose allocation is transferred to this object.
     */
    DeviceBuffer(DeviceBuffer&& other) noexcept
            : data_(std::exchange(other.data_, nullptr)),
              bytes_(std::exchange(other.bytes_, 0)),
              owns_(std::exchange(other.owns_, true)) {}

    /**
     * @brief Transfers ownership from another device buffer.
     * @param other Buffer whose allocation is transferred to this object.
     * @return Reference to this buffer.
     */
    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
        if (this != &other) {
            release();
            data_ = std::exchange(other.data_, nullptr);
            bytes_ = std::exchange(other.bytes_, 0);
            owns_ = std::exchange(other.owns_, true);
        }
        return *this;
    }

    /**
     * @brief Allocates or replaces the device-memory allocation.
     *
     * Passing zero releases the current allocation and leaves the buffer
     * empty. If the requested allocation fails, the existing allocation is
     * kept alive because it is released only after `cudaMalloc` succeeds.
     *
     * @param bytes Number of bytes to allocate. A value of zero clears the
     *              buffer.
     */
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
        owns_ = true;
    }

    /**
     * @brief Releases the owned device-memory allocation.
     *
     * The object becomes empty after this call. Calling `release` repeatedly
     * is safe.
     */
    void release() noexcept {
        if (data_ != nullptr && owns_) {
            CUDA_CHECK(cudaFree(data_));
        }
        data_ = nullptr;
        bytes_ = 0;
        owns_ = true;
    }

    /**
     * @brief Returns the raw device-memory pointer.
     * @return Mutable pointer to the allocation, or `nullptr` if empty.
     */
    [[nodiscard]] void* data() noexcept { return data_; }

    /**
     * @brief Returns the raw device-memory pointer for a const buffer.
     * @return Read-only pointer to the allocation, or `nullptr` if empty.
     */
    [[nodiscard]] const void* data() const noexcept { return data_; }

    /**
     * @brief Returns the allocation size.
     * @return Number of allocated bytes, or zero if the buffer is empty.
     */
    [[nodiscard]] std::size_t bytes() const noexcept { return bytes_; }

    /**
     * @brief Checks whether the buffer owns no device memory.
     * @return `true` if the buffer is empty; otherwise `false`.
     */
    [[nodiscard]] bool empty() const noexcept { return data_ == nullptr; }

private:
    void* data_ = nullptr;
    std::size_t bytes_ = 0;
    bool owns_ = true;

};
