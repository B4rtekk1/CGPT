/** @file tensor.cu CUDA tensor storage, transfer, and elementwise utility operations. */

#include "core/tensor.h"
#include "core/cuda_check.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <algorithm>
#include <vector>

namespace {


/**
 * @brief Fills a contiguous CUDA buffer with a scalar value.
 *
 * Each thread writes at most one element. The input value is converted to the
 * destination CUDA scalar type @p T before storage.
 *
 * @tparam T Destination element type.
 * @param data Pointer to device memory containing @p count elements.
 * @param count Number of elements to initialize.
 * @param value F32 scalar converted to @p T for every element.
 */
template <typename T>
__global__ void fill_kernel(T* data, std::size_t count, float value) {
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count) {
        data[index] = static_cast<T>(value);
    }
}


/**
 * @brief Generates a row-major identity matrix in CUDA memory.
 *
 * Elements on the main diagonal are set to one and every other element is set
 * to zero. Rectangular matrices are supported.
 *
 * @tparam T Destination element type.
 * @param data Pointer to a device buffer containing `rows * columns` elements.
 * @param rows Number of matrix rows.
 * @param columns Number of matrix columns.
 */
template <typename T>
__global__ void eye_kernel(
    T* data,
    std::size_t rows,
    std::size_t columns
) {
    const std::size_t index =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    const std::size_t count = rows * columns;

    if (index < count) {
        const std::size_t row = index / columns;
        const std::size_t column = index % columns;

        data[index] = static_cast<T>((row == column) ? 1.0f : 0.0f);
    }
}


/**
 * @brief Launches the typed CUDA fill kernel for a tensor.
 *
 * @tparam T CUDA storage type corresponding to the tensor dtype.
 * @param tensor CUDA tensor whose complete storage is initialized.
 * @param value Scalar value written to every element.
 * @param stream CUDA stream used for the asynchronous kernel launch.
 *
 * @note Kernel-launch errors are checked by the caller.
 */
template <typename T>
void launch_fill(Tensor& tensor, float value, cudaStream_t stream) {
    constexpr int threads = 256;
    const int blocks = static_cast<int>((tensor.numel() + threads - 1) / threads);
    fill_kernel<<<blocks, threads, 0, stream>>>(
        static_cast<T*>(tensor.raw_data()), tensor.numel(), value);
}


/**
 * @brief Fills a CPU or CUDA tensor with a scalar value.
 *
 * CPU tensors are initialized through Tensor::copy_from_host(). CUDA tensors
 * dispatch to a dtype-specific kernel supporting F32, F16, and BF16 storage.
 *
 * @param tensor Tensor to initialize.
 * @param value Scalar value written to every tensor element.
 * @param stream CUDA stream used when @p tensor resides on the GPU.
 *
 * @note The CUDA path is asynchronous with respect to the host.
 */
void fill(Tensor& tensor, float value, cudaStream_t stream) {
    if (tensor.device_type() == DeviceType::CPU) {
        tensor.copy_from_host(std::vector<float>(tensor.numel(), value));
    } else if (tensor.dtype() == Dtype::F16) {
        launch_fill<__half>(tensor, value, stream);
    } else if (tensor.dtype() == Dtype::BF16) {
        launch_fill<__nv_bfloat16>(tensor, value, stream);
    } else {
        launch_fill<float>(tensor, value, stream);
    }
}


/**
 * @brief Converts an F32 host span to a typed host vector.
 *
 * @tparam T Destination host scalar type.
 * @param source F32 source values.
 * @return Newly allocated vector containing the converted values.
 */
template <typename T>
std::vector<T> convert_from_float(const std::span<const float> source) {
    std::vector<T> converted(source.size());
    std::transform(source.begin(), source.end(), converted.begin(),
                   [](const float value) { return static_cast<T>(value); });
    return converted;
}


/**
 * @brief Converts a typed host array to an F32 destination span.
 *
 * @tparam T Source host scalar type.
 * @param source Pointer to at least `destination.size()` source elements.
 * @param destination F32 span receiving converted values.
 */
template <typename T>
void convert_to_float(const T* source, const std::span<float> destination) {
    std::transform(source, source + destination.size(), destination.begin(),
                   [](const T value) { return static_cast<float>(value); });
}

} // namespace


/**
 * @brief Copies F32 host values into this tensor.
 *
 * Values are converted to the tensor's storage dtype when necessary. CUDA
 * tensors use a synchronous host-to-device copy; CPU tensors use a direct
 * memory copy after conversion.
 *
 * @param source Host values. The span length must equal numel().
 *
 * @throws std::invalid_argument If the source length differs from numel().
 * @throws std::runtime_error If an underlying CUDA copy fails.
 *
 * @note Supported storage dtypes are F32, F16, and BF16.
 */
void Tensor::copy_from_host(const std::span<const float> source) {
    if (source.size() != numel()) {
        throw std::invalid_argument("Tensor: invalid input size");
    }

    if (dtype_ == Dtype::F32) {
        if (device_type_ == DeviceType::CUDA) {
            CUDA_CHECK(cudaMemcpy(
                storage_.data(), source.data(), nbytes(), cudaMemcpyHostToDevice));
        } else {
            std::memcpy(host_storage_.data(), source.data(), nbytes());
        }
        return;
    }

    if (dtype_ == Dtype::F16) {
        const auto converted = convert_from_float<__half>(source);
        if (device_type_ == DeviceType::CUDA) {
            CUDA_CHECK(cudaMemcpy(
                storage_.data(), converted.data(), nbytes(), cudaMemcpyHostToDevice));
        } else {
            std::memcpy(host_storage_.data(), converted.data(), nbytes());
        }
        return;
    }

    const auto converted = convert_from_float<__nv_bfloat16>(source);
    if (device_type_ == DeviceType::CUDA) {
        CUDA_CHECK(cudaMemcpy(
            storage_.data(), converted.data(), nbytes(), cudaMemcpyHostToDevice));
    } else {
        std::memcpy(host_storage_.data(), converted.data(), nbytes());
    }
}


/**
 * @brief Copies this tensor to an F32 host span.
 *
 * CUDA storage is first transferred to host memory. F16 and BF16 elements are
 * then expanded to F32.
 *
 * @param destination Host span receiving the tensor values. Its length must
 * equal numel().
 *
 * @throws std::invalid_argument If the destination length differs from numel().
 * @throws std::runtime_error If an underlying CUDA copy fails.
 */
void Tensor::copy_to_host(const std::span<float> destination) const {
    if (destination.size() != numel()) {
        throw std::invalid_argument("Tensor: invalid output size");
    }

    if (dtype_ == Dtype::F32) {
        if (device_type_ == DeviceType::CUDA) {
            CUDA_CHECK(cudaMemcpy(
                destination.data(), storage_.data(), nbytes(), cudaMemcpyDeviceToHost));
        } else {
            std::memcpy(destination.data(), host_storage_.data(), nbytes());
        }
        return;
    }

    std::vector<std::uint8_t> host_bytes;
    const void* source = host_storage_.data();
    if (device_type_ == DeviceType::CUDA) {
        host_bytes.resize(nbytes());
        CUDA_CHECK(cudaMemcpy(
            host_bytes.data(), storage_.data(), nbytes(), cudaMemcpyDeviceToHost));
        source = host_bytes.data();
    }

    if (dtype_ == Dtype::F16) {
        convert_to_float(static_cast<const __half*>(source), destination);
    } else {
        convert_to_float(static_cast<const __nv_bfloat16*>(source), destination);
    }
}


/**
 * @brief Allocates an uninitialized tensor.
 *
 * @param shape Tensor dimensions.
 * @param device_type Target device.
 * @param dtype Element storage type.
 * @return Newly allocated tensor whose contents are unspecified.
 */
Tensor Tensor::empty(
    std::vector<std::size_t> shape,
    const DeviceType device_type,
    const Dtype dtype) {
    return Tensor(std::move(shape), device_type, dtype);
}


/**
 * @brief Allocates a tensor initialized to zero.
 *
 * CUDA tensors are cleared with cudaMemsetAsync(); CPU tensors are cleared with
 * std::memset().
 *
 * @param shape Tensor dimensions.
 * @param device_type Target device.
 * @param stream CUDA stream used for GPU initialization.
 * @param dtype Element storage type.
 * @return Newly allocated zero-filled tensor.
 *
 * @note GPU initialization is asynchronous with respect to the host.
 */
Tensor Tensor::zeros(
    std::vector<std::size_t> shape,
    const DeviceType device_type,
    cudaStream_t stream,
    const Dtype dtype) {
    Tensor result(std::move(shape), device_type, dtype);

    if (device_type == DeviceType::CUDA) {
        // cudaMemset is faster than a kernel for zeros.
        CUDA_CHECK(cudaMemsetAsync(
            result.raw_data(), 0, result.nbytes(), stream));
    } else {
        std::memset(result.raw_data(), 0, result.nbytes());
    }

    return result;
}


/**
 * @brief Allocates a tensor initialized to one.
 *
 * @param shape Tensor dimensions.
 * @param device_type Target device.
 * @param stream CUDA stream used for GPU initialization.
 * @param dtype Element storage type.
 * @return Newly allocated tensor containing ones.
 */
Tensor Tensor::ones(
    std::vector<std::size_t> shape,
    const DeviceType device_type,
    cudaStream_t stream,
    const Dtype dtype) {
    Tensor result(std::move(shape), device_type, dtype);
    fill(result, 1.0f, stream);
    return result;
}


/**
 * @brief Allocates a tensor initialized to a caller-provided scalar.
 *
 * @param shape Tensor dimensions.
 * @param value Scalar value assigned to every element.
 * @param device_type Target device.
 * @param stream CUDA stream used for GPU initialization.
 * @param dtype Element storage type.
 * @return Newly allocated tensor containing @p value.
 */
Tensor Tensor::full(
    std::vector<std::size_t> shape,
    const float value,
    const DeviceType device_type,
    cudaStream_t stream,
    const Dtype dtype) {
    Tensor result(std::move(shape), device_type, dtype);
    fill(result, value, stream);
    return result;
}


/**
 * @brief Creates a two-dimensional identity matrix.
 *
 * The result has shape `[rows, columns]`. Rectangular matrices contain ones on
 * the first `min(rows, columns)` diagonal positions and zeros elsewhere.
 *
 * @param rows Number of matrix rows.
 * @param columns Number of matrix columns.
 * @param device_type Target device.
 * @param stream CUDA stream used for GPU generation.
 * @param dtype Element storage type.
 * @return Newly allocated identity matrix.
 *
 * @throws std::runtime_error If CUDA kernel launch validation fails.
 *
 * @note GPU generation is asynchronous with respect to the host.
 */
Tensor Tensor::eye(
    std::size_t rows,
    std::size_t columns,
    const DeviceType device_type,
    cudaStream_t stream,
    const Dtype dtype
) {
    Tensor result(std::vector<std::size_t>{rows, columns}, device_type, dtype);

    const std::size_t count = rows * columns;

    if (device_type != DeviceType::CUDA) {
        std::vector<float> values(count, 0.0f);
        for (std::size_t diagonal = 0;
             diagonal < std::min(rows, columns);
             ++diagonal) {
            values[diagonal * columns + diagonal] = 1.0f;
        }
        result.copy_from_host(values);
        return result;
    }

    constexpr int threads = 256;
    const int blocks = static_cast<int>((count + threads - 1) / threads);

    if (dtype == Dtype::F16) {
        eye_kernel<<<blocks, threads, 0, stream>>>(
            static_cast<__half*>(result.raw_data()), rows, columns);
    } else if (dtype == Dtype::BF16) {
        eye_kernel<<<blocks, threads, 0, stream>>>(
            static_cast<__nv_bfloat16*>(result.raw_data()), rows, columns);
    } else {
        eye_kernel<<<blocks, threads, 0, stream>>>(
            static_cast<float*>(result.raw_data()), rows, columns);
    }

    CUDA_CHECK(cudaGetLastError());
    return result;
}
