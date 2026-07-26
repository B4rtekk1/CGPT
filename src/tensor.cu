#include "../include/core/tensor.h"
#include "../include/core/cuda_check.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <algorithm>
#include <vector>

namespace {

template <typename T>
__global__ void fill_kernel(T* data, std::size_t count, float value) {
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count) {
        data[index] = static_cast<T>(value);
    }
}

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

template <typename T>
void launch_fill(Tensor& tensor, float value, cudaStream_t stream) {
    constexpr int threads = 256;
    const int blocks = static_cast<int>((tensor.numel() + threads - 1) / threads);
    fill_kernel<<<blocks, threads, 0, stream>>>(
        static_cast<T*>(tensor.raw_data()), tensor.numel(), value);
}

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

template <typename T>
std::vector<T> convert_from_float(const std::span<const float> source) {
    std::vector<T> converted(source.size());
    std::transform(source.begin(), source.end(), converted.begin(),
                   [](const float value) { return static_cast<T>(value); });
    return converted;
}

template <typename T>
void convert_to_float(const T* source, const std::span<float> destination) {
    std::transform(source, source + destination.size(), destination.begin(),
                   [](const T value) { return static_cast<float>(value); });
}

} // namespace

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

Tensor Tensor::empty(
    std::vector<std::size_t> shape,
    const DeviceType device_type,
    const Dtype dtype) {
    return Tensor(std::move(shape), device_type, dtype);
}

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
    }

    return result;
}

Tensor Tensor::ones(
    std::vector<std::size_t> shape,
    const DeviceType device_type,
    cudaStream_t stream,
    const Dtype dtype) {
    Tensor result(std::move(shape), device_type, dtype);
    fill(result, 1.0f, stream);
    return result;
}

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
