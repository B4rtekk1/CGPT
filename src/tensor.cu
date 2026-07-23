#include "../include/core/tensor.h"
#include "../include/core/cuda_check.h"

namespace {

__global__ void fill_kernel(float* data, std::size_t count, float value) {
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count) {
        data[index] = value;
    }
}

__global__ void eye_kernel(
    float* data,
    std::size_t rows,
    std::size_t columns
) {
    const std::size_t index =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    const std::size_t count = rows * columns;

    if (index < count) {
        const std::size_t row = index / columns;
        const std::size_t column = index % columns;

        data[index] = (row == column) ? 1.0f : 0.0f;
    }
}

void fill(Tensor& tensor, float value, cudaStream_t stream) {
    if (tensor.device_type() != DeviceType::CUDA) {
        std::vector<float> values(tensor.numel(), value);
        tensor.copy_from_host(values);
        return;
    }

    constexpr int threads = 256;
    const int blocks = static_cast<int>(
        (tensor.numel() + threads - 1) / threads);

    fill_kernel<<<blocks, threads, 0, stream>>>(
        tensor.data(), tensor.numel(), value);
}

} // namespace

Tensor Tensor::empty(
    std::vector<std::size_t> shape,
    DeviceType device,
    Dtype dtype) {
    return Tensor(std::move(shape), device, dtype);
}

Tensor Tensor::zeros(
    std::vector<std::size_t> shape,
    DeviceType device,
    cudaStream_t stream,
    Dtype dtype) {
    Tensor result(std::move(shape), device, dtype);

    if (device == DeviceType::CUDA) {
        // cudaMemset is faster than a kernel for zeros.
        CUDA_CHECK(cudaMemsetAsync(
            result.data(), 0, result.numel() * dtype_size(result.dtype()), stream));
    }

    return result;
}

Tensor Tensor::ones(
    std::vector<std::size_t> shape,
    DeviceType device,
    cudaStream_t stream,
    Dtype dtype) {
    Tensor result(std::move(shape), device, dtype);
    fill(result, 1.0f, stream);
    return result;
}

Tensor Tensor::full(
    std::vector<std::size_t> shape,
    float value,
    DeviceType device,
    cudaStream_t stream,
    Dtype dtype) {
    Tensor result(std::move(shape), device, dtype);
    fill(result, value, stream);
    return result;
}

Tensor Tensor::eye(
    std::size_t rows,
    std::size_t columns,
    DeviceType device,
    cudaStream_t stream,
    Dtype dtype
) {
    Tensor result(std::vector<std::size_t>{rows, columns}, device, dtype);

    const std::size_t count = rows * columns;

    if (device != DeviceType::CUDA) {
        for (std::size_t row = 0; row < rows; ++row) {
            for (std::size_t column = 0; column < columns; ++column) {
                result.data()[row * columns + column] = row == column ? 1.0f : 0.0f;
            }
        }
        return result;
    }

    constexpr int threads = 256;
    const int blocks = static_cast<int>((count + threads - 1) / threads);

    eye_kernel<<<blocks, threads, 0, stream>>>(
        result.data(), rows, columns
    );

    CUDA_CHECK(cudaGetLastError());
    return result;
}
