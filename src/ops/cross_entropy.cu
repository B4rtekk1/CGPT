#include "ops/cross_entropy.h"

#include "core/cuda_check.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cmath>
#include <stdexcept>

namespace {

constexpr int kThreads = 256;

template <typename T>
__global__ void cross_entropy_loss_kernel(
    float* loss, const T* logits, const bpe::TokenId* targets,
    const std::size_t rows, const std::size_t vocabulary_size) {
    const std::size_t row = blockIdx.x;
    if (row >= rows) return;

    __shared__ float reductions[kThreads];
    const T* row_logits = logits + row * vocabulary_size;
    float maximum = -INFINITY;
    for (std::size_t column = threadIdx.x; column < vocabulary_size; column += blockDim.x) {
        maximum = fmaxf(maximum, static_cast<float>(row_logits[column]));
    }
    reductions[threadIdx.x] = maximum;
    __syncthreads();
    for (int stride = static_cast<int>(blockDim.x) / 2; stride > 0; stride /= 2) {
        if (threadIdx.x < stride) reductions[threadIdx.x] = fmaxf(reductions[threadIdx.x], reductions[threadIdx.x + stride]);
        __syncthreads();
    }
    maximum = reductions[0];

    float sum = 0.0f;
    for (std::size_t column = threadIdx.x; column < vocabulary_size; column += blockDim.x) {
        sum += expf(static_cast<float>(row_logits[column]) - maximum);
    }
    reductions[threadIdx.x] = sum;
    __syncthreads();
    for (int stride = static_cast<int>(blockDim.x) / 2; stride > 0; stride /= 2) {
        if (threadIdx.x < stride) reductions[threadIdx.x] += reductions[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        const bpe::TokenId target = targets[row];
        if (target < vocabulary_size) {
            atomicAdd(loss, (logf(reductions[0]) + maximum - static_cast<float>(row_logits[target])) /
                            static_cast<float>(rows));
        }
    }
}

template <typename T>
__global__ void cross_entropy_gradient_kernel(
    T* gradient, const T* logits, const bpe::TokenId* targets,
    const std::size_t rows, const std::size_t vocabulary_size) {
    const std::size_t row = blockIdx.x;
    if (row >= rows) return;

    __shared__ float reductions[kThreads];
    const T* row_logits = logits + row * vocabulary_size;
    float maximum = -INFINITY;
    for (std::size_t column = threadIdx.x; column < vocabulary_size; column += blockDim.x) {
        maximum = fmaxf(maximum, static_cast<float>(row_logits[column]));
    }
    reductions[threadIdx.x] = maximum;
    __syncthreads();
    for (int stride = static_cast<int>(blockDim.x) / 2; stride > 0; stride /= 2) {
        if (threadIdx.x < stride) reductions[threadIdx.x] = fmaxf(reductions[threadIdx.x], reductions[threadIdx.x + stride]);
        __syncthreads();
    }
    maximum = reductions[0];

    float sum = 0.0f;
    for (std::size_t column = threadIdx.x; column < vocabulary_size; column += blockDim.x) {
        sum += expf(static_cast<float>(row_logits[column]) - maximum);
    }
    reductions[threadIdx.x] = sum;
    __syncthreads();
    for (int stride = static_cast<int>(blockDim.x) / 2; stride > 0; stride /= 2) {
        if (threadIdx.x < stride) reductions[threadIdx.x] += reductions[threadIdx.x + stride];
        __syncthreads();
    }

    const bpe::TokenId target = targets[row];
    T* row_gradient = gradient + row * vocabulary_size;
    const float scale = 1.0f / static_cast<float>(rows);
    for (std::size_t column = threadIdx.x; column < vocabulary_size; column += blockDim.x) {
        float value = expf(static_cast<float>(row_logits[column]) - maximum) / reductions[0];
        if (column == target) value -= 1.0f;
        row_gradient[column] = static_cast<T>(value * scale);
    }
}

void validate(const Tensor& loss, const Tensor& gradient, const Tensor& logits,
              const bpe::TokenId* targets, const std::size_t target_count) {
    if (targets == nullptr || logits.device_type() != DeviceType::CUDA || logits.dim() != 2 ||
        !is_floating_point(logits.dtype()) || logits.size(0) == 0 || logits.size(1) == 0) {
        throw std::invalid_argument("cross_entropy: logits and targets must be non-empty CUDA inputs");
    }
    if (target_count != logits.size(0) || gradient.device_type() != DeviceType::CUDA ||
        gradient.dtype() != logits.dtype() || gradient.shape() != logits.shape()) {
        throw std::invalid_argument("cross_entropy: invalid gradient or target count");
    }
    if (loss.device_type() != DeviceType::CUDA || loss.dtype() != Dtype::F32 || loss.shape() != std::vector<std::size_t>{1}) {
        throw std::invalid_argument("cross_entropy: loss must be a CUDA F32 tensor with shape [1]");
    }
}

template <typename T>
void launch(Tensor& loss, Tensor& gradient, const Tensor& logits, const bpe::TokenId* targets,
            const std::size_t rows, const std::size_t vocabulary_size, cudaStream_t stream) {
    cross_entropy_loss_kernel<T><<<static_cast<unsigned>(rows), kThreads, 0, stream>>>(
        static_cast<float*>(loss.raw_data()), static_cast<const T*>(logits.raw_data()), targets, rows, vocabulary_size);
    cross_entropy_gradient_kernel<T><<<static_cast<unsigned>(rows), kThreads, 0, stream>>>(
        static_cast<T*>(gradient.raw_data()), static_cast<const T*>(logits.raw_data()), targets, rows, vocabulary_size);
}

} // namespace

void cross_entropy_forward_backward(Tensor& loss, Tensor& gradient, const Tensor& logits,
                                    const bpe::TokenId* device_targets,
                                    const std::size_t target_count, cudaStream_t stream) {
    validate(loss, gradient, logits, device_targets, target_count);
    CUDA_CHECK(cudaMemsetAsync(loss.raw_data(), 0, sizeof(float), stream));
    const auto rows = logits.size(0);
    const auto vocabulary_size = logits.size(1);
    switch (logits.dtype()) {
        case Dtype::F32: launch<float>(loss, gradient, logits, device_targets, rows, vocabulary_size, stream); break;
        case Dtype::F16: launch<__half>(loss, gradient, logits, device_targets, rows, vocabulary_size, stream); break;
        case Dtype::BF16: launch<__nv_bfloat16>(loss, gradient, logits, device_targets, rows, vocabulary_size, stream); break;
        default: throw std::invalid_argument("cross_entropy: unsupported logits dtype");
    }
    CUDA_CHECK(cudaGetLastError());
}
