#include "ops/rmsnorm.h"
#include "core/cuda_check.h"
#include "device_guard.h"
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace {

__inline__ __device__ float warp_reduce_sum(float value) {
    constexpr unsigned mask = 0xffffffffu;
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(mask, value, offset);
    }
    return value;
}

__inline__ __device__ float block_reduce_sum(float value) {
    value = warp_reduce_sum(value);

    __shared__ float warp_sums[8];
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int warp = static_cast<int>(threadIdx.x) >> 5;
    if (lane == 0) {
        warp_sums[warp] = value;
    }
    __syncthreads();

    if (warp == 0) {
        const int warp_count = (static_cast<int>(blockDim.x) + 31) >> 5;
        value = lane < warp_count ? warp_sums[lane] : 0.0f;
        value = warp_reduce_sum(value);
        if (lane == 0) {
            warp_sums[0] = value;
        }
    }
    __syncthreads();
    return warp_sums[0];
}

template <int ItemsPerThread>
__launch_bounds__(256)
__global__ void rmsnorm_vectorized_cached_kernel(
    float* __restrict__ output,
    const float* __restrict__ input,
    const float* __restrict__ weight,
    int vector_count,
    float inverse_hidden,
    float epsilon
) {
    const int tid = static_cast<int>(threadIdx.x);
    const int block_size = static_cast<int>(blockDim.x);
    const std::size_t row_offset =
        static_cast<std::size_t>(blockIdx.x) * vector_count;
    const auto* row_input =
        reinterpret_cast<const float4*>(input) + row_offset;
    auto* row_output = reinterpret_cast<float4*>(output) + row_offset;

    float4 values[ItemsPerThread];
    float sum_squares = 0.0f;
#pragma unroll
    for (int item = 0; item < ItemsPerThread; ++item) {
        const int index = tid + item * block_size;
        float4 value = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        if (index < vector_count) {
            value = row_input[index];
            sum_squares = fmaf(value.x, value.x, sum_squares);
            sum_squares = fmaf(value.y, value.y, sum_squares);
            sum_squares = fmaf(value.z, value.z, sum_squares);
            sum_squares = fmaf(value.w, value.w, sum_squares);
        }
        values[item] = value;
    }

    const float inv_rms =
        rsqrtf(block_reduce_sum(sum_squares) * inverse_hidden + epsilon);
    const auto* weight_vectors = reinterpret_cast<const float4*>(weight);

#pragma unroll
    for (int item = 0; item < ItemsPerThread; ++item) {
        const int index = tid + item * block_size;
        if (index < vector_count) {
            const float4 value = values[item];
            const float4 scale = weight_vectors[index];
            row_output[index] = make_float4(
                value.x * inv_rms * scale.x,
                value.y * inv_rms * scale.y,
                value.z * inv_rms * scale.z,
                value.w * inv_rms * scale.w);
        }
    }
}

template <typename T, int ItemsPerThread>
__launch_bounds__(256)
__global__ void rmsnorm_scalar_cached_kernel(
    T* __restrict__ output,
    const T* __restrict__ input,
    const T* __restrict__ weight,
    int hidden,
    float inverse_hidden,
    float epsilon
) {
    const int tid = static_cast<int>(threadIdx.x);
    const int block_size = static_cast<int>(blockDim.x);
    const std::size_t row_offset =
        static_cast<std::size_t>(blockIdx.x) * hidden;
    const T* row_input = input + row_offset;
    T* row_output = output + row_offset;

    float values[ItemsPerThread];
    float sum_squares = 0.0f;
#pragma unroll
    for (int item = 0; item < ItemsPerThread; ++item) {
        const int index = tid + item * block_size;
        float value = 0.0f;
        if (index < hidden) {
            value = static_cast<float>(row_input[index]);
            sum_squares = fmaf(value, value, sum_squares);
        }
        values[item] = value;
    }

    const float inv_rms =
        rsqrtf(block_reduce_sum(sum_squares) * inverse_hidden + epsilon);

#pragma unroll
    for (int item = 0; item < ItemsPerThread; ++item) {
        const int index = tid + item * block_size;
        if (index < hidden) {
            row_output[index] = static_cast<T>(
                values[item] * inv_rms * static_cast<float>(weight[index]));
        }
    }
}

template <bool Vectorized>
__launch_bounds__(256)
__global__ void rmsnorm_streaming_kernel(
    float* __restrict__ output,
    const float* __restrict__ input,
    const float* __restrict__ weight,
    int hidden,
    float inverse_hidden,
    float epsilon
) {
    const int tid = static_cast<int>(threadIdx.x);
    const int block_size = static_cast<int>(blockDim.x);
    const std::size_t row_offset =
        static_cast<std::size_t>(blockIdx.x) * hidden;
    const float* row_input = input + row_offset;
    float* row_output = output + row_offset;

    float sum_squares = 0.0f;
    if constexpr (Vectorized) {
        const auto* input_vectors =
            reinterpret_cast<const float4*>(row_input);
        for (int index = tid; index < hidden / 4; index += block_size) {
            const float4 value = input_vectors[index];
            sum_squares = fmaf(value.x, value.x, sum_squares);
            sum_squares = fmaf(value.y, value.y, sum_squares);
            sum_squares = fmaf(value.z, value.z, sum_squares);
            sum_squares = fmaf(value.w, value.w, sum_squares);
        }
    } else {
        for (int index = tid; index < hidden; index += block_size) {
            const float value = row_input[index];
            sum_squares = fmaf(value, value, sum_squares);
        }
    }

    const float inv_rms =
        rsqrtf(block_reduce_sum(sum_squares) * inverse_hidden + epsilon);

    if constexpr (Vectorized) {
        const auto* input_vectors =
            reinterpret_cast<const float4*>(row_input);
        auto* output_vectors = reinterpret_cast<float4*>(row_output);
        const auto* weight_vectors =
            reinterpret_cast<const float4*>(weight);
        for (int index = tid; index < hidden / 4; index += block_size) {
            const float4 value = input_vectors[index];
            const float4 scale = weight_vectors[index];
            output_vectors[index] = make_float4(
                value.x * inv_rms * scale.x,
                value.y * inv_rms * scale.y,
                value.z * inv_rms * scale.z,
                value.w * inv_rms * scale.w);
        }
    } else {
        for (int index = tid; index < hidden; index += block_size) {
            row_output[index] =
                row_input[index] * inv_rms * weight[index];
        }
    }
}

template <typename T>
__launch_bounds__(256)
__global__ void rmsnorm_scalar_streaming_kernel(
    T* __restrict__ output,
    const T* __restrict__ input,
    const T* __restrict__ weight,
    int hidden,
    float inverse_hidden,
    float epsilon
) {
    const int tid = static_cast<int>(threadIdx.x);
    const int block_size = static_cast<int>(blockDim.x);
    const std::size_t row_offset =
        static_cast<std::size_t>(blockIdx.x) * hidden;
    const T* row_input = input + row_offset;
    T* row_output = output + row_offset;

    float sum_squares = 0.0f;
    for (int index = tid; index < hidden; index += block_size) {
        const float value = static_cast<float>(row_input[index]);
        sum_squares = fmaf(value, value, sum_squares);
    }

    const float inv_rms =
        rsqrtf(block_reduce_sum(sum_squares) * inverse_hidden + epsilon);

    for (int index = tid; index < hidden; index += block_size) {
        row_output[index] = static_cast<T>(
            static_cast<float>(row_input[index]) * inv_rms *
            static_cast<float>(weight[index]));
    }
}

void validate_rmsnorm(
    const Tensor& output,
    const Tensor& input,
    const Tensor& weight,
    float epsilon
) {
    if (!std::isfinite(epsilon) || epsilon <= 0.0f) {
        throw std::invalid_argument("RMSNorm epsilon must be finite and positive");
    }
    if (input.device_type() != DeviceType::CUDA ||
        weight.device_type() != DeviceType::CUDA ||
        output.device_type() != DeviceType::CUDA) {
        throw std::invalid_argument("RMSNorm requires CUDA tensors");
    }
    if (!is_floating_point(input.dtype()) ||
        input.dtype() != weight.dtype() || input.dtype() != output.dtype()) {
        throw std::invalid_argument(
            "RMSNorm input, weight, and output must have the same floating dtype");
    }
    if (input.shape().size() != 2) {
        throw std::invalid_argument("RMSNorm input must have shape [rows, hidden]");
    }
    if (weight.shape().size() != 1) {
        throw std::invalid_argument("RMSNorm weight must have shape [hidden]");
    }

    const std::size_t rows_size = input.size(0);
    const std::size_t hidden_size = input.size(1);
    if (weight.size(0) != hidden_size) {
        throw std::invalid_argument(
            "RMSNorm weight size must match input hidden size"
        );
    }
    if (output.shape() != input.shape()) {
        throw std::invalid_argument(
            "RMSNorm output shape must match input shape"
        );
    }
    if (rows_size > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
        hidden_size > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::invalid_argument("RMSNorm dimensions exceed supported range");
    }
}

template <typename T>
void launch_scalar_rmsnorm(
    Tensor& output,
    const Tensor& input,
    const Tensor& weight,
    int rows,
    int hidden,
    float inverse_hidden,
    float epsilon,
    cudaStream_t stream
) {
    const int threads = hidden <= 32 ? 32 : hidden <= 256 ? 128 : 256;
    const int items_per_thread = (hidden + threads - 1) / threads;
    auto* output_data = static_cast<T*>(output.raw_data());
    const auto* input_data = static_cast<const T*>(input.raw_data());
    const auto* weight_data = static_cast<const T*>(weight.raw_data());

    if (items_per_thread == 1) {
        rmsnorm_scalar_cached_kernel<T, 1><<<rows, threads, 0, stream>>>(
            output_data, input_data, weight_data, hidden, inverse_hidden, epsilon);
    } else if (items_per_thread == 2) {
        rmsnorm_scalar_cached_kernel<T, 2><<<rows, threads, 0, stream>>>(
            output_data, input_data, weight_data, hidden, inverse_hidden, epsilon);
    } else if (items_per_thread <= 4) {
        rmsnorm_scalar_cached_kernel<T, 4><<<rows, threads, 0, stream>>>(
            output_data, input_data, weight_data, hidden, inverse_hidden, epsilon);
    } else {
        rmsnorm_scalar_streaming_kernel<T><<<rows, threads, 0, stream>>>(
            output_data, input_data, weight_data, hidden, inverse_hidden, epsilon);
    }
}

} // namespace

void rmsnorm_forward(
    Tensor& output,
    const Tensor& input,
    const Tensor& weight,
    float epsilon,
    cudaStream_t stream
) {
    validate_rmsnorm(output, input, weight, epsilon);

    const std::size_t rows_size = input.size(0);
    const std::size_t hidden_size = input.size(1);

    DeviceGuard device_guard(0);

    const auto rows = static_cast<int>(rows_size);
    const auto hidden = static_cast<int>(hidden_size);

    const float inverse_hidden = 1.0f / static_cast<float>(hidden);

    if (input.dtype() == Dtype::F32 &&
        hidden % 4 == 0 &&
        reinterpret_cast<std::uintptr_t>(input.raw_data()) % alignof(float4) == 0 &&
        reinterpret_cast<std::uintptr_t>(output.raw_data()) % alignof(float4) == 0 &&
        reinterpret_cast<std::uintptr_t>(weight.raw_data()) % alignof(float4) == 0) {
        const int vector_count = hidden / 4;
        const int threads = vector_count <= 32 ? 32 :
                            vector_count <= 256 ? 128 : 256;
        const int items_per_thread =
            (vector_count + threads - 1) / threads;
        if (items_per_thread == 1) {
            rmsnorm_vectorized_cached_kernel<1><<<rows, threads, 0, stream>>>(
                output.data(), input.data(), weight.data(), vector_count,
                inverse_hidden, epsilon);
        } else if (items_per_thread == 2) {
            rmsnorm_vectorized_cached_kernel<2><<<rows, threads, 0, stream>>>(
                output.data(), input.data(), weight.data(), vector_count,
                inverse_hidden, epsilon);
        } else if (items_per_thread <= 4) {
            rmsnorm_vectorized_cached_kernel<4><<<rows, threads, 0, stream>>>(
                output.data(), input.data(), weight.data(), vector_count,
                inverse_hidden, epsilon);
        } else {
            rmsnorm_streaming_kernel<true><<<rows, threads, 0, stream>>>(
                output.data(), input.data(), weight.data(), hidden,
                inverse_hidden, epsilon);
        }
    } else if (input.dtype() == Dtype::F16) {
        launch_scalar_rmsnorm<__half>(
            output, input, weight, rows, hidden, inverse_hidden, epsilon, stream);
    } else if (input.dtype() == Dtype::BF16) {
        launch_scalar_rmsnorm<__nv_bfloat16>(
            output, input, weight, rows, hidden, inverse_hidden, epsilon, stream);
    } else {
        launch_scalar_rmsnorm<float>(
            output, input, weight, rows, hidden, inverse_hidden, epsilon, stream);
    }

    CUDA_CHECK(cudaGetLastError());
}

Tensor rmsnorm_forward(
    const Tensor& input,
    const Tensor& weight,
    float epsilon,
    cudaStream_t stream
) {
    Tensor output(input.shape(), input.device_type(), input.dtype());
    rmsnorm_forward(output, input, weight, epsilon, stream);
    return output;
}
