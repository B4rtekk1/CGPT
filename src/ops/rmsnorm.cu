#include "ops/rmsnorm.h"
#include "cuda_check.h"
#include "device_guard.h"
#include <cmath>
#include <limits>
#include <stdexcept>

namespace {

__inline__ __device__ float warp_reduce_sum(float value) {
    // Using the active-lane mask also keeps this correct for a partial warp.
    const unsigned mask = __activemask();
    for (int offset = 16; offset > 0; offset /= 2) {
        value += __shfl_down_sync(mask, value, offset);
    }
    return value;
}

template <bool Vectorized>
__global__ void rmsnorm_kernel(
    float* __restrict__ output,
    const float* __restrict__ input,
    const float* __restrict__ weight,
    int hidden,
    float epsilon
) {
    const auto row = static_cast<int>(blockIdx.x);
    const auto tid = static_cast<int>(threadIdx.x);
    const auto block_size = static_cast<int>(blockDim.x);

    const float* row_input = input + static_cast<size_t>(row) * hidden;
    float* row_output = output + static_cast<size_t>(row) * hidden;

    float sum_squares = 0.0f;

    if constexpr (Vectorized) {
        const auto row_input_vec = reinterpret_cast<const float4*>(row_input);
        for (int i = tid; i < hidden / 4; i += block_size) {
            const float4 x = row_input_vec[i];
            sum_squares = fmaf(x.x, x.x, sum_squares);
            sum_squares = fmaf(x.y, x.y, sum_squares);
            sum_squares = fmaf(x.z, x.z, sum_squares);
            sum_squares = fmaf(x.w, x.w, sum_squares);
        }
    } else {
        for (int i = tid; i < hidden; i += block_size) {
            const float x = row_input[i];
            sum_squares = fmaf(x, x, sum_squares);
        }
    }

    sum_squares = warp_reduce_sum(sum_squares);

    __shared__ float warp_sums[32];

    const int lane = tid & 31;
    const int warp = tid >> 5;

    if (lane == 0) {
        warp_sums[warp] = sum_squares;
    }
    __syncthreads();

    float total = 0.0f;
    if (warp == 0) {
        const int warp_count = (block_size + 31) / 32;
        total = lane < warp_count ? warp_sums[lane] : 0.0f;
        total = warp_reduce_sum(total);
    }

    __shared__ float inv_rms = 0.0f;
    if (tid == 0) {
        inv_rms = rsqrtf(total / static_cast<float>(hidden) + epsilon);
    }
    __syncthreads();

    if constexpr (Vectorized) {
        const auto row_input_vec = reinterpret_cast<const float4*>(row_input);
        auto row_output_vec = reinterpret_cast<float4*>(row_output);
        const auto weight_vec = reinterpret_cast<const float4*>(weight);
        for (int i = tid; i < hidden / 4; i += block_size) {
            const float4 x = row_input_vec[i];
            const float4 w = weight_vec[i];
            row_output_vec[i] = make_float4(
                x.x * inv_rms * w.x,
                x.y * inv_rms * w.y,
                x.z * inv_rms * w.z,
                x.w * inv_rms * w.w
            );
        }
    } else {
        for (int i = tid; i < hidden; i += block_size) {
            row_output[i] = row_input[i] * inv_rms * weight[i];
        }
    }
}

} // namespace

Tensor rmsnorm_forward(
    const Tensor& input,
    const Tensor& weight,
    float epsilon,
    cudaStream_t stream
) {
    if (!std::isfinite(epsilon) || epsilon <= 0.0f) {
        throw std::invalid_argument("RMSNorm epsilon must be finite and positive");
    }
    if (input.device_type() != DeviceType::CUDA ||
        weight.device_type() != DeviceType::CUDA) {
        throw std::invalid_argument("RMSNorm requires CUDA tensors");
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
    if (rows_size > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
        hidden_size > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::invalid_argument("RMSNorm dimensions exceed supported range");
    }

    DeviceGuard device_guard(0);

    const auto rows = static_cast<int>(rows_size);
    const auto hidden = static_cast<int>(hidden_size);
    Tensor output(input.shape(), DeviceType::CUDA);

    if (rows == 0 || hidden == 0) {
        return output;
    }

    const bool vectorized = (hidden % 4) == 0;
    const int work_items = vectorized ? hidden / 4 : hidden;
    // Avoid launching mostly idle warps for small workloads. All selected
    // sizes are multiples of a warp, which keeps the reduction efficient.
    const int threads = work_items <= 32 ? 32 :
                        work_items <= 64 ? 64 :
                        work_items <= 128 ? 128 : 256;

    if (vectorized) {
        rmsnorm_kernel<true><<<rows, threads, 0, stream>>>(
            output.data(), input.data(), weight.data(), hidden, epsilon
        );
    } else {
        rmsnorm_kernel<false><<<rows, threads, 0, stream>>>(
            output.data(), input.data(), weight.data(), hidden, epsilon
        );
    }

    CUDA_CHECK(cudaGetLastError());
    return output;
}
