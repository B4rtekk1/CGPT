#include "ops/rmsnorm.h"
#include "cuda_check.h"
#include <cmath>
#include <limits>
#include <stdexcept>

__inline__ __device__ float warp_reduce_sum(float value) {
    for (int offset = 16; offset > 0; offset /= 2) {
        value += __shfl_down_sync(0xffffffff, value, offset);
    }
    return value;
}

__global__ void rmsnorm_kernel(
    float* __restrict__ output,
    const float* __restrict__ input,
    const float* __restrict__ weight,
    int hidden,
    float epsilon
) {
    const int row = static_cast<int>(blockIdx.x);
    const int tid = static_cast<int>(threadIdx.x);
    const int block_size = static_cast<int>(blockDim.x);

    const float* row_input = input + static_cast<size_t>(row) * hidden;
    float* row_output = output + static_cast<size_t>(row) * hidden;

    float sum_squares = 0.0f;

    for (int i = tid; i < hidden; i += block_size) {
        const float x = row_input[i];
        sum_squares = fmaf(x, x, sum_squares);
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

    __shared__ float inv_rms;
    if (tid == 0) {
        inv_rms = rsqrtf(total / static_cast<float>(hidden) + epsilon);
    }
    __syncthreads();

    for (int i = tid; i < hidden; i += block_size) {
        row_output[i] = row_input[i] * inv_rms * weight[i];
    }
}

Tensor rmsnorm_forward(
    const Tensor& input,
    const Tensor& weight,
    float epsilon,
    cudaStream_t stream
) {
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

    const int rows = static_cast<int>(rows_size);
    const int hidden = static_cast<int>(hidden_size);
    Tensor output(input.shape(), DeviceType::CUDA);

    constexpr int threads = 256;
    rmsnorm_kernel<<<rows, threads, 0, stream>>>(
        output.data(), input.data(), weight.data(), hidden, epsilon
    );

    CUDA_CHECK(cudaGetLastError());
    return output;
}
