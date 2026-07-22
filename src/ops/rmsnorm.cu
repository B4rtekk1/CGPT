#include "ops/rmsnorm.h"
#include "cuda_check.h"

#include <cmath>
#include <limits>
#include <stdexcept>

__global__ void rmsnorm_kernel(
    float* output,
    const float* input,
    const float* weight,
    int hidden,
    float epsilon
) {
    const int row = static_cast<int>(blockIdx.x);
    const int tid = static_cast<int>(threadIdx.x);
    const int block_size = static_cast<int>(blockDim.x);

    extern __shared__ float shared_sum[];

    const float* row_input = input + row * hidden;
    float* row_output = output + row * hidden;

    float sum_squares = 0.0f;
    for (int i = tid; i < hidden; i += block_size) {
        const float value = row_input[i];
        sum_squares += value * value;
    }

    shared_sum[tid] = sum_squares;
    __syncthreads();

    for (int stride = block_size / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            shared_sum[tid] += shared_sum[tid + stride];
        }
        __syncthreads();
    }

    const float rms = sqrtf(shared_sum[0] / static_cast<float>(hidden) + epsilon);

    for (int i = tid; i < hidden; i += block_size) {
        row_output[i] = row_input[i] / rms * weight[i];
    }
}

Tensor rmsnorm_forward(
    const Tensor& input,
    const Tensor& weight,
    float epsilon,
    cudaStream_t stream
) {
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
    Tensor output(input.shape());
    constexpr int threads = 256;

    rmsnorm_kernel<<<rows, threads, threads * sizeof(float), stream>>>(
        output.data(), input.data(), weight.data(), hidden, epsilon
    );

    CUDA_CHECK(cudaGetLastError());
    return output;
}
