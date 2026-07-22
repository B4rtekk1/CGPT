#include "ops/rmsnorm.h"
#include "cuda_check.h"

#include <cmath>

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

void rmsnorm_forward(
    float* output,
    const float* input,
    const float* weight,
    int rows,
    int hidden,
    float epsilon,
    cudaStream_t stream
) {
    constexpr int threads = 256;

    rmsnorm_kernel<<<rows, threads, threads * sizeof(float), stream>>>(
        output, input, weight, hidden, epsilon
    );

    CUDA_CHECK(cudaGetLastError());
}
