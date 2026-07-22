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
    const int row = blockIdx.x;
    const int tid = threadIdx.x;

    const float* row_input = input + static_cast<size_t>(row) * hidden;
    float* row_output = output + static_cast<size_t>(row) * hidden;

    float sum_squares = 0.0f;

    for (int i = tid; i < hidden; i += blockDim.x) {
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
        const int warp_count = (blockDim.x + 31) / 32;
        total = lane < warp_count ? warp_sums[lane] : 0.0f;
        total = warp_reduce_sum(total);
    }

    __shared__ float inv_rms;
    if (tid == 0) {
        inv_rms = rsqrtf(total / static_cast<float>(hidden) + epsilon);
    }
    __syncthreads();

    for (int i = tid; i < hidden; i += blockDim.x) {
        row_output[i] = row_input[i] * inv_rms * weight[i];
    }
}