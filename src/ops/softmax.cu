/**
 * @file softmax.cu
 * @brief CUDA implementation of numerically stable row-wise softmax.
 *
 * The implementation performs row-wise maximum and sum reductions in FP32,
 * stores centered exponentials once, and reuses them for final normalization.
 */

#include "ops/softmax.h"

#include "core/cuda_check.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <math_constants.h>

#include <stdexcept>

namespace {

constexpr int kThreads = 512;
constexpr int kWarpSize = 32;
constexpr int kWarps = kThreads / kWarpSize;

template <typename T>
/** @brief Converts a supported CUDA scalar value to FP32. */
__device__ __forceinline__ float to_float(const T value) { return static_cast<float>(value); }

template <typename T>
/** @brief Converts an FP32 value to a supported CUDA scalar type. */
__device__ __forceinline__ T from_float(const float value) { return static_cast<T>(value); }

/** @brief Reduces values to a block-wide maximum using warp shuffles. */
__device__ __forceinline__ float reduce_max(float value, float* const scratch) {
    const int lane = threadIdx.x & (kWarpSize - 1);
    const int warp = threadIdx.x / kWarpSize;
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        value = fmaxf(value, __shfl_down_sync(0xffffffffu, value, offset));
    if (lane == 0) scratch[warp] = value;
    __syncthreads();
    if (warp == 0) {
        value = lane < kWarps ? scratch[lane] : -CUDART_INF_F;
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            value = fmaxf(value, __shfl_down_sync(0xffffffffu, value, offset));
        if (lane == 0) scratch[0] = value;
    }
    __syncthreads();
    return scratch[0];
}

/** @brief Reduces values to a block-wide sum using warp shuffles. */
__device__ __forceinline__ float reduce_sum(float value, float* const scratch) {
    const int lane = threadIdx.x & (kWarpSize - 1);
    const int warp = threadIdx.x / kWarpSize;
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        value += __shfl_down_sync(0xffffffffu, value, offset);
    if (lane == 0) scratch[warp] = value;
    __syncthreads();
    if (warp == 0) {
        value = lane < kWarps ? scratch[lane] : 0.0f;
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            value += __shfl_down_sync(0xffffffffu, value, offset);
        if (lane == 0) scratch[0] = value;
    }
    __syncthreads();
    return scratch[0];
}

template <typename T>
/** @brief Computes one row-wise softmax using FP32 reductions. */
__global__ __launch_bounds__(kThreads)
void softmax_fused_kernel(T* const __restrict__ output, const T* const __restrict__ input,
                          const std::size_t rows, const std::size_t columns) {
    const std::size_t row = blockIdx.x;
    if (row >= rows) return;

    __shared__ float scratch[kWarps];
    constexpr std::size_t stride = kThreads;
    constexpr std::size_t unrolled_stride = 4 * stride;
    const std::size_t tid = threadIdx.x;
    const T* const row_input = input + row * columns;
    T* const row_output = output + row * columns;

    float m0 = -CUDART_INF_F, m1 = -CUDART_INF_F, m2 = -CUDART_INF_F, m3 = -CUDART_INF_F;
    std::size_t column = tid;
    for (; column + 3 * stride < columns; column += unrolled_stride) {
        m0 = fmaxf(m0, to_float(row_input[column]));
        m1 = fmaxf(m1, to_float(row_input[column + stride]));
        m2 = fmaxf(m2, to_float(row_input[column + 2 * stride]));
        m3 = fmaxf(m3, to_float(row_input[column + 3 * stride]));
    }
    float local_max = fmaxf(fmaxf(m0, m1), fmaxf(m2, m3));
    for (; column < columns; column += stride) local_max = fmaxf(local_max, to_float(row_input[column]));
    const float maximum = reduce_max(local_max, scratch);

    // Store exp(x - max) once and reuse it after the sum reduction.  This
    // avoids a second exp pass; reductions remain FP32 for stable normalization.
    float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
    column = tid;
    for (; column + 3 * stride < columns; column += unrolled_stride) {
        const float e0 = __expf(to_float(row_input[column]) - maximum);
        const float e1 = __expf(to_float(row_input[column + stride]) - maximum);
        const float e2 = __expf(to_float(row_input[column + 2 * stride]) - maximum);
        const float e3 = __expf(to_float(row_input[column + 3 * stride]) - maximum);
        s0 += e0; s1 += e1; s2 += e2; s3 += e3;
        row_output[column] = from_float<T>(e0);
        row_output[column + stride] = from_float<T>(e1);
        row_output[column + 2 * stride] = from_float<T>(e2);
        row_output[column + 3 * stride] = from_float<T>(e3);
    }
    float local_sum = (s0 + s1) + (s2 + s3);
    for (; column < columns; column += stride) {
        const float value = __expf(to_float(row_input[column]) - maximum);
        local_sum += value;
        row_output[column] = from_float<T>(value);
    }
    const float inverse_sum = __frcp_rn(reduce_sum(local_sum, scratch));

    for (column = tid; column < columns; column += stride)
        row_output[column] = from_float<T>(to_float(row_output[column]) * inverse_sum);
}

/** @brief Validates the shape, device, and dtype contract for softmax. */
void validate(const Tensor& output, const Tensor& input) {
    if (input.device_type() != DeviceType::CUDA || input.dim() != 2 ||
        !is_floating_point(input.dtype()) || input.size(0) == 0 || input.size(1) == 0 ||
        output.device_type() != DeviceType::CUDA || output.dtype() != input.dtype() ||
        output.shape() != input.shape())
        throw std::invalid_argument("softmax: input and output must be non-empty matching 2D CUDA floating-point tensors");
}

template <typename T>
/** @brief Launches the fused softmax kernel for a concrete element type. */
void launch(Tensor& output, const Tensor& input, const cudaStream_t stream) {
    softmax_fused_kernel<T><<<static_cast<unsigned>(input.size(0)), kThreads, 0, stream>>>(
        static_cast<T*>(output.raw_data()), static_cast<const T*>(input.raw_data()), input.size(0), input.size(1));
}

} // namespace

/**
 * @brief Dispatches the CUDA softmax implementation for the input dtype.
 * @param output Destination tensor.
 * @param input Source logits tensor.
 * @param stream CUDA stream used for the operation.
 */
void softmax_forward(Tensor& output, const Tensor& input, const cudaStream_t stream) {
    validate(output, input);
    switch (input.dtype()) {
        case Dtype::F32: launch<float>(output, input, stream); break;
        case Dtype::F16: launch<__half>(output, input, stream); break;
        case Dtype::BF16: launch<__nv_bfloat16>(output, input, stream); break;
        default: throw std::invalid_argument("softmax: unsupported input dtype");
    }
    CUDA_CHECK(cudaGetLastError());
}
