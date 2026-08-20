/**
 * @file cross_entropy(3).cu
 * @brief Fused CUDA implementation of softmax cross-entropy and its gradient.
 *
 * The kernel supports FP32, FP16, and BF16 logits. It computes the numerically
 * stable softmax using a row-wise maximum, reuses the gradient buffer for
 * centered exponentials, and avoids allocating a separate probability matrix.
 */

#include "ops/cross_entropy.h"

#include "core/cuda_check.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <math_constants.h>

#include <stdexcept>

namespace {

// A GPT-sized vocabulary keeps all 16 warps busy while each thread owns
// enough values to expose instruction-level parallelism to the exp pipeline.
constexpr int kThreads = 512;
constexpr int kWarpSize = 32;
constexpr int kWarps = kThreads / kWarpSize;
static_assert(kThreads % kWarpSize == 0);

/**
 * @brief Converts a supported CUDA scalar type to FP32 for arithmetic.
 * @tparam T Source scalar type.
 * @param value Value to convert.
 * @return The value represented as a float.
 */
template <typename T>
__device__ __forceinline__ float to_float(T value) {
    return static_cast<float>(value);
}

/**
 * @brief Converts an FP32 arithmetic result to a supported output type.
 * @tparam T Destination scalar type.
 * @param value Value to convert.
 * @return The converted value.
 */
template <typename T>
__device__ __forceinline__ T from_float(float value) {
    return static_cast<T>(value);
}

/** @brief Performs a warp-wide maximum reduction. */
__device__ __forceinline__ float warp_reduce_max(float value) {
#pragma unroll
    for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
        value = fmaxf(value, __shfl_down_sync(0xffffffffu, value, offset));
    }
    return value;
}

/** @brief Performs a warp-wide sum reduction. */
__device__ __forceinline__ float warp_reduce_sum(float value) {
#pragma unroll
    for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(0xffffffffu, value, offset);
    }
    return value;
}

/**
 * @brief Performs a block-wide maximum reduction using warp scratch storage.
 * @param value Per-thread value.
 * @param warp_scratch Shared-memory storage with one slot per warp.
 * @return Block-wide maximum, available to every thread after synchronization.
 */
__device__ __forceinline__ float block_reduce_max(float value, float* warp_scratch) {
    const int lane = static_cast<int>(threadIdx.x) & (kWarpSize - 1);
    const int warp = static_cast<int>(threadIdx.x) / kWarpSize;

    value = warp_reduce_max(value);
    if (lane == 0) warp_scratch[warp] = value;
    __syncthreads();

    if (warp == 0) {
        value = lane < kWarps ? warp_scratch[lane] : -CUDART_INF_F;
        value = warp_reduce_max(value);
        if (lane == 0) warp_scratch[0] = value;
    }
    __syncthreads();
    return warp_scratch[0];
}

/**
 * @brief Performs a block-wide sum reduction using warp scratch storage.
 * @param value Per-thread value.
 * @param warp_scratch Shared-memory storage with one slot per warp.
 * @return Block-wide sum, available to every thread after synchronization.
 */
__device__ __forceinline__ float block_reduce_sum(float value, float* warp_scratch) {
    const int lane = static_cast<int>(threadIdx.x) & (kWarpSize - 1);
    const int warp = static_cast<int>(threadIdx.x) / kWarpSize;

    value = warp_reduce_sum(value);
    if (lane == 0) warp_scratch[warp] = value;
    __syncthreads();

    if (warp == 0) {
        value = lane < kWarps ? warp_scratch[lane] : 0.0f;
        value = warp_reduce_sum(value);
        if (lane == 0) warp_scratch[0] = value;
    }
    __syncthreads();
    return warp_scratch[0];
}

/**
 * @brief Fused row-wise softmax cross-entropy forward and backward kernel.
 *
 * One block handles one row. The kernel first finds the row maximum, computes
 * centered exponentials and their denominator, accumulates the mean loss, and
 * finally transforms the staged exponentials in @p gradient into dL/dlogits.
 * Invalid target IDs produce a zero target contribution while still producing
 * the softmax gradient.
 *
 * @tparam T Logit and gradient scalar type: float, __half, or __nv_bfloat16.
 * @param loss Device scalar receiving the accumulated mean loss.
 * @param gradient Output gradient with the same shape and dtype as @p logits.
 * @param logits Input logits with shape [rows, vocabulary_size].
 * @param targets Device target-token array.
 * @param rows Number of rows.
 * @param vocabulary_size Number of classes per row.
 * @param inv_rows Reciprocal of @p rows.
 */
template <typename T>
__global__ __launch_bounds__(kThreads)
void cross_entropy_fused_kernel(
    float* __restrict__ loss,
    T* __restrict__ gradient,
    const T* __restrict__ logits,
    const bpe::TokenId* __restrict__ targets,
    const std::size_t rows,
    const std::size_t vocabulary_size,
    const float inv_rows) {

    const std::size_t row = blockIdx.x;
    if (row >= rows) return;

    __shared__ float warp_scratch[kWarps];

    const T* __restrict__ row_logits = logits + row * vocabulary_size;
    T* __restrict__ row_gradient = gradient + row * vocabulary_size;

    constexpr auto kStride = static_cast<std::size_t>(kThreads);
    constexpr std::size_t kUnrolledStride = 4 * kStride;
    const auto tid = static_cast<std::size_t>(threadIdx.x);

    float max0 = -CUDART_INF_F;
    float max1 = -CUDART_INF_F;
    float max2 = -CUDART_INF_F;
    float max3 = -CUDART_INF_F;

    std::size_t column = tid;
    for (; column + 3 * kStride < vocabulary_size; column += kUnrolledStride) {
        max0 = fmaxf(max0, to_float(row_logits[column]));
        max1 = fmaxf(max1, to_float(row_logits[column + kStride]));
        max2 = fmaxf(max2, to_float(row_logits[column + 2 * kStride]));
        max3 = fmaxf(max3, to_float(row_logits[column + 3 * kStride]));
    }

    float local_max = fmaxf(fmaxf(max0, max1), fmaxf(max2, max3));
    for (; column < vocabulary_size; column += kStride) {
        local_max = fmaxf(local_max, to_float(row_logits[column]));
    }
    const float maximum = block_reduce_max(local_max, warp_scratch);


    // Accumulate the normalizer in FP32.  The final pass is intentionally
    // recomputed from logits rather than staging exponentials in gradient:
    // this removes a full gradient read plus an intermediate global write,
    // and avoids quantizing the intermediate for FP16/BF16 gradients.
    float sum0 = 0.0f;
    float sum1 = 0.0f;
    float sum2 = 0.0f;
    float sum3 = 0.0f;

    column = tid;
    for (; column + 3 * kStride < vocabulary_size; column += kUnrolledStride) {
        const float e0 = __expf(to_float(row_logits[column]) - maximum);
        const float e1 = __expf(to_float(row_logits[column + kStride]) - maximum);
        const float e2 = __expf(to_float(row_logits[column + 2 * kStride]) - maximum);
        const float e3 = __expf(to_float(row_logits[column + 3 * kStride]) - maximum);

        sum0 += e0;
        sum1 += e1;
        sum2 += e2;
        sum3 += e3;

    }

    float local_sum = (sum0 + sum1) + (sum2 + sum3);
    for (; column < vocabulary_size; column += kStride) {
        const float e = __expf(to_float(row_logits[column]) - maximum);
        local_sum += e;
    }
    const float denominator = block_reduce_sum(local_sum, warp_scratch);
    const float inv_denominator = __frcp_rn(denominator);

    const bpe::TokenId target = targets[row];
    const bool valid_target = target < vocabulary_size;


    if (threadIdx.x == 0 && valid_target) {
        const float target_logit = to_float(row_logits[static_cast<std::size_t>(target)]);
        const float row_loss = __logf(denominator) + maximum - target_logit;
        atomicAdd(loss, row_loss * inv_rows);
    }

    column = tid;
    for (; column + 3 * kStride < vocabulary_size; column += kUnrolledStride) {
        float v0 = __expf(to_float(row_logits[column]) - maximum) * inv_denominator;
        float v1 = __expf(to_float(row_logits[column + kStride]) - maximum) * inv_denominator;
        float v2 = __expf(to_float(row_logits[column + 2 * kStride]) - maximum) * inv_denominator;
        float v3 = __expf(to_float(row_logits[column + 3 * kStride]) - maximum) * inv_denominator;

        if (valid_target) {
            const auto target_column = static_cast<std::size_t>(target);
            if (column == target_column) v0 -= 1.0f;
            if (column + kStride == target_column) v1 -= 1.0f;
            if (column + 2 * kStride == target_column) v2 -= 1.0f;
            if (column + 3 * kStride == target_column) v3 -= 1.0f;
        }

        row_gradient[column] = from_float<T>(v0 * inv_rows);
        row_gradient[column + kStride] = from_float<T>(v1 * inv_rows);
        row_gradient[column + 2 * kStride] = from_float<T>(v2 * inv_rows);
        row_gradient[column + 3 * kStride] = from_float<T>(v3 * inv_rows);
    }

    for (; column < vocabulary_size; column += kStride) {
        float value = __expf(to_float(row_logits[column]) - maximum) * inv_denominator;
        if (valid_target && column == static_cast<std::size_t>(target)) value -= 1.0f;
        row_gradient[column] = from_float<T>(value * inv_rows);
    }
}

template <typename T>
__global__ __launch_bounds__(kThreads)
void cross_entropy_forward_kernel(
    float* __restrict__ loss,
    const T* __restrict__ logits,
    const bpe::TokenId* __restrict__ targets,
    const std::size_t rows,
    const std::size_t vocabulary_size,
    const float inv_rows) {

    const std::size_t row = blockIdx.x;
    if (row >= rows) return;
    __shared__ float warp_scratch[kWarps];
    const T* const row_logits = logits + row * vocabulary_size;
    constexpr auto kStride = static_cast<std::size_t>(kThreads);
    const auto tid = static_cast<std::size_t>(threadIdx.x);

    float local_max = -CUDART_INF_F;
    for (std::size_t column = tid; column < vocabulary_size; column += kStride)
        local_max = fmaxf(local_max, to_float(row_logits[column]));
    const float maximum = block_reduce_max(local_max, warp_scratch);

    float local_sum = 0.0f;
    for (std::size_t column = tid; column < vocabulary_size; column += kStride)
        local_sum += __expf(to_float(row_logits[column]) - maximum);
    const float denominator = block_reduce_sum(local_sum, warp_scratch);

    const bpe::TokenId target = targets[row];
    if (threadIdx.x == 0 && target < vocabulary_size) {
        const float row_loss = __logf(denominator) + maximum -
            to_float(row_logits[static_cast<std::size_t>(target)]);
        atomicAdd(loss, row_loss * inv_rows);
    }
}

/**
 * @brief Validates tensors and target metadata for cross-entropy.
 * @throws std::invalid_argument If inputs are not compatible CUDA tensors or
 *         if the target count does not match the number of rows.
 */
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
    if (loss.device_type() != DeviceType::CUDA || loss.dtype() != Dtype::F32 ||
        loss.shape() != std::vector<std::size_t>{1}) {
        throw std::invalid_argument("cross_entropy: loss must be a CUDA F32 tensor with shape [1]");
    }
}

/**
 * @brief Launches the fused cross-entropy kernel for one scalar type.
 * @tparam T CUDA scalar type used by logits and gradient.
 */
template <typename T>
void launch(Tensor& loss, Tensor& gradient, const Tensor& logits,
            const bpe::TokenId* targets, const std::size_t rows,
            const std::size_t vocabulary_size, cudaStream_t stream) {
    const float inv_rows = 1.0f / static_cast<float>(rows);

    cross_entropy_fused_kernel<T><<<static_cast<unsigned>(rows), kThreads, 0, stream>>>(
        static_cast<float*>(loss.raw_data()),
        static_cast<T*>(gradient.raw_data()),
        static_cast<const T*>(logits.raw_data()),
        targets,
        rows,
        vocabulary_size,
        inv_rows);
}

template <typename T>
void launch_forward(Tensor& loss, const Tensor& logits,
                    const bpe::TokenId* targets, const std::size_t rows,
                    const std::size_t vocabulary_size, cudaStream_t stream) {
    cross_entropy_forward_kernel<T><<<static_cast<unsigned>(rows), kThreads, 0, stream>>>(
        static_cast<float*>(loss.raw_data()), static_cast<const T*>(logits.raw_data()),
        targets, rows, vocabulary_size, 1.0f / static_cast<float>(rows));
}

} // namespace

/**
 * @brief Computes mean softmax cross-entropy and its logits gradient.
 *
 * Supported logits and gradient dtypes are FP32, FP16, and BF16. The loss is
 * always accumulated in a one-element CUDA FP32 tensor. Target IDs must point
 * to device memory and the target count must equal the number of logits rows.
 *
 * @param[out] loss One-element CUDA FP32 tensor receiving the mean loss.
 * @param[out] gradient Gradient with respect to @p logits.
 * @param[in] logits Input logits with shape [rows, vocabulary_size].
 * @param[in] device_targets Device pointer to target IDs.
 * @param target_count Number of target IDs.
 * @param stream CUDA stream used for the asynchronous operation.
 * @throws std::invalid_argument If inputs, shapes, dtypes, or targets are invalid.
 * @throws cudaError_t If a CUDA memory operation or kernel launch fails.
 */
void cross_entropy_forward_backward(Tensor& loss, Tensor& gradient, const Tensor& logits,
                                    const bpe::TokenId* device_targets,
                                    const std::size_t target_count, cudaStream_t stream) {
    validate(loss, gradient, logits, device_targets, target_count);

    CUDA_CHECK(cudaMemsetAsync(loss.raw_data(), 0, sizeof(float), stream));

    const auto rows = logits.size(0);
    const auto vocabulary_size = logits.size(1);

    switch (logits.dtype()) {
        case Dtype::F32:
            launch<float>(loss, gradient, logits, device_targets, rows, vocabulary_size, stream);
            break;
        case Dtype::F16:
            launch<__half>(loss, gradient, logits, device_targets, rows, vocabulary_size, stream);
            break;
        case Dtype::BF16:
            launch<__nv_bfloat16>(loss, gradient, logits, device_targets, rows, vocabulary_size, stream);
            break;
        default:
            throw std::invalid_argument("cross_entropy: unsupported logits dtype");
    }

    CUDA_CHECK(cudaGetLastError());
}

void cross_entropy_forward(Tensor& loss, const Tensor& logits,
                           const bpe::TokenId* device_targets,
                           const std::size_t target_count, cudaStream_t stream) {
    if (device_targets == nullptr || logits.device_type() != DeviceType::CUDA ||
        logits.dim() != 2 || !is_floating_point(logits.dtype()) || logits.size(0) == 0 ||
        logits.size(1) == 0 || target_count != logits.size(0))
        throw std::invalid_argument("cross_entropy: invalid logits or target count");
    if (loss.device_type() != DeviceType::CUDA || loss.dtype() != Dtype::F32 ||
        loss.shape() != std::vector<std::size_t>{1})
        throw std::invalid_argument("cross_entropy: loss must be a CUDA F32 tensor with shape [1]");

    CUDA_CHECK(cudaMemsetAsync(loss.raw_data(), 0, sizeof(float), stream));
    const auto rows = logits.size(0);
    const auto vocabulary_size = logits.size(1);
    switch (logits.dtype()) {
        case Dtype::F32:
            launch_forward<float>(loss, logits, device_targets, rows, vocabulary_size, stream);
            break;
        case Dtype::F16:
            launch_forward<__half>(loss, logits, device_targets, rows, vocabulary_size, stream);
            break;
        case Dtype::BF16:
            launch_forward<__nv_bfloat16>(loss, logits, device_targets, rows, vocabulary_size, stream);
            break;
        default:
            throw std::invalid_argument("cross_entropy: unsupported logits dtype");
    }
    CUDA_CHECK(cudaGetLastError());
}
