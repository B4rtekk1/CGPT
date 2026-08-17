/**
 * @file rmsnorm_backward(1).cu
 * @brief CUDA backward pass for RMSNorm.
 */

#include "ops/backward/rmsnorm_backward.h"
#include "core/cuda_check.h"
#include "core/device_guard.h"

#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include <limits>
#include <stdexcept>
#include <cstdint>
#include <type_traits>
#include <unordered_map>

namespace {
    /** @brief Performs a warp-wide FP32 sum reduction. */
    __forceinline__ __device__ float warp_reduce_sum(float value) {
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            value += __shfl_down_sync(0xffffffff, value, offset);
        }
        return value;
    }

    /** @brief Performs a block-wide FP32 sum reduction. */
    __forceinline__ __device__ float block_reduce_sum(float value) {
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

    /**
     * @brief Computes the inverse RMS for one row after block reduction.
     * @param value Per-thread sum of squared input values.
     * @param inverse_hidden Reciprocal hidden dimension.
     * @param epsilon Numerical stability constant.
     * @return `1 / sqrt(mean(x^2) + epsilon)`.
     */
    __forceinline__ __device__ float block_reduce_rms(
        float value,
        const float inverse_hidden,
        const float epsilon
    ) {
        value = warp_reduce_sum(value);
        __shared__ float warp_values[8];
        const int lane = static_cast<int>(threadIdx.x) & 31;
        const int warp = static_cast<int>(threadIdx.x) >> 5;
        if (lane == 0) {
            warp_values[warp] = value;
        }
        __syncthreads();

        if (warp == 0) {
            const int warp_count = (static_cast<int>(blockDim.x) + 31) >> 5;
            value = lane < warp_count ? warp_values[lane] : 0.0f;
            value = warp_reduce_sum(value);
            if (lane == 0) {
                warp_values[0] = rsqrtf(value * inverse_hidden + epsilon);
            }
        }
        __syncthreads();
        return warp_values[0];
    }

    /** @brief Unpacks two FP16 or BF16 values stored in a 32-bit word. */
    template<typename T>
    __forceinline__ __device__ float2 packed_u32_to_float2(
        const unsigned int value
    ) {
        union Packed16 {
            unsigned int bits;
            __half2 halves;
            __nv_bfloat162 bfloat;
        } packed{};
        packed.bits = value;
        if constexpr (std::is_same_v<T, __half>) {
            return __half22float2(packed.halves);
        } else {
            return __bfloat1622float2(packed.bfloat);
        }
    }

    /** @brief Packs two FP32 values into FP16 or BF16 storage. */
    template<typename T>
    __forceinline__ __device__ unsigned int floats_to_packed_u32(
        const float first,
        const float second
    ) {
        union Packed16 {
            unsigned int bits;
            __half2 halves;
            __nv_bfloat162 bfloat;
        } packed{};
        if constexpr (std::is_same_v<T, __half>) {
            packed.halves = __floats2half2_rn(first, second);
        } else {
            packed.bfloat = __floats2bfloat162_rn(first, second);
        }
        return packed.bits;
    }

    /**
     * @brief Register-cached 16-bit backward kernel for 4096-wide model rows.
     *
     * Each thread owns sixteen elements and reads x, dy, and weight once. The
     * packed values remain live across the RMS and weighted-dot reductions.
     */
    /**
     * @brief Optimized 16-bit input-gradient kernel for hidden size 4096.
     * @tparam T Either __half or __nv_bfloat16.
     */
    template<typename T>
    __launch_bounds__(256, 2)
    __global__ void rmsnorm_grad_input_16bit_4096_kernel(
        T * __restrict__ grad_input,
        const T * __restrict__ grad_output,
        const T * __restrict__ input,
        const T * __restrict__ weight,
        float * __restrict__ inv_rms,
        const float epsilon
    ) {
        constexpr int kThreads = 256;
        constexpr int kVectorsPerRow = 512;
        constexpr float kInverseHidden = 1.0f / 4096.0f;
        const int tid = static_cast<int>(threadIdx.x);
        const std::size_t row_vector_offset =
            static_cast<std::size_t>(blockIdx.x) * kVectorsPerRow;
        const auto *input128 = reinterpret_cast<const uint4 *>(input) + row_vector_offset;
        const auto *grad_output128 =
            reinterpret_cast<const uint4 *>(grad_output) + row_vector_offset;
        auto *grad_input128 = reinterpret_cast<uint4 *>(grad_input) + row_vector_offset;
        const auto *weight128 = reinterpret_cast<const uint4 *>(weight);

        const uint4 x0 = input128[tid];
        const uint4 x1 = input128[tid + kThreads];
        const uint4 dy0 = grad_output128[tid];
        const uint4 dy1 = grad_output128[tid + kThreads];

        float sum_squares = 0.0f;
#define RMSNORM_ACCUMULATE_SQUARES(packed_value) \
        { \
            const float2 value = packed_u32_to_float2<T>(packed_value); \
            sum_squares = fmaf(value.x, value.x, sum_squares); \
            sum_squares = fmaf(value.y, value.y, sum_squares); \
        }
        RMSNORM_ACCUMULATE_SQUARES(x0.x);
        RMSNORM_ACCUMULATE_SQUARES(x0.y);
        RMSNORM_ACCUMULATE_SQUARES(x0.z);
        RMSNORM_ACCUMULATE_SQUARES(x0.w);
        RMSNORM_ACCUMULATE_SQUARES(x1.x);
        RMSNORM_ACCUMULATE_SQUARES(x1.y);
        RMSNORM_ACCUMULATE_SQUARES(x1.z);
        RMSNORM_ACCUMULATE_SQUARES(x1.w);
#undef RMSNORM_ACCUMULATE_SQUARES

        const float row_inv_rms =
            block_reduce_rms(sum_squares, kInverseHidden, epsilon);
        if (tid == 0) {
            inv_rms[blockIdx.x] = row_inv_rms;
        }

        const uint4 weight0 = weight128[tid];
        const uint4 weight1 = weight128[tid + kThreads];
        float weighted_dot = 0.0f;
#define RMSNORM_ACCUMULATE_DOT(packed_x, packed_dy, packed_weight) \
        { \
            const float2 x = packed_u32_to_float2<T>(packed_x); \
            const float2 dy = packed_u32_to_float2<T>(packed_dy); \
            const float2 w = packed_u32_to_float2<T>(packed_weight); \
            weighted_dot = fmaf(dy.x * w.x, x.x, weighted_dot); \
            weighted_dot = fmaf(dy.y * w.y, x.y, weighted_dot); \
        }
        RMSNORM_ACCUMULATE_DOT(x0.x, dy0.x, weight0.x);
        RMSNORM_ACCUMULATE_DOT(x0.y, dy0.y, weight0.y);
        RMSNORM_ACCUMULATE_DOT(x0.z, dy0.z, weight0.z);
        RMSNORM_ACCUMULATE_DOT(x0.w, dy0.w, weight0.w);
        RMSNORM_ACCUMULATE_DOT(x1.x, dy1.x, weight1.x);
        RMSNORM_ACCUMULATE_DOT(x1.y, dy1.y, weight1.y);
        RMSNORM_ACCUMULATE_DOT(x1.z, dy1.z, weight1.z);
        RMSNORM_ACCUMULATE_DOT(x1.w, dy1.w, weight1.w);
#undef RMSNORM_ACCUMULATE_DOT

        const float correction = block_reduce_sum(weighted_dot) * kInverseHidden *
            row_inv_rms * row_inv_rms * row_inv_rms;

#define RMSNORM_GRAD_PAIR(packed_x, packed_dy, packed_weight) \
        [&] { \
            const float2 x = packed_u32_to_float2<T>(packed_x); \
            const float2 dy = packed_u32_to_float2<T>(packed_dy); \
            const float2 w = packed_u32_to_float2<T>(packed_weight); \
            return floats_to_packed_u32<T>( \
                fmaf(-x.x, correction, dy.x * w.x * row_inv_rms), \
                fmaf(-x.y, correction, dy.y * w.y * row_inv_rms)); \
        }()
        uint4 dx0;
        dx0.x = RMSNORM_GRAD_PAIR(x0.x, dy0.x, weight0.x);
        dx0.y = RMSNORM_GRAD_PAIR(x0.y, dy0.y, weight0.y);
        dx0.z = RMSNORM_GRAD_PAIR(x0.z, dy0.z, weight0.z);
        dx0.w = RMSNORM_GRAD_PAIR(x0.w, dy0.w, weight0.w);
        grad_input128[tid] = dx0;

        uint4 dx1;
        dx1.x = RMSNORM_GRAD_PAIR(x1.x, dy1.x, weight1.x);
        dx1.y = RMSNORM_GRAD_PAIR(x1.y, dy1.y, weight1.y);
        dx1.z = RMSNORM_GRAD_PAIR(x1.z, dy1.z, weight1.z);
        dx1.w = RMSNORM_GRAD_PAIR(x1.w, dy1.w, weight1.w);
#undef RMSNORM_GRAD_PAIR
        grad_input128[tid + kThreads] = dx1;
    }

    /**
     * @brief Cached-register RMSNorm input-gradient kernel for small rows.
     * @tparam T Input and gradient storage type.
     */
    template<typename T>
    __launch_bounds__(256)
    __global__ void rmsnorm_grad_input_cached_kernel(
        T * __restrict__ grad_input,
        const T * __restrict__ grad_output,
        const T * __restrict__ input,
        const T * __restrict__ weight,
        float * __restrict__ inv_rms,
        const int hidden,
        const float inverse_hidden,
        const float epsilon
    ) {
        constexpr int kThreads = 256;
        const int tid = static_cast<int>(threadIdx.x);
        const std::size_t row_offset = static_cast<std::size_t>(blockIdx.x) * hidden;
        const T *row_input = input + row_offset;
        const T *row_grad_output = grad_output + row_offset;
        T *row_grad_input = grad_input + row_offset;

        float x_values[4];
        float dy_values[4];
        float weight_values[4];
        float sum_squares = 0.0f;
        float weighted_dot = 0.0f;
#pragma unroll
        for (int item = 0; item < 4; ++item) {
            const int index = tid + item * kThreads;
            float x = 0.0f;
            float dy = 0.0f;
            float scale = 0.0f;
            if (index < hidden) {
                x = static_cast<float>(row_input[index]);
                dy = static_cast<float>(row_grad_output[index]);
                scale = static_cast<float>(weight[index]);
                sum_squares = fmaf(x, x, sum_squares);
                weighted_dot = fmaf(dy * scale, x, weighted_dot);
            }
            x_values[item] = x;
            dy_values[item] = dy;
            weight_values[item] = scale;
        }

        const float row_inv_rms = block_reduce_rms(sum_squares, inverse_hidden, epsilon);
        if (tid == 0) {
            inv_rms[blockIdx.x] = row_inv_rms;
        }
        const float correction = block_reduce_sum(weighted_dot) * inverse_hidden *
                                 row_inv_rms * row_inv_rms * row_inv_rms;

#pragma unroll
        for (int item = 0; item < 4; ++item) {
            const int index = tid + item * kThreads;
            if (index < hidden) {
                row_grad_input[index] = static_cast<T>(
                    fmaf(-x_values[item], correction,
                         dy_values[item] * weight_values[item] * row_inv_rms));
            }
        }
    }

    /**
     * @brief General RMSNorm input-gradient kernel.
     * @tparam T Input and gradient storage type.
     */
    template<typename T>
    __launch_bounds__(256)
    __global__ void rmsnorm_grad_input_kernel(
        T * __restrict__ grad_input,
        const T * __restrict__ grad_output,
        const T * __restrict__ input,
        const T * __restrict__ weight,
        float * __restrict__ inv_rms,
        const int hidden,
        const float inverse_hidden,
        const float epsilon
    ) {
        const int column = static_cast<int>(threadIdx.x);
        const std::size_t row_offset = static_cast<std::size_t>(blockIdx.x) * hidden;
        const T *row_input = input + row_offset;
        const T *row_grad_output = grad_output + row_offset;
        T *row_grad_input = grad_input + row_offset;

        float sum_squares = 0.0f;
        const int stride = static_cast<int>(blockDim.x);
        for (int index = column; index < hidden; index += stride * 4) {
#pragma unroll
            for (int unroll = 0; unroll < 4; ++unroll) {
                const int current = index + unroll * stride;
                if (current < hidden) {
                    const float x = static_cast<float>(row_input[current]);
                    sum_squares = fmaf(x, x, sum_squares);
                }
            }
        }
        __shared__ float shared_inv_rms;
        const float square_sum = block_reduce_sum(sum_squares);
        if (threadIdx.x == 0) {
            shared_inv_rms = rsqrtf(square_sum * inverse_hidden + epsilon);
            inv_rms[blockIdx.x] = shared_inv_rms;
        }
        __syncthreads();

        float weighted_dot = 0.0f;
        for (int index = column; index < hidden; index += stride * 4) {
#pragma unroll
            for (int unroll = 0; unroll < 4; ++unroll) {
                const int current = index + unroll * stride;
                if (current < hidden) {
                    const float x = static_cast<float>(row_input[current]);
                    const float dy_weight = static_cast<float>(row_grad_output[current]) *
                                            static_cast<float>(weight[current]);
                    weighted_dot = fmaf(dy_weight, x, weighted_dot);
                }
            }
        }

        const float row_inv_rms = shared_inv_rms;
        const float correction = block_reduce_sum(weighted_dot) * inverse_hidden *
                                 row_inv_rms * row_inv_rms * row_inv_rms;

        for (int index = column; index < hidden; index += stride * 4) {
#pragma unroll
            for (int unroll = 0; unroll < 4; ++unroll) {
                const int current = index + unroll * stride;
                if (current < hidden) {
                    const float x = static_cast<float>(row_input[current]);
                    const float dy_weight = static_cast<float>(row_grad_output[current]) *
                                            static_cast<float>(weight[current]);
                    row_grad_input[current] = static_cast<T>(
                        fmaf(-x, correction, dy_weight * row_inv_rms));
                }
            }
        }
    }

    /**
     * @brief Computes the gradient of the RMSNorm scale vector.
     * @tparam T Input and gradient storage type.
     */
    template<typename T>
    __launch_bounds__(256)
    __global__ void rmsnorm_grad_weight_kernel(
        T * __restrict__ grad_weight,
        const T * __restrict__ grad_output,
        const T * __restrict__ input,
        const float * __restrict__ inv_rms,
        const int rows,
        const int hidden
    ) {
        const int column = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x)
                         + static_cast<int>(threadIdx.x);
        if (column >= hidden) return;

        // Adjacent lanes own adjacent columns, so every iteration issues
        // coalesced row reads.  The former one-CTA-per-column mapping made
        // adjacent lanes read rows separated by `hidden` elements.
        float grad = 0.0f;
        for (int row = 0; row < rows; ++row) {
            const std::size_t row_offset = static_cast<std::size_t>(row) * hidden;
            grad = fmaf(static_cast<float>(grad_output[row_offset + column]),
                        static_cast<float>(input[row_offset + column]) * inv_rms[row], grad);
        }
        grad_weight[column] = static_cast<T>(grad);
    }

    /**
     * @brief Validates RMSNorm backward tensors and epsilon.
     * @throws std::invalid_argument If shapes, devices, dtypes, dimensions, or
     *         epsilon are invalid.
     */
    void validate_rmsnorm_backward(
        const Tensor &grad_input,
        const Tensor &grad_weight,
        const Tensor &grad_output,
        const Tensor &input,
        const Tensor &weight,
        const float epsilon
    ) {
        if (!std::isfinite(epsilon) || epsilon <= 0.0f) {
            throw std::invalid_argument("rmsnorm_backward: epsilon must be finite and positive");
        }
        const Tensor *tensors[] = {&grad_input, &grad_weight, &grad_output, &input, &weight};
        for (const Tensor *tensor: tensors) {
            if (tensor->device_type() != DeviceType::CUDA) {
                throw std::invalid_argument("rmsnorm_backward: all tensors must be CUDA tensors");
            }
            if (!is_floating_point(tensor->dtype()) || tensor->dtype() != input.dtype()) {
                throw std::invalid_argument(
                    "rmsnorm_backward: all tensors must have the same floating dtype");
            }
        }
        if (input.dim() != 2 || grad_output.shape() != input.shape() ||
            grad_input.shape() != input.shape()) {
            throw std::invalid_argument(
                "rmsnorm_backward: input, grad_output and grad_input must have shape [rows, hidden]");
        }
        if (weight.dim() != 1 || grad_weight.dim() != 1 ||
            weight.size(0) != input.size(1) || grad_weight.size(0) != input.size(1)) {
            throw std::invalid_argument(
                "rmsnorm_backward: weight and grad_weight must have shape [hidden]");
        }
        if (input.size(0) > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
            input.size(1) > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
            throw std::invalid_argument("rmsnorm_backward: dimensions exceed supported range");
        }
    }

    /**
     * @brief Selects and launches the optimized RMSNorm backward kernels.
     * @tparam T Input and gradient storage type.
     */
    template<typename T>
    void launch_rmsnorm_backward(
        Tensor &grad_input, Tensor &grad_weight, const Tensor &grad_output,
        const Tensor &input, const Tensor &weight, const int rows, const int hidden,
        const float epsilon, cudaStream_t stream
    ) {
        constexpr int threads = 256;
        const float inverse_hidden = 1.0f / static_cast<float>(hidden);
        // Workspace is cached per host thread and CUDA stream.  This keeps the
        // operation asynchronous after its first use instead of paying for a
        // synchronizing cudaMalloc/cudaFree pair on every backward call.
        thread_local std::unordered_map<std::uintptr_t, DeviceBuffer> inv_rms_buffers;
        DeviceBuffer& inv_rms_buffer =
            inv_rms_buffers[reinterpret_cast<std::uintptr_t>(stream)];
        const std::size_t required_bytes = static_cast<std::size_t>(rows) * sizeof(float);
        if (inv_rms_buffer.bytes() < required_bytes) {
            inv_rms_buffer.allocate(required_bytes);
        }
        auto *inv_rms = static_cast<float *>(inv_rms_buffer.data());
        const bool aligned_128 =
            (reinterpret_cast<std::uintptr_t>(grad_input.raw_data()) & 0x0fu) == 0 &&
            (reinterpret_cast<std::uintptr_t>(grad_output.raw_data()) & 0x0fu) == 0 &&
            (reinterpret_cast<std::uintptr_t>(input.raw_data()) & 0x0fu) == 0 &&
            (reinterpret_cast<std::uintptr_t>(weight.raw_data()) & 0x0fu) == 0;
        if constexpr (std::is_same_v<T, __half> || std::is_same_v<T, __nv_bfloat16>) {
            if (hidden == 4096 && aligned_128) {
                rmsnorm_grad_input_16bit_4096_kernel<T><<<rows, threads, 0, stream>>>(
                    static_cast<T *>(grad_input.raw_data()),
                    static_cast<const T *>(grad_output.raw_data()),
                    static_cast<const T *>(input.raw_data()),
                    static_cast<const T *>(weight.raw_data()),
                    inv_rms, epsilon);
            } else if (hidden <= 1024) {
                rmsnorm_grad_input_cached_kernel<T><<<rows, threads, 0, stream>>>(
                    static_cast<T *>(grad_input.raw_data()), static_cast<const T *>(grad_output.raw_data()),
                    static_cast<const T *>(input.raw_data()), static_cast<const T *>(weight.raw_data()),
                    inv_rms, hidden, inverse_hidden, epsilon);
            } else {
                rmsnorm_grad_input_kernel<T><<<rows, threads, 0, stream>>>(
                    static_cast<T *>(grad_input.raw_data()),
                    static_cast<const T *>(grad_output.raw_data()),
                    static_cast<const T *>(input.raw_data()),
                    static_cast<const T *>(weight.raw_data()),
                    inv_rms, hidden, inverse_hidden, epsilon);
            }
        } else if (hidden <= 1024) {
            rmsnorm_grad_input_cached_kernel<T><<<rows, threads, 0, stream>>>(
                static_cast<T *>(grad_input.raw_data()), static_cast<const T *>(grad_output.raw_data()),
                static_cast<const T *>(input.raw_data()), static_cast<const T *>(weight.raw_data()),
                inv_rms, hidden, inverse_hidden, epsilon);
        } else {
            rmsnorm_grad_input_kernel<T><<<rows, threads, 0, stream>>>(
                static_cast<T *>(grad_input.raw_data()),
                static_cast<const T *>(grad_output.raw_data()),
                static_cast<const T *>(input.raw_data()),
                static_cast<const T *>(weight.raw_data()),
                inv_rms, hidden, inverse_hidden, epsilon);
        }
        const int weight_blocks = (hidden + threads - 1) / threads;
        rmsnorm_grad_weight_kernel<T><<<weight_blocks, threads, 0, stream>>>(
            static_cast<T *>(grad_weight.raw_data()),
            static_cast<const T *>(grad_output.raw_data()),
            static_cast<const T *>(input.raw_data()), inv_rms, rows, hidden);
    }
}

/**
 * @brief Computes gradients through RMSNorm.
 *
 * For `y = weight * input / sqrt(mean(input^2) + epsilon)`, this function
 * computes gradients for both the input and the scale vector. It supports
 * FP32, FP16, and BF16 CUDA tensors and selects specialized kernels based on
 * hidden size, alignment, and data type.
 *
 * @param[out] grad_input Gradient with respect to the input matrix.
 * @param[out] grad_weight Gradient with respect to the scale vector.
 * @param[in] grad_output Gradient from the following operation.
 * @param[in] input RMSNorm input with shape [rows, hidden].
 * @param[in] weight Scale vector with shape [hidden].
 * @param epsilon Positive numerical stability constant.
 * @param stream CUDA stream used for asynchronous launches.
 * @throws std::invalid_argument If tensors or epsilon are invalid.
 */
void rmsnorm_backward(
    Tensor& grad_input, Tensor& grad_weight, const Tensor& grad_output,
    const Tensor& input, const Tensor& weight, const float epsilon, cudaStream_t stream) {
    validate_rmsnorm_backward(grad_input, grad_weight, grad_output, input, weight, epsilon);
    DeviceGuard device_guard(0);
    const int rows = static_cast<int>(input.size(0));
    const int hidden = static_cast<int>(input.size(1));
    if (input.dtype() == Dtype::F16) {
        launch_rmsnorm_backward<__half>(grad_input, grad_weight, grad_output, input, weight,
                                        rows, hidden, epsilon, stream);
    } else if (input.dtype() == Dtype::BF16) {
        launch_rmsnorm_backward<__nv_bfloat16>(grad_input, grad_weight, grad_output, input,
                                               weight, rows, hidden, epsilon, stream);
    } else {
        launch_rmsnorm_backward<float>(grad_input, grad_weight, grad_output, input, weight,
                                       rows, hidden, epsilon, stream);
    }
    CUDA_CHECK(cudaGetLastError());
}
