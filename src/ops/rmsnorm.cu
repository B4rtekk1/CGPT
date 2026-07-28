/**
* @file rmsnorm.cu
 * @brief CUDA implementation of Root Mean Square Layer Normalization.
 *
 * The implementation contains cached and streaming kernels, a vectorized FP32
 * path, vectorized FP16 and scalar BF16 fallbacks, warp-level reductions and host-side launch
 * selection based on tensor shape and data type.
 */

#include "ops/rmsnorm.h"
#include "core/cuda_check.h"
#include "device_guard.h"
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <limits>
#include <stdexcept>

namespace {
    /**
     * @brief Reduces a value across one NVIDIA warp using shuffle instructions.
     * @param value Per-lane partial sum.
     * @return Sum broadcast to lane zero; other lanes contain intermediate values.
     */
    __inline__ __device__ float warp_reduce_sum(float value) {
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            constexpr unsigned mask = 0xffffffffu;
            value += __shfl_down_sync(mask, value, offset);
        }
        return value;
    }

    /**
     * @brief Reduces a value across the entire CUDA thread block.
     * @param value Per-thread partial sum.
     * @return Block sum in thread zero.
     */
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

    /**
     * @brief Vectorized FP32 RMSNorm kernel that caches one row in registers.
     * @tparam ItemsPerThread Number of packed `float4` values cached per thread.
     * @param output Destination tensor.
     * @param input Source tensor.
     * @param weight Per-feature scale tensor.
     * @param vector_count Number of `float4` vectors in one row.
     * @param inverse_hidden Reciprocal of the scalar hidden dimension.
     * @param epsilon Numerical stability term.
     */
    template<int ItemsPerThread>
    __launch_bounds__(256)
    __global__ void rmsnorm_vectorized_cached_kernel(
        float * __restrict__ output,
        const float * __restrict__ input,
        const float * __restrict__ weight,
        int vector_count,
        float inverse_hidden,
        float epsilon
    ) {
        const int tid = static_cast<int>(threadIdx.x);
        const int block_size = static_cast<int>(blockDim.x);
        const std::size_t row_offset =
                static_cast<std::size_t>(blockIdx.x) * vector_count;
        const auto *row_input =
                reinterpret_cast<const float4 *>(input) + row_offset;
        auto *row_output = reinterpret_cast<float4 *>(output) + row_offset;

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
        const auto *weight_vectors = reinterpret_cast<const float4 *>(weight);

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

    /**
     * @brief Scalar RMSNorm kernel that caches one row in registers.
     * @tparam T Tensor element type.
     * @tparam ItemsPerThread Number of scalar values cached per thread.
     * @param output Destination tensor.
     * @param input Source tensor.
     * @param weight Per-feature scale tensor.
     * @param hidden Number of scalar elements in one row.
     * @param inverse_hidden Reciprocal of @p hidden.
     * @param epsilon Numerical stability term.
     */
    template<typename T, int ItemsPerThread>
    __launch_bounds__(256)
    __global__ void rmsnorm_scalar_cached_kernel(
        T * __restrict__ output,
        const T * __restrict__ input,
        const T * __restrict__ weight,
        int hidden,
        float inverse_hidden,
        float epsilon
    ) {
        const int tid = static_cast<int>(threadIdx.x);
        const int block_size = static_cast<int>(blockDim.x);
        const std::size_t row_offset =
                static_cast<std::size_t>(blockIdx.x) * hidden;
        const T *row_input = input + row_offset;
        T *row_output = output + row_offset;

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

    /**
     * @brief FP32 streaming RMSNorm kernel for rows too large to cache.
     * @tparam Vectorized Uses aligned `float4` memory operations when true.
     * @param output Destination tensor.
     * @param input Source tensor.
     * @param weight Per-feature scale tensor.
     * @param hidden Number of scalar elements in one row.
     * @param inverse_hidden Reciprocal of @p hidden.
     * @param epsilon Numerical stability term.
     */
    template<bool Vectorized>
    __launch_bounds__(256)
    __global__ void rmsnorm_streaming_kernel(
        float * __restrict__ output,
        const float * __restrict__ input,
        const float * __restrict__ weight,
        int hidden,
        float inverse_hidden,
        float epsilon
    ) {
        const int tid = static_cast<int>(threadIdx.x);
        const int block_size = static_cast<int>(blockDim.x);
        const std::size_t row_offset =
                static_cast<std::size_t>(blockIdx.x) * hidden;
        const float *row_input = input + row_offset;
        float *row_output = output + row_offset;

        float sum_squares = 0.0f;
        if constexpr (Vectorized) {
            const auto *input_vectors =
                    reinterpret_cast<const float4 *>(row_input);
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
            const auto *input_vectors =
                    reinterpret_cast<const float4 *>(row_input);
            auto *output_vectors = reinterpret_cast<float4 *>(row_output);
            const auto *weight_vectors =
                    reinterpret_cast<const float4 *>(weight);
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

    /**
     * @brief Scalar streaming RMSNorm kernel for FP16 and BF16 tensors.
     * @tparam T Tensor element type.
     * @param output Destination tensor.
     * @param input Source tensor.
     * @param weight Per-feature scale tensor.
     * @param hidden Number of scalar elements in one row.
     * @param inverse_hidden Reciprocal of @p hidden.
     * @param epsilon Numerical stability term.
     */
    template<typename T>
    __launch_bounds__(256)
    __global__ void rmsnorm_scalar_streaming_kernel(
        T * __restrict__ output,
        const T * __restrict__ input,
        const T * __restrict__ weight,
        int hidden,
        float inverse_hidden,
        float epsilon
    ) {
        const int tid = static_cast<int>(threadIdx.x);
        const int block_size = static_cast<int>(blockDim.x);
        const std::size_t row_offset =
                static_cast<std::size_t>(blockIdx.x) * hidden;
        const T *row_input = input + row_offset;
        T *row_output = output + row_offset;

        float sum_squares = 0.0f;
        for (int index = tid; index < hidden; index += block_size) {
            const auto value = static_cast<float>(row_input[index]);
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

    /**
     * @brief Validates tensor shape, dtype, device and epsilon constraints.
     * @param output Destination tensor.
     * @param input Rank-two source tensor with shape `[rows, hidden]`.
     * @param weight Rank-one scale tensor with shape `[hidden]`.
     * @param epsilon Positive finite numerical stability term.
     * @throws std::invalid_argument When the RMSNorm contract is violated.
     */
    void validate_rmsnorm(
        const Tensor &output,
        const Tensor &input,
        const Tensor &weight,
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

    /**
     * @brief Selects and launches a scalar cached or streaming RMSNorm kernel.
     * @tparam T Tensor element type.
     * @param output Destination tensor.
     * @param input Source tensor.
     * @param weight Per-feature scale tensor.
     * @param rows Number of input rows, and therefore CUDA blocks.
     * @param hidden Number of scalar elements in one row.
     * @param inverse_hidden Reciprocal of @p hidden.
     * @param epsilon Numerical stability term.
     * @param stream CUDA stream that receives the kernel launch.
     */
    template<typename T>
    void launch_scalar_rmsnorm(
        Tensor &output,
        const Tensor &input,
        const Tensor &weight,
        int rows,
        int hidden,
        float inverse_hidden,
        float epsilon,
        cudaStream_t stream
    ) {
        const int threads = hidden <= 32 ? 32 : hidden <= 256 ? 128 : 256;
        const int items_per_thread = (hidden + threads - 1) / threads;
        auto *output_data = static_cast<T *>(output.raw_data());
        const auto *input_data = static_cast<const T *>(input.raw_data());
        const auto *weight_data = static_cast<const T *>(weight.raw_data());

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


    /**
     * @brief Converts one packed 32-bit word to two FP16 values.
     */
    __forceinline__ __device__ __half2 packed_u32_to_half2(unsigned int value) {
        union PackedHalf2 {
            unsigned int bits;
            __half2 halves;
        } packed{};
        packed.bits = value;
        return packed.halves;
    }

    /**
     * @brief Converts two FP16 values to one packed 32-bit word.
     */
    __forceinline__ __device__ unsigned int half2_to_packed_u32(__half2 value) {
        union PackedHalf2 {
            unsigned int bits;
            __half2 halves;
        } packed{};
        packed.halves = value;
        return packed.bits;
    }

    /**
     * @brief FP16 RMSNorm specialized for hidden size 4096.
     *
     * One 256-thread block processes one row. Each thread loads two 128-bit
     * vectors, corresponding to sixteen FP16 elements. Input values remain in
     * registers between reduction and normalization, so the input row is read
     * exactly once from global memory.
     */
    __launch_bounds__(256, 2)
    __global__ void rmsnorm_fp16_4096_kernel(
        __half * __restrict__ output,
        const __half * __restrict__ input,
        const __half * __restrict__ weight,
        float epsilon
    ) {
        constexpr int kHidden = 4096;
        constexpr int kThreads = 256;
        constexpr int kVectorsPerRow = kHidden * static_cast<int>(sizeof(__half)) /
                                       static_cast<int>(sizeof(uint4));
        static_assert(kVectorsPerRow == 512);

        const int tid = static_cast<int>(threadIdx.x);
        const std::size_t row_vector_offset =
            static_cast<std::size_t>(blockIdx.x) * kVectorsPerRow;

        const auto *input128 = reinterpret_cast<const uint4 *>(input) + row_vector_offset;
        auto *output128 = reinterpret_cast<uint4 *>(output) + row_vector_offset;
        const auto *weight128 = reinterpret_cast<const uint4 *>(weight);

        const uint4 packed0 = input128[tid];
        const uint4 packed1 = input128[tid + kThreads];

        const float2 v0 = __half22float2(packed_u32_to_half2(packed0.x));
        const float2 v1 = __half22float2(packed_u32_to_half2(packed0.y));
        const float2 v2 = __half22float2(packed_u32_to_half2(packed0.z));
        const float2 v3 = __half22float2(packed_u32_to_half2(packed0.w));
        const float2 v4 = __half22float2(packed_u32_to_half2(packed1.x));
        const float2 v5 = __half22float2(packed_u32_to_half2(packed1.y));
        const float2 v6 = __half22float2(packed_u32_to_half2(packed1.z));
        const float2 v7 = __half22float2(packed_u32_to_half2(packed1.w));

        float sum_squares = 0.0f;
#define RMSNORM_ACCUMULATE_PAIR(value) \
        sum_squares = fmaf((value).x, (value).x, sum_squares); \
        sum_squares = fmaf((value).y, (value).y, sum_squares)
        RMSNORM_ACCUMULATE_PAIR(v0);
        RMSNORM_ACCUMULATE_PAIR(v1);
        RMSNORM_ACCUMULATE_PAIR(v2);
        RMSNORM_ACCUMULATE_PAIR(v3);
        RMSNORM_ACCUMULATE_PAIR(v4);
        RMSNORM_ACCUMULATE_PAIR(v5);
        RMSNORM_ACCUMULATE_PAIR(v6);
        RMSNORM_ACCUMULATE_PAIR(v7);
#undef RMSNORM_ACCUMULATE_PAIR

        constexpr float kInverseHidden = 1.0f / 4096.0f;
        const float inv_rms =
            rsqrtf(block_reduce_sum(sum_squares) * kInverseHidden + epsilon);

        const uint4 scale0 = weight128[tid];
        const uint4 scale1 = weight128[tid + kThreads];

#define RMSNORM_NORMALIZE_PAIR(value, packed_scale) \
        __floats2half2_rn( \
            (value).x * inv_rms * __low2float(packed_u32_to_half2(packed_scale)), \
            (value).y * inv_rms * __high2float(packed_u32_to_half2(packed_scale)))

        uint4 result0;
        result0.x = half2_to_packed_u32(RMSNORM_NORMALIZE_PAIR(v0, scale0.x));
        result0.y = half2_to_packed_u32(RMSNORM_NORMALIZE_PAIR(v1, scale0.y));
        result0.z = half2_to_packed_u32(RMSNORM_NORMALIZE_PAIR(v2, scale0.z));
        result0.w = half2_to_packed_u32(RMSNORM_NORMALIZE_PAIR(v3, scale0.w));

        uint4 result1;
        result1.x = half2_to_packed_u32(RMSNORM_NORMALIZE_PAIR(v4, scale1.x));
        result1.y = half2_to_packed_u32(RMSNORM_NORMALIZE_PAIR(v5, scale1.y));
        result1.z = half2_to_packed_u32(RMSNORM_NORMALIZE_PAIR(v6, scale1.z));
        result1.w = half2_to_packed_u32(RMSNORM_NORMALIZE_PAIR(v7, scale1.w));
#undef RMSNORM_NORMALIZE_PAIR

        output128[tid] = result0;
        output128[tid + kThreads] = result1;
    }

    /**
     * @brief Vectorized FP16 RMSNorm kernel caching packed values in registers.
     * @tparam ItemsPerThread Number of `half2` values cached per thread.
     */
    template<int ItemsPerThread>
    __launch_bounds__(256)
    __global__ void rmsnorm_half2_cached_kernel(
        __half * __restrict__ output,
        const __half * __restrict__ input,
        const __half * __restrict__ weight,
        int pair_count,
        float inverse_hidden,
        float epsilon
    ) {
        const int tid = static_cast<int>(threadIdx.x);
        const int block_size = static_cast<int>(blockDim.x);
        const std::size_t row_offset =
                static_cast<std::size_t>(blockIdx.x) * pair_count;
        const auto *row_input = reinterpret_cast<const __half2 *>(input) + row_offset;
        auto *row_output = reinterpret_cast<__half2 *>(output) + row_offset;
        const auto *weight_pairs = reinterpret_cast<const __half2 *>(weight);

        float2 values[ItemsPerThread];
        float sum_squares = 0.0f;
#pragma unroll
        for (int item = 0; item < ItemsPerThread; ++item) {
            const int index = tid + item * block_size;
            float2 value = make_float2(0.0f, 0.0f);
            if (index < pair_count) {
                value = __half22float2(row_input[index]);
                sum_squares = fmaf(value.x, value.x, sum_squares);
                sum_squares = fmaf(value.y, value.y, sum_squares);
            }
            values[item] = value;
        }

        const float inv_rms =
                rsqrtf(block_reduce_sum(sum_squares) * inverse_hidden + epsilon);

#pragma unroll
        for (int item = 0; item < ItemsPerThread; ++item) {
            const int index = tid + item * block_size;
            if (index < pair_count) {
                const float2 scale = __half22float2(weight_pairs[index]);
                const float2 value = values[item];
                row_output[index] = __floats2half2_rn(
                    value.x * inv_rms * scale.x,
                    value.y * inv_rms * scale.y);
            }
        }
    }

    /**
     * @brief Vectorized FP16 streaming fallback for very wide rows.
     */
    __launch_bounds__(256)
    __global__ void rmsnorm_half2_streaming_kernel(
        __half * __restrict__ output,
        const __half * __restrict__ input,
        const __half * __restrict__ weight,
        int pair_count,
        float inverse_hidden,
        float epsilon
    ) {
        const int tid = static_cast<int>(threadIdx.x);
        const int block_size = static_cast<int>(blockDim.x);
        const std::size_t row_offset =
                static_cast<std::size_t>(blockIdx.x) * pair_count;
        const auto *row_input = reinterpret_cast<const __half2 *>(input) + row_offset;
        auto *row_output = reinterpret_cast<__half2 *>(output) + row_offset;
        const auto *weight_pairs = reinterpret_cast<const __half2 *>(weight);

        float sum_squares = 0.0f;
        for (int index = tid; index < pair_count; index += block_size) {
            const float2 value = __half22float2(row_input[index]);
            sum_squares = fmaf(value.x, value.x, sum_squares);
            sum_squares = fmaf(value.y, value.y, sum_squares);
        }

        const float inv_rms =
                rsqrtf(block_reduce_sum(sum_squares) * inverse_hidden + epsilon);

        for (int index = tid; index < pair_count; index += block_size) {
            const float2 value = __half22float2(row_input[index]);
            const float2 scale = __half22float2(weight_pairs[index]);
            row_output[index] = __floats2half2_rn(
                value.x * inv_rms * scale.x,
                value.y * inv_rms * scale.y);
        }
    }

    void launch_half_rmsnorm(
        Tensor &output,
        const Tensor &input,
        const Tensor &weight,
        int rows,
        int hidden,
        float inverse_hidden,
        float epsilon,
        cudaStream_t stream
    ) {
        auto *output_data = static_cast<__half *>(output.raw_data());
        const auto *input_data = static_cast<const __half *>(input.raw_data());
        const auto *weight_data = static_cast<const __half *>(weight.raw_data());

        const bool packed = (hidden & 1) == 0 &&
            (reinterpret_cast<std::uintptr_t>(input_data) & 0x3u) == 0 &&
            (reinterpret_cast<std::uintptr_t>(output_data) & 0x3u) == 0 &&
            (reinterpret_cast<std::uintptr_t>(weight_data) & 0x3u) == 0;

        if (!packed) {
            launch_scalar_rmsnorm<__half>(
                output, input, weight, rows, hidden,
                inverse_hidden, epsilon, stream);
            return;
        }

        constexpr int threads = 256;

        const bool aligned_128 =
            (reinterpret_cast<std::uintptr_t>(input_data) & 0x0fu) == 0 &&
            (reinterpret_cast<std::uintptr_t>(output_data) & 0x0fu) == 0 &&
            (reinterpret_cast<std::uintptr_t>(weight_data) & 0x0fu) == 0;

        if (hidden == 4096 && aligned_128) {
            rmsnorm_fp16_4096_kernel<<<rows, threads, 0, stream>>>(
                output_data, input_data, weight_data, epsilon);
            return;
        }

        const int pair_count = hidden / 2;
        const int items_per_thread = (pair_count + threads - 1) / threads;

        if (items_per_thread == 1) {
            rmsnorm_half2_cached_kernel<1><<<rows, threads, 0, stream>>>(
                output_data, input_data, weight_data, pair_count,
                inverse_hidden, epsilon);
        } else if (items_per_thread <= 2) {
            rmsnorm_half2_cached_kernel<2><<<rows, threads, 0, stream>>>(
                output_data, input_data, weight_data, pair_count,
                inverse_hidden, epsilon);
        } else if (items_per_thread <= 4) {
            rmsnorm_half2_cached_kernel<4><<<rows, threads, 0, stream>>>(
                output_data, input_data, weight_data, pair_count,
                inverse_hidden, epsilon);
        } else if (items_per_thread <= 8) {
            rmsnorm_half2_cached_kernel<8><<<rows, threads, 0, stream>>>(
                output_data, input_data, weight_data, pair_count,
                inverse_hidden, epsilon);
        } else {
            rmsnorm_half2_streaming_kernel<<<rows, threads, 0, stream>>>(
                output_data, input_data, weight_data, pair_count,
                inverse_hidden, epsilon);
        }
    }
} // namespace

/**
 * @brief Applies RMS normalization over the last tensor dimension.
 *
 * For each row, computes `x * rsqrt(mean(x^2) + epsilon) * weight`.
 *
 * @param output Destination tensor with the same shape as `input`.
 * @param input Source tensor.
 * @param weight One-dimensional scale tensor matching the last input dimension.
 * @param epsilon Numerical stability term.
 * @param stream CUDA stream used for asynchronous execution.
 *
 * @throws std::invalid_argument If tensors are not CUDA-resident floating-point
 *         tensors of matching dtype, or their shapes do not satisfy the API
 *         contract.
 *
 * @note The function validates and enqueues work only; it does not synchronize
 *       @p stream. The FP32 `float4` path is selected only when all buffers are
 *       suitably aligned and the hidden dimension is divisible by four.
 */
void rmsnorm_forward(
    Tensor &output,
    const Tensor &input,
    const Tensor &weight,
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
        const int threads = vector_count <= 32 ? 32 : vector_count <= 256 ? 128 : 256;
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
        launch_half_rmsnorm(
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

/**
 * @brief Allocates an output tensor and applies RMS normalization.
 *
 * This convenience overload creates a CUDA output tensor whose shape, dtype
 * and device match @p input, then delegates to the preallocated-output
 * overload.
 *
 * @param input Source tensor with shape `[rows, hidden]`.
 * @param weight One-dimensional scale tensor with shape `[hidden]`.
 * @param epsilon Positive finite numerical stability term.
 * @param stream CUDA stream used for asynchronous execution.
 * @return Newly allocated normalized tensor.
 * @throws std::invalid_argument Under the same conditions as the
 *         preallocated-output overload.
 */
Tensor rmsnorm_forward(
    const Tensor &input,
    const Tensor &weight,
    float epsilon,
    cudaStream_t stream
) {
    Tensor output(input.shape(), input.device_type(), input.dtype());
    rmsnorm_forward(output, input, weight, epsilon, stream);
    return output;
}
