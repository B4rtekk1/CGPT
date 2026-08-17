/**
 * @file rope_backward.cu
 * @brief CUDA backward pass for Rotary Positional Embedding (RoPE).
 *
 * The implementation provides a generic path for arbitrary supported layouts
 * and a specialized FP16 path for the common 32-query-head, 8-key-head,
 * 128-dimensional configuration.
 */

#include "ops/backward/rope_backward.h"
#include "core/cuda_check.h"
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include <algorithm>
#include <limits>
#include <stdexcept>

namespace {
    constexpr std::uint32_t kThreadsPerBlock = 256U;
    constexpr std::uint32_t kMaxGridX = 65535U;

    /** @brief Converts a supported RoPE cache or gradient value to FP32. */
    template<typename T>
    __device__ __forceinline__ float to_float(const T value) {
        return static_cast<float>(value);
    }

    template<>
    __device__ __forceinline__ float to_float<half>(const half value) {
        return __half2float(value);
    }

    template<>
    __device__ __forceinline__ float to_float<__nv_bfloat16>(const __nv_bfloat16 value) {
        return __bfloat162float(value);
    }

    /**
     * @brief Applies the transpose of the forward rotation to Q and K gradients.
     *
     * Each pair `(x0, x1)` is transformed with the inverse rotation:
     * `dx0 = dy0*cos + dy1*sin` and `dx1 = -dy0*sin + dy1*cos`.
     *
     * @tparam T Element type of gradients and trigonometric caches.
     * @param grad_query Output query gradient.
     * @param grad_key Output key gradient.
     * @param grad_rotated_query Gradient after the forward RoPE operation.
     * @param grad_rotated_key Gradient after the forward RoPE operation.
     * @param cos_cache Cosine cache indexed by position and rotary pair.
     * @param sin_cache Sine cache indexed by position and rotary pair.
     * @param token_count Number of batch/sequence tokens.
     * @param sequence_length Sequence length used to derive token positions.
     * @param query_head_count Number of query heads.
     * @param key_head_count Number of key heads.
     * @param head_dim Head dimension.
     * @param rotary_pair_count Number of rotated coordinate pairs.
     * @param position_offset Offset into the trigonometric caches.
     */
    template<typename T>
    __global__ void rope_backward_kernel(
        T * __restrict__ grad_query,
        T * __restrict__ grad_key,
        const T * __restrict__ grad_rotated_query,
        const T * __restrict__ grad_rotated_key,
        const T * __restrict__ cos_cache,
        const T * __restrict__ sin_cache,
        const std::size_t token_count,
        const std::uint32_t sequence_length,
        const std::uint32_t query_head_count,
        const std::uint32_t key_head_count,
        const std::uint32_t head_dim,
        const std::uint32_t rotary_pair_count,
        const std::uint32_t position_offset
    ) {
        for (std::size_t token = blockIdx.x; token < token_count; token += gridDim.x) {
            const auto position = static_cast<std::uint32_t>(token % sequence_length);
            const std::size_t cache_offset = static_cast<std::size_t>(position + position_offset) * rotary_pair_count;
            const std::size_t query_token_offset = token * static_cast<std::size_t>(query_head_count) * head_dim;
            const std::size_t key_token_offset =
                token * static_cast<std::size_t>(key_head_count) * head_dim;

            for (std::uint32_t pair = threadIdx.x; pair < rotary_pair_count; pair += blockDim.x) {
                const float cosine = to_float(cos_cache[cache_offset + pair]);
                const float sine = to_float(sin_cache[cache_offset + pair]);
                const std::size_t pair_offset = static_cast<std::size_t>(pair) * 2U;

                for (std::uint32_t head = 0; head < query_head_count; ++head) {
                    const std::size_t offset = query_token_offset +
                                               static_cast<std::size_t>(head) * head_dim + pair_offset;
                    const float dy0 = to_float(grad_rotated_query[offset]);
                    const float dy1 = to_float(grad_rotated_query[offset + 1U]);
                    grad_query[offset] = static_cast<T>(fmaf(dy1, sine, dy0 * cosine));
                    grad_query[offset + 1U] =
                            static_cast<T>(fmaf(-dy0, sine, dy1 * cosine));
                }

                for (std::uint32_t head = 0; head < key_head_count; ++head) {
                    const std::size_t offset = key_token_offset +
                                               static_cast<std::size_t>(head) * head_dim + pair_offset;
                    const float dy0 = to_float(grad_rotated_key[offset]);
                    const float dy1 = to_float(grad_rotated_key[offset + 1U]);
                    grad_key[offset] = static_cast<T>(fmaf(dy1, sine, dy0 * cosine));
                    grad_key[offset + 1U] =
                            static_cast<T>(fmaf(-dy0, sine, dy1 * cosine));
                }
            }
        }
    }

    /**
     * @brief Register/shared-memory optimized FP16 backward path for 32/8 heads.
     *
     * This specialization assumes 32 query heads, 8 key heads, head dimension
     * 128, and a full 128-dimensional rotation.
     */
    __global__ __launch_bounds__(256) void rope_backward_f16_32_8_128_kernel(
        half * __restrict__ grad_query,
        half * __restrict__ grad_key,
        const half * __restrict__ grad_rotated_query,
        const half * __restrict__ grad_rotated_key,
        const half * __restrict__ cos_cache,
        const half * __restrict__ sin_cache,
        const std::uint32_t sequence_length,
        const std::uint32_t position_offset
    ) {
        constexpr std::uint32_t query_heads = 32U;
        constexpr std::uint32_t key_heads = 8U;
        constexpr std::uint32_t head_dim = 128U;
        constexpr std::uint32_t pairs_per_head = 64U;
        constexpr std::uint32_t total_heads = query_heads + key_heads;
        constexpr std::uint32_t work_per_token = total_heads * pairs_per_head;

        const std::uint32_t token = blockIdx.x;
        const std::uint32_t position = token % sequence_length;
        const std::size_t cache_base =
            static_cast<std::size_t>(position + position_offset) * pairs_per_head;

        __shared__ float shared_cos[pairs_per_head];
        __shared__ float shared_sin[pairs_per_head];
        if (threadIdx.x < pairs_per_head) {
            shared_cos[threadIdx.x] = __half2float(cos_cache[cache_base + threadIdx.x]);
            shared_sin[threadIdx.x] = __half2float(sin_cache[cache_base + threadIdx.x]);
        }
        __syncthreads();

        const std::size_t query_token_base =
            static_cast<std::size_t>(token) * query_heads * head_dim;
        const std::size_t key_token_base =
            static_cast<std::size_t>(token) * key_heads * head_dim;

#pragma unroll
        for (std::uint32_t work = threadIdx.x;
             work < work_per_token;
             work += blockDim.x) {
            const std::uint32_t pair = work & (pairs_per_head - 1U);
            const std::uint32_t combined_head = work >> 6U;
            const std::size_t pair_offset = static_cast<std::size_t>(pair) * 2U;

            const half *input_pair;
            half *output_pair;
            if (combined_head < query_heads) {
                const std::size_t offset = query_token_base +
                    static_cast<std::size_t>(combined_head) * head_dim + pair_offset;
                input_pair = grad_rotated_query + offset;
                output_pair = grad_query + offset;
            } else {
                const std::size_t offset = key_token_base +
                    static_cast<std::size_t>(combined_head - query_heads) * head_dim + pair_offset;
                input_pair = grad_rotated_key + offset;
                output_pair = grad_key + offset;
            }

            const float2 gradient =
                __half22float2(*reinterpret_cast<const half2 *>(input_pair));
            const float cosine = shared_cos[pair];
            const float sine = shared_sin[pair];
            *reinterpret_cast<half2 *>(output_pair) = __floats2half2_rn(
                fmaf(gradient.y, sine, gradient.x * cosine),
                fmaf(-gradient.x, sine, gradient.y * cosine));
        }
    }

    /** @brief Converts a size value to uint32_t or throws on overflow. */
    [[nodiscard]] std::uint32_t checked_u32(const std::size_t value, const char *name) {
        if (value > std::numeric_limits<std::uint32_t>::max()) {
            throw std::overflow_error(std::string("RoPE backward: ") + name + " exceeds uint32_t range");
        }
        return static_cast<std::uint32_t>(value);
    }

    /** @brief Validates that a tensor is a CUDA rank-four tensor. */
    void validate_tensor(const Tensor &tensor, const char *name) {
        if (tensor.device_type() != DeviceType::CUDA) {
            throw std::invalid_argument(std::string("RoPE backward: ") + name +
                                        " must be a CUDA tensor");
        }
        if (tensor.shape().size() != 4) {
            throw std::invalid_argument(std::string("RoPE backward: ") + name +
                                        " must have shape [batch, sequence, heads, head_dim]");
        }
    }

    /**
     * @brief Validates RoPE gradient tensors, caches, options, and dimensions.
     * @throws std::invalid_argument For incompatible devices, shapes, dtypes,
     *         rotary dimensions, or cache coverage.
     * @throws std::overflow_error If a launch parameter exceeds uint32_t range.
     */
    void validate_inputs(
        const Tensor &grad_query, const Tensor &grad_key,
        const Tensor &grad_rotated_query, const Tensor &grad_rotated_key,
        const Tensor &cos_cache, const Tensor &sin_cache, const RopeOptions &options
    ) {
        validate_tensor(grad_query, "grad_query");
        validate_tensor(grad_key, "grad_key");
        validate_tensor(grad_rotated_query, "grad_rotated_query");
        validate_tensor(grad_rotated_key, "grad_rotated_key");

        if (grad_query.shape() != grad_rotated_query.shape() ||
            grad_key.shape() != grad_rotated_key.shape() ||
            grad_query.shape()[0] != grad_key.shape()[0] ||
            grad_query.shape()[1] != grad_key.shape()[1] ||
            grad_query.shape()[3] != grad_key.shape()[3]) {
            throw std::invalid_argument("RoPE backward: incompatible Q/K gradient shapes");
        }
        if (cos_cache.device_type() != DeviceType::CUDA ||
            sin_cache.device_type() != DeviceType::CUDA ||
            cos_cache.shape().size() != 2 || sin_cache.shape() != cos_cache.shape()) {
            throw std::invalid_argument(
                "RoPE backward: caches must be CUDA tensors with identical [max_sequence, rotary_dim / 2] shapes");
        }
        if (grad_query.dtype() != grad_key.dtype() ||
            grad_query.dtype() != grad_rotated_query.dtype() ||
            grad_query.dtype() != grad_rotated_key.dtype() ||
            grad_query.dtype() != cos_cache.dtype() || grad_query.dtype() != sin_cache.dtype()) {
            throw std::invalid_argument("RoPE backward: all tensors must have the same dtype");
        }

        const std::size_t head_dim = grad_query.shape()[3];
        const std::size_t sequence_length = grad_query.shape()[1];
        const std::size_t rotary_dim = options.rotary_dim == 0 ? head_dim : options.rotary_dim;
        if (head_dim == 0 || sequence_length == 0) {
            return;
        }
        if ((head_dim & 1U) != 0U || rotary_dim == 0 || rotary_dim > head_dim ||
            (rotary_dim & 1U) != 0U) {
            throw std::invalid_argument(
                "RoPE backward: head_dim and rotary_dim must be even, with rotary_dim in [2, head_dim]");
        }
        if (cos_cache.shape()[1] != rotary_dim / 2U ||
            options.position_offset > std::numeric_limits<std::size_t>::max() - sequence_length ||
            cos_cache.shape()[0] < options.position_offset + sequence_length) {
            throw std::invalid_argument("RoPE backward: cache does not cover the requested rotary positions");
        }
        static_cast<void>(checked_u32(sequence_length, "sequence length"));
        static_cast<void>(checked_u32(grad_query.shape()[2], "query head count"));
        static_cast<void>(checked_u32(grad_key.shape()[2], "key head count"));
        static_cast<void>(checked_u32(head_dim, "head dimension"));
        static_cast<void>(checked_u32(rotary_dim / 2U, "rotary pair count"));
        static_cast<void>(checked_u32(options.position_offset, "position offset"));
    }

    /**
     * @brief Launches the generic RoPE backward kernel for one element type.
     * @tparam T Element type used by gradients and caches.
     */
    template<typename T>
    void launch(
        Tensor &grad_query, Tensor &grad_key,
        const Tensor &grad_rotated_query, const Tensor &grad_rotated_key,
        const Tensor &cos_cache, const Tensor &sin_cache,
        const std::size_t token_count, const RopeOptions &options, cudaStream_t stream
    ) {
        const auto grid_size = static_cast<std::uint32_t>(
            std::min(token_count, static_cast<std::size_t>(kMaxGridX)));
        const auto rotary_pairs = checked_u32(
            (options.rotary_dim == 0 ? grad_query.shape()[3] : options.rotary_dim) / 2U,
            "rotary pair count");
        const std::uint32_t block_size = rotary_pairs <= 32U ? 32U
            : rotary_pairs <= 64U ? 64U
            : rotary_pairs <= 128U ? 128U
            : kThreadsPerBlock;
        rope_backward_kernel<T><<<grid_size, block_size, 0, stream>>>(
            static_cast<T *>(grad_query.raw_data()), static_cast<T *>(grad_key.raw_data()),
            static_cast<const T *>(grad_rotated_query.raw_data()),
            static_cast<const T *>(grad_rotated_key.raw_data()),
            static_cast<const T *>(cos_cache.raw_data()), static_cast<const T *>(sin_cache.raw_data()),
            token_count, checked_u32(grad_query.shape()[1], "sequence length"),
            checked_u32(grad_query.shape()[2], "query head count"),
            checked_u32(grad_key.shape()[2], "key head count"),
            checked_u32(grad_query.shape()[3], "head dimension"),
            rotary_pairs,
            checked_u32(options.position_offset, "position offset"));
    }
}

/**
 * @brief Computes gradients through Rotary Positional Embedding.
 *
 * The function supports FP32, FP16, and BF16 tensors with query/key layouts
 * `[batch, sequence, heads, head_dim]`. When `rotary_dim` is smaller than the
 * head dimension, the unrotated suffix is copied before the rotated prefix is
 * processed. The operation is asynchronous with respect to the supplied CUDA
 * stream.
 *
 * @param[out] grad_query Gradient with respect to the original query tensor.
 * @param[out] grad_key Gradient with respect to the original key tensor.
 * @param[in] grad_rotated_query Gradient after RoPE for queries.
 * @param[in] grad_rotated_key Gradient after RoPE for keys.
 * @param[in] cos_cache Cosine cache with shape [max_sequence, rotary_dim / 2].
 * @param[in] sin_cache Sine cache with shape [max_sequence, rotary_dim / 2].
 * @param stream CUDA stream used for copies and kernel launches.
 * @param rope_options RoPE dimension and position-offset configuration.
 * @throws std::invalid_argument If tensors or RoPE options are incompatible.
 * @throws std::overflow_error If dimensions cannot be represented by launch types.
 */
void rope_backward(
    Tensor &grad_query, Tensor &grad_key,
    const Tensor &grad_rotated_query, const Tensor &grad_rotated_key,
    const Tensor &cos_cache, const Tensor &sin_cache,
    cudaStream_t stream, const RopeOptions &rope_options
) {
    validate_inputs(grad_query, grad_key, grad_rotated_query, grad_rotated_key,
                    cos_cache, sin_cache, rope_options);
    if (grad_query.numel() == 0 || grad_key.numel() == 0 || grad_query.shape()[1] == 0) {
        return;
    }
    // A full-width rotation overwrites every output element, so copying the
    // complete Q/K tensors first only doubles global-memory traffic.
    const std::size_t rotary_dim = rope_options.rotary_dim == 0
        ? grad_query.shape()[3]
        : rope_options.rotary_dim;
    if (rotary_dim < grad_query.shape()[3]) {
        if (grad_query.raw_data() != grad_rotated_query.raw_data()) {
            CUDA_CHECK(cudaMemcpyAsync(grad_query.raw_data(), grad_rotated_query.raw_data(),
                grad_query.nbytes(), cudaMemcpyDeviceToDevice, stream));
        }
        if (grad_key.raw_data() != grad_rotated_key.raw_data()) {
            CUDA_CHECK(cudaMemcpyAsync(grad_key.raw_data(), grad_rotated_key.raw_data(),
                grad_key.nbytes(), cudaMemcpyDeviceToDevice, stream));
        }
    }
    if (grad_query.shape()[0] != 0 &&
        grad_query.shape()[1] > std::numeric_limits<std::size_t>::max() /
        grad_query.shape()[0]) {
        throw std::overflow_error("RoPE backward: batch times sequence overflow");
    }
    const std::size_t token_count = grad_query.shape()[0] * grad_query.shape()[1];
    if (token_count == 0) {
        return;
    }

    switch (grad_query.dtype()) {
        case Dtype::F32:
            launch<float>(grad_query, grad_key, grad_rotated_query, grad_rotated_key,
                          cos_cache, sin_cache, token_count, rope_options, stream);
            break;
        case Dtype::F16:
            if (grad_query.shape()[2] == 32U &&
                grad_key.shape()[2] == 8U &&
                grad_query.shape()[3] == 128U &&
                rotary_dim == 128U &&
                token_count <= kMaxGridX) {
                rope_backward_f16_32_8_128_kernel<<<
                    static_cast<std::uint32_t>(token_count), kThreadsPerBlock, 0, stream
                >>>(
                    static_cast<half *>(grad_query.raw_data()),
                    static_cast<half *>(grad_key.raw_data()),
                    static_cast<const half *>(grad_rotated_query.raw_data()),
                    static_cast<const half *>(grad_rotated_key.raw_data()),
                    static_cast<const half *>(cos_cache.raw_data()),
                    static_cast<const half *>(sin_cache.raw_data()),
                    checked_u32(grad_query.shape()[1], "sequence length"),
                    checked_u32(rope_options.position_offset, "position offset"));
            } else {
                launch<half>(grad_query, grad_key, grad_rotated_query, grad_rotated_key,
                             cos_cache, sin_cache, token_count, rope_options, stream);
            }
            break;
        case Dtype::BF16:
            launch<__nv_bfloat16>(grad_query, grad_key, grad_rotated_query, grad_rotated_key,
                                  cos_cache, sin_cache, token_count, rope_options, stream);
            break;
        default:
            throw std::invalid_argument("RoPE backward: unsupported dtype");
    }
    CUDA_CHECK(cudaGetLastError());
}
