#include "ops/rope.h"
#include "core/cuda_check.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <string>

namespace {
    /** Number of threads executed together by NVIDIA hardware as one warp. */
constexpr std::uint32_t WARP_SIZE = 32U;
    /** Maximum block size used by the RoPE launcher. */
constexpr std::uint32_t MAX_THREADS_PER_BLOCK = 256U;
    /** Conservative one-dimensional grid limit supported by all target devices. */
constexpr std::uint32_t MAX_PORTABLE_GRID_X = 65535U;

/**
 * @brief Type-specific vectorized memory access used by RoPE kernel.
 *
 * Each specialization load and stores two adjacent tensor elements as one
 * logical rotary pair and converts arithmetic operands to FP32.
 * This keeps the rotation math uniform for FP32, Fp16 and BF16 tensors.
 *
 * @tparam T CUDA storage type of the tensor.
 */
template<typename T>
struct RopeTypeTraits;

/** @brief FP32 implementation of vectorized RoPE memory operations. */
template<>
struct RopeTypeTraits<float> {
    __device__ __forceinline__ static float2 load_pair(
        const float* const pointer
    ) {
        return *reinterpret_cast<const float2*>(pointer);
    }

    __device__ __forceinline__ static void store_pair(
        float* const pointer,
        const float2 value
    ) {
        *reinterpret_cast<float2*>(pointer) = value;
    }

    __device__ __forceinline__ static float load_scalar(
        const float* const pointer
    ) {
        return *pointer;
    }
};

/** @brief FP16 implementation using packed @c half2 accesses. */
template<>
struct RopeTypeTraits<half> {
    __device__ __forceinline__ static float2 load_pair(
        const half* const pointer
    ) {
        return __half22float2(*reinterpret_cast<const half2*>(pointer));
    }

    __device__ __forceinline__ static void store_pair(
        half* const pointer,
        const float2 value
    ) {
        *reinterpret_cast<half2*>(pointer) =
            __floats2half2_rn(value.x, value.y);
    }

    __device__ __forceinline__ static float load_scalar(
        const half* const pointer
    ) {
        return __half2float(*pointer);
    }
};

/** @brief BF16 implementation using packed CUDA @c __nv_bfloat162 accesses. */
template<>
struct RopeTypeTraits<__nv_bfloat16> {
    __device__ __forceinline__ static float2 load_pair(
        const __nv_bfloat16* const pointer
    ) {
        return __bfloat1622float2(
            *reinterpret_cast<const __nv_bfloat162*>(pointer)
        );
    }

    __device__ __forceinline__ static void store_pair(
        __nv_bfloat16* const pointer,
        const float2 value
    ) {
        *reinterpret_cast<__nv_bfloat162*>(pointer) =
            __floats2bfloat162_rn(value.x, value.y);
    }

    __device__ __forceinline__ static float load_scalar(
        const __nv_bfloat16* const pointer
    ) {
        return __bfloat162float(*pointer);
    }
};

/** @brief Applies a two-dimensional rotary transform in place.
 *
 * Given an adjacent pair `(x0, x1)`, computes:
 * @code
 * y0 = x0 * cos(theta) - x1 * sin(theta)
 * y1 = x0 * sin(theta) + x1 * cos(theta)
 * @endcode
 *
 * Input values are converted to FP32 for arithemtic and converted back to the
 * tensor storage type on write. Fused multiply-add instructions are used where avaible
 *
 * @tparam T Tensor storage type: @c float, @c half or @c __nv_bfloat16
 * @param pointer Address of two adcjacent tensor elements. The adress must be
 * aligned for the corresponding vector type.
 * @param cosine Cosine coefficient for the current rotary pair.
 * @param sine Sine coefficient for the current rotary pair.
 */
template<typename T>
__device__ __forceinline__ void rotate_pair(
    T* const pointer,
    const float cosine,
    const float sine
) {
    using Traits = RopeTypeTraits<T>;

    const float2 value = Traits::load_pair(pointer);
    const float2 rotated = make_float2(
        fmaf(-value.y, sine, value.x * cosine),
        fmaf(value.x, sine, value.y * cosine)
    );

    Traits::store_pair(pointer, rotated);
}

/**
 * @brief Applies RoPE to query and key tensors in one fused CUDA kernel.
 *
 * A block processes one or more flattened tokens using a grid-stride loop.
 * Threads are distributed across rotary pairs. Each thread loads one cosine
 * and sine pair and reuses it for every query and key head belonging to the
 * token. Fusing Q and K avoids a second kernel launch and repeated cache loads.
 *
 * Tensor memory layout:
 * @code
 * query      [batch, sequence, query_heads, head_dim]
 * key        [batch, sequence, key_heads,   head_dim]
 * cos_cache  [max_sequence, rotary_dim / 2]
 * sin_cache  [max_sequence, rotary_dim / 2]
 * @endcode
 *
 * Only the first @p rotary_pair_count pairs of each head are rotated. Elements
 * beyond `2 * rotary_pair_count` remain unchanged.
 *
 * @tparam T Tensor storage type.
 * @param query Mutable query tensor data.
 * @param key Mutable key tensor data.
 * @param cos_cache Precomputed cosine coefficients.
 * @param sin_cache Precomputed sine coefficients.
 * @param token_count Number of flattened `[batch, sequence]` tokens.
 * @param sequence_length Sequence dimension used to recover token position.
 * @param query_head_count Number of query heads.
 * @param key_head_count Number of key/value heads.
 * @param head_dim Number of elements in one attention head.
 * @param rotary_pair_count Number of rotated element pairs per head.
 * @param position_offset Position added to every sequence index when reading
 *        the caches, typically used during autoregressive decoding.
 */
template<typename T>
__global__ void rope_qk_kernel(
    T* __restrict__ query,
    T* __restrict__ key,
    const T* __restrict__ cos_cache,
    const T* __restrict__ sin_cache,
    const std::size_t token_count,
    const std::uint32_t sequence_length,
    const std::uint32_t query_head_count,
    const std::uint32_t key_head_count,
    const std::uint32_t head_dim,
    const std::uint32_t rotary_pair_count,
    const std::uint32_t position_offset
) {
    for (std::size_t token_index = blockIdx.x;
         token_index < token_count;
         token_index += gridDim.x) {
        const auto sequence_index = static_cast<std::uint32_t>(
            token_index % sequence_length
        );
        const std::size_t cache_base =
            static_cast<std::size_t>(position_offset + sequence_index) *
            rotary_pair_count;

        const std::size_t query_token_base =
            token_index * query_head_count * head_dim;
        const std::size_t key_token_base =
            token_index * key_head_count * head_dim;

        for (std::uint32_t pair_index = threadIdx.x;
             pair_index < rotary_pair_count;
             pair_index += blockDim.x) {
            const float cosine = RopeTypeTraits<T>::load_scalar(
                cos_cache + cache_base + pair_index
            );
            const float sine = RopeTypeTraits<T>::load_scalar(
                sin_cache + cache_base + pair_index
            );
            const std::size_t pair_offset =
                static_cast<std::size_t>(pair_index) * 2U;

#pragma unroll 1
            for (std::uint32_t head_index = 0;
                 head_index < query_head_count;
                 ++head_index) {
                T* const pair_pointer =
                    query + query_token_base +
                    static_cast<std::size_t>(head_index) * head_dim +
                    pair_offset;
                rotate_pair(pair_pointer, cosine, sine);
            }

#pragma unroll 1
            for (std::uint32_t head_index = 0;
                 head_index < key_head_count;
                 ++head_index) {
                T* const pair_pointer =
                    key + key_token_base +
                    static_cast<std::size_t>(head_index) * head_dim +
                    pair_offset;
                rotate_pair(pair_pointer, cosine, sine);
            }
        }
    }
}

[[nodiscard]] std::uint32_t checked_u32(
    const std::size_t value,
    const char* const description
) {
    if (value > std::numeric_limits<std::uint32_t>::max()) {
        throw std::overflow_error(
            std::string("RoPE: ") + description + " exceeds uint32 range"
        );
    }

    return static_cast<std::uint32_t>(value);
}

[[nodiscard]] std::size_t checked_product(
    const std::size_t left,
    const std::size_t right,
    const char* const description
) {
    if (left != 0 && right > std::numeric_limits<std::size_t>::max() / left) {
        throw std::overflow_error(
            std::string("RoPE: ") + description + " overflow"
        );
    }

    return left * right;
}

void validate_tensor_layout(const Tensor& tensor, const char* const name) {
    if (tensor.device_type() != DeviceType::CUDA) {
        throw std::invalid_argument(
            std::string("RoPE: ") + name + " must be a CUDA tensor"
        );
    }

    if (tensor.shape().size() != 4) {
        throw std::invalid_argument(
            std::string("RoPE: ") + name +
            " must have shape [batch, sequence, heads, head_dim]"
        );
    }
}

void validate_inputs(
    const Tensor& query,
    const Tensor& key,
    const Tensor& cos_cache,
    const Tensor& sin_cache,
    const RopeOptions& options
) {
    validate_tensor_layout(query, "query");
    validate_tensor_layout(key, "key");

    if (cos_cache.device_type() != DeviceType::CUDA ||
        sin_cache.device_type() != DeviceType::CUDA) {
        throw std::invalid_argument("RoPE: caches must be CUDA tensors");
    }

    if (cos_cache.shape().size() != 2 || sin_cache.shape().size() != 2) {
        throw std::invalid_argument(
            "RoPE: caches must have shape [max_sequence, rotary_dim / 2]"
        );
    }

    if (query.dtype() != key.dtype() ||
        query.dtype() != cos_cache.dtype() ||
        query.dtype() != sin_cache.dtype()) {
        throw std::invalid_argument(
            "RoPE: query, key and caches must have the same dtype"
        );
    }

    if (query.shape()[0] != key.shape()[0] ||
        query.shape()[1] != key.shape()[1] ||
        query.shape()[3] != key.shape()[3]) {
        throw std::invalid_argument(
            "RoPE: query and key must have equal batch, sequence and head_dim"
        );
    }

    if (cos_cache.shape() != sin_cache.shape()) {
        throw std::invalid_argument(
            "RoPE: cos_cache and sin_cache must have identical shapes"
        );
    }

    const std::size_t head_dim = query.shape()[3];
    const std::size_t rotary_dim =
        options.rotary_dim == 0 ? head_dim : options.rotary_dim;

    if (head_dim == 0 || query.shape()[1] == 0) {
        return;
    }

    if ((head_dim & 1U) != 0U) {
        throw std::invalid_argument(
            "RoPE: head_dim must be even for vectorized pair access"
        );
    }

    if (rotary_dim == 0 || rotary_dim > head_dim || (rotary_dim & 1U) != 0U) {
        throw std::invalid_argument(
            "RoPE: rotary_dim must be even, non-zero and <= head_dim"
        );
    }

    const std::size_t rotary_pair_count = rotary_dim / 2U;
    if (cos_cache.shape()[1] != rotary_pair_count) {
        throw std::invalid_argument(
            "RoPE: cache width must equal rotary_dim / 2"
        );
    }

    if (options.position_offset >
        std::numeric_limits<std::size_t>::max() - query.shape()[1]) {
        throw std::overflow_error("RoPE: position range overflow");
    }

    const std::size_t required_positions =
        options.position_offset + query.shape()[1];
    if (cos_cache.shape()[0] < required_positions) {
        throw std::invalid_argument(
            "RoPE: cache does not contain all requested positions"
        );
    }

    static_cast<void>(checked_u32(query.shape()[1], "sequence length"));
    static_cast<void>(checked_u32(query.shape()[2], "query head count"));
    static_cast<void>(checked_u32(key.shape()[2], "key head count"));
    static_cast<void>(checked_u32(head_dim, "head dimension"));
    static_cast<void>(checked_u32(rotary_pair_count, "rotary pair count"));
    static_cast<void>(checked_u32(options.position_offset, "position offset"));
    static_cast<void>(checked_product(
        query.shape()[0], query.shape()[1], "batch times sequence"
    ));
}

[[nodiscard]] std::uint32_t select_block_size(
    const std::uint32_t rotary_pair_count
) {
    if (rotary_pair_count <= WARP_SIZE) {
        return WARP_SIZE;
    }
    if (rotary_pair_count <= 64U) {
        return 64U;
    }
    if (rotary_pair_count <= 128U) {
        return 128U;
    }
    return MAX_THREADS_PER_BLOCK;
}

template<typename T>
void launch_rope(
    Tensor& query,
    Tensor& key,
    const Tensor& cos_cache,
    const Tensor& sin_cache,
    const std::size_t token_count,
    const std::uint32_t sequence_length,
    const std::uint32_t head_dim,
    const std::uint32_t rotary_pair_count,
    const std::uint32_t position_offset,
    cudaStream_t stream
) {
    const std::uint32_t query_head_count =
        checked_u32(query.shape()[2], "query head count");
    const std::uint32_t key_head_count =
        checked_u32(key.shape()[2], "key head count");

    const auto grid_size = static_cast<std::uint32_t>(
        std::min<std::size_t>(token_count, MAX_PORTABLE_GRID_X)
    );
    const std::uint32_t block_size = select_block_size(rotary_pair_count);

    rope_qk_kernel<T><<<grid_size, block_size, 0, stream>>>(
        static_cast<T*>(query.raw_data()),
        static_cast<T*>(key.raw_data()),
        static_cast<const T*>(cos_cache.raw_data()),
        static_cast<const T*>(sin_cache.raw_data()),
        token_count,
        sequence_length,
        query_head_count,
        key_head_count,
        head_dim,
        rotary_pair_count,
        position_offset
    );
}
} // namespace

void rope_forward(
    Tensor& query,
    Tensor& key,
    const Tensor& cos_cache,
    const Tensor& sin_cache,
    cudaStream_t stream,
    const RopeOptions& options
) {
    validate_inputs(query, key, cos_cache, sin_cache, options);

    if (query.numel() == 0 || key.numel() == 0 || query.shape()[1] == 0) {
        return;
    }

    const std::size_t token_count = checked_product(
        query.shape()[0], query.shape()[1], "batch times sequence"
    );
    if (token_count == 0) {
        return;
    }

    const std::size_t head_dim_value = query.shape()[3];
    const std::size_t rotary_dim =
        options.rotary_dim == 0 ? head_dim_value : options.rotary_dim;

    const std::uint32_t sequence_length =
        checked_u32(query.shape()[1], "sequence length");
    const std::uint32_t head_dim =
        checked_u32(head_dim_value, "head dimension");
    const std::uint32_t rotary_pair_count =
        checked_u32(rotary_dim / 2U, "rotary pair count");
    const std::uint32_t position_offset =
        checked_u32(options.position_offset, "position offset");

    switch (query.dtype()) {
        case Dtype::F32:
            launch_rope<float>(
                query, key, cos_cache, sin_cache, token_count,
                sequence_length, head_dim, rotary_pair_count,
                position_offset, stream
            );
            break;

        case Dtype::F16:
            launch_rope<half>(
                query, key, cos_cache, sin_cache, token_count,
                sequence_length, head_dim, rotary_pair_count,
                position_offset, stream
            );
            break;

        case Dtype::BF16:
            launch_rope<__nv_bfloat16>(
                query, key, cos_cache, sin_cache, token_count,
                sequence_length, head_dim, rotary_pair_count,
                position_offset, stream
            );
            break;

        default:
            throw std::invalid_argument("RoPE: unsupported dtype");
    }

    CUDA_CHECK(cudaGetLastError());
}
