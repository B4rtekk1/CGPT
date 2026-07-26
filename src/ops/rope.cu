#include "ops/rope.h"
#include "core/cuda_check.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace {
    constexpr int THREADS_PER_BLOCK = 256;
    constexpr std::uint32_t MAX_GRID_X = 65535U;

    template<typename T>
    struct RopeTypeTraits;

    template<>
    struct RopeTypeTraits<float> {
        using Scalar = float;

        __device__ __forceinline__ static float2 load_pair(
            const float* const pointer
            ) {
            return make_float2(pointer[0], pointer[1]);
        }

        __device__ __forceinline__ static void store_pair(
            float* const pointer,
            const float2 value
            ) {
            pointer[0] = value.x;
            pointer[1] = value.y;
        }

        __device__ __forceinline__ static float load_scalar(
            const float* const pointer
            ) {
            return *pointer;
        }
    };

    template<>
    struct RopeTypeTraits<half> {
        using Scalar = half;

        __device__ __forceinline__ static float2 load_pair(
            const half* const pointer
            ) {
            return __half22float2(
                *reinterpret_cast<const half2*>(pointer));
        }

        __device__ __forceinline__ static void store_pair(
            half* const pointer,
            const float2 value
            ) {
            *reinterpret_cast<half2*>(pointer) = __floats2half2_rn(value.x, value.y);
        }

        __device__ __forceinline__ static float load_scalar(
            const half* const pointer
            ) {
            return __half2float(*pointer);
        }
    };

    template<>
    struct RopeTypeTraits<__nv_bfloat16> {
        using Scalar = __nv_bfloat16;

        __device__ __forceinline__ static float2 load_pair(
            const __nv_bfloat16* const pointer
            ) {
            return __bfloat1622float2(
                *reinterpret_cast<const __nv_bfloat162*>(pointer));
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

    template<typename T>
    __global__ void rope_tensor_kernel(
        T* __restrict__ input,
        const  T* __restrict__ cos_cache,
        const T* __restrict__ sin_cache,
        const std::uint32_t sequence_length,
        const std::uint32_t head_count,
        const std::uint32_t head_dim,
        const std::uint32_t rotary_pair_count,
        const std::uint32_t position_offset
    ) {
        using Traits = RopeTypeTraits<T>;

        const std::uint32_t token_index = blockIdx.z;
        const std::uint32_t head_index = blockIdx.y;
        const std::uint32_t sequence_index = token_index % sequence_length;
        const std::uint32_t  position = position_offset + sequence_index;

        const std::size_t tensor_base = (static_cast<std::size_t>(token_index) * head_count + head_index) * head_dim;
        const std::size_t cache_base = static_cast<std::size_t>(position) * rotary_pair_count;

        for (std::uint32_t pair_index = blockIdx.x * blockDim.x + threadIdx.x; pair_index < rotary_pair_count; pair_index+=gridDim.x * blockDim.x) {
            T* const pair_pointer = input + tensor_base + static_cast<std::size_t>(pair_index) * 2U;

            const float2 value = Traits::load_pair(pair_pointer);
            const float cosine = Traits::load_scalar(cos_cache + cache_base + pair_index);
            const float sine = Traits::load_scalar(sin_cache + cache_base + pair_index);

            const float2 rotated = make_float2(
                fmaf(-value.y, sine, value.x * cosine),
                fmaf(value.x, sine, value.y * cosine)
                );

            Traits::store_pair(pair_pointer, rotated);
        }
    }

    [[nodiscard]] std::uint32_t checked_u32(
        const std::size_t value,
        const char* const description
        ) {
        if (value > std::numeric_limits<std::uint32_t>::max()) {
            throw std::overflow_error(
                std::string("Rope: ") + description + "exceeds the CUDA grid dimension limit");
        }

        return static_cast<std::uint32_t>(value);
    }

    [[nodiscard]] std::uint32_t checked_grid_dimension(
        const std::size_t value,
        const char* const description
    ) {
        if (value > 65535U) {
            throw std::overflow_error(
                std::string("RoPE: ") + description +
                " exceeds the CUDA grid dimension limit"
            );
        }

        return static_cast<std::uint32_t>(value);
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
                "RoPE: cos_cache and sin_cache must have shape "
                "[max_sequence, rotary_dim / 2]"
            );
        }

        if (query.dtype() != key.dtype() ||
            query.dtype() != cos_cache.dtype() ||
            query.dtype() != sin_cache.dtype()) {
            throw std::invalid_argument(
                "RoPE: query, key, cos_cache and sin_cache must have the same dtype"
            );
            }

        if (query.shape()[0] != key.shape()[0] ||
            query.shape()[1] != key.shape()[1] ||
            query.shape()[3] != key.shape()[3]) {
            throw std::invalid_argument(
                "RoPE: query and key must have equal batch, sequence and head_dim"
            );
            }

        if (query.shape()[1] == 0 || query.shape()[3] == 0) {
            return;
        }

        const std::size_t head_dim = query.shape()[3];
        const std::size_t rotary_dim =
            options.rotary_dim == 0 ? head_dim : options.rotary_dim;

        if (rotary_dim == 0 || rotary_dim > head_dim || rotary_dim % 2 != 0) {
            throw std::invalid_argument(
                "RoPE: rotary_dim must be even, non-zero and no greater than head_dim"
            );
        }

        const std::size_t pair_count = rotary_dim / 2;

        if ((query.dtype() == Dtype::F16 || query.dtype() == Dtype::BF16) &&
            head_dim % 2 != 0) {
            throw std::invalid_argument(
                "RoPE: FP16 and BF16 require an even head_dim for aligned pair loads"
            );
            }

        if (cos_cache.shape() != sin_cache.shape()) {
            throw std::invalid_argument(
                "RoPE: cos_cache and sin_cache must have identical shapes"
            );
        }

        if (cos_cache.shape()[1] != pair_count) {
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
                "RoPE: cache does not contain all requested token positions"
            );
        }

        static_cast<void>(checked_u32(query.shape()[1], "sequence length"));
        static_cast<void>(checked_u32(query.shape()[2], "query head count"));
        static_cast<void>(checked_u32(key.shape()[2], "key head count"));
        static_cast<void>(checked_u32(head_dim, "head dimension"));
        static_cast<void>(checked_u32(pair_count, "rotary pair count"));
        static_cast<void>(checked_u32(options.position_offset, "position offset"));

        static_cast<void>(checked_grid_dimension(query.shape()[2], "query head count"));
        static_cast<void>(checked_grid_dimension(key.shape()[2], "key head count"));

        if (query.shape()[0] != 0 &&
            query.shape()[1] >
                std::numeric_limits<std::size_t>::max() / query.shape()[0]) {
            throw std::overflow_error("RoPE: batch times sequence overflow");
                }

        static_cast<void>(checked_grid_dimension(
            query.shape()[0] * query.shape()[1],
            "batch times sequence"
        ));
    }

    [[nodiscard]] dim3 make_grid(
        const std::size_t batch_size,
        const std::size_t sequence_length,
        const std::size_t head_count,
        const std::uint32_t rotary_pair_count
        ) {
        const std::uint32_t blocks_x = std::max<std::uint32_t>(
        1U, std::min<std::uint32_t>(
            (rotary_pair_count + THREADS_PER_BLOCK - 1U) / THREADS_PER_BLOCK, MAX_GRID_X)
        );

        return {
            blocks_x,
            checked_grid_dimension(head_count, "head count"),
            checked_grid_dimension(batch_size* sequence_length, "batch times sequence")
            };
    }

    template<typename T>
    void launch_rope_tensor(
        Tensor& tensor,
        const Tensor& cos_cache,
        const Tensor& sin_cache,
        const std::uint32_t sequence_length,
        const std::uint32_t head_dim,
        const std::uint32_t rotary_pair_count,
        const std::uint32_t position_offset,
        cudaStream_t stream
    ) {
        const std::uint32_t head_count = checked_u32(tensor.shape()[2], "head count");

        rope_tensor_kernel<T><<<make_grid(
            tensor.shape()[0],
            tensor.shape()[1],
            tensor.shape()[2],
            rotary_pair_count),
        THREADS_PER_BLOCK,
        0,
        stream>>>(
            static_cast<T*>(tensor.raw_data()),
            static_cast<const T*>(cos_cache.raw_data()),
            static_cast<const T*>(sin_cache.raw_data()),
            sequence_length,
            head_count,
            head_dim,
            rotary_pair_count,
            position_offset
            );
    }

    template<typename T>
    void launch_rope(
        Tensor& query,
        Tensor& key,
        const Tensor& cos_cache,
        const Tensor& sin_cache,
        const std::uint32_t sequence_length,
        const std::uint32_t head_dim,
        const std::uint32_t rotary_pair_count,
        const std::uint32_t position_offset,
        cudaStream_t stream
        ) {
        launch_rope_tensor<T>(
            query,
            cos_cache,
            sin_cache,
            sequence_length,
            head_dim,
            rotary_pair_count,
            position_offset,
            stream
            );

        launch_rope_tensor<T>(
            key,
            cos_cache,
            sin_cache,
            sequence_length,
            head_dim,
            rotary_pair_count,
            position_offset,
            stream);
    }
}

void rope_forward(
    Tensor& query,
    Tensor& key,
    const Tensor& cos_cache,
    const Tensor& sin_cache,
    cudaStream_t stream,
    const RopeOptions& options) {

    validate_inputs(query, key, cos_cache, sin_cache, options);

    if (query.numel() == 0 || key.numel() == 0 || query.shape()[1] == 0) {
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
        checked_u32(rotary_dim / 2, "rotary pair count");
    const std::uint32_t position_offset =
        checked_u32(options.position_offset, "position offset");

    switch (query.dtype()) {
        case Dtype::F32:
            launch_rope<float>(
                query,
                key,
                cos_cache,
                sin_cache,
                sequence_length,
                head_dim,
                rotary_pair_count,
                position_offset,
                stream
            );
            break;

        case Dtype::F16:
            launch_rope<half>(
                query,
                key,
                cos_cache,
                sin_cache,
                sequence_length,
                head_dim,
                rotary_pair_count,
                position_offset,
                stream
            );
            break;

        case Dtype::BF16:
            launch_rope<__nv_bfloat16>(
                query,
                key,
                cos_cache,
                sin_cache,
                sequence_length,
                head_dim,
                rotary_pair_count,
                position_offset,
                stream
            );
            break;

        default:
            throw std::invalid_argument("RoPE: unsupported dtype");
    }

    CUDA_CHECK(cudaGetLastError());
}
