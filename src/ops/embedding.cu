/**
 * @file embedding.cu
 * @brief CUDA implementation of the token embedding operator.
 *
 * @details
 * This translation unit gathers rows from an embedding matrix using token identifiers stored in device memory.
 * One CUDA blok processes one token and cooperatively copies the corresponding row to the output tensor.
 *
 * The implementation supports FP32, FP16 and BF16 tensors. When pointer alignment and the hidden
 * dimension allow it, the implementation selects a vectorized copy using uint4, half2 or bfloat162 values.
 * Invalid token identifier=ers can optionally be converted to an all-zero embedding row.
 *
 * @note The public declaration are located in `include/ops/embedding.h`.
 * @warning `device_token_ids` must point to CUDA_accessible memory. This file does not copy token identifiers automatically in
 * `embedding_forward`; use embedding_upload_token_ids when needed.
 */

#include "ops/embedding.h"
#include "core/cuda_check.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <limits>
#include <stdexcept>

namespace {
    constexpr int WARP_SIZE = 32;

    [[nodiscard]] bool valid_block_size(const int block_size) noexcept {
        return block_size >= WARP_SIZE && block_size <= 1024 && block_size % WARP_SIZE == 0;
    }

    [[nodiscard]] bool aligned_to(const void *pointer, const std::uintptr_t alignment) noexcept {
        return reinterpret_cast<std::uintptr_t>(pointer) % alignment == 0;
    }

    void validate_embedding(
        const Tensor &output,
        const bpe::TokenId *token_ids,
        const std::size_t token_count,
        const Tensor &weight,
        const EmbeddingOptions &options
    ) {
        if (token_ids == nullptr) {
            throw std::invalid_argument("embedding_forward: token_ids cannot be null");
        }
        if (token_count == 0) {
            throw std::invalid_argument("embedding_forward: token_count must be positive");
        }
        if (!valid_block_size(options.block_size)) {
            throw std::invalid_argument(
                "embedding_forward: block_size must be a multiple of 32 in [32, 1024]");
        }
        if (weight.device_type() != DeviceType::CUDA ||
            output.device_type() != DeviceType::CUDA) {
            throw std::invalid_argument(
                "embedding_forward: output and weight must be CUDA tensors");
        }
        if (weight.dtype() != output.dtype()) {
            throw std::invalid_argument(
                "embedding_forward: output and weight must have the same dtype");
        }
        if (weight.dim() != 2) {
            throw std::invalid_argument(
                "embedding_forward: weight must have shape [vocabulary_size, hidden_size]");
        }
        if (output.dim() < 2) {
            throw std::invalid_argument(
                "embedding_forward: output must have at least two dimensions");
        }

        const std::size_t vocabulary_size = weight.size(0);
        const std::size_t hidden_size = weight.size(1);

        if (output.size(output.dim() - 1) != hidden_size) {
            throw std::invalid_argument(
                "embedding_forward: output last dimension must equal hidden_size");
        }
        if (hidden_size != 0 &&
            token_count > std::numeric_limits<std::size_t>::max() / hidden_size) {
            throw std::overflow_error("embedding_forward: output element count overflows");
        }
        if (output.numel() != token_count * hidden_size) {
            throw std::invalid_argument(
                "embedding_forward: output must contain token_count * hidden_size elements");
        }
        if (token_count > std::numeric_limits<unsigned int>::max()) {
            throw std::invalid_argument(
                "embedding_forward: token_count exceeds CUDA grid.x range");
        }
        if (vocabulary_size >
            static_cast<std::size_t>(std::numeric_limits<bpe::TokenId>::max())) {
            throw std::invalid_argument(
                "embedding_forward: vocabulary_size exceeds bpe::TokenId range");
        }
        if (hidden_size > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
            throw std::invalid_argument(
                "embedding_forward: hidden_size exceeds supported range");
        }
    }

    template<typename T, bool BoundsCheck>
    __global__ void embedding_scalar_kernel(
        T* __restrict__ output,
        const bpe::TokenId* __restrict__ token_ids,
        const T* __restrict__ weight,
        const bpe::TokenId vocabulary_size,
        const int hidden_size
        ) {
        const auto token_index = static_cast<std::size_t>(blockIdx.x);
        const bpe::TokenId token_id = token_ids[token_index];
        T* output_row = output + token_index * static_cast<std::size_t>(hidden_size);

        if constexpr (BoundsCheck) {
            if (token_id >= vocabulary_size) {
                for (int feature = static_cast<int>(threadIdx.x); feature < hidden_size; feature += static_cast<int>(blockDim.x)) {
                    output_row[feature] = T{};
                }
                return;
            }
        }

        const T* weight_row = weight + static_cast<std::size_t>(token_id) * hidden_size;

        for (int feature = static_cast<int>(threadIdx.x); feature < hidden_size; feature += static_cast<int>(blockDim.x)) {
            output_row[feature] = weight_row[feature];
        }
    }

    template<typename PackedT, bool BoundsCheck>
    __global__ void embedding_packed_kernel(
        PackedT* __restrict__ output,
        const bpe::TokenId* __restrict__ token_ids,
        const PackedT* __restrict__ weight,
        const bpe::TokenId vocabulary_size,
        const int packed_hidden_size
        ) {
        const auto token_index = static_cast<std::size_t>(blockIdx.x);
        const bpe::TokenId token_id = token_ids[token_index];
        PackedT* output_row = output + token_index * static_cast<std::size_t>(packed_hidden_size);

        if constexpr (BoundsCheck) {
            if (token_id >= vocabulary_size) {
                for (int feature = static_cast<int>(threadIdx.x); feature < packed_hidden_size; feature += static_cast<int>(blockDim.x)) {
                    output_row[feature] = PackedT{};
                }
                return;
            }
        }

        const PackedT* weight_row = weight + static_cast<std::size_t>(token_id) * packed_hidden_size;

        for (int feature = static_cast<int>(threadIdx.x); feature < packed_hidden_size; feature += static_cast<int>(blockDim.x)) {
            output_row[feature] = weight_row[feature];
        }
    }

    template<typename T>
void launch_scalar(
    Tensor& output,
    const bpe::TokenId* token_ids,
    const Tensor& weight,
    const std::size_t token_count,
    const int hidden_size,
    cudaStream_t stream,
    const EmbeddingOptions& options
) {
        const auto grid = static_cast<unsigned int>(token_count);
        const auto vocabulary_size = static_cast<bpe::TokenId>(weight.size(0));

        if (options.bounds_check) {
            embedding_scalar_kernel<T, true><<<grid, options.block_size, 0, stream>>>(
                static_cast<T*>(output.raw_data()), token_ids,
                static_cast<const T*>(weight.raw_data()),
                vocabulary_size, hidden_size);
        } else {
            embedding_scalar_kernel<T, false><<<grid, options.block_size, 0, stream>>>(
                static_cast<T*>(output.raw_data()), token_ids,
                static_cast<const T*>(weight.raw_data()),
                vocabulary_size, hidden_size);
        }
    }

    template<typename PackedT>
    void launch_packed(
        Tensor& output,
        const bpe::TokenId* token_ids,
        const Tensor& weight,
        const std::size_t token_count,
        const int packed_hidden_size,
        cudaStream_t stream,
        const EmbeddingOptions& options
    ) {
        const auto grid = static_cast<unsigned int>(token_count);
        const auto vocabulary_size = static_cast<bpe::TokenId>(weight.size(0));

        if (options.bounds_check) {
            embedding_packed_kernel<PackedT, true><<<grid, options.block_size, 0, stream>>>(
                static_cast<PackedT*>(output.raw_data()), token_ids,
                static_cast<const PackedT*>(weight.raw_data()),
                vocabulary_size, packed_hidden_size);
        } else {
            embedding_packed_kernel<PackedT, false><<<grid, options.block_size, 0, stream>>>(
                static_cast<PackedT*>(output.raw_data()), token_ids,
                static_cast<const PackedT*>(weight.raw_data()),
                vocabulary_size, packed_hidden_size);
        }
    }

}

void embedding_upload_token_ids(
    bpe::TokenId* const device_destination,
    const std::span<const bpe::TokenId> token_ids,
    cudaStream_t stream
) {
    if (token_ids.empty()) {
        return;
    }
    if (device_destination == nullptr) {
        throw std::invalid_argument(
            "embedding_upload_token_ids: destination cannot be null");
    }

    CUDA_CHECK(cudaMemcpyAsync(
        device_destination,
        token_ids.data(),
        token_ids.size_bytes(),
        cudaMemcpyHostToDevice,
        stream));
}

void embedding_forward(
    Tensor& output,
    const bpe::TokenId* const device_token_ids,
    const std::size_t token_count,
    const Tensor& weight,
    cudaStream_t stream,
    const EmbeddingOptions& options
) {
    validate_embedding(output, device_token_ids, token_count, weight, options);

    const int hidden_size = static_cast<int>(weight.size(1));
    const std::size_t row_bytes = weight.nbytes() / weight.size(0);
    const bool can_pack_128 = row_bytes % sizeof(uint4) == 0 &&
        aligned_to(output.raw_data(), alignof(uint4)) &&
        aligned_to(weight.raw_data(), alignof(uint4));
    const bool can_pack = hidden_size % 2 == 0 &&
        aligned_to(output.raw_data(), alignof(std::uint32_t)) &&
        aligned_to(weight.raw_data(), alignof(std::uint32_t));

    if (can_pack_128) {
        launch_packed<uint4>(
            output, device_token_ids, weight, token_count,
            static_cast<int>(row_bytes / sizeof(uint4)), stream, options);
    } else if (weight.dtype() == Dtype::F32) {
        launch_scalar<float>(
            output, device_token_ids, weight, token_count, hidden_size, stream, options);
    } else if (weight.dtype() == Dtype::F16 && can_pack) {
        launch_packed<__half2>(
            output, device_token_ids, weight, token_count, hidden_size / 2, stream, options);
    } else if (weight.dtype() == Dtype::BF16 && can_pack) {
        launch_packed<__nv_bfloat162>(
            output, device_token_ids, weight, token_count, hidden_size / 2, stream, options);
    } else if (weight.dtype() == Dtype::F16) {
        launch_scalar<__half>(
            output, device_token_ids, weight, token_count, hidden_size, stream, options);
    } else if (weight.dtype() == Dtype::BF16) {
        launch_scalar<__nv_bfloat16>(
            output, device_token_ids, weight, token_count, hidden_size, stream, options);
    } else {
        throw std::invalid_argument("embedding_forward: unsupported dtype");
    }

    CUDA_CHECK(cudaGetLastError());
}
