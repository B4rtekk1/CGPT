/** @file embedding_backward.cu @brief CUDA backward pass for embedding lookup. */

#include "ops/backward/embedding_backward.h"
#include "core/cuda_check.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdint>
#include <limits>
#include <stdexcept>

namespace {
    constexpr int kWarpSize = 32;

    [[nodiscard]] bool valid_block_size(const int block_size) noexcept {
        return block_size >= kWarpSize && block_size <= 1024 && block_size % kWarpSize == 0;
    }

    [[nodiscard]] bool aligned_to(const void *pointer, const std::uintptr_t alignment) noexcept {
        return reinterpret_cast<std::uintptr_t>(pointer) % alignment == 0;
    }

    template<typename T, bool BoundsCheck>
    __global__ void embedding_backward_kernel(
        T * __restrict__ grad_weight,
        const T * __restrict__ grad_output,
        const bpe::TokenId * __restrict__ token_ids,
        const bpe::TokenId vocabulary_size,
        const int hidden_size
    ) {
        const std::size_t token = blockIdx.x;
        const bpe::TokenId token_id = token_ids[token];
        if constexpr (BoundsCheck) {
            if (token_id >= vocabulary_size) {
                return;
            }
        }

        T *const weight_row = grad_weight + static_cast<std::size_t>(token_id) * hidden_size;
        const T *const output_row = grad_output + token * static_cast<std::size_t>(hidden_size);
        for (int feature = static_cast<int>(threadIdx.x); feature < hidden_size;
             feature += static_cast<int>(blockDim.x)) {
            atomicAdd(weight_row + feature, output_row[feature]);
        }
    }

    template<typename PackedT, bool BoundsCheck>
    __global__ void embedding_backward_packed_kernel(
        PackedT * __restrict__ grad_weight,
        const PackedT * __restrict__ grad_output,
        const bpe::TokenId * __restrict__ token_ids,
        const bpe::TokenId vocabulary_size,
        const int packed_hidden_size
    ) {
        const std::size_t token = blockIdx.x;
        const bpe::TokenId token_id = token_ids[token];
        if constexpr (BoundsCheck) {
            if (token_id >= vocabulary_size) {
                return;
            }
        }

        PackedT *const weight_row =
            grad_weight + static_cast<std::size_t>(token_id) * packed_hidden_size;
        const PackedT *const output_row =
            grad_output + token * static_cast<std::size_t>(packed_hidden_size);
        for (int feature = static_cast<int>(threadIdx.x); feature < packed_hidden_size;
             feature += static_cast<int>(blockDim.x)) {
            atomicAdd(weight_row + feature, output_row[feature]);
        }
    }

    void validate_inputs(
        const Tensor &grad_weight,
        const Tensor &grad_output,
        const bpe::TokenId *token_ids,
        const std::size_t token_count,
        const EmbeddingBackwardOptions &options
    ) {
        if (token_ids == nullptr) {
            throw std::invalid_argument("embedding_backward: token_ids cannot be null");
        }
        if (token_count == 0) {
            throw std::invalid_argument("embedding_backward: token_count must be positive");
        }
        if (!valid_block_size(options.block_size)) {
            throw std::invalid_argument(
                "embedding_backward: block_size must be a multiple of 32 in [32, 1024]");
        }
        if (grad_weight.device_type() != DeviceType::CUDA ||
            grad_output.device_type() != DeviceType::CUDA) {
            throw std::invalid_argument("embedding_backward: tensors must be CUDA tensors");
        }
        if (grad_weight.dtype() != grad_output.dtype()) {
            throw std::invalid_argument("embedding_backward: tensors must have the same dtype");
        }
        if (grad_weight.dim() != 2 || grad_output.dim() < 2) {
            throw std::invalid_argument(
                "embedding_backward: grad_weight must be [vocabulary_size, hidden_size] and grad_output must have rank >= 2");
        }
        const std::size_t hidden_size = grad_weight.size(1);
        if (grad_output.size(grad_output.dim() - 1) != hidden_size) {
            throw std::invalid_argument("embedding_backward: hidden sizes must match");
        }
        if (hidden_size == 0 || token_count > std::numeric_limits<std::size_t>::max() / hidden_size ||
            grad_output.numel() != token_count * hidden_size) {
            throw std::invalid_argument(
                "embedding_backward: grad_output must contain token_count * hidden_size elements");
        }
        if (token_count > std::numeric_limits<unsigned int>::max() ||
            grad_weight.size(0) > static_cast<std::size_t>(std::numeric_limits<bpe::TokenId>::max()) ||
            hidden_size > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
            throw std::invalid_argument("embedding_backward: dimensions exceed supported CUDA ranges");
        }
    }

    template<typename T>
    void launch(
        Tensor &grad_weight, const Tensor &grad_output,
        const bpe::TokenId *token_ids, const std::size_t token_count,
        cudaStream_t stream, const EmbeddingBackwardOptions &options
    ) {
        const auto vocabulary_size = static_cast<bpe::TokenId>(grad_weight.size(0));
        const auto grid = static_cast<unsigned int>(token_count);
        const int hidden_size = static_cast<int>(grad_weight.size(1));
        if (options.bounds_check) {
            embedding_backward_kernel<T, true><<<grid, options.block_size, 0, stream>>>(
                static_cast<T *>(grad_weight.raw_data()), static_cast<const T *>(grad_output.raw_data()),
                token_ids, vocabulary_size, hidden_size);
        } else {
            embedding_backward_kernel<T, false><<<grid, options.block_size, 0, stream>>>(
                static_cast<T *>(grad_weight.raw_data()), static_cast<const T *>(grad_output.raw_data()),
                token_ids, vocabulary_size, hidden_size);
        }
    }

    template<typename PackedT>
    void launch_packed(
        Tensor &grad_weight, const Tensor &grad_output,
        const bpe::TokenId *token_ids, const std::size_t token_count,
        cudaStream_t stream, const EmbeddingBackwardOptions &options
    ) {
        const auto vocabulary_size = static_cast<bpe::TokenId>(grad_weight.size(0));
        const auto grid = static_cast<unsigned int>(token_count);
        const int packed_hidden_size = static_cast<int>(grad_weight.size(1) / 2);
        if (options.bounds_check) {
            embedding_backward_packed_kernel<PackedT, true>
                <<<grid, options.block_size, 0, stream>>>(
                    static_cast<PackedT *>(grad_weight.raw_data()),
                    static_cast<const PackedT *>(grad_output.raw_data()),
                    token_ids, vocabulary_size, packed_hidden_size);
        } else {
            embedding_backward_packed_kernel<PackedT, false>
                <<<grid, options.block_size, 0, stream>>>(
                    static_cast<PackedT *>(grad_weight.raw_data()),
                    static_cast<const PackedT *>(grad_output.raw_data()),
                    token_ids, vocabulary_size, packed_hidden_size);
        }
    }
}

void embedding_backward(
    Tensor &grad_weight,
    const Tensor &grad_output,
    const bpe::TokenId *const device_token_ids,
    const std::size_t token_count,
    cudaStream_t stream,
    const EmbeddingBackwardOptions &options
) {
    validate_inputs(grad_weight, grad_output, device_token_ids, token_count, options);
    if (!options.accumulate_weight) {
        CUDA_CHECK(cudaMemsetAsync(grad_weight.raw_data(), 0, grad_weight.nbytes(), stream));
    }

    const bool can_pack = (grad_weight.size(1) & 1u) == 0 &&
        aligned_to(grad_weight.raw_data(), alignof(std::uint32_t)) &&
        aligned_to(grad_output.raw_data(), alignof(std::uint32_t));
    switch (grad_weight.dtype()) {
        case Dtype::F32:
            launch<float>(grad_weight, grad_output, device_token_ids, token_count, stream, options);
            break;
        case Dtype::F16:
            if (can_pack) {
                launch_packed<half2>(
                    grad_weight, grad_output, device_token_ids, token_count, stream, options);
            } else {
                launch<half>(
                    grad_weight, grad_output, device_token_ids, token_count, stream, options);
            }
            break;
        case Dtype::BF16:
            if (can_pack) {
                launch_packed<__nv_bfloat162>(
                    grad_weight, grad_output, device_token_ids, token_count, stream, options);
            } else {
                launch<__nv_bfloat16>(
                    grad_weight, grad_output, device_token_ids, token_count, stream, options);
            }
            break;
        default:
            throw std::invalid_argument("embedding_backward: unsupported dtype");
    }
    CUDA_CHECK(cudaGetLastError());
}
