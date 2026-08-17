#pragma once

#include "ops/embedding.h"

/** @brief Kernel configuration for embedding-table backpropagation. */
struct EmbeddingBackwardOptions {
    /** @brief CUDA thread-block size. */
    int block_size = 128;
    /** @brief Enables validation of token IDs against the vocabulary size. */
    bool bounds_check = true;
    /** @brief Adds to existing weight gradients instead of overwriting them. */
    bool accumulate_weight = false;
};

/**
 * @brief Computes or accumulates embedding-table gradients for a token-ID batch.
 *
 * For every token @c t, adds @c grad_output[t, :] to
 * @c grad_weight[token_ids[t], :]. Repeated token IDs are safely reduced with
 * CUDA atomic operations. Invalid IDs are skipped when @c bounds_check is set.
 *
 * Layouts: @c grad_weight is `[vocabulary_size, hidden_size]` and
 * @c grad_output is `[..., hidden_size]` with exactly
 * `token_count * hidden_size` elements. Both tensors must be CUDA tensors with
 * the same F32, F16 or BF16 dtype.
 *
 * @param grad_weight Destination embedding-table gradient.
 * @param grad_output Gradient of the embedding output.
 * @param device_token_ids CUDA token-ID buffer.
 * @param token_count Number of token IDs and output rows.
 * @param stream CUDA stream used for the operation.
 * @param options Backward-kernel configuration.
 */
void embedding_backward(
    Tensor& grad_weight,
    const Tensor& grad_output,
    const bpe::TokenId* device_token_ids,
    std::size_t token_count,
    cudaStream_t stream = nullptr,
    const EmbeddingBackwardOptions& options = {}
);