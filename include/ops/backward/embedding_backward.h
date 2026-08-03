#pragma once

#include "ops/embedding.h"

struct EmbeddingBackwardOptions {
    int block_size = 128;
    bool bounds_check = true;
    bool accumulate_weight = false;
};

/**
 * @brief Accumulates embedding-table gradients for a token-ID batch.
 *
 * For every token @c t, adds @c grad_output[t, :] to
 * @c grad_weight[token_ids[t], :]. Repeated token IDs are safely reduced with
 * CUDA atomic operations. Invalid IDs are skipped when @c bounds_check is set.
 *
 * Layouts: @c grad_weight is `[vocabulary_size, hidden_size]` and
 * @c grad_output is `[..., hidden_size]` with exactly
 * `token_count * hidden_size` elements. Both tensors must be CUDA tensors with
 * the same F32, F16 or BF16 dtype.
 */
void embedding_backward(
    Tensor& grad_weight,
    const Tensor& grad_output,
    const bpe::TokenId* device_token_ids,
    std::size_t token_count,
    cudaStream_t stream = nullptr,
    const EmbeddingBackwardOptions& options = {}
);
