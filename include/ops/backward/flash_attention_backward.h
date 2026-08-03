#pragma once

#include "ops/flash_attention.h"
/**
 * @brief Computes gradients of scaled dot-product grouped-query attention.
 *
 * All tensors use the same contiguous layout and dtype as the forward pass.
 * The implementation recomputes the softmax probabilities in registers/shared
 * memory and never allocates the [query_sequence, key_value_sequence] score
 * matrix. With @p accumulate_grads false (the default), all three output
 * gradient tensors are overwritten; otherwise their contributions are added.
 */
void flash_gqa_attention_backward(
    Tensor& grad_query,
    Tensor& grad_key,
    Tensor& grad_value,
    const Tensor& grad_output,
    const Tensor& query,
    const Tensor& key,
    const Tensor& value,
    cudaStream_t stream = nullptr,
    const FlashAttentionOptions& options = {},
    bool accumulate_grads = false
    );
