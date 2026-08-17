#pragma once

#include "ops/attention.h"
/**
 * @brief Computes gradients of scaled dot-product grouped-query attention.
 *
 * All tensors use the same contiguous layout and dtype as the forward pass.
 * The implementation recomputes the softmax probabilities in registers/shared
 * memory and never allocates the [query_sequence, key_value_sequence] score
 * matrix. With @p accumulate_grads false (the default), all three output
 * gradient tensors are overwritten; otherwise their contributions are added.
 *
 * @param grad_query Gradient with respect to the query tensor.
 * @param grad_key Gradient with respect to the key tensor.
 * @param grad_value Gradient with respect to the value tensor.
 * @param grad_output Gradient with respect to the attention output.
 * @param query Forward-pass query tensor.
 * @param key Forward-pass key tensor.
 * @param value Forward-pass value tensor.
 * @param stream CUDA stream used for the operation.
 * @param options Attention configuration.
 * @param accumulate_grads Adds to existing gradients when `true`.
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

/**
 * @brief Backpropagates attention using saved output and LSE statistics.
 *
 * The output and LSE statistics are saved by
 * flash_gqa_attention_forward_with_lse.  This is the preferred training API:
 * it avoids a separate softmax-statistics pass in the backward kernel.
 *
 * @param grad_query Gradient with respect to the query tensor.
 * @param grad_key Gradient with respect to the key tensor.
 * @param grad_value Gradient with respect to the value tensor.
 * @param grad_output Gradient with respect to the attention output.
 * @param output Forward-pass attention output.
 * @param logsumexp Saved per-row log-sum-exp statistics.
 * @param query Forward-pass query tensor.
 * @param key Forward-pass key tensor.
 * @param value Forward-pass value tensor.
 * @param stream CUDA stream used for the operation.
 * @param options Attention configuration.
 * @param accumulate_grads Adds to existing gradients when `true`.
 */
void flash_gqa_attention_backward_with_lse(
    Tensor& grad_query,
    Tensor& grad_key,
    Tensor& grad_value,
    const Tensor& grad_output,
    const Tensor& output,
    const Tensor& logsumexp,
    const Tensor& query,
    const Tensor& key,
    const Tensor& value,
    cudaStream_t stream = nullptr,
    const FlashAttentionOptions& options = {},
    bool accumulate_grads = false
    );