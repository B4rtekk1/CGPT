#pragma once

#include "core/tensor.h"
#include "tokenizer/bpe_tokenizer.h"

/**
 * @brief Computes mean next-token cross-entropy and its gradient with respect to
 * logits. `logits` and `gradient` are [rows, vocabulary_size]; targets is a
 * CUDA buffer containing one token id for each row; every ID must be smaller
 * than `vocabulary_size`. `loss` is an F32 CUDA tensor with shape [1].
 *
 * The result is numerically stable (log-sum-exp). The implementation stages
 * centered exponentials in `gradient`, avoiding a redundant exp() pass; on
 * return it contains (softmax(logits) - one_hot(target)) / rows.
 *
 * @param loss Destination F32 scalar loss tensor.
 * @param gradient Destination tensor receiving the logits gradient.
 * @param logits Input logits tensor.
 * @param device_targets CUDA target-token buffer.
 * @param target_count Number of target IDs and logits rows.
 * @param gradient_scale Multiplier applied only to the returned gradient. This
 * enables mixed-precision loss scaling while leaving @p loss unchanged.
 * @param stream CUDA stream used for the operation.
 */
void cross_entropy_forward_backward(
    Tensor& loss,
    Tensor& gradient,
    const Tensor& logits,
    const bpe::TokenId* device_targets,
    std::size_t target_count,
    cudaStream_t stream = nullptr,
    float gradient_scale = 1.0F);

/**
 * @brief Computes mean next-token cross-entropy without producing a gradient.
 */
void cross_entropy_forward(
    Tensor& loss,
    const Tensor& logits,
    const bpe::TokenId* device_targets,
    std::size_t target_count,
    cudaStream_t stream = nullptr);
