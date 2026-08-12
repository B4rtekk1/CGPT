#pragma once

#include "core/tensor.h"
#include "tokenizer/bpe_tokenizer.h"

#include <cstddef>

/**
 * Computes mean next-token cross-entropy and its gradient with respect to
 * logits. `logits` and `gradient` are [rows, vocabulary_size]; targets is a
 * CUDA buffer containing one token id for each row; every ID must be smaller
 * than `vocabulary_size`. `loss` is an F32 CUDA tensor with shape [1].
 *
 * The result is numerically stable (log-sum-exp) and `gradient` contains the
 * gradient of the mean loss: (softmax(logits) - one_hot(target)) / rows.
 */
void cross_entropy_forward_backward(
    Tensor& loss,
    Tensor& gradient,
    const Tensor& logits,
    const bpe::TokenId* device_targets,
    std::size_t target_count,
    cudaStream_t stream = nullptr);
