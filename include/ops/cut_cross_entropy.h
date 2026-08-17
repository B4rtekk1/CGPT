#pragma once

#include "core/tensor.h"
#include "tokenizer/bpe_tokenizer.h"

/**
 * @brief Computes Cut Cross-Entropy (CCE) for a linear classifier.
 *
 * Computes the mean cross-entropy for `input * classifier^T` without ever
 * materializing the [rows, vocabulary_size] logits tensor.  `input` is
 * [rows, hidden_size], `classifier` and `classifier_gradient` are
 * [vocabulary_size, hidden_size], and `input_gradient` is [rows, hidden_size].
 * All tensors are CUDA F32. Target IDs are a CUDA buffer with one value per
 * row. Gradients overwrite their output tensors.
 *
 * @param loss Destination scalar loss tensor.
 * @param input_gradient Destination gradient with respect to @p input.
 * @param classifier_gradient Destination gradient with respect to @p classifier.
 * @param input Input hidden states.
 * @param classifier Linear classifier weights.
 * @param device_targets CUDA target-token buffer.
 * @param target_count Number of target IDs and input rows.
 * @param stream CUDA stream used for the operation.
 */
void cut_cross_entropy_forward_backward(
    Tensor& loss,
    Tensor& input_gradient,
    Tensor& classifier_gradient,
    const Tensor& input,
    const Tensor& classifier,
    const bpe::TokenId* device_targets,
    std::size_t target_count,
    cudaStream_t stream = nullptr);