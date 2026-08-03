#pragma once

#include "ops/rope.h"

/**
 * @brief Computes input gradients for in-place rotary positional embeddings.
 *
 * Given gradients with respect to the rotated query and key tensors, computes
 * the gradients with respect to their pre-RoPE values. The tensor layouts are:
 * @code
 * grad_query          [batch, sequence, query_heads, head_dim]
 * grad_key            [batch, sequence, kv_heads,    head_dim]
 * grad_rotated_query  [batch, sequence, query_heads, head_dim]
 * grad_rotated_key    [batch, sequence, kv_heads,    head_dim]
 * cos_cache           [max_sequence, rotary_dim / 2]
 * sin_cache           [max_sequence, rotary_dim / 2]
 * @endcode
 *
 * The non-rotary tail of each head is copied unchanged. All tensors must be
 * CUDA tensors of the same supported dtype (F32, F16 or BF16).
 */
void rope_backward(
    Tensor& grad_query,
    Tensor& grad_key,
    const Tensor& grad_rotated_query,
    const Tensor& grad_rotated_key,
    const Tensor& cos_cache,
    const Tensor& sin_cache,
    cudaStream_t stream = nullptr,
    const RopeOptions& rope_options = {}
);
