#pragma once

#include "core/tensor.h"

/** @brief Configuration for rotary positional embeddings. */
struct RopeOptions {
    /** @brief Rotary dimension; zero uses the full head dimension. */
    std::size_t rotary_dim = 0;
    /** @brief First position represented by sequence index zero. */
    std::size_t position_offset = 0;
};

/**
 * @brief Applies rotary positional embeddings.
 *
 * Expected tensor layouts:
 * @code
 * query      [batch, sequence, query_heads, head_dim]
 * key        [batch, sequence, kv_heads,    head_dim]
 * cos_cache  [max_sequence, rotary_dim / 2]
 * sin_cache  [max_sequence, rotary_dim / 2]
 * @endcode
 *
 * Supported dtypes: F32, F16 and BF16. All four tensors must use the same dtype.
 *
 * @param query Query tensor modified in-place.
 * @param key Key tensor modified in-place.
 * @param cos_cache Cosine rotary-embedding cache.
 * @param sin_cache Sine rotary-embedding cache.
 * @param stream CUDA stream used for the operation.
 * @param rope_options Rotary-embedding configuration.
 */
void rope_forward(
    Tensor& query,
    Tensor& key,
    const Tensor& cos_cache,
    const Tensor& sin_cache,
    cudaStream_t stream,
    const RopeOptions& rope_options = {}
    );