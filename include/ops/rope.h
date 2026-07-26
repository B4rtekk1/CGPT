#pragma once

#include "core/tensor.h"

struct RopeOptions {
    std::size_t rotary_dim = 0;
    std::size_t position_offset = 0;
};

/**
 * @brief Applies rotary positional embeddings.
 *
 * Expected tensor layouts:
 *
 * @code
 * query      [batch, sequence, query_heads, head_dim]
 * key        [batch, sequence, kv_heads,    head_dim]
 * cos_cache  [max_sequence, rotary_dim / 2]
 * sin_cache  [max_sequence, rotary_dim / 2]
 * @endcode
 */
void rope_forward(
    Tensor& query,
    Tensor& key,
    const Tensor& cos_cache,
    const Tensor& sin_cache,
    cudaStream_t stream,
    const RopeOptions& rope_options = {}
    );