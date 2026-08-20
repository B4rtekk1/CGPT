/** @file CPU backward implementation of token embeddings. */
/** @file CPU backward implementation of token embeddings. */
#pragma once

#include "ops/backward/embedding_backward.h"

/** Accumulates embedding-weight gradients for the supplied token IDs. */
/** Accumulates embedding-weight gradients for the supplied token IDs. */
void embedding_backward_cpu(
    Tensor& grad_weight, const Tensor& grad_output,
    const bpe::TokenId* token_ids, std::size_t token_count,
    const EmbeddingBackwardOptions& options = {});
