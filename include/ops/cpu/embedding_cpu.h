#pragma once

#include "core/tensor.h"
#include "ops/embedding.h"

/** CPU AVX2 embedding lookup from host token IDs. */
void embedding_forward_cpu(
    Tensor& output,
    const bpe::TokenId* token_ids,
    std::size_t token_count,
    const Tensor& weight,
    const EmbeddingOptions& options = {}
);
