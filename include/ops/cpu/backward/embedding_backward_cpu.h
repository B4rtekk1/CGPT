#pragma once

#include "ops/backward/embedding_backward.h"

void embedding_backward_cpu(
    Tensor& grad_weight, const Tensor& grad_output,
    const bpe::TokenId* token_ids, std::size_t token_count,
    const EmbeddingBackwardOptions& options = {});
