/** @file CPU cross-entropy loss implementation. */
#pragma once

#include "core/tensor.h"
#include "tokenizer/bpe_tokenizer.h"

/** Computes token-level cross-entropy loss from CPU logits and targets. */
void cross_entropy_forward_cpu(
    Tensor& loss, const Tensor& logits, const bpe::TokenId* targets,
    std::size_t target_count);
