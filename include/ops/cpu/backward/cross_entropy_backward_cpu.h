#pragma once

#include "core/tensor.h"
#include "tokenizer/bpe_tokenizer.h"

/** CPU cross-entropy backward with mean loss and logits gradient. */
void cross_entropy_forward_backward_cpu(
    Tensor& loss, Tensor& gradient, const Tensor& logits,
    const bpe::TokenId* targets, std::size_t target_count,
    float gradient_scale = 1.0F);
