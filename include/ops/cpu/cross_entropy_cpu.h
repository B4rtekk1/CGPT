#pragma once

#include "core/tensor.h"
#include "tokenizer/bpe_tokenizer.h"

void cross_entropy_forward_cpu(
    Tensor& loss, const Tensor& logits, const bpe::TokenId* targets,
    std::size_t target_count);

void cross_entropy_forward_backward_cpu(
    Tensor& loss, Tensor& gradient, const Tensor& logits,
    const bpe::TokenId* targets, std::size_t target_count,
    float gradient_scale = 1.0F);
