#pragma once

#include "ops/backward/attention_backward.h"

void flash_gqa_attention_backward_cpu(
    Tensor& grad_query, Tensor& grad_key, Tensor& grad_value,
    const Tensor& grad_output, const Tensor& query, const Tensor& key,
    const Tensor& value, const FlashAttentionOptions& options = {},
    bool accumulate_grads = false);

void flash_gqa_attention_backward_with_lse_cpu(
    Tensor& grad_query, Tensor& grad_key, Tensor& grad_value,
    const Tensor& grad_output, const Tensor& output, const Tensor& logsumexp,
    const Tensor& query, const Tensor& key, const Tensor& value,
    const FlashAttentionOptions& options = {}, bool accumulate_grads = false);
