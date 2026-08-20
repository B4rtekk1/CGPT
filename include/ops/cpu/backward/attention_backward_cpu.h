/** @file CPU backward implementations of grouped-query flash attention. */
/** @file CPU backward implementations of grouped-query flash attention. */
#pragma once

#include "ops/backward/attention_backward.h"

/** Computes attention gradients without an explicitly supplied LSE tensor. */
/** Computes attention gradients without an explicitly supplied LSE tensor. */
void flash_gqa_attention_backward_cpu(
    Tensor& grad_query, Tensor& grad_key, Tensor& grad_value,
    const Tensor& grad_output, const Tensor& query, const Tensor& key,
    const Tensor& value, const FlashAttentionOptions& options = {},
    bool accumulate_grads = false);

/** Computes attention gradients using precomputed log-sum-exp values. */
/** Computes attention gradients using precomputed log-sum-exp values. */
void flash_gqa_attention_backward_with_lse_cpu(
    Tensor& grad_query, Tensor& grad_key, Tensor& grad_value,
    const Tensor& grad_output, const Tensor& output, const Tensor& logsumexp,
    const Tensor& query, const Tensor& key, const Tensor& value,
    const FlashAttentionOptions& options = {}, bool accumulate_grads = false);
