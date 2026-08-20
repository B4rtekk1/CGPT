/** @file CPU grouped-query flash-attention implementations. */
#pragma once

#include "core/tensor.h"
#include "ops/attention.h"

/** Computes grouped-query attention on the CPU. */
void flash_gqa_attention_forward_cpu(
    Tensor& output, const Tensor& query, const Tensor& key, const Tensor& value,
    const FlashAttentionOptions& options = {});

/** Computes grouped-query attention and returns the log-sum-exp values. */
void flash_gqa_attention_forward_with_lse_cpu(
    Tensor& output, Tensor& logsumexp, const Tensor& query, const Tensor& key,
    const Tensor& value, const FlashAttentionOptions& options = {});
