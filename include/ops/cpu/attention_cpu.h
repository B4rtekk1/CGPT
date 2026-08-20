#pragma once

#include "core/tensor.h"
#include "ops/attention.h"

void flash_gqa_attention_forward_cpu(
    Tensor& output, const Tensor& query, const Tensor& key, const Tensor& value,
    const FlashAttentionOptions& options = {});

void flash_gqa_attention_forward_with_lse_cpu(
    Tensor& output, Tensor& logsumexp, const Tensor& query, const Tensor& key,
    const Tensor& value, const FlashAttentionOptions& options = {});
