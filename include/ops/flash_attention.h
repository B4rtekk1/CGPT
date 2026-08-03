#pragma once

#include "core/tensor.h"

struct FlashAttentionOptions {
    std::size_t num_query_heads = 0;
    std::size_t num_kv_heads = 0;
    std::size_t head_dim = 0;

    float attention_scale = 0.0F;

    bool causal = true;

    std::size_t query_position_offset = 0;

    int key_tile_size = 16;
    int block_size = 128;

    bool use_tensor_cores = true;
};

void flash_gqa_attention_forward(
    Tensor& output,
    const Tensor& query,
    const Tensor& key,
    const Tensor& value,
    cudaStream_t stream = nullptr,
    const FlashAttentionOptions& options = {}
    );
