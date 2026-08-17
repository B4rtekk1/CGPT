#pragma once

#include "core/tensor.h"

/**
 * @brief Configuration for grouped-query FlashAttention.
 */
struct FlashAttentionOptions {
    /** @brief Number of query attention heads. */
    std::size_t num_query_heads = 0;
    /** @brief Number of key/value attention heads. */
    std::size_t num_kv_heads = 0;
    /** @brief Number of elements in each attention head. */
    std::size_t head_dim = 0;

    /** @brief Scale applied to query-key dot products. */
    float attention_scale = 0.0F;

    /** @brief Enables causal masking of future key positions. */
    bool causal = true;

    /** @brief Position offset of the first query token. */
    std::size_t query_position_offset = 0;

    /** @brief Number of key/value tokens processed by one tile. */
    int key_tile_size = 16;
    /** @brief CUDA thread-block size used by the kernel. */
    int block_size = 128;

    /** @brief Enables Tensor Core execution when supported. */
    bool use_tensor_cores = true;
};

/**
 * @brief Computes grouped-query scaled dot-product attention with FlashAttention.
 *
 * The operation uses tiled online softmax and avoids materializing the full
 * attention matrix. Query, key, value, and output tensors must satisfy the
 * layout and dtype requirements of the implementation.
 *
 * @param output Destination attention output.
 * @param query Query tensor.
 * @param key Key tensor.
 * @param value Value tensor.
 * @param stream CUDA stream used for the operation.
 * @param options Attention configuration.
 */
void flash_gqa_attention_forward(
    Tensor& output,
    const Tensor& query,
    const Tensor& key,
    const Tensor& value,
    cudaStream_t stream = nullptr,
    const FlashAttentionOptions& options = {}
    );

/**
 * @brief Training variant of flash_gqa_attention_forward.
 *
 * In addition to @p output,
 * stores the per-row log-sum-exp softmax statistic in @p logsumexp.  The LSE
 * tensor must be CUDA F32 with shape [batch, query_sequence, query_heads].
 * Pass it to flash_gqa_attention_backward_with_lse to avoid recomputing the
 * online-softmax statistics during backpropagation.
 *
 * @param output Destination attention output.
 * @param logsumexp Destination FP32 log-sum-exp tensor.
 * @param query Query tensor.
 * @param key Key tensor.
 * @param value Value tensor.
 * @param stream CUDA stream used for the operation.
 * @param options Attention configuration.
 */
void flash_gqa_attention_forward_with_lse(
    Tensor& output,
    Tensor& logsumexp,
    const Tensor& query,
    const Tensor& key,
    const Tensor& value,
    cudaStream_t stream = nullptr,
    const FlashAttentionOptions& options = {}
    );