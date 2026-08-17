#pragma once

#include "core/tensor.h"
#include "tokenizer/bpe_tokenizer.h"

#include <cuda_runtime.h>

#include <span>

/** @brief Kernel configuration for embedding lookup. */
struct EmbeddingOptions {
    /** @brief CUDA thread-block size. */
    int block_size = 128;
    /** @brief Enables validation of token IDs against the vocabulary size. */
    bool bounds_check = true;
};

/**
 * @brief Copies token IDs produced by BpeTokenizer to a preallocated CUDA buffer.
 * The destination must contain at least token_ids.size() elements.
 * @param device_destination Destination CUDA buffer.
 * @param token_ids Host token IDs to upload.
 * @param stream CUDA stream used for the asynchronous copy.
 */
void embedding_upload_token_ids(
    bpe::TokenId* device_destination,
    std::span<const bpe::TokenId> token_ids,
    cudaStream_t stream = nullptr
);

/**
 * @brief Looks up embeddings for token IDs already stored in CUDA memory.
 *
 * Layouts:
 *   device_token_ids: [token_count], bpe::TokenId (uint32), CUDA memory
 *   weight:           [vocabulary_size, hidden_size], CUDA F32/F16/BF16
 *   output:           [..., hidden_size], same dtype as weight
 *
 * @param output Destination embedding tensor.
 * @param device_token_ids CUDA token-ID buffer.
 * @param token_count Number of token IDs to process.
 * @param weight Embedding matrix.
 * @param stream CUDA stream used for the operation.
 * @param options Embedding-kernel configuration.
 */
void embedding_forward(
    Tensor& output,
    const bpe::TokenId* device_token_ids,
    std::size_t token_count,
    const Tensor& weight,
    cudaStream_t stream = nullptr,
    const EmbeddingOptions& options = {}
);

/**
 * @brief Convenience overload for a flattened tokenizer batch already uploaded to CUDA.
 * @param output Destination embedding tensor.
 * @param device_token_ids CUDA buffer containing flattened token IDs.
 * @param batch Flattened batch metadata.
 * @param weight Embedding matrix.
 * @param stream CUDA stream used for the operation.
 * @param options Embedding-kernel configuration.
 */
inline void embedding_forward(
    Tensor& output,
    const bpe::TokenId* device_token_ids,
    const bpe::EncodedBatch& batch,
    const Tensor& weight,
    cudaStream_t stream = nullptr,
    const EmbeddingOptions& options = {}
) {
    embedding_forward(
        output,
        device_token_ids,
        batch.tokens.size(),
        weight,
        stream,
        options
    );
}