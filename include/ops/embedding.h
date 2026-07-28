#pragma once

#include "core/tensor.h"
#include "tokenizer/bpe_tokenizer.h"

#include <cuda_runtime.h>

#include <span>

struct EmbeddingOptions {
    int block_size = 128;
    bool bounds_check = true;
};

/**
 * Copies token IDs produced by BpeTokenizer to a preallocated CUDA buffer.
 * The destination must contain at least token_ids.size() elements.
 */
void embedding_upload_token_ids(
    bpe::TokenId* device_destination,
    std::span<const bpe::TokenId> token_ids,
    cudaStream_t stream = nullptr
);

/**
 * Looks up embeddings for token IDs already stored in CUDA memory.
 *
 * Layouts:
 *   device_token_ids: [token_count], bpe::TokenId (uint32), CUDA memory
 *   weight:           [vocabulary_size, hidden_size], CUDA F32/F16/BF16
 *   output:           [..., hidden_size], same dtype as weight
 */
void embedding_forward(
    Tensor& output,
    const bpe::TokenId* device_token_ids,
    std::size_t token_count,
    const Tensor& weight,
    cudaStream_t stream = nullptr,
    const EmbeddingOptions& options = {}
);

/** Convenience overload for a flattened tokenizer batch already uploaded to CUDA. */
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