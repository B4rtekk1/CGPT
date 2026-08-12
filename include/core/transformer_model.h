#pragma once

#include "core/cublas_context.h"
#include "core/tensor.h"
#include "core/transformer_block.h"
#include "ops/embedding.h"

#include <cstddef>
#include <vector>

/** Configuration shared by all layers of a decoder-only Transformer. */
struct TransformerModelOptions {
    std::size_t vocabulary_size = 0;
    std::size_t num_layers = 0;
    TransformerBlockOptions block_options{};
};

/** Non-owning references to model parameters.  All tensors must remain alive
 * for the complete call to transformer_model_forward. */
struct TransformerModelWeights {
    const Tensor& token_embedding;                 // [vocabulary_size, hidden_size]
    const std::vector<TransformerBlockWeights>& layers;
    const Tensor& final_norm;                      // [hidden_size]
    const Tensor& lm_head;                         // [vocabulary_size, hidden_size]
};

/** Reusable activation buffers for one fixed batch and sequence shape. */
struct TransformerModelWorkspace {
    Tensor hidden;
    Tensor layer_output;
    Tensor normalized;
    std::vector<TransformerBlockWorkspace> layers;

    TransformerModelWorkspace(
        const TransformerModelOptions& options,
        std::size_t batch_size,
        std::size_t sequence_length,
        Dtype dtype = Dtype::F16
    );
};

/**
 * Runs a complete decoder-only Transformer forward pass.
 *
 * Layouts: token_ids is a CUDA buffer with [batch_size * sequence_length]
 * entries; logits is [batch_size * sequence_length, vocabulary_size].  The
 * RoPE caches are shared by all layers and must cover the requested positions.
 */
void transformer_model_forward(
    Tensor& logits,
    const bpe::TokenId* device_token_ids,
    std::size_t batch_size,
    std::size_t sequence_length,
    const TransformerModelWeights& weights,
    TransformerModelWorkspace& workspace,
    const Tensor& cos_cache,
    const Tensor& sin_cache,
    const CublasLtContext& cublas_context,
    cudaStream_t stream,
    const TransformerModelOptions& options,
    std::size_t position_offset = 0
);
