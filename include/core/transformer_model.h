#pragma once

#include "core/cublas_context.h"
#include "core/tensor.h"
#include "core/transformer_block.h"
#include "ops/embedding.h"

#include <cstddef>
#include <vector>

/**
 * @brief Configuration shared by all layers of a decoder-only Transformer.
 */
struct TransformerModelOptions {
    /** @brief Number of tokens in the model vocabulary. */
    std::size_t vocabulary_size = 0;
    /** @brief Number of Transformer blocks. */
    std::size_t num_layers = 0;
    /** @brief Architecture and execution options for each block. */
    TransformerBlockOptions block_options{};
};

/**
 * @brief Non-owning references to decoder-only Transformer parameters.
 *
 * All referenced tensors must remain alive for the complete forward or
 * backward operation that uses this structure.
 */
struct TransformerModelWeights {
    /** @brief Token embedding matrix `[vocabulary_size, hidden_size]`. */
    const Tensor& token_embedding;                 // [vocabulary_size, hidden_size]
    /** @brief Parameters of all Transformer blocks. */
    const std::vector<TransformerBlockWeights>& layers;
    /** @brief Final RMSNorm parameters `[hidden_size]`. */
    const Tensor& final_norm;                      // [hidden_size]
    /** @brief Language-model head matrix `[vocabulary_size, hidden_size]`. */
    const Tensor& lm_head;                         // [vocabulary_size, hidden_size]
};

/**
 * @brief Reusable activation buffers for one fixed batch and sequence shape.
 */
struct TransformerModelWorkspace {
    /** @brief Current hidden-state buffer. */
    Tensor hidden;
    /** @brief Output of the currently processed layer. */
    Tensor layer_output;
    /** @brief Final normalized hidden states. */
    Tensor normalized;
    /** @brief Per-layer forward workspaces. */
    std::vector<TransformerBlockWorkspace> layers;
    /** @brief Saved inputs for the individual Transformer layers. */
    std::vector<Tensor> layer_inputs;

    /**
     * @brief Allocates activation buffers for a model shape.
     * @param options Model and block configuration.
     * @param batch_size Number of sequences in a batch.
     * @param sequence_length Number of tokens per sequence.
     * @param dtype Data type used by the activation buffers.
     */
    TransformerModelWorkspace(
        const TransformerModelOptions& options,
        std::size_t batch_size,
        std::size_t sequence_length,
        Dtype dtype = Dtype::F16
    );
};

/**
 * @brief Writable gradients for all trainable model parameters.
 */
struct TransformerModelGradients {
    /** @brief Gradient of the token embedding matrix. */
    Tensor& token_embedding;
    /** @brief Gradients for all Transformer blocks. */
    std::vector<TransformerBlockGradients> layers;
    /** @brief Gradient of the final normalization parameters. */
    Tensor& final_norm;
    /** @brief Gradient of the language-model head. */
    Tensor& lm_head;
};

/**
 * @brief Reusable temporary tensors for model backpropagation.
 */
struct TransformerModelBackwardWorkspace {
    /** @brief Gradient of the normalized final hidden states. */
    Tensor grad_normalized;
    /** @brief Gradient of the model hidden states. */
    Tensor grad_hidden;
    /** @brief Gradient passed to the preceding layer. */
    Tensor grad_previous;
    /** @brief Temporary language-model head bias or reduction buffer. */
    Tensor lm_head_bias;
    /** @brief Per-layer backward workspaces. */
    std::vector<TransformerBlockBackwardWorkspace> layers;

    /**
     * @brief Allocates backward buffers for a model shape.
     * @param options Model and block configuration.
     * @param batch_size Number of sequences in a batch.
     * @param sequence_length Number of tokens per sequence.
     * @param dtype Data type used by the temporary tensors.
     */
    TransformerModelBackwardWorkspace(const TransformerModelOptions& options,
        std::size_t batch_size, std::size_t sequence_length, Dtype dtype = Dtype::F16);
};

/**
 * @brief Runs a complete decoder-only Transformer forward pass.
 *
 * @param logits Destination logits tensor with shape
 *        `[batch_size * sequence_length, vocabulary_size]`.
 * @param device_token_ids CUDA buffer containing `[batch_size * sequence_length]`
 *        token IDs.
 * @param batch_size Number of sequences in the batch.
 * @param sequence_length Number of tokens per sequence.
 * @param weights Non-owning model parameter references.
 * @param workspace Reusable forward activation buffers.
 * @param cos_cache Shared cosine rotary-embedding cache.
 * @param sin_cache Shared sine rotary-embedding cache.
 * @param cublas_context cuBLASLt context used by model operations.
 * @param stream CUDA stream used for the operation.
 * @param options Model and block configuration.
 * @param position_offset Offset into the rotary-embedding caches.
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

/**
 * @brief Backpropagates a completed transformer_model_forward call.
 * @param grad_logits Gradient of the output logits.
 * @param device_token_ids CUDA buffer containing the input token IDs.
 * @param batch_size Number of sequences in the batch.
 * @param sequence_length Number of tokens per sequence.
 * @param weights Non-owning model parameter references.
 * @param gradients Destination parameter gradients.
 * @param forward_workspace Workspace produced by the forward pass.
 * @param backward_workspace Reusable backward-pass buffers.
 * @param cos_cache Shared cosine rotary-embedding cache.
 * @param sin_cache Shared sine rotary-embedding cache.
 * @param cublas_context cuBLASLt context used by model operations.
 * @param stream CUDA stream used for the operation.
 * @param options Model and block configuration.
 * @param position_offset Offset into the rotary-embedding caches.
 */
void transformer_model_backward(
    const Tensor& grad_logits,
    const bpe::TokenId* device_token_ids,
    std::size_t batch_size,
    std::size_t sequence_length,
    const TransformerModelWeights& weights,
    TransformerModelGradients gradients,
    const TransformerModelWorkspace& forward_workspace,
    TransformerModelBackwardWorkspace& backward_workspace,
    const Tensor& cos_cache,
    const Tensor& sin_cache,
    const CublasLtContext& cublas_context,
    cudaStream_t stream,
    const TransformerModelOptions& options,
    std::size_t position_offset = 0);