#pragma once

#include <cstddef>

#include "ops/linear.h"

/**
 * @brief Architecture and execution parameters for one Transformer block.
 */
struct TransformerBlockOptions {
    /** @brief Hidden representation size. */
    std::size_t hidden_size = 0;
    /** @brief Feed-forward intermediate representation size. */
    std::size_t intermediate_size = 0;
    /** @brief Number of query attention heads. */
    std::size_t num_query_heads = 0;
    /** @brief Number of key/value attention heads used by GQA or MQA. */
    std::size_t num_kv_heads = 0;
    /** @brief Width of one attention head. */
    std::size_t head_dim = 0;
    /** @brief Number of dimensions to which rotary embeddings are applied. */
    std::size_t rotary_dim =0;

    /** @brief Epsilon used by RMS normalization. */
    float rms_epsilon = 1.0e-5F;
    /** @brief Enables causal attention masking. */
    bool causal = true;

    /** @brief Options passed to the block's linear projections. */
    LinearOptions linear_options{};
};

/**
 * @brief Read-only trainable parameters used by one Transformer block.
 */
struct TransformerBlockWeights {
    /** @brief Attention input RMS-normalization parameters. */
    const Tensor& attention_norm;
    /** @brief Query projection weights. */
    const Tensor& q_projection;
    /** @brief Key projection weights. */
    const Tensor& k_projection;
    /** @brief Query RMSNorm scale applied before RoPE. */
    const Tensor& q_norm;
    /** @brief Key RMSNorm scale applied before RoPE. */
    const Tensor& k_norm;
    /** @brief Value projection weights. */
    const Tensor& v_projection;
    /** @brief Attention output projection weights. */
    const Tensor& o_projection;

    /** @brief Feed-forward input RMS-normalization parameters. */
    const Tensor& ffn_norm;
    /** @brief SwiGLU gating projection weights. */
    const Tensor& gate_proj;
    /** @brief SwiGLU up-projection weights. */
    const Tensor& up_proj;
    /** @brief Feed-forward down-projection weights. */
    const Tensor& down_proj;
};

/**
 * @brief Forward-pass tensors reused by a Transformer block.
 */
struct TransformerBlockWorkspace {
    /** @brief Attention normalization output. */
    Tensor norm;

    /** @brief Query activations. */
    Tensor query;
    /** @brief Key activations. */
    Tensor key;
    /** @brief Value activations. */
    Tensor value;
    /** @brief Q/K values before QK-Norm, saved for backward. */
    Tensor query_pre_norm;
    Tensor key_pre_norm;
    /** @brief Output produced by the attention operation. */
    Tensor attention_output;
    /** @brief FP32 softmax log-sum-exp values saved for the backward pass. */
    Tensor attention_logsumexp;
    /** @brief Attention output after the output projection. */
    Tensor attention_projection;

    /** @brief SwiGLU gate projection output. */
    Tensor gate;
    /** @brief SwiGLU up-projection output. */
    Tensor up;
    /** @brief Activated SwiGLU output. */
    Tensor activated;
    /** @brief Feed-forward projection output. */
    Tensor ffn_output;
};

/**
 * @brief Writable gradients of the trainable tensors in one Transformer block.
 */
struct TransformerBlockGradients {
    /** @brief Gradient of the attention normalization parameters. */
    Tensor& attention_norm;
    /** @brief Gradient of the query projection weights. */
    Tensor& q_projection;
    /** @brief Gradient of the key projection weights. */
    Tensor& k_projection;
    /** @brief Gradient of the query RMSNorm scale. */
    Tensor& q_norm;
    /** @brief Gradient of the key RMSNorm scale. */
    Tensor& k_norm;
    /** @brief Gradient of the value projection weights. */
    Tensor& v_projection;
    /** @brief Gradient of the attention output projection weights. */
    Tensor& o_projection;
    /** @brief Gradient of the feed-forward normalization parameters. */
    Tensor& ffn_norm;
    /** @brief Gradient of the SwiGLU gate projection weights. */
    Tensor& gate_proj;
    /** @brief Gradient of the SwiGLU up-projection weights. */
    Tensor& up_proj;
    /** @brief Gradient of the feed-forward down-projection weights. */
    Tensor& down_proj;
};

/**
 * @brief Temporary tensors used by transformer_block_backward.
 */
struct TransformerBlockBackwardWorkspace {
    /** @brief Gradient with respect to the attention normalization input. */
    Tensor grad_attention_norm_input;
    /** @brief Attention normalization output saved for backward computation. */
    Tensor attention_norm_output;
    /** @brief Gradient with respect to the feed-forward normalization input. */
    Tensor grad_ffn_norm_input;
    /** @brief Gradient propagated through the residual connection. */
    Tensor grad_residual;
    /** @brief Gradient of the projected attention output. */
    Tensor grad_attention_projection;
    /** @brief Gradient of the attention output. */
    Tensor grad_attention_output;
    /** @brief Gradient of the query activations. */
    Tensor grad_query;
    /** @brief Gradient of the key activations. */
    Tensor grad_key;
    /** @brief Gradient of the value activations. */
    Tensor grad_value;
    /** @brief Gradient of the query before rotary embedding. */
    Tensor grad_query_pre_rope;
    /** @brief Gradient of the key before rotary embedding. */
    Tensor grad_key_pre_rope;
    /** @brief Gradient of the SwiGLU gate projection output. */
    Tensor grad_gate;
    /** @brief Gradient of the SwiGLU up-projection output. */
    Tensor grad_up;
    /** @brief Gradient of the activated SwiGLU output. */
    Tensor grad_activated;
    /** @brief Temporary hidden-layer bias buffer. */
    Tensor hidden_bias;
    /** @brief Temporary intermediate-layer bias buffer. */
    Tensor intermediate_bias;
    /** @brief Temporary key/value bias buffer. */
    Tensor kv_bias;

    /**
     * @brief Allocates backward-pass workspace tensors.
     * @param options Transformer block dimensions and execution options.
     * @param batch_size Number of sequences in a batch.
     * @param sequence_length Number of tokens processed per sequence.
     * @param dtype Data type used for the temporary tensors.
     */
    TransformerBlockBackwardWorkspace(const TransformerBlockOptions& options,
        std::size_t batch_size, std::size_t sequence_length, Dtype dtype = Dtype::F16);
};

/**
 * @brief Executes the forward pass of one decoder-only Transformer block.
 *
 * The block applies independent pre-RMSNorm attention and SwiGLU branches to
 * the same input, then combines both with the residual connection:
 * `output = input + attention(RMSNorm_attn(input)) + MLP(RMSNorm_ffn(input))`.
 * Intermediate results needed by training are written to @p workspace.
 *
 * @param output Destination tensor for the block output.
 * @param input Input hidden states.
 * @param weights Read-only block parameters.
 * @param workspace Forward-pass workspace receiving intermediate tensors.
 * @param cos_cache Cosine rotary-embedding cache.
 * @param sin_cache Sine rotary-embedding cache.
 * @param cublas_context cuBLASLt context used by linear projections.
 * @param stream CUDA stream used for the operation.
 * @param options Block architecture and execution options.
 * @param position_offset Offset into the rotary-embedding cache.
 */
void transformer_block_forward(
    Tensor& output,
    const Tensor& input,
    const TransformerBlockWeights& weights,
    TransformerBlockWorkspace& workspace,
    const Tensor& cos_cache,
    const Tensor& sin_cache,
    const CublasLtContext& cublas_context,
    cudaStream_t stream,
    const TransformerBlockOptions& options,
    std::size_t position_offset = 0,
    Tensor* key_cache = nullptr,
    Tensor* value_cache = nullptr,
    std::size_t cache_length = 0
    );

/**
 * @brief Backpropagates through one decoder-only Transformer block.
 *
 * The matching forward pass must have been run into @p forward_workspace
 * beforehand, since it contains
 * the saved Q/K/V and MLP activations. Parameter gradients are overwritten.
 *
 * @param grad_input Destination gradient with respect to the block input.
 * @param grad_output Gradient with respect to the block output.
 * @param input Input hidden states from the forward pass.
 * @param weights Read-only block parameters.
 * @param gradients Destination parameter gradients; existing values are overwritten.
 * @param forward_workspace Workspace produced by the matching forward pass.
 * @param backward_workspace Temporary tensors used by the backward pass.
 * @param cos_cache Cosine rotary-embedding cache.
 * @param sin_cache Sine rotary-embedding cache.
 * @param cublas_context cuBLASLt context used by linear projections.
 * @param stream CUDA stream used for the operation.
 * @param options Block architecture and execution options.
 * @param position_offset Offset into the rotary-embedding cache.
 */
void transformer_block_backward(
    Tensor& grad_input,
    const Tensor& grad_output,
    const Tensor& input,
    const TransformerBlockWeights& weights,
    TransformerBlockGradients gradients,
    const TransformerBlockWorkspace& forward_workspace,
    TransformerBlockBackwardWorkspace& backward_workspace,
    const Tensor& cos_cache,
    const Tensor& sin_cache,
    const CublasLtContext& cublas_context,
    cudaStream_t stream,
    const TransformerBlockOptions& options,
    std::size_t position_offset = 0);
