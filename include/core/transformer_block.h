#pragma once

#include <cstddef>

#include "ops/linear.h"

struct TransformerBlockOptions {
    std::size_t hidden_size = 0;
    std::size_t intermediate_size = 0;
    std::size_t num_query_heads = 0;
    std::size_t num_kv_heads = 0;
    std::size_t head_dim = 0;
    std::size_t rotary_dim =0;

    float rms_epsilon = 1.0e-5F;
    bool causal = true;

    LinearOptions linear_options{};
};

struct TransformerBlockWeights {
    const Tensor& attention_norm;
    const Tensor& q_projection;
    const Tensor& k_projection;
    const Tensor& v_projection;
    const Tensor& o_projection;

    const Tensor& ffn_norm;
    const Tensor& gate_proj;
    const Tensor& up_proj;
    const Tensor& down_proj;
};

struct TransformerBlockWorkspace {
    Tensor norm;

    Tensor query;
    Tensor key;
    Tensor value;
    Tensor attention_output;
    // FP32 softmax log-sum-exp saved by the training Flash Attention API.
    Tensor attention_logsumexp;
    Tensor attention_projection;

    Tensor gate;
    Tensor up;
    Tensor activated;
    Tensor ffn_output;
};

/** Gradients of the trainable tensors in one Transformer block. */
struct TransformerBlockGradients {
    Tensor& attention_norm;
    Tensor& q_projection;
    Tensor& k_projection;
    Tensor& v_projection;
    Tensor& o_projection;
    Tensor& ffn_norm;
    Tensor& gate_proj;
    Tensor& up_proj;
    Tensor& down_proj;
};

/** Temporary CUDA buffers used by transformer_block_backward. */
struct TransformerBlockBackwardWorkspace {
    Tensor grad_attention_norm_input;
    Tensor attention_norm_output;
    Tensor grad_ffn_norm_input;
    Tensor grad_residual;
    Tensor grad_attention_projection;
    Tensor grad_attention_output;
    Tensor grad_query;
    Tensor grad_key;
    Tensor grad_value;
    Tensor grad_query_pre_rope;
    Tensor grad_key_pre_rope;
    Tensor grad_gate;
    Tensor grad_up;
    Tensor grad_activated;
    Tensor hidden_bias;
    Tensor intermediate_bias;
    Tensor kv_bias;

    TransformerBlockBackwardWorkspace(const TransformerBlockOptions& options,
        std::size_t batch_size, std::size_t sequence_length, Dtype dtype = Dtype::F16);
};

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
    std::size_t position_offset = 0
    );

/**
 * Backpropagates one decoder-only Transformer block. The matching forward pass
 * must have been run into @p forward_workspace beforehand, since it contains
 * the saved Q/K/V and MLP activations. Parameter gradients are overwritten.
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
