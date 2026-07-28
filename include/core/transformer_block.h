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
    Tensor attention_projection;

    Tensor gate;
    Tensor up;
    Tensor activated;
    Tensor ffn_output;
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
