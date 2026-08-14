#pragma once

#include "core/transformer_model.h"

#include <cstddef>
#include <cstdint>
#include <vector>

/** Options for initializing decoder-only Transformer weights. */
struct WeightInitializationOptions {
    std::uint64_t seed = 0xC0FFEEULL;
    float embedding_stddev = 0.02F;
    float weight_stddev = 0.02F;
};

/** Mutable view used only while constructing or initializing a model. */
struct MutableTransformerBlockWeights {
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

struct MutableTransformerModelWeights {
    Tensor& token_embedding;
    std::vector<MutableTransformerBlockWeights>& layers;
    Tensor& final_norm;
    Tensor& lm_head;
};

/**
 * Initializes all model parameters in-place on their existing device/dtype.
 * Matrix weights use a seeded normal distribution; RMSNorm weights are set to
 * one. The supplied model options are also used to validate every shape.
 */
void initialize_transformer_weights(
    MutableTransformerModelWeights weights,
    const TransformerModelOptions& options,
    const WeightInitializationOptions& initialization = {});

/** Initializes one ordinary trainable tensor with a seeded normal distribution. */
void initialize_weight_tensor(
    Tensor& tensor,
    float standard_deviation,
    std::uint64_t seed);

/** Initializes one RMSNorm scale tensor to one. */
void initialize_norm_tensor(Tensor& tensor);
