#pragma once

#include "core/transformer_model.h"

#include <cstddef>
#include <cstdint>
#include <vector>

/**
 * @brief Options for initializing decoder-only Transformer weights.
 */
struct WeightInitializationOptions {
    /** @brief Seed used by the random number generator. */
    std::uint64_t seed = 0xC0FFEEULL;
    /** @brief Standard deviation used for token embeddings. */
    float embedding_stddev = 0.02F;
    /** @brief Standard deviation used for ordinary matrix weights. */
    float weight_stddev = 0.02F;
};

/**
 * @brief Mutable view of one Transformer block's parameters.
 *
 * The references are non-owning and are intended for model construction or
 * parameter initialization.
 */
struct MutableTransformerBlockWeights {
    /** @brief Attention normalization parameters. */
    Tensor& attention_norm;
    /** @brief Query projection weights. */
    Tensor& q_projection;
    /** @brief Key projection weights. */
    Tensor& k_projection;
    /** @brief Value projection weights. */
    Tensor& v_projection;
    /** @brief Attention output projection weights. */
    Tensor& o_projection;
    /** @brief Feed-forward normalization parameters. */
    Tensor& ffn_norm;
    /** @brief SwiGLU gate projection weights. */
    Tensor& gate_proj;
    /** @brief SwiGLU up-projection weights. */
    Tensor& up_proj;
    /** @brief Feed-forward down-projection weights. */
    Tensor& down_proj;
};

/**
 * @brief Mutable view of all parameters in a Transformer model.
 */
struct MutableTransformerModelWeights {
    /** @brief Token embedding matrix. */
    Tensor& token_embedding;
    /** @brief Mutable views of all Transformer block parameters. */
    std::vector<MutableTransformerBlockWeights>& layers;
    /** @brief Final normalization parameters. */
    Tensor& final_norm;
    /** @brief Language-model head matrix. */
    Tensor& lm_head;
};

/**
 * @brief Initializes all model parameters in-place on their existing device and data type.
 *
 * Matrix weights use a seeded normal distribution; RMSNorm weights are set to
 * one. The supplied model options are also used to validate every shape.
 *
 * @param weights Mutable references to all model parameters.
 * @param options Model configuration used for shape validation.
 * @param initialization Random initialization parameters.
 */
void initialize_transformer_weights(
    MutableTransformerModelWeights weights,
    const TransformerModelOptions& options,
    const WeightInitializationOptions& initialization = {});

/**
 * @brief Initializes one ordinary trainable tensor with a seeded normal distribution.
 * @param tensor Tensor to initialize.
 * @param standard_deviation Standard deviation of the normal distribution.
 * @param seed Random-number-generator seed.
 */
void initialize_weight_tensor(
    Tensor& tensor,
    float standard_deviation,
    std::uint64_t seed);

/**
 * @brief Initializes one RMSNorm scale tensor to one.
 * @param tensor Tensor to initialize.
 */
void initialize_norm_tensor(Tensor& tensor);