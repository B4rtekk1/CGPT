/** @file weight_initialization.cu CUDA implementation of Transformer weight initialization. */

#include "core/weight_initialization.h"

#include <cmath>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>

namespace {
    /**
     * @brief Requires a floating-point CUDA tensor.
     *
     * @param tensor Tensor to validate.
     * @param name Diagnostic tensor name.
     *
     * @throws std::invalid_argument If the tensor is not CUDA-resident or does not
     * use a floating-point dtype.
     */
    void validate_floating(const Tensor &tensor, const char *name) {
        if (tensor.device_type() != DeviceType::CUDA || !is_floating_point(tensor.dtype()))
            throw std::invalid_argument(std::string(name) + " must be a floating-point CUDA tensor");
    }


    /**
     * @brief Requires a floating-point CUDA tensor with an exact shape.
     *
     * @param tensor Tensor to validate.
     * @param shape Expected dimensions.
     * @param name Diagnostic tensor name.
     *
     * @throws std::invalid_argument If device, dtype, or shape is invalid.
     */
    void validate_shape(const Tensor &tensor, const std::vector<std::size_t> &shape, const char *name) {
        validate_floating(tensor, name);
        if (tensor.shape() != shape)
            throw std::invalid_argument(std::string(name) + " has an invalid shape");
    }


    /**
     * @brief Validates a positive finite initialization scalar.
     *
     * @param value Value to validate.
     * @param name Diagnostic parameter name.
     *
     * @throws std::invalid_argument If @p value is non-finite or non-positive.
     */
    void validate_positive_finite(const float value, const char *name) {
        if (!std::isfinite(value) || value <= 0.0F)
            throw std::invalid_argument(std::string(name) + " must be finite and positive");
    }
} // namespace


/**
 * @brief Initializes a CUDA tensor from a zero-mean normal distribution.
 *
 * Random values are generated deterministically on the host using
 * `std::mt19937_64`, then converted and copied into the destination tensor.
 *
 * @param tensor Floating-point CUDA tensor to initialize.
 * @param standard_deviation Positive standard deviation of the distribution.
 * @param seed Deterministic random-number-generator seed.
 *
 * @throws std::invalid_argument If the tensor or standard deviation is invalid.
 * @throws std::runtime_error If copying values to CUDA fails.
 */
void initialize_weight_tensor(Tensor &tensor, const float standard_deviation,
                              const std::uint64_t seed) {
    validate_floating(tensor, "weight");
    validate_positive_finite(standard_deviation, "standard deviation");

    std::mt19937_64 generator(seed);
    std::normal_distribution<float> distribution(0.0F, standard_deviation);
    std::vector<float> values(tensor.numel());
    for (float &value: values) value = distribution(generator);
    tensor.copy_from_host(values);
}


/**
 * @brief Initializes a normalization scale tensor to ones.
 *
 * @param tensor Floating-point CUDA tensor to initialize.
 *
 * @throws std::invalid_argument If @p tensor is not a floating-point CUDA tensor.
 */
void initialize_norm_tensor(Tensor &tensor) {
    validate_floating(tensor, "norm");
    tensor.copy_from_host(std::vector<float>(tensor.numel(), 1.0F));
}


/**
 * @brief Initializes every parameter tensor of a Transformer model.
 *
 * Normalization scales are initialized to one. Matrix parameters are sampled
 * from zero-mean normal distributions with deterministic, layer-specific seeds.
 * Output and down projections use residual scaling
 * @f[
 *     \sigma_{residual} =
 *     \frac{\sigma_{weight}}{\sqrt{2\,N_{layers}}}.
 * @f]
 *
 * If the LM head and token embedding are the same Tensor object, the shared
 * storage is initialized only once.
 *
 * @param weights Mutable references to all model parameter tensors.
 * @param options Model dimensions and layer count.
 * @param initialization Initialization standard deviations and base seed.
 *
 * @throws std::invalid_argument If model dimensions, layer counts,
 * initialization scalars, tensor devices, dtypes, or tensor shapes are invalid.
 *
 * @note The current implementation validates `embedding_stddev`, but the token
 * embedding is initialized through the common matrix path using
 * `weight_stddev`.
 */
void initialize_transformer_weights(
    const MutableTransformerModelWeights weights,
    const TransformerModelOptions &options,
    const WeightInitializationOptions &initialization) {
    if (options.vocabulary_size == 0 || options.num_layers == 0)
        throw std::invalid_argument("model options must contain a vocabulary and at least one layer");
    if (weights.layers.size() != options.num_layers)
        throw std::invalid_argument("model layer count does not match model options");
    validate_positive_finite(initialization.embedding_stddev, "embedding standard deviation");
    validate_positive_finite(initialization.weight_stddev, "weight standard deviation");

    const auto &block = options.block_options;
    const auto matrix = [&](Tensor &tensor, const std::vector<std::size_t> &shape,
                            const char *name, const std::uint64_t seed) {
        validate_shape(tensor, shape, name);
        initialize_weight_tensor(tensor, initialization.weight_stddev, seed);
    };
    const auto norm = [&](Tensor &tensor, const char *name) {
        validate_shape(tensor, {block.hidden_size}, name);
        initialize_norm_tensor(tensor);
    };
    const auto head_norm = [&](Tensor &tensor, const char *name) {
        validate_shape(tensor, {block.head_dim}, name);
        initialize_norm_tensor(tensor);
    };

    matrix(weights.token_embedding, {options.vocabulary_size, block.hidden_size},
           "token embedding", initialization.seed);
    // A tied classifier is the very same Tensor as the input embedding and
    // must not be reinitialized with an independent random matrix.
    if (&weights.lm_head != &weights.token_embedding) {
        matrix(weights.lm_head, {options.vocabulary_size, block.hidden_size},
               "lm head", initialization.seed + 1);
    }
    norm(weights.final_norm, "final norm");

    const float residual_stddev = initialization.weight_stddev /
                                  std::sqrt(2.0F * static_cast<float>(options.num_layers));
    for (std::size_t layer = 0; layer < weights.layers.size(); ++layer) {
        auto &current = weights.layers[layer];
        const std::uint64_t base = initialization.seed + 2 + layer * 9;
        norm(current.attention_norm, "attention norm");
        norm(current.ffn_norm, "ffn norm");
        matrix(current.q_projection,
               {block.num_query_heads * block.head_dim, block.hidden_size}, "q projection", base);
        matrix(current.k_projection,
               {block.num_kv_heads * block.head_dim, block.hidden_size}, "k projection", base + 1);
        head_norm(current.q_norm, "q norm");
        head_norm(current.k_norm, "k norm");
        matrix(current.v_projection,
               {block.num_kv_heads * block.head_dim, block.hidden_size}, "v projection", base + 2);
        validate_shape(current.o_projection,
                       {block.hidden_size, block.num_query_heads * block.head_dim}, "o projection");
        initialize_weight_tensor(current.o_projection, residual_stddev, base + 3);
        matrix(current.gate_proj, {block.intermediate_size, block.hidden_size}, "gate projection", base + 4);
        matrix(current.up_proj, {block.intermediate_size, block.hidden_size}, "up projection", base + 5);
        validate_shape(current.down_proj, {block.hidden_size, block.intermediate_size}, "down projection");
        initialize_weight_tensor(current.down_proj, residual_stddev, base + 6);
    }
}
