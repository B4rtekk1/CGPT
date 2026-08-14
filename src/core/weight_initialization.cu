#include "core/weight_initialization.h"

#include <cmath>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>

namespace {

void validate_floating(const Tensor& tensor, const char* name) {
    if (tensor.device_type() != DeviceType::CUDA || !is_floating_point(tensor.dtype()))
        throw std::invalid_argument(std::string(name) + " must be a floating-point CUDA tensor");
}

void validate_shape(const Tensor& tensor, const std::vector<std::size_t>& shape, const char* name) {
    validate_floating(tensor, name);
    if (tensor.shape() != shape)
        throw std::invalid_argument(std::string(name) + " has an invalid shape");
}

void validate_positive_finite(const float value, const char* name) {
    if (!std::isfinite(value) || value <= 0.0F)
        throw std::invalid_argument(std::string(name) + " must be finite and positive");
}

} // namespace

void initialize_weight_tensor(Tensor& tensor, const float standard_deviation,
                              const std::uint64_t seed) {
    validate_floating(tensor, "weight");
    validate_positive_finite(standard_deviation, "standard deviation");

    std::mt19937_64 generator(seed);
    std::normal_distribution<float> distribution(0.0F, standard_deviation);
    std::vector<float> values(tensor.numel());
    for (float& value : values) value = distribution(generator);
    tensor.copy_from_host(values);
}

void initialize_norm_tensor(Tensor& tensor) {
    validate_floating(tensor, "norm");
    tensor.copy_from_host(std::vector<float>(tensor.numel(), 1.0F));
}

void initialize_transformer_weights(
    const MutableTransformerModelWeights weights,
    const TransformerModelOptions& options,
    const WeightInitializationOptions& initialization) {
    if (options.vocabulary_size == 0 || options.num_layers == 0)
        throw std::invalid_argument("model options must contain a vocabulary and at least one layer");
    if (weights.layers.size() != options.num_layers)
        throw std::invalid_argument("model layer count does not match model options");
    validate_positive_finite(initialization.embedding_stddev, "embedding standard deviation");
    validate_positive_finite(initialization.weight_stddev, "weight standard deviation");

    const auto& block = options.block_options;
    const auto matrix = [&](Tensor& tensor, const std::vector<std::size_t>& shape,
                            const char* name, const std::uint64_t seed) {
        validate_shape(tensor, shape, name);
        initialize_weight_tensor(tensor, initialization.weight_stddev, seed);
    };
    const auto norm = [&](Tensor& tensor, const char* name) {
        validate_shape(tensor, {block.hidden_size}, name);
        initialize_norm_tensor(tensor);
    };

    matrix(weights.token_embedding, {options.vocabulary_size, block.hidden_size},
           "token embedding", initialization.seed);
    matrix(weights.lm_head, {options.vocabulary_size, block.hidden_size},
           "lm head", initialization.seed + 1);
    norm(weights.final_norm, "final norm");

    for (std::size_t layer = 0; layer < weights.layers.size(); ++layer) {
        auto& current = weights.layers[layer];
        const std::uint64_t base = initialization.seed + 2 + layer * 9;
        norm(current.attention_norm, "attention norm");
        norm(current.ffn_norm, "ffn norm");
        matrix(current.q_projection,
               {block.num_query_heads * block.head_dim, block.hidden_size}, "q projection", base);
        matrix(current.k_projection,
               {block.num_kv_heads * block.head_dim, block.hidden_size}, "k projection", base + 1);
        matrix(current.v_projection,
               {block.num_kv_heads * block.head_dim, block.hidden_size}, "v projection", base + 2);
        matrix(current.o_projection,
               {block.hidden_size, block.num_query_heads * block.head_dim}, "o projection", base + 3);
        matrix(current.gate_proj, {block.intermediate_size, block.hidden_size}, "gate projection", base + 4);
        matrix(current.up_proj, {block.intermediate_size, block.hidden_size}, "up projection", base + 5);
        matrix(current.down_proj, {block.hidden_size, block.intermediate_size}, "down projection", base + 6);
    }
}
