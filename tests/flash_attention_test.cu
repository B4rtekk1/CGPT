#include "core/cuda_check.h"
#include "ops/attention.h"
#include "ops/backward/attention_backward.h"
#include "cuda_benchmark.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

std::size_t index4(
    std::size_t batch, std::size_t sequence, std::size_t head,
    std::size_t dimension, std::size_t sequence_size,
    std::size_t head_count, std::size_t head_dim
) {
    return ((batch * sequence_size + sequence) * head_count + head) * head_dim
        + dimension;
}

void expect_close(
    const std::vector<float>& actual, const std::vector<float>& expected,
    float tolerance
) {
    if (actual.size() != expected.size()) {
        throw std::runtime_error("Flash attention test: result size mismatch");
    }

    for (std::size_t i = 0; i < actual.size(); ++i) {
        const float difference = std::fabs(actual[i] - expected[i]);
        if (!std::isfinite(actual[i]) || difference > tolerance) {
            throw std::runtime_error(
                "Flash attention test: mismatch at index " + std::to_string(i)
            );
        }
    }
}

std::vector<float> attention_reference(
    const std::vector<float>& query, const std::vector<float>& key,
    const std::vector<float>& value, std::size_t batch_size,
    std::size_t query_sequence, std::size_t key_value_sequence,
    const FlashAttentionOptions& options
) {
    std::vector<float> output(query.size());
    const std::size_t heads_per_kv =
        options.num_query_heads / options.num_kv_heads;
    const float scale = options.attention_scale > 0.0F
        ? options.attention_scale
        : 1.0F / std::sqrt(static_cast<float>(options.head_dim));

    for (std::size_t batch = 0; batch < batch_size; ++batch) {
        for (std::size_t q_pos = 0; q_pos < query_sequence; ++q_pos) {
            const std::size_t visible_keys = options.causal
                ? std::min(key_value_sequence,
                    options.query_position_offset + q_pos + 1U)
                : key_value_sequence;
            for (std::size_t q_head = 0; q_head < options.num_query_heads;
                 ++q_head) {
                const std::size_t kv_head = q_head / heads_per_kv;
                std::vector<float> scores(visible_keys);
                float max_score = -INFINITY;
                for (std::size_t k_pos = 0; k_pos < visible_keys; ++k_pos) {
                    float score = 0.0F;
                    for (std::size_t d = 0; d < options.head_dim; ++d) {
                        score += query[index4(batch, q_pos, q_head, d,
                            query_sequence, options.num_query_heads,
                            options.head_dim)]
                            * key[index4(batch, k_pos, kv_head, d,
                                key_value_sequence, options.num_kv_heads,
                                options.head_dim)];
                    }
                    scores[k_pos] = score * scale;
                    max_score = std::max(max_score, scores[k_pos]);
                }

                float denominator = 0.0F;
                for (float& score : scores) {
                    score = std::exp(score - max_score);
                    denominator += score;
                }
                for (std::size_t d = 0; d < options.head_dim; ++d) {
                    float result = 0.0F;
                    for (std::size_t k_pos = 0; k_pos < visible_keys; ++k_pos) {
                        result += scores[k_pos]
                            * value[index4(batch, k_pos, kv_head, d,
                                key_value_sequence, options.num_kv_heads,
                                options.head_dim)];
                    }
                    output[index4(batch, q_pos, q_head, d, query_sequence,
                        options.num_query_heads, options.head_dim)] =
                        result / denominator;
                }
            }
        }
    }
    return output;
}

struct AttentionGradients {
    std::vector<float> query;
    std::vector<float> key;
    std::vector<float> value;
};

AttentionGradients attention_backward_reference(
    const std::vector<float>& grad_output, const std::vector<float>& query,
    const std::vector<float>& key, const std::vector<float>& value,
    std::size_t batch_size, std::size_t query_sequence,
    std::size_t key_value_sequence, const FlashAttentionOptions& options
) {
    AttentionGradients result{
        std::vector<float>(query.size()), std::vector<float>(key.size()),
        std::vector<float>(value.size())};
    const std::size_t heads_per_kv = options.num_query_heads / options.num_kv_heads;
    const float scale = options.attention_scale > 0.0F
        ? options.attention_scale : 1.0F / std::sqrt(static_cast<float>(options.head_dim));

    for (std::size_t batch = 0; batch < batch_size; ++batch) {
        for (std::size_t q_pos = 0; q_pos < query_sequence; ++q_pos) {
            const std::size_t visible = options.causal
                ? std::min(key_value_sequence, options.query_position_offset + q_pos + 1U)
                : key_value_sequence;
            for (std::size_t q_head = 0; q_head < options.num_query_heads; ++q_head) {
                const std::size_t kv_head = q_head / heads_per_kv;
                std::vector<float> probability(visible);
                std::vector<float> d_probability(visible);
                float maximum = -INFINITY;
                for (std::size_t k_pos = 0; k_pos < visible; ++k_pos) {
                    float score = 0.0F;
                    float dp = 0.0F;
                    for (std::size_t d = 0; d < options.head_dim; ++d) {
                        const auto qi = index4(batch, q_pos, q_head, d, query_sequence,
                            options.num_query_heads, options.head_dim);
                        const auto ki = index4(batch, k_pos, kv_head, d, key_value_sequence,
                            options.num_kv_heads, options.head_dim);
                        score += query[qi] * key[ki];
                        dp += grad_output[qi] * value[ki];
                    }
                    probability[k_pos] = score * scale;
                    d_probability[k_pos] = dp;
                    maximum = std::max(maximum, probability[k_pos]);
                }
                float normalizer = 0.0F;
                for (float& p : probability) {
                    p = std::exp(p - maximum);
                    normalizer += p;
                }
                float softmax_dot = 0.0F;
                for (std::size_t k_pos = 0; k_pos < visible; ++k_pos) {
                    probability[k_pos] /= normalizer;
                    softmax_dot += probability[k_pos] * d_probability[k_pos];
                }
                for (std::size_t k_pos = 0; k_pos < visible; ++k_pos) {
                    const float d_score = probability[k_pos] * (d_probability[k_pos] - softmax_dot);
                    for (std::size_t d = 0; d < options.head_dim; ++d) {
                        const auto qi = index4(batch, q_pos, q_head, d, query_sequence,
                            options.num_query_heads, options.head_dim);
                        const auto ki = index4(batch, k_pos, kv_head, d, key_value_sequence,
                            options.num_kv_heads, options.head_dim);
                        result.query[qi] += scale * d_score * key[ki];
                        result.key[ki] += scale * d_score * query[qi];
                        result.value[ki] += probability[k_pos] * grad_output[qi];
                    }
                }
            }
        }
    }
    return result;
}

std::vector<float> values(std::size_t count, float phase) {
    std::vector<float> result(count);
    for (std::size_t i = 0; i < count; ++i) {
        result[i] = 0.45F * std::sin(phase + static_cast<float>(i) * 0.31F)
            + 0.1F * std::cos(static_cast<float>(i) * 0.17F);
    }
    return result;
}

void run_case(
    std::size_t batch_size, std::size_t query_sequence,
    std::size_t key_value_sequence, FlashAttentionOptions options,
    Dtype dtype, float tolerance, cudaStream_t stream = nullptr
) {
    const std::vector<std::size_t> query_shape{
        batch_size, query_sequence, options.num_query_heads, options.head_dim};
    const std::vector<std::size_t> key_value_shape{
        batch_size, key_value_sequence, options.num_kv_heads, options.head_dim};
    const std::vector<float> query_values = values(
        batch_size * query_sequence * options.num_query_heads * options.head_dim,
        0.2F);
    const std::vector<float> key_values = values(
        batch_size * key_value_sequence * options.num_kv_heads * options.head_dim,
        0.7F);
    const std::vector<float> value_values = values(key_values.size(), 1.4F);
    const std::vector<float> expected = attention_reference(
        query_values, key_values, value_values, batch_size, query_sequence,
        key_value_sequence, options);

    Tensor query(query_shape, dtype);
    Tensor key(key_value_shape, dtype);
    Tensor value(key_value_shape, dtype);
    Tensor output(query_shape, dtype);
    query.copy_from_host(query_values);
    key.copy_from_host(key_values);
    value.copy_from_host(value_values);

    flash_gqa_attention_forward(output, query, key, value, stream, options);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    std::vector<float> actual(output.numel());
    output.copy_to_host(actual);
    expect_close(actual, expected, tolerance);
}

template <typename Function>
void expect_invalid_argument(Function&& function) {
    try {
        function();
    } catch (const std::invalid_argument&) {
        return;
    }
    throw std::runtime_error("Flash attention test: invalid input was accepted");
}

FlashAttentionOptions valid_options() {
    FlashAttentionOptions options;
    options.num_query_heads = 2;
    options.num_kv_heads = 1;
    options.head_dim = 32;
    options.block_size = 32;
    options.key_tile_size = 2;
    options.use_tensor_cores = true;
    return options;
}

void test_validation() {
    const auto call = [](const FlashAttentionOptions& options) {
        Tensor query({1, 2, 2, 32}, Dtype::F16);
        Tensor key({1, 3, 1, 32}, Dtype::F16);
        Tensor value({1, 3, 1, 32}, Dtype::F16);
        Tensor output({1, 2, 2, 32}, Dtype::F16);
        flash_gqa_attention_forward(output, query, key, value, nullptr, options);
    };

    FlashAttentionOptions options = valid_options();
    options.num_query_heads = 0;
    expect_invalid_argument([&] { call(options); });
    options = valid_options(); options.num_query_heads = 3;
    expect_invalid_argument([&] { call(options); });
    options = valid_options(); options.head_dim = 16;
    expect_invalid_argument([&] { call(options); });
    options = valid_options(); options.head_dim = 256;
    expect_invalid_argument([&] { call(options); });
    options = valid_options(); options.use_tensor_cores = false;
    expect_invalid_argument([&] { call(options); });

    options = valid_options();
    expect_invalid_argument([&] {
        Tensor q({1, 2, 32}, Dtype::F16); Tensor k({1, 3, 1, 32}, Dtype::F16);
        Tensor v({1, 3, 1, 32}, Dtype::F16); Tensor o({1, 2, 2, 32}, Dtype::F16);
        flash_gqa_attention_forward(o, q, k, v, nullptr, options);
    });
    expect_invalid_argument([&] {
        Tensor q({1, 2, 2, 32}, Dtype::F16); Tensor k({1, 3, 2, 32}, Dtype::F16);
        Tensor v({1, 3, 2, 32}, Dtype::F16); Tensor o({1, 2, 2, 32}, Dtype::F16);
        flash_gqa_attention_forward(o, q, k, v, nullptr, options);
    });
    expect_invalid_argument([&] {
        Tensor q({1, 2, 2, 32}, Dtype::F16); Tensor k({1, 3, 1, 32}, Dtype::F16);
        Tensor v({1, 2, 1, 32}, Dtype::F16); Tensor o({1, 2, 2, 32}, Dtype::F16);
        flash_gqa_attention_forward(o, q, k, v, nullptr, options);
    });
    expect_invalid_argument([&] {
        Tensor q({1, 2, 2, 32}, Dtype::F16); Tensor k({1, 3, 1, 32}, Dtype::F16);
        Tensor v({1, 3, 1, 32}, Dtype::F16); Tensor o({1, 2, 2, 31}, Dtype::F16);
        flash_gqa_attention_forward(o, q, k, v, nullptr, options);
    });
    expect_invalid_argument([&] {
        Tensor q({1, 2, 2, 32}, Dtype::F32); Tensor k({1, 3, 1, 32}, Dtype::F16);
        Tensor v({1, 3, 1, 32}, Dtype::F16); Tensor o({1, 2, 2, 32}, Dtype::F16);
        flash_gqa_attention_forward(o, q, k, v, nullptr, options);
    });
    options.query_position_offset = 2;
    expect_invalid_argument([&] { call(options); });
}

void test_custom_stream() {
    cudaStream_t stream = nullptr;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    try {
        auto options = valid_options();
        options.causal = false;
        run_case(1, 2, 3, options, Dtype::F16, 3.0e-2F, stream);
    } catch (...) {
        cudaStreamDestroy(stream);
        throw;
    }
    CUDA_CHECK(cudaStreamDestroy(stream));
}

void run_backward_case(
    std::size_t batch, std::size_t query_sequence, std::size_t key_sequence,
    const FlashAttentionOptions& options
) {
    const std::vector<std::size_t> query_shape{
        batch, query_sequence, options.num_query_heads, options.head_dim};
    const std::vector<std::size_t> key_shape{
        batch, key_sequence, options.num_kv_heads, options.head_dim};
    const auto query_values = values(
        batch * query_sequence * options.num_query_heads * options.head_dim, 0.2F);
    const auto key_values = values(
        batch * key_sequence * options.num_kv_heads * options.head_dim, 0.7F);
    const auto value_values = values(key_values.size(), 1.4F);
    const auto grad_output_values = values(query_values.size(), 2.1F);
    const auto expected = attention_backward_reference(
        grad_output_values, query_values, key_values, value_values,
        batch, query_sequence, key_sequence, options);

    Tensor query(query_shape, Dtype::F16), key(key_shape, Dtype::F16);
    Tensor value(key_shape, Dtype::F16), grad_output(query_shape, Dtype::F16);
    Tensor grad_query(query_shape, Dtype::F16), grad_key(key_shape, Dtype::F16);
    Tensor grad_value(key_shape, Dtype::F16);
    query.copy_from_host(query_values);
    key.copy_from_host(key_values);
    value.copy_from_host(value_values);
    grad_output.copy_from_host(grad_output_values);
    flash_gqa_attention_backward(
        grad_query, grad_key, grad_value, grad_output, query, key, value,
        nullptr, options);
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<float> actual_query(grad_query.numel());
    std::vector<float> actual_key(grad_key.numel());
    std::vector<float> actual_value(grad_value.numel());
    grad_query.copy_to_host(actual_query);
    grad_key.copy_to_host(actual_key);
    grad_value.copy_to_host(actual_value);
    expect_close(actual_query, expected.query, 3.0e-2F);
    expect_close(actual_key, expected.key, 3.0e-2F);
    expect_close(actual_value, expected.value, 3.0e-2F);

    // The training API saves LSE in forward and consumes it in backward.
    // Compare this path with the same FP32 reference as the legacy backward.
    Tensor output(query_shape, Dtype::F16);
    Tensor logsumexp(
        {batch, query_sequence, options.num_query_heads}, Dtype::F32);
    flash_gqa_attention_forward_with_lse(
        output, logsumexp, query, key, value, nullptr, options);
    flash_gqa_attention_backward_with_lse(
        grad_query, grad_key, grad_value, grad_output, output, logsumexp,
        query, key, value, nullptr, options);
    CUDA_CHECK(cudaDeviceSynchronize());
    grad_query.copy_to_host(actual_query);
    grad_key.copy_to_host(actual_key);
    grad_value.copy_to_host(actual_value);
    expect_close(actual_query, expected.query, 3.0e-2F);
    expect_close(actual_key, expected.key, 3.0e-2F);
    expect_close(actual_value, expected.value, 3.0e-2F);
}

void test_backward() {
    auto options = valid_options();
    options.num_query_heads = 4;
    options.num_kv_heads = 2;
    options.head_dim = 32;
    options.causal = true;
    options.query_position_offset = 2;
    options.attention_scale = 0.23F;
    run_backward_case(2, 3, 5, options);

    // Exercises the CTA-cooperative 4:1 GQA/head_dim=128 specialization.
    options.num_query_heads = 4;
    options.num_kv_heads = 1;
    options.head_dim = 128;
    options.causal = true;
    options.query_position_offset = 32;
    options.attention_scale = 0.11F;
    run_backward_case(1, 3, 35, options);
}

void benchmark_kernel() {
    FlashAttentionOptions options;
    options.num_query_heads = 32;
    options.num_kv_heads = 8;
    options.head_dim = 128;
    options.use_tensor_cores = true;
    options.causal = false;

    constexpr std::size_t kBatchSize = 1;
    constexpr std::size_t kQuerySequence = 512;
    constexpr std::size_t kKeyValueSequence = 512;
    const std::vector<std::size_t> query_shape{
        kBatchSize, kQuerySequence, options.num_query_heads, options.head_dim};
    const std::vector<std::size_t> key_value_shape{
        kBatchSize, kKeyValueSequence, options.num_kv_heads, options.head_dim};

    Tensor query(query_shape, Dtype::F16);
    Tensor key(key_value_shape, Dtype::F16);
    Tensor value(key_value_shape, Dtype::F16);
    Tensor output(query_shape, Dtype::F16);

    test::benchmark_cuda_launches("Flash attention kernel", [&](cudaStream_t stream) {
        flash_gqa_attention_forward(output, query, key, value, stream, options);
    }, 1000, 50);

    Tensor logsumexp(
        {kBatchSize, kQuerySequence, options.num_query_heads}, Dtype::F32);
    test::benchmark_cuda_launches("Flash attention LSE forward kernel", [&](cudaStream_t stream) {
        flash_gqa_attention_forward_with_lse(
            output, logsumexp, query, key, value, stream, options);
    }, 1000, 50);

    Tensor grad_output(query_shape, Dtype::F16);
    Tensor grad_query(query_shape, Dtype::F16);
    Tensor grad_key(key_value_shape, Dtype::F16);
    Tensor grad_value(key_value_shape, Dtype::F16);
    test::benchmark_cuda_launches("Flash attention backward kernel", [&](cudaStream_t stream) {
        flash_gqa_attention_backward(
            grad_query, grad_key, grad_value, grad_output,
            query, key, value, stream, options);
    }, 20, 5);

    test::benchmark_cuda_launches("Flash attention LSE backward kernel", [&](cudaStream_t stream) {
        flash_gqa_attention_backward_with_lse(
            grad_query, grad_key, grad_value, grad_output, output, logsumexp,
            query, key, value, stream, options);
    }, 20, 5);
}

} // namespace

int main() {
    try {
        // GQA, multiple batches and a final partial key tile.
        auto options = valid_options();
        options.num_query_heads = 4;
        options.num_kv_heads = 2;
        options.head_dim = 32;
        options.block_size = 32;
        options.key_tile_size = 2;
        options.causal = false;
        options.attention_scale = 0.37F;
        options.use_tensor_cores = true;
        run_case(2, 3, 5, options, Dtype::F16, 3.0e-2F);

        // Causal decoding against a K/V cache uses the position offset.
        options.causal = true;
        options.query_position_offset = 3;
        options.attention_scale = 0.0F;
        run_case(1, 2, 5, options, Dtype::F16, 3.0e-2F);

        // Single-token decoding at the end of a multi-tile K/V cache.  This
        // specifically verifies that the causal position offset exposes every
        // cache entry, including entries in the final (partial) 16-token tile.
        options.num_query_heads = 4;
        options.num_kv_heads = 2;
        options.head_dim = 32;
        options.block_size = 64;
        options.key_tile_size = 16;
        options.causal = true;
        options.query_position_offset = 16;
        options.attention_scale = 0.19F;
        options.use_tensor_cores = true;
        run_case(1, 1, 17, options, Dtype::F16, 3.0e-2F);

        // Cover each Tensor Core kernel specialization.
        options.num_query_heads = 2;
        options.num_kv_heads = 1;
        options.head_dim = 64;
        options.block_size = 32;
        options.key_tile_size = 16;
        options.causal = false;
        options.query_position_offset = 0;
        options.use_tensor_cores = true;
        run_case(1, 2, 16, options, Dtype::F16, 3.0e-2F);

        options.head_dim = 128;
        options.key_tile_size = 3;
        options.use_tensor_cores = true;
        run_case(1, 3, 5, options, Dtype::F16, 3.0e-2F);

        test_custom_stream();
        test_backward();
        test_validation();
        benchmark_kernel();
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return EXIT_FAILURE;
    }
    std::cout << "Flash attention tests passed.\n";
    return EXIT_SUCCESS;
}
