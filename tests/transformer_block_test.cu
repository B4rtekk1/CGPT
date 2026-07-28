#include "core/cuda_check.h"
#include "core/transformer_block.h"
#include "cuda_benchmark.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr std::size_t kBatch = 1;
constexpr std::size_t kSequence = 2;
constexpr std::size_t kHidden = 64;
constexpr std::size_t kIntermediate = 64;
constexpr std::size_t kQueryHeads = 2;
constexpr std::size_t kKvHeads = 1;
constexpr std::size_t kHeadDim = 32;
constexpr float kEpsilon = 1.0e-5F;

void expect_close(const std::vector<float>& actual, const std::vector<float>& expected,
                  float tolerance) {
    if (actual.size() != expected.size()) {
        throw std::runtime_error("Transformer block test: result size mismatch");
    }
    for (std::size_t i = 0; i < actual.size(); ++i) {
        if (!std::isfinite(actual[i]) || std::fabs(actual[i] - expected[i]) > tolerance) {
            throw std::runtime_error("Transformer block test: mismatch at index " +
                                     std::to_string(i));
        }
    }
}

std::vector<float> make_values(std::size_t count, float phase) {
    std::vector<float> result(count);
    for (std::size_t i = 0; i < count; ++i) {
        result[i] = 0.12F * std::sin(phase + 0.13F * static_cast<float>(i));
    }
    return result;
}

std::vector<float> rmsnorm(const std::vector<float>& input, const std::vector<float>& weight,
                           std::size_t rows, std::size_t columns) {
    std::vector<float> output(input.size());
    for (std::size_t row = 0; row < rows; ++row) {
        float sum = 0.0F;
        for (std::size_t column = 0; column < columns; ++column) {
            const float value = input[row * columns + column];
            sum += value * value;
        }
        const float scale = 1.0F / std::sqrt(sum / static_cast<float>(columns) + kEpsilon);
        for (std::size_t column = 0; column < columns; ++column) {
            const std::size_t index = row * columns + column;
            output[index] = input[index] * scale * weight[column];
        }
    }
    return output;
}

std::vector<float> linear(const std::vector<float>& input, const std::vector<float>& weight,
                          std::size_t rows, std::size_t input_features,
                          std::size_t output_features) {
    std::vector<float> output(rows * output_features);
    for (std::size_t row = 0; row < rows; ++row) {
        for (std::size_t output_feature = 0; output_feature < output_features; ++output_feature) {
            float sum = 0.0F;
            for (std::size_t input_feature = 0; input_feature < input_features; ++input_feature) {
                sum += input[row * input_features + input_feature] *
                    weight[output_feature * input_features + input_feature];
            }
            output[row * output_features + output_feature] = sum;
        }
    }
    return output;
}

std::vector<float> reference_forward(const std::vector<float>& input,
                                     const std::vector<float>& attention_norm,
                                     const std::vector<float>& q_weight,
                                     const std::vector<float>& k_weight,
                                     const std::vector<float>& v_weight,
                                     const std::vector<float>& o_weight,
                                     const std::vector<float>& ffn_norm,
                                     const std::vector<float>& gate_weight,
                                     const std::vector<float>& up_weight,
                                     const std::vector<float>& down_weight) {
    constexpr std::size_t rows = kBatch * kSequence;
    const auto norm = rmsnorm(input, attention_norm, rows, kHidden);
    const auto query = linear(norm, q_weight, rows, kHidden, kHidden);
    const auto key = linear(norm, k_weight, rows, kHidden, kHeadDim);
    const auto value = linear(norm, v_weight, rows, kHidden, kHeadDim);

    std::vector<float> attention(rows * kHidden);
    const float scale = 1.0F / std::sqrt(static_cast<float>(kHeadDim));
    for (std::size_t position = 0; position < kSequence; ++position) {
        for (std::size_t head = 0; head < kQueryHeads; ++head) {
            std::vector<float> scores(position + 1U);
            float max_score = -INFINITY;
            for (std::size_t key_position = 0; key_position <= position; ++key_position) {
                float score = 0.0F;
                for (std::size_t dimension = 0; dimension < kHeadDim; ++dimension) {
                    score += query[position * kHidden + head * kHeadDim + dimension] *
                        key[key_position * kHeadDim + dimension];
                }
                scores[key_position] = score * scale;
                max_score = std::max(max_score, scores[key_position]);
            }
            float denominator = 0.0F;
            for (float& score : scores) {
                score = std::exp(score - max_score);
                denominator += score;
            }
            for (std::size_t dimension = 0; dimension < kHeadDim; ++dimension) {
                float result = 0.0F;
                for (std::size_t key_position = 0; key_position <= position; ++key_position) {
                    result += scores[key_position] * value[key_position * kHeadDim + dimension];
                }
                attention[position * kHidden + head * kHeadDim + dimension] = result / denominator;
            }
        }
    }

    const auto attention_projection = linear(attention, o_weight, rows, kHidden, kHidden);
    std::vector<float> residual(rows * kHidden);
    for (std::size_t i = 0; i < residual.size(); ++i) residual[i] = input[i] + attention_projection[i];
    const auto ffn_input = rmsnorm(residual, ffn_norm, rows, kHidden);
    const auto gate = linear(ffn_input, gate_weight, rows, kHidden, kIntermediate);
    const auto up = linear(ffn_input, up_weight, rows, kHidden, kIntermediate);
    std::vector<float> activated(gate.size());
    for (std::size_t i = 0; i < activated.size(); ++i) {
        activated[i] = gate[i] / (1.0F + std::exp(-gate[i])) * up[i];
    }
    const auto ffn_output = linear(activated, down_weight, rows, kIntermediate, kHidden);
    for (std::size_t i = 0; i < residual.size(); ++i) residual[i] += ffn_output[i];
    return residual;
}

struct BlockFixture {
    TransformerBlockOptions options{kHidden, kIntermediate, kQueryHeads, kKvHeads, kHeadDim, kHeadDim,
                                    kEpsilon, true, {ComputeType::F32}};
    Tensor input{{kBatch * kSequence, kHidden}, Dtype::F16};
    Tensor output{{kBatch * kSequence, kHidden}, Dtype::F16};
    Tensor attention_norm{{kHidden}, Dtype::F16};
    Tensor q{{kHidden, kHidden}, Dtype::F16};
    Tensor k{{kHeadDim, kHidden}, Dtype::F16};
    Tensor v{{kHeadDim, kHidden}, Dtype::F16};
    Tensor o{{kHidden, kHidden}, Dtype::F16};
    Tensor ffn_norm{{kHidden}, Dtype::F16};
    Tensor gate{{kIntermediate, kHidden}, Dtype::F16};
    Tensor up{{kIntermediate, kHidden}, Dtype::F16};
    Tensor down{{kHidden, kIntermediate}, Dtype::F16};
    Tensor cos_cache{{kSequence, kHeadDim / 2}, Dtype::F16};
    Tensor sin_cache{{kSequence, kHeadDim / 2}, Dtype::F16};
    TransformerBlockWorkspace workspace{
        Tensor{{kBatch * kSequence, kHidden}, Dtype::F16},
        Tensor{{kBatch, kSequence, kQueryHeads, kHeadDim}, Dtype::F16},
        Tensor{{kBatch, kSequence, kKvHeads, kHeadDim}, Dtype::F16},
        Tensor{{kBatch, kSequence, kKvHeads, kHeadDim}, Dtype::F16},
        Tensor{{kBatch, kSequence, kQueryHeads, kHeadDim}, Dtype::F16},
        Tensor{{kBatch * kSequence, kHidden}, Dtype::F16},
        Tensor{{kBatch * kSequence, kIntermediate}, Dtype::F16},
        Tensor{{kBatch * kSequence, kIntermediate}, Dtype::F16},
        Tensor{{kBatch * kSequence, kIntermediate}, Dtype::F16},
        Tensor{{kBatch * kSequence, kHidden}, Dtype::F16}};

    BlockFixture() {
        input.copy_from_host(make_values(input.numel(), 0.1F));
        attention_norm.copy_from_host(make_values(kHidden, 1.0F));
        ffn_norm.copy_from_host(make_values(kHidden, 1.4F));
        q.copy_from_host(make_values(q.numel(), 0.2F)); k.copy_from_host(make_values(k.numel(), 0.3F));
        v.copy_from_host(make_values(v.numel(), 0.4F)); o.copy_from_host(make_values(o.numel(), 0.5F));
        gate.copy_from_host(make_values(gate.numel(), 0.6F)); up.copy_from_host(make_values(up.numel(), 0.7F));
        down.copy_from_host(make_values(down.numel(), 0.8F));
        cos_cache.copy_from_host(std::vector<float>(cos_cache.numel(), 1.0F));
        sin_cache.copy_from_host(std::vector<float>(sin_cache.numel(), 0.0F));
    }

    TransformerBlockWeights weights() const {
        return {attention_norm, q, k, v, o, ffn_norm, gate, up, down};
    }
};

void test_forward() {
    BlockFixture fixture;
    const auto expected = reference_forward(
        make_values(fixture.input.numel(), 0.1F), make_values(kHidden, 1.0F),
        make_values(fixture.q.numel(), 0.2F), make_values(fixture.k.numel(), 0.3F),
        make_values(fixture.v.numel(), 0.4F), make_values(fixture.o.numel(), 0.5F),
        make_values(kHidden, 1.4F), make_values(fixture.gate.numel(), 0.6F),
        make_values(fixture.up.numel(), 0.7F), make_values(fixture.down.numel(), 0.8F));
    const CublasLtContext context;
    transformer_block_forward(fixture.output, fixture.input, fixture.weights(), fixture.workspace,
                              fixture.cos_cache, fixture.sin_cache, context, nullptr, fixture.options);
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<float> actual(fixture.output.numel());
    fixture.output.copy_to_host(actual);
    expect_close(actual, expected, 8.0e-2F);
    if (fixture.workspace.query.shape() != std::vector<std::size_t>{kBatch, kSequence, kQueryHeads, kHeadDim}) {
        throw std::runtime_error("Transformer block test: workspace shape was not restored");
    }
}

template <typename Function> void expect_invalid_argument(Function&& function) {
    try { function(); } catch (const std::invalid_argument&) { return; }
    throw std::runtime_error("Transformer block test: invalid input was accepted");
}

void test_validation() {
    BlockFixture fixture;
    const CublasLtContext context;
    auto options = fixture.options;
    options.hidden_size = 0;
    expect_invalid_argument([&] { transformer_block_forward(fixture.output, fixture.input, fixture.weights(), fixture.workspace, fixture.cos_cache, fixture.sin_cache, context, nullptr, options); });
    Tensor wrong_output({kBatch * kSequence, kHidden - 1}, Dtype::F16);
    expect_invalid_argument([&] { transformer_block_forward(wrong_output, fixture.input, fixture.weights(), fixture.workspace, fixture.cos_cache, fixture.sin_cache, context, nullptr, fixture.options); });
    fixture.workspace.gate.reshape({kBatch, kIntermediate * kSequence});
    try {
        transformer_block_forward(fixture.output, fixture.input, fixture.weights(), fixture.workspace,
                                  fixture.cos_cache, fixture.sin_cache, context, nullptr,
                                  fixture.options);
    } catch (const std::runtime_error&) {
        return;
    }
    throw std::runtime_error("Transformer block test: invalid workspace was accepted");
}

void benchmark_kernel() {
    constexpr std::size_t batch = 1, sequence = 16, hidden = 128, intermediate = 256;
    constexpr std::size_t query_heads = 2, kv_heads = 1, head_dim = 64;
    TransformerBlockOptions options{hidden, intermediate, query_heads, kv_heads, head_dim, head_dim, kEpsilon, true, {ComputeType::F32}};
    Tensor input({batch * sequence, hidden}, Dtype::F16), output({batch * sequence, hidden}, Dtype::F16);
    Tensor norm({hidden}, Dtype::F16), q({hidden, hidden}, Dtype::F16), k({kv_heads * head_dim, hidden}, Dtype::F16), v({kv_heads * head_dim, hidden}, Dtype::F16), o({hidden, hidden}, Dtype::F16), gate({intermediate, hidden}, Dtype::F16), up({intermediate, hidden}, Dtype::F16), down({hidden, intermediate}, Dtype::F16), cos({sequence, head_dim / 2}, Dtype::F16), sin({sequence, head_dim / 2}, Dtype::F16);
    TransformerBlockWeights weights{norm, q, k, v, o, norm, gate, up, down};
    TransformerBlockWorkspace workspace{Tensor({batch * sequence, hidden}, Dtype::F16), Tensor({batch, sequence, query_heads, head_dim}, Dtype::F16), Tensor({batch, sequence, kv_heads, head_dim}, Dtype::F16), Tensor({batch, sequence, kv_heads, head_dim}, Dtype::F16), Tensor({batch, sequence, query_heads, head_dim}, Dtype::F16), Tensor({batch * sequence, hidden}, Dtype::F16), Tensor({batch * sequence, intermediate}, Dtype::F16), Tensor({batch * sequence, intermediate}, Dtype::F16), Tensor({batch * sequence, intermediate}, Dtype::F16), Tensor({batch * sequence, hidden}, Dtype::F16)};
    const CublasLtContext context;
    test::benchmark_cuda_launches("Transformer block", [&](cudaStream_t stream) { transformer_block_forward(output, input, weights, workspace, cos, sin, context, stream, options); });
}

} // namespace

int main() {
    try {
        test_forward();
        test_validation();
        benchmark_kernel();
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return EXIT_FAILURE;
    }
    std::cout << "Transformer block tests passed.\n";
    return EXIT_SUCCESS;
}
