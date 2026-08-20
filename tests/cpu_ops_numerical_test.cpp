#include "ops/cpu/linear_cpu.h"
#include "ops/cpu/softmax_cpu.h"
#include "ops/cpu/rmsnorm_cpu.h"
#include "ops/cpu/swiglu_cpu.h"
#include "ops/cpu/embedding_cpu.h"
#include "ops/cpu/cross_entropy_cpu.h"
#include "ops/cpu/rope_cpu.h"
#include "ops/cpu/attention_cpu.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void expect_close(const std::vector<float>& actual, const std::vector<float>& expected,
                 float tolerance, const std::string& name) {
    if (actual.size() != expected.size())
        throw std::runtime_error(name + ": size mismatch");
    for (std::size_t i = 0; i < actual.size(); ++i) {
        const float error = std::fabs(actual[i] - expected[i]);
        const float scale = std::max(1.0f, std::fabs(expected[i]));
        if (!std::isfinite(actual[i]) || error > tolerance * scale)
            throw std::runtime_error(name + ": mismatch at index " + std::to_string(i));
    }
}

std::vector<float> values(const Tensor& tensor) {
    std::vector<float> result(tensor.numel());
    tensor.copy_to_host(result);
    return result;
}

void test_linear() {
    Tensor input({2, 3, 5}, DeviceType::CPU);
    Tensor weight({7, 5}, DeviceType::CPU);
    Tensor bias({7}, DeviceType::CPU);
    Tensor output({2, 3, 7}, DeviceType::CPU);
    std::vector<float> x(input.numel()), w(weight.numel()), b(bias.numel());
    for (std::size_t i = 0; i < x.size(); ++i) x[i] = 0.13f * float(int(i % 11) - 5);
    for (std::size_t i = 0; i < w.size(); ++i) w[i] = 0.09f * float(int(i % 13) - 6);
    for (std::size_t i = 0; i < b.size(); ++i) b[i] = 0.17f * float(int(i) - 3);
    input.copy_from_host(x); weight.copy_from_host(w); bias.copy_from_host(b);
    linear_forward_cpu(output, input, weight, bias);
    std::vector<float> expected(output.numel());
    for (std::size_t row = 0; row < 6; ++row)
        for (std::size_t o = 0; o < 7; ++o) {
            float sum = b[o];
            for (std::size_t i = 0; i < 5; ++i) sum += x[row * 5 + i] * w[o * 5 + i];
            expected[row * 7 + o] = sum;
        }
    expect_close(values(output), expected, 2e-6f, "CPU linear");
}

void test_softmax() {
    constexpr std::size_t rows = 3, columns = 37;
    Tensor input({rows, columns}, DeviceType::CPU), output({rows, columns}, DeviceType::CPU);
    std::vector<float> x(rows * columns);
    for (std::size_t i = 0; i < x.size(); ++i) x[i] = 0.31f * float(int(i % 17) - 8);
    x[0] = 1000.0f; x[1] = 999.0f;
    input.copy_from_host(x); softmax_forward_cpu(output, input);
    std::vector<float> expected(x.size());
    for (std::size_t r = 0; r < rows; ++r) {
        const auto begin = x.begin() + r * columns;
        const float maximum = *std::max_element(begin, begin + columns);
        float sum = 0.0f;
        for (std::size_t c = 0; c < columns; ++c) sum += std::exp(x[r * columns + c] - maximum);
        for (std::size_t c = 0; c < columns; ++c) expected[r * columns + c] = std::exp(x[r * columns + c] - maximum) / sum;
    }
    expect_close(values(output), expected, 3e-5f, "CPU softmax");
}

void test_rmsnorm_and_swiglu() {
    constexpr float epsilon = 1e-5f;
    Tensor input({3, 13}, DeviceType::CPU), weight({13}, DeviceType::CPU), output({3, 13}, DeviceType::CPU);
    std::vector<float> x(input.numel()), w(weight.numel());
    for (std::size_t i = 0; i < x.size(); ++i) x[i] = 0.2f * float(int(i % 9) - 4);
    for (std::size_t i = 0; i < w.size(); ++i) w[i] = 0.5f + 0.03f * float(i);
    input.copy_from_host(x); weight.copy_from_host(w); rmsnorm_forward_cpu(output, input, weight, epsilon);
    std::vector<float> expected(x.size());
    for (std::size_t r = 0; r < 3; ++r) {
        float sum = 0.0f; for (std::size_t c = 0; c < 13; ++c) sum += x[r * 13 + c] * x[r * 13 + c];
        const float inv = 1.0f / std::sqrt(sum / 13.0f + epsilon);
        for (std::size_t c = 0; c < 13; ++c) expected[r * 13 + c] = x[r * 13 + c] * w[c] * inv;
    }
    expect_close(values(output), expected, 3e-5f, "CPU RMSNorm");

    Tensor gate({2, 13}, DeviceType::CPU), up({2, 13}, DeviceType::CPU), swiglu({2, 13}, DeviceType::CPU);
    std::vector<float> g(gate.numel()), u(up.numel());
    for (std::size_t i = 0; i < g.size(); ++i) { g[i] = 0.17f * float(int(i % 15) - 7); u[i] = 0.11f * float(int(i % 9) - 4); }
    gate.copy_from_host(g); up.copy_from_host(u); swiglu_forward_cpu(swiglu, gate, up);
    for (std::size_t i = 0; i < g.size(); ++i) expected[i] = g[i] / (1.0f + std::exp(-g[i])) * u[i];
    std::vector<float> swiglu_expected(expected.begin(), expected.begin() + g.size());
    expect_close(values(swiglu), swiglu_expected, 3e-5f, "CPU SwiGLU");
}

void test_embedding_and_cross_entropy() {
    Tensor table({5, 11}, DeviceType::CPU), output({4, 11}, DeviceType::CPU);
    std::vector<float> w(table.numel());
    for (std::size_t i = 0; i < w.size(); ++i) w[i] = 0.07f * float(int(i % 19) - 9);
    table.copy_from_host(w);
    const bpe::TokenId ids[] = {4, 1, 3, 0};
    embedding_forward_cpu(output, ids, 4, table);
    std::vector<float> expected(44);
    for (std::size_t r = 0; r < 4; ++r) std::copy_n(w.data() + std::size_t(ids[r]) * 11, 11, expected.data() + r * 11);
    expect_close(values(output), expected, 0.0f, "CPU embedding");

    Tensor logits({3, 5}, DeviceType::CPU), loss({1}, DeviceType::CPU);
    logits.copy_from_host(std::vector<float>{2.0f, 1.0f, 0.0f, -1.0f, 3.0f, -2.0f, 0.5f, 4.0f, 1.0f, -0.5f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f});
    const bpe::TokenId targets[] = {4, 1, 0};
    cross_entropy_forward_cpu(loss, logits, targets, 3);
    const auto l = values(logits); float expected_loss = 0.0f;
    for (std::size_t r = 0; r < 3; ++r) { float sum = 0.0f; for (std::size_t c = 0; c < 5; ++c) sum += std::exp(l[r * 5 + c]); expected_loss += std::log(sum) - l[r * 5 + targets[r]]; }
    expect_close(values(loss), std::vector<float>{expected_loss / 3.0f}, 2e-6f, "CPU cross entropy");
}

void test_rope() {
    Tensor query({1, 2, 2, 4}, DeviceType::CPU), key({1, 2, 1, 4}, DeviceType::CPU);
    Tensor cosine({2, 2}, DeviceType::CPU), sine({2, 2}, DeviceType::CPU);
    query.copy_from_host(std::vector<float>(16, 1.0f)); key.copy_from_host(std::vector<float>(8, 2.0f));
    cosine.copy_from_host(std::vector<float>{0.0f, 1.0f, 1.0f, 0.0f}); sine.copy_from_host(std::vector<float>{1.0f, 0.0f, 0.0f, 1.0f});
    rope_forward_cpu(query, key, cosine, sine);
    expect_close(values(query), std::vector<float>{ -1, 1, 1, 1, -1, 1, 1, 1, 1, 1, -1, 1, 1, 1, -1, 1 }, 2e-6f, "CPU RoPE query");
    expect_close(values(key), std::vector<float>{ -2, 2, 2, 2, 2, 2, -2, 2 }, 2e-6f, "CPU RoPE key");
}

void test_attention() {
    constexpr std::size_t query_tokens = 3, key_tokens = 4, heads = 2, dimension = 3;
    Tensor query({1, query_tokens, heads, dimension}, DeviceType::CPU);
    Tensor key({1, key_tokens, 1, dimension}, DeviceType::CPU);
    Tensor value({1, key_tokens, 1, dimension}, DeviceType::CPU);
    Tensor output(query.shape(), DeviceType::CPU), lse({1, query_tokens, heads}, DeviceType::CPU);
    std::vector<float> q(query.numel()), k(key.numel()), v(value.numel());
    for (std::size_t i = 0; i < q.size(); ++i) q[i] = 0.08f * float(int(i % 9) - 4);
    for (std::size_t i = 0; i < k.size(); ++i) k[i] = 0.11f * float(int(i % 7) - 3);
    for (std::size_t i = 0; i < v.size(); ++i) v[i] = 0.13f * float(int(i % 11) - 5);
    query.copy_from_host(q); key.copy_from_host(k); value.copy_from_host(v);
    FlashAttentionOptions options{}; options.num_query_heads = heads; options.num_kv_heads = 1;
    options.head_dim = dimension; options.attention_scale = 0.7f; options.causal = true;
    flash_gqa_attention_forward_with_lse_cpu(output, lse, query, key, value, options);
    std::vector<float> expected(output.numel());
    for (std::size_t qi = 0; qi < query_tokens; ++qi) for (std::size_t h = 0; h < heads; ++h) {
        const std::size_t qb = (qi * heads + h) * dimension, last = qi + 1;
        float maximum = -INFINITY;
        for (std::size_t kj = 0; kj < last; ++kj) { float dot = 0; for (std::size_t d = 0; d < dimension; ++d) dot += q[qb + d] * k[kj * dimension + d]; maximum = std::max(maximum, dot * options.attention_scale); }
        float denominator = 0;
        for (std::size_t kj = 0; kj < last; ++kj) { float dot = 0; for (std::size_t d = 0; d < dimension; ++d) dot += q[qb + d] * k[kj * dimension + d]; denominator += std::exp(dot * options.attention_scale - maximum); }
        for (std::size_t d = 0; d < dimension; ++d) { float sum = 0; for (std::size_t kj = 0; kj < last; ++kj) { float dot = 0; for (std::size_t z = 0; z < dimension; ++z) dot += q[qb + z] * k[kj * dimension + z]; sum += std::exp(dot * options.attention_scale - maximum) * v[kj * dimension + d]; } expected[qb + d] = sum / denominator; }
        if (std::fabs(values(lse)[qi * heads + h] - (std::log(denominator) + maximum)) > 3e-6f) throw std::runtime_error("CPU attention: LSE mismatch");
    }
    expect_close(values(output), expected, 3e-6f, "CPU attention");
}

} // namespace

int main() {
    try { test_linear(); test_softmax(); test_rmsnorm_and_swiglu(); test_embedding_and_cross_entropy(); test_rope(); test_attention(); }
    catch (const std::exception& error) { std::cerr << error.what() << '\n'; return EXIT_FAILURE; }
    std::cout << "CPU numerical operation tests passed.\n";
    return EXIT_SUCCESS;
}
