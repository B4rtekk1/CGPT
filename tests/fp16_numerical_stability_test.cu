#include "core/cuda_check.h"
#include "ops/attention.h"
#include "ops/backward/attention_backward.h"
#include "ops/backward/embedding_backward.h"
#include "ops/backward/linear_backward.h"
#include "ops/backward/rmsnorm_backward.h"
#include "ops/backward/swiglu_backward.h"
#include "ops/backward/rope_backward.h"
#include "ops/cross_entropy.h"
#include "ops/embedding.h"
#include "ops/linear.h"
#include "ops/rmsnorm.h"
#include "ops/rope.h"
#include "ops/softmax.h"
#include "ops/swiglu.h"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void expect_finite(const Tensor& tensor, const char* name) {
    std::vector<float> values(tensor.numel());
    tensor.copy_to_host(values);
    for (const float value : values) {
        if (!std::isfinite(value))
            throw std::runtime_error(std::string("FP16 numerical stability failure: ") + name);
    }
}

void fill_values(Tensor& tensor, const float scale = 64.0F) {
    std::vector<float> values(tensor.numel());
    for (std::size_t i = 0; i < values.size(); ++i)
        values[i] = (i % 3 == 0 ? scale : (i % 3 == 1 ? -scale : 1.0e-3F));
    tensor.copy_from_host(values);
}

void test_elementwise_and_normalization() {
    Tensor input({2, 8}, Dtype::F16), output({2, 8}, Dtype::F16);
    Tensor weight({8}, Dtype::F16), grad_output({2, 8}, Dtype::F16);
    Tensor grad_input({2, 8}, Dtype::F16), grad_weight({8}, Dtype::F16);
    fill_values(input); fill_values(weight, 1.0F); fill_values(grad_output, 2.0F);

    rmsnorm_forward(output, input, weight, 1.0e-5F);
    rmsnorm_backward(grad_input, grad_weight, grad_output, input, weight, 1.0e-5F);
    expect_finite(output, "RMSNorm forward");
    expect_finite(grad_input, "RMSNorm backward input");
    expect_finite(grad_weight, "RMSNorm backward weight");

    Tensor gate({2, 8}, Dtype::F16), up({2, 8}, Dtype::F16), activated({2, 8}, Dtype::F16);
    Tensor grad_gate({2, 8}, Dtype::F16), grad_up({2, 8}, Dtype::F16);
    fill_values(gate); fill_values(up); fill_values(grad_output, 1.0F);
    swiglu_forward(activated, gate, up);
    swiglu_backward(grad_gate, grad_up, grad_output, gate, up);
    expect_finite(activated, "SwiGLU forward");
    expect_finite(grad_gate, "SwiGLU backward gate");
    expect_finite(grad_up, "SwiGLU backward up");
}

void test_softmax_and_cross_entropy() {
    Tensor logits({2, 8}, Dtype::F16), probabilities({2, 8}, Dtype::F16);
    Tensor gradient({2, 8}, Dtype::F16), loss({1});
    fill_values(logits, 64.0F);
    bpe::TokenId* targets = nullptr;
    CUDA_CHECK(cudaMalloc(&targets, 2 * sizeof(*targets)));
    const bpe::TokenId host_targets[] = {0, 3};
    CUDA_CHECK(cudaMemcpy(targets, host_targets, sizeof(host_targets), cudaMemcpyHostToDevice));

    softmax_forward(probabilities, logits);
    cross_entropy_forward_backward(loss, gradient, logits, targets, 2);
    CUDA_CHECK(cudaDeviceSynchronize());
    expect_finite(probabilities, "Softmax");
    expect_finite(loss, "Cross-entropy loss");
    expect_finite(gradient, "Cross-entropy gradient");
    cudaFree(targets);
}

void test_linear_and_attention() {
    const CublasLtContext context;
    Tensor input({2, 4}, Dtype::F16), weight({4, 4}, Dtype::F16), output({2, 4}, Dtype::F16);
    Tensor bias({4}, Dtype::F16), grad_output({2, 4}, Dtype::F16);
    Tensor grad_input({2, 4}, Dtype::F16), grad_weight({4, 4}, Dtype::F16), grad_bias({4}, Dtype::F16);
    fill_values(input, 32.0F); fill_values(weight, 32.0F); fill_values(bias, 1.0F); fill_values(grad_output, 1.0F);
    linear_forward(output, input, weight, bias, context);
    linear_backward(grad_input, grad_weight, grad_bias, grad_output, input, weight, context);
    expect_finite(output, "Linear forward");
    expect_finite(grad_input, "Linear backward input");
    expect_finite(grad_weight, "Linear backward weight");
    expect_finite(grad_bias, "Linear backward bias");

    Tensor query({1, 2, 2, 32}, Dtype::F16), key({1, 3, 1, 32}, Dtype::F16);
    Tensor value({1, 3, 1, 32}, Dtype::F16), attention_output({1, 2, 2, 32}, Dtype::F16);
    Tensor grad_attention_output({1, 2, 2, 32}, Dtype::F16);
    Tensor grad_query({1, 2, 2, 32}, Dtype::F16), grad_key({1, 3, 1, 32}, Dtype::F16), grad_value({1, 3, 1, 32}, Dtype::F16);
    fill_values(query, 16.0F); fill_values(key, 16.0F); fill_values(value, 16.0F); fill_values(grad_attention_output, 1.0F);
    FlashAttentionOptions options{2, 1, 32, 1.0F / std::sqrt(32.0F), false, 0, 2, 128, true};
    flash_gqa_attention_forward(attention_output, query, key, value, nullptr, options);
    flash_gqa_attention_backward(grad_query, grad_key, grad_value, grad_attention_output, query, key, value, nullptr, options);
    CUDA_CHECK(cudaDeviceSynchronize());
    expect_finite(attention_output, "FlashAttention forward");
    expect_finite(grad_query, "FlashAttention backward query");
    expect_finite(grad_key, "FlashAttention backward key");
    expect_finite(grad_value, "FlashAttention backward value");
}

void test_rope_and_embedding() {
    Tensor query({1, 2, 1, 8}, Dtype::F16), key({1, 2, 1, 8}, Dtype::F16);
    Tensor cosine({2, 4}, Dtype::F16), sine({2, 4}, Dtype::F16);
    Tensor grad_rotated_query({1, 2, 1, 8}, Dtype::F16), grad_rotated_key({1, 2, 1, 8}, Dtype::F16);
    Tensor grad_query({1, 2, 1, 8}, Dtype::F16), grad_key({1, 2, 1, 8}, Dtype::F16);
    fill_values(query); fill_values(key); fill_values(grad_rotated_query, 1.0F); fill_values(grad_rotated_key, 1.0F);
    cosine.copy_from_host(std::vector<float>(cosine.numel(), 0.7071F));
    sine.copy_from_host(std::vector<float>(sine.numel(), 0.7071F));
    rope_forward(query, key, cosine, sine, nullptr);
    rope_backward(grad_query, grad_key, grad_rotated_query, grad_rotated_key, cosine, sine);
    expect_finite(query, "RoPE forward query"); expect_finite(key, "RoPE forward key");
    expect_finite(grad_query, "RoPE backward query"); expect_finite(grad_key, "RoPE backward key");

    Tensor embedding_weight({4, 8}, Dtype::F16), embedding_output({3, 8}, Dtype::F16), embedding_gradient({4, 8}, Dtype::F16);
    Tensor embedding_grad_output({3, 8}, Dtype::F16);
    fill_values(embedding_weight); fill_values(embedding_grad_output, 1.0F);
    bpe::TokenId* ids = nullptr;
    CUDA_CHECK(cudaMalloc(&ids, 3 * sizeof(*ids)));
    const bpe::TokenId host_ids[] = {0, 1, 3};
    CUDA_CHECK(cudaMemcpy(ids, host_ids, sizeof(host_ids), cudaMemcpyHostToDevice));
    embedding_forward(embedding_output, ids, 3, embedding_weight);
    embedding_backward(embedding_gradient, embedding_grad_output, ids, 3);
    CUDA_CHECK(cudaDeviceSynchronize());
    expect_finite(embedding_output, "Embedding forward"); expect_finite(embedding_gradient, "Embedding backward");
    cudaFree(ids);
}

} // namespace

int main() {
    try {
        test_elementwise_and_normalization();
        test_softmax_and_cross_entropy();
        test_linear_and_attention();
        test_rope_and_embedding();
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return EXIT_FAILURE;
    }
    std::cout << "FP16 numerical stability tests passed.\n";
    return EXIT_SUCCESS;
}
