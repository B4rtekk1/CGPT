#include "core/cuda_check.h"
#include "ops/cut_cross_entropy.h"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {
void expect_close(const Tensor& tensor, const std::vector<float>& expected, float tolerance = 2e-5f) {
    std::vector<float> actual(tensor.numel()); tensor.copy_to_host(actual);
    for (std::size_t i = 0; i < actual.size(); ++i)
        if (std::fabs(actual[i] - expected[i]) > tolerance) {
            std::cerr << "CCE mismatch at " << i << ": " << actual[i] << " expected " << expected[i] << '\n';
            throw std::runtime_error("CCE: unexpected result");
        }
}
void test_cce() {
    Tensor input({2, 2}), classifier({3, 2}), loss({1}), grad_input({2, 2}), grad_classifier({3, 2});
    input.copy_from_host(std::vector<float>{1, 2, 3, 4});
    classifier.copy_from_host(std::vector<float>{1, 0, 0, 1, 1, 1});
    bpe::TokenId* targets; CUDA_CHECK(cudaMalloc(&targets, 2 * sizeof(*targets)));
    const std::vector<bpe::TokenId> host_targets{2, 0}; CUDA_CHECK(cudaMemcpy(targets, host_targets.data(), 2 * sizeof(*targets), cudaMemcpyHostToDevice));
    cut_cross_entropy_forward_backward(loss, grad_input, grad_classifier, input, classifier, targets, 2);
    CUDA_CHECK(cudaDeviceSynchronize()); cudaFree(targets);
    const float e1 = std::exp(1.0f), e2 = std::exp(2.0f), e3 = std::exp(3.0f), e4 = std::exp(4.0f), e7 = std::exp(7.0f);
    expect_close(loss, {(std::log(e1 + e2 + e3) - 3.0f + std::log(e3 + e4 + e7) - 3.0f) / 2});
    expect_close(grad_input, {-0.1223642f, -0.0450153f, -0.0234640f, 0.4915309f});
    expect_close(grad_classifier, {-1.429500f, -1.876000f, 0.193000f, 0.339000f, 1.736500f, 2.537000f}, 2e-3f);
}
}
int main() { try { test_cce(); } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return EXIT_FAILURE; } std::cout << "Cut Cross-Entropy tests passed.\n"; }
