#include "core/cuda_check.h"
#include "ops/backward/linear_backward.h"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

void expect_close(const Tensor& tensor, const std::vector<float>& expected, const float tolerance = 1e-5f) {
    std::vector<float> actual(tensor.numel());
    tensor.copy_to_host(actual);
    for (std::size_t i = 0; i < actual.size(); ++i)
        if (!std::isfinite(actual[i]) || std::fabs(actual[i] - expected[i]) > tolerance)
            throw std::runtime_error("linear backward: incorrect result");
}

void test_backward_and_accumulation() {
    Tensor input({2, 3}), weight({2, 3}), grad_output({2, 2});
    Tensor grad_input({2, 3}), grad_weight({2, 3}), grad_bias({2});
    input.copy_from_host(std::vector<float>{1, 2, 3, 4, 5, 6});
    weight.copy_from_host(std::vector<float>{1, 0, 2, -1, 3, 1});
    grad_output.copy_from_host(std::vector<float>{2, -1, 3, 4});
    const CublasLtContext context;
    linear_backward(grad_input, grad_weight, grad_bias, grad_output, input, weight, context);
    CUDA_CHECK(cudaDeviceSynchronize());
    expect_close(grad_input, {3, -3, 3, -1, 12, 10});
    expect_close(grad_weight, {14, 19, 24, 15, 18, 21});
    expect_close(grad_bias, {5, 3});
    LinearBackwardOptions options;
    options.accumulate_input = options.accumulate_weight = options.accumulate_bias = true;
    linear_backward(grad_input, grad_weight, grad_bias, grad_output, input, weight, context, nullptr, options);
    CUDA_CHECK(cudaDeviceSynchronize());
    expect_close(grad_input, {6, -6, 6, -2, 24, 20});
    expect_close(grad_weight, {28, 38, 48, 30, 36, 42});
    expect_close(grad_bias, {10, 6});
}

void test_batched_f16() {
    Tensor input({2, 2, 2}, Dtype::F16), weight({3, 2}, Dtype::F16), grad_output({2, 2, 3}, Dtype::F16);
    Tensor grad_input({2, 2, 2}, Dtype::F16), grad_weight({3, 2}, Dtype::F16), grad_bias({3}, Dtype::F16);
    input.copy_from_host(std::vector<float>{1, 2, 3, 4, 5, 6, 7, 8});
    weight.copy_from_host(std::vector<float>{1, 2, 3, 4, 5, 6});
    grad_output.copy_from_host(std::vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
    const CublasLtContext context;
    linear_backward(grad_input, grad_weight, grad_bias, grad_output, input, weight, context);
    CUDA_CHECK(cudaDeviceSynchronize());
    expect_close(grad_input, {22, 28, 49, 64, 76, 100, 103, 136}, 0.1f);
    expect_close(grad_weight, {118, 140, 134, 160, 150, 180}, 0.2f);
    expect_close(grad_bias, {22, 26, 30}, 0.1f);
}

} // namespace

int main() {
    try { test_backward_and_accumulation(); test_batched_f16(); }
    catch (const std::exception& error) { std::cerr << error.what() << '\n'; return EXIT_FAILURE; }
    std::cout << "Linear backward tests passed.\n";
    return EXIT_SUCCESS;
}
