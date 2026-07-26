#include "core/cuda_check.h"
#include "ops/linear.h"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

void expect_close(
    const std::vector<float>& actual,
    const std::vector<float>& expected,
    float tolerance = 1.0e-5f
) {
    if (actual.size() != expected.size()) {
        throw std::runtime_error("Linear test: result size mismatch");
    }

    for (std::size_t index = 0; index < actual.size(); ++index) {
        if (!std::isfinite(actual[index]) ||
            std::fabs(actual[index] - expected[index]) > tolerance) {
            throw std::runtime_error("Linear test: result value mismatch");
        }
    }
}

std::vector<float> read_tensor(const Tensor& tensor) {
    std::vector<float> values(tensor.numel());
    tensor.copy_to_host(values);
    return values;
}

void test_matrix_multiplication() {
    Tensor input({2, 3});
    Tensor weight({2, 3});
    Tensor output({2, 2});

    input.copy_from_host(std::vector<float>{1.0f, 2.0f, 3.0f, -1.0f, 0.0f, 2.0f});
    weight.copy_from_host(std::vector<float>{1.0f, 0.0f, 2.0f, -1.0f, 3.0f, 1.0f});

    const CublasLtContext context;
    linear_forward(output, input, weight, context);
    CUDA_CHECK(cudaDeviceSynchronize());

    expect_close(read_tensor(output), {7.0f, 8.0f, 3.0f, 3.0f});
}

void test_batched_input() {
    Tensor input({2, 2, 3});
    Tensor weight({2, 3});
    Tensor output({2, 2, 2});

    input.copy_from_host(std::vector<float>{
        1.0f, 2.0f, 3.0f,
        4.0f, 5.0f, 6.0f,
        -1.0f, 0.0f, 1.0f,
        2.0f, -2.0f, 1.0f
    });
    weight.copy_from_host(std::vector<float>{1.0f, 2.0f, 0.0f, -1.0f, 0.0f, 1.0f});

    const CublasLtContext context;
    linear_forward(output, input, weight, context);
    CUDA_CHECK(cudaDeviceSynchronize());

    expect_close(read_tensor(output), {
        5.0f, 2.0f,
        14.0f, 2.0f,
        -1.0f, 2.0f,
        -2.0f, -1.0f
    });
}

void test_fused_bias_and_cached_plan() {
    Tensor input({2, 3});
    Tensor weight({2, 3});
    Tensor bias({2});
    Tensor output({2, 2});

    input.copy_from_host(std::vector<float>{
        1.0f, 2.0f, 3.0f,
        -1.0f, 0.0f, 2.0f});
    weight.copy_from_host(std::vector<float>{
        1.0f, 0.0f, 2.0f,
        -1.0f, 3.0f, 1.0f});
    bias.copy_from_host(std::vector<float>{0.5f, -1.0f});

    const CublasLtContext context;
    linear_forward(output, input, weight, bias, context);
    linear_forward(output, input, weight, bias, context);
    CUDA_CHECK(cudaDeviceSynchronize());

    expect_close(read_tensor(output), {7.5f, 7.0f, 3.5f, 2.0f});
}

void test_irregular_tf32_shape() {
    const Tensor input = Tensor::ones({9, 9});
    const Tensor weight = Tensor::ones({9, 9});
    const Tensor bias = Tensor::full({9}, 0.25f);
    Tensor output({9, 9});

    const CublasLtContext context;
    linear_forward(output, input, weight, bias, context);
    CUDA_CHECK(cudaDeviceSynchronize());

    expect_close(read_tensor(output), std::vector<float>(81, 9.25f));
}

void test_reduced_precision(const Dtype dtype, const float tolerance) {
    Tensor input({2, 4}, dtype);
    Tensor weight({3, 4}, dtype);
    Tensor bias({3}, dtype);
    Tensor output({2, 3}, dtype);

    input.copy_from_host(std::vector<float>{
        1.0f, 2.0f, -1.0f, 0.5f,
        -2.0f, 1.0f, 3.0f, -1.0f});
    weight.copy_from_host(std::vector<float>{
        1.0f, 0.0f, 2.0f, -1.0f,
        0.5f, 1.0f, 0.0f, 2.0f,
        -1.0f, 2.0f, 1.0f, 0.0f});
    bias.copy_from_host(std::vector<float>{0.25f, -0.5f, 1.0f});

    const CublasLtContext context;
    linear_forward(output, input, weight, bias, context);
    CUDA_CHECK(cudaDeviceSynchronize());

    expect_close(
        read_tensor(output),
        {-1.25f, 3.0f, 3.0f, 5.25f, -2.5f, 8.0f},
        tolerance);
}

template <typename Function>
void expect_invalid_argument(Function&& function) {
    try {
        function();
    } catch (const std::invalid_argument&) {
        return;
    }
    throw std::runtime_error("Linear test: invalid arguments were accepted");
}

void test_validation() {
    const CublasLtContext context;
    const Tensor valid_input({2, 3});
    Tensor valid_weight({2, 3});
    Tensor valid_output({2, 2});

    expect_invalid_argument([&] {
        Tensor input({3});
        linear_forward(valid_output, input, valid_weight, context);
    });
    expect_invalid_argument([&] {
        Tensor weight({3});
        linear_forward(valid_output, valid_input, weight, context);
    });
    expect_invalid_argument([&] {
        Tensor output({2, 3});
        linear_forward(output, valid_input, valid_weight, context);
    });
    expect_invalid_argument([&] {
        Tensor input({2, 4});
        linear_forward(valid_output, input, valid_weight, context);
    });
    expect_invalid_argument([&] {
        Tensor input({2, 3}, DeviceType::CPU);
        Tensor weight({2, 3}, DeviceType::CPU);
        Tensor output({2, 2}, DeviceType::CPU);
        linear_forward(output, input, weight, context);
    });
}

} // namespace

int main() {
    try {
        test_matrix_multiplication();
        test_batched_input();
        test_fused_bias_and_cached_plan();
        test_irregular_tf32_shape();
        test_reduced_precision(Dtype::F16, 2.0e-2f);
        test_reduced_precision(Dtype::BF16, 5.0e-2f);
        test_validation();
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return EXIT_FAILURE;
    }

    std::cout << "Linear tests passed.\n";
    return EXIT_SUCCESS;
}
