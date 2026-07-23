#include "../include/core/tensor.h"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

void expect_values(
    const std::vector<float>& actual,
    const std::vector<float>& expected,
    float tolerance = 0.0f) {
    if (actual.size() != expected.size()) {
        throw std::runtime_error("Tensor test: unexpected value count");
    }

    for (std::size_t index = 0; index < actual.size(); ++index) {
        if (std::fabs(actual[index] - expected[index]) > tolerance) {
            throw std::runtime_error("Tensor test: unexpected value");
        }
    }
}

std::vector<float> read_values(const Tensor& tensor) {
    std::vector<float> values(tensor.numel());
    tensor.copy_to_host(values);
    return values;
}

void test_shape_and_empty() {
    const Tensor tensor = Tensor::empty({2, 3, 4}, DeviceType::CPU);

    if (tensor.shape() != std::vector<std::size_t>{2, 3, 4} ||
        tensor.dim() != 3 || tensor.numel() != 24 ||
        tensor.size(1) != 3 || tensor.device_type() != DeviceType::CPU) {
        throw std::runtime_error("Tensor test: invalid shape metadata");
    }
}

void test_cpu_factories() {
    expect_values(
        read_values(Tensor::zeros({2, 3}, DeviceType::CPU)),
        {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f});
    expect_values(
        read_values(Tensor::ones({2, 3}, DeviceType::CPU)),
        {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f});
    expect_values(
        read_values(Tensor::full({2, 2}, -2.5f, DeviceType::CPU)),
        {-2.5f, -2.5f, -2.5f, -2.5f});
    expect_values(
        read_values(Tensor::eye(2, 3, DeviceType::CPU)),
        {1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f});
}

void test_cuda_factories() {
    expect_values(
        read_values(Tensor::zeros({2, 3})),
        {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f});
    expect_values(
        read_values(Tensor::ones({2, 3})),
        {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f});
    expect_values(
        read_values(Tensor::full({2, 2}, 3.25f)),
        {3.25f, 3.25f, 3.25f, 3.25f});
    expect_values(
        read_values(Tensor::eye(3, 2)),
        {1.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f});
}

void test_copy_validation() {
    Tensor tensor({2, 2}, DeviceType::CPU);

    try {
        tensor.copy_from_host(std::vector<float>{1.0f});
        throw std::runtime_error("Tensor accepted an undersized input");
    } catch (const std::invalid_argument&) {
    }

    try {
        std::vector<float> output(3);
        tensor.copy_to_host(output);
        throw std::runtime_error("Tensor accepted an undersized output");
    } catch (const std::invalid_argument&) {
    }
}

void test_invalid_shapes() {
    try {
        const Tensor tensor(std::vector<std::size_t>{});
        (void)tensor;
        throw std::runtime_error("Tensor accepted an empty shape");
    } catch (const std::invalid_argument&) {
    }

    try {
        const Tensor tensor({2, 0});
        (void)tensor;
        throw std::runtime_error("Tensor accepted a zero dimension");
    } catch (const std::invalid_argument&) {
    }
}

} // namespace

int main() {
    try {
        test_shape_and_empty();
        test_cpu_factories();
        test_cuda_factories();
        test_copy_validation();
        test_invalid_shapes();
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return EXIT_FAILURE;
    }

    std::cout << "Tensor tests passed.\n";
    return EXIT_SUCCESS;
}
