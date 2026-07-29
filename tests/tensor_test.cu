#include "../include/core/tensor.h"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <vector>
#include <utility>

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

void test_cpu_gpu_copy_round_trip() {
    const std::vector<float> expected{1.0f, -2.0f, 3.5f, 4.25f};
    Tensor gpu_tensor({2, 2});
    gpu_tensor.copy_from_host(expected);

    std::vector<float> actual(expected.size());
    gpu_tensor.copy_to_host(actual);
    expect_values(actual, expected);

    Tensor cpu_tensor({2, 2}, DeviceType::CPU);
    cpu_tensor.copy_from_host(actual);
    actual.assign(expected.size(), 0.0f);
    cpu_tensor.copy_to_host(actual);
    expect_values(actual, expected);
}

void test_reduced_precision_storage() {
    const std::vector<float> values{1.0f, -2.5f, 3.25f, 0.125f};

    for (const Dtype dtype : {Dtype::F16, Dtype::BF16}) {
        Tensor gpu_tensor({2, 2}, dtype);
        gpu_tensor.copy_from_host(values);
        expect_values(read_values(gpu_tensor), values, 1.0e-2f);
        if (gpu_tensor.nbytes() != values.size() * 2) {
            throw std::runtime_error("Tensor test: reduced dtype uses invalid storage size");
        }

        Tensor cpu_tensor({2, 2}, DeviceType::CPU, dtype);
        cpu_tensor.copy_from_host(values);
        expect_values(read_values(cpu_tensor), values, 1.0e-2f);
    }
}

void test_tensor_copy_and_move() {
    const std::vector<float> expected{1.0f, 2.0f, 3.0f, 4.0f};
    Tensor original({2, 2});
    original.copy_from_host(expected);

    Tensor copied(original);
    original.copy_from_host(std::vector<float>{9.0f, 9.0f, 9.0f, 9.0f});
    expect_values(read_values(copied), expected);

    Tensor assigned({1});
    assigned = copied;
    expect_values(read_values(assigned), expected);

    Tensor moved(std::move(assigned));
    expect_values(read_values(moved), expected);
    if (assigned.numel() != 0) {
        throw std::runtime_error("Moved-from Tensor should be empty");
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

    try {
        const Tensor tensor({2, 3}, DeviceType::CPU);
        (void)tensor.size(2);
        throw std::runtime_error("Tensor accepted an out-of-range axis");
    } catch (const std::out_of_range&) {
    }

    try {
        const auto max_dimension = std::numeric_limits<std::size_t>::max();
        const Tensor tensor({max_dimension, 2}, DeviceType::CPU);
        (void)tensor;
        throw std::runtime_error("Tensor accepted an overflowing shape");
    } catch (const std::invalid_argument&) {
    }
}

void test_self_assignment() {
    Tensor tensor({2, 2}, DeviceType::CPU);
    const std::vector<float> expected{1.0f, 2.0f, 3.0f, 4.0f};
    tensor.copy_from_host(expected);
    expect_values(read_values(tensor), expected);
}

} // namespace

int main() {
    try {
        test_shape_and_empty();
        test_cpu_factories();
        test_cuda_factories();
        test_copy_validation();
        test_cpu_gpu_copy_round_trip();
        test_reduced_precision_storage();
        test_tensor_copy_and_move();
        test_invalid_shapes();
        test_self_assignment();
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return EXIT_FAILURE;
    }

    std::cout << "Tensor tests passed.\n";
    return EXIT_SUCCESS;
}
