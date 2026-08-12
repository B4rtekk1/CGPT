#include "optim/adamw.h"

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

void expect_close(const Tensor& tensor, const std::vector<float>& expected, float tolerance = 1.0e-6f) {
    std::vector<float> actual(tensor.numel());
    tensor.copy_to_host(actual);
    if (actual.size() != expected.size()) throw std::runtime_error("AdamW: incorrect result size");
    for (std::size_t i = 0; i < actual.size(); ++i) {
        if (std::fabs(actual[i] - expected[i]) > tolerance) {
            throw std::runtime_error("AdamW: incorrect result");
        }
    }
}

void test_update_and_state(DeviceType device, Dtype dtype, float tolerance) {
    Tensor parameter({2}, device, dtype);
    Tensor gradient({2}, device, dtype);
    parameter.copy_from_host(std::vector<float>{1.0f, -2.0f});
    gradient.copy_from_host(std::vector<float>{0.1f, -0.2f});
    AdamWState state = AdamWState::for_parameter(parameter);
    AdamWOptions options{.learning_rate = 0.1f, .beta1 = 0.9f, .beta2 = 0.999f, .epsilon = 1.0e-8f, .weight_decay = 0.01f};

    adamw_step(parameter, gradient, state, options);
    expect_close(parameter, {0.899f, -1.898f}, tolerance);
    expect_close(state.first_moment, {0.01f, -0.02f}, tolerance);
    expect_close(state.second_moment, {0.00001f, 0.00004f}, tolerance);
    if (state.step != 1) throw std::runtime_error("AdamW: step counter was not updated");

    adamw_step(parameter, gradient, state, options);
    expect_close(parameter, {0.798101f, -1.796102f}, tolerance);
    if (state.step != 2) throw std::runtime_error("AdamW: step counter was not updated twice");
}

void test_validation() {
    Tensor parameter({1}, DeviceType::CPU);
    Tensor gradient({2}, DeviceType::CPU);
    AdamWState state = AdamWState::for_parameter(parameter);
    try {
        adamw_step(parameter, gradient, state);
        throw std::runtime_error("AdamW accepted incompatible tensors");
    } catch (const std::invalid_argument&) {
    }
}

} // namespace

int main() {
    try {
        test_update_and_state(DeviceType::CPU, Dtype::F32, 2.0e-6f);
        test_update_and_state(DeviceType::CPU, Dtype::F16, 2.0e-3f);
        test_update_and_state(DeviceType::CPU, Dtype::BF16, 1.0e-2f);
        test_update_and_state(DeviceType::CUDA, Dtype::F32, 2.0e-6f);
        test_update_and_state(DeviceType::CUDA, Dtype::F16, 2.0e-3f);
        test_update_and_state(DeviceType::CUDA, Dtype::BF16, 1.0e-2f);
        test_validation();
        std::cout << "AdamW tests passed.\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
