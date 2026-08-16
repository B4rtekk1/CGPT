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

    if (!adamw_step(parameter, gradient, state, options)) throw std::runtime_error("AdamW: valid update was skipped");
    expect_close(parameter, {0.899f, -1.898f}, tolerance);
    expect_close(state.first_moment, {0.01f, -0.02f}, tolerance);
    expect_close(state.second_moment, {0.00001f, 0.00004f}, tolerance);
    if (state.step != 1) throw std::runtime_error("AdamW: step counter was not updated");

    if (!adamw_step(parameter, gradient, state, options)) throw std::runtime_error("AdamW: valid update was skipped");
    expect_close(parameter, {0.798101f, -1.796102f}, tolerance);
    if (state.step != 2) throw std::runtime_error("AdamW: step counter was not updated twice");
}

void test_numerical_stability(DeviceType device, Dtype dtype, float tolerance) {
    Tensor parameter({2}, device, dtype);
    Tensor gradient({2}, device, dtype);
    parameter.copy_from_host(std::vector<float>{1.0f, -1.0f});
    gradient.copy_from_host(std::vector<float>{8.0f, 0.0f});
    AdamWState state = AdamWState::for_parameter(parameter);
    AdamWOptions options{.learning_rate = 0.1f, .weight_decay = 0.0f, .loss_scale = 4.0f, .max_grad_norm = 1.0f};
    if (!adamw_step(parameter, gradient, state, options)) throw std::runtime_error("AdamW: stable update was skipped");
    // After unscaling, norm is 2 and clipping reduces the effective gradient to 1.
    expect_close(parameter, {0.9f, -1.0f}, tolerance);
    expect_close(state.master_parameter, {0.9f, -1.0f}, 2.0e-6f);

    const std::vector<float> before = [&] { std::vector<float> result(2); parameter.copy_to_host(result); return result; }();
    gradient.copy_from_host(std::vector<float>{std::numeric_limits<float>::infinity(), 0.0f});
    if (adamw_step(parameter, gradient, state, options)) throw std::runtime_error("AdamW: accepted an infinite gradient");
    expect_close(parameter, before, tolerance);
    if (state.step != 1) throw std::runtime_error("AdamW: advanced step after non-finite gradient");
}

void test_validation() {
    Tensor parameter({1}, DeviceType::CPU);
    Tensor gradient({2}, DeviceType::CPU);
    AdamWState state = AdamWState::for_parameter(parameter);
    try {
        static_cast<void>(adamw_step(parameter, gradient, state));
        throw std::runtime_error("AdamW accepted incompatible tensors");
    } catch (const std::invalid_argument&) {
    }
}

void test_batched_non_finite_gradient() {
    Tensor parameter({2}, DeviceType::CUDA, Dtype::F32);
    Tensor gradient({2}, DeviceType::CUDA, Dtype::F32);
    parameter.copy_from_host(std::vector<float>{1.0f, -1.0f});
    gradient.copy_from_host(std::vector<float>{std::numeric_limits<float>::infinity(), 0.0f});
    AdamWState state = AdamWState::for_parameter(parameter);
    AdamWWorkspace workspace;
    const AdamWBatchEntry entry{&parameter, &gradient, &state};

    gradient.copy_from_host(std::vector<float>{1.0f, 0.0f});
    AdamWOptions options{.learning_rate = 0.1f, .weight_decay = 0.0f};
    adamw_step_many_async(std::span<const AdamWBatchEntry>(&entry, 1), options, workspace);
    if (!adamw_check(workspace)) throw std::runtime_error("AdamW batch skipped a finite gradient");
    expect_close(parameter, {0.9f, -1.0f});
    if (state.step != 1) throw std::runtime_error("AdamW batch did not advance step after a finite gradient");

    gradient.copy_from_host(std::vector<float>{std::numeric_limits<float>::infinity(), 0.0f});
    adamw_step_many_async(std::span<const AdamWBatchEntry>(&entry, 1), {}, workspace);
    if (adamw_check(workspace)) throw std::runtime_error("AdamW batch accepted an infinite gradient");
    expect_close(parameter, {0.9f, -1.0f});
    if (state.step != 1) throw std::runtime_error("AdamW batch advanced step after non-finite gradient");
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
        test_numerical_stability(DeviceType::CPU, Dtype::F32, 2.0e-6f);
        test_numerical_stability(DeviceType::CPU, Dtype::F16, 2.0e-3f);
        test_numerical_stability(DeviceType::CPU, Dtype::BF16, 1.0e-2f);
        test_numerical_stability(DeviceType::CUDA, Dtype::F32, 2.0e-6f);
        test_numerical_stability(DeviceType::CUDA, Dtype::F16, 2.0e-3f);
        test_numerical_stability(DeviceType::CUDA, Dtype::BF16, 1.0e-2f);
        test_validation();
        test_batched_non_finite_gradient();
        std::cout << "AdamW tests passed.\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
