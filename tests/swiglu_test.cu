#include "core/cuda_check.h"
#include "ops/swiglu.h"
#include "cuda_benchmark.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

float silu(const float value) {
    return value / (1.0f + std::exp(-value));
}

void expect_close(
    const std::vector<float>& actual,
    const std::vector<float>& expected,
    const float tolerance
) {
    if (actual.size() != expected.size()) {
        throw std::runtime_error("SwiGLU test: result size mismatch");
    }

    for (std::size_t index = 0; index < actual.size(); ++index) {
        const float error = std::fabs(actual[index] - expected[index]);
        const float scale = std::max(1.0f, std::fabs(expected[index]));
        if (!std::isfinite(actual[index]) || error > tolerance * scale) {
            throw std::runtime_error("SwiGLU test: result value mismatch");
        }
    }
}

std::vector<float> read_tensor(const Tensor& tensor) {
    std::vector<float> values(tensor.numel());
    tensor.copy_to_host(values);
    return values;
}

std::vector<float> reference(
    const std::vector<float>& gate,
    const std::vector<float>& up
) {
    std::vector<float> expected(gate.size());
    for (std::size_t index = 0; index < gate.size(); ++index) {
        expected[index] = silu(gate[index]) * up[index];
    }
    return expected;
}

void test_separate(
    const Dtype dtype,
    const float tolerance,
    const std::size_t elements
) {
    std::vector<float> gate(elements);
    std::vector<float> up(elements);
    for (std::size_t index = 0; index < elements; ++index) {
        gate[index] = 0.125f * static_cast<float>(static_cast<int>(index % 17) - 8);
        up[index] = 0.25f * static_cast<float>(static_cast<int>(index % 11) - 5);
    }

    Tensor gate_tensor({3, elements / 3}, dtype);
    Tensor up_tensor({3, elements / 3}, dtype);
    Tensor output({3, elements / 3}, dtype);
    gate_tensor.copy_from_host(gate);
    up_tensor.copy_from_host(up);

    swiglu_forward(output, gate_tensor, up_tensor);
    CUDA_CHECK(cudaDeviceSynchronize());
    expect_close(read_tensor(output), reference(gate, up), tolerance);
}

void test_fused(
    const Dtype dtype,
    const float tolerance,
    const std::size_t intermediate_size
) {
    constexpr std::size_t rows = 6;
    std::vector<float> gate_up(rows * 2 * intermediate_size);
    for (std::size_t index = 0; index < gate_up.size(); ++index) {
        gate_up[index] = 0.125f * static_cast<float>(
            static_cast<int>(index % 29) - 14
        );
    }

    Tensor input({2, 3, 2 * intermediate_size}, dtype);
    Tensor output({2, 3, intermediate_size}, dtype);
    input.copy_from_host(gate_up);

    swiglu_forward(output, input);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> gate;
    std::vector<float> up;
    for (std::size_t row = 0; row < rows; ++row) {
        const auto offset = row * 2 * intermediate_size;
        gate.insert(
            gate.end(),
            gate_up.begin() + offset,
            gate_up.begin() + offset + intermediate_size
        );
        up.insert(
            up.end(),
            gate_up.begin() + offset + intermediate_size,
            gate_up.begin() + offset + 2 * intermediate_size
        );
    }
    expect_close(read_tensor(output), reference(gate, up), tolerance);
}

template <typename Function>
void expect_invalid_argument(Function&& function) {
    try {
        function();
    } catch (const std::invalid_argument&) {
        return;
    }
    throw std::runtime_error("SwiGLU test: invalid arguments were accepted");
}

template <typename Function>
void expect_overflow_error(Function&& function) {
    try {
        function();
    } catch (const std::overflow_error&) {
        return;
    }
    throw std::runtime_error("SwiGLU test: overflow was not rejected");
}

void test_validation() {
    Tensor gate({2, 3});
    Tensor up({2, 3});
    Tensor output({2, 3});

    expect_invalid_argument([&] {
        Tensor wrong_output({2, 2});
        swiglu_forward(wrong_output, gate, up);
    });
    expect_invalid_argument([&] { swiglu_forward(output, gate, Tensor({3, 2})); });
    expect_invalid_argument([&] {
        Tensor half_gate({2, 3}, Dtype::F16);
        swiglu_forward(output, half_gate, up);
    });
    expect_invalid_argument([&] {
        Tensor cpu_output({2, 3}, DeviceType::CPU);
        Tensor cpu_gate({2, 3}, DeviceType::CPU);
        Tensor cpu_up({2, 3}, DeviceType::CPU);
        swiglu_forward(cpu_output, cpu_gate, cpu_up);
    });

    expect_invalid_argument([&] {
        Tensor wrong_output({2, 2});
        Tensor input({2, 6});
        swiglu_forward(wrong_output, input);
    });
    expect_invalid_argument([&] {
        Tensor fused_output({2, 3});
        Tensor odd_input({2, 5});
        swiglu_forward(fused_output, odd_input);
    });
    expect_invalid_argument([&] {
        Tensor fused_output({2, 2, 3});
        Tensor fused_input({2, 6});
        swiglu_forward(fused_output, fused_input);
    });
    expect_invalid_argument([&] {
        Tensor cpu_output({2, 3}, DeviceType::CPU);
        Tensor cpu_input({2, 6}, DeviceType::CPU);
        swiglu_forward(cpu_output, cpu_input);
    });
    expect_invalid_argument([&] {
        Tensor f16_output({2, 3}, Dtype::F16);
        Tensor f32_input({2, 6});
        swiglu_forward(f16_output, f32_input);
    });
    expect_overflow_error([&] {
        Tensor output({65536, 1});
        Tensor input({65536, 2});
        swiglu_forward(output, input);
    });
}

void test_custom_stream() {
    cudaStream_t stream = nullptr;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    try {
        Tensor gate({1, 3});
        Tensor up({1, 3});
        Tensor output({1, 3});
        gate.copy_from_host(std::vector<float>{-1.0f, 0.0f, 1.0f});
        up.copy_from_host(std::vector<float>{2.0f, 3.0f, 4.0f});
        swiglu_forward(output, gate, up, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        expect_close(read_tensor(output), reference({-1.0f, 0.0f, 1.0f}, {2.0f, 3.0f, 4.0f}), 1.0e-5f);
    } catch (...) {
        cudaStreamDestroy(stream);
        throw;
    }
    CUDA_CHECK(cudaStreamDestroy(stream));
}

void benchmark_kernel() {
    constexpr std::size_t kRows = 512;
    constexpr std::size_t kIntermediateSize = 4096;
    Tensor input({kRows, 2 * kIntermediateSize}, Dtype::F16);
    Tensor output({kRows, kIntermediateSize}, Dtype::F16);

    test::benchmark_cuda_launches("SwiGLU kernel", [&](cudaStream_t stream) {
        swiglu_forward(output, input, stream);
    });
}

} // namespace

int main() {
    try {
        // Odd sizes use scalar kernels; even sizes use packed half2/BF16x2 kernels.
        test_separate(Dtype::F32, 1.0e-5f, 513);
        test_separate(Dtype::F16, 3.0e-3f, 513);
        test_separate(Dtype::F16, 3.0e-3f, 516);
        test_separate(Dtype::BF16, 2.0e-2f, 513);
        test_separate(Dtype::BF16, 2.0e-2f, 516);

        test_fused(Dtype::F32, 1.0e-5f, 513);
        test_fused(Dtype::F16, 3.0e-3f, 513);
        test_fused(Dtype::F16, 3.0e-3f, 514);
        test_fused(Dtype::BF16, 2.0e-2f, 513);
        test_fused(Dtype::BF16, 2.0e-2f, 514);
        test_validation();
        test_custom_stream();
        benchmark_kernel();
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return EXIT_FAILURE;
    }

    std::cout << "SwiGLU tests passed.\n";
    return EXIT_SUCCESS;
}
