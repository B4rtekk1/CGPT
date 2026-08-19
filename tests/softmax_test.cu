#include "core/cuda_check.h"
#include "ops/softmax.h"
#include "cuda_benchmark.h"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <algorithm>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
void expect_close(const Tensor& tensor, const std::vector<float>& expected,
                  const float tolerance) {
    std::vector<float> actual(tensor.numel());
    tensor.copy_to_host(actual);
    if (actual.size() != expected.size())
        throw std::runtime_error("Softmax: result size mismatch");
    for (std::size_t i = 0; i < actual.size(); ++i) {
        const float error = std::fabs(actual[i] - expected[i]);
        const float scale = std::max(1.0f, std::fabs(expected[i]));
        if (!std::isfinite(actual[i]) || error > tolerance * scale)
            throw std::runtime_error("Softmax: unexpected value at index " +
                std::to_string(i) + " (actual=" + std::to_string(actual[i]) +
                ", expected=" + std::to_string(expected[i]) + ")");
    }
}

std::vector<float> reference(const std::vector<float>& input,
                             const std::size_t rows,
                             const std::size_t columns) {
    std::vector<float> output(input.size());
    for (std::size_t row = 0; row < rows; ++row) {
        const auto begin = input.begin() + row * columns;
        const auto end = begin + columns;
        const float maximum = *std::max_element(begin, end);
        float denominator = 0.0f;
        for (std::size_t column = 0; column < columns; ++column) {
            const float value = std::exp(input[row * columns + column] - maximum);
            output[row * columns + column] = value;
            denominator += value;
        }
        for (std::size_t column = 0; column < columns; ++column)
            output[row * columns + column] /= denominator;
    }
    return output;
}

void test_forward() {
    Tensor input({2, 3}), output({2, 3});
    input.copy_from_host(std::vector<float>{1.0f, 2.0f, 3.0f, 1000.0f, 1000.0f, 1000.0f});
    softmax_forward(output, input);
    CUDA_CHECK(cudaDeviceSynchronize());
    expect_close(output, {0.0900306f, 0.2447285f, 0.6652409f, 1.0f / 3.0f, 1.0f / 3.0f, 1.0f / 3.0f}, 1.0e-5f);
}

void test_numerical_reference(const Dtype dtype, const float tolerance,
                              const std::size_t rows, const std::size_t columns) {
    std::vector<float> values(rows * columns);
    for (std::size_t index = 0; index < values.size(); ++index) {
        // Include large offsets and a non-periodic pattern to exercise the
        // max-subtraction, tail, and multi-iteration reduction paths.
        values[index] = 0.17f * static_cast<float>(index % 19)
            - 0.31f * static_cast<float>(index % 13);
        if (index % 37 == 0)
            values[index] -= 12.0f;
    }

    Tensor input({rows, columns}, dtype);
    Tensor output({rows, columns}, dtype);
    input.copy_from_host(values);
    softmax_forward(output, input);
    CUDA_CHECK(cudaDeviceSynchronize());
    expect_close(output, reference(values, rows, columns), tolerance);

    std::vector<float> actual(output.numel());
    output.copy_to_host(actual);
    for (std::size_t row = 0; row < rows; ++row) {
        float sum = 0.0f;
        for (std::size_t column = 0; column < columns; ++column)
            sum += actual[row * columns + column];
        if (std::fabs(sum - 1.0f) > tolerance * 2.0f)
            throw std::runtime_error("Softmax: row is not normalized");
    }
}

void benchmark_kernel() {
    constexpr std::size_t kRows = 512, kColumns = 50'257;
    Tensor input({kRows, kColumns}), output({kRows, kColumns});
    test::benchmark_cuda_launches("Softmax kernel", [&](cudaStream_t stream) { softmax_forward(output, input, stream); }, 1000, 50);
}
} // namespace

int main() {
    try {
        test_forward();
        test_numerical_reference(Dtype::F32, 2.0e-5f, 3, 7);
        test_numerical_reference(Dtype::F32, 2.0e-5f, 2, 513);
        test_numerical_reference(Dtype::F16, 4.0e-3f, 3, 513);
        test_numerical_reference(Dtype::BF16, 3.0e-2f, 3, 1025);
        benchmark_kernel();
    }
    catch (const std::exception& error) { std::cerr << error.what() << '\n'; return EXIT_FAILURE; }
    std::cout << "Softmax tests passed.\n";
    return EXIT_SUCCESS;
}
