#include "cuda_check.h"
#include "ops/rmsnorm.h"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>

namespace {

void expect_close(
    const std::vector<float>& actual,
    const std::vector<float>& expected,
    float tolerance
) {
    if (actual.size() != expected.size()) {
        std::cerr << "Size mismatch: got " << actual.size()
                  << ", expected " << expected.size() << '\n';
        std::exit(EXIT_FAILURE);
    }

    for (std::size_t i = 0; i < actual.size(); ++i) {
        const float difference = std::fabs(actual[i] - expected[i]);
        if (difference > tolerance) {
            std::cerr << "Mismatch at index " << i
                      << ": got " << actual[i]
                      << ", expected " << expected[i]
                      << ", difference " << difference << '\n';
            std::exit(EXIT_FAILURE);
        }
    }
}

std::vector<float> rmsnorm_reference(
    const std::vector<float>& input,
    const std::vector<float>& weight,
    int rows,
    int hidden,
    float epsilon
) {
    std::vector<float> expected(input.size());

    for (int row = 0; row < rows; ++row) {
        float sum_squares = 0.0f;
        for (int column = 0; column < hidden; ++column) {
            const float value = input[row * hidden + column];
            sum_squares += value * value;
        }

        const float rms = std::sqrt(
            sum_squares / static_cast<float>(hidden) + epsilon
        );

        for (int column = 0; column < hidden; ++column) {
            const int index = row * hidden + column;
            expected[index] = input[index] / rms * weight[column];
        }
    }

    return expected;
}

void run_case(int rows, int hidden, float epsilon) {
    std::vector<float> input(rows * hidden);
    std::vector<float> weight(hidden);

    for (int row = 0; row < rows; ++row) {
        for (int column = 0; column < hidden; ++column) {
            input[row * hidden + column] =
                0.25f * static_cast<float>((row + 1) * (column % 7 - 3));
        }
    }
    for (int column = 0; column < hidden; ++column) {
        weight[column] = 0.5f + 0.125f * static_cast<float>(column % 5);
    }

    const std::vector<float> expected =
        rmsnorm_reference(input, weight, rows, hidden, epsilon);

    float* device_output = nullptr;
    float* device_input = nullptr;
    float* device_weight = nullptr;

    CUDA_CHECK(cudaMalloc(&device_output, input.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&device_input, input.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&device_weight, weight.size() * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(
        device_input,
        input.data(),
        input.size() * sizeof(float),
        cudaMemcpyHostToDevice
    ));
    CUDA_CHECK(cudaMemcpy(
        device_weight,
        weight.data(),
        weight.size() * sizeof(float),
        cudaMemcpyHostToDevice
    ));

    rmsnorm_forward(
        device_output,
        device_input,
        device_weight,
        rows,
        hidden,
        epsilon
    );
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> actual(input.size());
    CUDA_CHECK(cudaMemcpy(
        actual.data(),
        device_output,
        actual.size() * sizeof(float),
        cudaMemcpyDeviceToHost
    ));

    CUDA_CHECK(cudaFree(device_weight));
    CUDA_CHECK(cudaFree(device_input));
    CUDA_CHECK(cudaFree(device_output));

    expect_close(actual, expected, 1.0e-4f);
}

} // namespace

int main() {
    // Exercises several rows and a hidden size below one CUDA block.
    run_case(3, 7, 1.0e-5f);

    // Exercises the second loop iteration in each thread and the reduction.
    run_case(2, 513, 1.0e-5f);

    std::cout << "RMSNorm tests passed.\n";
    return EXIT_SUCCESS;
}
