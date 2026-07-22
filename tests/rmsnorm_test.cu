#include "cuda_check.h"
#include "ops/rmsnorm.h"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
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

void run_case(std::vector<std::size_t> shape, float epsilon) {
    const int rows = static_cast<int>(shape[0]);
    const int hidden = static_cast<int>(shape[1]);
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

    Tensor input_tensor(shape);
    Tensor weight_tensor({shape[1]});
    input_tensor.copy_from_host(input);
    weight_tensor.copy_from_host(weight);

    Tensor output_tensor = rmsnorm_forward(input_tensor, weight_tensor, epsilon);
    if (output_tensor.shape() != shape || output_tensor.dim() != 2) {
        std::cerr << "RMSNorm returned an invalid output shape\n";
        std::exit(EXIT_FAILURE);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> actual(input.size());
    output_tensor.copy_to_host(actual);

    expect_close(actual, expected, 1.0e-4f);
}

void expect_invalid_argument(
    const Tensor& input,
    const Tensor& weight
) {
    try {
        const Tensor output = rmsnorm_forward(input, weight, 1.0e-5f);
        (void)output;
    } catch (const std::invalid_argument&) {
        return;
    }

    std::cerr << "Expected RMSNorm to reject invalid tensor shapes\n";
    std::exit(EXIT_FAILURE);
}

} // namespace

int main() {
    // Exercises several rows and a hidden size below one CUDA block.
    run_case({3, 7}, 1.0e-5f);

    // Exercises the second loop iteration in each thread and the reduction.
    run_case({2, 513}, 1.0e-5f);

    expect_invalid_argument(Tensor({2, 4, 1}), Tensor({4}));
    expect_invalid_argument(Tensor({2, 4}), Tensor({2, 4}));

    Tensor mismatched_input({2, 4});
    Tensor mismatched_weight({3});
    expect_invalid_argument(mismatched_input, mismatched_weight);

    std::cout << "RMSNorm tests passed.\n";
    return EXIT_SUCCESS;
}
