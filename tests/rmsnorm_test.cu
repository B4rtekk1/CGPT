#include "../include/core/cuda_check.h"
#include "ops/rmsnorm.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {

void print_tensor(
    const char* name,
    const std::vector<float>& values,
    const std::vector<std::size_t>& shape,
    std::size_t max_rows = 8,
    std::size_t max_columns = 12
) {
    std::cout << name << " shape=[";
    for (std::size_t dimension = 0; dimension < shape.size(); ++dimension) {
        if (dimension != 0) {
            std::cout << ", ";
        }
        std::cout << shape[dimension];
    }
    std::cout << "]\n";

    if (shape.size() != 2 || shape[0] == 0 || shape[1] == 0) {
        std::cout << "  <empty or non-matrix tensor>\n";
        return;
    }

    const std::size_t rows = shape[0];
    const std::size_t columns = shape[1];
    const std::size_t rows_to_print = std::min(rows, max_rows);
    const std::size_t head_columns = std::min(columns, max_columns);
    const std::size_t tail_columns =
        columns > max_columns ? std::min<std::size_t>(3, columns - head_columns) : 0;

    std::cout << std::fixed << std::setprecision(5);
    for (std::size_t row = 0; row < rows_to_print; ++row) {
        std::cout << "  [";
        for (std::size_t column = 0; column < head_columns; ++column) {
            if (column != 0) {
                std::cout << ", ";
            }
            std::cout << std::setw(9) << values[row * columns + column];
        }

        if (tail_columns != 0) {
            std::cout << ", ...";
            for (std::size_t column = columns - tail_columns; column < columns; ++column) {
                std::cout << ", " << std::setw(9) << values[row * columns + column];
            }
        }
        std::cout << "]\n";
    }

    if (rows_to_print < rows) {
        std::cout << "  ... (" << rows - rows_to_print << " more row(s))\n";
    }
}

void expect_close(
    const std::vector<float>& actual,
    const std::vector<float>& expected,
    float tolerance
) {
    if (actual.size() != expected.size()) {
        std::cerr << "Size mismatch: got " << actual.size()
                  << ", expected " << expected.size() << '\n';
        throw std::runtime_error("RMSNorm result size mismatch");
    }

    for (std::size_t i = 0; i < actual.size(); ++i) {
        const float difference = std::fabs(actual[i] - expected[i]);
        const float scale = std::max(1.0f, std::fabs(expected[i]));
        if (!std::isfinite(actual[i]) ||
            !std::isfinite(expected[i]) ||
            difference > tolerance * scale) {
            std::cerr << "Mismatch at index " << i
                      << ": got " << actual[i]
                      << ", expected " << expected[i]
                      << ", difference " << difference << '\n';
            throw std::runtime_error("RMSNorm result mismatch");
        }
    }
}

std::vector<float> rmsnorm_reference(
    const std::vector<float>& input,
    const std::vector<float>& weight,
    std::size_t rows,
    std::size_t hidden,
    float epsilon
) {
    std::vector<float> expected(input.size());

    for (std::size_t row = 0; row < rows; ++row) {
        double sum_squares = 0.0;
        for (std::size_t column = 0; column < hidden; ++column) {
            const float value = input[row * hidden + column];
            sum_squares += static_cast<double>(value) * value;
        }

        const float rms = static_cast<float>(std::sqrt(
            sum_squares / static_cast<double>(hidden) + epsilon
        ));

        for (std::size_t column = 0; column < hidden; ++column) {
            const std::size_t index = row * hidden + column;
            expected[index] = input[index] / rms * weight[column];
        }
    }

    return expected;
}

void run_case(const std::vector<std::size_t>& shape, float epsilon) {
    if (shape.size() != 2 || shape[0] == 0 || shape[1] == 0) {
        throw std::invalid_argument("RMSNorm shape must be [rows, hidden]");
    }
    if (shape[0] > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
        shape[1] > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::invalid_argument("RMSNorm shape exceeds the supported range");
    }
    if (shape[0] > std::numeric_limits<std::size_t>::max() / shape[1]) {
        throw std::invalid_argument("RMSNorm shape has too many elements");
    }

    const std::size_t rows = shape[0];
    const std::size_t hidden = shape[1];
    const std::size_t elements = rows * hidden;
    std::vector<float> input(elements);
    std::vector<float> weight(hidden);

    for (std::size_t row = 0; row < rows; ++row) {
        for (std::size_t column = 0; column < hidden; ++column) {
            // Keep the multiplication signed.  `row` is size_t, so doing
            // the multiplication before the cast would convert negative
            // values to very large unsigned integers.
            input[row * hidden + column] =
                0.25f * static_cast<float>(row + 1) *
                static_cast<float>(static_cast<int>(column % 7) - 3);
        }
    }
    for (std::size_t column = 0; column < hidden; ++column) {
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
        throw std::runtime_error("RMSNorm returned an invalid output shape");
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> actual(input.size());
    output_tensor.copy_to_host(actual);

    print_tensor("RMSNorm output", actual, shape);
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
    throw std::runtime_error("RMSNorm accepted invalid tensor shapes");
}

void test_invalid_epsilon() {
    Tensor input({1, 4});
    Tensor weight({4});

    for (const float epsilon : {0.0f, -1.0f, NAN, INFINITY}) {
        try {
            const Tensor output = rmsnorm_forward(input, weight, epsilon);
            (void)output;
        } catch (const std::invalid_argument&) {
            continue;
        }
        std::cerr << "RMSNorm accepted invalid epsilon\n";
        throw std::runtime_error("RMSNorm accepted invalid epsilon");
    }
}

void test_custom_stream() {
    cudaStream_t stream = nullptr;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    try {
        Tensor input({2, 7});
        Tensor weight = Tensor::ones({7}, DeviceType::CUDA, stream);
        input.copy_from_host(std::vector<float>(14, 1.0f));

        Tensor output = rmsnorm_forward(input, weight, 1.0e-5f, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));

        const auto values = [&] {
            std::vector<float> result(output.numel());
            output.copy_to_host(result);
            return result;
        }();

        for (const float value : values) {
            if (std::fabs(value - 1.0f) > 1.0e-4f) {
                throw std::runtime_error("RMSNorm stream result is incorrect");
            }
        }
    } catch (...) {
        cudaStreamDestroy(stream);
        throw;
    }

    CUDA_CHECK(cudaStreamDestroy(stream));
}

void test_device_type() {
    Tensor cpu_tensor({2, 3}, DeviceType::CPU);
    if (cpu_tensor.device_type() != DeviceType::CPU ||
        cpu_tensor.deviceType() != DeviceType::CPU) {
        std::cerr << "Tensor did not preserve CPU device type\n";
        throw std::runtime_error("Tensor did not preserve CPU device type");
    }

    const std::vector<float> values{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    cpu_tensor.copy_from_host(values);

    std::vector<float> copied(values.size());
    cpu_tensor.copy_to_host(copied);
    expect_close(copied, values, 0.0f);
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

    test_device_type();
    expect_invalid_argument(
        Tensor({2, 4}, DeviceType::CPU),
        Tensor({4}, DeviceType::CPU)
    );
    test_invalid_epsilon();
    test_custom_stream();

    std::cout << "RMSNorm tests passed.\n";
    return EXIT_SUCCESS;
}
