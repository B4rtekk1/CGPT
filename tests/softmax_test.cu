#include "core/cuda_check.h"
#include "ops/softmax.h"
#include "cuda_benchmark.h"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {
void expect_close(const Tensor& tensor, const std::vector<float>& expected) {
    std::vector<float> actual(tensor.numel());
    tensor.copy_to_host(actual);
    for (std::size_t i = 0; i < actual.size(); ++i)
        if (std::fabs(actual[i] - expected[i]) > 1.0e-5f)
            throw std::runtime_error("Softmax: unexpected value");
}

void test_forward() {
    Tensor input({2, 3}), output({2, 3});
    input.copy_from_host(std::vector<float>{1.0f, 2.0f, 3.0f, 1000.0f, 1000.0f, 1000.0f});
    softmax_forward(output, input);
    CUDA_CHECK(cudaDeviceSynchronize());
    expect_close(output, {0.0900306f, 0.2447285f, 0.6652409f, 1.0f / 3.0f, 1.0f / 3.0f, 1.0f / 3.0f});
}

void benchmark_kernel() {
    constexpr std::size_t kRows = 512, kColumns = 50'257;
    Tensor input({kRows, kColumns}), output({kRows, kColumns});
    test::benchmark_cuda_launches("Softmax kernel", [&](cudaStream_t stream) { softmax_forward(output, input, stream); }, 1000, 50);
}
} // namespace

int main() {
    try { test_forward(); benchmark_kernel(); }
    catch (const std::exception& error) { std::cerr << error.what() << '\n'; return EXIT_FAILURE; }
    std::cout << "Softmax tests passed.\n";
    return EXIT_SUCCESS;
}
