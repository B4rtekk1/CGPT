#include "core/cuda_check.h"
#include "ops/cross_entropy.h"
#include "cuda_benchmark.h"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

class DeviceTargets {
public:
    explicit DeviceTargets(const std::vector<bpe::TokenId>& values) {
        CUDA_CHECK(cudaMalloc(&data_, values.size() * sizeof(bpe::TokenId)));
        CUDA_CHECK(cudaMemcpy(data_, values.data(), values.size() * sizeof(bpe::TokenId), cudaMemcpyHostToDevice));
    }
    ~DeviceTargets() { if (data_ != nullptr) cudaFree(data_); }
    [[nodiscard]] bpe::TokenId* get() const { return data_; }
private:
    bpe::TokenId* data_ = nullptr;
};

void expect_close(const Tensor& tensor, const std::vector<float>& expected, const float tolerance = 1.0e-5f) {
    std::vector<float> actual(tensor.numel());
    tensor.copy_to_host(actual);
    if (actual.size() != expected.size()) throw std::runtime_error("Cross entropy: output size mismatch");
    for (std::size_t i = 0; i < actual.size(); ++i)
        if (std::fabs(actual[i] - expected[i]) > tolerance) throw std::runtime_error("Cross entropy: unexpected value");
}

void test_loss_and_gradient() {
    Tensor logits({2, 3});
    Tensor gradient({2, 3});
    Tensor loss({1});
    logits.copy_from_host(std::vector<float>{1.0f, 2.0f, 3.0f, 1.0f, 1.0f, 1.0f});
    DeviceTargets targets({2, 0});
    cross_entropy_forward_backward(loss, gradient, logits, targets.get(), 2);
    CUDA_CHECK(cudaDeviceSynchronize());
    const float expected_loss = (std::log(std::exp(1.0f) + std::exp(2.0f) + std::exp(3.0f)) - 3.0f + std::log(3.0f)) / 2.0f;
    expect_close(loss, {expected_loss});
    expect_close(gradient, {0.0450153f, 0.1223642f, -0.1673795f, -0.3333333f, 0.1666667f, 0.1666667f});
}

void test_validation() {
    Tensor logits({2, 3}); Tensor gradient({2, 3}); Tensor loss({1}); DeviceTargets targets({0, 1});
    try { cross_entropy_forward_backward(loss, gradient, logits, targets.get(), 1); }
    catch (const std::invalid_argument&) { return; }
    throw std::runtime_error("Cross entropy: accepted invalid target count");
}

void benchmark_kernel() {
    constexpr std::size_t kRows = 512;
    constexpr std::size_t kVocabularySize = 50'257;

    Tensor logits({kRows, kVocabularySize});
    Tensor gradient({kRows, kVocabularySize});
    Tensor loss({1});
    DeviceTargets targets(std::vector<bpe::TokenId>(kRows, 0));

    test::benchmark_cuda_launches("Cross entropy kernel", [&](cudaStream_t stream) {
        cross_entropy_forward_backward(loss, gradient, logits, targets.get(), kRows, stream);
    }, 1000, 50);
}

} // namespace

int main() {
    try { test_loss_and_gradient(); test_validation(); benchmark_kernel(); }
    catch (const std::exception& error) { std::cerr << error.what() << '\n'; return EXIT_FAILURE; }
    std::cout << "Cross entropy tests passed.\n";
    return EXIT_SUCCESS;
}
