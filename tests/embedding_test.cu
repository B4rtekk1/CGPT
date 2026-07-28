#include "core/cuda_check.h"
#include "ops/embedding.h"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

void expect_close(const std::vector<float>& actual, const std::vector<float>& expected,
                  const float tolerance) {
    if (actual.size() != expected.size()) {
        throw std::runtime_error("Embedding test: result size mismatch");
    }

    for (std::size_t index = 0; index < actual.size(); ++index) {
        if (!std::isfinite(actual[index]) ||
            std::fabs(actual[index] - expected[index]) > tolerance) {
            throw std::runtime_error("Embedding test: result value mismatch");
        }
    }
}

std::vector<float> read_tensor(const Tensor& tensor) {
    std::vector<float> values(tensor.numel());
    tensor.copy_to_host(values);
    return values;
}

template <typename Function>
void expect_invalid_argument(Function&& function) {
    try {
        function();
    } catch (const std::invalid_argument&) {
        return;
    }
    throw std::runtime_error("Embedding test: invalid arguments were accepted");
}

class DeviceTokenIds {
public:
    explicit DeviceTokenIds(const std::size_t count) {
        CUDA_CHECK(cudaMalloc(&data_, count * sizeof(*data_)));
    }

    DeviceTokenIds(const DeviceTokenIds&) = delete;
    DeviceTokenIds& operator=(const DeviceTokenIds&) = delete;

    ~DeviceTokenIds() noexcept {
        if (data_ != nullptr) {
            static_cast<void>(cudaFree(data_));
        }
    }

    [[nodiscard]] bpe::TokenId* get() const noexcept { return data_; }

private:
    bpe::TokenId* data_ = nullptr;
};

void test_lookup_and_upload() {
    Tensor weight({4, 3});
    Tensor output({2, 2, 3});
    weight.copy_from_host(std::vector<float>{
        1.0f, 2.0f, 3.0f,
        4.0f, 5.0f, 6.0f,
        7.0f, 8.0f, 9.0f,
        10.0f, 11.0f, 12.0f});

    DeviceTokenIds token_ids(4);
    const std::vector<bpe::TokenId> host_ids{3, 1, 0, 2};
    embedding_upload_token_ids(token_ids.get(), host_ids);
    embedding_forward(output, token_ids.get(), host_ids.size(), weight);
    CUDA_CHECK(cudaDeviceSynchronize());

    expect_close(read_tensor(output), {
        10.0f, 11.0f, 12.0f,
        4.0f, 5.0f, 6.0f,
        1.0f, 2.0f, 3.0f,
        7.0f, 8.0f, 9.0f}, 1.0e-5f);
}

void test_bounds_check() {
    Tensor weight({2, 2});
    Tensor output({3, 2});
    weight.copy_from_host(std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f});

    DeviceTokenIds token_ids(3);
    embedding_upload_token_ids(token_ids.get(), std::vector<bpe::TokenId>{1, 8, 0});
    embedding_forward(output, token_ids.get(), 3, weight);
    CUDA_CHECK(cudaDeviceSynchronize());

    expect_close(read_tensor(output), {3.0f, 4.0f, 0.0f, 0.0f, 1.0f, 2.0f}, 1.0e-5f);
}

void test_reduced_precision_and_stream() {
    cudaStream_t stream = nullptr;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    try {
        for (const Dtype dtype : {Dtype::F16, Dtype::BF16}) {
            Tensor weight({3, 4}, dtype);
            Tensor output({2, 4}, dtype);
            weight.copy_from_host(std::vector<float>{
                0.25f, 1.5f, -2.0f, 4.0f,
                5.0f, -6.0f, 7.0f, 8.0f,
                9.0f, 10.0f, 11.0f, -12.0f});
            DeviceTokenIds token_ids(2);
            embedding_upload_token_ids(token_ids.get(), std::vector<bpe::TokenId>{2, 0}, stream);
            embedding_forward(output, token_ids.get(), 2, weight, stream,
                              EmbeddingOptions{.block_size = 32});
            CUDA_CHECK(cudaStreamSynchronize(stream));
            expect_close(read_tensor(output),
                         {9.0f, 10.0f, 11.0f, -12.0f, 0.25f, 1.5f, -2.0f, 4.0f},
                         dtype == Dtype::F16 ? 1.0e-2f : 5.0e-2f);
        }
    } catch (...) {
        cudaStreamDestroy(stream);
        throw;
    }

    CUDA_CHECK(cudaStreamDestroy(stream));
}

void test_validation() {
    Tensor weight({2, 3});
    Tensor output({2, 3});
    DeviceTokenIds token_ids(2);

    expect_invalid_argument([&] { embedding_forward(output, nullptr, 2, weight); });
    expect_invalid_argument([&] { embedding_forward(output, token_ids.get(), 0, weight); });
    expect_invalid_argument([&] {
        embedding_forward(output, token_ids.get(), 2, weight, nullptr,
                          EmbeddingOptions{.block_size = 31});
    });
    expect_invalid_argument([&] {
        Tensor bad_output({2, 2});
        embedding_forward(bad_output, token_ids.get(), 2, weight);
    });
    expect_invalid_argument([&] {
        Tensor bad_weight({2, 3, 1});
        embedding_forward(output, token_ids.get(), 2, bad_weight);
    });
    expect_invalid_argument([&] {
        Tensor cpu_weight({2, 3}, DeviceType::CPU);
        Tensor cpu_output({2, 3}, DeviceType::CPU);
        embedding_forward(cpu_output, token_ids.get(), 2, cpu_weight);
    });
    expect_invalid_argument([&] {
        Tensor f16_output({2, 3}, Dtype::F16);
        embedding_forward(f16_output, token_ids.get(), 2, weight);
    });
}

} // namespace

int main() {
    try {
        test_lookup_and_upload();
        test_bounds_check();
        test_reduced_precision_and_stream();
        test_validation();
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return EXIT_FAILURE;
    }

    std::cout << "Embedding tests passed.\n";
    return EXIT_SUCCESS;
}
