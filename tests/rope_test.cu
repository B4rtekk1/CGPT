#include "core/cuda_check.h"
#include "ops/rope.h"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {
    void expect_close(const std::vector<float>& actual,
                      const std::vector<float>& expected,
                      const float tolerance) {
        if (actual.size() != expected.size()) {
            throw std::runtime_error("RoPE test: result size mismatch");
        }
        for (std::size_t i = 0; i < actual.size(); ++i) {
            if (!std::isfinite(actual[i]) ||
                std::fabs(actual[i] - expected[i]) > tolerance) {
                throw std::runtime_error("RoPE test: result value mismatch");
            }
        }
    }

    std::vector<float> read_tensor(const Tensor& tensor) {
        std::vector<float> values(tensor.numel());
        tensor.copy_to_host(values);
        return values;
    }

    void test_rotation_and_partial_dimension(const Dtype dtype, const float tolerance) {
        // [batch=1, sequence=2, heads=2, head_dim=4], rotating only the first pair.
        Tensor query({1, 2, 2, 4}, dtype);
        Tensor key({1, 2, 1, 4}, dtype);
        Tensor cosine({2, 1}, dtype);
        Tensor sine({2, 1}, dtype);
        const std::vector<float> q{
            1, 2, 30, 40,  3, 4, 50, 60,
            5, 6, 70, 80,  7, 8, 90, 100};
        const std::vector<float> k{1, -2, 30, 40, 5, -6, 70, 80};
        query.copy_from_host(q);
        key.copy_from_host(k);
        cosine.copy_from_host(std::vector<float>{1.0f, 0.0f});
        sine.copy_from_host(std::vector<float>{0.0f, 1.0f});

        rope_forward(query, key, cosine, sine, nullptr, RopeOptions{.rotary_dim = 2});
        CUDA_CHECK(cudaDeviceSynchronize());
        expect_close(read_tensor(query), {
            1, 2, 30, 40,  3, 4, 50, 60,
            -6, 5, 70, 80, -8, 7, 90, 100}, tolerance);
        expect_close(read_tensor(key), {1, -2, 30, 40, 6, 5, 70, 80}, tolerance);
    }

    template <typename Function>
    void expect_invalid_argument(Function&& function) {
        try {
            function();
        } catch (const std::invalid_argument&) {
            return;
        }
        throw std::runtime_error("RoPE test: invalid arguments were accepted");
    }

    void test_position_offset_and_validation() {
        Tensor query({1, 1, 1, 2});
        Tensor key({1, 1, 1, 2});
        Tensor cosine({2, 1});
        Tensor sine({2, 1});
        query.copy_from_host(std::vector<float>{2, 3});
        key.copy_from_host(std::vector<float>{4, 5});
        cosine.copy_from_host(std::vector<float>{1, 0});
        sine.copy_from_host(std::vector<float>{0, 1});
        rope_forward(query, key, cosine, sine, nullptr, RopeOptions{.position_offset = 1});
        CUDA_CHECK(cudaDeviceSynchronize());
        expect_close(read_tensor(query), {-3, 2}, 1.0e-5f);
        expect_close(read_tensor(key), {-5, 4}, 1.0e-5f);

        expect_invalid_argument([] {
            Tensor q({1, 1, 1, 3}); Tensor k({1, 1, 1, 3});
            Tensor c({1, 1}); Tensor s({1, 1});
            rope_forward(q, k, c, s, nullptr);
        });
        expect_invalid_argument([] {
            Tensor q({1, 2, 1, 4}); Tensor k({1, 2, 1, 4});
            Tensor c({2, 1}); Tensor s({2, 1});
            rope_forward(q, k, c, s, nullptr, RopeOptions{.rotary_dim = 2});
        });
        expect_invalid_argument([] {
            Tensor q({1, 1, 1, 4}, DeviceType::CPU);
            Tensor k({1, 1, 1, 4}, DeviceType::CPU);
            Tensor c({1, 2}, DeviceType::CPU);
            Tensor s({1, 2}, DeviceType::CPU);
            rope_forward(q, k, c, s, nullptr);
        });
    }
}

int main() {
    try {
        test_rotation_and_partial_dimension(Dtype::F32, 1.0e-5f);
        test_rotation_and_partial_dimension(Dtype::F16, 3.0e-3f);
        test_rotation_and_partial_dimension(Dtype::BF16, 2.0e-2f);
        test_position_offset_and_validation();
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return EXIT_FAILURE;
    }
    std::cout << "RoPE tests passed.\n";
    return EXIT_SUCCESS;
}
