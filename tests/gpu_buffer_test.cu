#include "core/gpu_buffer.h"

#include <cstdlib>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {

template <typename Exception, typename Function>
void expect_throw(Function&& function, const char* message) {
    try {
        function();
    } catch (const Exception&) {
        return;
    }
    throw std::runtime_error(message);
}

void test_invalid_copies() {
    GpuBuffer<int> buffer(2);

    expect_throw<std::invalid_argument>(
        [&] { buffer.copy_from_host(nullptr, 1); },
        "GpuBuffer accepted a null source");
    expect_throw<std::invalid_argument>(
        [&] { buffer.copy_to_host(nullptr, 1); },
        "GpuBuffer accepted a null destination");
    expect_throw<std::out_of_range>(
        [&] { buffer.copy_from_host(std::vector<int>{1, 2, 3}.data(), 3); },
        "GpuBuffer accepted an oversized source");
    expect_throw<std::out_of_range>(
        [&] { buffer.copy_to_host(std::vector<int>(3).data(), 3); },
        "GpuBuffer accepted an oversized destination");
}

void test_reallocation_and_release() {
    GpuBuffer<float> buffer(2);
    if (buffer.size() != 2 || buffer.empty()) {
        throw std::runtime_error("GpuBuffer allocation metadata is incorrect");
    }

    buffer.allocate(4);
    if (buffer.size() != 4 || buffer.empty()) {
        throw std::runtime_error("GpuBuffer reallocation failed");
    }

    buffer.allocate(0);
    if (buffer.size() != 0 || !buffer.empty() || buffer.data() != nullptr) {
        throw std::runtime_error("GpuBuffer zero allocation did not release storage");
    }
}

void test_allocation_overflow() {
    expect_throw<std::invalid_argument>(
        [] {
            GpuBuffer<int> buffer(
                std::numeric_limits<std::size_t>::max() / sizeof(int) + 1);
        },
        "GpuBuffer accepted an overflowing allocation");
}

} // namespace

int main() {
    try {
        test_invalid_copies();
        test_reallocation_and_release();
        test_allocation_overflow();
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return EXIT_FAILURE;
    }

    std::cout << "GpuBuffer tests passed.\n";
    return EXIT_SUCCESS;
}
