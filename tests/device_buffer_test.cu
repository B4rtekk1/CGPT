#include "cuda_check.h"
#include "device_buffer.h"

#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <utility>
#include <vector>

namespace {

void expect(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void test_empty_buffer() {
    const DeviceBuffer buffer;

    expect(buffer.empty(), "Default buffer should be empty");
    expect(buffer.data() == nullptr, "Default buffer should have null data");
    expect(buffer.bytes() == 0, "Default buffer should have zero bytes");
}

void test_allocate_and_copy() {
    constexpr std::size_t byte_count = 8 * sizeof(int);
    const std::vector<int> expected{0, 1, 2, 3, 4, 5, 6, 7};
    std::vector<int> actual(expected.size());

    DeviceBuffer buffer(byte_count);
    expect(!buffer.empty(), "Allocated buffer should not be empty");
    expect(buffer.data() != nullptr, "Allocated buffer should have data");
    expect(buffer.bytes() == byte_count, "Allocated buffer has wrong size");

    CUDA_CHECK(cudaMemcpy(
        buffer.data(), expected.data(), byte_count, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(
        actual.data(), buffer.data(), byte_count, cudaMemcpyDeviceToHost));
    expect(actual == expected, "DeviceBuffer data was not copied correctly");
}

void test_zero_allocation_releases_buffer() {
    DeviceBuffer buffer(sizeof(float));
    expect(!buffer.empty(), "Buffer allocation failed");

    buffer.allocate(0);

    expect(buffer.empty(), "Zero allocation should release the buffer");
    expect(buffer.data() == nullptr, "Released buffer should have null data");
    expect(buffer.bytes() == 0, "Released buffer should have zero bytes");
}

void test_move_operations() {
    DeviceBuffer source(sizeof(float));
    const void* source_data = source.data();

    DeviceBuffer moved(std::move(source));
    expect(moved.data() == source_data, "Move construction lost the allocation");
    expect(source.empty(), "Moved-from buffer should be empty");

    DeviceBuffer assigned;
    assigned = std::move(moved);
    expect(assigned.data() == source_data, "Move assignment lost the allocation");
    expect(moved.empty(), "Move-assigned buffer should be empty");
}

} // namespace

int main() {
    try {
        test_empty_buffer();
        test_allocate_and_copy();
        test_zero_allocation_releases_buffer();
        test_move_operations();
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return EXIT_FAILURE;
    }

    std::cout << "DeviceBuffer tests passed.\n";
    return EXIT_SUCCESS;
}
