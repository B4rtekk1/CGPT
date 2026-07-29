#include "core/cuda_check.h"
#include "core/device_buffer.h"
#include "core/kv_cache.h"

#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <utility>
#include <vector>

namespace {

void expect(const bool condition, const char* const message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

template <typename Exception, typename Function>
void expect_throw(Function&& function, const char* const message) {
    try {
        std::forward<Function>(function)();
    } catch (const Exception&) {
        return;
    }
    throw std::runtime_error(message);
}

std::vector<__half> to_half(const std::vector<float>& values) {
    std::vector<__half> result(values.size());
    for (std::size_t index = 0; index < values.size(); ++index) {
        result[index] = __float2half(values[index]);
    }
    return result;
}

DeviceBuffer copy_to_device(const std::vector<__half>& values) {
    DeviceBuffer buffer(values.size() * sizeof(__half));
    CUDA_CHECK(cudaMemcpy(
        buffer.data(), values.data(), buffer.bytes(), cudaMemcpyHostToDevice));
    return buffer;
}

std::vector<float> read_device(const __half* const source, const std::size_t count) {
    std::vector<__half> host_values(count);
    CUDA_CHECK(cudaMemcpy(
        host_values.data(), source, count * sizeof(__half), cudaMemcpyDeviceToHost));

    std::vector<float> values(count);
    for (std::size_t index = 0; index < count; ++index) {
        values[index] = __half2float(host_values[index]);
    }
    return values;
}

void expect_values(
    const std::vector<float>& actual,
    const std::vector<float>& expected,
    const char* const message
) {
    expect(actual.size() == expected.size(), message);
    for (std::size_t index = 0; index < actual.size(); ++index) {
        expect(actual[index] == expected[index], message);
    }
}

void test_validation() {
    expect_throw<std::invalid_argument>(
        [] {
            const KVCache cache(KVCacheConfig{});
            static_cast<void>(cache);
        },
        "KVCache accepted an invalid configuration");

    KVCache cache({1, 2, 4, 1, 2});
    expect_throw<std::out_of_range>(
        [&] { static_cast<void>(cache.key_sequence(1, 0)); },
        "KVCache accepted an invalid layer");
    expect_throw<std::out_of_range>(
        [&] { static_cast<void>(cache.value_sequence(0, 2)); },
        "KVCache accepted an invalid batch");
    expect_throw<std::out_of_range>(
        [&] { cache.advance_sequence_length(0, 0, 5); },
        "KVCache allowed a sequence length beyond its capacity");
}

void test_write_layout() {
    KVCache cache({2, 2, 4, 1, 2});
    const auto keys = to_half({1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F, 7.0F, 8.0F});
    const auto values = to_half({11.0F, 12.0F, 13.0F, 14.0F, 15.0F, 16.0F, 17.0F, 18.0F});
    const DeviceBuffer device_keys = copy_to_device(keys);
    const DeviceBuffer device_values = copy_to_device(values);

    cache.write(
        1,
        2,
        1,
        2,
        static_cast<const __half*>(device_keys.data()),
        static_cast<const __half*>(device_values.data()));
    CUDA_CHECK(cudaDeviceSynchronize());

    expect_values(
        read_device(cache.key_sequence(1, 0), 8),
        {0.0F, 0.0F, 1.0F, 2.0F, 3.0F, 4.0F, 0.0F, 0.0F},
        "KVCache stored the first batch in an unexpected location");
    expect_values(
        read_device(cache.value_sequence(1, 1), 8),
        {0.0F, 0.0F, 15.0F, 16.0F, 17.0F, 18.0F, 0.0F, 0.0F},
        "KVCache stored the second batch in an unexpected location");
}

void test_append_and_move() {
    KVCache cache({1, 2, 4, 1, 2});
    const auto first = to_half({1.0F, 2.0F, 3.0F, 4.0F});
    const auto second = to_half({5.0F, 6.0F, 7.0F, 8.0F});
    const DeviceBuffer device_first = copy_to_device(first);
    const DeviceBuffer device_second = copy_to_device(second);

    cache.append(
        0,
        2,
        1,
        static_cast<const __half*>(device_first.data()),
        static_cast<const __half*>(device_first.data()));
    cache.append(
        0,
        2,
        1,
        static_cast<const __half*>(device_second.data()),
        static_cast<const __half*>(device_second.data()));
    CUDA_CHECK(cudaDeviceSynchronize());

    expect(cache.sequence_length(0, 0) == 2 && cache.sequence_length(0, 1) == 2,
           "KVCache did not advance sequence lengths after append");
    expect_values(
        read_device(cache.key_sequence(0, 1), 4),
        {3.0F, 4.0F, 7.0F, 8.0F},
        "KVCache append stored values in an unexpected location");

    KVCache moved(std::move(cache));
    expect(moved.sequence_length(0, 0) == 2,
           "KVCache move construction lost sequence lengths");
}

} // namespace

int main() {
    try {
        test_validation();
        test_write_layout();
        test_append_and_move();
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return EXIT_FAILURE;
    }

    std::cout << "KV cache tests passed.\n";
    return EXIT_SUCCESS;
}
