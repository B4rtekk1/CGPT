#include <cuda_runtime.h>
#include <iostream>
#include <vector>

#include "../include/core/cuda_check.h"

__global__ static void add_one(float* values, int count) {
    if (const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
        index < static_cast<unsigned int>(count)) {
        values[index] += 1.0f;
    }
}

int main() {
    constexpr int count = 1 << 20;

    std::vector<float> host_values(count, 2.0f);
    float* device_values = nullptr;

    CUDA_CHECK(cudaMalloc(&device_values, count * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(
        device_values,
        host_values.data(),
        count * sizeof(float),
        cudaMemcpyHostToDevice
    ));

    constexpr int threads_per_block = 256;
    constexpr int blocks =
        (count + threads_per_block - 1) / threads_per_block;

    add_one<<<blocks, threads_per_block>>>(device_values, count);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(
        host_values.data(),
        device_values,
        count * sizeof(float),
        cudaMemcpyDeviceToHost
    ));

    CUDA_CHECK(cudaFree(device_values));

    for (float value : host_values) {
        if (value != 3.0f) {
            std::cerr << "Test failed: expected 3.0, got "
                      << value << '\n';
            return 1;
        }
    }

    std::cout << "CUDA runtime works. GPU computed "
              << count << " values correctly.\n";
}
