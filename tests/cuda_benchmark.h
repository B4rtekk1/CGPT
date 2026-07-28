#pragma once

#include "core/cuda_check.h"

#include <iostream>

namespace test {

template <typename Launch>
void benchmark_cuda_launches(const char* name, Launch&& launch) {
    constexpr int kWarmupIterations = 100;
    constexpr int kMeasuredIterations = 10000;

    cudaStream_t stream = nullptr;
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    try {
        for (int iteration = 0; iteration < kWarmupIterations; ++iteration) {
            launch(stream);
        }
        CUDA_CHECK(cudaStreamSynchronize(stream));

        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));
        CUDA_CHECK(cudaEventRecord(start, stream));
        for (int iteration = 0; iteration < kMeasuredIterations; ++iteration) {
            launch(stream);
        }
        CUDA_CHECK(cudaEventRecord(stop, stream));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float elapsed_ms = 0.0F;
        CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
        std::cout << name << ": "
                  << elapsed_ms / static_cast<float>(kMeasuredIterations)
                  << " ms (average of " << kMeasuredIterations
                  << " launches; " << kWarmupIterations << " warmup launches)\n";
    } catch (...) {
        if (stop != nullptr) {
            cudaEventDestroy(stop);
        }
        if (start != nullptr) {
            cudaEventDestroy(start);
        }
        cudaStreamDestroy(stream);
        throw;
    }
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaStreamDestroy(stream));
}

} // namespace test
