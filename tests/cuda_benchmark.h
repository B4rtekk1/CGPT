#pragma once

#include "core/cuda_check.h"

#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>

namespace test {

inline bool benchmark_results_are_collected() {
    return std::getenv("CGPT_BENCHMARK_RESULTS_FILE") != nullptr;
}

inline void report_cuda_benchmark(
    const char* name,
    const float average_ms,
    const int measured_iterations,
    const int warmup_iterations
) {
    if (const char* const results_file = std::getenv("CGPT_BENCHMARK_RESULTS_FILE")) {
        std::ofstream output(results_file, std::ios::app);
        if (output) {
            output << name << '\t' << std::fixed << std::setprecision(6)
                   << average_ms << '\n';
            return;
        }
    }

    std::cout << name << ": " << average_ms
              << " ms (average of " << measured_iterations
              << " launches; " << warmup_iterations << " warmup launches)\n";
}

template <typename Launch>
void benchmark_cuda_launches(
    const char* name,
    Launch&& launch,
    const int measured_iterations = 10000,
    const int warmup_iterations = 100
) {

    cudaStream_t stream = nullptr;
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    try {
        for (int iteration = 0; iteration < warmup_iterations; ++iteration) {
            launch(stream);
        }
        CUDA_CHECK(cudaStreamSynchronize(stream));

        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));
        CUDA_CHECK(cudaEventRecord(start, stream));
        for (int iteration = 0; iteration < measured_iterations; ++iteration) {
            launch(stream);
        }
        CUDA_CHECK(cudaEventRecord(stop, stream));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float elapsed_ms = 0.0F;
        CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
        report_cuda_benchmark(
            name,
            elapsed_ms / static_cast<float>(measured_iterations),
            measured_iterations,
            warmup_iterations);
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
