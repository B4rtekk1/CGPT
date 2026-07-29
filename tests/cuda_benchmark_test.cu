#include "cuda_benchmark.h"

#include "core/cuda_check.h"

#include <cassert>
#include <iostream>
#include <sstream>
#include <string>

namespace {

void test_benchmark_reports_average_launch_time() {
    int* value = nullptr;
    CUDA_CHECK(cudaMalloc(&value, sizeof(*value)));

    std::ostringstream output;
    auto* const previous_output = std::cout.rdbuf(output.rdbuf());
    try {
        test::benchmark_cuda_launches("cudaMemsetAsync", [value](cudaStream_t stream) {
            CUDA_CHECK(cudaMemsetAsync(value, 0, sizeof(*value), stream));
        });
    } catch (...) {
        std::cout.rdbuf(previous_output);
        cudaFree(value);
        throw;
    }
    std::cout.rdbuf(previous_output);

    CUDA_CHECK(cudaFree(value));

    const std::string report = output.str();
    if (test::benchmark_results_are_collected()) {
        assert(report.empty());
        return;
    }
    assert(report.starts_with("cudaMemsetAsync: "));
    assert(report.find(" ms (average of 10000 launches; 100 warmup launches)\n") != std::string::npos);
}

} // namespace

int main() {
    test_benchmark_reports_average_launch_time();
    std::cout << "CUDA benchmark tests passed.\n";
    return 0;
}
