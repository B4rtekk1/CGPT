#include <cuda_runtime_api.h>
#include <cupti_activity.h>

#include <cstdlib>
#include <fstream>
#include <map>
#include <mutex>
#include <string>

namespace {

constexpr std::size_t kActivityBufferSize = 8U * 1024U * 1024U;

struct KernelStatistics {
    std::uint64_t total_nanoseconds = 0;
    std::uint64_t launches = 0;
};

std::mutex statistics_mutex;
std::map<std::string, KernelStatistics> statistics;

void CUPTIAPI request_activity_buffer(
    std::uint8_t** buffer,
    std::size_t* size,
    std::size_t* max_records
) {
    *buffer = static_cast<std::uint8_t*>(std::malloc(kActivityBufferSize));
    *size = *buffer == nullptr ? 0 : kActivityBufferSize;
    *max_records = 0;
}

void record_kernel(const CUpti_ActivityKernel12& kernel) {
    if (kernel.name == nullptr || kernel.end <= kernel.start) {
        return;
    }

    std::lock_guard lock(statistics_mutex);
    KernelStatistics& entry = statistics[kernel.name];
    entry.total_nanoseconds += kernel.end - kernel.start;
    ++entry.launches;
}

void CUPTIAPI complete_activity_buffer(
    CUcontext,
    std::uint32_t,
    std::uint8_t* buffer,
    std::size_t,
    std::size_t valid_size
) {
    CUpti_Activity* record = nullptr;
    while (cuptiActivityGetNextRecord(buffer, valid_size, &record) == CUPTI_SUCCESS) {
        if (record->kind == CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL ||
            record->kind == CUPTI_ACTIVITY_KIND_KERNEL) {
            record_kernel(*reinterpret_cast<const CUpti_ActivityKernel12*>(record));
        }
    }
    std::free(buffer);
}

std::string tsv_safe_name(const std::string& name) {
    std::string safe = name;
    for (char& character : safe) {
        if (character == '\t' || character == '\n' || character == '\r') {
            character = ' ';
        }
    }
    return safe;
}

class KernelProfiler {
public:
    KernelProfiler() {
        const char* const path = std::getenv("CGPT_KERNEL_PROFILE_RESULTS_FILE");
        if (path == nullptr || *path == '\0') {
            return;
        }
        output_path_ = path;

        if (cuptiActivityRegisterCallbacks(request_activity_buffer, complete_activity_buffer) != CUPTI_SUCCESS) {
            return;
        }
        enabled_ = cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL) == CUPTI_SUCCESS;
    }

    ~KernelProfiler() {
        if (!enabled_) {
            return;
        }

        // CUPTI only emits completed records. Synchronize before the final flush so
        // the report includes launches still queued when a test exits.
        cudaDeviceSynchronize();
        cuptiActivityFlushAll(CUPTI_ACTIVITY_FLAG_FLUSH_FORCED);
        cuptiActivityDisable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL);

        std::lock_guard lock(statistics_mutex);
        std::ofstream output(output_path_, std::ios::app);
        if (!output) {
            return;
        }
        for (const auto& [name, entry] : statistics) {
            if (entry.launches == 0) {
                continue;
            }
            const double average_ms = static_cast<double>(entry.total_nanoseconds) /
                                      static_cast<double>(entry.launches) / 1'000'000.0;
            output << "CUPTI: " << tsv_safe_name(name) << '\t'
                   << average_ms << '\n';
        }
    }

private:
    bool enabled_ = false;
    std::string output_path_;
};

// Every test executable receives this translation unit. The collector remains
// dormant for ordinary test runs and is enabled only by the run_all target.
KernelProfiler kernel_profiler;

} // namespace
