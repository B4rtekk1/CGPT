#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <string_view>
#include <vector>

namespace {

struct TestResult {
    std::string_view name;
    int exit_code;
    std::chrono::duration<double> duration;
};

struct KernelBenchmark {
    std::string name;
    double average_ms;
};

std::string quote_command_argument(const std::string_view argument) {
    std::string quoted{"\""};
    for (const char character : argument) {
        if (character == '\"') {
            quoted += "\\\"";
        } else {
            quoted += character;
        }
    }
    quoted += '\"';
    return quoted;
}

int normalize_exit_code(const int status) {
#ifdef _WIN32
    return status;
#else
    if (status == -1) {
        return status;
    }
    return (status >> 8) & 0xff;
#endif
}

const char* status_name(const int exit_code) {
    if (exit_code == 0) {
        return "PASS";
    }
    if (exit_code == 77) {
        return "SKIP";
    }
    return "FAIL";
}

bool set_benchmark_results_file(const std::filesystem::path& path) {
#ifdef _WIN32
    return _putenv_s("CGPT_BENCHMARK_RESULTS_FILE", path.string().c_str()) == 0;
#else
    return setenv("CGPT_BENCHMARK_RESULTS_FILE", path.c_str(), 1) == 0;
#endif
}

void clear_benchmark_results_file() {
#ifdef _WIN32
    _putenv_s("CGPT_BENCHMARK_RESULTS_FILE", "");
#else
    unsetenv("CGPT_BENCHMARK_RESULTS_FILE");
#endif
}

std::vector<KernelBenchmark> read_kernel_benchmarks(const std::filesystem::path& path) {
    std::ifstream input(path);
    std::vector<KernelBenchmark> benchmarks;
    std::string line;
    while (std::getline(input, line)) {
        const std::size_t separator = line.find('\t');
        if (separator == std::string::npos) {
            continue;
        }
        try {
            benchmarks.push_back({line.substr(0, separator), std::stod(line.substr(separator + 1))});
        } catch (const std::exception&) {
            // Ignore incomplete records if a test terminates while writing one.
        }
    }
    return benchmarks;
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 3 || (argc - 1) % 2 != 0) {
        std::cerr << "Usage: " << argv[0]
                  << " <test-name> <test-executable> [<test-name> <test-executable> ...]\n";
        return 2;
    }

    const auto results_file = std::filesystem::temp_directory_path() /
                              ("cgpt_kernel_benchmarks_" +
                               std::to_string(std::chrono::steady_clock::now()
                                                  .time_since_epoch()
                                                  .count()) +
                               ".tsv");
    std::error_code remove_error;
    std::filesystem::remove(results_file, remove_error);
    if (!set_benchmark_results_file(results_file)) {
        std::cerr << "Cannot enable kernel benchmark collection\n";
        return EXIT_FAILURE;
    }

    std::vector<TestResult> results;
    results.reserve(static_cast<std::size_t>((argc - 1) / 2));

    for (int argument = 1; argument < argc; argument += 2) {
        const std::string_view name = argv[argument];
        const std::string_view executable = argv[argument + 1];
        std::cout << "\n== " << name << " ==\n" << std::flush;

        const auto start = std::chrono::steady_clock::now();
        const int exit_code = normalize_exit_code(
            std::system(quote_command_argument(executable).c_str()));
        const auto finish = std::chrono::steady_clock::now();
        results.push_back({name, exit_code, finish - start});
    }

    std::cout << "\n========== Test benchmark summary ==========\n";
    std::cout << std::left << std::setw(30) << "Test"
              << std::setw(10) << "Status"
              << "Duration (s)\n";

    bool failed = false;
    double total_seconds = 0.0;
    for (const TestResult& result : results) {
        const double seconds = result.duration.count();
        total_seconds += seconds;
        std::cout << std::left << std::setw(30) << result.name
                  << std::setw(10) << status_name(result.exit_code)
                  << std::fixed << std::setprecision(3) << seconds << '\n';
        failed = failed || (result.exit_code != 0 && result.exit_code != 77);
    }
    std::cout << "Total duration: " << std::fixed << std::setprecision(3)
              << total_seconds << " s\n";

    clear_benchmark_results_file();
    const std::vector<KernelBenchmark> kernel_benchmarks =
        read_kernel_benchmarks(results_file);
    std::filesystem::remove(results_file, remove_error);

    std::cout << "\n========== Kernel benchmark summary ==========\n";
    if (kernel_benchmarks.empty()) {
        std::cout << "No CUDA kernel benchmarks were reported.\n";
    } else {
        std::cout << std::left << std::setw(30) << "Kernel"
                  << "Average time (ms)\n";
        double total_average_ms = 0.0;
        for (const KernelBenchmark& benchmark : kernel_benchmarks) {
            total_average_ms += benchmark.average_ms;
            std::cout << std::left << std::setw(30) << benchmark.name
                      << std::fixed << std::setprecision(6)
                      << benchmark.average_ms << '\n';
        }
        std::cout << "Mean of kernel averages: " << std::fixed << std::setprecision(6)
                  << total_average_ms / static_cast<double>(kernel_benchmarks.size())
                  << " ms\n";
    }

    return failed ? EXIT_FAILURE : EXIT_SUCCESS;
}
