#include "core/cuda_check.h"
#include "core/cuda_graph.h"

#include <cstdlib>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <utility>

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

class Stream {
public:
    Stream() {
        CUDA_CHECK(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking));
    }

    Stream(const Stream&) = delete;
    Stream& operator=(const Stream&) = delete;

    ~Stream() noexcept {
        if (stream_ != nullptr) {
            static_cast<void>(cudaStreamDestroy(stream_));
        }
    }

    [[nodiscard]] cudaStream_t get() const noexcept { return stream_; }

private:
    cudaStream_t stream_ = nullptr;
};

class DeviceValue {
public:
    DeviceValue() {
        CUDA_CHECK(cudaMalloc(&data_, sizeof(*data_)));
    }

    DeviceValue(const DeviceValue&) = delete;
    DeviceValue& operator=(const DeviceValue&) = delete;

    ~DeviceValue() noexcept {
        if (data_ != nullptr) {
            static_cast<void>(cudaFree(data_));
        }
    }

    [[nodiscard]] int* get() const noexcept { return data_; }

    int read() const {
        int value = 0;
        CUDA_CHECK(cudaMemcpy(&value, data_, sizeof(value), cudaMemcpyDeviceToHost));
        return value;
    }

private:
    int* data_ = nullptr;
};

void capture_memset(CudaGraph& graph, const cudaStream_t stream, int* const value,
                    const unsigned char pattern) {
    graph.capture(stream, [=] {
        CUDA_CHECK(cudaMemsetAsync(value, pattern, sizeof(*value), stream));
    });
}

void test_default_state_and_validation() {
    CudaGraph graph;

    expect(!graph.initialized(), "Default graph should not be initialized");
    expect(graph.graph() == nullptr, "Default graph handle should be null");
    expect(graph.executable() == nullptr, "Default executable handle should be null");

    expect_throw<std::logic_error>([&] { graph.launch(nullptr); },
                                   "Launching an empty graph should fail");
    expect_throw<std::logic_error>([&] { graph.upload(nullptr); },
                                   "Uploading an empty graph should fail");
    expect_throw<std::invalid_argument>([&] { graph.capture(nullptr, {}); },
                                        "Capturing an empty callback should fail");
    expect_throw<std::invalid_argument>([&] { graph.instantiate(nullptr); },
                                        "Instantiating a null graph should fail");

    graph.reset();
    expect(!graph.initialized(), "Resetting an empty graph should be safe");
}

void test_capture_upload_launch_and_reset() {
    Stream stream;
    DeviceValue value;
    CudaGraph graph;

    capture_memset(graph, stream.get(), value.get(), 0x2A);
    expect(graph.initialized(), "Captured graph should be initialized");
    expect(graph.graph() != nullptr, "Captured graph handle should not be null");
    expect(graph.executable() != nullptr, "Captured executable should not be null");

    graph.upload(stream.get());
    graph.launch(stream.get());
    CUDA_CHECK(cudaStreamSynchronize(stream.get()));
    expect(value.read() == 0x2A2A2A2A, "Captured graph produced an unexpected value");

    graph.reset();
    expect(!graph.initialized(), "Reset should release the executable graph");
    expect(graph.graph() == nullptr, "Reset should release the source graph");
    expect(graph.executable() == nullptr, "Reset should clear the executable handle");
}

void test_recapture_and_exception_recovery() {
    Stream stream;
    DeviceValue value;
    CudaGraph graph;

    capture_memset(graph, stream.get(), value.get(), 0x11);
    capture_memset(graph, stream.get(), value.get(), 0x22);
    expect(graph.graph() != nullptr, "Recapture should create a graph");

    graph.launch(stream.get());
    CUDA_CHECK(cudaStreamSynchronize(stream.get()));
    expect(value.read() == 0x22222222, "Recaptured graph was not launched");

    expect_throw<std::runtime_error>([&] {
        graph.capture(stream.get(), [] { throw std::runtime_error("capture failure"); });
    }, "Capture callback exception should propagate");
    expect(!graph.initialized(), "Failed capture should leave no initialized graph");

    capture_memset(graph, stream.get(), value.get(), 0x33);
    graph.launch(stream.get());
    CUDA_CHECK(cudaStreamSynchronize(stream.get()));
    expect(value.read() == 0x33333333, "Graph should recover after a failed capture");
}

void test_manual_instantiation_and_move_operations() {
    cudaGraph_t raw_graph = nullptr;
    CUDA_CHECK(cudaGraphCreate(&raw_graph, 0));

    CudaGraph graph;
    graph.instantiate(raw_graph);
    expect(graph.initialized(), "Manually instantiated graph should be initialized");
    expect(graph.graph() == raw_graph, "Graph should take ownership of the supplied handle");

    CudaGraph moved(std::move(graph));
    expect(!graph.initialized(), "Move construction should empty the source graph");
    expect(moved.graph() == raw_graph, "Move construction should retain the source graph");

    CudaGraph assigned;
    assigned = std::move(moved);
    expect(!moved.initialized(), "Move assignment should empty the source graph");
    expect(assigned.graph() == raw_graph, "Move assignment should retain the source graph");
    assigned.launch(nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());
}

} // namespace

int main() {
    try {
        test_default_state_and_validation();
        test_capture_upload_launch_and_reset();
        test_recapture_and_exception_recovery();
        test_manual_instantiation_and_move_operations();
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return EXIT_FAILURE;
    }

    std::cout << "CUDA graph tests passed.\n";
    return EXIT_SUCCESS;
}
