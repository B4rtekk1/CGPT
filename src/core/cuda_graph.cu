#include "core/cuda_graph.h"

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>
#include <utility>

namespace {

[[noreturn]] void throw_cuda_error(
    const cudaError_t status,
    const char* const operation
) {
    throw std::runtime_error(
        std::string(operation) + " failed: " + cudaGetErrorString(status)
    );
}

void check_cuda(
    const cudaError_t status,
    const char* const operation
) {
    if (status != cudaSuccess) {
        throw_cuda_error(status, operation);
    }
}

void destroy_graph_noexcept(cudaGraph_t& graph) noexcept {
    if (graph != nullptr) {
        static_cast<void>(cudaGraphDestroy(graph));
        graph = nullptr;
    }
}

void destroy_executable_noexcept(cudaGraphExec_t& executable) noexcept {
    if (executable != nullptr) {
        static_cast<void>(cudaGraphExecDestroy(executable));
        executable = nullptr;
    }
}

} // namespace

CudaGraph::CudaGraph(CudaGraph&& other) noexcept
    : graph_(std::exchange(other.graph_, nullptr)),
      executable_(std::exchange(other.executable_, nullptr)) {}

CudaGraph& CudaGraph::operator=(CudaGraph&& other) noexcept {
    if (this != &other) {
        reset();
        graph_ = std::exchange(other.graph_, nullptr);
        executable_ = std::exchange(other.executable_, nullptr);
    }
    return *this;
}

CudaGraph::~CudaGraph() noexcept {
    reset();
}

void CudaGraph::capture(
    const cudaStream_t stream,
    const std::function<void()>& capture_body,
    const cudaStreamCaptureMode mode
) {
    if (!capture_body) {
        throw std::invalid_argument("CudaGraph::capture: capture callback is empty");
    }

    reset();

    check_cuda(
        cudaStreamBeginCapture(stream, mode),
        "cudaStreamBeginCapture"
    );

    try {
        capture_body();
    } catch (...) {
        cudaGraph_t abandoned_graph = nullptr;
        const cudaError_t end_status =
            cudaStreamEndCapture(stream, &abandoned_graph);

        destroy_graph_noexcept(abandoned_graph);

        // Clear a possible sticky error caused by invalidated capture. The
        // original C++ exception remains the most useful error to propagate.
        if (end_status != cudaSuccess) {
            static_cast<void>(cudaGetLastError());
        }
        throw;
    }

    cudaGraph_t captured_graph = nullptr;
    check_cuda(
        cudaStreamEndCapture(stream, &captured_graph),
        "cudaStreamEndCapture"
    );

    if (captured_graph == nullptr) {
        throw std::runtime_error(
            "CudaGraph::capture: CUDA returned an empty graph"
        );
    }

    try {
        instantiate(captured_graph);
    } catch (...) {
        destroy_graph_noexcept(captured_graph);
        throw;
    }
}

void CudaGraph::instantiate(cudaGraph_t graph) {
    if (graph == nullptr) {
        throw std::invalid_argument("CudaGraph::instantiate: graph is null");
    }

    reset();

    cudaGraphExec_t executable = nullptr;

#if CUDART_VERSION >= 12000
    const cudaError_t status = cudaGraphInstantiate(
        &executable,
        graph,
        0
    );
#else
    const cudaError_t status = cudaGraphInstantiate(
        &executable,
        graph,
        nullptr,
        nullptr,
        0
    );
#endif

    if (status != cudaSuccess) {
        destroy_executable_noexcept(executable);
        throw_cuda_error(status, "cudaGraphInstantiate");
    }

    graph_ = graph;
    executable_ = executable;
}

void CudaGraph::launch(const cudaStream_t stream) const {
    if (executable_ == nullptr) {
        throw std::logic_error(
            "CudaGraph::launch: graph has not been instantiated"
        );
    }

    check_cuda(
        cudaGraphLaunch(executable_, stream),
        "cudaGraphLaunch"
    );
}

void CudaGraph::upload(const cudaStream_t stream) const {
    if (executable_ == nullptr) {
        throw std::logic_error(
            "CudaGraph::upload: graph has not been instantiated"
        );
    }

    check_cuda(
        cudaGraphUpload(executable_, stream),
        "cudaGraphUpload"
    );
}

void CudaGraph::reset() noexcept {
    destroy_executable_noexcept(executable_);
    destroy_graph_noexcept(graph_);
}

bool CudaGraph::initialized() const noexcept {
    return executable_ != nullptr;
}

cudaGraph_t CudaGraph::graph() const noexcept {
    return graph_;
}

cudaGraphExec_t CudaGraph::executable() const noexcept {
    return executable_;
}
