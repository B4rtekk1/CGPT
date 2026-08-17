/**
 * @file cuda_graph(1).cu
 * @brief RAII implementation for CUDA graph capture and execution.
 */

#include "core/cuda_graph.h"

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>
#include <utility>

namespace {
    /** @brief Throws a descriptive C++ exception for a CUDA runtime error. */
    [[noreturn]] void throw_cuda_error(
        const cudaError_t status,
        const char *const operation
    ) {
        throw std::runtime_error(
            std::string(operation) + " failed: " + cudaGetErrorString(status)
        );
    }

    /** @brief Checks a CUDA status and throws when it is not cudaSuccess. */
    void check_cuda(
        const cudaError_t status,
        const char *const operation
    ) {
        if (status != cudaSuccess) {
            throw_cuda_error(status, operation);
        }
    }

    /** @brief Destroys a CUDA graph handle without throwing. */
    void destroy_graph_noexcept(cudaGraph_t &graph) noexcept {
        if (graph != nullptr) {
            static_cast<void>(cudaGraphDestroy(graph));
            graph = nullptr;
        }
    }

    /** @brief Destroys an executable CUDA graph handle without throwing. */
    void destroy_executable_noexcept(cudaGraphExec_t &executable) noexcept {
        if (executable != nullptr) {
            static_cast<void>(cudaGraphExecDestroy(executable));
            executable = nullptr;
        }
    }
} // namespace

/** @brief Transfers ownership of graph and executable handles. */
CudaGraph::CudaGraph(CudaGraph &&other) noexcept
    : graph_(std::exchange(other.graph_, nullptr)),
      executable_(std::exchange(other.executable_, nullptr)) {
}

/** @brief Releases current handles and takes ownership from another object. */
CudaGraph &CudaGraph::operator=(CudaGraph &&other) noexcept {
    if (this != &other) {
        reset();
        graph_ = std::exchange(other.graph_, nullptr);
        executable_ = std::exchange(other.executable_, nullptr);
    }
    return *this;
}

/** @brief Destroys the owned CUDA graph resources. */
CudaGraph::~CudaGraph() noexcept {
    reset();
}

/**
 * @brief Captures stream work and instantiates the resulting CUDA graph.
 *
 * The previous graph is reset before capture. If the callback throws, capture
 * is ended best-effort, the abandoned graph is destroyed, and the original
 * exception is rethrown.
 *
 * @param stream Stream whose operations should be captured.
 * @param capture_body Callback that enqueues operations during capture.
 * @param mode CUDA stream capture mode.
 * @throws std::invalid_argument If the callback is empty or stream is null.
 * @throws std::runtime_error If CUDA capture or graph instantiation fails.
 */
void CudaGraph::capture(
    cudaStream_t stream,
    const std::function<void()> &capture_body,
    const cudaStreamCaptureMode mode
) {
    if (!capture_body) {
        throw std::invalid_argument("CudaGraph::capture: capture callback is empty");
    }
    if (stream == nullptr) {
        throw std::invalid_argument("CudaGraph::capture: stream must not be null");
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

/**
 * @brief Instantiates an executable CUDA graph from a graph handle.
 * @param graph Captured CUDA graph to own.
 * @throws std::invalid_argument If @p graph is null.
 * @throws std::runtime_error If CUDA graph instantiation fails.
 */
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

/**
 * @brief Launches the instantiated graph on a CUDA stream.
 * @param stream Destination CUDA stream.
 * @throws std::logic_error If no executable graph has been instantiated.
 * @throws std::runtime_error If cudaGraphLaunch fails.
 */
void CudaGraph::launch(cudaStream_t stream) const {
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

/**
 * @brief Uploads the executable graph to the device ahead of launch.
 * @param stream Stream used for graph upload.
 * @throws std::logic_error If no executable graph has been instantiated.
 * @throws std::runtime_error If cudaGraphUpload fails.
 */
void CudaGraph::upload(cudaStream_t stream) const {
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

/** @brief Destroys both graph and executable handles, if present. */
void CudaGraph::reset() noexcept {
    destroy_executable_noexcept(executable_);
    destroy_graph_noexcept(graph_);
}

/** @brief Indicates whether an executable graph is currently available. */
bool CudaGraph::initialized() const noexcept {
    return executable_ != nullptr;
}

/** @brief Returns the owned capture graph handle without transferring ownership. */
cudaGraph_t CudaGraph::graph() const noexcept {
    return graph_;
}

/** @brief Returns the owned executable graph handle without transferring ownership. */
cudaGraphExec_t CudaGraph::executable() const noexcept {
    return executable_;
}
