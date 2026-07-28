#pragma once

#include <cuda_runtime.h>

#include <functional>
/**
 * @brief RAII wrapper around CUDA stream capture and executable CUDA graphs.
 *
 * A CudaGraph records a fixed sequence of CUDA operations submitted to a
 * stream and replays that sequence with a single cudaGraphLaunch call.
 *
 * The captured operations keep the device-pointer arguments used during
 * capture. Tensor storage and other referenced CUDA resources therefore must
 * remain alive and at stable addresses for as long as the executable graph is
 * used.
 */
class CudaGraph {
public:
    CudaGraph() noexcept = default;

    CudaGraph(const CudaGraph&) = delete;
    CudaGraph& operator=(const CudaGraph&) = delete;

    CudaGraph(CudaGraph&& graph) noexcept;
    CudaGraph& operator=(CudaGraph&& graph) noexcept;

    ~CudaGraph() noexcept;

    /**
     * @brief Captures CUDA work submitted by @p capture_body to @p stream.
     *
     * Existing graph state is destroyed before capture begins. The callback must enqueue CUDA work on the
     * supplied stream and must not synchronize that stream. cuBLASLt plans and all temporary buffers
     * should be created before calling this function.
     *
     * @param stream Stream on which capture is performed. Must not be null when legacy-default-stream
     * semantics would make capture invalid.
     * @param capture_body Callback that enqueues the operations to capture
     * @param mode CUDA stream capture mode.
     *
     * @throws std::invalid_argument If the callback is empty.
     * @throws std::runtime_error If CUDA capture or graph instantiation fails.
     */
    void capture(
        const cudaStream_t stream,
        const std::function<void()>& capture_body,
        const cudaStreamCaptureMode mode = cudaStreamCaptureModeThreadLocal
    );

    /**
     * @brief Instantiates an already created CUDA graph.
     *
     * This is useful when graph construction is performed manually with the CUDA graph API instead of stream capture.
     * Ownership of @p graph is transferred to this object on success.
     *
     * @param graph CUDA graph whose ownership is transferred
     * @throws std::invalid_argument If @p graph is null
     * @throws std::runtime_error If graph instantiation failed
     */
    void instantiate(cudaGraph_t graph);

    /**
     * @brief Enqueues the executable graph on @p stream.
     *
     * The function is asynchronous with respect to the host
     *
     * @trows std::logic_error If o executable graph is avaible.
     * @throws std:;runtime_error If cudaGraphLaunch fails.
     */
    void launch(cudaStream_t stream) const;

    /**
     * @brief Uploads graph nodes to the device before the first launch.
     *
     * Uploading is optional. It can move part of the first-launch setup cost
     * outside a latency-sensitive region.
     *
     * @throws std::logic_error If no executable graph is available.
     * @throws std::runtime_error If cudaGraphUpload fails.
     */
    void upload(cudaStream_t stream) const;

    /** @brief Destroys the executable and source graph, if present. */
    void reset() noexcept;

    [[nodiscard]] bool initialized() const noexcept;
    [[nodiscard]] cudaGraph_t graph() const noexcept;
    [[nodiscard]] cudaGraphExec_t executable() const noexcept;

private:
    cudaGraph_t graph_ = nullptr;
    cudaGraphExec_t executable_ = nullptr;
};
