#pragma once

#include "core/tensor.h"

/** @brief Controls accumulation behavior for SwiGLU backward outputs. */
struct SwiGLUBackwardOptions {
    /** @brief Adds to grad_gate instead of overwriting it. */
    bool accumulate_gate = false;
    /** @brief Adds to grad_up instead of overwriting it. */
    bool accumulate_up = false;
    /** @brief Controls accumulation for a packed gate/up gradient, if used. */
    bool accumulate_gate_up = false;
};

/**
 * @brief Computes gradients of the SwiGLU activation.
 *
 * For separate gate and up projections, the operation computes gradients with
 * respect to both inputs from the output gradient. Output buffers are either
 * overwritten or accumulated according to @p options.
 *
 * @param grad_gate Gradient with respect to the gate projection.
 * @param grad_up Gradient with respect to the up projection.
 * @param grad_output Gradient with respect to the SwiGLU output.
 * @param gate Forward-pass gate projection.
 * @param up Forward-pass up projection.
 * @param stream CUDA stream used for the operation.
 * @param options Gradient accumulation options.
 */
void swiglu_backward(
    Tensor& grad_gate,
    Tensor& grad_up,
    const Tensor& grad_output,
    const Tensor& gate,
    const Tensor& up,
    cudaStream_t stream = nullptr,
    const SwiGLUBackwardOptions& options = {}
);