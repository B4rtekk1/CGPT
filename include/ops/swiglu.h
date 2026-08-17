#pragma once
#include "core/tensor.h"

/**
 * @brief Computes the SwiGLU activation from separate gate and up projections.
 *
 * The operation applies the SiLU activation to @p gate and multiplies it
 * element-wise by @p up. All tensors must have compatible shapes and storage
 * types supported by the implementation.
 *
 * @param output Destination tensor.
 * @param gate Gate projection tensor.
 * @param up Up projection tensor.
 * @param stream CUDA stream used for the operation.
 */
void swiglu_forward(
    Tensor& output,
    const Tensor& gate,
    const Tensor& up,
    cudaStream_t stream = nullptr
    );

/**
 * @brief Computes SwiGLU from a tensor containing concatenated projections.
 *
 * The input is expected to contain the gate and up projections in the layout
 * required by the implementation.
 *
 * @param output Destination tensor.
 * @param gate_up Tensor containing the concatenated gate and up projections.
 * @param stream CUDA stream used for the operation.
 */
void swiglu_forward(
    Tensor& output,
    const Tensor& gate_up,
    cudaStream_t stream = nullptr
    );