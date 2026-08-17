#pragma once

#include "ops/linear.h"

#include <cstddef>

/**
 * @brief Controls whether linear backward overwrites or accumulates gradients.
 */
struct LinearBackwardOptions {
    /** @brief Arithmetic mode used by cuBLASLt. */
    ComputeType compute_type = ComputeType::F32;
    /** @brief Maximum temporary workspace size in bytes. */
    std::size_t workspace_bytes = 32ULL * 1024ULL * 1024ULL;

    /** @brief Adds to the input gradient instead of overwriting it. */
    bool accumulate_input = false;
    /** @brief Adds to the weight gradient instead of overwriting it. */
    bool accumulate_weight = false;
    /** @brief Adds to the bias gradient instead of overwriting it. */
    bool accumulate_bias = false;
};

/**
 * @brief Computes the backward pass for a linear layer.
 * @param grad_input Gradient of the input tensor.
 * @param grad_weight Gradient of the weight tensor.
 * @param grad_bias Gradient of the bias tensor.
 * @param grad_output Gradient of the output tensor.
 * @param input Input tensor.
 * @param weight Weight tensor.
 * @param cublas_lt_context cuBLASLt context used by the matrix operations.
 * @param stream CUDA stream used for the operation.
 * @param options Linear backward configuration.
 */
void linear_backward(
    Tensor& grad_input,
    Tensor& grad_weight,
    Tensor& grad_bias,
    const Tensor& grad_output,
    const Tensor& input,
    const Tensor& weight,
    const CublasLtContext& cublas_lt_context,
    cudaStream_t stream = nullptr,
    const LinearBackwardOptions& options = {}
);