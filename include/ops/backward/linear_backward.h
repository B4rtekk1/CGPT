#pragma once

#include "ops/linear.h"

#include <cstddef>

/**
 * @brief Controls whether linear backward overwrites or accumulate gradients.
 */
struct LinearBackwardOptions {
    ComputeType compute_type = ComputeType::F32;
    std::size_t workspace_bytes = 32ULL * 1024ULL * 1024ULL;

    bool accumulate_input = false;
    bool accumulate_weight = false;
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
 * @param cublas_lt_context
 * @param stream CUDA stream
 * @param options LinearBackward options
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
