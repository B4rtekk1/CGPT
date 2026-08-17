#pragma once

#include "core/tensor.h"

/**
 * @brief Computes a numerically stable row-wise softmax.
 *
 * @p input and @p output
 * must be CUDA floating-point tensors of identical shape [rows, columns] and
 * dtype (F32, F16, or BF16).  Each row is processed by one CUDA block, with
 * FP32 reductions and an in-register logit-to-probability pipeline.
 *
 * @param output Destination tensor containing row-wise probabilities.
 * @param input Source tensor containing row-wise logits.
 * @param stream CUDA stream used for the operation.
 */
void softmax_forward(
    Tensor& output,
    const Tensor& input,
    cudaStream_t stream = nullptr);