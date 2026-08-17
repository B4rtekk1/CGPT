#pragma once

#include "core/cublas_context.h"
#include "core/tensor.h"

/**
 * @brief Configuration for a cuBLASLt linear projection.
 */
struct LinearOptions {
    /** @brief Arithmetic mode used by cuBLASLt. */
    ComputeType compute_type = ComputeType::TF32;
    /** @brief Maximum temporary workspace size in bytes. */
    std::size_t workspace_bytes = 32U * 1024U * 1024U;
};

/**
 * @brief Computes a linear projection without bias.
 *
 * The operation evaluates the matrix product defined by @p input and
 * @p weight using the supplied cuBLASLt context.
 *
 * @param output Destination tensor.
 * @param input Input activation tensor.
 * @param weight Projection weight tensor.
 * @param cublas_context cuBLASLt context used for the matrix multiplication.
 * @param stream CUDA stream used for the operation.
 * @param options Linear-operation configuration.
 */
void linear_forward(
    Tensor& output,
    const Tensor& input,
    const Tensor& weight,
    const CublasLtContext& cublas_context,
    cudaStream_t stream = nullptr,
    const LinearOptions& options = {}
    );

/**
 * @brief Computes a linear projection with an added bias.
 *
 * @param output Destination tensor.
 * @param input Input activation tensor.
 * @param weight Projection weight tensor.
 * @param bias Bias tensor added to the projection result.
 * @param cublas_context cuBLASLt context used for the matrix multiplication.
 * @param stream CUDA stream used for the operation.
 * @param options Linear-operation configuration.
 */
void linear_forward(
    Tensor& output,
    const Tensor& input,
    const Tensor& weight,
    const Tensor& bias,
    const CublasLtContext& cublas_context,
    cudaStream_t stream = nullptr,
    const LinearOptions& options = {}
    );