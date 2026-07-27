#pragma once

#include "../core/tensor.h"

/**
 * @brief Applies Root Mean Square Layer Normalization to a preallocated tensor.
 *
 * For every row of @p input, the operation computes
 * `output[i] = input[i] * rsqrt(mean(input[i]^2) + epsilon) * weight`.
 *
 * @param output CUDA destination tensor with the same shape and dtype as
 *        @p input.
 * @param input CUDA source tensor with shape `[rows, hidden]`.
 * @param weight CUDA scale tensor with shape `[hidden]` and the same dtype as
 *        @p input.
 * @param epsilon Positive, finite numerical stability term.
 * @param stream CUDA stream receiving the asynchronous kernel launch.
 *
 * @throws std::invalid_argument If a tensor is not CUDA-resident, dtypes do
 *         not match, shapes are incompatible, or @p epsilon is not finite and
 *         positive.
 *
 * @note This function does not synchronize @p stream.
 */
void rmsnorm_forward(
    Tensor& output,
    const Tensor& input,
    const Tensor& weight,
    float epsilon,
    cudaStream_t stream = nullptr
);

/**
 * @brief Allocates an output tensor and applies Root Mean Square Layer
 *        Normalization.
 *
 * The output inherits the shape, CUDA device and dtype of @p input. See the
 * preallocated-output overload for the operation formula and requirements.
 *
 * @param input CUDA source tensor with shape `[rows, hidden]`.
 * @param weight CUDA scale tensor with shape `[hidden]`.
 * @param epsilon Positive, finite numerical stability term.
 * @param stream CUDA stream receiving the asynchronous kernel launch.
 * @return A newly allocated normalized tensor.
 * @throws std::invalid_argument If the RMSNorm input contract is not met.
 */
Tensor rmsnorm_forward(
    const Tensor& input,
    const Tensor& weight,
    float epsilon,
    cudaStream_t stream = nullptr
);
