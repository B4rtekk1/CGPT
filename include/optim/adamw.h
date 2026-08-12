#pragma once

#include <cstdint>

#include "core/tensor.h"

/** Settings for decoupled Adam weight decay (AdamW). */
struct AdamWOptions {
    float learning_rate = 1.0e-3f;
    float beta1 = 0.9f;
    float beta2 = 0.999f;
    float epsilon = 1.0e-8f;
    float weight_decay = 1.0e-2f;
};

/** Per-parameter AdamW state. Moment buffers are kept in F32. */
struct AdamWState {
    Tensor first_moment;
    Tensor second_moment;
    std::uint64_t step = 0;

    [[nodiscard]] static AdamWState for_parameter(
        const Tensor& parameter, cudaStream_t stream = nullptr);
};

/**
 * Performs one AdamW update in-place and advances @p state.
 *
 * Parameters and gradients must have the same F32, F16, or BF16 dtype, shape,
 * and device. Moment buffers are always F32. The weight-decay term is
 * decoupled from the gradient, as in AdamW.
 */
void adamw_step(
    Tensor& parameter,
    const Tensor& gradient,
    AdamWState& state,
    const AdamWOptions& options = {},
    cudaStream_t stream = nullptr);
