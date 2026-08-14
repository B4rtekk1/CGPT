#pragma once

#include "core/tensor.h"

#include <cstdint>
#include <limits>

/** Settings for decoupled Adam weight decay (AdamW). */
struct AdamWOptions {
    float learning_rate = 1.0e-3f;
    float beta1 = 0.9f;
    float beta2 = 0.999f;
    float epsilon = 1.0e-8f;
    float weight_decay = 1.0e-2f;
    /** Divisor for gradients produced from a loss multiplied by this value. */
    float loss_scale = 1.0f;
    /** Per-tensor L2 clipping threshold; infinity disables clipping. */
    float max_grad_norm = std::numeric_limits<float>::infinity();
};

/** Per-parameter AdamW state. All optimizer state and master weights use F32. */
struct AdamWState {
    Tensor master_parameter;
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
 * and device. Moment buffers and master weights are always F32. Gradients are
 * unscaled by @c options.loss_scale before clipping and updating. If a NaN or
 * Inf is found, no state is changed and false is returned. The weight-decay
 * term is decoupled from the gradient, as in AdamW.
 */
[[nodiscard]] bool adamw_step(
    Tensor& parameter,
    const Tensor& gradient,
    AdamWState& state,
    const AdamWOptions& options = {},
    cudaStream_t stream = nullptr);
