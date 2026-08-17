#pragma once

#include "core/tensor.h"
#include "core/device_buffer.h"

#include <cstdint>
#include <limits>
#include <span>
#include <vector>

/** @brief Settings for decoupled Adam weight decay (AdamW). */
struct AdamWOptions {
    /** @brief Learning rate applied to each update. */
    float learning_rate = 1.0e-3f;
    /** @brief Exponential decay rate for the first moment. */
    float beta1 = 0.9f;
    /** @brief Exponential decay rate for the second moment. */
    float beta2 = 0.999f;
    /** @brief Numerical stability constant. */
    float epsilon = 1.0e-8f;
    /** @brief Decoupled weight-decay coefficient. */
    float weight_decay = 1.0e-2f;
    /** Divisor for gradients produced from a loss multiplied by this value. */
    float loss_scale = 1.0f;
    /** @brief Per-tensor L2 clipping threshold; infinity disables clipping. */
    float max_grad_norm = std::numeric_limits<float>::infinity();
};

/**
 * @brief Per-parameter AdamW state.
 *
 * All optimizer state and master weights use F32 storage.
 */
struct AdamWState {
    /** @brief F32 master copy of the optimized parameter. */
    Tensor master_parameter;
    /** @brief First-moment estimate. */
    Tensor first_moment;
    /** @brief Second-moment estimate. */
    Tensor second_moment;
    /** @brief Number of optimizer updates applied to this state. */
    std::uint64_t step = 0;

    /** @brief Creates optimizer state matching a parameter tensor. */
    [[nodiscard]] static AdamWState for_parameter(
        const Tensor& parameter, cudaStream_t stream = nullptr);
};

/** @brief Parameter, gradient, and state participating in a batched update. */
struct AdamWBatchEntry {
    /** @brief Parameter tensor updated in-place. */
    Tensor* parameter = nullptr;
    /** @brief Gradient tensor associated with the parameter. */
    const Tensor* gradient = nullptr;
    /** @brief AdamW state associated with the parameter. */
    AdamWState* state = nullptr;
};

/** Reusable device-side reductions for a batched AdamW update. */
class AdamWWorkspace {
public:
    /** @brief Constructs an empty optimizer workspace. */
    AdamWWorkspace();
    AdamWWorkspace(const AdamWWorkspace&) = delete;
    AdamWWorkspace& operator=(const AdamWWorkspace&) = delete;
    AdamWWorkspace(AdamWWorkspace&&) noexcept = default;
    AdamWWorkspace& operator=(AdamWWorkspace&&) noexcept = default;

private:
    friend void adamw_step_many_async(std::span<const AdamWBatchEntry>,
                                      const AdamWOptions&, AdamWWorkspace&, cudaStream_t);
    friend bool adamw_check(const AdamWWorkspace&, cudaStream_t);
    DeviceBuffer non_finite_;
    DeviceBuffer squared_norm_;
};

/**
 * @brief Enqueues one global-norm AdamW update for all entries.
 *
 * The function
 * synchronizes the preceding gradient-statistics phase to reject NaN/Inf
 * gradients before changing host-side step counters; the parameter-update
 * kernels themselves remain asynchronous.  The caller must keep entries and
 * tensors alive until stream work completes.
 *
 * When the batch contains a NaN or Inf, no update is enqueued and no state
 * counter is advanced.  Use adamw_check() to retrieve that result.
 *
 * @param entries Parameter entries participating in the update.
 * @param options Optimizer hyperparameters.
 * @param workspace Reusable reduction workspace.
 * @param stream CUDA stream used for the operation.
 */
void adamw_step_many_async(
    std::span<const AdamWBatchEntry> entries,
    const AdamWOptions& options,
    AdamWWorkspace& workspace,
    cudaStream_t stream = nullptr);

/**
 * @brief Synchronizes the stream and reports whether the last batch was finite.
 * @param workspace Workspace containing the non-finite flag.
 * @param stream CUDA stream to synchronize.
 * @return `true` if the last batch contained only finite gradients.
 */
[[nodiscard]] bool adamw_check(const AdamWWorkspace& workspace,
                               cudaStream_t stream = nullptr);

/**
 * @brief Performs one AdamW update in-place and advances @p state.
 *
 * Parameters and gradients must have the same F32, F16, or BF16 dtype, shape,
 * and device. Moment buffers and master weights are always F32. Gradients are
 * unscaled by @c options.loss_scale before clipping and updating. If a NaN or
 * Inf is found, no state is changed and false is returned. The weight-decay
 * term is decoupled from the gradient, as in AdamW.
 *
 * @param parameter Parameter tensor to update.
 * @param gradient Gradient tensor for @p parameter.
 * @param state Optimizer state associated with @p parameter.
 * @param options Optimizer hyperparameters.
 * @param stream CUDA stream used for the update.
 * @return `true` if the update was applied; `false` for non-finite gradients.
 */
[[nodiscard]] bool adamw_step(
    Tensor& parameter,
    const Tensor& gradient,
    AdamWState& state,
    const AdamWOptions& options = {},
    cudaStream_t stream = nullptr);