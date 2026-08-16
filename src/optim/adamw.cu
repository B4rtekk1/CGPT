/**
 * @file adamw.cu
 * @brief AdamW optimizer implementation for CPU and CUDA.
 *
 * @details
 * This file implements AdamW parameter updates for both the CPU and GPU.
 * Gradients may use the parameter type (F32, F16, or BF16), while master
 * parameters and optimizer moments are stored in F32 to reduce precision loss
 * during training. The implementation also supports loss scaling, detection
 * of non-finite gradient values, and global gradient clipping.
 */

#include "optim/adamw.h"
#include "core/cuda_check.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {
    /**
     * @brief Validates AdamW hyperparameters.
     * @param options Parameter update options.
     * @throws std::invalid_argument If any option has an invalid value.
     */
    void validate_options(const AdamWOptions& options) {
        if (!std::isfinite(options.learning_rate) || options.learning_rate < 0.0f ||
        !std::isfinite(options.beta1) || options.beta1 < 0.0f || options.beta1 >= 1.0f ||
        !std::isfinite(options.beta2) || options.beta2 < 0.0f || options.beta2 >= 1.0f ||
        !std::isfinite(options.epsilon) || options.epsilon <= 0.0f ||
        !std::isfinite(options.weight_decay) || options.weight_decay < 0.0f ||
        !std::isfinite(options.loss_scale) || options.loss_scale <= 0.0f ||
        options.max_grad_norm <= 0.0f || std::isnan(options.max_grad_norm)) {
            throw std::invalid_argument("adamw_step: invalid optimizer options");
        }
    }

    /**
     * @brief Validates parameters, gradients, and optimizer state buffers.
     * @param parameter Model parameters being updated.
     * @param gradient Parameter gradients.
     * @param state AdamW master-parameter and moment buffers.
     * @throws std::invalid_argument If tensor types, shapes, or devices differ.
     */
    void validate_tensors(const Tensor& parameter, const Tensor& gradient, const AdamWState& state) {
        if (!is_floating_point(parameter.dtype()) || gradient.dtype() != parameter.dtype()) {
            throw std::invalid_argument("adamw_step: parameters and gradients must have the same floating dtype");
        }
        const Tensor* tensors[] = {&gradient, &state.master_parameter, &state.first_moment, &state.second_moment};
        for (const Tensor* tensor : tensors) {
            if (tensor->shape() != parameter.shape() || tensor->device_type() != parameter.device_type()) {
                throw std::invalid_argument("adamw_step: tensors must have matching shapes and devices");
            }
        }
        if (state.master_parameter.dtype() != Dtype::F32 || state.first_moment.dtype() != Dtype::F32 || state.second_moment.dtype() != Dtype::F32) {
            throw std::invalid_argument("adamw_step: master parameter and moment buffers must be F32");
        }
    }

    template <typename T>
    /**
     * @brief Computes the squared gradient norm and detects non-finite values.
     * @tparam T Gradient element type, such as float, __half, or bfloat16.
     * @param gradient Device gradient buffer.
     * @param count Number of gradient elements.
     * @param loss_scale Gradient scaling factor.
     * @param non_finite Atomically set when the gradient contains NaN or Inf.
     * @param squared_norm Squared gradient norm accumulator.
     */
    __global__ void gradient_statistics_kernel(
        const T* __restrict__ gradient, std::size_t count, float loss_scale,
        unsigned int* __restrict__ non_finite, float* __restrict__ squared_norm) {
        // Accumulate in shared memory first.  A global atomic for every element
        // serializes this otherwise bandwidth-bound kernel, especially for
        // large tensors.  One atomic per block keeps the result in float while
        // reducing the number of global atomics by roughly blockDim.x.
        __shared__ float block_sum[256];
        __shared__ unsigned int block_has_non_finite;

        if (threadIdx.x == 0) block_has_non_finite = 0;
        __syncthreads();

        const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
        float contribution = 0.0f;
        if (index < count) {
            const float grad = static_cast<float>(gradient[index]) / loss_scale;
            if (isfinite(grad)) {
                contribution = grad * grad;
            } else {
                atomicExch(&block_has_non_finite, 1U);
            }
        }
        block_sum[threadIdx.x] = contribution;
        __syncthreads();

        for (unsigned int stride = blockDim.x / 2; stride != 0; stride >>= 1) {
            if (threadIdx.x < stride) block_sum[threadIdx.x] += block_sum[threadIdx.x + stride];
            __syncthreads();
        }

        if (threadIdx.x == 0) {
            if (block_has_non_finite) atomicExch(non_finite, 1U);
            atomicAdd(squared_norm, block_sum[0]);
        }
    }

    template <typename T>
    /**
     * @brief Performs the AdamW update on the GPU.
     * @tparam T Parameter and gradient element type.
     * @param parameter Model parameters updated in place.
     * @param gradient Parameter gradients.
     * @param master_parameter F32 master parameters.
     * @param first_moment First gradient moment.
     * @param second_moment Second gradient moment.
     * @param count Number of parameters.
     * @param beta1 First-moment decay coefficient.
     * @param beta2 Second-moment decay coefficient.
     * @param bias_correction1 First-moment bias correction.
     * @param bias_correction2 Second-moment bias correction.
     * @param learning_rate Learning rate.
     * @param epsilon Denominator stability constant.
     * @param weight_decay Decoupled weight decay coefficient.
     * @param loss_scale Gradient unscaling factor.
     * @param gradient_scale Additional scale resulting from gradient clipping.
     */
    __global__ void adamw_kernel(
        T* parameter, const T* gradient, float* master_parameter, float* first_moment, float* second_moment, std::size_t count,
        float beta1, float beta2, float bias_correction1, float bias_correction2, float learning_rate, float epsilon, float weight_decay,
        float loss_scale, float gradient_scale) {
        const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
        if (index >= count) return;

        const float grad = static_cast<float>(gradient[index]) / loss_scale * gradient_scale;
        const float first = first_moment[index] = beta1 * first_moment[index] + (1.0f - beta1) * grad;
        const float second = second_moment[index] = beta2 * second_moment[index] + (1.0f - beta2) * grad * grad;
        const float normalized = first / bias_correction1 / (sqrtf(second / bias_correction2) + epsilon);
        const float updated = master_parameter[index] * (1.0f - learning_rate * weight_decay) - learning_rate * normalized;
        master_parameter[index] = updated;
        parameter[index] = static_cast<T>(updated);
    }

    /**
     * @brief Performs an AdamW step on the CPU.
     * @return false if the gradient contains NaN/Inf; otherwise true.
     */
    bool adamw_cpu(
    Tensor& parameter, const Tensor& gradient, AdamWState& state, const AdamWOptions& options,
    float bias_correction1, float bias_correction2) {
        std::vector<float> gradient_data(gradient.numel());
        gradient.copy_to_host(gradient_data);
        double squared_norm = 0.0;
        for (const float value : gradient_data) {
            const float grad = value / options.loss_scale;
            if (!std::isfinite(grad)) return false;
            squared_norm += static_cast<double>(grad) * grad;
        }
        const float norm = static_cast<float>(std::sqrt(squared_norm));
        const float gradient_scale = norm > options.max_grad_norm ? options.max_grad_norm / norm : 1.0f;
        std::vector<float> parameter_data(parameter.numel());
        state.master_parameter.copy_to_host(parameter_data);
        std::vector<float> master_data = parameter_data;
        auto* first_moment = static_cast<float*>(state.first_moment.raw_data());
        auto* second_moment = static_cast<float*>(state.second_moment.raw_data());

        for (std::size_t index = 0; index < parameter.numel(); ++index) {
            const float grad = gradient_data[index] / options.loss_scale * gradient_scale;
            const float first = first_moment[index] = options.beta1 * first_moment[index] + (1.0f - options.beta1) * grad;
            const float second = second_moment[index] = options.beta2 * second_moment[index] + (1.0f - options.beta2) * grad * grad;
            const float normalized = (first / bias_correction1) /
                                     (std::sqrt(second / bias_correction2) + options.epsilon);
            master_data[index] = master_data[index] * (1.0f - options.learning_rate * options.weight_decay) -
                                 options.learning_rate * normalized;
            parameter_data[index] = master_data[index];
        }
        state.master_parameter.copy_from_host(master_data);
        parameter.copy_from_host(parameter_data);
        return true;
    }

    template <typename T>
    /**
     * @brief Performs an AdamW step on a CUDA device.
     * @tparam T Parameter and gradient element type.
     * @return false if the gradient is invalid; otherwise true.
     */
    bool adamw_cuda(Tensor& parameter, const Tensor& gradient, AdamWState& state, const AdamWOptions& options,
                    float bias_correction1, float bias_correction2, cudaStream_t stream) {
        constexpr int threads = 256;
        const auto blocks = static_cast<unsigned int>((parameter.numel() + threads - 1) / threads);
        float gradient_scale = 1.0f;
        {
            // Release temporary buffers before launching the update, so their
            // synchronous cudaFree cannot serialize the update itself.
            DeviceBuffer non_finite_device(sizeof(unsigned int));
            DeviceBuffer squared_norm_device(sizeof(float));
            CUDA_CHECK(cudaMemsetAsync(non_finite_device.data(), 0, sizeof(unsigned int), stream));
            CUDA_CHECK(cudaMemsetAsync(squared_norm_device.data(), 0, sizeof(float), stream));
            gradient_statistics_kernel<T><<<blocks, threads, 0, stream>>>(
                static_cast<const T*>(gradient.raw_data()), parameter.numel(), options.loss_scale,
                static_cast<unsigned int*>(non_finite_device.data()), static_cast<float*>(squared_norm_device.data()));
            CUDA_CHECK(cudaGetLastError());
            unsigned int non_finite = 0;
            float squared_norm = 0.0f;
            CUDA_CHECK(cudaMemcpyAsync(&non_finite, non_finite_device.data(), sizeof(non_finite), cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK(cudaMemcpyAsync(&squared_norm, squared_norm_device.data(), sizeof(squared_norm), cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));
            if (non_finite != 0 || !std::isfinite(squared_norm)) return false;
            const float norm = sqrtf(squared_norm);
            gradient_scale = norm > options.max_grad_norm ? options.max_grad_norm / norm : 1.0f;
        }
        adamw_kernel<T><<<blocks, threads, 0, stream>>>(
            static_cast<T*>(parameter.raw_data()), static_cast<const T*>(gradient.raw_data()),
            static_cast<float*>(state.master_parameter.raw_data()), static_cast<float*>(state.first_moment.raw_data()),
            static_cast<float*>(state.second_moment.raw_data()), parameter.numel(), options.beta1, options.beta2,
            bias_correction1, bias_correction2, options.learning_rate, options.epsilon, options.weight_decay,
            options.loss_scale, gradient_scale);
        CUDA_CHECK(cudaGetLastError());
        return true;
    }
}

/**
 * @brief Creates a zero-initialized AdamW state matching the parameters.
 *
 * The master parameter is initialized as an F32 copy of the input parameters,
 * and both moment buffers are zero-initialized.
 *
 * @param parameter Parameter tensor for which the state is created.
 * @param stream CUDA stream used to initialize the buffers.
 * @return New optimizer state with its step counter set to zero.
 * @throws std::invalid_argument If the parameters are not floating-point.
 */
AdamWState AdamWState::for_parameter(const Tensor& parameter, cudaStream_t stream) {
    if (!is_floating_point(parameter.dtype())) {
        throw std::invalid_argument("AdamWState: parameter must have a floating dtype");
    }
    Tensor master_parameter(parameter.shape(), parameter.device_type(), Dtype::F32);
    std::vector<float> parameter_data(parameter.numel());
    parameter.copy_to_host(parameter_data);
    master_parameter.copy_from_host(parameter_data);
    return {
        std::move(master_parameter),
        Tensor::zeros(parameter.shape(), parameter.device_type(), stream, Dtype::F32),
        Tensor::zeros(parameter.shape(), parameter.device_type(), stream, Dtype::F32),
        0};
}

/**
 * @brief Performs one AdamW optimizer step.
 *
 * The implementation is selected automatically based on the parameter device.
 * Before the update, the gradient is unscaled, validated, and globally
 * clipped. The state step counter is incremented only after a successful
 * update.
 *
 * @param parameter Model parameters updated in place.
 * @param gradient Gradient corresponding to the parameters.
 * @param state AdamW state associated with the parameters.
 * @param options Optimizer hyperparameters.
 * @param stream CUDA stream used for GPU operations.
 * @return true if the parameters were updated; false if the gradient contains
 *         NaN/Inf.
 * @throws std::invalid_argument If argument types, shapes, or options are invalid.
 * @throws std::overflow_error If the step counter reached uint64_t maximum.
 */
bool adamw_step(Tensor& parameter, const Tensor& gradient, AdamWState& state,
                const AdamWOptions& options, cudaStream_t stream) {
    validate_options(options);
    validate_tensors(parameter, gradient, state);
    if (state.step == std::numeric_limits<std::uint64_t>::max()) {
        throw std::overflow_error("adamw_step: step counter overflow");
    }

    const std::uint64_t next_step = state.step + 1;
    const auto bias_correction1 = static_cast<float>(1.0 - std::pow(static_cast<double>(options.beta1), next_step));
    const auto bias_correction2 = static_cast<float>(1.0 - std::pow(static_cast<double>(options.beta2), next_step));

    if (parameter.device_type() == DeviceType::CPU) {
        const bool updated = adamw_cpu(parameter, gradient, state, options, bias_correction1, bias_correction2);
        if (updated) state.step = next_step;
        return updated;
    } else {
        const auto launch = [&]<typename T>() -> bool {
            return adamw_cuda<T>(parameter, gradient, state, options, bias_correction1, bias_correction2, stream);
        };
        bool updated = false;
        switch (parameter.dtype()) {
            case Dtype::F32: updated = launch.operator()<float>(); break;
            case Dtype::F16: updated = launch.operator()<__half>(); break;
            case Dtype::BF16: updated = launch.operator()<__nv_bfloat16>(); break;
            default: throw std::invalid_argument("adamw_step: unsupported dtype");
        }
        if (updated) state.step = next_step;
        return updated;
    }
}
