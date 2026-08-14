#include "optim/adamw.h"
#include "core/cuda_check.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {
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
    __global__ void gradient_statistics_kernel(
        const T* gradient, std::size_t count, float loss_scale, unsigned int* non_finite, float* squared_norm) {
        const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
        if (index >= count) return;
        const float grad = static_cast<float>(gradient[index]) / loss_scale;
        if (!isfinite(grad)) {
            atomicExch(non_finite, 1U);
            return;
        }
        atomicAdd(squared_norm, grad * grad);
    }

    template <typename T>
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
