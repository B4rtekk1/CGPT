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
        !std::isfinite(options.weight_decay) || options.weight_decay < 0.0f) {
            throw std::invalid_argument("adamw_step: invalid optimizer options");
        }
    }

    void validate_tensors(const Tensor& parameter, const Tensor& gradient, const AdamWState& state) {
        if (!is_floating_point(parameter.dtype()) || gradient.dtype() != parameter.dtype()) {
            throw std::invalid_argument("adamw_step: parameters and gradients must have the same floating dtype");
        }
        const Tensor* tensors[] = {&gradient, &state.first_moment, &state.second_moment};
        for (const Tensor* tensor : tensors) {
            if (tensor->shape() != parameter.shape() || tensor->device_type() != parameter.device_type()) {
                throw std::invalid_argument("adamw_step: tensors must have matching shapes and devices");
            }
        }
        if (state.first_moment.dtype() != Dtype::F32 || state.second_moment.dtype() != Dtype::F32) {
            throw std::invalid_argument("adamw_step: moment buffers must be F32");
        }
    }

    template <typename T>
    __global__ void adamw_kernel(
        T* parameter, const T* gradient, float* first_moment, float* second_moment, std::size_t count,
        float beta1, float beta2, float bias_correction1, float bias_correction2, float learning_rate, float epsilon, float weight_decay) {
        const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
        if (index >= count) return;

        const float grad = static_cast<float>(gradient[index]);
        const float first = first_moment[index] = beta1 * first_moment[index] + (1.0f - beta1) * grad;
        const float second = second_moment[index] = beta2 * second_moment[index] + (1.0f - beta2) * grad * grad;
        const float normalized = (first / bias_correction1) / (sqrtf(second / bias_correction2) + epsilon);
        parameter[index] = static_cast<T>(static_cast<float>(parameter[index]) *
            (1.0f - learning_rate * weight_decay) - learning_rate * normalized);
    }

    void adamw_cpu(
    Tensor& parameter, const Tensor& gradient, AdamWState& state, const AdamWOptions& options,
    float bias_correction1, float bias_correction2) {
        std::vector<float> parameter_data(parameter.numel());
        std::vector<float> gradient_data(gradient.numel());
        parameter.copy_to_host(parameter_data);
        gradient.copy_to_host(gradient_data);
        auto* first_moment = static_cast<float*>(state.first_moment.raw_data());
        auto* second_moment = static_cast<float*>(state.second_moment.raw_data());

        for (std::size_t index = 0; index < parameter.numel(); ++index) {
            const float grad = gradient_data[index];
            const float first = first_moment[index] = options.beta1 * first_moment[index] + (1.0f - options.beta1) * grad;
            const float second = second_moment[index] = options.beta2 * second_moment[index] + (1.0f - options.beta2) * grad * grad;
            const float normalized = (first / bias_correction1) /
                                     (std::sqrt(second / bias_correction2) + options.epsilon);
            parameter_data[index] = parameter_data[index] * (1.0f - options.learning_rate * options.weight_decay) -
                                    options.learning_rate * normalized;
        }
        parameter.copy_from_host(parameter_data);
    }
}

AdamWState AdamWState::for_parameter(const Tensor& parameter, cudaStream_t stream) {
    if (!is_floating_point(parameter.dtype())) {
        throw std::invalid_argument("AdamWState: parameter must have a floating dtype");
    }
    return {
        Tensor::zeros(parameter.shape(), parameter.device_type(), stream, Dtype::F32),
        Tensor::zeros(parameter.shape(), parameter.device_type(), stream, Dtype::F32),
        0};
}

void adamw_step(Tensor& parameter, const Tensor& gradient, AdamWState& state,
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
        adamw_cpu(parameter, gradient, state, options, bias_correction1, bias_correction2);
    } else {
        constexpr int threads = 256;
        const auto blocks = static_cast<unsigned int>((parameter.numel() + threads - 1) / threads);
        const auto launch = [&]<typename T>() {
            adamw_kernel<T><<<blocks, threads, 0, stream>>>(
                static_cast<T*>(parameter.raw_data()), static_cast<const T*>(gradient.raw_data()),
                static_cast<float*>(state.first_moment.raw_data()), static_cast<float*>(state.second_moment.raw_data()),
                parameter.numel(), options.beta1, options.beta2, bias_correction1, bias_correction2,
                options.learning_rate, options.epsilon, options.weight_decay);
        };
        switch (parameter.dtype()) {
            case Dtype::F32: launch.template operator()<float>(); break;
            case Dtype::F16: launch.template operator()<__half>(); break;
            case Dtype::BF16: launch.template operator()<__nv_bfloat16>(); break;
            default: throw std::invalid_argument("adamw_step: unsupported dtype");
        }
        CUDA_CHECK(cudaGetLastError());
    }
    state.step = next_step;
}
