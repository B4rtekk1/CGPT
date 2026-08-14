/**
 * @file rmsnorm_backward.cu
 * @brief CUDA backward pass for RMSNorm.
 */

#include "ops/backward/rmsnorm_backward.h"
#include "core/cuda_check.h"
#include "core/device_guard.h"

#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include <limits>
#include <stdexcept>
#include <cstdint>
#include <unordered_map>

namespace {
    __forceinline__ __device__ float warp_reduce_sum(float value) {
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            value += __shfl_down_sync(0xffffffff, value, offset);
        }
        return value;
    }

    __forceinline__ __device__ float block_reduce_sum(float value) {
        value = warp_reduce_sum(value);
        __shared__ float warp_sums[8];
        const int lane = static_cast<int>(threadIdx.x) & 31;
        const int warp = static_cast<int>(threadIdx.x) >> 5;
        if (lane == 0) {
            warp_sums[warp] = value;
        }
        __syncthreads();

        if (warp == 0) {
            const int warp_count = (static_cast<int>(blockDim.x) + 31) >> 5;
            value = lane < warp_count ? warp_sums[lane] : 0.0f;
            value = warp_reduce_sum(value);
            if (lane == 0) {
                warp_sums[0] = value;
            }
        }
        __syncthreads();
        return warp_sums[0];
    }

    template<typename T>
    __launch_bounds__(256)
    __global__ void rmsnorm_grad_input_kernel(
        T * __restrict__ grad_input,
        const T * __restrict__ grad_output,
        const T * __restrict__ input,
        const T * __restrict__ weight,
        float * __restrict__ inv_rms,
        const int hidden,
        const float inverse_hidden,
        const float epsilon
    ) {
        const int column = static_cast<int>(threadIdx.x);
        const std::size_t row_offset = static_cast<std::size_t>(blockIdx.x) * hidden;
        const T *row_input = input + row_offset;
        const T *row_grad_output = grad_output + row_offset;
        T *row_grad_input = grad_input + row_offset;

        float sum_squares = 0.0f;
        for (int index = column; index < hidden; index += static_cast<int>(blockDim.x)) {
            const float x = static_cast<float>(row_input[index]);
            sum_squares = fmaf(x, x, sum_squares);
        }
        __shared__ float shared_inv_rms;
        const float square_sum = block_reduce_sum(sum_squares);
        if (threadIdx.x == 0) {
            shared_inv_rms = rsqrtf(square_sum * inverse_hidden + epsilon);
            inv_rms[blockIdx.x] = shared_inv_rms;
        }
        __syncthreads();

        float weighted_dot = 0.0f;
        for (int index = column; index < hidden; index += static_cast<int>(blockDim.x)) {
            const auto x = static_cast<float>(row_input[index]);
            const float dy_weight = static_cast<float>(row_grad_output[index]) *
                                    static_cast<float>(weight[index]);
            weighted_dot = fmaf(dy_weight, x, weighted_dot);
        }

        const float row_inv_rms = shared_inv_rms;
        const float correction = block_reduce_sum(weighted_dot) * inverse_hidden *
                                 row_inv_rms * row_inv_rms * row_inv_rms;

        for (int index = column; index < hidden; index += static_cast<int>(blockDim.x)) {
            const auto x = static_cast<float>(row_input[index]);
            const float dy_weight = static_cast<float>(row_grad_output[index]) *
                                    static_cast<float>(weight[index]);
            row_grad_input[index] = static_cast<T>(fmaf(-x, correction, dy_weight * row_inv_rms));
        }
    }

    template<typename T>
    __launch_bounds__(256)
    __global__ void rmsnorm_grad_weight_kernel(
        T * __restrict__ grad_weight,
        const T * __restrict__ grad_output,
        const T * __restrict__ input,
        const float * __restrict__ inv_rms,
        const int rows,
        const int hidden
    ) {
        const int column = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x)
                         + static_cast<int>(threadIdx.x);
        if (column >= hidden) return;

        // Adjacent lanes own adjacent columns, so every iteration issues
        // coalesced row reads.  The former one-CTA-per-column mapping made
        // adjacent lanes read rows separated by `hidden` elements.
        float grad = 0.0f;
        for (int row = 0; row < rows; ++row) {
            const std::size_t row_offset = static_cast<std::size_t>(row) * hidden;
            grad = fmaf(static_cast<float>(grad_output[row_offset + column]),
                        static_cast<float>(input[row_offset + column]) * inv_rms[row], grad);
        }
        grad_weight[column] = static_cast<T>(grad);
    }

    void validate_rmsnorm_backward(
        const Tensor &grad_input,
        const Tensor &grad_weight,
        const Tensor &grad_output,
        const Tensor &input,
        const Tensor &weight,
        const float epsilon
    ) {
        if (!std::isfinite(epsilon) || epsilon <= 0.0f) {
            throw std::invalid_argument("rmsnorm_backward: epsilon must be finite and positive");
        }
        const Tensor *tensors[] = {&grad_input, &grad_weight, &grad_output, &input, &weight};
        for (const Tensor *tensor: tensors) {
            if (tensor->device_type() != DeviceType::CUDA) {
                throw std::invalid_argument("rmsnorm_backward: all tensors must be CUDA tensors");
            }
            if (!is_floating_point(tensor->dtype()) || tensor->dtype() != input.dtype()) {
                throw std::invalid_argument(
                    "rmsnorm_backward: all tensors must have the same floating dtype");
            }
        }
        if (input.dim() != 2 || grad_output.shape() != input.shape() ||
            grad_input.shape() != input.shape()) {
            throw std::invalid_argument(
                "rmsnorm_backward: input, grad_output and grad_input must have shape [rows, hidden]");
        }
        if (weight.dim() != 1 || grad_weight.dim() != 1 ||
            weight.size(0) != input.size(1) || grad_weight.size(0) != input.size(1)) {
            throw std::invalid_argument(
                "rmsnorm_backward: weight and grad_weight must have shape [hidden]");
        }
        if (input.size(0) > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
            input.size(1) > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
            throw std::invalid_argument("rmsnorm_backward: dimensions exceed supported range");
        }
    }

    template<typename T>
    void launch_rmsnorm_backward(
        Tensor &grad_input, Tensor &grad_weight, const Tensor &grad_output,
        const Tensor &input, const Tensor &weight, const int rows, const int hidden,
        const float epsilon, cudaStream_t stream
    ) {
        constexpr int threads = 256;
        const float inverse_hidden = 1.0f / static_cast<float>(hidden);
        // Workspace is cached per host thread and CUDA stream.  This keeps the
        // operation asynchronous after its first use instead of paying for a
        // synchronizing cudaMalloc/cudaFree pair on every backward call.
        thread_local std::unordered_map<std::uintptr_t, DeviceBuffer> inv_rms_buffers;
        DeviceBuffer& inv_rms_buffer =
            inv_rms_buffers[reinterpret_cast<std::uintptr_t>(stream)];
        const std::size_t required_bytes = static_cast<std::size_t>(rows) * sizeof(float);
        if (inv_rms_buffer.bytes() < required_bytes) {
            inv_rms_buffer.allocate(required_bytes);
        }
        auto *inv_rms = static_cast<float *>(inv_rms_buffer.data());
        rmsnorm_grad_input_kernel<T><<<rows, threads, 0, stream>>>(
            static_cast<T *>(grad_input.raw_data()), static_cast<const T *>(grad_output.raw_data()),
            static_cast<const T *>(input.raw_data()), static_cast<const T *>(weight.raw_data()),
            inv_rms, hidden, inverse_hidden, epsilon);
        const int weight_blocks = (hidden + threads - 1) / threads;
        rmsnorm_grad_weight_kernel<T><<<weight_blocks, threads, 0, stream>>>(
            static_cast<T *>(grad_weight.raw_data()), static_cast<const T *>(grad_output.raw_data()),
            static_cast<const T *>(input.raw_data()), inv_rms, rows, hidden);
    }
}

void rmsnorm_backward(
    Tensor& grad_input, Tensor& grad_weight, const Tensor& grad_output,
    const Tensor& input, const Tensor& weight, const float epsilon, cudaStream_t stream) {
    validate_rmsnorm_backward(grad_input, grad_weight, grad_output, input, weight, epsilon);
    DeviceGuard device_guard(0);
    const int rows = static_cast<int>(input.size(0));
    const int hidden = static_cast<int>(input.size(1));
    if (input.dtype() == Dtype::F16) {
        launch_rmsnorm_backward<__half>(grad_input, grad_weight, grad_output, input, weight,
                                        rows, hidden, epsilon, stream);
    } else if (input.dtype() == Dtype::BF16) {
        launch_rmsnorm_backward<__nv_bfloat16>(grad_input, grad_weight, grad_output, input,
                                               weight, rows, hidden, epsilon, stream);
    } else {
        launch_rmsnorm_backward<float>(grad_input, grad_weight, grad_output, input, weight,
                                       rows, hidden, epsilon, stream);
    }
    CUDA_CHECK(cudaGetLastError());
}
