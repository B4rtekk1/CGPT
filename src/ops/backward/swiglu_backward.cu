/**
 * @file swiglu_backward.cu
 * @brief CUDA backward implementation of the SwiGLU activation.
 *
 * Both separated-input (`gate`, `up`) and fused-input (`gate_up`) layouts are supported.
 * FP16 and BF16 use packed two-element vector paths whenever the last dimension permits vector access.
 *
 */

#include "ops/backward/swiglu_backward.h"
#include "core/cuda_check.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

namespace {
    constexpr int THREADS_PER_BLOCK = 256;
    constexpr int MAX_BLOCKS_PER_ROW = 32;

    template<typename T>
    struct CudaTypeTraits;

    template<>
    struct CudaTypeTraits<float> {
        __device__ __forceinline__ static float load(const float value) {
            return value;
        }

        __device__ __forceinline__ static float store(const float value) {
            return value;
        }
    };

    template<>
    struct CudaTypeTraits<half> {
        __device__ __forceinline__ static float load(const half value) {
            return __half2float(value);
        }

        __device__ __forceinline__ static half store(const float value) {
            return __float2half(value);
        }
    };

    template<>
    struct CudaTypeTraits<__nv_bfloat16> {
        __device__ __forceinline__ static float load(const __nv_bfloat16 value) {
            return __bfloat162float(value);
        }

        __device__ __forceinline__ static __nv_bfloat16 store(const float value) {
            return __float2bfloat16_rn(value);
        }
    };

    __device__ __forceinline__ float sigmoid(const float value) {
        return 1.0f / (1.0f + expf(-value));
    }

    __device__ __forceinline__ float silu(const float value) {
        return value * sigmoid(value);
    }

    __device__ __forceinline__ float silu_derivative(const float value) {
        const float sigmoid_value = sigmoid(value);
        return sigmoid_value * (1.0f + value * (1.0f - sigmoid_value));
    }

    template<typename T, bool ACCUMULATE_GATE, bool ACCUMULATE_UP>
    __global__ void swiglu_separate_scalar_backward_kernel(
        T * __restrict__ grad_gate,
        T * __restrict__ grad_up,
        const T * __restrict__ grad_output,
        const T * __restrict__ gate,
        const T * __restrict__ up,
        const std::uint32_t element_count
    ) {
        const std::uint32_t stride = gridDim.x * blockDim.x;
        for (std::uint32_t index =
                     blockIdx.x * blockDim.x + threadIdx.x;
             index < element_count;
             index += stride) {
            const float grad_output_value =
                    CudaTypeTraits<T>::load(grad_output[index]);
            const float gate_value = CudaTypeTraits<T>::load(gate[index]);
            const float up_value = CudaTypeTraits<T>::load(up[index]);

            float grad_gate_value =
                    grad_output_value * up_value * silu_derivative(gate_value);
            float grad_up_value =
                    grad_output_value * silu(gate_value);

            if constexpr (ACCUMULATE_GATE) {
                grad_gate_value += CudaTypeTraits<T>::load(grad_gate[index]);
            }
            if constexpr (ACCUMULATE_UP) {
                grad_up_value += CudaTypeTraits<T>::load(grad_up[index]);
            }

            grad_gate[index] = CudaTypeTraits<T>::store(grad_gate_value);
            grad_up[index] = CudaTypeTraits<T>::store(grad_up_value);
        }
    }

    template<bool ACCUMULATE_GATE, bool ACCUMULATE_UP>
    __global__ void swiglu_separate_half2_backward_kernel(
        half2 * __restrict__ grad_gate,
        half2 * __restrict__ grad_up,
        const half2 * __restrict__ grad_output,
        const half2 * __restrict__ gate,
        const half2 * __restrict__ up,
        const std::uint32_t vector_count
    ) {
        const std::uint32_t stride = gridDim.x * blockDim.x;

        for (std::uint32_t index =
                     blockIdx.x * blockDim.x + threadIdx.x;
             index < vector_count;
             index += stride) {
            const float2 grad_output_values = __half22float2(grad_output[index]);
            const float2 gate_values = __half22float2(gate[index]);
            const float2 up_values = __half22float2(up[index]);

            float2 grad_gate_values{
                grad_output_values.x * up_values.x *
                silu_derivative(gate_values.x),
                grad_output_values.y * up_values.y *
                silu_derivative(gate_values.y)
            };
            float2 grad_up_values{
                grad_output_values.x * silu(gate_values.x),
                grad_output_values.y * silu(gate_values.y)
            };

            if constexpr (ACCUMULATE_GATE) {
                const float2 previous = __half22float2(grad_gate[index]);
                grad_gate_values.x += previous.x;
                grad_gate_values.y += previous.y;
            }
            if constexpr (ACCUMULATE_UP) {
                const float2 previous = __half22float2(grad_up[index]);
                grad_up_values.x += previous.x;
                grad_up_values.y += previous.y;
            }

            grad_gate[index] = __floats2half2_rn(
                grad_gate_values.x, grad_gate_values.y);
            grad_up[index] = __floats2half2_rn(
                grad_up_values.x, grad_up_values.y);
        }
    }

    template<bool ACCUMULATE_GATE, bool ACCUMULATE_UP>
    __global__ void swiglu_separate_bfloat162_backward_kernel(
        __nv_bfloat162* __restrict__ grad_gate,
        __nv_bfloat162* __restrict__ grad_up,
        const __nv_bfloat162* __restrict__ grad_output,
        const __nv_bfloat162* __restrict__ gate,
        const __nv_bfloat162* __restrict__ up,
        const std::uint32_t vector_count
    ) {
        const std::uint32_t stride = gridDim.x * blockDim.x;

        for (std::uint32_t index =
                 blockIdx.x * blockDim.x + threadIdx.x;
             index < vector_count;
             index += stride) {
            const float2 grad_output_values =
                __bfloat1622float2(grad_output[index]);
            const float2 gate_values = __bfloat1622float2(gate[index]);
            const float2 up_values = __bfloat1622float2(up[index]);

            float2 grad_gate_values{
                grad_output_values.x * up_values.x *
                    silu_derivative(gate_values.x),
                grad_output_values.y * up_values.y *
                    silu_derivative(gate_values.y)
            };
            float2 grad_up_values{
                grad_output_values.x * silu(gate_values.x),
                grad_output_values.y * silu(gate_values.y)
            };

            if constexpr (ACCUMULATE_GATE) {
                const float2 previous = __bfloat1622float2(grad_gate[index]);
                grad_gate_values.x += previous.x;
                grad_gate_values.y += previous.y;
            }
            if constexpr (ACCUMULATE_UP) {
                const float2 previous = __bfloat1622float2(grad_up[index]);
                grad_up_values.x += previous.x;
                grad_up_values.y += previous.y;
            }

            grad_gate[index] = __floats2bfloat162_rn(
                grad_gate_values.x, grad_gate_values.y);
            grad_up[index] = __floats2bfloat162_rn(
                grad_up_values.x, grad_up_values.y);
        }
    }

    template<typename T, bool ACCUMULATE>
    __global__ void swiglu_fused_scalar_backward_kernel(
        T* __restrict__ grad_gate_up,
        const T* __restrict__ grad_output,
        const T* __restrict__ gate_up,
        const std::uint32_t intermediate_size
    ) {
        const std::uint32_t row = blockIdx.y;
        const std::uint32_t output_row_offset = row * intermediate_size;
        const std::uint32_t input_row_offset = row * (2U * intermediate_size);

        for (std::uint32_t column =
                 blockIdx.x * blockDim.x + threadIdx.x;
             column < intermediate_size;
             column += gridDim.x * blockDim.x) {
            const std::uint32_t output_index = output_row_offset + column;
            const std::uint32_t gate_index = input_row_offset + column;
            const std::uint32_t up_index =
                input_row_offset + intermediate_size + column;

            const float grad_output_value =
                CudaTypeTraits<T>::load(grad_output[output_index]);
            const float gate_value =
                CudaTypeTraits<T>::load(gate_up[gate_index]);
            const float up_value =
                CudaTypeTraits<T>::load(gate_up[up_index]);

            float grad_gate_value =
                grad_output_value * up_value * silu_derivative(gate_value);
            float grad_up_value =
                grad_output_value * silu(gate_value);

            if constexpr (ACCUMULATE) {
                grad_gate_value +=
                    CudaTypeTraits<T>::load(grad_gate_up[gate_index]);
                grad_up_value +=
                    CudaTypeTraits<T>::load(grad_gate_up[up_index]);
            }

            grad_gate_up[gate_index] =
                CudaTypeTraits<T>::store(grad_gate_value);
            grad_gate_up[up_index] =
                CudaTypeTraits<T>::store(grad_up_value);
        }
    }

    template<bool ACCUMULATE>
    __global__ void swiglu_fused_half2_backward_kernel(
        half2* __restrict__ grad_gate_up,
        const half2* __restrict__ grad_output,
        const half2* __restrict__ gate_up,
        const std::uint32_t vector_size
    ) {
        const std::uint32_t row = blockIdx.y;
        const std::uint32_t output_row_offset = row * vector_size;
        const std::uint32_t input_row_offset = row * (2U * vector_size);

        for (std::uint32_t column =
                 blockIdx.x * blockDim.x + threadIdx.x;
             column < vector_size;
             column += gridDim.x * blockDim.x) {
            const std::uint32_t output_index = output_row_offset + column;
            const std::uint32_t gate_index = input_row_offset + column;
            const std::uint32_t up_index =
                input_row_offset + vector_size + column;

            const float2 grad_output_values =
                __half22float2(grad_output[output_index]);
            const float2 gate_values = __half22float2(gate_up[gate_index]);
            const float2 up_values = __half22float2(gate_up[up_index]);

            float2 grad_gate_values{
                grad_output_values.x * up_values.x *
                    silu_derivative(gate_values.x),
                grad_output_values.y * up_values.y *
                    silu_derivative(gate_values.y)
            };
            float2 grad_up_values{
                grad_output_values.x * silu(gate_values.x),
                grad_output_values.y * silu(gate_values.y)
            };

            if constexpr (ACCUMULATE) {
                const float2 previous_gate =
                    __half22float2(grad_gate_up[gate_index]);
                const float2 previous_up =
                    __half22float2(grad_gate_up[up_index]);
                grad_gate_values.x += previous_gate.x;
                grad_gate_values.y += previous_gate.y;
                grad_up_values.x += previous_up.x;
                grad_up_values.y += previous_up.y;
            }

            grad_gate_up[gate_index] = __floats2half2_rn(
                grad_gate_values.x, grad_gate_values.y);
            grad_gate_up[up_index] = __floats2half2_rn(
                grad_up_values.x, grad_up_values.y);
        }
    }

    template<bool ACCUMULATE>
    __global__ void swiglu_fused_bfloat162_backward_kernel(
        __nv_bfloat162* __restrict__ grad_gate_up,
        const __nv_bfloat162* __restrict__ grad_output,
        const __nv_bfloat162* __restrict__ gate_up,
        const std::uint32_t vector_size
    ) {
        const std::uint32_t row = blockIdx.y;
        const std::uint32_t output_row_offset = row * vector_size;
        const std::uint32_t input_row_offset = row * (2U * vector_size);

        for (std::uint32_t column =
                 blockIdx.x * blockDim.x + threadIdx.x;
             column < vector_size;
             column += gridDim.x * blockDim.x) {
            const std::uint32_t output_index = output_row_offset + column;
            const std::uint32_t gate_index = input_row_offset + column;
            const std::uint32_t up_index =
                input_row_offset + vector_size + column;

            const float2 grad_output_values =
                __bfloat1622float2(grad_output[output_index]);
            const float2 gate_values =
                __bfloat1622float2(gate_up[gate_index]);
            const float2 up_values =
                __bfloat1622float2(gate_up[up_index]);

            float2 grad_gate_values{
                grad_output_values.x * up_values.x *
                    silu_derivative(gate_values.x),
                grad_output_values.y * up_values.y *
                    silu_derivative(gate_values.y)
            };
            float2 grad_up_values{
                grad_output_values.x * silu(gate_values.x),
                grad_output_values.y * silu(gate_values.y)
            };

            if constexpr (ACCUMULATE) {
                const float2 previous_gate =
                    __bfloat1622float2(grad_gate_up[gate_index]);
                const float2 previous_up =
                    __bfloat1622float2(grad_gate_up[up_index]);
                grad_gate_values.x += previous_gate.x;
                grad_gate_values.y += previous_gate.y;
                grad_up_values.x += previous_up.x;
                grad_up_values.y += previous_up.y;
            }

            grad_gate_up[gate_index] = __floats2bfloat162_rn(
                grad_gate_values.x, grad_gate_values.y);
            grad_gate_up[up_index] = __floats2bfloat162_rn(
                grad_up_values.x, grad_up_values.y);
        }
    }

    [[nodiscard]] int get_block_count(const std::uint32_t element_count) {
        const std::uint32_t blocks =
            (element_count + THREADS_PER_BLOCK - 1U) / THREADS_PER_BLOCK;
        return static_cast<int>(std::min<std::uint32_t>(blocks, 65535U));
    }

    [[nodiscard]] int get_blocks_per_row(const std::uint32_t row_size) {
        const std::uint32_t blocks =
            (row_size + THREADS_PER_BLOCK - 1U) / THREADS_PER_BLOCK;
        return static_cast<int>(
            std::max<std::uint32_t>(
                1U,
                std::min<std::uint32_t>(blocks, MAX_BLOCKS_PER_ROW)
            )
        );
    }

    [[nodiscard]] std::uint32_t checked_u32(
        const std::size_t value,
        const char* const description
    ) {
        if (value > std::numeric_limits<std::uint32_t>::max()) {
            throw std::overflow_error(
                std::string("SwiGLU backward: ") + description +
                " exceeds the 32-bit kernel indexing limit"
            );
        }
        return static_cast<std::uint32_t>(value);
    }

    void validate_supported_dtype(const Dtype dtype) {
        if (dtype != Dtype::F32 &&
            dtype != Dtype::F16 &&
            dtype != Dtype::BF16) {
            throw std::invalid_argument(
                "SwiGLU backward: unsupported dtype"
            );
        }
    }

    void validate_separate_tensors(
        const Tensor& grad_gate,
        const Tensor& grad_up,
        const Tensor& grad_output,
        const Tensor& gate,
        const Tensor& up
    ) {
        if (grad_gate.shape() != gate.shape() ||
            grad_up.shape() != up.shape() ||
            grad_output.shape() != gate.shape() ||
            gate.shape() != up.shape()) {
            throw std::invalid_argument(
                "SwiGLU backward: all separate tensors must have the same shape"
            );
        }

        if (grad_gate.device_type() != DeviceType::CUDA ||
            grad_up.device_type() != DeviceType::CUDA ||
            grad_output.device_type() != DeviceType::CUDA ||
            gate.device_type() != DeviceType::CUDA ||
            up.device_type() != DeviceType::CUDA) {
            throw std::invalid_argument(
                "SwiGLU backward requires CUDA tensors"
            );
        }

        if (grad_gate.dtype() != gate.dtype() ||
            grad_up.dtype() != gate.dtype() ||
            grad_output.dtype() != gate.dtype() ||
            up.dtype() != gate.dtype()) {
            throw std::invalid_argument(
                "SwiGLU backward: all separate tensors must have the same dtype"
            );
        }

        validate_supported_dtype(gate.dtype());
    }

    void validate_fused_tensors(
        const Tensor& grad_gate_up,
        const Tensor& grad_output,
        const Tensor& gate_up
    ) {
        if (grad_gate_up.shape() != gate_up.shape()) {
            throw std::invalid_argument(
                "SwiGLU backward: grad_gate_up must match gate_up shape"
            );
        }

        if (gate_up.shape().empty() || grad_output.shape().empty()) {
            throw std::invalid_argument(
                "SwiGLU backward: tensors must have at least one dimension"
            );
        }

        if (gate_up.shape().back() % 2 != 0) {
            throw std::invalid_argument(
                "SwiGLU backward: gate_up last dimension must be even"
            );
        }

        const std::size_t intermediate_size = gate_up.shape().back() / 2;
        if (grad_output.shape().back() != intermediate_size) {
            throw std::invalid_argument(
                "SwiGLU backward: grad_output last dimension must equal half "
                "of gate_up last dimension"
            );
        }

        if (grad_output.shape().size() != gate_up.shape().size() ||
            !std::equal(
                grad_output.shape().begin(),
                grad_output.shape().end() - 1,
                gate_up.shape().begin()
            )) {
            throw std::invalid_argument(
                "SwiGLU backward: grad_output leading dimensions must match gate_up"
            );
        }

        if (grad_gate_up.device_type() != DeviceType::CUDA ||
            grad_output.device_type() != DeviceType::CUDA ||
            gate_up.device_type() != DeviceType::CUDA) {
            throw std::invalid_argument(
                "SwiGLU backward requires CUDA tensors"
            );
        }

        if (grad_gate_up.dtype() != gate_up.dtype() ||
            grad_output.dtype() != gate_up.dtype()) {
            throw std::invalid_argument(
                "SwiGLU backward: fused tensors must have the same dtype"
            );
        }

        if (gate_up.numel() != grad_output.numel() * 2) {
            throw std::invalid_argument(
                "SwiGLU backward: gate_up must contain twice as many elements "
                "as grad_output"
            );
        }

        if (intermediate_size != 0) {
            const std::size_t row_count =
                grad_output.numel() / intermediate_size;
            if (row_count > 65535) {
                throw std::overflow_error(
                    "SwiGLU backward: fused kernel supports at most 65535 rows"
                );
            }
        }

        validate_supported_dtype(gate_up.dtype());
    }

    template<typename T>
    void launch_separate_scalar(
        Tensor& grad_gate,
        Tensor& grad_up,
        const Tensor& grad_output,
        const Tensor& gate,
        const Tensor& up,
        cudaStream_t stream,
        const bool accumulate_gate,
        const bool accumulate_up
    ) {
        const std::uint32_t count =
            checked_u32(grad_output.numel(), "element count");
        const int blocks = get_block_count(count);

#define LAUNCH_SEPARATE_SCALAR(AG, AU) \
        swiglu_separate_scalar_backward_kernel<T, AG, AU><<< \
            blocks, THREADS_PER_BLOCK, 0, stream>>>( \
                static_cast<T*>(grad_gate.raw_data()), \
                static_cast<T*>(grad_up.raw_data()), \
                static_cast<const T*>(grad_output.raw_data()), \
                static_cast<const T*>(gate.raw_data()), \
                static_cast<const T*>(up.raw_data()), count)

        if (accumulate_gate) {
            if (accumulate_up) {
                LAUNCH_SEPARATE_SCALAR(true, true);
            } else {
                LAUNCH_SEPARATE_SCALAR(true, false);
            }
        } else if (accumulate_up) {
            LAUNCH_SEPARATE_SCALAR(false, true);
        } else {
            LAUNCH_SEPARATE_SCALAR(false, false);
        }

#undef LAUNCH_SEPARATE_SCALAR
    }

    void launch_separate_half2(
        Tensor& grad_gate,
        Tensor& grad_up,
        const Tensor& grad_output,
        const Tensor& gate,
        const Tensor& up,
        cudaStream_t stream,
        const bool accumulate_gate,
        const bool accumulate_up
    ) {
        const std::uint32_t count =
            checked_u32(grad_output.numel() / 2, "FP16 vector count");
        const int blocks = get_block_count(count);

#define LAUNCH_SEPARATE_HALF2(AG, AU) \
        swiglu_separate_half2_backward_kernel<AG, AU><<< \
            blocks, THREADS_PER_BLOCK, 0, stream>>>( \
                static_cast<half2*>(grad_gate.raw_data()), \
                static_cast<half2*>(grad_up.raw_data()), \
                static_cast<const half2*>(grad_output.raw_data()), \
                static_cast<const half2*>(gate.raw_data()), \
                static_cast<const half2*>(up.raw_data()), count)

        if (accumulate_gate) {
            if (accumulate_up) {
                LAUNCH_SEPARATE_HALF2(true, true);
            } else {
                LAUNCH_SEPARATE_HALF2(true, false);
            }
        } else if (accumulate_up) {
            LAUNCH_SEPARATE_HALF2(false, true);
        } else {
            LAUNCH_SEPARATE_HALF2(false, false);
        }

#undef LAUNCH_SEPARATE_HALF2
    }

    void launch_separate_bfloat162(
        Tensor& grad_gate,
        Tensor& grad_up,
        const Tensor& grad_output,
        const Tensor& gate,
        const Tensor& up,
        cudaStream_t stream,
        const bool accumulate_gate,
        const bool accumulate_up
    ) {
        const std::uint32_t count =
            checked_u32(grad_output.numel() / 2, "BF16 vector count");
        const int blocks = get_block_count(count);

#define LAUNCH_SEPARATE_BFLOAT162(AG, AU) \
        swiglu_separate_bfloat162_backward_kernel<AG, AU><<< \
            blocks, THREADS_PER_BLOCK, 0, stream>>>( \
                static_cast<__nv_bfloat162*>(grad_gate.raw_data()), \
                static_cast<__nv_bfloat162*>(grad_up.raw_data()), \
                static_cast<const __nv_bfloat162*>(grad_output.raw_data()), \
                static_cast<const __nv_bfloat162*>(gate.raw_data()), \
                static_cast<const __nv_bfloat162*>(up.raw_data()), count)

        if (accumulate_gate) {
            if (accumulate_up) {
                LAUNCH_SEPARATE_BFLOAT162(true, true);
            } else {
                LAUNCH_SEPARATE_BFLOAT162(true, false);
            }
        } else if (accumulate_up) {
            LAUNCH_SEPARATE_BFLOAT162(false, true);
        } else {
            LAUNCH_SEPARATE_BFLOAT162(false, false);
        }

#undef LAUNCH_SEPARATE_BFLOAT162
    }

    template<typename T>
    void launch_fused_scalar(
        Tensor& grad_gate_up,
        const Tensor& grad_output,
        const Tensor& gate_up,
        cudaStream_t stream,
        const bool accumulate
    ) {
        const std::uint32_t intermediate_size =
            checked_u32(grad_output.shape().back(), "intermediate size");
        const std::uint32_t row_count = checked_u32(
            grad_output.numel() / grad_output.shape().back(), "row count");

        const dim3 blocks(
            get_blocks_per_row(intermediate_size), row_count);

        if (accumulate) {
            swiglu_fused_scalar_backward_kernel<T, true><<<
                blocks, THREADS_PER_BLOCK, 0, stream>>>(
                    static_cast<T*>(grad_gate_up.raw_data()),
                    static_cast<const T*>(grad_output.raw_data()),
                    static_cast<const T*>(gate_up.raw_data()),
                    intermediate_size);
        } else {
            swiglu_fused_scalar_backward_kernel<T, false><<<
                blocks, THREADS_PER_BLOCK, 0, stream>>>(
                    static_cast<T*>(grad_gate_up.raw_data()),
                    static_cast<const T*>(grad_output.raw_data()),
                    static_cast<const T*>(gate_up.raw_data()),
                    intermediate_size);
        }
    }

    void launch_fused_half2(
        Tensor& grad_gate_up,
        const Tensor& grad_output,
        const Tensor& gate_up,
        cudaStream_t stream,
        const bool accumulate
    ) {
        const std::uint32_t vector_size =
            checked_u32(grad_output.shape().back() / 2, "FP16 vector size");
        const std::uint32_t row_count = checked_u32(
            grad_output.numel() / grad_output.shape().back(), "row count");
        const dim3 blocks(get_blocks_per_row(vector_size), row_count);

        if (accumulate) {
            swiglu_fused_half2_backward_kernel<true><<<
                blocks, THREADS_PER_BLOCK, 0, stream>>>(
                    static_cast<half2*>(grad_gate_up.raw_data()),
                    static_cast<const half2*>(grad_output.raw_data()),
                    static_cast<const half2*>(gate_up.raw_data()),
                    vector_size);
        } else {
            swiglu_fused_half2_backward_kernel<false><<<
                blocks, THREADS_PER_BLOCK, 0, stream>>>(
                    static_cast<half2*>(grad_gate_up.raw_data()),
                    static_cast<const half2*>(grad_output.raw_data()),
                    static_cast<const half2*>(gate_up.raw_data()),
                    vector_size);
        }
    }

    void launch_fused_bfloat162(
        Tensor& grad_gate_up,
        const Tensor& grad_output,
        const Tensor& gate_up,
        cudaStream_t stream,
        const bool accumulate
    ) {
        const std::uint32_t vector_size =
            checked_u32(grad_output.shape().back() / 2, "BF16 vector size");
        const std::uint32_t row_count = checked_u32(
            grad_output.numel() / grad_output.shape().back(), "row count");
        const dim3 blocks(get_blocks_per_row(vector_size), row_count);

        if (accumulate) {
            swiglu_fused_bfloat162_backward_kernel<true><<<
                blocks, THREADS_PER_BLOCK, 0, stream>>>(
                    static_cast<__nv_bfloat162*>(grad_gate_up.raw_data()),
                    static_cast<const __nv_bfloat162*>(grad_output.raw_data()),
                    static_cast<const __nv_bfloat162*>(gate_up.raw_data()),
                    vector_size);
        } else {
            swiglu_fused_bfloat162_backward_kernel<false><<<
                blocks, THREADS_PER_BLOCK, 0, stream>>>(
                    static_cast<__nv_bfloat162*>(grad_gate_up.raw_data()),
                    static_cast<const __nv_bfloat162*>(grad_output.raw_data()),
                    static_cast<const __nv_bfloat162*>(gate_up.raw_data()),
                    vector_size);
        }
    }
}

void swiglu_backward(
    Tensor& grad_gate,
    Tensor& grad_up,
    const Tensor& grad_output,
    const Tensor& gate,
    const Tensor& up,
    cudaStream_t stream,
    const SwiGLUBackwardOptions& options
) {
    validate_separate_tensors(
        grad_gate, grad_up, grad_output, gate, up);

    if (grad_output.numel() == 0) {
        return;
    }

    const bool can_vectorize = grad_output.numel() % 2 == 0;

    switch (grad_output.dtype()) {
        case Dtype::F32:
            launch_separate_scalar<float>(
                grad_gate, grad_up, grad_output, gate, up, stream,
                options.accumulate_gate, options.accumulate_up);
            break;

        case Dtype::F16:
            if (can_vectorize) {
                launch_separate_half2(
                    grad_gate, grad_up, grad_output, gate, up, stream,
                    options.accumulate_gate, options.accumulate_up);
            } else {
                launch_separate_scalar<half>(
                    grad_gate, grad_up, grad_output, gate, up, stream,
                    options.accumulate_gate, options.accumulate_up);
            }
            break;

        case Dtype::BF16:
            if (can_vectorize) {
                launch_separate_bfloat162(
                    grad_gate, grad_up, grad_output, gate, up, stream,
                    options.accumulate_gate, options.accumulate_up);
            } else {
                launch_separate_scalar<__nv_bfloat16>(
                    grad_gate, grad_up, grad_output, gate, up, stream,
                    options.accumulate_gate, options.accumulate_up);
            }
            break;

        default:
            throw std::invalid_argument(
                "SwiGLU backward: unsupported dtype");
    }

    CUDA_CHECK(cudaGetLastError());
}

void swiglu_backward(
    Tensor& grad_gate_up,
    const Tensor& grad_output,
    const Tensor& gate_up,
    cudaStream_t stream,
    const SwiGLUBackwardOptions& options
) {
    validate_fused_tensors(grad_gate_up, grad_output, gate_up);

    if (grad_output.numel() == 0) {
        return;
    }

    const bool can_vectorize =
        grad_output.shape().back() % 2 == 0;

    switch (grad_output.dtype()) {
        case Dtype::F32:
            launch_fused_scalar<float>(
                grad_gate_up, grad_output, gate_up, stream,
                options.accumulate_gate_up);
            break;

        case Dtype::F16:
            if (can_vectorize) {
                launch_fused_half2(
                    grad_gate_up, grad_output, gate_up, stream,
                    options.accumulate_gate_up);
            } else {
                launch_fused_scalar<half>(
                    grad_gate_up, grad_output, gate_up, stream,
                    options.accumulate_gate_up);
            }
            break;

        case Dtype::BF16:
            if (can_vectorize) {
                launch_fused_bfloat162(
                    grad_gate_up, grad_output, gate_up, stream,
                    options.accumulate_gate_up);
            } else {
                launch_fused_scalar<__nv_bfloat16>(
                    grad_gate_up, grad_output, gate_up, stream,
                    options.accumulate_gate_up);
            }
            break;

        default:
            throw std::invalid_argument(
                "SwiGLU backward: unsupported dtype");
    }

    CUDA_CHECK(cudaGetLastError());
}

