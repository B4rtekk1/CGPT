#include "ops/swiglu.h"

#include "core/cuda_check.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <stdexcept>

namespace {
    constexpr int THREADS_PER_BLOCK = 256;

    template <typename T>
    struct CudaTypeTraits;

    template<>
    struct CudaTypeTraits<float> {
        __device__ static float load(const float value) {
            return value;
        }

        __device__ static float store(float value) {
            return value;
        }
    };

    template<>
    struct CudaTypeTraits<half> {
        __device__ static float load(const half value) {
            return __half2float(value);
        }

        __device__ static half store(const float value) {
            return __float2half_rn(value);
        }
    };

    template<>
    struct CudaTypeTraits<__nv_bfloat16> {
        __device__ static float load(const __nv_bfloat16 value) {
            return __bfloat162float(value);
        }

        __device__ static __nv_bfloat16 store(const float value) {
            return __float2bfloat16_rn(value);
        }
    };

    __device__ __forceinline__ float silu(const float value) {
        return value / (1.0f + __expf(-value));
    }

    template<typename T>
    __global__ void swiglu_separate_kernel(
        T* __restrict__ output,
        const T* __restrict__ gate,
        const T* __restrict__ up,
        const std::size_t element_count) {

        const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
        if (index >= element_count) {
            return;
        }

        const float gate_value = CudaTypeTraits<T>::load(gate[index]);
        const float up_value = CudaTypeTraits<T>::load(up[index]);
        const float result = silu(gate_value) * up_value;
        output[index] = CudaTypeTraits<T>::store(result);
    }

    template<typename T>
    __global__ void swiglu_fused_kernel(
        T* __restrict__ output,
        const T* __restrict__ gate_up,
        const std::size_t intermediate_size,
        std::size_t element_count
        ) {
        const std::size_t output_index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
        if (output_index >= element_count) {
            return;
        }
        const std::size_t row =
        output_index / intermediate_size;

        const std::size_t column =
            output_index % intermediate_size;

        const std::size_t input_row_offset =
            row * (2 * intermediate_size);

        const std::size_t gate_index =
            input_row_offset + column;

        const std::size_t up_index =
            input_row_offset + intermediate_size + column;

        const float gate_value =
            CudaTypeTraits<T>::load(gate_up[gate_index]);

        const float up_value =
            CudaTypeTraits<T>::load(gate_up[up_index]);

        const float result =
            silu(gate_value) * up_value;

        output[output_index] =
            CudaTypeTraits<T>::store(result);
    }

    [[nodiscard]] int get_block_count(
        const std::size_t element_count) {
        return static_cast<int>((element_count + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);
    }

    void validate_separate_tensors(
        const Tensor& output,
        const Tensor& gate,
        const Tensor& up
        ) {
        if (output.numel() != gate.numel()) {
            throw std::invalid_argument(
                "SwiGLU: output and gate must have the same number of elements"
            );
        }

        if (output.shape() != gate.shape() || gate.shape() != up.shape()) {
            throw std::invalid_argument(
                "SwiGLU: output, gate and up must have the same shape"
            );
        }

        if (output.device_type() != DeviceType::CUDA ||
            gate.device_type() != DeviceType::CUDA ||
            up.device_type() != DeviceType::CUDA) {
            throw std::invalid_argument("SwiGLU requires CUDA tensors");
        }

        if (gate.numel() != up.numel()) {
            throw std::invalid_argument(
                "SwiGLU: gate and up must have the same number of elements"
            );
        }

        if (
            output.dtype() != gate.dtype()
            || gate.dtype() != up.dtype()
        ) {
            throw std::invalid_argument(
                "SwiGLU: output, gate and up must have the same dtype"
            );
        }
    }

    void validate_fused_tensors(
    const Tensor& output,
    const Tensor& gate_up
) {

        if (output.device_type() != DeviceType::CUDA ||
            gate_up.device_type() != DeviceType::CUDA) {
            throw std::invalid_argument("SwiGLU requires CUDA tensors");
        }

        if (output.dtype() != gate_up.dtype()) {
            throw std::invalid_argument(
                "SwiGLU: output and gate_up must have the same dtype"
            );
        }

        if (gate_up.shape().empty()) {
            throw std::invalid_argument(
                "SwiGLU: gate_up must have at least one dimension"
            );
        }

        const std::size_t gate_up_last_dimension =
            gate_up.shape().back();

        if (gate_up_last_dimension % 2 != 0) {
            throw std::invalid_argument(
                "SwiGLU: the last gate_up dimension must be even"
            );
        }

        const std::size_t intermediate_size =
            gate_up_last_dimension / 2;

        if (output.shape().empty()) {
            throw std::invalid_argument(
                "SwiGLU: output must have at least one dimension"
            );
        }

        if (output.shape().back() != intermediate_size) {
            throw std::invalid_argument(
                "SwiGLU: output last dimension must equal half of gate_up last dimension"
            );
        }

        if (output.shape().size() != gate_up.shape().size() ||
            !std::equal(output.shape().begin(), output.shape().end() - 1,
                        gate_up.shape().begin())) {
            throw std::invalid_argument(
                "SwiGLU: output shape must match gate_up except for its last dimension"
            );
        }

        if (gate_up.numel() != output.numel() * 2) {
            throw std::invalid_argument(
                "SwiGLU: gate_up must contain twice as many elements as output"
            );
        }
    }

    template<typename T>
void launch_separate_swiglu(
    Tensor& output,
    const Tensor& gate,
    const Tensor& up,
    cudaStream_t stream
) {
        const std::size_t element_count = output.numel();

        if (element_count == 0) {
            return;
        }

        swiglu_separate_kernel<T><<<
            get_block_count(element_count),
            THREADS_PER_BLOCK,
            0,
            stream
        >>>(
            static_cast<T*>(output.raw_data()),
            static_cast<const T*>(gate.raw_data()),
            static_cast<const T*>(up.raw_data()),
            element_count
        );

        CUDA_CHECK(cudaGetLastError());
    }

    template<typename T>
    void launch_fused_swiglu(
        Tensor& output,
        const Tensor& gate_up,
        cudaStream_t stream
    ) {
        const std::size_t element_count =
            output.numel();

        if (element_count == 0) {
            return;
        }

        const std::size_t intermediate_size =
            output.shape().back();

        swiglu_fused_kernel<T><<<
            get_block_count(element_count),
            THREADS_PER_BLOCK,
            0,
            stream
        >>>(
            static_cast<T*>(output.raw_data()),
            static_cast<const T*>(gate_up.raw_data()),
            intermediate_size,
            element_count
        );

        CUDA_CHECK(cudaGetLastError());
    }

}

void swiglu_forward(
    Tensor& output,
    const Tensor& gate,
    const Tensor& up,
    cudaStream_t stream
) {
    validate_separate_tensors(output, gate, up);

    switch (output.dtype()) {
        case Dtype::F32:
            launch_separate_swiglu<float>(
                output,
                gate,
                up,
                stream
            );
            break;

        case Dtype::F16:
            launch_separate_swiglu<half>(
                output,
                gate,
                up,
                stream
            );
            break;

        case Dtype::BF16:
            launch_separate_swiglu<__nv_bfloat16>(
                output,
                gate,
                up,
                stream
            );
            break;

        default:
            throw std::invalid_argument(
                "SwiGLU: unsupported dtype"
            );
    }
}

void swiglu_forward(
    Tensor& output,
    const Tensor& gate_up,
    cudaStream_t stream
) {
    validate_fused_tensors(output, gate_up);

    switch (output.dtype()) {
        case Dtype::F32:
            launch_fused_swiglu<float>(
                output,
                gate_up,
                stream
            );
            break;

        case Dtype::F16:
            launch_fused_swiglu<half>(
                output,
                gate_up,
                stream
            );
            break;

        case Dtype::BF16:
            launch_fused_swiglu<__nv_bfloat16>(
                output,
                gate_up,
                stream
            );
            break;

        default:
            throw std::invalid_argument(
                "SwiGLU: unsupported dtype"
            );
    }
}
