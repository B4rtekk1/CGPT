/**
 * @file swiglu(4).cu
 * @brief CUDA implementation of the SwiGLU activation.
 *
 * Both separated-input (`gate`, `up`) and fused-input (`gate_up`) layouts are supported.
 * FP16 and BF16 use packed two-element vector paths whenever the last
 * dimension permits vector access.
 */

#include "ops/swiglu.h"

#include "core/cuda_check.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <string>

namespace {
    constexpr int THREADS_PER_BLOCK = 256;
    constexpr int MAX_BLOCKS_PER_ROW = 32;

    /** @brief Converts supported CUDA scalar types to and from FP32 arithmetic. */
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
            return __float2half_rn(value);
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

    /**
     * @brief Computes the SiLU activation using the CUDA fast exponential.
     * @param value Input value.
     * @return `value / (1 + exp(-value))`.
     */
    __device__ __forceinline__ float silu(const float value) {
        return value / (1.0f + __expf(-value));
    }

    /**
     * @brief Scalar SwiGLU kernel for separate gate and up tensors.
     * @tparam T Tensor element type.
     */
    template<typename T>
    __global__ void swiglu_separate_scalar_kernel(
        T* __restrict__ output,
        const T* __restrict__ gate,
        const T* __restrict__ up,
        const std::uint32_t element_count
    ) {
        const std::uint32_t stride =
            gridDim.x * blockDim.x;

        for (std::uint32_t index =
                 blockIdx.x * blockDim.x + threadIdx.x;
             index < element_count;
             index += stride) {
            const float gate_value = CudaTypeTraits<T>::load(gate[index]);
            const float up_value = CudaTypeTraits<T>::load(up[index]);
            output[index] = CudaTypeTraits<T>::store(silu(gate_value) * up_value);
        }
    }

    /** @brief Packed FP16 SwiGLU kernel for separate gate and up tensors, */
    __global__ void swiglu_separate_half2_kernel(
        half2* __restrict__ output,
        const half2* __restrict__ gate,
        const half2* __restrict__ up,
        const std::uint32_t vector_count
    ) {
        const auto stride =
            gridDim.x * blockDim.x;

        for (auto index =
                 blockIdx.x * blockDim.x + threadIdx.x;
             index < vector_count;
             index += stride) {
            const float2 gate_values = __half22float2(gate[index]);
            const float2 up_values = __half22float2(up[index]);

            output[index] = __floats2half2_rn(
                silu(gate_values.x) * up_values.x,
                silu(gate_values.y) * up_values.y
            );
        }
    }

    /** @brief Packed BF16 SwiGLU kernel for separate gate a nd up tensors.*/
    __global__ void swiglu_separate_bfloat162_kernel(
        __nv_bfloat162* __restrict__ output,
        const __nv_bfloat162* __restrict__ gate,
        const __nv_bfloat162* __restrict__ up,
        const std::uint32_t vector_count
    ) {
        const std::uint32_t stride =
            gridDim.x * blockDim.x;

        for (std::uint32_t index =
                 blockIdx.x * blockDim.x + threadIdx.x;
             index < vector_count;
             index += stride) {
            const float2 gate_values = __bfloat1622float2(gate[index]);
            const float2 up_values = __bfloat1622float2(up[index]);

            output[index] = __floats2bfloat162_rn(
                silu(gate_values.x) * up_values.x,
                silu(gate_values.y) * up_values.y
            );
        }
    }

    /**
     * @brief Scalar SwiGLU kernel for fused `[gate, up]` rows.
     * @tparam T Tensor element type.
     */
    template<typename T>
    __global__ void swiglu_fused_scalar_kernel(
        T* __restrict__ output,
        const T* __restrict__ gate_up,
        const std::uint32_t intermediate_size
    ) {
        const std::uint32_t row = blockIdx.y;
        const std::uint32_t output_row_offset = row * intermediate_size;
        const std::uint32_t input_row_offset = row * (2U * intermediate_size);

        for (auto column =
                 blockIdx.x * blockDim.x + threadIdx.x;
             column < intermediate_size;
             column += gridDim.x * blockDim.x) {
            const float gate_value =
                CudaTypeTraits<T>::load(gate_up[input_row_offset + column]);
            const float up_value =
                CudaTypeTraits<T>::load(
                    gate_up[input_row_offset + intermediate_size + column]
                );

            output[output_row_offset + column] =
                CudaTypeTraits<T>::store(silu(gate_value) * up_value);
        }
    }

    /** @brief Packed FP16 SwiGLU kernel for fused `[gate, up]` rows. */
    __global__ void swiglu_fused_half2_kernel(
        half2* __restrict__ output,
        const half2* __restrict__ gate_up,
        const std::uint32_t vector_size
    ) {
        const std::uint32_t row = blockIdx.y;
        const std::uint32_t output_row_offset = row * vector_size;
        const std::uint32_t input_row_offset = row * (2U * vector_size);

        for (auto column =
                 blockIdx.x * blockDim.x + threadIdx.x;
             column < vector_size;
             column += gridDim.x * blockDim.x) {
            const float2 gate_values =
                __half22float2(gate_up[input_row_offset + column]);
            const float2 up_values =
                __half22float2(gate_up[input_row_offset + vector_size + column]);

            output[output_row_offset + column] = __floats2half2_rn(
                silu(gate_values.x) * up_values.x,
                silu(gate_values.y) * up_values.y
            );
        }
    }

    /** @brief Packed BF16 SwiGLU kernel for fused `[gate, up]` rows. */
    __global__ void swiglu_fused_bfloat162_kernel(
        __nv_bfloat162* __restrict__ output,
        const __nv_bfloat162* __restrict__ gate_up,
        const std::uint32_t vector_size
    ) {
        const std::uint32_t row = blockIdx.y;
        const std::uint32_t output_row_offset = row * vector_size;
        const std::uint32_t input_row_offset = row * (2U * vector_size);

        for (auto column =
                 blockIdx.x * blockDim.x + threadIdx.x;
             column < vector_size;
             column += gridDim.x * blockDim.x) {
            const float2 gate_values =
                __bfloat1622float2(gate_up[input_row_offset + column]);
            const float2 up_values =
                __bfloat1622float2(gate_up[input_row_offset + vector_size + column]);

            output[output_row_offset + column] = __floats2bfloat162_rn(
                silu(gate_values.x) * up_values.x,
                silu(gate_values.y) * up_values.y
            );
        }
    }

    /** @brief Returns capped one-dimensional grid size for an elementwise kernel. */
    [[nodiscard]] int get_block_count(const std::uint32_t element_count) {
        const std::uint32_t blocks =
            (element_count + THREADS_PER_BLOCK - 1U) / THREADS_PER_BLOCK;
        return static_cast<int>(std::min<std::uint32_t>(blocks, 65535U));
    }

    /** @brief Returns the number of x-dimensional blocks assigned to each row. */
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

    /**
     * @brief Converts a size to 32-bit indexing after an overflow check.
     * @throws std::overflow_error When the value exceeds `uint32_t`.
     */
    [[nodiscard]] std::uint32_t checked_u32(
        const std::size_t value,
        const char* const description
    ) {
        if (value > std::numeric_limits<std::uint32_t>::max()) {
            throw std::overflow_error(
                std::string("SwiGLU: ") + description +
                " exceeds the 32-bit kernel indexing limit"
            );
        }
        return static_cast<std::uint32_t>(value);
    }

    /** @brief Validates the separate-input SwiGLU tensor contract. */
    void validate_separate_tensors(
        const Tensor& output,
        const Tensor& gate,
        const Tensor& up
    ) {
        if (output.shape() != gate.shape() || gate.shape() != up.shape()) {
            throw std::invalid_argument(
                "SwiGLU: output, gate and up must have the same shape"
            );
        }

        if (output.numel() != gate.numel() || gate.numel() != up.numel()) {
            throw std::invalid_argument(
                "SwiGLU: output, gate and up must have the same number of elements"
            );
        }

        if (output.device_type() != DeviceType::CUDA ||
            gate.device_type() != DeviceType::CUDA ||
            up.device_type() != DeviceType::CUDA) {
            throw std::invalid_argument("SwiGLU requires CUDA tensors");
        }

        if (output.dtype() != gate.dtype() || gate.dtype() != up.dtype()) {
            throw std::invalid_argument(
                "SwiGLU: output, gate and up must have the same dtype"
            );
        }
    }

    /** @brief Validates the fused-input SwiGLU tensor contract. */
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

        if (output.shape().empty()) {
            throw std::invalid_argument(
                "SwiGLU: output must have at least one dimension"
            );
        }

        const std::size_t gate_up_last_dimension = gate_up.shape().back();
        if (gate_up_last_dimension % 2 != 0) {
            throw std::invalid_argument(
                "SwiGLU: the last gate_up dimension must be even"
            );
        }

        const std::size_t intermediate_size = gate_up_last_dimension / 2;
        if (output.shape().back() != intermediate_size) {
            throw std::invalid_argument(
                "SwiGLU: output last dimension must equal half of gate_up last dimension"
            );
        }

        if (output.shape().size() != gate_up.shape().size() ||
            !std::equal(
                output.shape().begin(),
                output.shape().end() - 1,
                gate_up.shape().begin()
            )) {
            throw std::invalid_argument(
                "SwiGLU: output shape must match gate_up except for its last dimension"
            );
        }

        if (gate_up.numel() != output.numel() * 2) {
            throw std::invalid_argument(
                "SwiGLU: gate_up must contain twice as many elements as output"
            );
        }

        if (intermediate_size == 0) {
            return;
        }

        const std::size_t row_count = output.numel() / intermediate_size;
        if (row_count > 65535) {
            throw std::overflow_error(
                "SwiGLU: fused kernel supports at most 65535 rows per launch"
            );
        }
    }

    template<typename T>
    /** @brief Launches the scalar separate-input implementation. */
    void launch_separate_scalar(
        Tensor& output,
        const Tensor& gate,
        const Tensor& up,
        cudaStream_t stream
    ) {
        const std::uint32_t element_count =
            checked_u32(output.numel(), "element count");

        swiglu_separate_scalar_kernel<T><<<
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
    }

    /** @brief Launches the packed FP16 separate-input implementation. */
    void launch_separate_half2(
        Tensor& output,
        const Tensor& gate,
        const Tensor& up,
        cudaStream_t stream
    ) {
        const std::uint32_t vector_count =
            checked_u32(output.numel() / 2, "FP16 vector count");

        swiglu_separate_half2_kernel<<<
            get_block_count(vector_count),
            THREADS_PER_BLOCK,
            0,
            stream
        >>>(
            static_cast<half2*>(output.raw_data()),
            static_cast<const half2*>(gate.raw_data()),
            static_cast<const half2*>(up.raw_data()),
            vector_count
        );
    }

    /** @brief Launches the packed BF16 separate-input implementation. */
    void launch_separate_bfloat162(
        Tensor& output,
        const Tensor& gate,
        const Tensor& up,
        cudaStream_t stream
    ) {
        const std::uint32_t vector_count =
            checked_u32(output.numel() / 2, "BF16 vector count");

        swiglu_separate_bfloat162_kernel<<<
            get_block_count(vector_count),
            THREADS_PER_BLOCK,
            0,
            stream
        >>>(
            static_cast<__nv_bfloat162*>(output.raw_data()),
            static_cast<const __nv_bfloat162*>(gate.raw_data()),
            static_cast<const __nv_bfloat162*>(up.raw_data()),
            vector_count
        );
    }

    template<typename T>
    /** @brief Launches the scalar fused-input implementation. */
    void launch_fused_scalar(
        Tensor& output,
        const Tensor& gate_up,
        cudaStream_t stream
    ) {
        const std::uint32_t intermediate_size =
            checked_u32(output.shape().back(), "intermediate size");
        const std::uint32_t row_count = checked_u32(
            output.numel() / output.shape().back(),
            "row count"
        );

        swiglu_fused_scalar_kernel<T><<<
            dim3(get_blocks_per_row(intermediate_size), row_count),
            THREADS_PER_BLOCK,
            0,
            stream
        >>>(
            static_cast<T*>(output.raw_data()),
            static_cast<const T*>(gate_up.raw_data()),
            intermediate_size
        );
    }

    /** @brief Launches the packed FP16 fused-input implementation. */
    void launch_fused_half2(
        Tensor& output,
        const Tensor& gate_up,
        cudaStream_t stream
    ) {
        const std::uint32_t vector_size =
            checked_u32(output.shape().back() / 2, "FP16 vector size");
        const std::uint32_t row_count = checked_u32(
            output.numel() / output.shape().back(),
            "row count"
        );

        swiglu_fused_half2_kernel<<<
            dim3(get_blocks_per_row(vector_size), row_count),
            THREADS_PER_BLOCK,
            0,
            stream
        >>>(
            static_cast<half2*>(output.raw_data()),
            static_cast<const half2*>(gate_up.raw_data()),
            vector_size
        );
    }

    /** @brief Launches the packed BF16 fused-input implementation. */
    void launch_fused_bfloat162(
        Tensor& output,
        const Tensor& gate_up,
        cudaStream_t stream
    ) {
        const std::uint32_t vector_size =
            checked_u32(output.shape().back() / 2, "BF16 vector size");
        const std::uint32_t row_count = checked_u32(
            output.numel() / output.shape().back(),
            "row count"
        );

        swiglu_fused_bfloat162_kernel<<<
            dim3(get_blocks_per_row(vector_size), row_count),
            THREADS_PER_BLOCK,
            0,
            stream
        >>>(
            static_cast<__nv_bfloat162*>(output.raw_data()),
            static_cast<const __nv_bfloat162*>(gate_up.raw_data()),
            vector_size
        );
    }
}

/**
 * @brief Applies SwiGLU to separate gate and up tensors.
 *
 * Computes `output = SiLU(gate) * up` elementwise.
 *
 * @param output Destination tensor.
 * @param gate Gate projection tensor.
 * @param up Up projection tensor.
 * @param stream CUDA stream used for asynchronous execution.
 */
void swiglu_forward(
    Tensor& output,
    const Tensor& gate,
    const Tensor& up,
    cudaStream_t stream
) {
    validate_separate_tensors(output, gate, up);

    if (output.numel() == 0) {
        return;
    }

    const bool can_vectorize = output.numel() % 2 == 0;

    switch (output.dtype()) {
        case Dtype::F32:
            launch_separate_scalar<float>(output, gate, up, stream);
            break;

        case Dtype::F16:
            if (can_vectorize) {
                launch_separate_half2(output, gate, up, stream);
            } else {
                launch_separate_scalar<half>(output, gate, up, stream);
            }
            break;

        case Dtype::BF16:
            if (can_vectorize) {
                launch_separate_bfloat162(output, gate, up, stream);
            } else {
                launch_separate_scalar<__nv_bfloat16>(output, gate, up, stream);
            }
            break;

        default:
            throw std::invalid_argument("SwiGLU: unsupported dtype");
    }

    CUDA_CHECK(cudaGetLastError());
}

/**
 * @brief Applies SwiGLU to a fused `[gate, up]` tensor.
 *
 * The final dimension of `gate_up` must contain two equally sized contiguous
 * halves. The output final dimension equals one half of the input dimension.
 *
 * @param output Destination tensor.
 * @param gate_up Fused gate/up tensor.
 * @param stream CUDA stream used for asynchronous execution.
 */
void swiglu_forward(
    Tensor& output,
    const Tensor& gate_up,
    cudaStream_t stream
) {
    validate_fused_tensors(output, gate_up);

    if (output.numel() == 0) {
        return;
    }

    const bool can_vectorize = output.shape().back() % 2 == 0;

    switch (output.dtype()) {
        case Dtype::F32:
            launch_fused_scalar<float>(output, gate_up, stream);
            break;

        case Dtype::F16:
            if (can_vectorize) {
                launch_fused_half2(output, gate_up, stream);
            } else {
                launch_fused_scalar<half>(output, gate_up, stream);
            }
            break;

        case Dtype::BF16:
            if (can_vectorize) {
                launch_fused_bfloat162(output, gate_up, stream);
            } else {
                launch_fused_scalar<__nv_bfloat16>(output, gate_up, stream);
            }
            break;

        default:
            throw std::invalid_argument("SwiGLU: unsupported dtype");
    }

    CUDA_CHECK(cudaGetLastError());
}
