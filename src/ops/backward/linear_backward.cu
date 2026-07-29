/**
 * @file linear_backward.cu
 * @brief CUDA/cuBLASLt backward pass for a row-major linear layer.
 *
 * Forward convention:
 * Y = X W^T + b
 *
 * Backward equations:
 * dX = dY * W
 * dW = dY^T * X
 * db = reduce_rows(dY)
 */

#include "ops/backward/linear_backward.h"
#include "core/device_buffer.h"
#include "core/device_guard.h"

#include <cublasLt.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <string>

namespace {

const char* device_type_name(const DeviceType device_type) {
    switch (device_type) {
    case DeviceType::CPU: return "CPU";
    case DeviceType::CUDA: return "CUDA";
    }
    return "unknown";
}

void require_same_device(const Tensor& left, const Tensor& right) {
    if (left.device_type() != right.device_type()) {
        throw std::invalid_argument(
            "linear_backward: tensors must be on the same device, got " +
            std::string(device_type_name(left.device_type())) + " and " +
            device_type_name(right.device_type()));
    }
}

void set_row_major(cublasLtMatrixLayout_t layout) {
    constexpr cublasLtOrder_t order = CUBLASLT_ORDER_ROW;
    CUBLAS_CHECK(cublasLtMatrixLayoutSetAttribute(
        layout, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order)));
}

void validate_linear_backward(
    const Tensor& grad_input,
    const Tensor& grad_weight,
    const Tensor& grad_bias,
    const Tensor& grad_output,
    const Tensor& input,
    const Tensor& weight,
    const LinearBackwardOptions& options
) {
    if (input.dim() < 2) {
        throw std::invalid_argument("linear_backward: input must have at least 2 dimensions");
    }
    if (weight.dim() != 2) {
        throw std::invalid_argument(
            "linear_backward: weight must have shape [out_features, in_features]");
    }
    if (grad_output.dim() != input.dim() || grad_input.shape() != input.shape()) {
        throw std::invalid_argument(
            "linear_backward: grad_output and grad_input must match input rank and leading dimensions");
    }

    const std::size_t input_features = input.size(input.dim() - 1);
    const std::size_t output_features = weight.size(0);
    if (weight.size(1) != input_features) {
        throw std::invalid_argument(
            "linear_backward: input feature size does not match weight");
    }
    for (std::size_t axis = 0; axis + 1 < input.dim(); ++axis) {
        if (grad_output.size(axis) != input.size(axis)) {
            throw std::invalid_argument(
                "linear_backward: grad_output leading dimensions must match input");
        }
    }
    if (grad_output.size(grad_output.dim() - 1) != output_features) {
        throw std::invalid_argument(
            "linear_backward: grad_output last dimension must equal out_features");
    }
    if (grad_weight.dim() != 2 || grad_weight.size(0) != output_features ||
        grad_weight.size(1) != input_features) {
        throw std::invalid_argument(
            "linear_backward: grad_weight must have shape [out_features, in_features]");
    }
    if (grad_bias.dim() != 1 || grad_bias.size(0) != output_features) {
        throw std::invalid_argument(
            "linear_backward: grad_bias must have shape [out_features]");
    }

    const Tensor* tensors[] = {
        &grad_input, &grad_weight, &grad_bias, &grad_output, &input, &weight};
    for (const Tensor* tensor : tensors) {
        if (tensor->device_type() != DeviceType::CUDA) {
            throw std::invalid_argument("linear_backward: tensors must be on a CUDA device");
        }
        if (!is_floating_point(tensor->dtype()) || tensor->dtype() != input.dtype()) {
            throw std::invalid_argument(
                "linear_backward: all tensors must have the same floating dtype");
        }
        require_same_device(input, *tensor);
    }
    if (!is_valid_compute_type(options.compute_type)) {
        throw std::invalid_argument("linear_backward: invalid compute type");
    }
}

/** Executes one row-major cuBLASLt matrix product with optional accumulation. */
void matmul(
    cublasLtHandle_t handle,
    Tensor& output,
    const Tensor& left,
    const Tensor& right,
    const std::size_t rows,
    const std::size_t columns,
    const std::size_t inner,
    const cublasOperation_t transpose_left,
    const cublasOperation_t transpose_right,
    const ComputeType requested_compute_type,
    const std::size_t workspace_bytes,
    const bool accumulate,
    cudaStream_t stream
) {
    const cudaDataType_t data_type = to_cuda_dtype(output.dtype());
    const bool use_tf32 = output.dtype() == Dtype::F32 &&
        requested_compute_type == ComputeType::TF32 &&
        rows >= 8 && columns >= 8 && inner >= 8;
    const cublasComputeType_t compute_type = use_tf32
        ? CUBLAS_COMPUTE_32F_FAST_TF32
        : CUBLAS_COMPUTE_32F;

    cublasLtMatmulDesc_t operation = nullptr;
    cublasLtMatrixLayout_t left_layout = nullptr;
    cublasLtMatrixLayout_t right_layout = nullptr;
    cublasLtMatrixLayout_t output_layout = nullptr;
    cublasLtMatmulPreference_t preference = nullptr;

    CUBLAS_CHECK(cublasLtMatmulDescCreate(&operation, compute_type, CUDA_R_32F));
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(
        operation, CUBLASLT_MATMUL_DESC_TRANSA, &transpose_left, sizeof(transpose_left)));
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(
        operation, CUBLASLT_MATMUL_DESC_TRANSB, &transpose_right, sizeof(transpose_right)));

    const std::size_t left_rows = transpose_left == CUBLAS_OP_N ? rows : inner;
    const std::size_t left_columns = transpose_left == CUBLAS_OP_N ? inner : rows;
    const std::size_t right_rows = transpose_right == CUBLAS_OP_N ? inner : columns;
    const std::size_t right_columns = transpose_right == CUBLAS_OP_N ? columns : inner;
    CUBLAS_CHECK(cublasLtMatrixLayoutCreate(
        &left_layout, data_type, left_rows, left_columns, left_columns));
    CUBLAS_CHECK(cublasLtMatrixLayoutCreate(
        &right_layout, data_type, right_rows, right_columns, right_columns));
    CUBLAS_CHECK(cublasLtMatrixLayoutCreate(
        &output_layout, data_type, rows, columns, columns));
    set_row_major(left_layout);
    set_row_major(right_layout);
    set_row_major(output_layout);

    CUBLAS_CHECK(cublasLtMatmulPreferenceCreate(&preference));
    CUBLAS_CHECK(cublasLtMatmulPreferenceSetAttribute(
        preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
        &workspace_bytes, sizeof(workspace_bytes)));

    cublasLtMatmulHeuristicResult_t heuristic{};
    int returned_results = 0;
    const cublasStatus_t heuristic_status = cublasLtMatmulAlgoGetHeuristic(
        handle, operation, left_layout, right_layout, output_layout, output_layout,
        preference, 1, &heuristic, &returned_results);

    DeviceBuffer workspace;
    const cublasLtMatmulAlgo_t* algorithm = nullptr;
    std::size_t selected_workspace_bytes = 0;
    if (heuristic_status == CUBLAS_STATUS_SUCCESS && returned_results != 0 &&
        heuristic.state == CUBLAS_STATUS_SUCCESS && heuristic.workspaceSize <= workspace_bytes) {
        algorithm = &heuristic.algo;
        selected_workspace_bytes = heuristic.workspaceSize;
        if (selected_workspace_bytes != 0) {
            workspace.allocate(selected_workspace_bytes);
        }
    } else if (heuristic_status != CUBLAS_STATUS_SUCCESS &&
               heuristic_status != CUBLAS_STATUS_NOT_SUPPORTED) {
        CUBLAS_CHECK(heuristic_status);
    }

    constexpr float alpha = 1.0f;
    const float beta = accumulate ? 1.0f : 0.0f;
    cublasStatus_t status = cublasLtMatmul(
        handle, operation, &alpha,
        left.raw_data(), left_layout,
        right.raw_data(), right_layout,
        &beta,
        output.raw_data(), output_layout,
        output.raw_data(), output_layout,
        algorithm, workspace.data(), selected_workspace_bytes, stream);
    if (status == CUBLAS_STATUS_NOT_SUPPORTED && algorithm != nullptr) {
        status = cublasLtMatmul(
            handle, operation, &alpha,
            left.raw_data(), left_layout,
            right.raw_data(), right_layout,
            &beta,
            output.raw_data(), output_layout,
            output.raw_data(), output_layout,
            nullptr, nullptr, 0, stream);
    }
    CUBLAS_CHECK(status);

    CUBLAS_CHECK(cublasLtMatmulPreferenceDestroy(preference));
    CUBLAS_CHECK(cublasLtMatrixLayoutDestroy(output_layout));
    CUBLAS_CHECK(cublasLtMatrixLayoutDestroy(right_layout));
    CUBLAS_CHECK(cublasLtMatrixLayoutDestroy(left_layout));
    CUBLAS_CHECK(cublasLtMatmulDescDestroy(operation));
}

template <typename T>
__global__ void reduce_bias_kernel(
    T* grad_bias,
    const T* grad_output,
    const std::size_t rows,
    const std::size_t columns,
    const bool accumulate
) {
    extern __shared__ float partial[];
    const std::size_t column = blockIdx.x;
    float sum = 0.0f;
    for (std::size_t row = threadIdx.x; row < rows; row += blockDim.x) {
        sum += static_cast<float>(grad_output[row * columns + column]);
    }
    partial[threadIdx.x] = sum;
    __syncthreads();

    for (unsigned int stride = blockDim.x / 2; stride != 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            partial[threadIdx.x] += partial[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        const float value = accumulate
            ? static_cast<float>(grad_bias[column]) + partial[0]
            : partial[0];
        grad_bias[column] = static_cast<T>(value);
    }
}

void launch_bias_reduction(
    Tensor& grad_bias,
    const Tensor& grad_output,
    const std::size_t rows,
    const bool accumulate,
    cudaStream_t stream
) {
    constexpr unsigned int threads = 256;
    const std::size_t columns = grad_bias.numel();
    if (columns > std::numeric_limits<unsigned int>::max()) {
        throw std::invalid_argument("linear_backward: out_features exceeds CUDA grid.x range");
    }
    const dim3 blocks(static_cast<unsigned int>(columns));
    const std::size_t shared_bytes = threads * sizeof(float);
    if (grad_bias.dtype() == Dtype::F16) {
        reduce_bias_kernel<<<blocks, threads, shared_bytes, stream>>>(
            static_cast<__half*>(grad_bias.raw_data()),
            static_cast<const __half*>(grad_output.raw_data()), rows, columns, accumulate);
    } else if (grad_bias.dtype() == Dtype::BF16) {
        reduce_bias_kernel<<<blocks, threads, shared_bytes, stream>>>(
            static_cast<__nv_bfloat16*>(grad_bias.raw_data()),
            static_cast<const __nv_bfloat16*>(grad_output.raw_data()), rows, columns, accumulate);
    } else {
        reduce_bias_kernel<<<blocks, threads, shared_bytes, stream>>>(
            static_cast<float*>(grad_bias.raw_data()),
            static_cast<const float*>(grad_output.raw_data()), rows, columns, accumulate);
    }
    CUDA_CHECK(cudaGetLastError());
}

}

void linear_backward(
    Tensor& grad_input,
    Tensor& grad_weight,
    Tensor& grad_bias,
    const Tensor& grad_output,
    const Tensor& input,
    const Tensor& weight,
    const CublasLtContext& cublas_lt_context,
    cudaStream_t stream,
    const LinearBackwardOptions& options
) {
    validate_linear_backward(
        grad_input, grad_weight, grad_bias, grad_output, input, weight, options);

    const std::size_t input_features = input.size(input.dim() - 1);
    const std::size_t output_features = weight.size(0);
    const std::size_t rows = input.numel() / input_features;
    DeviceGuard device_guard(cublas_lt_context.device_index());

    constexpr cublasOperation_t no_transpose = CUBLAS_OP_N;
    constexpr cublasOperation_t transpose = CUBLAS_OP_T;
    matmul(cublas_lt_context.handle(), grad_input, grad_output, weight,
           rows, input_features, output_features, no_transpose, no_transpose,
           options.compute_type, options.workspace_bytes, options.accumulate_input, stream);
    matmul(cublas_lt_context.handle(), grad_weight, grad_output, input,
           output_features, input_features, rows, transpose, no_transpose,
           options.compute_type, options.workspace_bytes, options.accumulate_weight, stream);
    launch_bias_reduction(grad_bias, grad_output, rows, options.accumulate_bias, stream);
}
