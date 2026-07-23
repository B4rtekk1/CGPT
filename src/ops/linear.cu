#include "ops/linear.h"
#include "core/device_guard.h"
#include <cublasLt.h>

#include <stdexcept>
#include <string>

namespace {
    const char* device_type_name(DeviceType device_type) {
        switch (device_type) {
        case DeviceType::CPU:
            return "CPU";
        case DeviceType::CUDA:
            return "CUDA";
        }

        return "unknown";
    }

    void require_f32(const Tensor& tensor, const char* name) {
        if (tensor.dtype() != Dtype::F32) {
            throw std::invalid_argument(
                std::string(name) + " must have dtype F32, got " +
                std::string(dtype_name(tensor.dtype())));
        }
    }

    void require_same_device(const Tensor& a, const Tensor& b) {
        if (a.device_type() != b.device_type()) {
            throw std::invalid_argument(
                "Tensors must be on the same device, got " +
                std::string(device_type_name(a.device_type())) + " and " +
                std::string(device_type_name(b.device_type())));
        }
    }

    void set_row_major(cublasLtMatrixLayout_t layout) {
        const cublasLtOrder_t order = CUBLASLT_ORDER_ROW;
        CUBLAS_CHECK(cublasLtMatrixLayoutSetAttribute(
            layout,
            CUBLASLT_MATRIX_LAYOUT_ORDER,
            &order,
            sizeof(order)
        ));
    }
}

void linear_forward(
    Tensor& output,
    const Tensor& input,
    const Tensor& weight,
    const CublasLtContext& cublas_lt_context,
    cudaStream_t stream
    ) {
    if (input.dim() < 2) {
        throw std::invalid_argument("linear_forward: input must have at least 2 dimensions");
    }

    if (weight.dim() < 2) {
        throw std::invalid_argument("linear_forward: weight must have at least 2 dimensions");
    }

    if (output.dim() != input.dim()) {
        throw std::invalid_argument("linear_forward: output must have the same number of dimensions as input");
    }

    require_f32(input, "linear_forward: input");
    require_f32(weight, "linear_forward: weight");
    require_f32(output, "linear_forward: output");

    require_same_device(input, weight);
    require_same_device(input, output);

    if (input.device_type() != DeviceType::CUDA) {
        throw std::invalid_argument("linear_forward: input must be on CUDA device");
    }

    const std::size_t input_dim = input.size(input.dim() - 1);
    const std::size_t output_dim = weight.size(0);
    const std::size_t weight_input_dim = weight.size(1);

    if (input_dim != weight_input_dim) {
        throw std::invalid_argument(
            "linear_forward: input feature size does not match weight"
        );
    }

    for (std::size_t axis = 0; axis + 1 < input.dim(); ++axis) {
        if (input.size(axis) != output.size(axis)) {
            throw std::invalid_argument(
                "linear_forward: output leading dimensions must match input"
            );
        }
    }

    if (output.size(output.dim() - 1) != output_dim) {
        throw std::invalid_argument(
            "linear_forward: output last dimension must equal out_features"
        );
    }

    const std::size_t rows = input.numel() / input_dim;

    DeviceGuard device_guard(cublas_lt_context.device_index());

    cublasLtMatmulDesc_t operation = nullptr;
    cublasLtMatrixLayout_t input_layout = nullptr;
    cublasLtMatrixLayout_t weight_layout = nullptr;
    cublasLtMatrixLayout_t output_layout = nullptr;

    CUBLAS_CHECK(cublasLtMatmulDescCreate(
        &operation,
        CUBLAS_COMPUTE_32F,
        CUDA_R_32F
    ));
    // Y = X × Wᵀ
    const cublasOperation_t transpose_weight = CUBLAS_OP_T;

    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(
        operation,
        CUBLASLT_MATMUL_DESC_TRANSB,
        &transpose_weight,
        sizeof(transpose_weight)
    ));

    CUBLAS_CHECK(cublasLtMatrixLayoutCreate(
        &input_layout,
        CUDA_R_32F,
        rows,
        input_dim,
        input_dim
    ));

    CUBLAS_CHECK(cublasLtMatrixLayoutCreate(
        &weight_layout,
        CUDA_R_32F,
        output_dim,
        input_dim,
        input_dim
    ));

    CUBLAS_CHECK(cublasLtMatrixLayoutCreate(
        &output_layout,
        CUDA_R_32F,
        rows,
        output_dim,
        output_dim
    ));

    set_row_major(input_layout);
    set_row_major(weight_layout);
    set_row_major(output_layout);

    constexpr float alpha = 1.0f;
    constexpr float beta = 0.0f;

    CUBLAS_CHECK(cublasLtMatmul(
        cublas_lt_context.handle(),
        operation,
        &alpha,
        input.data(),
        input_layout,
        weight.data(),
        weight_layout,
        &beta,
        output.data(),
        output_layout,
        output.data(),
        output_layout,
        nullptr,
        nullptr,
        0,
        stream
    ));

    CUBLAS_CHECK(cublasLtMatrixLayoutDestroy(input_layout));
    CUBLAS_CHECK(cublasLtMatrixLayoutDestroy(weight_layout));
    CUBLAS_CHECK(cublasLtMatrixLayoutDestroy(output_layout));
    CUBLAS_CHECK(cublasLtMatmulDescDestroy(operation));
}
