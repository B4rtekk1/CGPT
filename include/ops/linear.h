#pragma once

#include "core/cublas_context.h"
#include "core/tensor.h"

struct LinearOptions {
    ComputeType compute_type = ComputeType::TF32;
    std::size_t workspace_bytes = 4U * 1024U * 1024U;
};

void linear_forward(
    Tensor& output,
    const Tensor& input,
    const Tensor& weight,
    const CublasLtContext& cublas_context,
    cudaStream_t stream = nullptr,
    const LinearOptions& options = {}
    );

void linear_forward(
    Tensor& output,
    const Tensor& input,
    const Tensor& weight,
    const Tensor& bias,
    const CublasLtContext& cublas_context,
    cudaStream_t stream = nullptr,
    const LinearOptions& options = {}
    );
