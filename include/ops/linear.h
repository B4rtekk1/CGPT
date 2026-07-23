#pragma once

#include "core/cublas_context.h"
#include "core/tensor.h"

#include <cuda_runtime.h>

void linear_forward(
    Tensor& output,
    const Tensor& input,
    const Tensor& weight,
    const CublasLtContext& cublas_context,
    cudaStream_t stream = nullptr
    );