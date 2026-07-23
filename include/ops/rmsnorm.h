#pragma once

#include <cuda_runtime.h>

#include "../core/tensor.h"

Tensor rmsnorm_forward(
    const Tensor& input,
    const Tensor& weight,
    float epsilon,
    cudaStream_t stream = nullptr
);
