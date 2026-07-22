#pragma once

#include <cuda_runtime.h>

#include "tensor.hpp"

Tensor rmsnorm_forward(
    const Tensor& input,
    const Tensor& weight,
    float epsilon,
    cudaStream_t stream = nullptr
);
