#pragma once

#include "../core/tensor.h"

void rmsnorm_forward(
    Tensor& output,
    const Tensor& input,
    const Tensor& weight,
    float epsilon,
    cudaStream_t stream = nullptr
);

Tensor rmsnorm_forward(
    const Tensor& input,
    const Tensor& weight,
    float epsilon,
    cudaStream_t stream = nullptr
);
