#pragma once

#include <cuda_runtime.h>

void rmsnorm_forward(
    float* output,
    const float* input,
    const float* weight,
    int rows,
    int hidden,
    float epsilon,
    cudaStream_t stream = nullptr
    );