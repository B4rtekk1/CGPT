#pragma once
#include "core/tensor.h"


void swiglu_forward(
    Tensor& output,
    const Tensor& gate,
    const Tensor& up,
    cudaStream_t stream = nullptr
    );

void swiglu_forward(
    Tensor& output,
    const Tensor& gate_up,
    cudaStream_t stream = nullptr
    );
