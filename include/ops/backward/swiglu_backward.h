#pragma once

#include "core/tensor.h"

struct SwiGLUBackwardOptions {
    bool accumulate_gate = false;
    bool accumulate_up = false;
    bool accumulate_gate_up = false;
};

void swiglu_backward(
    Tensor& grad_gate,
    Tensor& grad_up,
    const Tensor& grad_output,
    const Tensor& gate,
    const Tensor& up,
    cudaStream_t stream = nullptr,
    const SwiGLUBackwardOptions& options = {}
);