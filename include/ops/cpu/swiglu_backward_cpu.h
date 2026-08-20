#pragma once

#include "core/tensor.h"
#include "ops/backward/swiglu_backward.h"

void swiglu_backward_cpu(
    Tensor& grad_gate, Tensor& grad_up, const Tensor& grad_output,
    const Tensor& gate, const Tensor& up,
    const SwiGLUBackwardOptions& options = {});

void swiglu_backward_cpu(
    Tensor& grad_gate_up, const Tensor& grad_output, const Tensor& gate_up,
    const SwiGLUBackwardOptions& options = {});
