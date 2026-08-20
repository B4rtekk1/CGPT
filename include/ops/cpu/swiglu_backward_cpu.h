/** @file CPU backward implementation of the SwiGLU operation. */
#pragma once

#include "core/tensor.h"
#include "ops/backward/swiglu_backward.h"

/** Computes gradients for separate SwiGLU gate and up tensors. */
void swiglu_backward_cpu(
    Tensor& grad_gate, Tensor& grad_up, const Tensor& grad_output,
    const Tensor& gate, const Tensor& up,
    const SwiGLUBackwardOptions& options = {});

/** Computes gradients when the gate and up projections are packed together. */
void swiglu_backward_cpu(
    Tensor& grad_gate_up, const Tensor& grad_output, const Tensor& gate_up,
    const SwiGLUBackwardOptions& options = {});
