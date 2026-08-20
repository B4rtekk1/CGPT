/** @file CPU backward implementation of RMS normalization. */
/** @file CPU backward implementation of RMS normalization. */
#pragma once

#include "ops/backward/rmsnorm_backward.h"

/** Computes input and scale gradients for RMS normalization. */
/** Computes input and scale gradients for RMS normalization. */
void rmsnorm_backward_cpu(
    Tensor& grad_input, Tensor& grad_weight, const Tensor& grad_output,
    const Tensor& input, const Tensor& weight, float epsilon);
