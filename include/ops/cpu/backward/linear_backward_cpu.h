/** @file CPU backward implementation of a linear layer. */
/** @file CPU backward implementation of a linear layer. */
#pragma once

#include "ops/backward/linear_backward.h"

/** Computes input, weight, and bias gradients for a linear layer. */
/** Computes input, weight, and bias gradients for a linear layer. */
void linear_backward_cpu(
    Tensor& grad_input, Tensor& grad_weight, Tensor& grad_bias,
    const Tensor& grad_output, const Tensor& input, const Tensor& weight,
    const LinearBackwardOptions& options = {});
