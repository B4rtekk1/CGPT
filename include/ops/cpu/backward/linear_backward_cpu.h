#pragma once

#include "ops/backward/linear_backward.h"

void linear_backward_cpu(
    Tensor& grad_input, Tensor& grad_weight, Tensor& grad_bias,
    const Tensor& grad_output, const Tensor& input, const Tensor& weight,
    const LinearBackwardOptions& options = {});
