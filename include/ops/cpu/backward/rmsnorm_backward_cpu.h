#pragma once

#include "ops/backward/rmsnorm_backward.h"

void rmsnorm_backward_cpu(
    Tensor& grad_input, Tensor& grad_weight, const Tensor& grad_output,
    const Tensor& input, const Tensor& weight, float epsilon);
