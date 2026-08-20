#pragma once

#include "core/tensor.h"

/** CPU AVX2/FMA RMSNorm, modifying output in place. */
void rmsnorm_forward_cpu(
    Tensor& output,
    const Tensor& input,
    const Tensor& weight,
    float epsilon
);

/** Allocating CPU RMSNorm overload. */
Tensor rmsnorm_forward_cpu(const Tensor& input, const Tensor& weight, float epsilon);
