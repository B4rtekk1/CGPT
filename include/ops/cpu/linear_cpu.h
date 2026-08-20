#pragma once

#include "core/tensor.h"

/** CPU AVX2/FMA linear projection: output = input * weight^T. */
void linear_forward_cpu(Tensor& output, const Tensor& input, const Tensor& weight);

/** CPU AVX2/FMA linear projection with a per-output bias. */
void linear_forward_cpu(Tensor& output, const Tensor& input, const Tensor& weight,
                        const Tensor& bias);
