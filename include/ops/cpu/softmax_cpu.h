#pragma once

#include "core/tensor.h"

/**
 * @brief Numerically stable row-wise CPU softmax using AVX2/FMA.
 */
void softmax_forward_cpu(Tensor& output, const Tensor& input);