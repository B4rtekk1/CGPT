#pragma once

#include "core/tensor.h"

/**
 * @brief CPU AVX2/FMA implementation of output = SiLU(gate) * up.
 */
void swiglu_forward_cpu(
    Tensor& output,
    const Tensor& gate,
    const Tensor& up
);

/**
 * @brief CPU AVX2/FMA implementation for a fused [..., gate, up] tensor.
 */
void swiglu_forward_cpu(
    Tensor& output,
    const Tensor& gate_up
);