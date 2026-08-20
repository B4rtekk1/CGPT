#pragma once

#include "core/tensor.h"
#include "ops/rope.h"

/** CPU AVX2/FMA RoPE, modifying query and key in place. */
void rope_forward_cpu(
    Tensor& query,
    Tensor& key,
    const Tensor& cos_cache,
    const Tensor& sin_cache,
    const RopeOptions& rope_options = {}
);
