/** @file CPU backward implementation of rotary positional embeddings. */
#pragma once

#include "ops/backward/rope_backward.h"

/** Computes gradients through the rotary positional embedding transform. */
void rope_backward_cpu(
    Tensor& grad_query, Tensor& grad_key,
    const Tensor& grad_rotated_query, const Tensor& grad_rotated_key,
    const Tensor& cos_cache, const Tensor& sin_cache,
    const RopeOptions& rope_options = {});
