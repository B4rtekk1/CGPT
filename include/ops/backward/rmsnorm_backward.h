#include "core/tensor.h"

/**
 * @brief Computes gradients for `y = x * rsqrt(mean(x²) + epsilon) * weight`.
 *
 * All tensors must be CUDA tensors of the same floating-point dtype. `input`,
 * `grad_output` and `grad_input` have shape `[rows, hidden]`; `weight` and
 * `grad_weight` have shape `[hidden]`. Reductions and intermediate arithmetic
 * use FP32, including when the storage dtype is FP16 or BF16.
 */
void rmsnorm_backward(
    Tensor& grad_input,
    Tensor& grad_weight,
    const Tensor& grad_output,
    const Tensor& input,
    const Tensor& weight,
    float epsilon,
    cudaStream_t stream = nullptr
);
