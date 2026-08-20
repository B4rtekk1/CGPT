/**
 * @file swiglu_backward_cpu.cpp
 * @brief CPU implementation of the SwiGLU backward pass.
 *
 * Computes gradients for the gate and up-projection inputs of
 * @f$y = \operatorname{SiLU}(gate) \odot up@f$. The implementation supports
 * F32, F16, and BF16 tensors and parallelizes independent elements with
 * OpenMP.
 */
#include "ops/cpu/swiglu_backward_cpu.h"

#include <immintrin.h>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>

namespace {
    /**
     * @brief Loads one tensor element and converts it to single precision.
     *
     * F16 values are converted with F16C intrinsics. BF16 values are expanded
     * by placing their bits in the upper half of an IEEE-754 float.
     *
     * @param p Pointer to the beginning of the tensor storage.
     * @param t Tensor element type.
     * @param i Linear element index.
     * @return Element converted to F32.
     */
    float load1(const void *p, Dtype t, std::size_t i) {
        if (t == Dtype::F32)return static_cast<const float *>(p)[i];
        const auto h = static_cast<const std::uint16_t *>(p)[i];
        if (t == Dtype::F16)return _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(h)));
        std::uint32_t b = static_cast<std::uint32_t>(h) << 16;
        float x;
        std::memcpy(&x, &b, 4);
        return x;
    }

    /**
     * @brief Stores one single-precision value in the requested tensor format.
     *
     * BF16 conversion uses round-to-nearest-even. F16 conversion is performed
     * with the F16C conversion intrinsic.
     *
     * @param p Pointer to the beginning of the destination storage.
     * @param t Destination element type.
     * @param i Linear element index.
     * @param x Value to store.
     */
    void store1(void *p, Dtype t, std::size_t i, float x) {
        if (t == Dtype::F32) {
            static_cast<float *>(p)[i] = x;
            return;
        }
        if (t == Dtype::F16) {
            static_cast<std::uint16_t *>(p)[i] = static_cast<std::uint16_t>(_mm_cvtsi128_si32(_mm_cvtps_ph(_mm_set_ss(x), 0)));
            return;
        }
        std::uint32_t b;
        std::memcpy(&b, &x, 4);
        static_cast<std::uint16_t *>(p)[i] = static_cast<std::uint16_t>((b + 0x7fff + ((b >> 16) & 1)) >> 16);
    }

    /**
     * @brief Validates tensors used by the split-input SwiGLU backward path.
     *
     * All tensors must be CPU-resident, floating-point, have identical shapes,
     * and use the same data type.
     *
     * @param a Gate-gradient output tensor.
     * @param b Up-gradient output tensor.
     * @param c Output-gradient tensor.
     * @param d Gate input tensor.
     * @param e Up input tensor.
     * @throws std::invalid_argument If any tensor is incompatible.
     */
    void check(const Tensor &a, const Tensor &b, const Tensor &c, const Tensor &d, const Tensor &e) {
        if (a.device_type() != DeviceType::CPU || b.device_type() != DeviceType::CPU || c.device_type() !=
            DeviceType::CPU || d.device_type() != DeviceType::CPU || e.device_type() != DeviceType::CPU || a.shape() !=
            c.shape() || b.shape() != c.shape() || d.shape() != c.shape() || e.shape() != c.shape() || a.dtype() != c.
            dtype() || b.dtype() != c.dtype() || d.dtype() != c.dtype() || e.dtype() != c.dtype() || !
            is_floating_point(c.dtype()))
            throw std::invalid_argument("CPU SwiGLU backward: incompatible tensors");
    }

    /**
     * @brief Evaluates the SiLU activation in single precision.
     * @param x Input value.
     * @return @f$x\,\sigma(x)@f$.
     */
    inline float silu(float x) {
        const float s = 1.0f / (1.0f + std::exp(-x));
        return x * s;
    }

    /**
     * @brief Evaluates the derivative of the SiLU activation.
     * @param x Input value.
     * @return @f$\sigma(x)\left(1+x(1-\sigma(x))\right)@f$.
     */
    inline float silu_grad(float x) {
        const float s = 1.0f / (1.0f + std::exp(-x));
        return s * (1.0f + x * (1.0f - s));
    }
}

/**
 * @brief Computes SwiGLU input gradients for separate gate and up tensors.
 *
 * For @f$y=\operatorname{SiLU}(gate)\odot up@f$, this function computes
 * @f$\partial L/\partial gate = grad\_out\odot up\odot
 * \operatorname{SiLU}'(gate)@f$ and
 * @f$\partial L/\partial up = grad\_out\odot\operatorname{SiLU}(gate)@f$.
 * Each element is independent and is processed in parallel with OpenMP.
 *
 * @param gg Destination gradient with respect to the gate input.
 * @param gu Destination gradient with respect to the up input.
 * @param go Gradient with respect to the SwiGLU output.
 * @param gate Gate-projection input used during the forward pass.
 * @param up Up-projection input used during the forward pass.
 * @param o Backward options controlling gradient accumulation.
 *
 * @throws std::invalid_argument If tensors are not matching CPU floating-point
 *         tensors.
 *
 * @note When @c o.accumulate_gate or @c o.accumulate_up is enabled, the newly
 *       computed gradient is added to the existing destination value.
 */
void swiglu_backward_cpu(Tensor &gg, Tensor &gu, const Tensor &go, const Tensor &gate, const Tensor &up,
                         const SwiGLUBackwardOptions &o) {
    check(gg, gu, go, gate, up);
    const auto n = go.numel();
    const auto t = go.dtype();
#pragma omp parallel for schedule(static)
    for (std::int64_t i = 0; i < static_cast<std::int64_t>(n); ++i) {
        const auto j = static_cast<std::size_t>(i);
        const float g = load1(go.raw_data(), t, j), x = load1(gate.raw_data(), t, j), u = load1(up.raw_data(), t, j);
        const float dg = g * u * silu_grad(x), du = g * silu(x);
        store1(gg.raw_data(), t, j, dg + (o.accumulate_gate ? load1(gg.raw_data(), t, j) : 0));
        store1(gu.raw_data(), t, j, du + (o.accumulate_up ? load1(gu.raw_data(), t, j) : 0));
    }
}

/**
 * @brief Computes SwiGLU gradients for a fused gate-up representation.
 *
 * The last dimension of @p gu is interpreted as two contiguous halves:
 * `[gate, up]`. The output @p ggu uses the same layout and stores the gradients
 * for both halves.
 *
 * @param ggu Destination fused gradient tensor with shape `[..., 2H]`.
 * @param go Output-gradient tensor with shape `[..., H]`.
 * @param gu Fused forward input tensor containing gate and up values with shape
 *        `[..., 2H]`.
 * @param o Backward options controlling fused-gradient accumulation.
 *
 * @throws std::invalid_argument If tensors are not CPU-resident floating-point
 *         tensors with a valid fused layout.
 *
 * @note If @c o.accumulate_gate_up is enabled, gradients are added to the
 *       existing values in @p ggu.
 */
void swiglu_backward_cpu(Tensor &ggu, const Tensor &go, const Tensor &gu, const SwiGLUBackwardOptions &o) {
    if (ggu.device_type() != DeviceType::CPU || go.device_type() != DeviceType::CPU || gu.device_type() !=
        DeviceType::CPU || gu.dim() < 1 || gu.shape().back() != 2 * go.shape().back() || go.numel() * 2 != gu.numel() ||
        ggu.shape() != gu.shape() || ggu.dtype() != go.dtype() || gu.dtype() != go.dtype() || !
        is_floating_point(go.dtype()))
        throw std::invalid_argument("CPU SwiGLU backward: invalid fused tensors");
    const auto n = go.numel();
    const auto h = go.shape().back();
    const Dtype t = go.dtype();
#pragma omp parallel for schedule(static)
    for (std::int64_t i = 0; i < static_cast<std::int64_t>(n); ++i) {
        const auto j = static_cast<std::size_t>(i), row = j / h, col = j % h;
        const auto gb = row * 2 * h + col, ub = gb + h;
        const float g = load1(go.raw_data(), t, j), x = load1(gu.raw_data(), t, gb), u = load1(gu.raw_data(), t, ub);
        const float dg = g * u * silu_grad(x), du = g * silu(x);
        store1(ggu.raw_data(), t, gb, dg + (o.accumulate_gate_up ? load1(ggu.raw_data(), t, gb) : 0));
        store1(ggu.raw_data(), t, ub, du + (o.accumulate_gate_up ? load1(ggu.raw_data(), t, ub) : 0));
    }
}
