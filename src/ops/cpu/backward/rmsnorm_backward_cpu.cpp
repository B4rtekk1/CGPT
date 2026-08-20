/**
 * @file rmsnorm_backward_cpu.cpp
 * @brief CPU implementation of the RMSNorm backward pass.
 *
 * Computes gradients with respect to both the input activations and the
 * learnable scale vector for two-dimensional tensors. F32, F16, and BF16
 * storage are supported while all intermediate arithmetic is performed in F32.
 */
#include "ops/cpu/backward/rmsnorm_backward_cpu.h"

#include <immintrin.h>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace {
    /**
     * @brief Loads one tensor element and converts it to F32.
     * @param p Pointer to tensor storage.
     * @param t Tensor element type.
     * @param i Linear element index.
     * @return Loaded value in single precision.
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
     * @brief Stores one F32 value in F32, F16, or BF16 representation.
     * @param p Pointer to destination storage.
     * @param t Destination element type.
     * @param i Linear element index.
     * @param x Value to store.
     *
     * @note BF16 conversion uses round-to-nearest-even.
     */
    void store1(void *p, Dtype t, std::size_t i, float x) {
        if (t == Dtype::F32) {
            static_cast<float *>(p)[i] = x;
            return;
        }
        if (t == Dtype::F16) {
            static_cast<std::uint16_t *>(p)[i] = static_cast<std::uint16_t>(_mm_cvtsi128_si32(
                _mm_cvtps_ph(_mm_set_ss(x), 0)));
            return;
        }
        std::uint32_t b;
        std::memcpy(&b, &x, 4);
        static_cast<std::uint16_t *>(p)[i] = static_cast<std::uint16_t>((b + 0x7fff + ((b >> 16) & 1)) >> 16);
    }
}

/**
 * @brief Computes gradients for a two-dimensional RMSNorm operation.
 *
 * Given
 * @f$y_{r,i}=x_{r,i} w_i / \sqrt{\frac{1}{H}\sum_j x_{r,j}^2+\varepsilon}@f$,
 * the function computes gradients with respect to @p x and @p w. Per-row
 * inverse RMS values and correction dot products are calculated first, then
 * the input gradient and weight gradient are produced in separate OpenMP
 * passes.
 *
 * @param gi Destination input-gradient tensor with shape `[rows, hidden]`.
 * @param gw Destination scale-gradient tensor with shape `[hidden]`.
 * @param go Upstream output gradient with shape `[rows, hidden]`.
 * @param x Forward input tensor with shape `[rows, hidden]`.
 * @param w Forward scale vector with shape `[hidden]`.
 * @param eps Positive finite numerical-stability constant used by the forward
 *        normalization.
 *
 * @throws std::invalid_argument If tensors are not compatible CPU
 *         floating-point tensors or if @p eps is not positive and finite.
 *
 * @note The destination tensors are overwritten; this implementation does not
 *       accumulate into existing gradients.
 */
void rmsnorm_backward_cpu(Tensor &gi, Tensor &gw, const Tensor &go, const Tensor &x, const Tensor &w, float eps) {
    if (gi.device_type() != DeviceType::CPU || gw.device_type() != DeviceType::CPU || go.device_type() !=
        DeviceType::CPU || x.device_type() != DeviceType::CPU || w.device_type() != DeviceType::CPU || x.dim() != 2 ||
        go.shape() != x.shape() || gi.shape() != x.shape() || w.dim() != 1 || gw.dim() != 1 || w.shape()[0] != x.shape()
        [1] || gw.shape()[0] != x.shape()[1] || !is_floating_point(x.dtype()) || x.dtype() != gi.dtype() || x.dtype() !=
        gw.dtype() || x.dtype() != go.dtype() || x.dtype() != w.dtype() || !(eps > 0 && std::isfinite(eps)))
        throw
                std::invalid_argument("CPU RMSNorm backward: incompatible tensors or epsilon");
    const auto rows = x.shape()[0];
    const auto hidden = x.shape()[1];
    const Dtype t = x.dtype();
    const auto *xp = x.raw_data();
    const auto *gp = go.raw_data();
    const auto *wp = w.raw_data();
    std::vector<float> inv(rows), dot(rows);
#pragma omp parallel for schedule(static)
    for (std::int64_t r = 0; r < static_cast<std::int64_t>(rows); ++r) {
        const auto base = static_cast<std::size_t>(r) * hidden;
        float ss = 0, dd = 0;
        for (std::size_t i = 0; i < hidden; ++i) {
            const float xv = load1(xp, t, base + i), dy = load1(gp, t, base + i), wv = load1(wp, t, i);
            ss = std::fma(xv, xv, ss);
            dd = std::fma(dy * wv, xv, dd);
        }
        inv[static_cast<std::size_t>(r)] = 1.0f / std::sqrt(ss / static_cast<float>(hidden) + eps);
        dot[static_cast<std::size_t>(r)] = dd;
    }
#pragma omp parallel for schedule(static)
    for (std::int64_t r = 0; r < static_cast<std::int64_t>(rows); ++r) {
        const auto rr = static_cast<std::size_t>(r);
        const auto base = rr * hidden;
        const float iv = inv[rr];
        const float corr = dot[rr] * iv * iv * iv / static_cast<float>(hidden);
        for (std::size_t i = 0; i < hidden; ++i) {
            const float xv = load1(xp, t, base + i), dy = load1(gp, t, base + i), wv = load1(wp, t, i);
            store1(gi.raw_data(), t, base + i, dy * wv * iv - xv * corr);
        }
    }
#pragma omp parallel for schedule(static)
    for (std::int64_t i = 0; i < static_cast<std::int64_t>(hidden); ++i) {
        const auto j = static_cast<std::size_t>(i);
        float s = 0;
        for (std::size_t r = 0; r < rows; ++r) {
            const auto base = r * hidden;
            s = std::fma(load1(gp, t, base + j), load1(xp, t, base + j) * inv[r], s);
        }
        store1(gw.raw_data(), t, j, s);
    }
}
