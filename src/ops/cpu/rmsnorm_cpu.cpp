/**
 * @file rmsnorm_cpu.cpp
 * @brief AVX2-accelerated CPU implementation of RMS normalization.
 *
 * The implementation normalizes each row of a two-dimensional tensor by its
 * root-mean-square value and applies a learned per-feature scale. Computation
 * is performed in single precision for F32, F16, and BF16 tensors, while the
 * result is converted back to the original tensor data type.
 */

#include "ops/cpu/rmsnorm_cpu.h"
#include <immintrin.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace {
    /**
     * @brief Loads one tensor element and converts it to single precision.
     *
     * F32 values are loaded directly. F16 values are converted with F16C,
     * while BF16 values are reconstructed by placing their bits in the upper
     * half of an IEEE 754 single-precision value.
     *
     * @param p Pointer to the beginning of the tensor storage.
     * @param t Data type of the stored elements.
     * @param i Zero-based element index.
     * @return Element value converted to `float`.
     */
    inline float load1(const void *p, Dtype t, std::size_t i) {
        if (t == Dtype::F32)return static_cast<const float *>(p)[i];
        auto h = static_cast<const std::uint16_t *>(p)[i];
        if (t == Dtype::F16)return _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(h)));
        std::uint32_t b = static_cast<std::uint32_t>(h) << 16;
        float x;
        std::memcpy(&x, &b, 4);
        return x;
    }

    /**
     * @brief Stores one single-precision value in the requested tensor format.
     *
     * F16 conversion uses F16C. BF16 conversion applies round-to-nearest-even
     * before discarding the lower 16 bits of the single-precision value.
     *
     * @param p Pointer to the beginning of the tensor storage.
     * @param t Destination element data type.
     * @param i Zero-based destination element index.
     * @param x Value to store.
     */
    inline void store1(void *p, Dtype t, std::size_t i, float x) {
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

    /**
     * @brief Computes the reciprocal root-mean-square normalization factor.
     *
     * The function evaluates
     * \f$1 / \sqrt{\mathrm{sum}/\mathrm{hidden}+\varepsilon}\f$ using the
     * hardware reciprocal-square-root approximation followed by one
     * Newton-Raphson refinement step.
     *
     * @param sum Sum of squared values in one input row.
     * @param hidden Number of features in the row.
     * @param eps Positive numerical-stability constant.
     * @return Refined reciprocal RMS value.
     */
    inline float inverse_rms(float sum, std::size_t hidden, float eps) {
        const float value = sum / static_cast<float>(hidden) + eps;
        __m128 x = _mm_rsqrt_ss(_mm_set_ss(value));
        // One Newton-Raphson refinement: x <- x * (3 - value*x*x) / 2.
        x = _mm_mul_ss(x, _mm_mul_ss(_mm_set_ss(0.5f),
                                     _mm_sub_ss(_mm_set_ss(3.0f), _mm_mul_ss(_mm_set_ss(value), _mm_mul_ss(x, x)))));
        return _mm_cvtss_f32(x);
    }
}

/**
 * @brief Applies RMS normalization to a two-dimensional CPU tensor.
 *
 * Each row is normalized independently according to
 * \f$
 *   y_{r,i} = x_{r,i} w_i /
 *   \sqrt{\frac{1}{H}\sum_{j=0}^{H-1}x_{r,j}^{2}+\varepsilon}.
 * \f$
 * Rows are distributed across OpenMP threads. The squared-sum reduction and
 * output scaling use AVX2/FMA operations where possible, with scalar handling
 * for remaining elements. Intermediate arithmetic is performed in F32.
 *
 * @param out Preallocated output tensor with the same shape and type as @p in.
 * @param in Non-empty two-dimensional input tensor of shape `[rows, hidden]`.
 * @param w One-dimensional scale tensor containing `hidden` elements.
 * @param eps Positive finite numerical-stability constant.
 *
 * @throws std::invalid_argument If tensors are not CPU-resident floating-point
 *         tensors, their shapes or data types are incompatible, or @p eps is
 *         not positive and finite.
 *
 * @note Supported tensor data types are the floating-point types recognized by
 *       `is_floating_point`, with explicit conversion paths for F32, F16, and
 *       BF16 storage.
 * @note The function requires a build target supporting the AVX2, FMA, and F16C
 *       instruction sets.
 */
void rmsnorm_forward_cpu(Tensor &out, const Tensor &in, const Tensor &w, float eps) {
    if (out.device_type() != DeviceType::CPU || in.device_type() != DeviceType::CPU || w.device_type() !=
        DeviceType::CPU || in.dim() != 2 || w.dim() != 1 || out.shape() != in.shape() || w.shape()[0] != in.shape()[1]
        || out.dtype() != in.dtype() || w.dtype() != in.dtype() || !is_floating_point(in.dtype()) || !(
            eps > 0 && std::isfinite(eps)))
        throw std::invalid_argument("CPU RMSNorm: incompatible tensors or epsilon");
    const auto rows = in.shape()[0];
    const auto hidden = in.shape()[1];
    const Dtype t = in.dtype();
    std::vector<float> wf(hidden);
    for (std::size_t i = 0; i < hidden; ++i)wf[i] = load1(w.raw_data(), t, i);
    const auto *src = in.raw_data();
    auto *dst = out.raw_data();
#pragma omp parallel for schedule(static)
    for (std::int64_t r = 0; r < static_cast<std::int64_t>(rows); ++r) {
        const std::size_t base = static_cast<std::size_t>(r) * hidden;
        __m256 a = _mm256_setzero_ps(), b = a, c = a, d = a;
        std::size_t i = 0;
        for (; i + 31 < hidden; i += 32) {
            for (int j = 0; j < 4; ++j) {
                alignas(32) float x[8];
                for (int z = 0; z < 8; ++z)x[z] = load1(src, t, base + i + j * 8 + z);
                const auto v = _mm256_load_ps(x);
                const auto q = _mm256_fmadd_ps(v, v, _mm256_setzero_ps());
                if (j == 0)a = _mm256_add_ps(a, q);
                else if (j == 1)b = _mm256_add_ps(b, q);
                else if (j == 2)c = _mm256_add_ps(c, q);
                else d = _mm256_add_ps(d, q);
            }
        }
        alignas(32) float sums[8];
        _mm256_store_ps(sums, _mm256_add_ps(_mm256_add_ps(a, b), _mm256_add_ps(c, d)));
        float ss = 0;
        for (float x: sums)ss += x;
        for (; i < hidden; ++i) {
            float x = load1(src, t, base + i);
            ss = std::fma(x, x, ss);
        }
        const float inv = inverse_rms(ss, hidden, eps);
        const auto vi = _mm256_set1_ps(inv);
        for (i = 0; i + 7 < hidden; i += 8) {
            alignas(32)float x[8];
            for (int z = 0; z < 8; ++z)x[z] = load1(src, t, base + i + z);
            const auto vx = _mm256_load_ps(x);
            const auto vw = _mm256_loadu_ps(wf.data() + i);
            _mm256_store_ps(x, _mm256_mul_ps(_mm256_mul_ps(vx, vw), vi));
            for (int z = 0; z < 8; ++z)store1(dst, t, base + i + z, x[z]);
        }
        for (; i < hidden; ++i)store1(dst, t, base + i, load1(src, t, base + i) * wf[i] * inv);
    }
}

/**
 * @brief Allocates and returns the RMS-normalized result.
 *
 * This convenience overload creates a CPU output tensor matching the input
 * shape and data type, then delegates computation to the preallocated-output
 * overload.
 *
 * @param in Non-empty two-dimensional CPU input tensor.
 * @param w One-dimensional per-feature scale tensor.
 * @param eps Positive finite numerical-stability constant.
 * @return Newly allocated CPU tensor containing the normalized values.
 *
 * @throws std::invalid_argument Under the same conditions as the
 *         preallocated-output overload.
 */
Tensor rmsnorm_forward_cpu(const Tensor &in, const Tensor &w, float eps) {
    Tensor out(in.shape(), in.dtype(), DeviceType::CPU);
    rmsnorm_forward_cpu(out, in, w, eps);
    return out;
}
