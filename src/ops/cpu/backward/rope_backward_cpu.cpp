/**
 * @file rope_backward_cpu.cpp
 * @brief CPU implementation of the backward pass for rotary position embeddings.
 *
 * Applies the inverse two-dimensional RoPE rotation to query and key gradients.
 * F32, F16, and BF16 tensors are supported, and batch-position pairs are
 * parallelized with OpenMP.
 */
#include "ops/cpu/rope_backward_cpu.h"

#include <immintrin.h>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>

namespace {
    /**
     * @brief Loads one element and converts it to F32.
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
     * @brief Stores one F32 value using the requested tensor representation.
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
            static_cast<std::uint16_t *>(p)[i] = static_cast<std::uint16_t>(_mm_cvtsi128_si32(_mm_cvtps_ph(_mm_set_ss(x), 0)));
            return;
        }
        std::uint32_t b;
        std::memcpy(&b, &x, 4);
        static_cast<std::uint16_t *>(p)[i] = static_cast<std::uint16_t>((b + 0x7fff + ((b >> 16) & 1)) >> 16);
    }

    /**
     * @brief Validates RoPE backward tensors, cache dimensions, and options.
     *
     * Query and key tensors must use `[B, S, H, D]` layouts. Gradient outputs
     * must match their corresponding rotated-gradient inputs. The cosine and
     * sine caches must have shape `[cache_length, rotary_dim / 2]`.
     *
     * @param a Query-gradient output tensor.
     * @param b Key-gradient output tensor.
     * @param c Rotated query-gradient input tensor.
     * @param d Rotated key-gradient input tensor.
     * @param co Cosine cache.
     * @param si Sine cache.
     * @param o RoPE configuration.
     * @throws std::invalid_argument If tensor metadata or cache bounds are invalid.
     */
    void check(const Tensor &a, const Tensor &b, const Tensor &c, const Tensor &d, const Tensor &co, const Tensor &si,
               const RopeOptions &o) {
        if (a.device_type() != DeviceType::CPU || b.device_type() != DeviceType::CPU || c.device_type() !=
            DeviceType::CPU || d.device_type() != DeviceType::CPU || co.device_type() != DeviceType::CPU || si.
            device_type() != DeviceType::CPU || a.shape() != c.shape() || b.shape() != d.shape() || a.shape()[0] != b.
            shape()[0] || a.shape()[1] != b.shape()[1] || a.shape()[3] != b.shape()[3] || a.dtype() != b.dtype() || a.
            dtype() != c.dtype() || a.dtype() != d.dtype() || a.dtype() != co.dtype() || co.shape() != si.shape() || !
            is_floating_point(a.dtype()))
            throw std::invalid_argument("CPU RoPE backward: incompatible tensors");
        const auto rd = o.rotary_dim ? o.rotary_dim : a.shape()[3];
        if (!rd || rd > a.shape()[3] || (rd & 1) || co.dim() != 2 || co.shape()[1] != rd / 2 || a.shape()[1] > co.
            shape()[0] || o.position_offset > co.shape()[0] - a.shape()[1])
            throw std::invalid_argument(
                "CPU RoPE backward: invalid rotary dimension or cache range");
    }
}

/**
 * @brief Applies the RoPE backward transformation to query and key gradients.
 *
 * For each rotated pair `(x, y)` and cached angle `(c, s)`, the inverse
 * rotation is
 * @f$g_x = x c + y s@f$ and @f$g_y = -x s + y c@f$.
 * Dimensions outside @c rotary_dim are copied unchanged. The operation is
 * out-of-place and parallelized across batch-position pairs.
 *
 * @param gq Destination gradient with respect to the unrotated query tensor.
 * @param gk Destination gradient with respect to the unrotated key tensor.
 * @param rq Gradient with respect to the rotated query tensor, shape `[B,S,Q,D]`.
 * @param rk Gradient with respect to the rotated key tensor, shape `[B,S,K,D]`.
 * @param co Cosine cache with shape `[cache_length, rotary_dim / 2]`.
 * @param si Sine cache with the same shape and type as @p co.
 * @param o RoPE options, including rotary dimension and position offset.
 *
 * @throws std::invalid_argument If tensors, data types, dimensions, or cache
 *         ranges are incompatible.
 */
void rope_backward_cpu(Tensor &gq, Tensor &gk, const Tensor &rq, const Tensor &rk, const Tensor &co, const Tensor &si,
                       const RopeOptions &o) {
    check(gq, gk, rq, rk, co, si, o);
    const auto B = gq.shape()[0], S = gq.shape()[1], Q = gq.shape()[2], K = gk.shape()[2], D = gq.shape()[3], R = (
        o.rotary_dim ? o.rotary_dim : D), P = R / 2;
    const auto t = gq.dtype();
#pragma omp parallel for schedule(static)
    for (std::int64_t z = 0; z < static_cast<std::int64_t>(B * S); ++z) {
        const auto token = static_cast<std::size_t>(z), pos = o.position_offset + token % S;
        for (std::size_t h = 0; h < Q; ++h) {
            const auto base = (token * Q + h) * D;
            for (std::size_t p = 0; p < P; ++p) {
                const float c = load1(co.raw_data(), t, pos * P + p), s = load1(si.raw_data(), t, pos * P + p), x =
                        load1(rq.raw_data(), t, base + 2 * p), y = load1(rq.raw_data(), t, base + 2 * p + 1);
                store1(gq.raw_data(), t, base + 2 * p, std::fma(y, s, x * c));
                store1(gq.raw_data(), t, base + 2 * p + 1, std::fma(-x, s, y * c));
            }
            for (std::size_t d = R; d < D; ++d)store1(gq.raw_data(), t, base + d, load1(rq.raw_data(), t, base + d));
        }
        for (std::size_t h = 0; h < K; ++h) {
            const auto base = (token * K + h) * D;
            for (std::size_t p = 0; p < P; ++p) {
                const float c = load1(co.raw_data(), t, pos * P + p), s = load1(si.raw_data(), t, pos * P + p), x =
                        load1(rk.raw_data(), t, base + 2 * p), y = load1(rk.raw_data(), t, base + 2 * p + 1);
                store1(gk.raw_data(), t, base + 2 * p, std::fma(y, s, x * c));
                store1(gk.raw_data(), t, base + 2 * p + 1, std::fma(-x, s, y * c));
            }
            for (std::size_t d = R; d < D; ++d)store1(gk.raw_data(), t, base + d, load1(rk.raw_data(), t, base + d));
        }
    }
}
