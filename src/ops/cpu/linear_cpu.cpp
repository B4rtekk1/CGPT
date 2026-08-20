#include "ops/cpu/linear_cpu.h"

#include <immintrin.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>

namespace {
    inline __m256 load8(const void *p, Dtype t, std::size_t i) {
        if (t == Dtype::F32) return _mm256_loadu_ps(static_cast<const float *>(p) + i);
        const auto *h = static_cast<const std::uint16_t *>(p) + i;
        if (t == Dtype::F16) return _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i *>(h)));
        return _mm256_castsi256_ps(_mm256_slli_epi32(
            _mm256_cvtepu16_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i *>(h))), 16));
    }

    inline float load1(const void *p, Dtype t, std::size_t i) {
        if (t == Dtype::F32) return static_cast<const float *>(p)[i];
        const auto h = static_cast<const std::uint16_t *>(p)[i];
        if (t == Dtype::F16) return _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(h)));
        std::uint32_t bits = std::uint32_t(h) << 16;
        float x;
        std::memcpy(&x, &bits, 4);
        return x;
    }

    inline void store1(void *p, Dtype t, std::size_t i, float x) {
        if (t == Dtype::F32) {
            static_cast<float *>(p)[i] = x;
            return;
        }
        if (t == Dtype::F16) {
            static_cast<std::uint16_t *>(p)[i] = std::uint16_t(_mm_cvtsi128_si32(_mm_cvtps_ph(_mm_set_ss(x), 0)));
            return;
        }
        std::uint32_t bits;
        std::memcpy(&bits, &x, 4);
        static_cast<std::uint16_t *>(p)[i] = std::uint16_t((bits + 0x7fff + ((bits >> 16) & 1)) >> 16);
    }

    inline void validate(const Tensor &out, const Tensor &in, const Tensor &w, const Tensor *bias) {
        if (out.device_type() != DeviceType::CPU || in.device_type() != DeviceType::CPU || w.device_type() !=
            DeviceType::CPU ||
            (bias && bias->device_type() != DeviceType::CPU) || in.dim() < 1 || w.dim() != 2 || out.dim() != in.dim() ||
            !is_floating_point(in.dtype()) || in.dtype() != w.dtype() || out.dtype() != in.dtype() ||
            (bias && (bias->dim() != 1 || bias->dtype() != in.dtype() || bias->shape()[0] != w.shape()[0])))
            throw std::invalid_argument("CPU linear: incompatible CPU floating-point tensors");
        if (in.shape().back() != w.shape()[1] || out.shape().back() != w.shape()[0]) throw std::invalid_argument(
            "CPU linear: shape mismatch");
        for (std::size_t i = 0; i + 1 < in.dim(); ++i) if (out.shape()[i] != in.shape()[i]) throw std::invalid_argument(
            "CPU linear: leading shape mismatch");
    }

    void apply(Tensor &out, const Tensor &in, const Tensor &w, const Tensor *bias) {
        validate(out, in, w, bias);
        const auto rows = in.numel() / in.shape().back(), I = in.shape().back(), O = w.shape()[0];
        const auto t = in.dtype();
        const auto *x = in.raw_data();
        const auto *weights = w.raw_data();
        const auto *b = bias ? bias->raw_data() : nullptr;
        auto *y = out.raw_data();
#pragma omp parallel for schedule(static)
        for (std::int64_t r = 0; r < static_cast<std::int64_t>(rows); ++r) {
            const std::size_t rb = std::size_t(r) * I, ob = std::size_t(r) * O;
            std::size_t o = 0;
            for (; o + 7 < O; o += 8) {
                __m256 a0 = _mm256_setzero_ps(), a1 = a0, a2 = a0, a3 = a0, a4 = a0, a5 = a0, a6 = a0, a7 = a0;
                std::size_t i = 0;
                for (; i + 7 < I; i += 8) {
                    const auto vx = load8(x, t, rb + i);
                    a0 = _mm256_fmadd_ps(vx, load8(weights, t, (o + 0) * I + i), a0);
                    a1 = _mm256_fmadd_ps(vx, load8(weights, t, (o + 1) * I + i), a1);
                    a2 = _mm256_fmadd_ps(vx, load8(weights, t, (o + 2) * I + i), a2);
                    a3 = _mm256_fmadd_ps(vx, load8(weights, t, (o + 3) * I + i), a3);
                    a4 = _mm256_fmadd_ps(vx, load8(weights, t, (o + 4) * I + i), a4);
                    a5 = _mm256_fmadd_ps(vx, load8(weights, t, (o + 5) * I + i), a5);
                    a6 = _mm256_fmadd_ps(vx, load8(weights, t, (o + 6) * I + i), a6);
                    a7 = _mm256_fmadd_ps(vx, load8(weights, t, (o + 7) * I + i), a7);
                }
                float sums[8];
                for (int z = 0; z < 8; ++z) {
                    __m256 v[8] = {a0, a1, a2, a3, a4, a5, a6, a7};
                    alignas(32) float q[8];
                    _mm256_store_ps(q, v[z]);
                    float s = 0;
                    for (float n: q)s += n;
                    for (std::size_t j = i; j < I; ++j)s = std::fma(load1(x, t, rb + j),
                                                                    load1(weights, t, (o + z) * I + j), s);
                    if (b)s += load1(b, t, o + z);
                    sums[z] = s;
                    store1(y, t, ob + o + z, s);
                }
            }
            for (; o < O; ++o) {
                float s = b ? load1(b, t, o) : 0;
                for (std::size_t i = 0; i < I; ++i)s = std::fma(load1(x, t, rb + i), load1(weights, t, o * I + i), s);
                store1(y, t, ob + o, s);
            }
        }
    }
}

void linear_forward_cpu(Tensor &o, const Tensor &i, const Tensor &w) { apply(o, i, w, nullptr); }
void linear_forward_cpu(Tensor &o, const Tensor &i, const Tensor &w, const Tensor &b) { apply(o, i, w, &b); }
