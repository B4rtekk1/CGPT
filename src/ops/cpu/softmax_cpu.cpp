#include "ops/cpu/softmax_cpu.h"
#include "core/simd_math.h"
#include <immintrin.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>

namespace {
    inline __m256 load8(const void *p, Dtype t, std::size_t i) {
        if (t == Dtype::F32)return _mm256_loadu_ps(static_cast<const float *>(p) + i);
        if (t == Dtype::F16)
            return _mm256_cvtph_ps(
                _mm_loadu_si128(reinterpret_cast<const __m128i *>(static_cast<const std::uint16_t *>(p) + i)));
        return _mm256_castsi256_ps(_mm256_slli_epi32(
            _mm256_cvtepu16_epi32(
                _mm_loadu_si128(reinterpret_cast<const __m128i *>(static_cast<const std::uint16_t *>(p) + i))), 16));
    }

    inline float load1(const void *p, Dtype t, std::size_t i) {
        if (t == Dtype::F32)return static_cast<const float *>(p)[i];
        auto h = static_cast<const std::uint16_t *>(p)[i];
        if (t == Dtype::F16)return _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(h)));
        std::uint32_t b = std::uint32_t(h) << 16;
        float x;
        std::memcpy(&x, &b, 4);
        return x;
    }

    inline void store1(void *p, Dtype t, std::size_t i, float x) {
        if (t == Dtype::F32) {
            static_cast<float *>(p)[i] = x;
            return;
        }
        std::uint32_t b;
        std::memcpy(&b, &x, 4);
        if (t == Dtype::BF16) {
            static_cast<std::uint16_t *>(p)[i] = std::uint16_t((b + 0x7fff + ((b >> 16) & 1)) >> 16);
            return;
        }
        static_cast<std::uint16_t *>(p)[i] = std::uint16_t(_mm_cvtsi128_si32(_mm_cvtps_ph(_mm_set_ss(x), 0)));
    }
}

void softmax_forward_cpu(Tensor &output, const Tensor &input) {
    if (output.device_type() != DeviceType::CPU || input.device_type() != DeviceType::CPU || input.dim() != 2 || input.
        size(0) == 0 || input.size(1) == 0 || output.shape() != input.shape() || output.dtype() != input.dtype() || !
        is_floating_point(input.dtype()))
        throw std::invalid_argument(
            "CPU softmax: matching non-empty 2D CPU floating-point tensors required");
    const auto rows = input.size(0);
    const auto cols = input.size(1);
    const Dtype t = input.dtype();
    const auto *src = input.raw_data();
    auto *dst = output.raw_data();
#pragma omp parallel for schedule(static)
    for (std::int64_t r = 0; r < static_cast<std::int64_t>(rows); ++r) {
        const auto base = std::size_t(r) * cols;
        float m = -INFINITY;
        __m256 m0 = _mm256_set1_ps(-INFINITY), m1 = m0, m2 = m0, m3 = m0;
        std::size_t i = 0;
        // For F16/BF16 the conversion is vectorized; F32 uses direct loads.
        for (; i + 31 < cols; i += 32) {
            m0 = _mm256_max_ps(m0, load8(src, t, base + i));
            m1 = _mm256_max_ps(m1, load8(src, t, base + i + 8));
            m2 = _mm256_max_ps(m2, load8(src, t, base + i + 16));
            m3 = _mm256_max_ps(m3, load8(src, t, base + i + 24));
        }
        alignas(32) float q[8];
        _mm256_store_ps(q, _mm256_max_ps(_mm256_max_ps(m0, m1), _mm256_max_ps(m2, m3)));
        for (float x: q)m = std::max(m, x);
        for (; i < cols; ++i)m = std::max(m, load1(src, t, base + i));
        __m256 s0 = _mm256_setzero_ps(), s1 = s0, s2 = s0, s3 = s0;
        const auto vm = _mm256_set1_ps(m);
        i = 0;
        for (; i + 31 < cols; i += 32) {
            s0 = _mm256_add_ps(s0, cgpt::cpu::simd::exp256_ps(_mm256_sub_ps(load8(src, t, base + i), vm)));
            s1 = _mm256_add_ps(s1, cgpt::cpu::simd::exp256_ps(_mm256_sub_ps(load8(src, t, base + i + 8), vm)));
            s2 = _mm256_add_ps(s2, cgpt::cpu::simd::exp256_ps(_mm256_sub_ps(load8(src, t, base + i + 16), vm)));
            s3 = _mm256_add_ps(s3, cgpt::cpu::simd::exp256_ps(_mm256_sub_ps(load8(src, t, base + i + 24), vm)));
        }
        _mm256_store_ps(q, _mm256_add_ps(_mm256_add_ps(s0, s1), _mm256_add_ps(s2, s3)));
        float sum = 0;
        for (float x: q)sum += x;
        for (; i < cols; ++i)sum += std::exp(load1(src, t, base + i) - m);
        for (i = 0; i < cols; ++i)store1(dst, t, base + i, std::exp(load1(src, t, base + i) - m) / sum);
    }
}
