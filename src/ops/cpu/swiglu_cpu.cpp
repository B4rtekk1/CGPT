#include "ops/cpu/swiglu_cpu.h"
#include "core/simd_math.h"
#include <immintrin.h>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>

namespace {
    inline __m256 load8(const void *p, Dtype t, std::size_t i) {
        if (t == Dtype::F32) return _mm256_loadu_ps(static_cast<const float *>(p) + i);
        if (t == Dtype::F16) return _mm256_cvtph_ps(
            _mm_loadu_si128(reinterpret_cast<const __m128i *>(static_cast<const std::uint16_t *>(p) + i)));
        return _mm256_castsi256_ps(_mm256_slli_epi32(
            _mm256_cvtepu16_epi32(
                _mm_loadu_si128(reinterpret_cast<const __m128i *>(static_cast<const std::uint16_t *>(p) + i))), 16));
    }

    inline float load1(const void *p, Dtype t, std::size_t i) {
        if (t == Dtype::F32)return static_cast<const float *>(p)[i];
        const auto h = static_cast<const std::uint16_t *>(p)[i];
        if (t == Dtype::F16)return _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(h)));
        std::uint32_t b = std::uint32_t(h) << 16;
        float x;
        std::memcpy(&x, &b, 4);
        return x;
    }

    inline void store8(void *p, Dtype t, std::size_t i, __m256 v) {
        if (t == Dtype::F32) {
            _mm256_storeu_ps(static_cast<float *>(p) + i, v);
            return;
        }
        if (t == Dtype::F16) {
            _mm_storeu_si128(reinterpret_cast<__m128i *>(static_cast<std::uint16_t *>(p) + i), _mm256_cvtps_ph(v, 0));
            return;
        }
        const __m256i b = _mm256_castps_si256(v), r = _mm256_add_epi32(
            b, _mm256_add_epi32(_mm256_set1_epi32(0x7fff),
                                _mm256_and_si256(_mm256_srli_epi32(b, 16), _mm256_set1_epi32(1))));
        const __m256i h = _mm256_srli_epi32(r, 16);
        _mm_storeu_si128(reinterpret_cast<__m128i *>(static_cast<std::uint16_t *>(p) + i),
                         _mm_packus_epi32(_mm256_castsi256_si128(h), _mm256_extracti128_si256(h, 1)));
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

    inline void check(Tensor &o, const Tensor &a, const Tensor *b = nullptr) {
        if (o.device_type() != DeviceType::CPU || a.device_type() != DeviceType::CPU || (
                b && b->device_type() != DeviceType::CPU) || o.shape() != a.shape() || (b && b->shape() != a.shape()) ||
            o.dtype() != a.dtype() || (b && b->dtype() != a.dtype()) || !is_floating_point(o.dtype()))throw
                std::invalid_argument("CPU SwiGLU: matching CPU floating-point tensors required");
    }

    template<bool Fused>
    void apply(Tensor &o, const Tensor &in, const Tensor *up) {
        const auto w = Fused ? o.shape().back() : o.numel(), rows = Fused ? o.numel() / w : 1;
        auto *d = o.raw_data();
        const auto *p = in.raw_data();
#pragma omp parallel for schedule(static)
        for (std::int64_t r = 0; r < static_cast<std::int64_t>(rows); ++r) {
            const auto rr = std::size_t(r), stride = dtype_size(o.dtype());
            const auto *g = static_cast<const std::uint8_t *>(p) + (Fused ? rr * 2 * w : rr * w) * stride;
            const auto *u = Fused
                                ? g + w * stride
                                : static_cast<const std::uint8_t *>(up->raw_data()) + rr * w * stride;
            std::size_t i = 0;
            for (; i + 31 < w; i += 32)for (int k = 0; k < 4; ++k) {
                auto x = load8(g, o.dtype(), k * 8 + i), y = load8(u, o.dtype(), k * 8 + i);
                auto z = cgpt::cpu::simd::silu256_ps(x);
                store8(d, o.dtype(), rr * w + i + k * 8, _mm256_mul_ps(z, y));
            }
            for (; i < w; ++i) {
                auto x = load1(g, o.dtype(), i), y = load1(u, o.dtype(), i);
                store1(d, o.dtype(), rr * w + i, x / (1 + std::exp(-x)) * y);
            }
        }
    }
}

void swiglu_forward_cpu(Tensor &output, const Tensor &gate, const Tensor &up) {
    check(output, gate, &up);
    apply<false>(output, gate, &up);
}

void swiglu_forward_cpu(Tensor &output, const Tensor &gate_up) {
    if (output.device_type() != DeviceType::CPU || gate_up.device_type() != DeviceType::CPU || !
        is_floating_point(output.dtype()) || output.dtype() != gate_up.dtype() || gate_up.shape().empty() || gate_up.
        shape().back() % 2 || output.shape().size() != gate_up.shape().size() || output.shape().back() != gate_up.
        shape().back() / 2)throw std::invalid_argument("CPU SwiGLU: invalid fused tensors");
    for (std::size_t i = 0; i + 1 < output.shape().size(); ++i)if (output.shape()[i] != gate_up.shape()[i])throw
            std::invalid_argument("CPU SwiGLU: shape mismatch");
    apply<true>(output, gate_up, nullptr);
}
