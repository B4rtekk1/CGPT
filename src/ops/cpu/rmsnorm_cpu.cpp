#include "ops/cpu/rmsnorm_cpu.h"
#include <immintrin.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace {
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
        if (t == Dtype::F16) {
            static_cast<std::uint16_t *>(p)[i] = std::uint16_t(_mm_cvtsi128_si32(_mm_cvtps_ph(_mm_set_ss(x), 0)));
            return;
        }
        std::uint32_t b;
        std::memcpy(&b, &x, 4);
        static_cast<std::uint16_t *>(p)[i] = std::uint16_t((b + 0x7fff + ((b >> 16) & 1)) >> 16);
    }

    inline float inverse_rms(float sum, std::size_t hidden, float eps) {
        const float value = sum / float(hidden) + eps;
        __m128 x = _mm_rsqrt_ss(_mm_set_ss(value));
        // One Newton-Raphson refinement: x <- x * (3 - value*x*x) / 2.
        x = _mm_mul_ss(x, _mm_mul_ss(_mm_set_ss(1.5f),
                                     _mm_sub_ss(_mm_set_ss(3.0f), _mm_mul_ss(_mm_set_ss(value), _mm_mul_ss(x, x)))));
        return _mm_cvtss_f32(x);
    }
}

void rmsnorm_forward_cpu(Tensor &out, const Tensor &in, const Tensor &w, float eps) {
    if (out.device_type() != DeviceType::CPU || in.device_type() != DeviceType::CPU || w.device_type() !=
        DeviceType::CPU || in.dim() != 2 || w.dim() != 1 || out.shape() != in.shape() || w.shape()[0] != in.shape()[1]
        || out.dtype() != in.dtype() || w.dtype() != in.dtype() || !is_floating_point(in.dtype()) || !(
            eps > 0 && std::isfinite(eps))) throw std::invalid_argument("CPU RMSNorm: incompatible tensors or epsilon");
    const auto rows = in.shape()[0];
    const auto hidden = in.shape()[1];
    const Dtype t = in.dtype();
    std::vector<float> wf(hidden);
    for (std::size_t i = 0; i < hidden; ++i)wf[i] = load1(w.raw_data(), t, i);
    const auto *src = in.raw_data();
    auto *dst = out.raw_data();
#pragma omp parallel for schedule(static)
    for (std::int64_t r = 0; r < static_cast<std::int64_t>(rows); ++r) {
        const std::size_t base = std::size_t(r) * hidden;
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

Tensor rmsnorm_forward_cpu(const Tensor &in, const Tensor &w, float eps) {
    Tensor out(in.shape(), in.dtype(), DeviceType::CPU);
    rmsnorm_forward_cpu(out, in, w, eps);
    return out;
}
