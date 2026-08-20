#include "ops/cpu/rope_backward_cpu.h"

#include <immintrin.h>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>

namespace {
    float load1(const void *p, Dtype t, std::size_t i) {
        if (t == Dtype::F32)return static_cast<const float *>(p)[i];
        const auto h = static_cast<const std::uint16_t *>(p)[i];
        if (t == Dtype::F16)return _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(h)));
        std::uint32_t b = std::uint32_t(h) << 16;
        float x;
        std::memcpy(&x, &b, 4);
        return x;
    }

    void store1(void *p, Dtype t, std::size_t i, float x) {
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

    void check(const Tensor &a, const Tensor &b, const Tensor &c, const Tensor &d, const Tensor &co, const Tensor &si,
               const RopeOptions &o) {
        if (a.device_type() != DeviceType::CPU || b.device_type() != DeviceType::CPU || c.device_type() !=
            DeviceType::CPU || d.device_type() != DeviceType::CPU || co.device_type() != DeviceType::CPU || si.
            device_type() != DeviceType::CPU || a.shape() != c.shape() || b.shape() != d.shape() || a.shape()[0] != b.
            shape()[0] || a.shape()[1] != b.shape()[1] || a.shape()[3] != b.shape()[3] || a.dtype() != b.dtype() || a.
            dtype() != c.dtype() || a.dtype() != d.dtype() || a.dtype() != co.dtype() || co.shape() != si.shape() || !
            is_floating_point(a.dtype()))throw std::invalid_argument("CPU RoPE backward: incompatible tensors");
        const auto rd = o.rotary_dim ? o.rotary_dim : a.shape()[3];
        if (!rd || rd > a.shape()[3] || (rd & 1) || co.dim() != 2 || co.shape()[1] != rd / 2 || a.shape()[1] > co.
            shape()[0] || o.position_offset > co.shape()[0] - a.shape()[1])throw std::invalid_argument(
            "CPU RoPE backward: invalid rotary dimension or cache range");
    }
}

void rope_backward_cpu(Tensor &gq, Tensor &gk, const Tensor &rq, const Tensor &rk, const Tensor &co, const Tensor &si,
                       const RopeOptions &o) {
    check(gq, gk, rq, rk, co, si, o);
    const auto B = gq.shape()[0], S = gq.shape()[1], Q = gq.shape()[2], K = gk.shape()[2], D = gq.shape()[3], R = (
        o.rotary_dim ? o.rotary_dim : D), P = R / 2;
    const auto t = gq.dtype();
#pragma omp parallel for schedule(static)
    for (std::int64_t z = 0; z < static_cast<std::int64_t>(B * S); ++z) {
        const auto token = std::size_t(z), pos = o.position_offset + token % S;
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
