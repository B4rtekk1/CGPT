#include "ops/cpu/swiglu_backward_cpu.h"

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

    void check(const Tensor &a, const Tensor &b, const Tensor &c, const Tensor &d, const Tensor &e) {
        if (a.device_type() != DeviceType::CPU || b.device_type() != DeviceType::CPU || c.device_type() !=
            DeviceType::CPU || d.device_type() != DeviceType::CPU || e.device_type() != DeviceType::CPU || a.shape() !=
            c.shape() || b.shape() != c.shape() || d.shape() != c.shape() || e.shape() != c.shape() || a.dtype() != c.
            dtype() || b.dtype() != c.dtype() || d.dtype() != c.dtype() || e.dtype() != c.dtype() || !
            is_floating_point(c.dtype()))throw std::invalid_argument("CPU SwiGLU backward: incompatible tensors");
    }

    inline float silu(float x) {
        const float s = 1.0f / (1.0f + std::exp(-x));
        return x * s;
    }

    inline float silu_grad(float x) {
        const float s = 1.0f / (1.0f + std::exp(-x));
        return s * (1.0f + x * (1.0f - s));
    }
}

void swiglu_backward_cpu(Tensor &gg, Tensor &gu, const Tensor &go, const Tensor &gate, const Tensor &up,
                         const SwiGLUBackwardOptions &o) {
    check(gg, gu, go, gate, up);
    const auto n = go.numel();
    const auto t = go.dtype();
#pragma omp parallel for schedule(static)
    for (std::int64_t i = 0; i < static_cast<std::int64_t>(n); ++i) {
        const auto j = std::size_t(i);
        const float g = load1(go.raw_data(), t, j), x = load1(gate.raw_data(), t, j), u = load1(up.raw_data(), t, j);
        const float dg = g * u * silu_grad(x), du = g * silu(x);
        store1(gg.raw_data(), t, j, dg + (o.accumulate_gate ? load1(gg.raw_data(), t, j) : 0));
        store1(gu.raw_data(), t, j, du + (o.accumulate_up ? load1(gu.raw_data(), t, j) : 0));
    }
}

void swiglu_backward_cpu(Tensor &ggu, const Tensor &go, const Tensor &gu, const SwiGLUBackwardOptions &o) {
    if (ggu.device_type() != DeviceType::CPU || go.device_type() != DeviceType::CPU || gu.device_type() !=
        DeviceType::CPU || gu.dim() < 1 || gu.shape().back() != 2 * go.shape().back() || go.numel() * 2 != gu.numel() ||
        ggu.shape() != gu.shape() || ggu.dtype() != go.dtype() || gu.dtype() != go.dtype() || !
        is_floating_point(go.dtype()))throw std::invalid_argument("CPU SwiGLU backward: invalid fused tensors");
    const auto n = go.numel();
    const auto h = go.shape().back();
    const Dtype t = go.dtype();
#pragma omp parallel for schedule(static)
    for (std::int64_t i = 0; i < static_cast<std::int64_t>(n); ++i) {
        const auto j = std::size_t(i), row = j / h, col = j % h;
        const auto gb = row * 2 * h + col, ub = gb + h;
        const float g = load1(go.raw_data(), t, j), x = load1(gu.raw_data(), t, gb), u = load1(gu.raw_data(), t, ub);
        const float dg = g * u * silu_grad(x), du = g * silu(x);
        store1(ggu.raw_data(), t, gb, dg + (o.accumulate_gate_up ? load1(ggu.raw_data(), t, gb) : 0));
        store1(ggu.raw_data(), t, ub, du + (o.accumulate_gate_up ? load1(ggu.raw_data(), t, ub) : 0));
    }
}
