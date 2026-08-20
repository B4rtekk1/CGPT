#include "ops/cpu/backward/linear_backward_cpu.h"

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
        std::uint32_t b = static_cast<std::uint32_t>(h) << 16;
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
            static_cast<std::uint16_t *>(p)[i] = static_cast<std::uint16_t>(_mm_cvtsi128_si32(
                _mm_cvtps_ph(_mm_set_ss(x), 0)));
            return;
        }
        std::uint32_t b;
        std::memcpy(&b, &x, 4);
        static_cast<std::uint16_t *>(p)[i] = static_cast<std::uint16_t>((b + 0x7fff + ((b >> 16) & 1)) >> 16);
    }

    void validate(const Tensor &gi, const Tensor &gw, const Tensor &gb, const Tensor &go, const Tensor &x,
                  const Tensor &w) {
        if (gi.device_type() != DeviceType::CPU || gw.device_type() != DeviceType::CPU || gb.device_type() !=
            DeviceType::CPU || go.device_type() != DeviceType::CPU || x.device_type() != DeviceType::CPU || w.
            device_type() != DeviceType::CPU || x.dim() < 1 || w.dim() != 2 || gi.shape() != x.shape() || go.dim() != x.
            dim() || x.dtype() != w.dtype() || x.dtype() != gi.dtype() || x.dtype() != gw.dtype() || x.dtype() != gb.
            dtype() || x.dtype() != go.dtype() || !is_floating_point(x.dtype()) || x.shape().back() != w.shape()[1] ||
            go.shape().back() != w.shape()[0] || gw.shape() != w.shape() || gb.dim() != 1 || gb.shape()[0] != w.shape()[
                0])
            throw std::invalid_argument("CPU linear backward: incompatible tensors");
        for (std::size_t i = 0; i + 1 < x.dim(); ++i)
            if (go.shape()[i] != x.shape()[i])
                throw std::invalid_argument(
                    "CPU linear backward: leading shape mismatch");
    }
}

void linear_backward_cpu(Tensor &gi, Tensor &gw, Tensor &gb, const Tensor &go, const Tensor &x, const Tensor &w,
                         const LinearBackwardOptions &o) {
    validate(gi, gw, gb, go, x, w);
    const auto rows = x.numel() / x.shape().back();
    const auto I = x.shape().back();
    const auto O = w.shape()[0];
    const Dtype t = x.dtype();
    const auto *xp = x.raw_data();
    const auto *wp = w.raw_data();
    const auto *gop = go.raw_data();
#pragma omp parallel for schedule(static)
    for (std::int64_t r = 0; r < static_cast<std::int64_t>(rows); ++r) {
        const auto rb = static_cast<std::size_t>(r) * I, ob = static_cast<std::size_t>(r) * O;
        for (std::size_t i = 0; i < I; ++i) {
            float s = 0;
            for (std::size_t j = 0; j < O; ++j)s = std::fma(load1(gop, t, ob + j), load1(wp, t, j * I + i), s);
            store1(gi.raw_data(), t, rb + i, s + (o.accumulate_input ? load1(gi.raw_data(), t, rb + i) : 0));
        }
    }
#pragma omp parallel for schedule(static)
    for (std::int64_t index = 0; index < static_cast<std::int64_t>(O * I); ++index) {
        const auto z = static_cast<std::size_t>(index), j = z / I, i = z % I;
        float s = 0;
        for (std::size_t r = 0; r < rows; ++r)s = std::fma(load1(gop, t, r * O + j), load1(xp, t, r * I + i), s);
        store1(gw.raw_data(), t, z, s + (o.accumulate_weight ? load1(gw.raw_data(), t, z) : 0));
    }
#pragma omp parallel for schedule(static)
    for (std::int64_t j = 0; j < static_cast<std::int64_t>(O); ++j) {
        float s = 0;
        for (std::size_t r = 0; r < rows; ++r)s += load1(gop, t, r * O + static_cast<std::size_t>(j));
        store1(gb.raw_data(), t, static_cast<std::size_t>(j),
               s + (o.accumulate_bias ? load1(gb.raw_data(), t, static_cast<std::size_t>(j)) : 0));
    }
}
