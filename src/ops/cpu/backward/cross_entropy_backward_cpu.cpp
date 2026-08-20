#include "ops/cpu/backward/cross_entropy_backward_cpu.h"

#include <immintrin.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {
    float load1(const void *p, Dtype t, std::size_t i) {
        if (t == Dtype::F32) return static_cast<const float *>(p)[i];
        const auto h = static_cast<const std::uint16_t *>(p)[i];
        if (t == Dtype::F16) return _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(h)));
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

    void validate(const Tensor &loss, const Tensor &gradient, const Tensor &logits,
                  const bpe::TokenId *targets, std::size_t n, float scale) {
        if (!targets || !n || logits.device_type() != DeviceType::CPU || logits.dim() != 2 ||
            !is_floating_point(logits.dtype()) || logits.size(0) != n || !logits.size(1) ||
            loss.device_type() != DeviceType::CPU || loss.dtype() != Dtype::F32 ||
            loss.shape() != std::vector<std::size_t>{1} || gradient.device_type() != DeviceType::CPU ||
            gradient.dtype() != logits.dtype() || gradient.shape() != logits.shape() ||
            !std::isfinite(scale) || scale <= 0.0F)
            throw std::invalid_argument("CPU cross entropy backward: invalid tensors, targets or scale");
    }
}

void cross_entropy_forward_backward_cpu(
    Tensor &loss, Tensor &gradient, const Tensor &logits,
    const bpe::TokenId *targets, std::size_t n, float scale) {
    validate(loss, gradient, logits, targets, n, scale);
    const auto vocab = logits.size(1);
    const auto *p = logits.raw_data();
    auto *g = gradient.raw_data();
    float total = 0.0F;
#pragma omp parallel for reduction(+:total) schedule(static)
    for (std::int64_t r = 0; r < static_cast<std::int64_t>(n); ++r) {
        const auto base = static_cast<std::size_t>(r) * vocab;
        float maximum = -std::numeric_limits<float>::infinity();
        for (std::size_t i = 0; i < vocab; ++i) maximum = std::max(maximum, load1(p, logits.dtype(), base + i));
        float sum = 0.0F;
        for (std::size_t i = 0; i < vocab; ++i) sum += std::exp(load1(p, logits.dtype(), base + i) - maximum);
        const float inv = 1.0F / sum;
        const float factor = scale / static_cast<float>(n);
        if (targets[r] < vocab) total += std::log(sum) + maximum - load1(p, logits.dtype(), base + targets[r]);
        for (std::size_t i = 0; i < vocab; ++i) {
            float value = std::exp(load1(p, logits.dtype(), base + i) - maximum) * inv;
            if (targets[r] == i) value -= 1.0F;
            store1(g, logits.dtype(), base + i, value * factor);
        }
    }
    *static_cast<float *>(loss.raw_data()) = total / static_cast<float>(n);
}
