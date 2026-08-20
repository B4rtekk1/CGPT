#include "ops/cpu/cross_entropy_cpu.h"

#include <immintrin.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace {
    float load1(const void *p, Dtype t, std::size_t i) {
        if (t == Dtype::F32) return static_cast<const float *>(p)[i];
        const auto h = static_cast<const std::uint16_t *>(p)[i];
        if (t == Dtype::F16) return _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(h)));
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

    __m256 load8(const void *p, Dtype t, std::size_t i) {
        if (t == Dtype::F32)return _mm256_loadu_ps(static_cast<const float *>(p) + i);
        const auto *h = static_cast<const std::uint16_t *>(p) + i;
        if (t == Dtype::F16)return _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i *>(h)));
        return _mm256_castsi256_ps(
            _mm256_slli_epi32(_mm256_cvtepu16_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i *>(h))), 16));
    }

    void validate_loss(const Tensor &loss, const Tensor &logits, const bpe::TokenId *targets, std::size_t n) {
        if (!targets || !n || logits.device_type() != DeviceType::CPU || logits.dim() != 2 || !
            is_floating_point(logits.dtype()) || logits.size(0) != n || !logits.size(1) || loss.device_type() !=
            DeviceType::CPU || loss.dtype() != Dtype::F32 || loss.shape() != std::vector<std::size_t>{1})
            throw
                    std::invalid_argument("CPU cross entropy: invalid tensors or targets");
    }

    void validate_gradient(const Tensor &gradient, const Tensor &logits, float scale) {
        if (gradient.device_type() != DeviceType::CPU || gradient.dtype() != logits.dtype() || gradient.shape() !=
            logits.shape() || !std::isfinite(scale) || scale <= 0)
            throw std::invalid_argument(
                "CPU cross entropy: invalid gradient or scale");
    }

    float row_loss(const void *data, Dtype t, std::size_t base, std::size_t vocab, bpe::TokenId target) {
        float maximum = -std::numeric_limits<float>::infinity();
        __m256 m = _mm256_set1_ps(-std::numeric_limits<float>::infinity());
        std::size_t i = 0;
        for (; i + 7 < vocab; i += 8)m = _mm256_max_ps(m, load8(data, t, base + i));
        alignas(32)float q[8];
        _mm256_store_ps(q, m);
        for (float x: q)maximum = std::max(maximum, x);
        for (; i < vocab; ++i)maximum = std::max(maximum, load1(data, t, base + i));
        float sum = 0;
        for (i = 0; i < vocab; ++i)sum += std::exp(load1(data, t, base + i) - maximum);
        if (target >= vocab)return 0;
        return std::log(sum) + maximum - load1(data, t, base + target);
    }
}

void cross_entropy_forward_cpu(Tensor &loss, const Tensor &logits, const bpe::TokenId *targets, std::size_t n) {
    validate_loss(loss, logits, targets, n);
    const auto vocab = logits.size(1);
    const auto *p = logits.raw_data();
    float total = 0;
#pragma omp parallel for reduction(+:total) schedule(static)
    for (std::int64_t r = 0; r < static_cast<std::int64_t>(n); ++r)
        total += row_loss(p, logits.dtype(),
                          std::size_t(r) * vocab, vocab, targets[r]);
    *static_cast<float *>(loss.raw_data()) = total / float(n);
}

void cross_entropy_forward_backward_cpu(Tensor &loss, Tensor &gradient, const Tensor &logits,
                                        const bpe::TokenId *targets, std::size_t n, float scale) {
    validate_loss(loss, logits, targets, n);
    validate_gradient(gradient, logits, scale);
    const auto vocab = logits.size(1);
    const auto *p = logits.raw_data();
    auto *g = gradient.raw_data();
    float total = 0;
#pragma omp parallel for reduction(+:total) schedule(static)
    for (std::int64_t r = 0; r < static_cast<std::int64_t>(n); ++r) {
        const std::size_t base = std::size_t(r) * vocab;
        float maximum = -std::numeric_limits<float>::infinity();
        for (std::size_t i = 0; i < vocab; ++i)maximum = std::max(maximum, load1(p, logits.dtype(), base + i));
        float sum = 0;
        for (std::size_t i = 0; i < vocab; ++i)sum += std::exp(load1(p, logits.dtype(), base + i) - maximum);
        const float inv = 1.0f / sum, factor = scale / float(n);
        if (targets[r] < vocab)total += std::log(sum) + maximum - load1(p, logits.dtype(), base + targets[r]);
        for (std::size_t i = 0; i < vocab; ++i) {
            float v = std::exp(load1(p, logits.dtype(), base + i) - maximum) * inv;
            if (targets[r] == i)v -= 1;
            store1(g, logits.dtype(), base + i, v * factor);
        }
    }
    *static_cast<float *>(loss.raw_data()) = total / float(n);
}
