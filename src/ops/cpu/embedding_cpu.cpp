#include "ops/cpu/embedding_cpu.h"

#include <immintrin.h>
#include <cstdint>
#include <cstring>
#include <stdexcept>

namespace {
    inline __m256 load8(const void *p, Dtype t, std::size_t i) {
        if (t == Dtype::F32)return _mm256_loadu_ps(static_cast<const float *>(p) + i);
        const auto *h = static_cast<const std::uint16_t *>(p) + i;
        if (t == Dtype::F16)return _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i *>(h)));
        return _mm256_castsi256_ps(
            _mm256_slli_epi32(_mm256_cvtepu16_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i *>(h))), 16));
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
        const auto bits = _mm256_castps_si256(v), rounded = _mm256_add_epi32(
            bits, _mm256_add_epi32(_mm256_set1_epi32(0x7fff),
                                   _mm256_and_si256(_mm256_srli_epi32(bits, 16), _mm256_set1_epi32(1))));
        const auto h = _mm256_srli_epi32(rounded, 16);
        _mm_storeu_si128(reinterpret_cast<__m128i *>(static_cast<std::uint16_t *>(p) + i),
                         _mm_packus_epi32(_mm256_castsi256_si128(h), _mm256_extracti128_si256(h, 1)));
    }
}

void embedding_forward_cpu(Tensor &out, const bpe::TokenId *ids, std::size_t count, const Tensor &w,
                           const EmbeddingOptions &opts) {
    if (!ids || !count || out.device_type() != DeviceType::CPU || w.device_type() != DeviceType::CPU || w.dim() != 2 ||
        out.dim() < 1 || out.dtype() != w.dtype() || !is_floating_point(w.dtype()) || out.shape().back() != w.shape()[1]
        || out.numel() != count * w.shape()[1])throw std::invalid_argument(
        "CPU embedding: incompatible tensors or token count");
    const auto vocab = w.shape()[0], hidden = w.shape()[1];
    const Dtype t = w.dtype();
    const auto *src = w.raw_data();
    auto *dst = out.raw_data();
#pragma omp parallel for schedule(static)
    for (std::int64_t r = 0; r < static_cast<std::int64_t>(count); ++r) {
        const auto id = ids[r];
        auto *d = static_cast<std::uint8_t *>(dst) + std::size_t(r) * hidden * dtype_size(t);
        if (id >= vocab && opts.bounds_check) {
            std::memset(d, 0, hidden * dtype_size(t));
            continue;
        }
        const auto *s = static_cast<const std::uint8_t *>(src) + std::size_t(id) * hidden * dtype_size(t);
        std::size_t i = 0;
        for (; i + 7 < hidden; i += 8)store8(d, t, i, load8(s, t, i));
        for (; i < hidden; ++i)std::memcpy(d + i * dtype_size(t), s + i * dtype_size(t), dtype_size(t));
    }
}
