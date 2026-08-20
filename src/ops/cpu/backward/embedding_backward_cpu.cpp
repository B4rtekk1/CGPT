#include "ops/cpu/backward/embedding_backward_cpu.h"

#include <immintrin.h>
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
}

void embedding_backward_cpu(Tensor &gw, const Tensor &go, const bpe::TokenId *ids, std::size_t count,
                            const EmbeddingBackwardOptions &o) {
    if (!ids || !count || gw.device_type() != DeviceType::CPU || go.device_type() != DeviceType::CPU || gw.dim() != 2 ||
        go.dim() < 1 || gw.dtype() != go.dtype() || !is_floating_point(gw.dtype()) || go.shape().back() != gw.shape()[1]
        || go.numel() != count * gw.shape()[1])
        throw std::invalid_argument(
            "CPU embedding backward: incompatible tensors or token count");
    const auto vocab = gw.shape()[0], hidden = gw.shape()[1];
    const Dtype t = gw.dtype();
    const auto *gp = go.raw_data();
    auto *wp = gw.raw_data();
#pragma omp parallel for schedule(static)
    for (std::int64_t f = 0; f < static_cast<std::int64_t>(hidden); ++f) {
        const auto feature = static_cast<std::size_t>(f);
        if (!o.accumulate_weight)for (std::size_t row = 0; row < vocab; ++row)store1(wp, t, row * hidden + feature, 0);
        for (std::size_t token = 0; token < count; ++token) {
            const auto id = ids[token];
            if (o.bounds_check && id >= vocab)continue;
            const auto index = static_cast<std::size_t>(id) * hidden + feature;
            const float old = o.accumulate_weight ? load1(wp, t, index) : 0;
            store1(wp, t, index, old + load1(gp, t, token * hidden + feature));
        }
    }
}
