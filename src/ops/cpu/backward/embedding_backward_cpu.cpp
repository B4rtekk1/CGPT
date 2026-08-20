#include "ops/cpu/backward/embedding_backward_cpu.h"

#include <immintrin.h>
#include <cstdint>
#include <cstring>
#include <stdexcept>

namespace {
    /**
     * @brief Loads one tensor element and expands it to F32.
     *
     * @param p Pointer to the source tensor storage.
     * @param t Source element type: F32, F16, or BF16.
     * @param i Linear element index.
     * @return The selected element represented as an F32 value.
     *
     * @note F16 conversion requires the F16C instruction set.
     */
    float load1(const void *p, Dtype t, std::size_t i) {
        if (t == Dtype::F32)return static_cast<const float *>(p)[i];
        const auto h = static_cast<const std::uint16_t *>(p)[i];
        if (t == Dtype::F16)return _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(h)));
        std::uint32_t b = static_cast<std::uint32_t>(h) << 16;
        float x;
        std::memcpy(&x, &b, 4);
        return x;
    }

    /**
     * @brief Converts and stores one F32 value in tensor storage.
     *
     * BF16 conversion uses round-to-nearest-even rounding. F16 conversion is
     * performed by the F16C instruction set.
     *
     * @param p Pointer to the destination tensor storage.
     * @param t Destination element type.
     * @param i Linear element index.
     * @param x F32 value to store.
     */
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

/**
 * @brief Accumulates gradients for an embedding weight matrix on the CPU.
 *
 * For every token position `token` and embedding feature `feature`, the
 * upstream gradient is added to the row selected by `ids[token]`:
 * @f[
 *     \nabla_W[\mathrm{id}_{token}, feature] \mathrel{+}=
 *     \nabla_Y[token, feature].
 * @f]
 *
 * The weight-gradient tensor has shape `[vocabulary_size, hidden_size]`.
 * The upstream gradient may have any number of leading dimensions, provided its
 * final dimension equals `hidden_size` and its total element count equals
 * `count * hidden_size`.
 *
 * Work is parallelized over embedding features. This avoids concurrent writes
 * between OpenMP workers even when the same token identifier occurs multiple
 * times.
 *
 * @param gw Destination gradient for the embedding weight matrix.
 * @param go Upstream gradient for all token positions.
 * @param ids Pointer to an array containing @p count token identifiers.
 * @param count Number of token identifiers and embedding rows represented by
 * @p go.
 * @param o Backward options controlling gradient accumulation and out-of-range
 * token handling.
 *
 * @throws std::invalid_argument If @p ids is null, @p count is zero, tensors
 * are not compatible CPU floating-point tensors, or their shapes are invalid.
 *
 * @note When `accumulate_weight` is false, every gradient element is cleared
 * before token contributions are accumulated.
 * @note When `bounds_check` is true, token identifiers outside the vocabulary
 * are ignored. Disabling bounds checking requires every identifier to be valid.
 */
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
