/**
 * @file rope_cpu.cpp
 * @brief AVX2-optimized CPU implementation of rotary position embeddings.
 *
 * This translation unit applies interleaved rotary position embeddings to
 * query and key tensors stored in F32, F16, or BF16 format. Rotations are
 * computed in FP32 and written back in the original storage type.
 */

#include "ops/cpu/rope_cpu.h"

#include <immintrin.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>

namespace {
    /**
     * @brief Loads one tensor element and converts it to FP32.
     *
     * @param p Pointer to the beginning of tensor storage.
     * @param t Storage data type of the tensor.
     * @param i Element index.
     * @return Element value represented as FP32.
     */
    inline float load1(const void *p, Dtype t, std::size_t i) {
        if (t == Dtype::F32) return static_cast<const float *>(p)[i];
        const auto h = static_cast<const std::uint16_t *>(p)[i];
        if (t == Dtype::F16) return _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(h)));
        std::uint32_t bits = static_cast<std::uint32_t>(h) << 16;
        float x;
        std::memcpy(&x, &bits, 4);
        return x;
    }

    /**
     * @brief Converts and stores one FP32 value in tensor storage.
     *
     * BF16 conversion uses round-to-nearest-even. F16 conversion is performed
     * using the F16C conversion instruction.
     *
     * @param p Pointer to the beginning of destination tensor storage.
     * @param t Storage data type of the destination tensor.
     * @param i Destination element index.
     * @param x FP32 value to store.
     */
    inline void store1(void *p, Dtype t, std::size_t i, float x) {
        if (t == Dtype::F32) {
            static_cast<float *>(p)[i] = x;
            return;
        }
        if (t == Dtype::F16) {
            static_cast<std::uint16_t *>(p)[i] = static_cast<std::uint16_t>(_mm_cvtsi128_si32(_mm_cvtps_ph(_mm_set_ss(x), 0)));
            return;
        }
        std::uint32_t bits;
        std::memcpy(&bits, &x, 4);
        static_cast<std::uint16_t *>(p)[i] = static_cast<std::uint16_t>((bits + 0x7fff + ((bits >> 16) & 1)) >> 16);
    }

    /**
     * @brief Applies eight independent rotary transformations using AVX2/FMA.
     *
     * Each adjacent pair \f$(x_j, y_j)\f$ is transformed as
     * \f[
     *     x'_j = x_j c_j - y_j s_j, \qquad
     *     y'_j = x_j s_j + y_j c_j.
     * \f]
     *
     * Input elements are converted to FP32, processed as two eight-wide AVX
     * vectors, and stored back using the original tensor data type.
     *
     * @param data Pointer to mutable query or key tensor storage.
     * @param t Storage data type.
     * @param base Index of the first element in the first pair.
     * @param c Pointer to eight cosine coefficients.
     * @param s Pointer to eight sine coefficients.
     */
    inline void rotate8(void *data, Dtype t, std::size_t base, const float *c, const float *s) {
        alignas(32) float x[8], y[8];
        for (int j = 0; j < 8; ++j) {
            x[j] = load1(data, t, base + 2 * j);
            y[j] = load1(data, t, base + 2 * j + 1);
        }
        const __m256 vx = _mm256_load_ps(x), vy = _mm256_load_ps(y);
        const __m256 vc = _mm256_loadu_ps(c), vs = _mm256_loadu_ps(s);
        const __m256 ox = _mm256_fmadd_ps(vx, vc, _mm256_mul_ps(_mm256_sub_ps(_mm256_setzero_ps(), vy), vs));
        const __m256 oy = _mm256_fmadd_ps(vx, vs, _mm256_mul_ps(vy, vc));
        _mm256_store_ps(x, ox);
        _mm256_store_ps(y, oy);
        for (int j = 0; j < 8; ++j) {
            store1(data, t, base + 2 * j, x[j]);
            store1(data, t, base + 2 * j + 1, y[j]);
        }
    }

    /**
     * @brief Applies rotary transformations to a contiguous head fragment.
     *
     * Complete groups of eight pairs are processed with AVX2. Any remaining
     * pairs are handled by a scalar FMA path.
     *
     * @param p Pointer to mutable query or key tensor storage.
     * @param t Storage data type.
     * @param base Index of the first element in the head fragment.
     * @param pairs Number of adjacent element pairs to rotate.
     * @param c Pointer to cosine coefficients for the fragment.
     * @param s Pointer to sine coefficients for the fragment.
     */
    inline void rotate_head(void *p, Dtype t, std::size_t base, std::size_t pairs, const float *c, const float *s) {
        std::size_t i = 0;
        for (; i + 7 < pairs; i += 8) rotate8(p, t, base + 2 * i, c + i, s + i);
        for (; i < pairs; ++i) {
            const float x = load1(p, t, base + 2 * i), y = load1(p, t, base + 2 * i + 1);
            store1(p, t, base + 2 * i, std::fma(-y, s[i], x * c[i]));
            store1(p, t, base + 2 * i + 1, std::fma(x, s[i], y * c[i]));
        }
    }

    /**
     * @brief Validates tensors and options used by the CPU RoPE implementation.
     *
     * Query and key tensors must use the layout `[batch, sequence, heads,
     * head_dim]`. Cosine and sine caches must use `[cached_positions,
     * rotary_dim / 2]` and must cover the requested sequence range.
     *
     * @param q Query tensor.
     * @param k Key tensor.
     * @param c Cosine cache tensor.
     * @param s Sine cache tensor.
     * @param o Rotary embedding options.
     *
     * @throws std::invalid_argument If tensor devices, ranks, shapes, data
     *         types, rotary dimensions, or cache ranges are incompatible.
     */
    void validate(const Tensor &q, const Tensor &k, const Tensor &c, const Tensor &s, const RopeOptions &o) {
        if (q.device_type() != DeviceType::CPU || k.device_type() != DeviceType::CPU || c.device_type() !=
            DeviceType::CPU || s.device_type() != DeviceType::CPU || q.dim() != 4 || k.dim() != 4 || c.dim() != 2 || s.
            dim() != 2 || q.dtype() != k.dtype() || q.dtype() != c.dtype() || c.shape() != s.shape() || q.shape()[0] !=
            k.shape()[0] || q.shape()[1] != k.shape()[1] || q.shape()[3] != k.shape()[3] || !
            is_floating_point(q.dtype()))
            throw std::invalid_argument(
                "CPU RoPE: incompatible CPU floating-point tensors");
        const auto dim = q.shape()[3], rd = o.rotary_dim ? o.rotary_dim : dim;
        if (!rd || rd > dim || (rd & 1) || c.shape()[1] != rd / 2 || q.shape()[1] > c.shape()[0] || o.position_offset >
            c.shape()[0] - q.shape()[1])
            throw std::invalid_argument(
                "CPU RoPE: invalid rotary dimension or cache range");
    }
}

/**
 * @brief Applies rotary position embeddings in place to query and key tensors.
 *
 * The function rotates adjacent feature pairs in the first @c rotary_dim
 * elements of every query and key head. When @c rotary_dim is zero, the entire
 * head dimension is rotated. Position coefficients are selected from @p c and
 * @p s beginning at @c position_offset.
 *
 * Query and key tensors use the layouts `[B, S, Q, D]` and `[B, S, K, D]`,
 * respectively. The head counts may differ, which permits grouped-query or
 * multi-query attention. Work is parallelized across batch-position pairs with
 * OpenMP, while groups of rotary pairs are processed using AVX2/FMA.
 *
 * @param q Query tensor modified in place.
 * @param k Key tensor modified in place.
 * @param c Cosine cache with shape `[cached_positions, rotary_dim / 2]`.
 * @param s Sine cache with the same shape and data type as @p c.
 * @param o Rotary embedding options containing the optional rotary dimension
 *          and starting position offset.
 *
 * @throws std::invalid_argument If the tensors are not compatible CPU
 *         floating-point tensors, if the rotary dimension is zero, odd, or
 *         larger than the head dimension, or if the cache does not cover the
 *         requested sequence positions.
 *
 * @note Rotation arithmetic is performed in FP32 and converted back to the
 *       original tensor storage type.
 * @note Elements after @c rotary_dim in each head remain unchanged.
 */
void rope_forward_cpu(Tensor &q, Tensor &k, const Tensor &c, const Tensor &s, const RopeOptions &o) {
    validate(q, k, c, s, o);
    const auto B = q.shape()[0], S = q.shape()[1], Q = q.shape()[2], K = k.shape()[2], D = q.shape()[3], P =
            (o.rotary_dim ? o.rotary_dim : D) / 2;
    const auto t = q.dtype();
    const auto *cp = c.raw_data();
    const auto *sp = s.raw_data();
    auto *qp = q.raw_data();
    auto *kp = k.raw_data();
#pragma omp parallel for schedule(static)
    for (std::int64_t ti = 0; ti < static_cast<std::int64_t>(B * S); ++ti) {
        const std::size_t token = static_cast<std::size_t>(ti), pos = o.position_offset + token % S, cb = pos * P, qb = token * Q * D
                , kb = token * K * D;
        alignas(32) float cv[8], sv[8];
        for (std::size_t i = 0; i < P; i += 8) {
            const std::size_t n = std::min<std::size_t>(8, P - i);
            for (std::size_t j = 0; j < n; ++j) {
                cv[j] = load1(cp, t, cb + i + j);
                sv[j] = load1(sp, t, cb + i + j);
            }
            for (std::size_t h = 0; h < Q; h += 4)
                for (std::size_t z = 0; z < std::min<std::size_t>(4, Q - h); ++z)
                    rotate_head(qp, t, qb + (h + z) * D, std::min<std::size_t>(P, i + 8) - i, cv, sv);
            for (std::size_t h = 0; h < K; h += 4)
                for (std::size_t z = 0; z < std::min<std::size_t>(4, K - h); ++z)
                    rotate_head(kp, t, kb + (h + z) * D, std::min<std::size_t>(P, i + 8) - i, cv, sv);
        }
    }
}
