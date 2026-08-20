/**
 * @file attention_cpu.cpp
 * @brief CPU implementation of grouped-query scaled dot-product attention.
 *
 * The implementation supports multi-head attention, grouped-query attention,
 * optional causal masking, optional log-sum-exp output, and F32/F16/BF16 tensor
 * storage. Dot products use AVX2 and FMA, while independent batch/query/head
 * tasks are parallelized with OpenMP.
 */

#include "ops/cpu/attention_cpu.h"

#include <immintrin.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace {
    /**
     * @brief Loads one tensor element and converts it to F32.
     *
     * @param p Pointer to the beginning of the tensor storage.
     * @param t Tensor element type.
     * @param i Element index.
     * @return The requested value represented as F32.
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
     * @brief Loads eight tensor elements and converts them to F32 lanes.
     *
     * @param p Pointer to the beginning of the tensor storage.
     * @param t Tensor element type.
     * @param i Index of the first element to load.
     * @return Eight values represented as an AVX F32 vector.
     */
    __m256 load8(const void *p, Dtype t, std::size_t i) {
        if (t == Dtype::F32)return _mm256_loadu_ps(static_cast<const float *>(p) + i);
        const auto *h = static_cast<const std::uint16_t *>(p) + i;
        if (t == Dtype::F16)return _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i *>(h)));
        return _mm256_castsi256_ps(
            _mm256_slli_epi32(_mm256_cvtepu16_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i *>(h))), 16));
    }

    /**
     * @brief Converts and stores one F32 value in tensor storage.
     *
     * BF16 conversion uses round-to-nearest-even before truncation.
     *
     * @param p Pointer to the beginning of the destination tensor storage.
     * @param t Destination tensor element type.
     * @param i Destination element index.
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

    /**
     * @brief Computes an F32 dot product between two tensor slices.
     *
     * Blocks of eight values are accumulated with AVX2 FMA instructions. Any
     * remaining elements are processed with scalar `std::fma` operations.
     *
     * @param a Pointer to the first tensor storage.
     * @param b Pointer to the second tensor storage.
     * @param t Common tensor element type.
     * @param ia Starting element index in @p a.
     * @param ib Starting element index in @p b.
     * @param d Number of elements in each slice.
     * @return Dot product accumulated in F32.
     */
    float dot8(const void *a, const void *b, Dtype t, std::size_t ia, std::size_t ib, std::size_t d) {
        __m256 s = _mm256_setzero_ps();
        std::size_t i = 0;
        for (; i + 7 < d; i += 8)s = _mm256_fmadd_ps(load8(a, t, ia + i), load8(b, t, ib + i), s);
        alignas(32)float q[8];
        _mm256_store_ps(q, s);
        float r = 0;
        for (float x: q)r += x;
        for (; i < d; ++i)r = std::fma(load1(a, t, ia + i), load1(b, t, ib + i), r);
        return r;
    }

    /**
     * @brief Validates grouped-query attention tensors and options.
     *
     * Query and output tensors use layout `[batch, query_sequence,
     * query_heads, head_dim]`. Key and value tensors use layout `[batch,
     * key_sequence, kv_heads, head_dim]`. The number of query heads must be an
     * integer multiple of the number of key/value heads.
     *
     * @param out Attention output tensor.
     * @param q Query tensor.
     * @param k Key tensor.
     * @param v Value tensor.
     * @param o Attention configuration.
     * @param lse Optional F32 tensor with shape
     *        `[batch, query_sequence, query_heads]`.
     *
     * @throws std::invalid_argument If tensor devices, shapes, data types,
     *         attention dimensions, causal ranges, or the optional LSE tensor
     *         are incompatible.
     */
    void validate(Tensor &out, const Tensor &q, const Tensor &k, const Tensor &v, const FlashAttentionOptions &o,
                  Tensor *lse) {
        if (out.device_type() != DeviceType::CPU || q.device_type() != DeviceType::CPU || k.device_type() !=
            DeviceType::CPU || v.device_type() != DeviceType::CPU || q.dim() != 4 || k.dim() != 4 || v.dim() != 4 || out
            .shape() != q.shape() || q.dtype() != k.dtype() || q.dtype() != v.dtype() || out.dtype() != q.dtype() || !
            is_floating_point(q.dtype()) || !o.num_query_heads || !o.num_kv_heads || !o.head_dim || o.num_query_heads %
            o.num_kv_heads || q.shape()[2] != o.num_query_heads || q.shape()[3] != o.head_dim || k.shape()[0] != q.
            shape()[0] || v.shape() != k.shape() || k.shape()[2] != o.num_kv_heads || k.shape()[3] != o.head_dim || (
                o.causal && (o.query_position_offset > k.shape()[1] || q.shape()[1] > k.shape()[1] - o.
                             query_position_offset)))
            throw std::invalid_argument(
                "CPU attention: incompatible tensors or options");
        if (lse && (lse->device_type() != DeviceType::CPU || lse->dtype() != Dtype::F32 || lse->shape() != std::vector<
                        std::size_t>{q.shape()[0], q.shape()[1], q.shape()[2]}))
            throw std::invalid_argument(
                "CPU attention: invalid logsumexp tensor");
    }

    /**
     * @brief Executes grouped-query scaled dot-product attention.
     *
     * Each query head is mapped to one key/value head by contiguous grouping.
     * Scores are scaled by `o.attention_scale` when it is positive; otherwise
     * the conventional `1 / sqrt(head_dim)` scale is used. Softmax is evaluated
     * with maximum subtraction for numerical stability.
     *
     * In causal mode, query position `qi` may attend through
     * `o.query_position_offset + qi`. The implementation computes each output
     * directly without materializing a complete attention probability tensor.
     * Independent `[batch, query, query_head]` tasks are parallelized with
     * OpenMP.
     *
     * @param out Attention output with the same shape and type as @p q.
     * @param lse Optional F32 log-sum-exp output, or `nullptr` when not needed.
     * @param q Query tensor with shape `[B, QS, H, D]`.
     * @param k Key tensor with shape `[B, KS, KV, D]`.
     * @param v Value tensor with shape `[B, KS, KV, D]`.
     * @param o Attention options describing the head counts, head dimension,
     *        scaling, causal mode, and query position offset.
     *
     * @throws std::invalid_argument If validation of the tensors or options
     *         fails.
     */
    void apply(Tensor &out, Tensor *lse, const Tensor &q, const Tensor &k, const Tensor &v,
               const FlashAttentionOptions &o) {
        validate(out, q, k, v, o, lse);
        const auto B = q.shape()[0], QS = q.shape()[1], KS = k.shape()[1], H = o.num_query_heads, KV = o.num_kv_heads, D
                = o.head_dim;
        const float scale = o.attention_scale > 0 ? o.attention_scale : 1.0f / std::sqrt(static_cast<float>(D));
        const auto t = q.dtype();
        const auto *qp = q.raw_data();
        const auto *kp = k.raw_data();
        const auto *vp = v.raw_data();
        auto *op = out.raw_data();
#pragma omp parallel for schedule(static)
        for (std::int64_t task = 0; task < static_cast<std::int64_t>(B * QS * H); ++task) {
            const std::size_t z = static_cast<std::size_t>(task), h = z % H, qi = (z / H) % QS, b = z / (H * QS), kvh =
                            h / (H / KV),
                    qb = ((b * QS + qi) * H + h) * D;
            float mx = -std::numeric_limits<float>::infinity();
            const std::size_t first = 0;
            const std::size_t last = o.causal ? (o.query_position_offset + qi + 1) : KS;
            for (std::size_t kj = first; kj < last; ++kj) {
                const auto kb = ((b * KS + kj) * KV + kvh) * D;
                mx = std::max(mx, dot8(qp, kp, t, qb, kb, D) * scale);
            }
            float den = 0;
            for (std::size_t kj = first; kj < last; ++kj) {
                const auto kb = ((b * KS + kj) * KV + kvh) * D;
                den += std::exp(dot8(qp, kp, t, qb, kb, D) * scale - mx);
            }
            if (lse)static_cast<float *>(lse->raw_data())[(b * QS + qi) * H + h] = std::log(den) + mx;
            for (std::size_t d = 0; d < D; ++d) {
                float sum = 0;
                for (std::size_t kj = first; kj < last; ++kj) {
                    const auto kb = ((b * KS + kj) * KV + kvh) * D;
                    sum += std::exp(dot8(qp, kp, t, qb, kb, D) * scale - mx) * load1(vp, t, kb + d);
                }
                store1(op, t, qb + d, sum / den);
            }
        }
    }
}

/**
 * @brief Computes grouped-query attention without returning log-sum-exp values.
 *
 * @param o Output tensor with the same shape and type as @p q.
 * @param q Query tensor with shape `[B, QS, num_query_heads, head_dim]`.
 * @param k Key tensor with shape `[B, KS, num_kv_heads, head_dim]`.
 * @param v Value tensor with the same shape as @p k.
 * @param x Attention configuration.
 *
 * @throws std::invalid_argument If the tensors or options are incompatible.
 *
 * @note The implementation requires AVX2 and FMA. F16 execution additionally
 *       requires F16C support.
 */
void flash_gqa_attention_forward_cpu(Tensor &o, const Tensor &q, const Tensor &k, const Tensor &v,
                                     const FlashAttentionOptions &x) { apply(o, nullptr, q, k, v, x); }

/**
 * @brief Computes grouped-query attention and per-query log-sum-exp values.
 *
 * @param o Output tensor with the same shape and type as @p q.
 * @param l F32 tensor with shape `[B, QS, num_query_heads]` receiving
 *        `log(sum(exp(score)))` for each query head.
 * @param q Query tensor with shape `[B, QS, num_query_heads, head_dim]`.
 * @param k Key tensor with shape `[B, KS, num_kv_heads, head_dim]`.
 * @param v Value tensor with the same shape as @p k.
 * @param x Attention configuration.
 *
 * @throws std::invalid_argument If the tensors, options, or LSE output are
 *         incompatible.
 *
 * @note The implementation requires AVX2 and FMA. F16 execution additionally
 *       requires F16C support.
 */
void flash_gqa_attention_forward_with_lse_cpu(Tensor &o, Tensor &l, const Tensor &q, const Tensor &k, const Tensor &v,
                                              const FlashAttentionOptions &x) { apply(o, &l, q, k, v, x); }
