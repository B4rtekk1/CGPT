#include "ops/cpu/backward/attention_backward_cpu.h"

#include <immintrin.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {
    /**
     * @brief Loads one tensor element and expands it to F32.
     *
     * @param p Pointer to source tensor storage.
     * @param t Source element type: F32, F16, or BF16.
     * @param i Linear element index.
     * @return The selected value represented as F32.
     *
     * @note F16 conversion requires CPU support for F16C.
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
     * F16 conversion uses F16C. BF16 conversion uses round-to-nearest-even
     * rounding before discarding the lower 16 bits.
     *
     * @param p Pointer to destination tensor storage.
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

    /**
     * @brief Computes a dot product between two tensor slices.
     *
     * Elements are converted to F32 before multiplication. Groups of eight
     * elements are accumulated with AVX2/FMA, followed by a scalar tail.
     *
     * @param a Pointer to the first tensor storage.
     * @param b Pointer to the second tensor storage.
     * @param t Common element type of both tensors.
     * @param ia Starting linear index in @p a.
     * @param ib Starting linear index in @p b.
     * @param d Number of elements in each slice.
     * @return F32 dot-product result.
     *
     * @note This helper requires AVX2 and FMA. F16 inputs additionally require
     * F16C.
     */
    float dot(const void *a, const void *b, Dtype t, std::size_t ia, std::size_t ib, std::size_t d) {
        __m256 s = _mm256_setzero_ps();
        std::size_t i = 0;
        for (; i + 7 < d; i += 8) {
            alignas(32)float x[8], y[8];
            for (int z = 0; z < 8; ++z) {
                x[z] = load1(a, t, ia + i + z);
                y[z] = load1(b, t, ib + i + z);
            }
            s = _mm256_fmadd_ps(_mm256_load_ps(x), _mm256_load_ps(y), s);
        }
        alignas(32)float q[8];
        _mm256_store_ps(q, s);
        float r = 0;
        for (float x: q)r += x;
        for (; i < d; ++i)r = std::fma(load1(a, t, ia + i), load1(b, t, ib + i), r);
        return r;
    }

    /**
     * @brief Validates tensors and options for GQA attention backpropagation.
     *
     * Query-related tensors use shape `[B, QS, H, D]`. Key- and value-related
     * tensors use shape `[B, KS, KV, D]`, where `H` must be divisible by `KV`.
     *
     * @param gq Destination gradient with respect to queries.
     * @param gk Destination gradient with respect to keys.
     * @param gv Destination gradient with respect to values.
     * @param go Upstream gradient with the same shape as @p q.
     * @param q Query tensor.
     * @param k Key tensor.
     * @param v Value tensor.
     * @param o Attention configuration, including head counts, head dimension,
     * scaling, causal mode, and query-position offset.
     *
     * @throws std::invalid_argument If tensors are not compatible CPU
     * floating-point tensors, shapes do not match the configured attention
     * layout, GQA head grouping is invalid, the scale is not finite, or the
     * causal range exceeds the key sequence.
     */
    void validate(const Tensor &gq, const Tensor &gk, const Tensor &gv, const Tensor &go, const Tensor &q,
                  const Tensor &k, const Tensor &v, const FlashAttentionOptions &o) {
        const Tensor *ts[] = {&gq, &gk, &gv, &go, &q, &k, &v};
        for (const auto *x: ts)
            if (x->device_type() != DeviceType::CPU || x->dtype() != q.dtype() || !
                is_floating_point(x->dtype()))
                throw std::invalid_argument(
                    "CPU attention backward: tensors must be CPU floating-point tensors of one dtype");
        if (q.dim() != 4 || k.dim() != 4 || v.dim() != 4 || go.shape() != q.shape() || gq.shape() != q.shape() || gk.
            shape() != k.shape() || gv.shape() != k.shape() || v.shape() != k.shape() || !o.num_query_heads || !o.
            num_kv_heads || !o.head_dim || o.num_query_heads % o.num_kv_heads || q.shape()[0] != k.shape()[0] || q.
            shape()[2] != o.num_query_heads || k.shape()[2] != o.num_kv_heads || q.shape()[3] != o.head_dim || k.shape()
            [3] != o.head_dim || !std::isfinite(o.attention_scale) || (
                o.causal && (o.query_position_offset > k.shape()[1] || q.shape()[1] > k.shape()[1] - o.
                             query_position_offset)))
            throw std::invalid_argument(
                "CPU attention backward: invalid tensors or options");
    }

    /**
     * @brief Clears a gradient tensor unless accumulation is requested.
     *
     * @param x Gradient tensor to conditionally reset.
     * @param accumulate When true, preserves the existing tensor contents.
     */
    void zero_if_needed(Tensor &x, bool accumulate) { if (!accumulate)std::memset(x.raw_data(), 0, x.nbytes()); }
}

/**
 * @brief Computes backward gradients for scaled grouped-query attention on CPU.
 *
 * The query and output-gradient tensors have shape `[B, QS, H, D]`. Key and
 * value tensors have shape `[B, KS, KV, D]`. Each group of `H / KV` query heads
 * shares one key/value head.
 *
 * Attention probabilities are reconstructed from Q and K using a numerically
 * stable softmax:
 * @f[
 *     P = softmax(scale \cdot QK^T).
 * @f]
 * The implementation then computes
 * @f[
 *     dV = P^T dO,
 * @f]
 * @f[
 *     dS = P \odot \left(dP - \sum_j P_j dP_j\right),
 * @f]
 * @f[
 *     dQ = scale \cdot dS K,
 *     \qquad
 *     dK = scale \cdot dS^T Q.
 * @f]
 *
 * In causal mode, query position `qi` may attend only to key positions up to
 * `query_position_offset + qi`. If `attention_scale` is non-positive, the
 * default scale `1 / sqrt(D)` is used.
 *
 * Work for @p gq is parallelized over batch, query position, and query head.
 * Work for @p gk and @p gv is parallelized over their individual elements,
 * avoiding concurrent writes when several query heads share a KV head.
 *
 * @param gq Destination gradient with respect to @p q.
 * @param gk Destination gradient with respect to @p k.
 * @param gv Destination gradient with respect to @p v.
 * @param go Upstream gradient with respect to the attention output.
 * @param q Query tensor with shape `[B, QS, H, D]`.
 * @param k Key tensor with shape `[B, KS, KV, D]`.
 * @param v Value tensor with shape `[B, KS, KV, D]`.
 * @param o Attention configuration.
 * @param acc When true, adds all computed gradients to existing destination
 * values. When false, destinations are overwritten.
 *
 * @throws std::invalid_argument If tensors or attention options are invalid.
 *
 * @note Computation is performed in F32, while destination storage may use F32,
 * F16, or BF16.
 * @note The implementation requires AVX2 and FMA; F16 storage additionally
 * requires F16C.
 */
void flash_gqa_attention_backward_cpu(Tensor &gq, Tensor &gk, Tensor &gv, const Tensor &go, const Tensor &q,
                                      const Tensor &k, const Tensor &v, const FlashAttentionOptions &o, bool acc) {
    validate(gq, gk, gv, go, q, k, v, o);
    zero_if_needed(gk, acc);
    zero_if_needed(gv, acc);
    const auto B = q.shape()[0], QS = q.shape()[1], KS = k.shape()[1], H = o.num_query_heads, KV = o.num_kv_heads, D = o
            .head_dim;
    const float scale = o.attention_scale > 0 ? o.attention_scale : 1.0f / std::sqrt(static_cast<float>(D));
    const Dtype t = q.dtype();
    const auto qp = q.raw_data(), kp = k.raw_data(), vp = v.raw_data(), gop = go.raw_data();
#pragma omp parallel for schedule(static)
    for (std::int64_t task = 0; task < static_cast<std::int64_t>(B * QS * H); ++task) {
        const auto z = static_cast<std::size_t>(task), hq = z % H, qi = (z / H) % QS, b = z / (H * QS), kvh =
                hq / (H / KV), qb =
                ((b * QS + qi) * H + hq) * D;
        const auto last = o.causal ? o.query_position_offset + qi + 1 : KS;
        std::vector<float> p(last);
        float mx = -std::numeric_limits<float>::infinity();
        for (std::size_t kj = 0; kj < last; ++kj)
            mx = std::max(
                mx, dot(qp, kp, t, qb, ((b * KS + kj) * KV + kvh) * D, D) * scale);
        float den = 0;
        for (std::size_t kj = 0; kj < last; ++kj) {
            p[kj] = std::exp(dot(qp, kp, t, qb, ((b * KS + kj) * KV + kvh) * D, D) * scale - mx);
            den += p[kj];
        }
        for (float &x: p)x /= den;
        float delta = 0;
        for (std::size_t kj = 0; kj < last; ++kj) {
            const auto kb = ((b * KS + kj) * KV + kvh) * D;
            delta += p[kj] * dot(gop, vp, t, qb, kb, D);
        }
        for (std::size_t d = 0; d < D; ++d) {
            float s = 0;
            for (std::size_t kj = 0; kj < last; ++kj) {
                const auto kb = ((b * KS + kj) * KV + kvh) * D;
                const float dp = dot(gop, vp, t, qb, kb, D), ds = p[kj] * (dp - delta);
                s += ds * load1(kp, t, kb + d);
            }
            const auto qi_index = qb + d;
            store1(gq.raw_data(), t, qi_index, s * scale + (acc ? load1(gq.raw_data(), t, qi_index) : 0));
        }
    }
#pragma omp parallel for schedule(static)
    for (std::int64_t task = 0; task < static_cast<std::int64_t>(B * KV * KS * D); ++task) {
        const auto z = static_cast<std::size_t>(task), d = z % D, kj = (z / D) % KS, kvh = (z / D / KS) % KV, b =
                z / (D * KS * KV);
        const auto kb = ((b * KS + kj) * KV + kvh) * D;
        float dk = 0, dv = 0;
        for (std::size_t qi = 0; qi < QS; ++qi) {
            const auto last = o.causal ? o.query_position_offset + qi + 1 : KS;
            if (kj >= last)continue;
            for (std::size_t hq = kvh * (H / KV); hq < (kvh + 1) * (H / KV); ++hq) {
                const auto qb = ((b * QS + qi) * H + hq) * D;
                float mx = -std::numeric_limits<float>::infinity();
                for (std::size_t kk = 0; kk < last; ++kk)
                    mx = std::max(
                        mx, dot(qp, kp, t, qb, ((b * KS + kk) * KV + kvh) * D,
                                D) * scale);
                float den = 0;
                for (std::size_t kk = 0; kk < last; ++kk)
                    den += std::exp(
                        dot(qp, kp, t, qb, ((b * KS + kk) * KV + kvh) * D, D) *
                        scale - mx);
                const float prob = std::exp(dot(qp, kp, t, qb, kb, D) * scale - mx) / den;
                const float dp = dot(gop, vp, t, qb, kb, D);
                float delta = 0;
                for (std::size_t kk = 0; kk < last; ++kk) {
                    const auto kkbase = ((b * KS + kk) * KV + kvh) * D;
                    float pp = std::exp(dot(qp, kp, t, qb, kkbase, D) * scale - mx) / den;
                    delta += pp * dot(gop, vp, t, qb, kkbase, D);
                }
                const float ds = prob * (dp - delta);
                dk += ds * load1(qp, t, qb + d) * scale;
                dv += prob * load1(gop, t, qb + d);
            }
        }
        const auto index = kb + d;
        store1(gk.raw_data(), t, index, dk + (acc ? load1(gk.raw_data(), t, index) : 0));
        store1(gv.raw_data(), t, index, dv + (acc ? load1(gv.raw_data(), t, index) : 0));
    }
}

/**
 * @brief Validates forward output metadata and runs GQA attention backward.
 *
 * This overload accepts the forward attention output and log-sum-exp tensor to
 * match an interface that preserves forward-pass intermediates. The current CPU
 * implementation validates these tensors but recomputes probabilities from
 * @p q and @p k by delegating to flash_gqa_attention_backward_cpu().
 *
 * @param gq Destination query gradient.
 * @param gk Destination key gradient.
 * @param gv Destination value gradient.
 * @param go Upstream output gradient.
 * @param output Forward attention output with the same shape and dtype as @p q.
 * @param lse F32 log-sum-exp tensor with shape `[B, QS, H]`.
 * @param q Forward query tensor.
 * @param k Forward key tensor.
 * @param v Forward value tensor.
 * @param o Attention configuration.
 * @param acc Whether computed gradients are accumulated.
 *
 * @throws std::invalid_argument If @p output or @p lse is incompatible, or if
 * the delegated backward validation fails.
 *
 * @note The values stored in @p output and @p lse are not currently consumed
 * after validation.
 */
void flash_gqa_attention_backward_with_lse_cpu(Tensor &gq, Tensor &gk, Tensor &gv, const Tensor &go,
                                               const Tensor &output, const Tensor &lse, const Tensor &q,
                                               const Tensor &k, const Tensor &v, const FlashAttentionOptions &o,
                                               bool acc) {
    if (output.device_type() != DeviceType::CPU || output.shape() != q.shape() || output.dtype() != q.dtype() || lse.
        device_type() != DeviceType::CPU || lse.dtype() != Dtype::F32 || lse.shape() != std::vector<std::size_t>{
            q.shape()[0], q.shape()[1], q.shape()[2]
        })
        throw std::invalid_argument("CPU attention backward: invalid output or logsumexp");
    flash_gqa_attention_backward_cpu(gq, gk, gv, go, q, k, v, o, acc);
}
