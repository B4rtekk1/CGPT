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
            static_cast<std::uint16_t *>(p)[i] = static_cast<std::uint16_t>(_mm_cvtsi128_si32(_mm_cvtps_ph(_mm_set_ss(x), 0)));
            return;
        }
        std::uint32_t b;
        std::memcpy(&b, &x, 4);
        static_cast<std::uint16_t *>(p)[i] = static_cast<std::uint16_t>((b + 0x7fff + ((b >> 16) & 1)) >> 16);
    }

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

    void validate(const Tensor &gq, const Tensor &gk, const Tensor &gv, const Tensor &go, const Tensor &q,
                  const Tensor &k, const Tensor &v, const FlashAttentionOptions &o) {
        const Tensor *ts[] = {&gq, &gk, &gv, &go, &q, &k, &v};
        for (const auto *x: ts)if (x->device_type() != DeviceType::CPU || x->dtype() != q.dtype() || !
                                   is_floating_point(x->dtype()))throw std::invalid_argument(
            "CPU attention backward: tensors must be CPU floating-point tensors of one dtype");
        if (q.dim() != 4 || k.dim() != 4 || v.dim() != 4 || go.shape() != q.shape() || gq.shape() != q.shape() || gk.
            shape() != k.shape() || gv.shape() != k.shape() || v.shape() != k.shape() || !o.num_query_heads || !o.
            num_kv_heads || !o.head_dim || o.num_query_heads % o.num_kv_heads || q.shape()[0] != k.shape()[0] || q.
            shape()[2] != o.num_query_heads || k.shape()[2] != o.num_kv_heads || q.shape()[3] != o.head_dim || k.shape()
            [3] != o.head_dim || !std::isfinite(o.attention_scale) || (
                o.causal && (o.query_position_offset > k.shape()[1] || q.shape()[1] > k.shape()[1] - o.
                             query_position_offset)))throw std::invalid_argument(
            "CPU attention backward: invalid tensors or options");
    }

    void zero_if_needed(Tensor &x, bool accumulate) { if (!accumulate)std::memset(x.raw_data(), 0, x.nbytes()); }
}

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
        const auto z = static_cast<std::size_t>(task), hq = z % H, qi = (z / H) % QS, b = z / (H * QS), kvh = hq / (H / KV), qb =
                ((b * QS + qi) * H + hq) * D;
        const auto last = o.causal ? o.query_position_offset + qi + 1 : KS;
        std::vector<float> p(last);
        float mx = -std::numeric_limits<float>::infinity();
        for (std::size_t kj = 0; kj < last; ++kj)mx = std::max(
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
        const auto z = static_cast<std::size_t>(task), d = z % D, kj = (z / D) % KS, kvh = (z / D / KS) % KV, b = z / (D * KS * KV);
        const auto kb = ((b * KS + kj) * KV + kvh) * D;
        float dk = 0, dv = 0;
        for (std::size_t qi = 0; qi < QS; ++qi) {
            const auto last = o.causal ? o.query_position_offset + qi + 1 : KS;
            if (kj >= last)continue;
            for (std::size_t hq = kvh * (H / KV); hq < (kvh + 1) * (H / KV); ++hq) {
                const auto qb = ((b * QS + qi) * H + hq) * D;
                float mx = -std::numeric_limits<float>::infinity();
                for (std::size_t kk = 0; kk < last; ++kk)mx = std::max(
                                                             mx, dot(qp, kp, t, qb, ((b * KS + kk) * KV + kvh) * D,
                                                                     D) * scale);
                float den = 0;
                for (std::size_t kk = 0; kk < last; ++kk)den += std::exp(
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

void flash_gqa_attention_backward_with_lse_cpu(Tensor &gq, Tensor &gk, Tensor &gv, const Tensor &go,
                                               const Tensor &output, const Tensor &lse, const Tensor &q,
                                               const Tensor &k, const Tensor &v, const FlashAttentionOptions &o,
                                               bool acc) {
    if (output.device_type() != DeviceType::CPU || output.shape() != q.shape() || output.dtype() != q.dtype() || lse.
        device_type() != DeviceType::CPU || lse.dtype() != Dtype::F32 || lse.shape() != std::vector<std::size_t>{
            q.shape()[0], q.shape()[1], q.shape()[2]
        })throw std::invalid_argument("CPU attention backward: invalid output or logsumexp");
    flash_gqa_attention_backward_cpu(gq, gk, gv, go, q, k, v, o, acc);
}
