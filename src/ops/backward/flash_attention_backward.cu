/**
 * @file flash_attention_backward.cu
 * @brief Memory-efficient CUDA backward pass for grouped-query attention.
 *
 * One warp owns one (batch, query position, query head) row, and a CTA handles
 * four rows. Each warp makes three streaming passes over K/V: online softmax
 * maximum/normalizer, softmax-dot-product reduction, then dQ/dK/dV. This avoids
 * the O(Q*K) attention-probability workspace. K/V are shared between GQA heads,
 * so their contributions use atomics; dQ has exactly one owner and needs none.
 */
#include "ops/flash_attention.h"

#include "core/cuda_check.h"
#include "core/device_guard.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <math_constants.h>

#include <cmath>
#include <climits>
#include <limits>
#include <stdexcept>

namespace {
constexpr int kThreads = 128;
constexpr int kWarpSize = 32;
constexpr int kWarpsPerBlock = kThreads / kWarpSize;

__device__ __forceinline__ float warp_sum(float value) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(0xffffffffU, value, offset);
    }
    return __shfl_sync(0xffffffffU, value, 0);
}

template <typename T>
__device__ __forceinline__ void atomic_add(T* address, float value) {
    atomicAdd(address, static_cast<T>(value));
}

template <typename T>
__launch_bounds__(kThreads)
__global__ void attention_backward_kernel(
    T* __restrict__ grad_query, T* __restrict__ grad_key, T* __restrict__ grad_value,
    const T* __restrict__ grad_output, const T* __restrict__ query,
    const T* __restrict__ key, const T* __restrict__ value,
    int batch_size, int query_sequence, int key_value_sequence,
    int query_heads, int kv_heads, int head_dim, float scale,
    bool causal, int query_position_offset, bool accumulate_grad_query
) {
    const int lane = static_cast<int>(threadIdx.x) & (kWarpSize - 1);
    const int warp = static_cast<int>(threadIdx.x) / kWarpSize;
    const std::size_t linear = static_cast<std::size_t>(blockIdx.x) * kWarpsPerBlock + warp;
    const std::size_t row_count = static_cast<std::size_t>(batch_size) * query_sequence * query_heads;
    if (linear >= row_count) return;
    const int query_head = static_cast<int>(linear % query_heads);
    const int query_pos = static_cast<int>((linear / query_heads) % query_sequence);
    const int batch = static_cast<int>(linear / (static_cast<std::size_t>(query_heads) * query_sequence));

    const int kv_head = query_head / (query_heads / kv_heads);
    const int visible = causal ? min(key_value_sequence, query_position_offset + query_pos + 1)
                               : key_value_sequence;
    const std::size_t q_base =
        (static_cast<std::size_t>(batch) * query_sequence + query_pos) * query_heads * head_dim
        + static_cast<std::size_t>(query_head) * head_dim;
    const std::size_t kv_batch_base = static_cast<std::size_t>(batch) * key_value_sequence * kv_heads * head_dim;
    // Each warp owns a complete attention row. This eliminates all block-wide
    // barriers from the token loop and lets a 128-thread CTA process four rows.
    float row_max = -CUDART_INF_F;
    float row_normalizer = 0.0f;
    for (int k_pos = 0; k_pos < visible; ++k_pos) {
        const std::size_t kv_base = kv_batch_base +
            (static_cast<std::size_t>(k_pos) * kv_heads + kv_head) * head_dim;
        float dot = 0.0f;
        for (int d = lane; d < head_dim; d += kWarpSize)
            dot = fmaf(static_cast<float>(query[q_base + d]), static_cast<float>(key[kv_base + d]), dot);
        dot = warp_sum(dot) * scale;
        const float new_max = fmaxf(row_max, dot);
        const float previous_scale = row_max == -CUDART_INF_F ? 0.0f : expf(row_max - new_max);
        row_normalizer = row_normalizer * previous_scale + expf(dot - new_max);
        row_max = new_max;
    }

    float softmax_dot = 0.0f;
    for (int k_pos = 0; k_pos < visible; ++k_pos) {
        const std::size_t kv_base = kv_batch_base +
            (static_cast<std::size_t>(k_pos) * kv_heads + kv_head) * head_dim;
        float score = 0.0f;
        float d_probability = 0.0f;
        for (int d = lane; d < head_dim; d += kWarpSize) {
            score = fmaf(static_cast<float>(query[q_base + d]), static_cast<float>(key[kv_base + d]), score);
            d_probability = fmaf(static_cast<float>(grad_output[q_base + d]), static_cast<float>(value[kv_base + d]), d_probability);
        }
        score = warp_sum(score) * scale;
        d_probability = warp_sum(d_probability);
        const float probability = expf(score - row_max) / row_normalizer;
        softmax_dot += probability * d_probability;
    }

    // A lane owns dimensions lane, lane+32, ... for the complete K/V stream.
    float d_query[4] = {};
    for (int k_pos = 0; k_pos < visible; ++k_pos) {
            const std::size_t kv_base = kv_batch_base +
                (static_cast<std::size_t>(k_pos) * kv_heads + kv_head) * head_dim;
            float score_partial = 0.0f;
            float d_probability_partial = 0.0f;
            for (int column = lane; column < head_dim; column += kWarpSize) {
                score_partial = fmaf(static_cast<float>(query[q_base + column]), static_cast<float>(key[kv_base + column]), score_partial);
                d_probability_partial = fmaf(static_cast<float>(grad_output[q_base + column]), static_cast<float>(value[kv_base + column]), d_probability_partial);
            }
            const float score = warp_sum(score_partial) * scale;
            const float d_probability = warp_sum(d_probability_partial);
            const float probability = expf(score - row_max) / row_normalizer;
            const float d_score = probability * (d_probability - softmax_dot);
            int owned = 0;
            for (int d = lane; d < head_dim; d += kWarpSize, ++owned) {
                d_query[owned] = fmaf(d_score * scale, static_cast<float>(key[kv_base + d]), d_query[owned]);
                atomic_add(grad_key + kv_base + d, d_score * scale * static_cast<float>(query[q_base + d]));
                atomic_add(grad_value + kv_base + d,
                           probability * static_cast<float>(grad_output[q_base + d]));
            }
    }
    int owned = 0;
    for (int d = lane; d < head_dim; d += kWarpSize, ++owned) {
        const float result = accumulate_grad_query
            ? static_cast<float>(grad_query[q_base + d]) + d_query[owned]
            : d_query[owned];
        grad_query[q_base + d] = static_cast<T>(result);
    }
}

template <typename T>
void launch_attention_backward(
    T* grad_query, T* grad_key, T* grad_value, const T* grad_output,
    const T* query, const T* key, const T* value, unsigned int blocks,
    int batch, int qs, int ks, int qh, int kh, int dim, float scale,
    bool causal, int query_position_offset, bool accumulate_grads,
    cudaStream_t stream
) {
    const auto args = dim3(blocks);
    attention_backward_kernel<T><<<args, kThreads, 0, stream>>>(
        grad_query, grad_key, grad_value, grad_output, query, key, value,
        batch, qs, ks, qh, kh, dim, scale, causal, query_position_offset,
        accumulate_grads);
}

void validate(const Tensor& gq, const Tensor& gk, const Tensor& gv, const Tensor& go,
              const Tensor& q, const Tensor& k, const Tensor& v, const FlashAttentionOptions& o) {
    const Tensor* tensors[] = {&gq, &gk, &gv, &go, &q, &k, &v};
    for (const Tensor* tensor : tensors) {
        if (tensor->device_type() != DeviceType::CUDA || tensor->dtype() != q.dtype())
            throw std::invalid_argument("flash_gqa_attention_backward: tensors must be CUDA and have one dtype");
    }
    if (q.dtype() != Dtype::F16 && q.dtype() != Dtype::BF16 && q.dtype() != Dtype::F32)
        throw std::invalid_argument("flash_gqa_attention_backward: unsupported dtype");
    if (q.dim() != 4 || k.dim() != 4 || v.dim() != 4 || go.shape() != q.shape() ||
        gq.shape() != q.shape() || gv.shape() != k.shape() || gk.shape() != k.shape())
        throw std::invalid_argument("flash_gqa_attention_backward: invalid gradient or operand shape");
    if (v.shape() != k.shape() || o.num_query_heads == 0 || o.num_kv_heads == 0 || o.head_dim == 0 ||
        o.num_query_heads % o.num_kv_heads != 0 || q.size(2) != o.num_query_heads ||
        k.size(2) != o.num_kv_heads || q.size(3) != o.head_dim || k.size(3) != o.head_dim ||
        q.size(0) != k.size(0))
        throw std::invalid_argument("flash_gqa_attention_backward: invalid attention configuration");
    if (o.head_dim > static_cast<std::size_t>(kThreads))
        throw std::invalid_argument("flash_gqa_attention_backward: head_dim must not exceed 128");
    if (!std::isfinite(o.attention_scale) || (o.attention_scale < 0.0f) ||
        (o.causal && o.query_position_offset + q.size(1) > k.size(1)))
        throw std::invalid_argument("flash_gqa_attention_backward: invalid scale or causal range");
    if (q.size(0) > INT_MAX || q.size(1) > INT_MAX || k.size(1) > INT_MAX || o.num_query_heads > INT_MAX ||
        o.num_kv_heads > INT_MAX || o.head_dim > INT_MAX)
        throw std::invalid_argument("flash_gqa_attention_backward: dimensions exceed CUDA limits");
}
}

void flash_gqa_attention_backward(
    Tensor& grad_query, Tensor& grad_key, Tensor& grad_value, const Tensor& grad_output,
    const Tensor& query, const Tensor& key, const Tensor& value, cudaStream_t stream,
    const FlashAttentionOptions& options, bool accumulate_grads
) {
    validate(grad_query, grad_key, grad_value, grad_output, query, key, value, options);
    DeviceGuard device_guard(0);
    if (!accumulate_grads) {
        CUDA_CHECK(cudaMemsetAsync(grad_key.raw_data(), 0, grad_key.nbytes(), stream));
        CUDA_CHECK(cudaMemsetAsync(grad_value.raw_data(), 0, grad_value.nbytes(), stream));
    }
    const int batch = static_cast<int>(query.size(0));
    const int qs = static_cast<int>(query.size(1));
    const int ks = static_cast<int>(key.size(1));
    const int qh = static_cast<int>(options.num_query_heads);
    const int kh = static_cast<int>(options.num_kv_heads);
    const int dim = static_cast<int>(options.head_dim);
    const unsigned long long rows = static_cast<unsigned long long>(batch) * qs * qh;
    const unsigned long long blocks = (rows + kWarpsPerBlock - 1) / kWarpsPerBlock;
    if (blocks > static_cast<unsigned long long>(std::numeric_limits<unsigned int>::max()))
        throw std::invalid_argument("flash_gqa_attention_backward: CUDA grid is too large");
    const float scale = options.attention_scale > 0.0f ? options.attention_scale : rsqrtf(static_cast<float>(dim));
    const auto launch = [&]<typename T>() {
        launch_attention_backward(
            static_cast<T*>(grad_query.raw_data()), static_cast<T*>(grad_key.raw_data()),
            static_cast<T*>(grad_value.raw_data()), static_cast<const T*>(grad_output.raw_data()),
            static_cast<const T*>(query.raw_data()), static_cast<const T*>(key.raw_data()),
            static_cast<const T*>(value.raw_data()), static_cast<unsigned int>(blocks),
            batch, qs, ks, qh, kh, dim, scale, options.causal,
            static_cast<int>(options.query_position_offset), accumulate_grads, stream);
    };
    if (query.dtype() == Dtype::F16) launch.template operator()<__half>();
    else if (query.dtype() == Dtype::BF16) launch.template operator()<__nv_bfloat16>();
    else launch.template operator()<float>();
    CUDA_CHECK(cudaGetLastError());
}
