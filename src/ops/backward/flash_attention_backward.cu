/**
 * @file flash_attention_backward.cu
 * @brief Memory-efficient CUDA backward pass for grouped-query attention.
 *
 * One CTA owns one (batch, query position, query head) row. It makes three
 * streaming passes over K/V: softmax maximum/normalizer, softmax-dot-product
 * reduction, then dQ/dK/dV. This avoids the O(Q*K) attention-probability
 * workspace. K/V are shared between GQA heads, so their contributions use
 * atomics; dQ has exactly one owning CTA and needs none.
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
constexpr int kThreads = 256;

__device__ __forceinline__ float block_sum(float value, float* scratch) {
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int warp = static_cast<int>(threadIdx.x) >> 5;
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(0xffffffffU, value, offset);
    }
    if (lane == 0) scratch[warp] = value;
    __syncthreads();
    value = threadIdx.x < 8 ? scratch[lane] : 0.0f;
    if (warp == 0) {
        for (int offset = 16; offset > 0; offset >>= 1) {
            value += __shfl_down_sync(0xffffffffU, value, offset);
        }
        if (lane == 0) scratch[0] = value;
    }
    __syncthreads();
    return scratch[0];
}

__device__ __forceinline__ float block_max(float value, float* scratch) {
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int warp = static_cast<int>(threadIdx.x) >> 5;
    for (int offset = 16; offset > 0; offset >>= 1) {
        value = fmaxf(value, __shfl_down_sync(0xffffffffU, value, offset));
    }
    if (lane == 0) scratch[warp] = value;
    __syncthreads();
    value = static_cast<int>(threadIdx.x) < 8 ? scratch[lane] : -CUDART_INF_F;
    if (warp == 0) {
        for (int offset = 16; offset > 0; offset >>= 1) {
            value = fmaxf(value, __shfl_down_sync(0xffffffffU, value, offset));
        }
        if (lane == 0) scratch[0] = value;
    }
    __syncthreads();
    return scratch[0];
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
    const int linear = static_cast<int>(blockIdx.x);
    const int query_head = linear % query_heads;
    const int query_pos = (linear / query_heads) % query_sequence;
    const int batch = linear / (query_heads * query_sequence);
    if (batch >= batch_size) return;

    const int kv_head = query_head / (query_heads / kv_heads);
    const int visible = causal ? min(key_value_sequence, query_position_offset + query_pos + 1)
                               : key_value_sequence;
    const std::size_t q_base =
        (static_cast<std::size_t>(batch) * query_sequence + query_pos) * query_heads * head_dim
        + static_cast<std::size_t>(query_head) * head_dim;
    const std::size_t kv_batch_base = static_cast<std::size_t>(batch) * key_value_sequence * kv_heads * head_dim;
    __shared__ float reduction[8];
    __shared__ float max_score;
    __shared__ float normalizer;
    __shared__ float softmax_dot;

    float row_max = -CUDART_INF_F;
    for (int k_pos = 0; k_pos < visible; ++k_pos) {
        const std::size_t kv_base = kv_batch_base +
            (static_cast<std::size_t>(k_pos) * kv_heads + kv_head) * head_dim;
        float dot = 0.0f;
        for (int d = static_cast<int>(threadIdx.x); d < head_dim; d += static_cast<int>(blockDim.x))
            dot = fmaf(static_cast<float>(query[q_base + d]), static_cast<float>(key[kv_base + d]), dot);
        dot = block_sum(dot, reduction) * scale;
        if (static_cast<int>(threadIdx.x) == 0) row_max = fmaxf(row_max, dot);
        __syncthreads();
    }
    if (static_cast<int>(threadIdx.x) == 0) max_score = row_max;
    __syncthreads();

    float local_normalizer = 0.0f;
    for (int k_pos = 0; k_pos < visible; ++k_pos) {
        const std::size_t kv_base = kv_batch_base +
            (static_cast<std::size_t>(k_pos) * kv_heads + kv_head) * head_dim;
        float dot = 0.0f;
        for (int d = static_cast<int>(threadIdx.x); d < head_dim; d += static_cast<int>(blockDim.x))
            dot = fmaf(static_cast<float>(query[q_base + d]), static_cast<float>(key[kv_base + d]), dot);
        dot = block_sum(dot, reduction) * scale;
        if (static_cast<int>(threadIdx.x) == 0) local_normalizer += expf(dot - max_score);
        __syncthreads();
    }
    if (threadIdx.x == 0) normalizer = local_normalizer;
    __syncthreads();

    float local_softmax_dot = 0.0f;
    for (int k_pos = 0; k_pos < visible; ++k_pos) {
        const std::size_t kv_base = kv_batch_base +
            (static_cast<std::size_t>(k_pos) * kv_heads + kv_head) * head_dim;
        float score = 0.0f;
        float d_probability = 0.0f;
        for (int d = static_cast<int>(threadIdx.x); d < head_dim; d += static_cast<int>(blockDim.x)) {
            score = fmaf(static_cast<float>(query[q_base + d]), static_cast<float>(key[kv_base + d]), score);
            d_probability = fmaf(static_cast<float>(grad_output[q_base + d]), static_cast<float>(value[kv_base + d]), d_probability);
        }
        score = block_sum(score, reduction) * scale;
        d_probability = block_sum(d_probability, reduction);
        if (threadIdx.x == 0)
            local_softmax_dot += expf(score - max_score) / normalizer * d_probability;
        __syncthreads();
    }
    if (threadIdx.x == 0) softmax_dot = local_softmax_dot;
    __syncthreads();

    // Every thread must participate in the reductions below, including when
    // head_dim is smaller than the CTA. Each active lane owns one (or more)
    // gradient dimensions across all streamed K/V rows.
    float d_query = 0.0f;
    for (int k_pos = 0; k_pos < visible; ++k_pos) {
            const std::size_t kv_base = kv_batch_base +
                (static_cast<std::size_t>(k_pos) * kv_heads + kv_head) * head_dim;
            float score_partial = 0.0f;
            float d_probability_partial = 0.0f;
            for (int column = static_cast<int>(threadIdx.x); column < head_dim; column += static_cast<int>(blockDim.x)) {
                score_partial = fmaf(static_cast<float>(query[q_base + column]), static_cast<float>(key[kv_base + column]), score_partial);
                d_probability_partial = fmaf(static_cast<float>(grad_output[q_base + column]), static_cast<float>(value[kv_base + column]), d_probability_partial);
            }
            const float score = block_sum(score_partial, reduction) * scale;
            const float d_probability = block_sum(d_probability_partial, reduction);
            const float probability = expf(score - max_score) / normalizer;
            const float d_score = probability * (d_probability - softmax_dot);
            for (int d = static_cast<int>(threadIdx.x); d < head_dim; d += static_cast<int>(blockDim.x)) {
                d_query = fmaf(d_score * scale, static_cast<float>(key[kv_base + d]), d_query);
                atomic_add(grad_key + kv_base + d, d_score * scale * static_cast<float>(query[q_base + d]));
                atomic_add(grad_value + kv_base + d,
                           probability * static_cast<float>(grad_output[q_base + d]));
            }
    }
    for (int d = static_cast<int>(threadIdx.x); d < head_dim; d += static_cast<int>(blockDim.x)) {
        if (accumulate_grad_query) atomic_add(grad_query + q_base + d, d_query);
        else grad_query[q_base + d] = static_cast<T>(d_query);
    }
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
        throw std::invalid_argument("flash_gqa_attention_backward: head_dim must not exceed 256");
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
        CUDA_CHECK(cudaMemsetAsync(grad_query.raw_data(), 0, grad_query.nbytes(), stream));
        CUDA_CHECK(cudaMemsetAsync(grad_key.raw_data(), 0, grad_key.nbytes(), stream));
        CUDA_CHECK(cudaMemsetAsync(grad_value.raw_data(), 0, grad_value.nbytes(), stream));
    }
    const int batch = static_cast<int>(query.size(0));
    const int qs = static_cast<int>(query.size(1));
    const int ks = static_cast<int>(key.size(1));
    const int qh = static_cast<int>(options.num_query_heads);
    const int kh = static_cast<int>(options.num_kv_heads);
    const int dim = static_cast<int>(options.head_dim);
    const unsigned long long blocks = static_cast<unsigned long long>(batch) * qs * qh;
    if (blocks > static_cast<unsigned long long>(std::numeric_limits<unsigned int>::max()))
        throw std::invalid_argument("flash_gqa_attention_backward: CUDA grid is too large");
    const float scale = options.attention_scale > 0.0f ? options.attention_scale : rsqrtf(static_cast<float>(dim));
    if (query.dtype() == Dtype::F16)
        attention_backward_kernel<__half><<<static_cast<unsigned int>(blocks), kThreads, 0, stream>>>(static_cast<__half*>(grad_query.raw_data()), static_cast<__half*>(grad_key.raw_data()), static_cast<__half*>(grad_value.raw_data()), static_cast<const __half*>(grad_output.raw_data()), static_cast<const __half*>(query.raw_data()), static_cast<const __half*>(key.raw_data()), static_cast<const __half*>(value.raw_data()), batch, qs, ks, qh, kh, dim, scale, options.causal, static_cast<int>(options.query_position_offset), accumulate_grads);
    else if (query.dtype() == Dtype::BF16)
        attention_backward_kernel<__nv_bfloat16><<<static_cast<unsigned int>(blocks), kThreads, 0, stream>>>(static_cast<__nv_bfloat16*>(grad_query.raw_data()), static_cast<__nv_bfloat16*>(grad_key.raw_data()), static_cast<__nv_bfloat16*>(grad_value.raw_data()), static_cast<const __nv_bfloat16*>(grad_output.raw_data()), static_cast<const __nv_bfloat16*>(query.raw_data()), static_cast<const __nv_bfloat16*>(key.raw_data()), static_cast<const __nv_bfloat16*>(value.raw_data()), batch, qs, ks, qh, kh, dim, scale, options.causal, static_cast<int>(options.query_position_offset), accumulate_grads);
    else
        attention_backward_kernel<float><<<static_cast<unsigned int>(blocks), kThreads, 0, stream>>>(static_cast<float*>(grad_query.raw_data()), static_cast<float*>(grad_key.raw_data()), static_cast<float*>(grad_value.raw_data()), static_cast<const float*>(grad_output.raw_data()), static_cast<const float*>(query.raw_data()), static_cast<const float*>(key.raw_data()), static_cast<const float*>(value.raw_data()), batch, qs, ks, qh, kh, dim, scale, options.causal, static_cast<int>(options.query_position_offset), accumulate_grads);
    CUDA_CHECK(cudaGetLastError());
}
