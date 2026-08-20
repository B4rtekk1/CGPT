/**
 * @file attention_backward.cu
 * @brief Memory-efficient CUDA backward pass for grouped-query attention.
 *
 * One warp owns one (batch, query position, query head) row, and a CTA handles
 * four rows. Each warp makes two streaming passes over K/V: a fused online
 * softmax-statistics + softmax-dot pass, then dQ/dK/dV. This avoids the
 * O(Q*K) attention-probability workspace. K/V are shared between GQA heads,
 * so their contributions use atomics; dQ has exactly one owner and needs none.
 *
 * Optimization notes vs. a naive 3-pass implementation:
 *  - Q and grad_output for a row are invariant across the k_pos loop, so they
 *    are loaded from global memory exactly once (registers; also staged to
 *    shared memory in the grouped path, where the other 3 warps in the GQA
 *    group need to read them for the dK/dV reduction). Previously they were
 *    re-read from global memory on every iteration of every pass.
 *  - softmax_dot (= sum_j P_ij * dP_ij) is accumulated online in the same
 *    sweep that computes row_max/row_normalizer, using the identical
 *    rescale-on-new-max update rule the normalizer itself already uses. This
 *    removes an entire redundant sweep over K/V that only existed to
 *    recompute scores already computed (and discarded) once before.
 *  - Net effect: Q*K/dO*V dot products drop from 5 to 4 per (row, key), and
 *    Q/grad_output global traffic drops from up to 3x (up to 7x in the
 *    grouped path) to exactly 1x.
 */
#include "ops/attention.h"

#include "core/cuda_check.h"
#include "core/device_guard.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <math_constants.h>
#include <mma.h>

#include <cmath>
#include <climits>
#include <limits>
#include <stdexcept>

namespace {
    constexpr int kThreads = 128;
    constexpr int kWarpSize = 32;
    constexpr int kWarpsPerBlock = kThreads / kWarpSize;
    constexpr int kMaxOwnedDims = 4; // head_dim <= 128 => at most 4 dims/lane

    // The intrinsic maps to the GPU's fast approximate exponential.  This is the
    // normal accuracy/performance trade-off for FP16/BF16 attention and avoids a
    // costly libdevice call on builds that don't pass --use_fast_math.  Set this
    // macro to 0 when validating against a strict FP32 reference.
#ifndef FLASH_ATTENTION_USE_FAST_MATH
#define FLASH_ATTENTION_USE_FAST_MATH 1
#endif

    __device__ __forceinline__ float attention_exp(float x) {
#if FLASH_ATTENTION_USE_FAST_MATH
        return __expf(x);
#else
        return expf(x);
#endif
    }

    // The score and dP reductions always have exactly the same producer lanes.
    // Reducing them together gives the compiler two independent dependency chains
    // between shuffle instructions and removes a second shuffle-loop branch.
    /** @brief Reduces two independent FP32 values across a warp. */
    __device__ __forceinline__ void warp_sum_pair(float &first, float &second) {
        for (int offset = 16; offset > 0; offset >>= 1) {
            first += __shfl_down_sync(0xffffffffU, first, offset);
            second += __shfl_down_sync(0xffffffffU, second, offset);
        }
        first = __shfl_sync(0xffffffffU, first, 0);
        second = __shfl_sync(0xffffffffU, second, 0);
    }

    /** @brief Atomically adds an FP32 value converted to storage type @p T. */
    template<typename T>
    __device__ __forceinline__ void atomic_add(T *address, float value) {
        atomicAdd(address, static_cast<T>(value));
    }

    // HeadDim is a compile-time specialization for the overwhelmingly common
    // 32/64/128 cases.  It makes the per-lane register loops fixed-size, while 0
    // preserves the generic path for unusual head dimensions.
    /**
     * @brief Memory-efficient streaming GQA attention backward kernel.
     *
     * The kernel computes dQ, dK, and dV without materializing the attention
     * probability matrix. It supports causal masking, grouped-query head
     * sharing, and compile-time head-dimension specializations.
     *
     * @tparam T Input and gradient storage type.
     * @tparam HeadDim Compile-time head dimension, or 0 for generic dispatch.
     * @tparam GroupedFour Enables the optimized four-warp GQA path.
     */
    template<typename T, int HeadDim, bool GroupedFour>
    __launch_bounds__(kThreads)
    __global__ void attention_backward_kernel(
        T * __restrict__ grad_query, T * __restrict__ grad_key, T * __restrict__ grad_value,
        const T * __restrict__ grad_output, const T * __restrict__ query,
        const T * __restrict__ key, const T * __restrict__ value,
        int batch_size, int query_sequence, int key_value_sequence,
        int query_heads, int kv_heads, int head_dim, float scale,
        bool causal, int query_position_offset, bool accumulate_grad_query
    ) {
        const int lane = static_cast<int>(threadIdx.x) & (kWarpSize - 1);
        const int warp = static_cast<int>(threadIdx.x) / kWarpSize;
        const int dimensions = HeadDim == 0 ? head_dim : HeadDim;
        const std::size_t linear = static_cast<std::size_t>(blockIdx.x) * kWarpsPerBlock + warp;
        const std::size_t row_count = static_cast<std::size_t>(batch_size) * query_sequence * query_heads;
        if (linear >= row_count) return;
        const int query_head = static_cast<int>(linear % query_heads);
        const int query_pos = static_cast<int>((linear / query_heads) % query_sequence);
        const int batch = static_cast<int>(linear / (static_cast<std::size_t>(query_heads) * query_sequence));

        const int kv_head = query_head / (query_heads / kv_heads);
        const int visible = causal
                                ? min(key_value_sequence, query_position_offset + query_pos + 1)
                                : key_value_sequence;
        const std::size_t q_base =
                (static_cast<std::size_t>(batch) * query_sequence + query_pos) * query_heads * head_dim
                + static_cast<std::size_t>(query_head) * head_dim;
        const std::size_t kv_batch_base = static_cast<std::size_t>(batch) * key_value_sequence * kv_heads * head_dim;

        // Q and grad_output for this row never change across k_pos. Cache them in
        // registers once instead of re-reading (and re-casting from T to float)
        // from global memory on every iteration of every pass below.
        float q_reg[kMaxOwnedDims];
        float do_reg[kMaxOwnedDims];
        {
            int owned = 0;
#pragma unroll
            for (int d = lane; d < dimensions; d += kWarpSize, ++owned) {
                q_reg[owned] = static_cast<float>(query[q_base + d]);
                do_reg[owned] = static_cast<float>(grad_output[q_base + d]);
            }
        }

        // Each warp owns a complete attention row. This eliminates all block-wide
        // barriers from the token loop and lets a 128-thread CTA process four rows.
        float row_max = -CUDART_INF_F;
        float row_normalizer = 0.0f;
        float d_numerator = 0.0f; // unnormalized sum_j exp(score_j - row_max) * dP_j

        if constexpr (GroupedFour) {
            // The four warps in the CTA are the four query heads which share one
            // K/V head. Stage this row's Q/dO into shared memory too, so the dK/dV
            // reduction later can read the other group members' rows without ever
            // touching global memory for Q/grad_output again.
            __shared__ float q_group[kWarpsPerBlock][128];
            __shared__ float do_group[kWarpsPerBlock][128];
            constexpr int kOwnedDimensions = (HeadDim + kWarpSize - 1) / kWarpSize;
#pragma unroll
            for (int owned = 0; owned < kOwnedDimensions; ++owned) {
                const int d = lane + owned * kWarpSize;
                q_group[warp][d] = q_reg[owned];
                do_group[warp][d] = do_reg[owned];
            }
            __syncthreads(); // Safe: qh/kh==4 forces row_count % 4 == 0, so no warp
            // in a grouped four-head launch ever early-returns above.

            // Fused pass: online softmax stats AND the unnormalized softmax_dot
            // numerator, accumulated together using the same rescale-on-new-max
            // rule the normalizer itself uses. Replaces two separate K/V sweeps.
            for (int k_pos = 0; k_pos < visible; ++k_pos) {
                const std::size_t kv_base = kv_batch_base +
                                            (static_cast<std::size_t>(k_pos) * kv_heads + kv_head) * head_dim;
                float score_partial = 0.0f;
                float d_probability_partial = 0.0f;
#pragma unroll
                for (int owned = 0; owned < kOwnedDimensions; ++owned) {
                    const int d = lane + owned * kWarpSize;
                    score_partial = fmaf(q_reg[owned], static_cast<float>(key[kv_base + d]), score_partial);
                    d_probability_partial = fmaf(do_reg[owned], static_cast<float>(value[kv_base + d]),
                                                 d_probability_partial);
                }
                warp_sum_pair(score_partial, d_probability_partial);
                const float score = score_partial * scale;
                const float d_probability = d_probability_partial;
                const float new_max = fmaxf(row_max, score);
                const float correction = row_max == -CUDART_INF_F ? 0.0f : attention_exp(row_max - new_max);
                const float weight = attention_exp(score - new_max);
                row_normalizer = row_normalizer * correction + weight;
                d_numerator = d_numerator * correction + weight * d_probability;
                row_max = new_max;
            }
            const float inv_row_normalizer = row_normalizer > 0.0f
                                                 ? __fdividef(1.0f, row_normalizer)
                                                 : 0.0f;
            const float softmax_dot = d_numerator * inv_row_normalizer;

            float d_query[kMaxOwnedDims] = {};
            // Cache their scalar P/dS values a tile at a time, then let each warp
            // own a disjoint quarter of dK/dV. This changes 4 atomics per output
            // element into one without allocating an O(Q*K) workspace.
            constexpr int kKeyTile = 32;
            __shared__ float probabilities[kWarpsPerBlock][kKeyTile];
            __shared__ float d_scores[kWarpsPerBlock][kKeyTile];

            for (int tile = 0; tile < visible; tile += kKeyTile) {
                const int tile_size = min(kKeyTile, visible - tile);
                for (int offset = 0; offset < tile_size; ++offset) {
                    const int k_pos = tile + offset;
                    const std::size_t kv_base = kv_batch_base +
                                                (static_cast<std::size_t>(k_pos) * kv_heads + kv_head) * head_dim;
                    float score_partial = 0.0f;
                    float d_probability_partial = 0.0f;
                    // Keep K in registers across the score dot product and the
                    // dQ accumulation below instead of loading it from global
                    // memory a second time in the same offset iteration.
                    float k_reg[kMaxOwnedDims];
#pragma unroll
                    for (int owned = 0; owned < kOwnedDimensions; ++owned) {
                        const int column = lane + owned * kWarpSize;
                        k_reg[owned] = static_cast<float>(key[kv_base + column]);
                        score_partial = fmaf(q_reg[owned], k_reg[owned], score_partial);
                        d_probability_partial = fmaf(do_reg[owned], static_cast<float>(value[kv_base + column]),
                                                     d_probability_partial);
                    }
                    warp_sum_pair(score_partial, d_probability_partial);
                    const float score = score_partial * scale;
                    const float d_probability = d_probability_partial;
                    const float probability = attention_exp(score - row_max) * inv_row_normalizer;
                    const float d_score = probability * (d_probability - softmax_dot);
                    if (lane == 0) {
                        probabilities[warp][offset] = probability;
                        d_scores[warp][offset] = d_score;
                    }
#pragma unroll
                    for (int owned = 0; owned < kOwnedDimensions; ++owned) {
                        d_query[owned] = fmaf(d_score * scale, k_reg[owned], d_query[owned]);
                    }
                }
                __syncthreads();

                const int d = warp * kWarpSize + lane;
                if (d < dimensions) {
                    for (int offset = 0; offset < tile_size; ++offset) {
                        const int k_pos = tile + offset;
                        const std::size_t kv_base = kv_batch_base +
                                                    (static_cast<std::size_t>(k_pos) * kv_heads + kv_head) * head_dim;
                        float d_key = 0.0f;
                        float d_value = 0.0f;
#pragma unroll
                        for (int group_head = 0; group_head < kWarpsPerBlock; ++group_head) {
                            d_key = fmaf(d_scores[group_head][offset] * scale,
                                         q_group[group_head][d], d_key);
                            d_value = fmaf(probabilities[group_head][offset],
                                           do_group[group_head][d], d_value);
                        }
                        atomic_add(grad_key + kv_base + d, d_key);
                        atomic_add(grad_value + kv_base + d, d_value);
                    }
                }
                __syncthreads();
            }

            int owned = 0;
#pragma unroll
            for (int d = lane; d < dimensions; d += kWarpSize, ++owned) {
                const float result = accumulate_grad_query
                                         ? static_cast<float>(grad_query[q_base + d]) + d_query[owned]
                                         : d_query[owned];
                grad_query[q_base + d] = static_cast<T>(result);
            }
        } else {
            // A lane owns dimensions lane, lane+32, ... for the complete K/V stream.
            // Fused pass: see the derivation above the grouped four-head branch.
            for (int k_pos = 0; k_pos < visible; ++k_pos) {
                const std::size_t kv_base = kv_batch_base +
                                            (static_cast<std::size_t>(k_pos) * kv_heads + kv_head) * head_dim;
                float score_partial = 0.0f;
                float d_probability_partial = 0.0f;
                int owned = 0;
#pragma unroll
                for (int d = lane; d < dimensions; d += kWarpSize, ++owned) {
                    score_partial = fmaf(q_reg[owned], static_cast<float>(key[kv_base + d]), score_partial);
                    d_probability_partial = fmaf(do_reg[owned], static_cast<float>(value[kv_base + d]),
                                                 d_probability_partial);
                }
                warp_sum_pair(score_partial, d_probability_partial);
                const float score = score_partial * scale;
                const float d_probability = d_probability_partial;
                const float new_max = fmaxf(row_max, score);
                const float correction = row_max == -CUDART_INF_F ? 0.0f : attention_exp(row_max - new_max);
                const float weight = attention_exp(score - new_max);
                row_normalizer = row_normalizer * correction + weight;
                d_numerator = d_numerator * correction + weight * d_probability;
                row_max = new_max;
            }
            const float inv_row_normalizer = row_normalizer > 0.0f
                                                 ? __fdividef(1.0f, row_normalizer)
                                                 : 0.0f;
            const float softmax_dot = d_numerator * inv_row_normalizer;

            float d_query[kMaxOwnedDims] = {};
            for (int k_pos = 0; k_pos < visible; ++k_pos) {
                const std::size_t kv_base = kv_batch_base +
                                            (static_cast<std::size_t>(k_pos) * kv_heads + kv_head) * head_dim;
                float score_partial = 0.0f;
                float d_probability_partial = 0.0f;
                // K is needed twice this iteration (score, then dQ/dK below). Keep
                // the lane's owned elements in registers instead of re-issuing the
                // global load a second time.
                float k_reg[kMaxOwnedDims];
                int owned = 0;
#pragma unroll
                for (int d = lane; d < dimensions; d += kWarpSize, ++owned) {
                    k_reg[owned] = static_cast<float>(key[kv_base + d]);
                    score_partial = fmaf(q_reg[owned], k_reg[owned], score_partial);
                    d_probability_partial = fmaf(do_reg[owned], static_cast<float>(value[kv_base + d]),
                                                 d_probability_partial);
                }
                warp_sum_pair(score_partial, d_probability_partial);
                const float score = score_partial * scale;
                const float d_probability = d_probability_partial;
                const float probability = attention_exp(score - row_max) * inv_row_normalizer;
                const float d_score = probability * (d_probability - softmax_dot);
                owned = 0;
#pragma unroll
                for (int d = lane; d < dimensions; d += kWarpSize, ++owned) {
                    d_query[owned] = fmaf(d_score * scale, k_reg[owned], d_query[owned]);
                    atomic_add(grad_key + kv_base + d, d_score * scale * q_reg[owned]);
                    atomic_add(grad_value + kv_base + d, probability * do_reg[owned]);
                }
            }

            int owned = 0;
#pragma unroll
            for (int d = lane; d < dimensions; d += kWarpSize, ++owned) {
                const float result = accumulate_grad_query
                                         ? static_cast<float>(grad_query[q_base + d]) + d_query[owned]
                                         : d_query[owned];
                grad_query[q_base + d] = static_cast<T>(result);
            }
        }
    }

    // Training fast path: the forward pass supplies LSE and O.  Thus
    // D_i = sum_d dO_id * O_id is computed once, and P_ij = exp(S_ij - LSE_i).
    // This removes the complete streaming statistics pass from the legacy kernel.
    /**
     * @brief Streaming attention backward kernel using forward LSE values.
     *
     * Reuses the supplied log-sum-exp tensor to reconstruct probabilities and
     * therefore skips the forward softmax-statistics pass.
     *
     * @tparam T Input and gradient storage type.
     * @tparam HeadDim Compile-time head dimension, or 0 for generic dispatch.
     * @tparam GroupedFour Enables the optimized four-warp GQA path.
     */
    template<typename T, int HeadDim, bool GroupedFour>
    __launch_bounds__(kThreads)
    __global__ void attention_backward_lse_kernel(
        T * __restrict__ grad_query, T * __restrict__ grad_key, T * __restrict__ grad_value,
        const T * __restrict__ grad_output, const T * __restrict__ output,
        const float * __restrict__ logsumexp, const T * __restrict__ query,
        const T * __restrict__ key, const T * __restrict__ value,
        int batch_size, int query_sequence, int key_value_sequence,
        int query_heads, int kv_heads, int head_dim, float scale,
        bool causal, int query_position_offset, bool accumulate_grad_query
    ) {
        const int lane = static_cast<int>(threadIdx.x) & (kWarpSize - 1);
        const int warp = static_cast<int>(threadIdx.x) / kWarpSize;
        const int dimensions = HeadDim == 0 ? head_dim : HeadDim;
        const std::size_t linear = static_cast<std::size_t>(blockIdx.x) * kWarpsPerBlock + warp;
        const std::size_t row_count = static_cast<std::size_t>(batch_size) * query_sequence * query_heads;
        if (linear >= row_count) return;

        const int query_head = static_cast<int>(linear % query_heads);
        const int query_pos = static_cast<int>((linear / query_heads) % query_sequence);
        const int batch = static_cast<int>(linear / (static_cast<std::size_t>(query_heads) * query_sequence));
        const int kv_head = query_head / (query_heads / kv_heads);
        const int visible = causal
                                ? min(key_value_sequence, query_position_offset + query_pos + 1)
                                : key_value_sequence;
        const std::size_t q_base =
                (static_cast<std::size_t>(batch) * query_sequence + query_pos) * query_heads * head_dim
                + static_cast<std::size_t>(query_head) * head_dim;
        const std::size_t kv_batch_base = static_cast<std::size_t>(batch) * key_value_sequence * kv_heads * head_dim;
        const float row_lse = logsumexp[linear];

        float q_reg[kMaxOwnedDims];
        float do_reg[kMaxOwnedDims];
        float d_output = 0.0f;
        int owned = 0;
#pragma unroll
        for (int d = lane; d < dimensions; d += kWarpSize, ++owned) {
            q_reg[owned] = static_cast<float>(query[q_base + d]);
            do_reg[owned] = static_cast<float>(grad_output[q_base + d]);
            d_output = fmaf(do_reg[owned], static_cast<float>(output[q_base + d]), d_output);
        }
        float ignored = 0.0f;
        warp_sum_pair(d_output, ignored);

        float d_query[kMaxOwnedDims] = {};
        if constexpr (GroupedFour) {
            // Keep the four GQA rows together.  This preserves the LSE fast path
            // while reducing four dK/dV atomics per element to one.
            __shared__ float q_group[kWarpsPerBlock][128];
            __shared__ float do_group[kWarpsPerBlock][128];
            __shared__ float probabilities[kWarpsPerBlock][32];
            __shared__ float d_scores[kWarpsPerBlock][32];
            constexpr int kOwnedDimensions = (HeadDim + kWarpSize - 1) / kWarpSize;
#pragma unroll
            for (int owned_index = 0; owned_index < kOwnedDimensions; ++owned_index) {
                const int d = lane + owned_index * kWarpSize;
                q_group[warp][d] = q_reg[owned_index];
                do_group[warp][d] = do_reg[owned_index];
            }
            __syncthreads();

            constexpr int kKeyTile = 32;
            for (int tile = 0; tile < visible; tile += kKeyTile) {
                const int tile_size = min(kKeyTile, visible - tile);
                for (int offset = 0; offset < tile_size; ++offset) {
                    const int k_pos = tile + offset;
                    const std::size_t kv_base = kv_batch_base +
                                                (static_cast<std::size_t>(k_pos) * kv_heads + kv_head) * head_dim;
                    float score_partial = 0.0f;
                    float d_probability_partial = 0.0f;
                    // Same K-register-caching rationale as the base kernel.
                    float k_reg[kMaxOwnedDims];
#pragma unroll
                    for (int owned_index = 0; owned_index < kOwnedDimensions; ++owned_index) {
                        const int d = lane + owned_index * kWarpSize;
                        k_reg[owned_index] = static_cast<float>(key[kv_base + d]);
                        score_partial = fmaf(q_reg[owned_index], k_reg[owned_index], score_partial);
                        d_probability_partial = fmaf(do_reg[owned_index], static_cast<float>(value[kv_base + d]),
                                                     d_probability_partial);
                    }
                    warp_sum_pair(score_partial, d_probability_partial);
                    const float probability = attention_exp(score_partial * scale - row_lse);
                    const float d_score = probability * (d_probability_partial - d_output);
                    if (lane == 0) {
                        probabilities[warp][offset] = probability;
                        d_scores[warp][offset] = d_score;
                    }
#pragma unroll
                    for (int owned_index = 0; owned_index < kOwnedDimensions; ++owned_index) {
                        d_query[owned_index] = fmaf(d_score * scale, k_reg[owned_index], d_query[owned_index]);
                    }
                }
                __syncthreads();

                const int d = warp * kWarpSize + lane;
                if (d < dimensions) {
                    for (int offset = 0; offset < tile_size; ++offset) {
                        const std::size_t kv_base = kv_batch_base +
                                                    (static_cast<std::size_t>(tile + offset) * kv_heads + kv_head) *
                                                    head_dim;
                        float d_key = 0.0f;
                        float d_value = 0.0f;
#pragma unroll
                        for (int group_head = 0; group_head < kWarpsPerBlock; ++group_head) {
                            d_key = fmaf(d_scores[group_head][offset] * scale,
                                         q_group[group_head][d], d_key);
                            d_value = fmaf(probabilities[group_head][offset],
                                           do_group[group_head][d], d_value);
                        }
                        atomic_add(grad_key + kv_base + d, d_key);
                        atomic_add(grad_value + kv_base + d, d_value);
                    }
                }
                __syncthreads();
            }
        } else {
            for (int k_pos = 0; k_pos < visible; ++k_pos) {
                const std::size_t kv_base = kv_batch_base +
                                            (static_cast<std::size_t>(k_pos) * kv_heads + kv_head) * head_dim;
                float score_partial = 0.0f;
                float d_probability_partial = 0.0f;
                // Same rationale as the base kernel: keep K in registers across
                // the two uses within this iteration instead of a second global load.
                float k_reg[kMaxOwnedDims];
                owned = 0;
#pragma unroll
                for (int d = lane; d < dimensions; d += kWarpSize, ++owned) {
                    k_reg[owned] = static_cast<float>(key[kv_base + d]);
                    score_partial = fmaf(q_reg[owned], k_reg[owned], score_partial);
                    d_probability_partial = fmaf(do_reg[owned], static_cast<float>(value[kv_base + d]),
                                                 d_probability_partial);
                }
                warp_sum_pair(score_partial, d_probability_partial);
                const float probability = attention_exp(score_partial * scale - row_lse);
                const float d_score = probability * (d_probability_partial - d_output);
                owned = 0;
#pragma unroll
                for (int d = lane; d < dimensions; d += kWarpSize, ++owned) {
                    d_query[owned] = fmaf(d_score * scale, k_reg[owned], d_query[owned]);
                    atomic_add(grad_key + kv_base + d, d_score * scale * q_reg[owned]);
                    atomic_add(grad_value + kv_base + d, probability * do_reg[owned]);
                }
            }
        }
        owned = 0;
#pragma unroll
        for (int d = lane; d < dimensions; d += kWarpSize, ++owned) {
            const float result = accumulate_grad_query
                                     ? static_cast<float>(grad_query[q_base + d]) + d_query[owned]
                                     : d_query[owned];
            grad_query[q_base + d] = static_cast<T>(result);
        }
    }

    /** @brief Launches a specialized non-LSE attention backward kernel. */
    template<typename T, int HeadDim>
    void launch_attention_backward_specialized(
        T *grad_query, T *grad_key, T *grad_value, const T *grad_output,
        const T *query, const T *key, const T *value, unsigned int blocks,
        int batch, int qs, int ks, int qh, int kh, int dim, float scale,
        bool causal, int query_position_offset, bool accumulate_grads,
        cudaStream_t stream
    ) {
        const auto args = dim3(blocks);
        if (qh / kh == kWarpsPerBlock && (dim == 64 || dim == 128)) {
            attention_backward_kernel<T, HeadDim, true><<<args, kThreads, 0, stream>>>(
                grad_query, grad_key, grad_value, grad_output, query, key, value,
                batch, qs, ks, qh, kh, dim, scale, causal, query_position_offset,
                accumulate_grads);
        } else {
            attention_backward_kernel<T, HeadDim, false><<<args, kThreads, 0, stream>>>(
                grad_query, grad_key, grad_value, grad_output, query, key, value,
                batch, qs, ks, qh, kh, dim, scale, causal, query_position_offset,
                accumulate_grads);
        }
    }

    /** @brief Launches the generic non-LSE attention backward path. */
    template<typename T>
    void launch_attention_backward(
        T *grad_query, T *grad_key, T *grad_value, const T *grad_output,
        const T *query, const T *key, const T *value, unsigned int blocks,
        int batch, int qs, int ks, int qh, int kh, int dim, float scale,
        bool causal, int query_position_offset, bool accumulate_grads,
        cudaStream_t stream
    ) {
        switch (dim) {
            case 32:
                launch_attention_backward_specialized<T, 32>(grad_query, grad_key, grad_value, grad_output,
                                                             query, key, value, blocks, batch, qs, ks, qh, kh, dim,
                                                             scale, causal,
                                                             query_position_offset, accumulate_grads, stream);
                break;
            case 64:
                launch_attention_backward_specialized<T, 64>(grad_query, grad_key, grad_value, grad_output,
                                                             query, key, value, blocks, batch, qs, ks, qh, kh, dim,
                                                             scale, causal,
                                                             query_position_offset, accumulate_grads, stream);
                break;
            case 128:
                launch_attention_backward_specialized<T, 128>(grad_query, grad_key, grad_value, grad_output,
                                                              query, key, value, blocks, batch, qs, ks, qh, kh, dim,
                                                              scale, causal,
                                                              query_position_offset, accumulate_grads, stream);
                break;
            default:
                launch_attention_backward_specialized<T, 0>(grad_query, grad_key, grad_value, grad_output,
                                                            query, key, value, blocks, batch, qs, ks, qh, kh, dim,
                                                            scale, causal,
                                                            query_position_offset, accumulate_grads, stream);
                break;
        }
    }

    /** @brief Launches a specialized LSE-based attention backward kernel. */
    template<typename T, int HeadDim>
    void launch_attention_backward_lse_specialized(
        T *grad_query, T *grad_key, T *grad_value, const T *grad_output, const T *output,
        const float *logsumexp, const T *query, const T *key, const T *value, unsigned int blocks,
        int batch, int qs, int ks, int qh, int kh, int dim, float scale,
        bool causal, int query_position_offset, bool accumulate_grads, cudaStream_t stream
    ) {
        if (qh / kh == kWarpsPerBlock && (dim == 64 || dim == 128)) {
            attention_backward_lse_kernel<T, HeadDim, true><<<dim3(blocks), kThreads, 0, stream>>>(
                grad_query, grad_key, grad_value, grad_output, output, logsumexp, query, key, value,
                batch, qs, ks, qh, kh, dim, scale, causal, query_position_offset, accumulate_grads);
        } else {
            attention_backward_lse_kernel<T, HeadDim, false><<<dim3(blocks), kThreads, 0, stream>>>(
                grad_query, grad_key, grad_value, grad_output, output, logsumexp, query, key, value,
                batch, qs, ks, qh, kh, dim, scale, causal, query_position_offset, accumulate_grads);
        }
    }

    /** @brief Launches the generic LSE-based attention backward path. */
    template<typename T>
    void launch_attention_backward_lse(
        T *grad_query, T *grad_key, T *grad_value, const T *grad_output, const T *output,
        const float *logsumexp, const T *query, const T *key, const T *value, unsigned int blocks,
        int batch, int qs, int ks, int qh, int kh, int dim, float scale,
        bool causal, int query_position_offset, bool accumulate_grads, cudaStream_t stream
    ) {
        switch (dim) {
            case 32: launch_attention_backward_lse_specialized<T, 32>(grad_query, grad_key, grad_value, grad_output,
                                                                      output, logsumexp, query, key, value, blocks,
                                                                      batch, qs, ks, qh, kh, dim, scale, causal,
                                                                      query_position_offset, accumulate_grads, stream);
                break;
            case 64: launch_attention_backward_lse_specialized<T, 64>(grad_query, grad_key, grad_value, grad_output,
                                                                      output, logsumexp, query, key, value, blocks,
                                                                      batch, qs, ks, qh, kh, dim, scale, causal,
                                                                      query_position_offset, accumulate_grads, stream);
                break;
            case 128: launch_attention_backward_lse_specialized<T, 128>(grad_query, grad_key, grad_value, grad_output,
                                                                        output, logsumexp, query, key, value, blocks,
                                                                        batch, qs, ks, qh, kh, dim, scale, causal,
                                                                        query_position_offset, accumulate_grads,
                                                                        stream);
                break;
            default: launch_attention_backward_lse_specialized<T, 0>(grad_query, grad_key, grad_value, grad_output,
                                                                     output, logsumexp, query, key, value, blocks,
                                                                     batch, qs, ks, qh, kh, dim, scale, causal,
                                                                     query_position_offset, accumulate_grads, stream);
                break;
        }
    }

    // Ampere Tensor-Core fast path for the dominant training shape:
    // FP16, D=128 and exactly four query heads per KV head.  The implementation
    // follows the FA2 backward decomposition while adapting it to consumer
    // Ampere's smaller shared-memory budget:
    //   1. preprocess D_i = dot(dO_i, O_i),
    //   2. one packed-GQA CTA computes dQ for four heads while sharing K/V,
    //   3. one CTA owns a K/V tile; warps partition output columns for dK/dV.
    // QK^T, dO V^T, dS K, dS^T Q and P^T dO all use native
    // mma.sync.aligned.m16n8k16 instructions.  Global-to-shared transfers use
    // 16-byte cp.async transactions and shared rows are padded to reduce bank
    // conflicts on SM80/SM86.
    constexpr int kAmpereTile = 16;
    constexpr int kAmpereDqKeyTile = 32;
    constexpr int kAmpereDkvQueryTile = 32;
    constexpr int kAmpereHeadDim = 128;
    constexpr int kAmpereGqaRatio = 4;
    constexpr int kAmpereSmemLd = 136;  // 272-byte row stride shifts bank phase.
    constexpr int kAmpereVectorsPerRow = kAmpereHeadDim / 8;
    constexpr float kLog2E = 1.4426950408889634074f;

    struct Mma16816Accumulator {
        float x[4];
    };

    __device__ __forceinline__ void clear_mma(Mma16816Accumulator &accumulator) {
        accumulator.x[0] = 0.0f;
        accumulator.x[1] = 0.0f;
        accumulator.x[2] = 0.0f;
        accumulator.x[3] = 0.0f;
    }

    __device__ __forceinline__ unsigned shared_address(const void *pointer) {
        return static_cast<unsigned>(__cvta_generic_to_shared(pointer));
    }

    __device__ __forceinline__ void ldmatrix_a_m16k16_row(
        unsigned (&fragment)[4], const __half *base, int leading_dimension, int lane
    ) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
        const int matrix = lane >> 3;
        const int row = lane & 7;
        const int row_offset = (matrix & 1) * 8;
        const int column_offset = (matrix >> 1) * 8;
        const unsigned address = shared_address(
            base + (row_offset + row) * leading_dimension + column_offset);
        asm volatile(
            "ldmatrix.sync.aligned.m8n8.x4.shared.b16 "
            "{%0, %1, %2, %3}, [%4];\n"
            : "=r"(fragment[0]), "=r"(fragment[1]),
              "=r"(fragment[2]), "=r"(fragment[3])
            : "r"(address));
#else
        fragment[0] = fragment[1] = fragment[2] = fragment[3] = 0;
#endif
    }

    // Loads a 16x8 B operand corresponding to a transposed row-major source
    // (QK^T and dO V^T).  Each source row is one output column of B.
    __device__ __forceinline__ void ldmatrix_b_k16n8_col(
        unsigned (&fragment)[2], const __half *base, int leading_dimension, int lane
    ) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
        const int source_lane = lane & 15;
        const int matrix = source_lane >> 3;
        const int column = source_lane & 7;
        const unsigned address = shared_address(
            base + column * leading_dimension + matrix * 8);
        asm volatile(
            "ldmatrix.sync.aligned.m8n8.x2.shared.b16 "
            "{%0, %1}, [%2];\n"
            : "=r"(fragment[0]), "=r"(fragment[1])
            : "r"(address));
#else
        fragment[0] = fragment[1] = 0;
#endif
    }

    // Loads a normal row-major KxN tile and transposes the ldmatrix result into
    // the column-major operand layout required by mma.sync row.col.
    __device__ __forceinline__ void ldmatrix_b_k16n8_row_transposed(
        unsigned (&fragment)[2], const __half *base, int leading_dimension, int lane
    ) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
        const int source_row = lane & 15;
        const unsigned address = shared_address(base + source_row * leading_dimension);
        asm volatile(
            "ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 "
            "{%0, %1}, [%2];\n"
            : "=r"(fragment[0]), "=r"(fragment[1])
            : "r"(address));
#else
        fragment[0] = fragment[1] = 0;
#endif
    }

    __device__ __forceinline__ void mma_m16n8k16(
        Mma16816Accumulator &accumulator,
        const unsigned (&a)[4], const unsigned (&b)[2]
    ) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
        asm volatile(
            "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
            "{%0, %1, %2, %3}, "
            "{%4, %5, %6, %7}, "
            "{%8, %9}, "
            "{%0, %1, %2, %3};\n"
            : "+f"(accumulator.x[0]), "+f"(accumulator.x[1]),
              "+f"(accumulator.x[2]), "+f"(accumulator.x[3])
            : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
              "r"(b[0]), "r"(b[1]));
#endif
    }

    __device__ __forceinline__ void cp_async_16(
        __half *destination, const __half *source, bool valid
    ) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
        const unsigned dst = shared_address(destination);
        const int source_bytes = valid ? 16 : 0;
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::
                     "r"(dst), "l"(source), "r"(source_bytes));
#else
        if (valid) {
            *reinterpret_cast<uint4 *>(destination) =
                *reinterpret_cast<const uint4 *>(source);
        } else {
            *reinterpret_cast<uint4 *>(destination) = make_uint4(0, 0, 0, 0);
        }
#endif
    }

    __device__ __forceinline__ void cp_async_commit() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
        asm volatile("cp.async.commit_group;\n" ::);
#endif
    }

    __device__ __forceinline__ void cp_async_wait_all() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
        asm volatile("cp.async.wait_group 0;\n" ::);
#endif
    }


    // The m16n8k16 accumulator maps one logical score row to a 4-lane
    // subgroup.  These reductions exchange only those four lanes, avoiding
    // full-warp softmax reductions and their extra dependency chain.
    __device__ __forceinline__ float subgroup4_max(float value) {
        value = fmaxf(value, __shfl_xor_sync(0xffffffffU, value, 1, 4));
        value = fmaxf(value, __shfl_xor_sync(0xffffffffU, value, 2, 4));
        return value;
    }

    __device__ __forceinline__ float subgroup4_sum(float value) {
        value += __shfl_xor_sync(0xffffffffU, value, 1, 4);
        value += __shfl_xor_sync(0xffffffffU, value, 2, 4);
        return value;
    }

    template<int Heads>
    __device__ __forceinline__ void load_q_or_do_group_async(
        __half *destination, const __half *source,
        int batch, int query_begin, int query_sequence,
        int query_heads, int first_head
    ) {
        constexpr int kVectorsPerHead = kAmpereTile * kAmpereVectorsPerRow;
        constexpr int kVectorCount = Heads * kVectorsPerHead;
        for (int vector = static_cast<int>(threadIdx.x);
             vector < kVectorCount;
             vector += static_cast<int>(blockDim.x)) {
            const int local_head = vector / kVectorsPerHead;
            const int rem = vector - local_head * kVectorsPerHead;
            const int row = rem / kAmpereVectorsPerRow;
            const int column_vector = rem - row * kAmpereVectorsPerRow;
            const int query_position = query_begin + row;
            const int query_head = first_head + local_head;
            const bool valid = query_position < query_sequence;
            const std::size_t source_offset = valid
                ? ((static_cast<std::size_t>(batch) * query_sequence + query_position)
                    * query_heads + query_head) * kAmpereHeadDim
                    + column_vector * 8
                : 0;
            __half *dst = destination
                + local_head * kAmpereTile * kAmpereSmemLd
                + row * kAmpereSmemLd + column_vector * 8;
            cp_async_16(dst, source + source_offset, valid);
        }
        cp_async_commit();
    }

    __device__ __forceinline__ void load_kv_tile_async(
        __half *key_destination, __half *value_destination,
        const __half *key, const __half *value,
        int batch, int key_begin, int key_sequence,
        int kv_heads, int kv_head
    ) {
        constexpr int kVectorCount = kAmpereTile * kAmpereVectorsPerRow;
        for (int vector = static_cast<int>(threadIdx.x);
             vector < kVectorCount;
             vector += static_cast<int>(blockDim.x)) {
            const int row = vector / kAmpereVectorsPerRow;
            const int column_vector = vector - row * kAmpereVectorsPerRow;
            const int key_position = key_begin + row;
            const bool valid = key_position < key_sequence;
            const std::size_t source_offset = valid
                ? ((static_cast<std::size_t>(batch) * key_sequence + key_position)
                    * kv_heads + kv_head) * kAmpereHeadDim
                    + column_vector * 8
                : 0;
            __half *key_dst = key_destination + row * kAmpereSmemLd + column_vector * 8;
            __half *value_dst = value_destination + row * kAmpereSmemLd + column_vector * 8;
            cp_async_16(key_dst, key + source_offset, valid);
            cp_async_16(value_dst, value + source_offset, valid);
        }
        cp_async_commit();
    }


    template<int Rows>
    __device__ __forceinline__ void load_kv_rows_async(
        __half *key_destination, __half *value_destination,
        const __half *key, const __half *value,
        int batch, int key_begin, int key_sequence,
        int kv_heads, int kv_head
    ) {
        static_assert(Rows % 16 == 0);
        constexpr int kVectorCount = Rows * kAmpereVectorsPerRow;
        for (int vector = static_cast<int>(threadIdx.x);
             vector < kVectorCount;
             vector += static_cast<int>(blockDim.x)) {
            const int row = vector / kAmpereVectorsPerRow;
            const int column_vector = vector - row * kAmpereVectorsPerRow;
            const int key_position = key_begin + row;
            const bool valid = key_position < key_sequence;
            const std::size_t source_offset = valid
                ? ((static_cast<std::size_t>(batch) * key_sequence + key_position)
                    * kv_heads + kv_head) * kAmpereHeadDim
                    + column_vector * 8
                : 0;
            __half *key_dst = key_destination
                + row * kAmpereSmemLd + column_vector * 8;
            __half *value_dst = value_destination
                + row * kAmpereSmemLd + column_vector * 8;
            cp_async_16(key_dst, key + source_offset, valid);
            cp_async_16(value_dst, value + source_offset, valid);
        }
        cp_async_commit();
    }

    template<int Heads, int Rows>
    __device__ __forceinline__ void load_q_or_do_rows_async(
        __half *destination, const __half *source,
        int batch, int query_begin, int query_sequence,
        int query_heads, int first_head
    ) {
        static_assert(Rows % 16 == 0);
        constexpr int kVectorsPerHead = Rows * kAmpereVectorsPerRow;
        constexpr int kVectorCount = Heads * kVectorsPerHead;
        constexpr int kHeadStorage = Rows * kAmpereSmemLd;
        for (int vector = static_cast<int>(threadIdx.x);
             vector < kVectorCount;
             vector += static_cast<int>(blockDim.x)) {
            const int local_head = vector / kVectorsPerHead;
            const int rem = vector - local_head * kVectorsPerHead;
            const int row = rem / kAmpereVectorsPerRow;
            const int column_vector = rem - row * kAmpereVectorsPerRow;
            const int query_position = query_begin + row;
            const int query_head = first_head + local_head;
            const bool valid = query_position < query_sequence;
            const std::size_t source_offset = valid
                ? ((static_cast<std::size_t>(batch) * query_sequence + query_position)
                    * query_heads + query_head) * kAmpereHeadDim
                    + column_vector * 8
                : 0;
            __half *dst = destination + local_head * kHeadStorage
                + row * kAmpereSmemLd + column_vector * 8;
            cp_async_16(dst, source + source_offset, valid);
        }
        cp_async_commit();
    }

    /**
     * Computes the two forward statistics required by backward when the caller
     * did not retain them:
     *   LSE_i   = log(sum_j exp(S_ij))
     *   delta_i = sum_j P_ij * (dO_i dot V_j)
     *
     * One CTA owns a 16-row query tile for all four query heads sharing a KV
     * head.  Q, dO, K and V matrix products use Tensor Cores.  This replaces
     * the old scalar per-row statistics sweep.
     */
    template<bool Causal>
    __global__ __launch_bounds__(128, 2)
    void attention_backward_stats_ampere_f16_d128_gqa4(
        float * __restrict__ logsumexp,
        float * __restrict__ delta,
        const __half * __restrict__ grad_output,
        const __half * __restrict__ query,
        const __half * __restrict__ key,
        const __half * __restrict__ value,
        int batch_size, int query_sequence, int key_sequence,
        int query_heads, int kv_heads,
        float scale_log2, int query_position_offset
    ) {
        const int warp = static_cast<int>(threadIdx.x) >> 5;
        const int lane = static_cast<int>(threadIdx.x) & 31;
        const int lane_group = lane >> 2;
        const int lane_in_group = lane & 3;
        const int query_tiles = (query_sequence + kAmpereTile - 1) / kAmpereTile;
        const int linear = static_cast<int>(blockIdx.x);
        const int query_tile = linear % query_tiles;
        const int kv_head = (linear / query_tiles) % kv_heads;
        const int batch = linear / (query_tiles * kv_heads);
        if (batch >= batch_size) return;

        const int query_begin = query_tile * kAmpereTile;
        const int query_head = kv_head * kAmpereGqaRatio + warp;
        constexpr int kQElements = kAmpereTile * kAmpereSmemLd;
        constexpr int kKVElements = kAmpereDqKeyTile * kAmpereSmemLd;

        extern __shared__ __align__(16) unsigned char shared_raw[];
        auto *storage = reinterpret_cast<__half *>(shared_raw);

        load_q_or_do_group_async<4>(
            storage, query, batch, query_begin, query_sequence,
            query_heads, kv_head * kAmpereGqaRatio);
        cp_async_wait_all();
        __syncthreads();
        unsigned query_fragments[8][4];
#pragma unroll
        for (int dimension_tile = 0; dimension_tile < 8; ++dimension_tile) {
            ldmatrix_a_m16k16_row(
                query_fragments[dimension_tile],
                storage + warp * kQElements + dimension_tile * 16,
                kAmpereSmemLd, lane);
        }
        __syncthreads();

        load_q_or_do_group_async<4>(
            storage, grad_output, batch, query_begin, query_sequence,
            query_heads, kv_head * kAmpereGqaRatio);
        cp_async_wait_all();
        __syncthreads();
        unsigned grad_output_fragments[8][4];
#pragma unroll
        for (int dimension_tile = 0; dimension_tile < 8; ++dimension_tile) {
            ldmatrix_a_m16k16_row(
                grad_output_fragments[dimension_tile],
                storage + warp * kQElements + dimension_tile * 16,
                kAmpereSmemLd, lane);
        }
        __syncthreads();

        __half *key_stage[2] = {storage, storage + 2 * kKVElements};
        __half *value_stage[2] = {storage + kKVElements, storage + 3 * kKVElements};

        const int upper_query_position = query_begin + lane_group;
        const int lower_query_position = upper_query_position + 8;
        float running_max_upper = -CUDART_INF_F;
        float running_max_lower = -CUDART_INF_F;
        float running_sum_upper = 0.0f;
        float running_sum_lower = 0.0f;
        float running_delta_upper = 0.0f;
        float running_delta_lower = 0.0f;

        const int max_visible_key = Causal
            ? min(key_sequence, query_position_offset + query_begin + kAmpereTile)
            : key_sequence;
        if (max_visible_key > 0) {
            load_kv_rows_async<kAmpereDqKeyTile>(
                key_stage[0], value_stage[0], key, value,
                batch, 0, key_sequence, kv_heads, kv_head);
        }

        for (int key_begin = 0, tile_index = 0;
             key_begin < max_visible_key;
             key_begin += kAmpereDqKeyTile, ++tile_index) {
            const int stage = tile_index & 1;
            cp_async_wait_all();
            __syncthreads();
            const int next_key_begin = key_begin + kAmpereDqKeyTile;
            if (next_key_begin < max_visible_key) {
                load_kv_rows_async<kAmpereDqKeyTile>(
                    key_stage[stage ^ 1], value_stage[stage ^ 1], key, value,
                    batch, next_key_begin, key_sequence, kv_heads, kv_head);
            }

            Mma16816Accumulator score[4];
            Mma16816Accumulator d_probability[4];
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                clear_mma(score[i]);
                clear_mma(d_probability[i]);
            }
#pragma unroll
            for (int dimension_tile = 0; dimension_tile < 8; ++dimension_tile) {
                const int dimension = dimension_tile * 16;
                unsigned key_b[4][2];
                unsigned value_b[4][2];
#pragma unroll
                for (int part = 0; part < 4; ++part) {
                    ldmatrix_b_k16n8_col(
                        key_b[part], key_stage[stage]
                            + part * 8 * kAmpereSmemLd + dimension,
                        kAmpereSmemLd, lane);
                    ldmatrix_b_k16n8_col(
                        value_b[part], value_stage[stage]
                            + part * 8 * kAmpereSmemLd + dimension,
                        kAmpereSmemLd, lane);
                    mma_m16n8k16(
                        score[part], query_fragments[dimension_tile], key_b[part]);
                    mma_m16n8k16(
                        d_probability[part], grad_output_fragments[dimension_tile],
                        value_b[part]);
                }
            }

            const int column_pair = lane_in_group * 2;
            float score_upper[8];
            float score_lower[8];
            float dp_upper[8];
            float dp_lower[8];
#pragma unroll
            for (int part = 0; part < 4; ++part) {
                const int column = column_pair + part * 8;
                const int key_position0 = key_begin + column;
                const int key_position1 = key_position0 + 1;
                const bool valid_upper0 = upper_query_position < query_sequence
                    && key_position0 < key_sequence
                    && (!Causal || key_position0 <= query_position_offset + upper_query_position);
                const bool valid_upper1 = upper_query_position < query_sequence
                    && key_position1 < key_sequence
                    && (!Causal || key_position1 <= query_position_offset + upper_query_position);
                const bool valid_lower0 = lower_query_position < query_sequence
                    && key_position0 < key_sequence
                    && (!Causal || key_position0 <= query_position_offset + lower_query_position);
                const bool valid_lower1 = lower_query_position < query_sequence
                    && key_position1 < key_sequence
                    && (!Causal || key_position1 <= query_position_offset + lower_query_position);
                score_upper[part * 2] = valid_upper0
                    ? score[part].x[0] * scale_log2 : -CUDART_INF_F;
                score_upper[part * 2 + 1] = valid_upper1
                    ? score[part].x[1] * scale_log2 : -CUDART_INF_F;
                score_lower[part * 2] = valid_lower0
                    ? score[part].x[2] * scale_log2 : -CUDART_INF_F;
                score_lower[part * 2 + 1] = valid_lower1
                    ? score[part].x[3] * scale_log2 : -CUDART_INF_F;
                dp_upper[part * 2] = d_probability[part].x[0];
                dp_upper[part * 2 + 1] = d_probability[part].x[1];
                dp_lower[part * 2] = d_probability[part].x[2];
                dp_lower[part * 2 + 1] = d_probability[part].x[3];
            }

            float local_max_upper = -CUDART_INF_F;
            float local_max_lower = -CUDART_INF_F;
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                local_max_upper = fmaxf(local_max_upper, score_upper[i]);
                local_max_lower = fmaxf(local_max_lower, score_lower[i]);
            }
            const float new_max_upper = fmaxf(
                running_max_upper, subgroup4_max(local_max_upper));
            const float new_max_lower = fmaxf(
                running_max_lower, subgroup4_max(local_max_lower));
            const float alpha_upper = running_max_upper == -CUDART_INF_F
                ? 0.0f : exp2f(running_max_upper - new_max_upper);
            const float alpha_lower = running_max_lower == -CUDART_INF_F
                ? 0.0f : exp2f(running_max_lower - new_max_lower);
            float local_sum_upper = 0.0f;
            float local_sum_lower = 0.0f;
            float local_delta_upper = 0.0f;
            float local_delta_lower = 0.0f;
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                if (score_upper[i] != -CUDART_INF_F) {
                    const float p = exp2f(score_upper[i] - new_max_upper);
                    local_sum_upper += p;
                    local_delta_upper = fmaf(p, dp_upper[i], local_delta_upper);
                }
                if (score_lower[i] != -CUDART_INF_F) {
                    const float p = exp2f(score_lower[i] - new_max_lower);
                    local_sum_lower += p;
                    local_delta_lower = fmaf(p, dp_lower[i], local_delta_lower);
                }
            }
            running_sum_upper = running_sum_upper * alpha_upper
                + subgroup4_sum(local_sum_upper);
            running_sum_lower = running_sum_lower * alpha_lower
                + subgroup4_sum(local_sum_lower);
            running_delta_upper = running_delta_upper * alpha_upper
                + subgroup4_sum(local_delta_upper);
            running_delta_lower = running_delta_lower * alpha_lower
                + subgroup4_sum(local_delta_lower);
            running_max_upper = new_max_upper;
            running_max_lower = new_max_lower;
            __syncthreads();
        }

        if (lane_in_group == 0) {
            if (upper_query_position < query_sequence) {
                const std::size_t row =
                    (static_cast<std::size_t>(batch) * query_sequence
                     + upper_query_position) * query_heads + query_head;
                logsumexp[row] = running_sum_upper > 0.0f
                    ? running_max_upper / kLog2E + __logf(running_sum_upper)
                    : -CUDART_INF_F;
                delta[row] = running_sum_upper > 0.0f
                    ? running_delta_upper / running_sum_upper : 0.0f;
            }
            if (lower_query_position < query_sequence) {
                const std::size_t row =
                    (static_cast<std::size_t>(batch) * query_sequence
                     + lower_query_position) * query_heads + query_head;
                logsumexp[row] = running_sum_lower > 0.0f
                    ? running_max_lower / kLog2E + __logf(running_sum_lower)
                    : -CUDART_INF_F;
                delta[row] = running_sum_lower > 0.0f
                    ? running_delta_lower / running_sum_lower : 0.0f;
            }
        }
    }

    /** Computes D_i = sum_d dO_i,d * O_i,d for all query rows. */
    __global__ __launch_bounds__(128, 4)
    void attention_backward_delta_f16_d128(
        float * __restrict__ delta,
        const __half * __restrict__ grad_output,
        const __half * __restrict__ output,
        std::size_t row_count
    ) {
        const int lane = static_cast<int>(threadIdx.x) & 31;
        const int warp = static_cast<int>(threadIdx.x) >> 5;
        const std::size_t row = static_cast<std::size_t>(blockIdx.x) * 4U + warp;
        if (row >= row_count) return;
        const std::size_t base = row * kAmpereHeadDim;
        float sum = 0.0f;
#pragma unroll
        for (int d = lane; d < kAmpereHeadDim; d += 32) {
            sum = fmaf(__half2float(grad_output[base + d]),
                       __half2float(output[base + d]), sum);
        }
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            sum += __shfl_down_sync(0xffffffffU, sum, offset);
        }
        if (lane == 0) delta[row] = sum;
    }


    template<bool Causal, bool Accumulate>
    __global__ __launch_bounds__(128, 2)
    void attention_backward_lse_dq_ampere_f16_d128_gqa4(
        __half * __restrict__ grad_query,
        const __half * __restrict__ grad_output,
        const float * __restrict__ delta,
        const float * __restrict__ logsumexp,
        const __half * __restrict__ query,
        const __half * __restrict__ key,
        const __half * __restrict__ value,
        int batch_size, int query_sequence, int key_sequence,
        int query_heads, int kv_heads,
        float scale_log2, int query_position_offset
    ) {
        const int warp = static_cast<int>(threadIdx.x) >> 5;
        const int lane = static_cast<int>(threadIdx.x) & 31;
        const int query_tiles = (query_sequence + kAmpereTile - 1) / kAmpereTile;
        const int linear = static_cast<int>(blockIdx.x);
        const int query_tile = linear % query_tiles;
        const int kv_head = (linear / query_tiles) % kv_heads;
        const int batch = linear / (query_tiles * kv_heads);
        if (batch >= batch_size) return;

        const int query_begin = query_tile * kAmpereTile;
        const int query_head = kv_head * kAmpereGqaRatio + warp;
        constexpr int kQElements = kAmpereTile * kAmpereSmemLd;
        constexpr int kKVElements = kAmpereDqKeyTile * kAmpereSmemLd;
        constexpr int kDscoreElements = kAmpereTile * kAmpereDqKeyTile;

        extern __shared__ __align__(16) unsigned char shared_raw[];
        auto *storage = reinterpret_cast<__half *>(shared_raw);
        __half *dscore_shared = storage + 4 * kKVElements;
        __half *warp_dscore = dscore_shared + warp * kDscoreElements;

        load_q_or_do_group_async<4>(
            storage, query, batch, query_begin, query_sequence,
            query_heads, kv_head * kAmpereGqaRatio);
        cp_async_wait_all();
        __syncthreads();
        unsigned query_fragments[8][4];
#pragma unroll
        for (int dimension_tile = 0; dimension_tile < 8; ++dimension_tile) {
            ldmatrix_a_m16k16_row(
                query_fragments[dimension_tile],
                storage + warp * kQElements + dimension_tile * 16,
                kAmpereSmemLd, lane);
        }
        __syncthreads();

        load_q_or_do_group_async<4>(
            storage, grad_output, batch, query_begin, query_sequence,
            query_heads, kv_head * kAmpereGqaRatio);
        cp_async_wait_all();
        __syncthreads();
        unsigned grad_output_fragments[8][4];
#pragma unroll
        for (int dimension_tile = 0; dimension_tile < 8; ++dimension_tile) {
            ldmatrix_a_m16k16_row(
                grad_output_fragments[dimension_tile],
                storage + warp * kQElements + dimension_tile * 16,
                kAmpereSmemLd, lane);
        }
        __syncthreads();

        Mma16816Accumulator grad_query_fragments[16];
#pragma unroll
        for (int i = 0; i < 16; ++i) clear_mma(grad_query_fragments[i]);

        __half *key_stage[2] = {storage, storage + 2 * kKVElements};
        __half *value_stage[2] = {storage + kKVElements, storage + 3 * kKVElements};

        const int max_visible_key = Causal
            ? min(key_sequence, query_position_offset + query_begin + kAmpereTile)
            : key_sequence;
        if (max_visible_key > 0) {
            load_kv_rows_async<kAmpereDqKeyTile>(
                key_stage[0], value_stage[0], key, value,
                batch, 0, key_sequence, kv_heads, kv_head);
        }

        for (int key_begin = 0, tile_index = 0;
             key_begin < max_visible_key;
             key_begin += kAmpereDqKeyTile, ++tile_index) {
            const int stage = tile_index & 1;
            cp_async_wait_all();
            __syncthreads();
            const int next_key_begin = key_begin + kAmpereDqKeyTile;
            if (next_key_begin < max_visible_key) {
                load_kv_rows_async<kAmpereDqKeyTile>(
                    key_stage[stage ^ 1], value_stage[stage ^ 1], key, value,
                    batch, next_key_begin, key_sequence, kv_heads, kv_head);
            }

            Mma16816Accumulator score[4];
            Mma16816Accumulator d_probability[4];
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                clear_mma(score[i]);
                clear_mma(d_probability[i]);
            }
#pragma unroll
            for (int dimension_tile = 0; dimension_tile < 8; ++dimension_tile) {
                const int dimension = dimension_tile * 16;
                unsigned key_b[4][2];
                unsigned value_b[4][2];
#pragma unroll
                for (int part = 0; part < 4; ++part) {
                    ldmatrix_b_k16n8_col(
                        key_b[part], key_stage[stage]
                            + part * 8 * kAmpereSmemLd + dimension,
                        kAmpereSmemLd, lane);
                    ldmatrix_b_k16n8_col(
                        value_b[part], value_stage[stage]
                            + part * 8 * kAmpereSmemLd + dimension,
                        kAmpereSmemLd, lane);
                    mma_m16n8k16(
                        score[part], query_fragments[dimension_tile], key_b[part]);
                    mma_m16n8k16(
                        d_probability[part], grad_output_fragments[dimension_tile],
                        value_b[part]);
                }
            }

            const int row0 = lane >> 2;
            const int row1 = row0 + 8;
            const int column_pair = (lane & 3) * 2;
            const int query_position0 = query_begin + row0;
            const int query_position1 = query_begin + row1;
            const std::size_t row_index0 =
                (static_cast<std::size_t>(batch) * query_sequence + query_position0)
                * query_heads + query_head;
            const std::size_t row_index1 =
                (static_cast<std::size_t>(batch) * query_sequence + query_position1)
                * query_heads + query_head;
            const float lse0 = query_position0 < query_sequence
                ? logsumexp[row_index0] * kLog2E : 0.0f;
            const float lse1 = query_position1 < query_sequence
                ? logsumexp[row_index1] * kLog2E : 0.0f;
            const float delta0 = query_position0 < query_sequence
                ? delta[row_index0] : 0.0f;
            const float delta1 = query_position1 < query_sequence
                ? delta[row_index1] : 0.0f;
            const float scale = scale_log2 / kLog2E;

#pragma unroll
            for (int part = 0; part < 4; ++part) {
                const int column = column_pair + part * 8;
                const int key_position0 = key_begin + column;
                const int key_position1 = key_position0 + 1;
                const bool valid00 = query_position0 < query_sequence
                    && key_position0 < key_sequence
                    && (!Causal || key_position0 <= query_position_offset + query_position0);
                const bool valid01 = query_position0 < query_sequence
                    && key_position1 < key_sequence
                    && (!Causal || key_position1 <= query_position_offset + query_position0);
                const bool valid10 = query_position1 < query_sequence
                    && key_position0 < key_sequence
                    && (!Causal || key_position0 <= query_position_offset + query_position1);
                const bool valid11 = query_position1 < query_sequence
                    && key_position1 < key_sequence
                    && (!Causal || key_position1 <= query_position_offset + query_position1);
                const float p00 = valid00
                    ? exp2f(score[part].x[0] * scale_log2 - lse0) : 0.0f;
                const float p01 = valid01
                    ? exp2f(score[part].x[1] * scale_log2 - lse0) : 0.0f;
                const float p10 = valid10
                    ? exp2f(score[part].x[2] * scale_log2 - lse1) : 0.0f;
                const float p11 = valid11
                    ? exp2f(score[part].x[3] * scale_log2 - lse1) : 0.0f;
                warp_dscore[row0 * kAmpereDqKeyTile + column] =
                    __float2half_rn(p00 * (d_probability[part].x[0] - delta0) * scale);
                warp_dscore[row0 * kAmpereDqKeyTile + column + 1] =
                    __float2half_rn(p01 * (d_probability[part].x[1] - delta0) * scale);
                warp_dscore[row1 * kAmpereDqKeyTile + column] =
                    __float2half_rn(p10 * (d_probability[part].x[2] - delta1) * scale);
                warp_dscore[row1 * kAmpereDqKeyTile + column + 1] =
                    __float2half_rn(p11 * (d_probability[part].x[3] - delta1) * scale);
            }
            __syncwarp();

#pragma unroll
            for (int key_subtile = 0; key_subtile < kAmpereDqKeyTile;
                 key_subtile += 16) {
                unsigned dscore_a[4];
                ldmatrix_a_m16k16_row(
                    dscore_a, warp_dscore + key_subtile,
                    kAmpereDqKeyTile, lane);
#pragma unroll
                for (int output_tile = 0; output_tile < 16; ++output_tile) {
                    unsigned key_b[2];
                    ldmatrix_b_k16n8_row_transposed(
                        key_b,
                        key_stage[stage] + key_subtile * kAmpereSmemLd
                            + output_tile * 8,
                        kAmpereSmemLd, lane);
                    mma_m16n8k16(
                        grad_query_fragments[output_tile], dscore_a, key_b);
                }
            }
            __syncthreads();
        }

        const int row0 = lane >> 2;
        const int row1 = row0 + 8;
        const int output_pair = (lane & 3) * 2;
        const int query_position0 = query_begin + row0;
        const int query_position1 = query_begin + row1;
#pragma unroll
        for (int output_tile = 0; output_tile < 16; ++output_tile) {
            const int column = output_tile * 8 + output_pair;
            if (query_position0 < query_sequence) {
                const std::size_t base =
                    ((static_cast<std::size_t>(batch) * query_sequence + query_position0)
                     * query_heads + query_head) * kAmpereHeadDim + column;
                float x0 = grad_query_fragments[output_tile].x[0];
                float x1 = grad_query_fragments[output_tile].x[1];
                if constexpr (Accumulate) {
                    x0 += __half2float(grad_query[base]);
                    x1 += __half2float(grad_query[base + 1]);
                }
                grad_query[base] = __float2half_rn(x0);
                grad_query[base + 1] = __float2half_rn(x1);
            }
            if (query_position1 < query_sequence) {
                const std::size_t base =
                    ((static_cast<std::size_t>(batch) * query_sequence + query_position1)
                     * query_heads + query_head) * kAmpereHeadDim + column;
                float x0 = grad_query_fragments[output_tile].x[2];
                float x1 = grad_query_fragments[output_tile].x[3];
                if constexpr (Accumulate) {
                    x0 += __half2float(grad_query[base]);
                    x1 += __half2float(grad_query[base + 1]);
                }
                grad_query[base] = __float2half_rn(x0);
                grad_query[base + 1] = __float2half_rn(x1);
            }
        }
    }


    template<bool Causal, bool Accumulate>
    __global__ __launch_bounds__(128, 2)
    void attention_backward_lse_dkv_ampere_f16_d128_gqa4(
        __half * __restrict__ grad_key,
        __half * __restrict__ grad_value,
        const __half * __restrict__ grad_output,
        const float * __restrict__ delta,
        const float * __restrict__ logsumexp,
        const __half * __restrict__ query,
        const __half * __restrict__ key,
        const __half * __restrict__ value,
        int batch_size, int query_sequence, int key_sequence,
        int query_heads, int kv_heads,
        float scale_log2, int query_position_offset
    ) {
        const int warp = static_cast<int>(threadIdx.x) >> 5;
        const int lane = static_cast<int>(threadIdx.x) & 31;
        const int key_tiles = (key_sequence + kAmpereTile - 1) / kAmpereTile;
        const int linear = static_cast<int>(blockIdx.x);
        const int key_tile = linear % key_tiles;
        const int kv_head = (linear / key_tiles) % kv_heads;
        const int batch = linear / (key_tiles * kv_heads);
        if (batch >= batch_size) return;
        const int key_begin = key_tile * kAmpereTile;

        constexpr int kKVElements = kAmpereTile * kAmpereSmemLd;
        constexpr int kQElements = kAmpereDkvQueryTile * kAmpereSmemLd;
        constexpr int kProbabilityElements = kAmpereTile * kAmpereDkvQueryTile;

        extern __shared__ __align__(16) unsigned char shared_raw[];
        auto *key_shared = reinterpret_cast<__half *>(shared_raw);
        __half *value_shared = key_shared + kKVElements;
        __half *query_shared = value_shared + kKVElements;
        __half *grad_output_shared = query_shared + kQElements;
        __half *probability_transposed = grad_output_shared + kQElements;
        __half *dscore_transposed = probability_transposed + kProbabilityElements;

        load_kv_rows_async<kAmpereTile>(
            key_shared, value_shared, key, value,
            batch, key_begin, key_sequence, kv_heads, kv_head);
        cp_async_wait_all();
        __syncthreads();

        // Each warp owns 32 output columns.  This removes the four-way FP32
        // reduction and cuts the live dK+dV accumulator set from 128 floats per
        // thread to 32 floats per thread.
        Mma16816Accumulator grad_key_fragments[4];
        Mma16816Accumulator grad_value_fragments[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            clear_mma(grad_key_fragments[i]);
            clear_mma(grad_value_fragments[i]);
        }

        int first_query = 0;
        if constexpr (Causal) {
            first_query = max(0, key_begin - query_position_offset);
            first_query = (first_query / kAmpereDkvQueryTile)
                * kAmpereDkvQueryTile;
        }
        const float scale = scale_log2 / kLog2E;

        for (int local_head = 0; local_head < kAmpereGqaRatio; ++local_head) {
            const int query_head = kv_head * kAmpereGqaRatio + local_head;
            for (int query_begin = first_query;
                 query_begin < query_sequence;
                 query_begin += kAmpereDkvQueryTile) {
                load_q_or_do_rows_async<1, kAmpereDkvQueryTile>(
                    query_shared, query, batch, query_begin, query_sequence,
                    query_heads, query_head);
                load_q_or_do_rows_async<1, kAmpereDkvQueryTile>(
                    grad_output_shared, grad_output, batch, query_begin,
                    query_sequence, query_heads, query_head);
                cp_async_wait_all();
                __syncthreads();

                // Two warps generate the 32x16 P and dS tiles.  The remaining
                // warps immediately consume them in the following phase.
                if (warp < 2) {
                    const int query_row_begin = warp * 16;
                    Mma16816Accumulator score[2];
                    Mma16816Accumulator d_probability[2];
                    clear_mma(score[0]); clear_mma(score[1]);
                    clear_mma(d_probability[0]); clear_mma(d_probability[1]);
#pragma unroll
                    for (int dimension_tile = 0; dimension_tile < 8; ++dimension_tile) {
                        const int dimension = dimension_tile * 16;
                        unsigned query_a[4], grad_output_a[4];
                        unsigned key_b0[2], key_b1[2], value_b0[2], value_b1[2];
                        ldmatrix_a_m16k16_row(
                            query_a,
                            query_shared + query_row_begin * kAmpereSmemLd + dimension,
                            kAmpereSmemLd, lane);
                        ldmatrix_a_m16k16_row(
                            grad_output_a,
                            grad_output_shared + query_row_begin * kAmpereSmemLd
                                + dimension,
                            kAmpereSmemLd, lane);
                        ldmatrix_b_k16n8_col(
                            key_b0, key_shared + dimension, kAmpereSmemLd, lane);
                        ldmatrix_b_k16n8_col(
                            key_b1, key_shared + 8 * kAmpereSmemLd + dimension,
                            kAmpereSmemLd, lane);
                        ldmatrix_b_k16n8_col(
                            value_b0, value_shared + dimension, kAmpereSmemLd, lane);
                        ldmatrix_b_k16n8_col(
                            value_b1, value_shared + 8 * kAmpereSmemLd + dimension,
                            kAmpereSmemLd, lane);
                        mma_m16n8k16(score[0], query_a, key_b0);
                        mma_m16n8k16(score[1], query_a, key_b1);
                        mma_m16n8k16(d_probability[0], grad_output_a, value_b0);
                        mma_m16n8k16(d_probability[1], grad_output_a, value_b1);
                    }

                    const int row0 = lane >> 2;
                    const int row1 = row0 + 8;
                    const int query_local0 = query_row_begin + row0;
                    const int query_local1 = query_row_begin + row1;
                    const int query_position0 = query_begin + query_local0;
                    const int query_position1 = query_begin + query_local1;
                    const int column_pair = (lane & 3) * 2;
                    const std::size_t row_index0 =
                        (static_cast<std::size_t>(batch) * query_sequence
                         + query_position0) * query_heads + query_head;
                    const std::size_t row_index1 =
                        (static_cast<std::size_t>(batch) * query_sequence
                         + query_position1) * query_heads + query_head;
                    const float lse0 = query_position0 < query_sequence
                        ? logsumexp[row_index0] * kLog2E : 0.0f;
                    const float lse1 = query_position1 < query_sequence
                        ? logsumexp[row_index1] * kLog2E : 0.0f;
                    const float delta0 = query_position0 < query_sequence
                        ? delta[row_index0] : 0.0f;
                    const float delta1 = query_position1 < query_sequence
                        ? delta[row_index1] : 0.0f;
#pragma unroll
                    for (int part = 0; part < 2; ++part) {
                        const int key_local = column_pair + part * 8;
                        const int key_position0 = key_begin + key_local;
                        const int key_position1 = key_position0 + 1;
                        const bool valid00 = query_position0 < query_sequence
                            && key_position0 < key_sequence
                            && (!Causal || key_position0 <= query_position_offset + query_position0);
                        const bool valid01 = query_position0 < query_sequence
                            && key_position1 < key_sequence
                            && (!Causal || key_position1 <= query_position_offset + query_position0);
                        const bool valid10 = query_position1 < query_sequence
                            && key_position0 < key_sequence
                            && (!Causal || key_position0 <= query_position_offset + query_position1);
                        const bool valid11 = query_position1 < query_sequence
                            && key_position1 < key_sequence
                            && (!Causal || key_position1 <= query_position_offset + query_position1);
                        const float p00 = valid00
                            ? exp2f(score[part].x[0] * scale_log2 - lse0) : 0.0f;
                        const float p01 = valid01
                            ? exp2f(score[part].x[1] * scale_log2 - lse0) : 0.0f;
                        const float p10 = valid10
                            ? exp2f(score[part].x[2] * scale_log2 - lse1) : 0.0f;
                        const float p11 = valid11
                            ? exp2f(score[part].x[3] * scale_log2 - lse1) : 0.0f;
                        probability_transposed[key_local * kAmpereDkvQueryTile
                            + query_local0] = __float2half_rn(p00);
                        probability_transposed[(key_local + 1) * kAmpereDkvQueryTile
                            + query_local0] = __float2half_rn(p01);
                        probability_transposed[key_local * kAmpereDkvQueryTile
                            + query_local1] = __float2half_rn(p10);
                        probability_transposed[(key_local + 1) * kAmpereDkvQueryTile
                            + query_local1] = __float2half_rn(p11);
                        dscore_transposed[key_local * kAmpereDkvQueryTile
                            + query_local0] = __float2half_rn(
                                p00 * (d_probability[part].x[0] - delta0) * scale);
                        dscore_transposed[(key_local + 1) * kAmpereDkvQueryTile
                            + query_local0] = __float2half_rn(
                                p01 * (d_probability[part].x[1] - delta0) * scale);
                        dscore_transposed[key_local * kAmpereDkvQueryTile
                            + query_local1] = __float2half_rn(
                                p10 * (d_probability[part].x[2] - delta1) * scale);
                        dscore_transposed[(key_local + 1) * kAmpereDkvQueryTile
                            + query_local1] = __float2half_rn(
                                p11 * (d_probability[part].x[3] - delta1) * scale);
                    }
                }
                __syncthreads();

                const int first_output_tile = warp * 4;
#pragma unroll
                for (int query_subtile = 0;
                     query_subtile < kAmpereDkvQueryTile;
                     query_subtile += 16) {
                    unsigned dscore_a[4], probability_a[4];
                    ldmatrix_a_m16k16_row(
                        dscore_a, dscore_transposed + query_subtile,
                        kAmpereDkvQueryTile, lane);
                    ldmatrix_a_m16k16_row(
                        probability_a, probability_transposed + query_subtile,
                        kAmpereDkvQueryTile, lane);
#pragma unroll
                    for (int local_tile = 0; local_tile < 4; ++local_tile) {
                        const int output_tile = first_output_tile + local_tile;
                        unsigned query_b[2], grad_output_b[2];
                        ldmatrix_b_k16n8_row_transposed(
                            query_b,
                            query_shared + query_subtile * kAmpereSmemLd
                                + output_tile * 8,
                            kAmpereSmemLd, lane);
                        ldmatrix_b_k16n8_row_transposed(
                            grad_output_b,
                            grad_output_shared + query_subtile * kAmpereSmemLd
                                + output_tile * 8,
                            kAmpereSmemLd, lane);
                        mma_m16n8k16(
                            grad_key_fragments[local_tile], dscore_a, query_b);
                        mma_m16n8k16(
                            grad_value_fragments[local_tile], probability_a,
                            grad_output_b);
                    }
                }
                __syncthreads();
            }
        }

        const int row0 = lane >> 2;
        const int row1 = row0 + 8;
        const int output_pair = (lane & 3) * 2;
        const int first_output_tile = warp * 4;
#pragma unroll
        for (int local_tile = 0; local_tile < 4; ++local_tile) {
            const int output_tile = first_output_tile + local_tile;
            const int column = output_tile * 8 + output_pair;
            const int key_position0 = key_begin + row0;
            const int key_position1 = key_begin + row1;
            if (key_position0 < key_sequence) {
                const std::size_t base =
                    ((static_cast<std::size_t>(batch) * key_sequence + key_position0)
                     * kv_heads + kv_head) * kAmpereHeadDim + column;
                float dk0 = grad_key_fragments[local_tile].x[0];
                float dk1 = grad_key_fragments[local_tile].x[1];
                float dv0 = grad_value_fragments[local_tile].x[0];
                float dv1 = grad_value_fragments[local_tile].x[1];
                if constexpr (Accumulate) {
                    dk0 += __half2float(grad_key[base]);
                    dk1 += __half2float(grad_key[base + 1]);
                    dv0 += __half2float(grad_value[base]);
                    dv1 += __half2float(grad_value[base + 1]);
                }
                grad_key[base] = __float2half_rn(dk0);
                grad_key[base + 1] = __float2half_rn(dk1);
                grad_value[base] = __float2half_rn(dv0);
                grad_value[base + 1] = __float2half_rn(dv1);
            }
            if (key_position1 < key_sequence) {
                const std::size_t base =
                    ((static_cast<std::size_t>(batch) * key_sequence + key_position1)
                     * kv_heads + kv_head) * kAmpereHeadDim + column;
                float dk0 = grad_key_fragments[local_tile].x[2];
                float dk1 = grad_key_fragments[local_tile].x[3];
                float dv0 = grad_value_fragments[local_tile].x[2];
                float dv1 = grad_value_fragments[local_tile].x[3];
                if constexpr (Accumulate) {
                    dk0 += __half2float(grad_key[base]);
                    dk1 += __half2float(grad_key[base + 1]);
                    dv0 += __half2float(grad_value[base]);
                    dv1 += __half2float(grad_value[base + 1]);
                }
                grad_key[base] = __float2half_rn(dk0);
                grad_key[base + 1] = __float2half_rn(dk1);
                grad_value[base] = __float2half_rn(dv0);
                grad_value[base + 1] = __float2half_rn(dv1);
            }
        }
    }

    bool current_device_supports_ampere_mma() {
        int device = 0;
        CUDA_CHECK(cudaGetDevice(&device));
        cudaDeviceProp properties{};
        CUDA_CHECK(cudaGetDeviceProperties(&properties, device));
        return properties.major >= 8;
    }

    template<bool Causal, bool Accumulate>
    void launch_attention_backward_ampere_f16_d128_gqa4_from_stats(
        __half *grad_query, __half *grad_key, __half *grad_value,
        const __half *grad_output,
        const float *delta, const float *logsumexp,
        const __half *query, const __half *key, const __half *value,
        int batch, int query_sequence, int key_sequence,
        int query_heads, int kv_heads, float scale,
        int query_position_offset, cudaStream_t stream
    ) {
        constexpr std::size_t kDqSharedBytes =
            (4U * kAmpereDqKeyTile * kAmpereSmemLd
             + 4U * kAmpereTile * kAmpereDqKeyTile) * sizeof(__half);
        const unsigned int dq_blocks = static_cast<unsigned int>(
            static_cast<std::size_t>(batch) * kv_heads
            * ((query_sequence + kAmpereTile - 1) / kAmpereTile));
        attention_backward_lse_dq_ampere_f16_d128_gqa4<Causal, Accumulate>
            <<<dq_blocks, 128, kDqSharedBytes, stream>>>(
                grad_query, grad_output, delta, logsumexp,
                query, key, value, batch, query_sequence, key_sequence,
                query_heads, kv_heads, scale * kLog2E, query_position_offset);

        constexpr std::size_t kDkvSharedBytes =
            (2U * kAmpereTile * kAmpereSmemLd
             + 2U * kAmpereDkvQueryTile * kAmpereSmemLd
             + 2U * kAmpereTile * kAmpereDkvQueryTile) * sizeof(__half);
        static_assert(kDkvSharedBytes <= 32U * 1024U);
        const unsigned int dkv_blocks = static_cast<unsigned int>(
            static_cast<std::size_t>(batch) * kv_heads
            * ((key_sequence + kAmpereTile - 1) / kAmpereTile));
        attention_backward_lse_dkv_ampere_f16_d128_gqa4<Causal, Accumulate>
            <<<dkv_blocks, 128, kDkvSharedBytes, stream>>>(
                grad_key, grad_value, grad_output, delta, logsumexp,
                query, key, value, batch, query_sequence, key_sequence,
                query_heads, kv_heads, scale * kLog2E, query_position_offset);
    }

    template<bool Causal, bool Accumulate>
    void launch_attention_backward_ampere_f16_d128_gqa4_no_lse_impl(
        __half *grad_query, __half *grad_key, __half *grad_value,
        const __half *grad_output, const __half *query,
        const __half *key, const __half *value,
        int batch, int query_sequence, int key_sequence,
        int query_heads, int kv_heads, float scale,
        int query_position_offset, cudaStream_t stream
    ) {
        const std::size_t row_count = static_cast<std::size_t>(batch)
            * query_sequence * query_heads;
        float *workspace = nullptr;
        CUDA_CHECK(cudaMallocAsync(
            reinterpret_cast<void **>(&workspace),
            2U * row_count * sizeof(float), stream));
        float *logsumexp = workspace;
        float *delta = workspace + row_count;

        constexpr std::size_t kStatsSharedBytes =
            4U * kAmpereDqKeyTile * kAmpereSmemLd * sizeof(__half);
        const unsigned int stats_blocks = static_cast<unsigned int>(
            static_cast<std::size_t>(batch) * kv_heads
            * ((query_sequence + kAmpereTile - 1) / kAmpereTile));
        attention_backward_stats_ampere_f16_d128_gqa4<Causal>
            <<<stats_blocks, 128, kStatsSharedBytes, stream>>>(
                logsumexp, delta, grad_output, query, key, value,
                batch, query_sequence, key_sequence, query_heads, kv_heads,
                scale * kLog2E, query_position_offset);

        launch_attention_backward_ampere_f16_d128_gqa4_from_stats<Causal, Accumulate>(
            grad_query, grad_key, grad_value, grad_output, delta, logsumexp,
            query, key, value, batch, query_sequence, key_sequence,
            query_heads, kv_heads, scale, query_position_offset, stream);

        const cudaError_t launch_error = cudaGetLastError();
        const cudaError_t free_error = cudaFreeAsync(workspace, stream);
        CUDA_CHECK(launch_error);
        CUDA_CHECK(free_error);
    }

    void launch_attention_backward_ampere_f16_d128_gqa4_no_lse(
        __half *grad_query, __half *grad_key, __half *grad_value,
        const __half *grad_output, const __half *query,
        const __half *key, const __half *value,
        int batch, int query_sequence, int key_sequence,
        int query_heads, int kv_heads, float scale,
        bool causal, int query_position_offset, bool accumulate,
        cudaStream_t stream
    ) {
        if (causal) {
            if (accumulate) {
                launch_attention_backward_ampere_f16_d128_gqa4_no_lse_impl<true, true>(
                    grad_query, grad_key, grad_value, grad_output, query, key, value,
                    batch, query_sequence, key_sequence, query_heads, kv_heads,
                    scale, query_position_offset, stream);
            } else {
                launch_attention_backward_ampere_f16_d128_gqa4_no_lse_impl<true, false>(
                    grad_query, grad_key, grad_value, grad_output, query, key, value,
                    batch, query_sequence, key_sequence, query_heads, kv_heads,
                    scale, query_position_offset, stream);
            }
        } else if (accumulate) {
            launch_attention_backward_ampere_f16_d128_gqa4_no_lse_impl<false, true>(
                grad_query, grad_key, grad_value, grad_output, query, key, value,
                batch, query_sequence, key_sequence, query_heads, kv_heads,
                scale, query_position_offset, stream);
        } else {
            launch_attention_backward_ampere_f16_d128_gqa4_no_lse_impl<false, false>(
                grad_query, grad_key, grad_value, grad_output, query, key, value,
                batch, query_sequence, key_sequence, query_heads, kv_heads,
                scale, query_position_offset, stream);
        }
    }

    template<bool Causal, bool Accumulate>
    void launch_attention_backward_lse_ampere_f16_d128_gqa4_impl(
        __half *grad_query, __half *grad_key, __half *grad_value,
        const __half *grad_output, const __half *output,
        const float *logsumexp, const __half *query,
        const __half *key, const __half *value,
        int batch, int query_sequence, int key_sequence,
        int query_heads, int kv_heads, float scale,
        int query_position_offset, cudaStream_t stream
    ) {
        const std::size_t row_count = static_cast<std::size_t>(batch)
            * query_sequence * query_heads;
        float *delta = nullptr;
        CUDA_CHECK(cudaMallocAsync(
            reinterpret_cast<void **>(&delta), row_count * sizeof(float), stream));
        const unsigned int delta_blocks = static_cast<unsigned int>((row_count + 3U) / 4U);
        attention_backward_delta_f16_d128<<<delta_blocks, 128, 0, stream>>>(
            delta, grad_output, output, row_count);

        launch_attention_backward_ampere_f16_d128_gqa4_from_stats<Causal, Accumulate>(
            grad_query, grad_key, grad_value, grad_output, delta, logsumexp,
            query, key, value, batch, query_sequence, key_sequence,
            query_heads, kv_heads, scale, query_position_offset, stream);

        const cudaError_t launch_error = cudaGetLastError();
        const cudaError_t free_error = cudaFreeAsync(delta, stream);
        CUDA_CHECK(launch_error);
        CUDA_CHECK(free_error);
    }

    void launch_attention_backward_lse_ampere_f16_d128_gqa4(
        __half *grad_query, __half *grad_key, __half *grad_value,
        const __half *grad_output, const __half *output,
        const float *logsumexp, const __half *query,
        const __half *key, const __half *value,
        int batch, int query_sequence, int key_sequence,
        int query_heads, int kv_heads, float scale,
        bool causal, int query_position_offset, bool accumulate,
        cudaStream_t stream
    ) {
        if (causal) {
            if (accumulate) {
                launch_attention_backward_lse_ampere_f16_d128_gqa4_impl<true, true>(
                    grad_query, grad_key, grad_value, grad_output, output, logsumexp,
                    query, key, value, batch, query_sequence, key_sequence,
                    query_heads, kv_heads, scale, query_position_offset, stream);
            } else {
                launch_attention_backward_lse_ampere_f16_d128_gqa4_impl<true, false>(
                    grad_query, grad_key, grad_value, grad_output, output, logsumexp,
                    query, key, value, batch, query_sequence, key_sequence,
                    query_heads, kv_heads, scale, query_position_offset, stream);
            }
        } else if (accumulate) {
            launch_attention_backward_lse_ampere_f16_d128_gqa4_impl<false, true>(
                grad_query, grad_key, grad_value, grad_output, output, logsumexp,
                query, key, value, batch, query_sequence, key_sequence,
                query_heads, kv_heads, scale, query_position_offset, stream);
        } else {
            launch_attention_backward_lse_ampere_f16_d128_gqa4_impl<false, false>(
                grad_query, grad_key, grad_value, grad_output, output, logsumexp,
                query, key, value, batch, query_sequence, key_sequence,
                query_heads, kv_heads, scale, query_position_offset, stream);
        }
    }

    /**
     * @brief Validates GQA attention tensors and configuration.
     * @throws std::invalid_argument If shapes, dtypes, dimensions, scale, or
     *         causal range are invalid.
     */
    void validate(const Tensor &gq, const Tensor &gk, const Tensor &gv, const Tensor &go,
                  const Tensor &q, const Tensor &k, const Tensor &v, const FlashAttentionOptions &o) {
        const Tensor *tensors[] = {&gq, &gk, &gv, &go, &q, &k, &v};
        for (const Tensor *tensor: tensors) {
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

/**
 * @brief Computes streaming backward gradients for grouped-query attention.
 *
 * Computes dQ, dK, and dV using an online softmax backward pass without an
 * O(query_length * key_length) attention-probability workspace.
 *
 * @param[out] grad_query Query gradient.
 * @param[out] grad_key Key gradient.
 * @param[out] grad_value Value gradient.
 * @param[in] grad_output Gradient with respect to the attention output.
 * @param[in] query Query tensor [batch, query_sequence, query_heads, head_dim].
 * @param[in] key Key tensor [batch, key_sequence, kv_heads, head_dim].
 * @param[in] value Value tensor with the same shape as @p key.
 * @param stream CUDA stream used for the operation.
 * @param options Attention dimensions, scaling, and causal configuration.
 * @param accumulate_grads Whether existing gradients should be preserved.
 * @throws std::invalid_argument If tensors or options are invalid.
 */
void flash_gqa_attention_backward(
    Tensor &grad_query, Tensor &grad_key, Tensor &grad_value, const Tensor &grad_output,
    const Tensor &query, const Tensor &key, const Tensor &value, cudaStream_t stream,
    const FlashAttentionOptions &options, bool accumulate_grads
) {
    validate(grad_query, grad_key, grad_value, grad_output, query, key, value, options);
    DeviceGuard device_guard(0);

    const int batch = static_cast<int>(query.size(0));
    const int qs = static_cast<int>(query.size(1));
    const int ks = static_cast<int>(key.size(1));
    const int qh = static_cast<int>(options.num_query_heads);
    const int kh = static_cast<int>(options.num_kv_heads);
    const int dim = static_cast<int>(options.head_dim);
    const float scale = options.attention_scale > 0.0f
        ? options.attention_scale : rsqrtf(static_cast<float>(dim));

    // The old implementation sent this common training shape through the
    // scalar/atomic fallback.  Reconstruct LSE and delta with Tensor Cores, then
    // use the same atomics-free dQ/dK/dV kernels as the saved-LSE API.
    if (query.dtype() == Dtype::F16 && dim == 128 && kh > 0
        && qh % kh == 0 && qh / kh == 4
        && current_device_supports_ampere_mma()) {
        launch_attention_backward_ampere_f16_d128_gqa4_no_lse(
            static_cast<__half *>(grad_query.raw_data()),
            static_cast<__half *>(grad_key.raw_data()),
            static_cast<__half *>(grad_value.raw_data()),
            static_cast<const __half *>(grad_output.raw_data()),
            static_cast<const __half *>(query.raw_data()),
            static_cast<const __half *>(key.raw_data()),
            static_cast<const __half *>(value.raw_data()),
            batch, qs, ks, qh, kh, scale, options.causal,
            static_cast<int>(options.query_position_offset),
            accumulate_grads, stream);
        return;
    }

    if (!accumulate_grads) {
        CUDA_CHECK(cudaMemsetAsync(grad_key.raw_data(), 0, grad_key.nbytes(), stream));
        CUDA_CHECK(cudaMemsetAsync(grad_value.raw_data(), 0, grad_value.nbytes(), stream));
    }
    const unsigned long long rows = static_cast<unsigned long long>(batch) * qs * qh;
    const unsigned long long blocks = (rows + kWarpsPerBlock - 1) / kWarpsPerBlock;
    if (blocks > static_cast<unsigned long long>(std::numeric_limits<unsigned int>::max()))
        throw std::invalid_argument("flash_gqa_attention_backward: CUDA grid is too large");
    const auto launch = [&]<typename T>() {
        launch_attention_backward(
            static_cast<T *>(grad_query.raw_data()), static_cast<T *>(grad_key.raw_data()),
            static_cast<T *>(grad_value.raw_data()), static_cast<const T *>(grad_output.raw_data()),
            static_cast<const T *>(query.raw_data()), static_cast<const T *>(key.raw_data()),
            static_cast<const T *>(value.raw_data()), static_cast<unsigned int>(blocks),
            batch, qs, ks, qh, kh, dim, scale, options.causal,
            static_cast<int>(options.query_position_offset), accumulate_grads, stream);
    };
    if (query.dtype() == Dtype::F16) launch.template operator()<__half>();
    else if (query.dtype() == Dtype::BF16) launch.template operator()<__nv_bfloat16>();
    else launch.template operator()<float>();
    CUDA_CHECK(cudaGetLastError());
}


/**
 * @brief Computes GQA attention backward using precomputed log-sum-exp values.
 *
 * The supplied forward output and LSE tensor allow probabilities to be
 * reconstructed directly, eliminating the statistics pass used by the basic
 * backward entry point. FP16 D=128 GQA4 uses a native Ampere mma.sync-specialized path.
 *
 * @param[out] grad_query Query gradient.
 * @param[out] grad_key Key gradient.
 * @param[out] grad_value Value gradient.
 * @param[in] grad_output Gradient with respect to the attention output.
 * @param[in] output Forward attention output.
 * @param[in] logsumexp FP32 tensor [batch, query_sequence, query_heads].
 * @param[in] query Query tensor.
 * @param[in] key Key tensor.
 * @param[in] value Value tensor.
 * @param stream CUDA stream used for the operation.
 * @param options Attention dimensions, scaling, and causal configuration.
 * @param accumulate_grads Whether existing gradients should be preserved.
 * @throws std::invalid_argument If tensors or options are invalid.
 */
void flash_gqa_attention_backward_with_lse(
    Tensor &grad_query, Tensor &grad_key, Tensor &grad_value, const Tensor &grad_output,
    const Tensor &output, const Tensor &logsumexp, const Tensor &query,
    const Tensor &key, const Tensor &value, cudaStream_t stream,
    const FlashAttentionOptions &options, bool accumulate_grads
) {
    validate(grad_query, grad_key, grad_value, grad_output, query, key, value, options);
    if (output.device_type() != DeviceType::CUDA || output.dtype() != query.dtype()
        || output.shape() != query.shape() || logsumexp.device_type() != DeviceType::CUDA
        || logsumexp.dtype() != Dtype::F32 || logsumexp.dim() != 3
        || logsumexp.size(0) != query.size(0) || logsumexp.size(1) != query.size(1)
        || logsumexp.size(2) != query.size(2)) {
        throw std::invalid_argument("flash_gqa_attention_backward_with_lse: invalid output or LSE tensor");
    }
    DeviceGuard device_guard(0);
    const int batch = static_cast<int>(query.size(0));
    const int qs = static_cast<int>(query.size(1));
    const int ks = static_cast<int>(key.size(1));
    const int qh = static_cast<int>(options.num_query_heads);
    const int kh = static_cast<int>(options.num_kv_heads);
    const int dim = static_cast<int>(options.head_dim);
    const float scale = options.attention_scale > 0.0f
                            ? options.attention_scale
                            : rsqrtf(static_cast<float>(dim));

    // Native Ampere Tensor-Core path for FP16 D=128 GQA4.  Four query heads
    // share each K/V tile in dQ, while the dK/dV kernel owns one K/V tile and
    // reduces the four head contributions inside the CTA.  The fast path writes
    // all gradients directly, so no preliminary memset is required.
    if (query.dtype() == Dtype::F16 && dim == 128 && kh > 0
        && qh % kh == 0 && qh / kh == 4
        && current_device_supports_ampere_mma()) {
        launch_attention_backward_lse_ampere_f16_d128_gqa4(
            static_cast<__half *>(grad_query.raw_data()),
            static_cast<__half *>(grad_key.raw_data()),
            static_cast<__half *>(grad_value.raw_data()),
            static_cast<const __half *>(grad_output.raw_data()),
            static_cast<const __half *>(output.raw_data()),
            static_cast<const float *>(logsumexp.raw_data()),
            static_cast<const __half *>(query.raw_data()),
            static_cast<const __half *>(key.raw_data()),
            static_cast<const __half *>(value.raw_data()),
            batch, qs, ks, qh, kh, scale, options.causal,
            static_cast<int>(options.query_position_offset),
            accumulate_grads, stream);
        return;
    }

    // Scalar-kernel fallback for everything the Tensor-Core path above doesn't
    // cover (FP32, BF16, non-128 head_dim, or a GQA ratio other than 4).
    if (!accumulate_grads) {
        CUDA_CHECK(cudaMemsetAsync(grad_key.raw_data(), 0, grad_key.nbytes(), stream));
        CUDA_CHECK(cudaMemsetAsync(grad_value.raw_data(), 0, grad_value.nbytes(), stream));
    }
    const unsigned long long rows = static_cast<unsigned long long>(batch) * qs * qh;
    const unsigned long long blocks = (rows + kWarpsPerBlock - 1) / kWarpsPerBlock;
    if (blocks > static_cast<unsigned long long>(std::numeric_limits<unsigned int>::max())) {
        throw std::invalid_argument("flash_gqa_attention_backward_with_lse: CUDA grid is too large");
    }
    const auto launch = [&]<typename T>() {
        launch_attention_backward_lse(
            static_cast<T *>(grad_query.raw_data()), static_cast<T *>(grad_key.raw_data()),
            static_cast<T *>(grad_value.raw_data()), static_cast<const T *>(grad_output.raw_data()),
            static_cast<const T *>(output.raw_data()), static_cast<const float *>(logsumexp.raw_data()),
            static_cast<const T *>(query.raw_data()), static_cast<const T *>(key.raw_data()),
            static_cast<const T *>(value.raw_data()), static_cast<unsigned int>(blocks),
            batch, qs, ks, qh, kh, dim, scale, options.causal,
            static_cast<int>(options.query_position_offset), accumulate_grads, stream);
    };
    if (query.dtype() == Dtype::F16) launch.template operator()<__half>();
    else if (query.dtype() == Dtype::BF16) launch.template operator()<__nv_bfloat16>();
    else launch.template operator()<float>();
    CUDA_CHECK(cudaGetLastError());
}
