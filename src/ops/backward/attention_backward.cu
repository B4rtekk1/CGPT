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

    // Tensor-Core training path for the dominant decoder shape: FP16, D=128 and
    // four query heads per KV head.  dQ and dK/dV deliberately use separate
    // ownership decompositions: a Q tile owns dQ, while a K tile owns dK/dV.
    // Consequently the latter has no global atomics.
    constexpr int kTcTile = 16;

    // Higher min-blocks-per-SM for better occupancy on laptop Ampere (16 SMs).
    /** @brief FP16 WMMA kernel computing query gradients from saved LSE. */
    __global__ __launch_bounds__(32, 6)
    void attention_backward_lse_dq_wmma(
        __half * __restrict__ grad_query, const __half * __restrict__ grad_output,
        const __half * __restrict__ output, const float * __restrict__ lse,
        const __half * __restrict__ query, const __half * __restrict__ key,
        const __half * __restrict__ value, int batch_size, int qs, int ks,
        int qh, int kh, float scale, bool causal, int position_offset,
        bool accumulate
    ) {
        using namespace nvcuda;
        const int lane = threadIdx.x;
        const int q_tiles = (qs + kTcTile - 1) / kTcTile;
        const int linear = blockIdx.x;
        const int qt = linear % q_tiles;
        const int head = (linear / q_tiles) % qh;
        const int batch = linear / (q_tiles * qh);
        if (batch >= batch_size) return;
        const int qb = qt * kTcTile, kv_head = head / 4;
        __shared__ __align__(16) __half q_s[kTcTile * 128], do_s[kTcTile * 128];
        __shared__ __align__(16) __half k_s[kTcTile * 128], v_s[kTcTile * 128];
        __shared__ __align__(16) __half ds_s[kTcTile * kTcTile];
        __shared__ float scores[kTcTile * kTcTile], dprob[kTcTile * kTcTile];
        __shared__ float dot[kTcTile], dq[kTcTile * 128], mma_s[kTcTile * kTcTile];
        for (int i = lane; i < kTcTile * 128; i += 32) {
            const int r = i / 128, d = i % 128, qp = qb + r;
            if (qp < qs) {
                const size_t off = ((size_t(batch) * qs + qp) * qh + head) * 128 + d;
                q_s[i] = query[off];
                do_s[i] = grad_output[off];
                dq[i] = 0.0f;
            } else {
                q_s[i] = __float2half(0);
                do_s[i] = __float2half(0);
                dq[i] = 0.0f;
            }
        }
        __syncthreads();
        for (int r = 0; r < kTcTile; ++r) {
            float x = 0.0f;
#pragma unroll
            for (int d = lane; d < 128; d += 32) {
                const int qp = qb + r;
                if (qp < qs) {
                    const size_t off = ((size_t(batch) * qs + qp) * qh + head) * 128 + d;
                    x = fmaf(__half2float(do_s[r * 128 + d]), __half2float(output[off]), x);
                }
            }
            float z = 0.0f;
            warp_sum_pair(x, z);
            if (lane == 0) dot[r] = x;
        }
        __syncthreads();
        for (int kb = 0; kb < ks; kb += kTcTile) {
            for (int i = lane; i < kTcTile * 128; i += 32) {
                const int r = i / 128, d = i % 128, kp = kb + r;
                if (kp < ks) {
                    const size_t off = ((size_t(batch) * ks + kp) * kh + kv_head) * 128 + d;
                    k_s[i] = key[off];
                    v_s[i] = value[off];
                } else {
                    k_s[i] = __float2half(0);
                    v_s[i] = __float2half(0);
                }
            }
            __syncthreads();
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> b;
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> c;
            wmma::fill_fragment(c, 0.0f);
#pragma unroll
            for (int d = 0; d < 128; d += 16) {
                wmma::load_matrix_sync(a, q_s + d, 128);
                wmma::load_matrix_sync(b, k_s + d, 128);
                wmma::mma_sync(c, a, b, c);
            }
            wmma::store_matrix_sync(scores, c, 16, wmma::mem_row_major);
            wmma::fill_fragment(c, 0.0f);
#pragma unroll
            for (int d = 0; d < 128; d += 16) {
                wmma::load_matrix_sync(a, do_s + d, 128);
                wmma::load_matrix_sync(b, v_s + d, 128);
                wmma::mma_sync(c, a, b, c);
            }
            wmma::store_matrix_sync(dprob, c, 16, wmma::mem_row_major);
            __syncwarp();
            for (int i = lane; i < 256; i += 32) {
                const int r = i / 16, col = i % 16, qp = qb + r, kp = kb + col;
                const bool valid = qp < qs && kp < ks && (!causal || kp <= position_offset + qp);
                const float p = valid
                                    ? attention_exp(scores[i] * scale - lse[(size_t(batch) * qs + qp) * qh + head])
                                    : 0.0f;
                ds_s[i] = __float2half_rn(p * (dprob[i] - dot[r]));
            }
            __syncwarp();
#pragma unroll
            for (int d = 0; d < 128; d += 16) {
                wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> da;
                wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> db;
                wmma::fill_fragment(c, 0.0f);
                wmma::load_matrix_sync(da, ds_s, 16);
                wmma::load_matrix_sync(db, k_s + d, 128);
                wmma::mma_sync(c, da, db, c);
                wmma::store_matrix_sync(mma_s, c, 16, wmma::mem_row_major);
                for (int i = lane; i < 256; i += 32) dq[(i / 16) * 128 + d + (i % 16)] += mma_s[i] * scale;
                __syncwarp();
            }
            __syncthreads();
        }
        for (int i = lane; i < kTcTile * 128; i += 32) {
            const int r = i / 128, d = i % 128, qp = qb + r;
            if (qp < qs) {
                const size_t off = ((size_t(batch) * qs + qp) * qh + head) * 128 + d;
                grad_query[off] = __float2half_rn(dq[i] + (accumulate ? __half2float(grad_query[off]) : 0.0f));
            }
        }
    }

    /** @brief FP16 WMMA kernel computing key and value gradients from saved LSE. */
    __global__ __launch_bounds__(32, 6)
    void attention_backward_lse_dkv_wmma(
        __half * __restrict__ grad_key, __half * __restrict__ grad_value, const __half * __restrict__ grad_output,
        const __half * __restrict__ output, const float * __restrict__ lse, const __half * __restrict__ query,
        const __half * __restrict__ key, const __half * __restrict__ value, int batch_size, int qs, int ks,
        int qh, int kh, float scale, bool causal, int position_offset, bool accumulate
    ) {
        using namespace nvcuda;
        const int lane = threadIdx.x;
        const int k_tiles = (ks + 15) / 16, linear = blockIdx.x, kt = linear % k_tiles;
        const int kvh = (linear / k_tiles) % kh, batch = linear / (k_tiles * kh);
        if (batch >= batch_size) return;
        const int kb = kt * 16;
        __shared__ __align__(16) __half q_s[2048], do_s[2048], k_s[2048], v_s[2048], ds_s[256], p_s[256];
        __shared__ float scores[256], dprob[256], dot[16], dk[2048], dv[2048], mma_s[256];
        for (int i = lane; i < 2048; i += 32) {
            const int r = i / 128, d = i % 128, kp = kb + r;
            if (kp < ks) {
                const size_t o = ((size_t(batch) * ks + kp) * kh + kvh) * 128 + d;
                k_s[i] = key[o];
                v_s[i] = value[o];
            } else {
                k_s[i] = __float2half(0);
                v_s[i] = __float2half(0);
            }
            dk[i] = dv[i] = 0.0f;
        }
        __syncthreads();
        const int q_tiles = (qs + 15) / 16;
        for (int hl = 0; hl < 4; ++hl)
            for (int qt = 0; qt < q_tiles; ++qt) {
                const int head = kvh * 4 + hl, qb = qt * 16;
                for (int i = lane; i < 2048; i += 32) {
                    const int r = i / 128, d = i % 128, qp = qb + r;
                    if (qp < qs) {
                        const size_t o = ((size_t(batch) * qs + qp) * qh + head) * 128 + d;
                        q_s[i] = query[o];
                        do_s[i] = grad_output[o];
                    } else {
                        q_s[i] = __float2half(0);
                        do_s[i] = __float2half(0);
                    }
                }
                __syncthreads();
                for (int r = 0; r < 16; ++r) {
                    float x = 0;
                    for (int d = lane; d < 128; d += 32) {
                        const int qp = qb + r;
                        if (qp < qs) {
                            const size_t o = ((size_t(batch) * qs + qp) * qh + head) * 128 + d;
                            x = fmaf(__half2float(do_s[r * 128 + d]), __half2float(output[o]), x);
                        }
                    }
                    float z = 0;
                    warp_sum_pair(x, z);
                    if (lane == 0)dot[r] = x;
                }
                wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a;
                wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> b;
                wmma::fragment<wmma::accumulator, 16, 16, 16, float> c;
                wmma::fill_fragment(c, 0);
                for (int d = 0; d < 128; d += 16) {
                    wmma::load_matrix_sync(a, q_s + d, 128);
                    wmma::load_matrix_sync(b, k_s + d, 128);
                    wmma::mma_sync(c, a, b, c);
                }
                wmma::store_matrix_sync(scores, c, 16, wmma::mem_row_major);
                wmma::fill_fragment(c, 0);
                for (int d = 0; d < 128; d += 16) {
                    wmma::load_matrix_sync(a, do_s + d, 128);
                    wmma::load_matrix_sync(b, v_s + d, 128);
                    wmma::mma_sync(c, a, b, c);
                }
                wmma::store_matrix_sync(dprob, c, 16, wmma::mem_row_major);
                __syncwarp();
                for (int i = lane; i < 256; i += 32) {
                    const int r = i / 16, col = i % 16, qp = qb + r, kp = kb + col;
                    const bool ok = qp < qs && kp < ks && (!causal || kp <= position_offset + qp);
                    const float p = ok
                                        ? attention_exp(scores[i] * scale - lse[(size_t(batch) * qs + qp) * qh + head])
                                        : 0;
                    p_s[i] = __float2half_rn(p);
                    ds_s[i] = __float2half_rn(p * (dprob[i] - dot[r]));
                }
                __syncwarp();
                for (int d = 0; d < 128; d += 16) {
                    wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::col_major> ta;
                    wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> tb;
                    wmma::fill_fragment(c, 0);
                    wmma::load_matrix_sync(ta, ds_s, 16);
                    wmma::load_matrix_sync(tb, q_s + d, 128);
                    wmma::mma_sync(c, ta, tb, c);
                    wmma::store_matrix_sync(mma_s, c, 16, wmma::mem_row_major);
                    for (int i = lane; i < 256; i += 32)dk[(i / 16) * 128 + d + i % 16] += mma_s[i] * scale;
                    wmma::fill_fragment(c, 0);
                    wmma::load_matrix_sync(ta, p_s, 16);
                    wmma::load_matrix_sync(tb, do_s + d, 128);
                    wmma::mma_sync(c, ta, tb, c);
                    wmma::store_matrix_sync(mma_s, c, 16, wmma::mem_row_major);
                    for (int i = lane; i < 256; i += 32)dv[(i / 16) * 128 + d + i % 16] += mma_s[i];
                    __syncwarp();
                }
                __syncthreads();
            }
        for (int i = lane; i < 2048; i += 32) {
            const int r = i / 128, d = i % 128, kp = kb + r;
            if (kp < ks) {
                const size_t o = ((size_t(batch) * ks + kp) * kh + kvh) * 128 + d;
                grad_key[o] = __float2half_rn(dk[i] + (accumulate ? __half2float(grad_key[o]) : 0));
                grad_value[o] = __float2half_rn(dv[i] + (accumulate ? __half2float(grad_value[o]) : 0));
            }
        }
    }

    /** @brief Launches the pair of FP16 WMMA LSE backward kernels. */
    void launch_attention_backward_lse_wmma(
        __half *gq, __half *gk, __half *gv, const __half *go, const __half *out, const float *lse,
        const __half *q, const __half *k, const __half *v, int batch, int qs, int ks, int qh, int kh,
        float scale, bool causal, int offset, bool accumulate, cudaStream_t stream
    ) {
        const unsigned int dq_blocks = static_cast<unsigned int>(batch * qh * ((qs + 15) / 16));
        const unsigned int dkv_blocks = static_cast<unsigned int>(batch * kh * ((ks + 15) / 16));
        attention_backward_lse_dq_wmma<<<dq_blocks, 32, 0, stream>>>(gq, go, out, lse, q, k, v, batch, qs, ks, qh, kh,
                                                                     scale, causal, offset, accumulate);
        attention_backward_lse_dkv_wmma<<<dkv_blocks, 32, 0, stream>>>(gk, gv, go, out, lse, q, k, v, batch, qs, ks, qh,
                                                                       kh, scale, causal, offset, accumulate);
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
 * backward entry point. FP16 may use an additional WMMA-specialized path.
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
    const float scale = options.attention_scale > 0.0f
                            ? options.attention_scale
                            : rsqrtf(static_cast<float>(dim));

    // Tensor-Core fast path. attention_backward_lse_dq_wmma / _dkv_wmma cover
    // FP16, head_dim==128, and an exact 4:1 GQA ratio -- the dominant decoder
    // training shape. Every S=QK^T, dP=dO*V^T, dQ=dS*K, dK=dS^T*Q, and
    // dV=P^T*dO product is done as a 16x16x16 WMMA tile instead of per-lane
    // scalar fmaf, and dK/dV are written directly with no atomics at all: each
    // block owns one contiguous K/V tile and iterates over every query
    // position that attends to it, rather than each query row atomically
    // scattering into K/V positions it shares with other rows. This code
    // path previously existed but was never invoked from here. Checked before
    // the scalar-kernel grid-size validation below, since its own grid
    // (batch*heads*ceil(seq/16)) is independent of and generally much smaller
    // than the row-per-warp grid that validation guards.
    if (query.dtype() == Dtype::F16 && dim == 128 && kh > 0 && qh % kh == 0 && qh / kh == 4) {
        launch_attention_backward_lse_wmma(
            static_cast<__half *>(grad_query.raw_data()), static_cast<__half *>(grad_key.raw_data()),
            static_cast<__half *>(grad_value.raw_data()), static_cast<const __half *>(grad_output.raw_data()),
            static_cast<const __half *>(output.raw_data()), static_cast<const float *>(logsumexp.raw_data()),
            static_cast<const __half *>(query.raw_data()), static_cast<const __half *>(key.raw_data()),
            static_cast<const __half *>(value.raw_data()), batch, qs, ks, qh, kh, scale, options.causal,
            static_cast<int>(options.query_position_offset), accumulate_grads, stream);
        CUDA_CHECK(cudaGetLastError());
        return;
    }

    // Scalar-kernel fallback for everything the Tensor-Core path above doesn't
    // cover (FP32, BF16, non-128 head_dim, or a GQA ratio other than 4).
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
