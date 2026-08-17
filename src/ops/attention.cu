/**
 * @file flash_attention.cu
 * @brief FP16 Tensor Core implementation of tiled Flash Attention for GQA.
 *
 * This translation unit implements the forward pass of scaled dot-product
 * attention without materializing the full @f$QK^T@f$ matrix in global memory.
 * Keys and values are streamed through shared-memory tiles, while each query
 * row maintains an online softmax state consisting of a running maximum,
 * normalization sum, and output accumulator.
 *
 * Supported tensor layouts:
 * @code
 * query:  [batch, query_sequence, query_heads, head_dim]
 * key:    [batch, key_value_sequence, kv_heads, head_dim]
 * value:  [batch, key_value_sequence, kv_heads, head_dim]
 * output: [batch, query_sequence, query_heads, head_dim]
 * @endcode
 *
 * The implementation currently supports:
 * - FP16 input and output tensors,
 * - head dimensions 32, 64, and 128,
 * - grouped-query attention where query_heads is divisible by kv_heads,
 * - optional causal masking,
 * - prefilling and cached decoding through query_position_offset.
 *
 * @note Tensor storage must be contiguous in the documented row-major layout.
 * @note The kernels use WMMA and therefore require a Tensor Core-capable GPU.
 */
#include "ops/attention.h"

#include "core/cuda_check.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>

#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

namespace {

/// Number of threads in a CUDA warp.
constexpr int kWarpSize = 32;

/// Number of K/V rows processed per iteration by the generic per-head kernel.
/// 32 keeps all lanes busy in the warp reductions and reduces the number of
/// tile loads vs 16. On low-SM laptop GPUs (RTX 3050 class) this is a good
/// compromise between arithmetic intensity and occupancy.
constexpr int kBlockN = 32;

/// Number of K/V rows processed per iteration by the grouped 4:1 GQA kernel.
constexpr int kGroupedBlockN = 32;

// Prefer smaller query tiles on consumer Ampere (few SMs, limited registers).
// BlockM=32 gives higher occupancy than 64 while still using full WMMA 16x16.
constexpr int kPreferredBlockM = 32;

/**
 * @brief Computes a warp-wide sum using shuffle-down instructions.
 *
 * @param value Value contributed by the calling lane.
 * @return The complete sum in lane 0. Values returned by other lanes are
 *         partial and must not be used as the final reduction result.
 */
__device__ __forceinline__ float warp_reduce_sum(float value) {
    #pragma unroll
    for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(0xFFFFFFFFU, value, offset);
    }
    return value;
}

/**
 * @brief Computes a warp-wide maximum using shuffle-down instructions.
 *
 * @param value Value contributed by the calling lane.
 * @return The complete maximum in lane 0. Values returned by other lanes are
 *         partial and must not be used as the final reduction result.
 */
__device__ __forceinline__ float warp_reduce_max(float value) {
    #pragma unroll
    for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
        value = fmaxf(value, __shfl_down_sync(0xFFFFFFFFU, value, offset));
    }
    return value;
}

/**
 * @brief Cooperatively loads a rectangular FP16 tile into shared memory.
 *
 * Threads copy 16-byte vectors, corresponding to eight contiguous FP16 values.
 * Rows beyond @p valid_rows are explicitly zero-filled, allowing callers to
 * process a partial final query tile without separate boundary kernels.
 *
 * @tparam Rows Number of rows in the destination tile.
 * @tparam HeadDim Number of FP16 elements in each row; must be divisible by 8.
 * @param destination Shared-memory destination of size Rows * HeadDim.
 * @param source Pointer to row zero of the selected tensor/head slice.
 * @param first_row First logical source row requested by the tile.
 * @param valid_rows Total number of valid rows in the source slice.
 * @param source_row_stride Distance between consecutive source rows, in FP16
 *        elements rather than bytes.
 *
 * @pre destination is aligned sufficiently for uint4 stores.
 * @pre Every valid source row is aligned sufficiently for uint4 loads.
 */
template <int Rows, int HeadDim>
__device__ __forceinline__ void load_rows_f16(
    __half* __restrict__ destination,
    const __half* __restrict__ source,
    int first_row,
    int valid_rows,
    std::int64_t source_row_stride
) {
    static_assert(HeadDim % 8 == 0);
    constexpr int kVectorsPerRow = HeadDim / 8;
    constexpr int kVectorCount = Rows * kVectorsPerRow;

    auto* destination_vector = reinterpret_cast<uint4*>(destination);

    for (int vector_index = static_cast<int>(threadIdx.x);
         vector_index < kVectorCount;
         vector_index += static_cast<int>(blockDim.x)) {
        const int row = vector_index / kVectorsPerRow;
        const int column_vector = vector_index % kVectorsPerRow;
        const int source_row = first_row + row;

        if (source_row < valid_rows) {
            const __half* source_ptr =
                source + static_cast<std::int64_t>(source_row) * source_row_stride
                + column_vector * 8;
            destination_vector[vector_index] =
                *reinterpret_cast<const uint4*>(source_ptr);
        } else {
            destination_vector[vector_index] = make_uint4(0U, 0U, 0U, 0U);
        }
    }
}

/**
 * @brief Cooperatively loads matching K and V tiles into shared memory.
 *
 * Key and value rows use the same logical indices and stride. Invalid rows in
 * the final tile are zero-filled. The attention mask still marks those rows as
 * invalid, so the zero-fill exists only to make vectorized WMMA input safe.
 *
 * @tparam HeadDim Number of FP16 elements per attention head.
 * @tparam Rows Number of K/V rows loaded in one iteration.
 * @param key_shared Shared-memory key tile of size Rows * HeadDim.
 * @param value_shared Shared-memory value tile of size Rows * HeadDim.
 * @param key Pointer to row zero of the selected key head.
 * @param value Pointer to row zero of the selected value head.
 * @param first_row First K/V sequence position requested by the tile.
 * @param valid_rows Total number of valid K/V rows.
 * @param source_row_stride Distance between consecutive rows, in FP16 elements.
 */
template <int HeadDim, int Rows>
__device__ __forceinline__ void load_kv_rows_f16(
    __half* __restrict__ key_shared,
    __half* __restrict__ value_shared,
    const __half* __restrict__ key,
    const __half* __restrict__ value,
    int first_row,
    int valid_rows,
    std::int64_t source_row_stride
) {
    static_assert(HeadDim % 8 == 0);
    constexpr int kVectorsPerRow = HeadDim / 8;
    constexpr int kVectorCount = Rows * kVectorsPerRow;

    auto* key_shared_vector = reinterpret_cast<uint4*>(key_shared);
    auto* value_shared_vector = reinterpret_cast<uint4*>(value_shared);

    for (int vector_index = static_cast<int>(threadIdx.x);
         vector_index < kVectorCount;
         vector_index += static_cast<int>(blockDim.x)) {
        const int row = vector_index / kVectorsPerRow;
        const int column_vector = vector_index % kVectorsPerRow;
        const int source_row = first_row + row;

        if (source_row < valid_rows) {
            const std::int64_t offset =
                static_cast<std::int64_t>(source_row) * source_row_stride
                + column_vector * 8;
            key_shared_vector[vector_index] =
                *reinterpret_cast<const uint4*>(key + offset);
            value_shared_vector[vector_index] =
                *reinterpret_cast<const uint4*>(value + offset);
        } else {
            const uint4 zero = make_uint4(0U, 0U, 0U, 0U);
            key_shared_vector[vector_index] = zero;
            value_shared_vector[vector_index] = zero;
        }
    }
}

/**
 * @brief Computes one tiled FP16 GQA attention forward pass per query head.
 *
 * Grid mapping:
 * @code
 * blockIdx.x -> (batch, query_head, query_tile)
 * @endcode
 * A CTA owns BlockM query rows for exactly one query head. Every warp owns 16
 * of those rows and performs both matrix products with WMMA:
 * @f$S = QK^T@f$ and @f$O = PV@f$.
 *
 * Keys and values are streamed in kBlockN-row tiles. For each query row, the
 * kernel updates the numerically stable online-softmax recurrence:
 * @f[
 * m' = max(m, max(S_t)),
 * l' = l exp(m-m') + sum(exp(S_t-m')),
 * O' = O exp(m-m') + exp(S_t-m')V_t.
 * @f]
 * Final output is @f$O/l@f$. Therefore no full attention matrix is written to
 * global memory.
 *
 * @tparam HeadDim Attention head width. Supported values: 32, 64, 128.
 * @tparam BlockM Number of query rows owned by a CTA. Supported values: 16, 32,
 *         64; current dispatch uses 64.
 * @param output Contiguous FP16 tensor [B, Q, Hq, D].
 * @param logsumexp Optional contiguous FP32 tensor [B, Q, Hq] receiving the
 *        per-row log-sum-exp statistic. Pass nullptr when the statistic is not
 *        required.
 * @param query Contiguous FP16 tensor [B, Q, Hq, D].
 * @param key Contiguous FP16 tensor [B, K, Hkv, D].
 * @param value Contiguous FP16 tensor [B, K, Hkv, D].
 * @param batch_size Number of batches B.
 * @param query_sequence Query sequence length Q.
 * @param key_value_sequence Key/value sequence length K.
 * @param num_query_heads Number of query heads Hq.
 * @param num_kv_heads Number of key/value heads Hkv.
 * @param scale Multiplicative score scale, normally 1/sqrt(D).
 * @param causal Whether to mask keys to the right of each absolute query.
 * @param query_position_offset Absolute position represented by query row zero;
 *        used when Q contains only newly decoded tokens.
 */
template <int HeadDim, int BlockM>
__global__ void flash_gqa_f16_tensor_core_kernel(
    __half* __restrict__ output,
    float* __restrict__ logsumexp,
    const __half* __restrict__ query,
    const __half* __restrict__ key,
    const __half* __restrict__ value,
    int batch_size,
    int query_sequence,
    int key_value_sequence,
    int num_query_heads,
    int num_kv_heads,
    float scale,
    bool causal,
    int query_position_offset
) {
    static_assert(HeadDim == 32 || HeadDim == 64 || HeadDim == 128);
    static_assert(BlockM == 16 || BlockM == 32 || BlockM == 64);
    static_assert(BlockM % 16 == 0);

    constexpr int kWarps = BlockM / 16;
    constexpr int kThreads = kWarps * kWarpSize;
    static_assert(kThreads == 64 || kThreads == 128);

    const int query_tiles = (query_sequence + BlockM - 1) / BlockM;
    const int linear = static_cast<int>(blockIdx.x);
    const int query_tile = linear % query_tiles;
    const int query_head = (linear / query_tiles) % num_query_heads;
    const int batch = linear / (query_tiles * num_query_heads);
    if (batch >= batch_size) {
        return;
    }

    const int heads_per_kv = num_query_heads / num_kv_heads;
    const int kv_head = query_head / heads_per_kv;
    const int query_begin = query_tile * BlockM;

    const std::int64_t query_head_base =
        (static_cast<std::int64_t>(batch) * query_sequence * num_query_heads
         + query_head) * HeadDim;
    const std::int64_t kv_head_base =
        (static_cast<std::int64_t>(batch) * key_value_sequence * num_kv_heads
         + kv_head) * HeadDim;
    const std::int64_t query_row_stride =
        static_cast<std::int64_t>(num_query_heads) * HeadDim;
    const std::int64_t kv_row_stride =
        static_cast<std::int64_t>(num_kv_heads) * HeadDim;

    extern __shared__ __align__(16) unsigned char shared_raw[];
    auto* query_shared = reinterpret_cast<__half*>(shared_raw);
    auto* ptr = shared_raw + BlockM * HeadDim * sizeof(__half);

    auto* key_shared = reinterpret_cast<__half*>(ptr);
    ptr += kBlockN * HeadDim * sizeof(__half);

    auto* value_shared = reinterpret_cast<__half*>(ptr);
    ptr += kBlockN * HeadDim * sizeof(__half);

    auto* scores = reinterpret_cast<float*>(ptr);
    ptr += BlockM * kBlockN * sizeof(float);

    auto* probabilities = reinterpret_cast<__half*>(ptr);
    ptr += BlockM * kBlockN * sizeof(__half);

    auto* output_accumulator = reinterpret_cast<__half*>(ptr);
    ptr += BlockM * HeadDim * sizeof(__half);

    auto* mma_tiles = reinterpret_cast<float*>(ptr);
    ptr += kWarps * 16 * 16 * sizeof(float);

    auto* running_max = reinterpret_cast<float*>(ptr);
    ptr += BlockM * sizeof(float);

    auto* running_sum = reinterpret_cast<float*>(ptr);

    load_rows_f16<BlockM, HeadDim>(
        query_shared,
        query + query_head_base,
        query_begin,
        query_sequence,
        query_row_stride);

    for (int index = static_cast<int>(threadIdx.x);
         index < BlockM * HeadDim;
         index += kThreads) {
        output_accumulator[index] = __float2half_rn(0.0F);
    }
    for (int row = static_cast<int>(threadIdx.x); row < BlockM; row += kThreads) {
        running_max[row] = -INFINITY;
        running_sum[row] = 0.0F;
    }
    __syncthreads();

    const int max_visible_key = causal
        ? min(key_value_sequence,
              query_position_offset + query_begin + BlockM)
        : key_value_sequence;

    const int warp = static_cast<int>(threadIdx.x) / kWarpSize;
    const int lane = static_cast<int>(threadIdx.x) & (kWarpSize - 1);
    const int warp_row_begin = warp * 16;
    float* warp_scores = scores + warp_row_begin * kBlockN;
    __half* warp_probabilities =
        probabilities + warp_row_begin * kBlockN;
    float* warp_mma_tile = mma_tiles + warp * 16 * 16;

    for (int key_begin = 0;
         key_begin < max_visible_key;
         key_begin += kBlockN) {
        load_kv_rows_f16<HeadDim, kBlockN>(
            key_shared,
            value_shared,
            key + kv_head_base,
            value + kv_head_base,
            key_begin,
            key_value_sequence,
            kv_row_stride);
        __syncthreads();

        {
            using namespace nvcuda;
            const __half* warp_query =
                query_shared + warp_row_begin * HeadDim;

            #pragma unroll
            for (int subtile = 0; subtile < kBlockN; subtile += 16) {
                wmma::fragment<wmma::matrix_a, 16, 16, 16, __half,
                               wmma::row_major> query_fragment;
                wmma::fragment<wmma::matrix_b, 16, 16, 16, __half,
                               wmma::col_major> key_fragment;
                wmma::fragment<wmma::accumulator, 16, 16, 16, float>
                    score_fragment;
                wmma::fill_fragment(score_fragment, 0.0F);

                #pragma unroll
                for (int dimension = 0;
                     dimension < HeadDim;
                     dimension += 16) {
                    wmma::load_matrix_sync(
                        query_fragment, warp_query + dimension, HeadDim);
                    wmma::load_matrix_sync(
                        key_fragment,
                        key_shared + subtile * HeadDim + dimension,
                        HeadDim);
                    wmma::mma_sync(
                        score_fragment,
                        query_fragment,
                        key_fragment,
                        score_fragment);
                }
                wmma::store_matrix_sync(
                    warp_scores + subtile,
                    score_fragment,
                    kBlockN,
                    wmma::mem_row_major);
            }
        }
        __syncwarp();

        #pragma unroll
        for (int local_row = 0; local_row < 16; ++local_row) {
            const int row = warp_row_begin + local_row;
            const int query_position = query_begin + row;
            const int key_position = key_begin + lane;

            const bool valid =
                query_position < query_sequence
                && key_position < key_value_sequence
                && (!causal
                    || key_position
                        <= query_position_offset + query_position);
            const float score = valid
                ? warp_scores[local_row * kBlockN + lane] * scale
                : -INFINITY;

            float tile_max = warp_reduce_max(score);
            tile_max = __shfl_sync(0xFFFFFFFFU, tile_max, 0);

            const float old_max = running_max[row];
            const float new_max = fmaxf(old_max, tile_max);
            const float old_scale = old_max == -INFINITY
                ? 0.0F
                : __expf(old_max - new_max);
            const float probability =
                valid ? __expf(score - new_max) : 0.0F;

            float tile_sum = warp_reduce_sum(probability);
            tile_sum = __shfl_sync(0xFFFFFFFFU, tile_sum, 0);

            warp_probabilities[local_row * kBlockN + lane] =
                __float2half_rn(probability);

            for (int dimension = lane;
                 dimension < HeadDim;
                 dimension += kWarpSize) {
                const int accumulator_index = row * HeadDim + dimension;
                output_accumulator[accumulator_index] = __float2half_rn(
                    __half2float(output_accumulator[accumulator_index])
                    * old_scale);
            }

            if (lane == 0) {
                running_max[row] = new_max;
                running_sum[row] =
                    running_sum[row] * old_scale + tile_sum;
            }
        }
        __syncwarp();

        // Multiply the current probability tile by V. Each warp still owns
        // the same 16 query rows, so no inter-warp accumulation is required.
        {
            using namespace nvcuda;

            #pragma unroll
            for (int dimension = 0;
                 dimension < HeadDim;
                 dimension += 16) {
                wmma::fragment<wmma::accumulator, 16, 16, 16, float>
                    output_fragment;
                wmma::fill_fragment(output_fragment, 0.0F);

                #pragma unroll
                for (int subtile = 0; subtile < kBlockN; subtile += 16) {
                    wmma::fragment<wmma::matrix_a, 16, 16, 16, __half,
                                   wmma::row_major> probability_fragment;
                    wmma::fragment<wmma::matrix_b, 16, 16, 16, __half,
                                   wmma::row_major> value_fragment;
                    wmma::load_matrix_sync(
                        probability_fragment,
                        warp_probabilities + subtile,
                        kBlockN);
                    wmma::load_matrix_sync(
                        value_fragment,
                        value_shared + subtile * HeadDim + dimension,
                        HeadDim);
                    wmma::mma_sync(
                        output_fragment,
                        probability_fragment,
                        value_fragment,
                        output_fragment);
                }

                wmma::store_matrix_sync(
                    warp_mma_tile,
                    output_fragment,
                    16,
                    wmma::mem_row_major);
                __syncwarp();

                for (int index = lane; index < 16 * 16; index += kWarpSize) {
                    const int local_row = index / 16;
                    const int column = index % 16;
                    const int row = warp_row_begin + local_row;
                    const int accumulator_index =
                        row * HeadDim + dimension + column;
                    output_accumulator[accumulator_index] = __float2half_rn(
                        __half2float(output_accumulator[accumulator_index])
                        + warp_mma_tile[index]);
                }
                __syncwarp();
            }
        }

        // K/V are shared by every warp and may only be overwritten after all
        // warps have finished the current P @ V operation.
        __syncthreads();
    }

    for (int index = static_cast<int>(threadIdx.x);
         index < BlockM * HeadDim;
         index += kThreads) {
        const int row = index / HeadDim;
        const int dimension = index % HeadDim;
        const int query_position = query_begin + row;
        if (query_position < query_sequence) {
            const float denominator = running_sum[row];
            const float result = denominator > 0.0F
                ? __half2float(output_accumulator[index]) / denominator
                : 0.0F;
            const std::int64_t output_offset =
                query_head_base
                + static_cast<std::int64_t>(query_position)
                    * query_row_stride
                + dimension;
            output[output_offset] = __float2half_rn(result);
            if (logsumexp != nullptr && dimension == 0) {
                logsumexp[(static_cast<std::int64_t>(batch) * query_sequence
                           + query_position) * num_query_heads + query_head] =
                    denominator > 0.0F ? running_max[row] + __logf(denominator)
                                       : -INFINITY;
            }
        }
    }
}


/**
 * @brief Specialized GQA kernel that reuses each K/V tile across a head group.
 *
 * Grid mapping:
 * @code
 * blockIdx.x -> (batch, kv_head, query_tile)
 * warp       -> one query head associated with kv_head
 * @endcode
 *
 * Unlike the generic kernel, a CTA owns all query heads mapped to one K/V head.
 * K and V are therefore loaded once per 32-row sequence tile and consumed by
 * four independent warps. Each warp maintains its own online-softmax state and
 * output accumulator, so synchronization is required only around shared K/V
 * tile replacement.
 *
 * This specialization is intentionally constrained to D=128 and a 4:1 GQA
 * ratio. Its larger float accumulator improves numerical behavior but requires
 * more dynamic shared memory than the generic path.
 *
 * @tparam HeadDim Must be 128.
 * @tparam GroupSize Must be 4 query heads per K/V head.
 * @param output Contiguous FP16 tensor [B, Q, Hq, D].
 * @param logsumexp Optional contiguous FP32 tensor [B, Q, Hq] receiving the
 *        per-row log-sum-exp statistic. Pass nullptr when the statistic is not
 *        required.
 * @param query Contiguous FP16 tensor [B, Q, Hq, D].
 * @param key Contiguous FP16 tensor [B, K, Hkv, D].
 * @param value Contiguous FP16 tensor [B, K, Hkv, D].
 * @param batch_size Number of batches B.
 * @param query_sequence Query sequence length Q.
 * @param key_value_sequence Key/value sequence length K.
 * @param num_query_heads Number of query heads Hq.
 * @param num_kv_heads Number of key/value heads Hkv.
 * @param scale Multiplicative score scale, normally 1/sqrt(D).
 * @param causal Whether to enable the causal mask.
 * @param query_position_offset Absolute position represented by query row zero.
 */
template <int HeadDim, int GroupSize>
__global__ __launch_bounds__(128, 1)
void flash_gqa_grouped_f16_tensor_core_kernel(
    __half* __restrict__ output,
    float* __restrict__ logsumexp,
    const __half* __restrict__ query,
    const __half* __restrict__ key,
    const __half* __restrict__ value,
    int batch_size,
    int query_sequence,
    int key_value_sequence,
    int num_query_heads,
    int num_kv_heads,
    float scale,
    bool causal,
    int query_position_offset
) {
    static_assert(HeadDim == 128);
    static_assert(GroupSize == 4);
    constexpr int kBlockM = 16;
    constexpr int kWarps = GroupSize;
    constexpr int kThreads = kWarps * kWarpSize;
    constexpr int kN = kGroupedBlockN;

    const int query_tiles = (query_sequence + kBlockM - 1) / kBlockM;
    const int linear = static_cast<int>(blockIdx.x);
    const int query_tile = linear % query_tiles;
    const int kv_head = (linear / query_tiles) % num_kv_heads;
    const int batch = linear / (query_tiles * num_kv_heads);
    if (batch >= batch_size) {
        return;
    }

    const int query_begin = query_tile * kBlockM;
    const int first_query_head = kv_head * GroupSize;
    const std::int64_t kv_row_stride =
        static_cast<std::int64_t>(num_kv_heads) * HeadDim;
    const std::int64_t kv_head_base =
        (static_cast<std::int64_t>(batch) * key_value_sequence * num_kv_heads
         + kv_head) * HeadDim;

    extern __shared__ __align__(16) unsigned char shared_raw[];
    auto* query_shared = reinterpret_cast<__half*>(shared_raw);
    auto* ptr = shared_raw + GroupSize * kBlockM * HeadDim * sizeof(__half);
    auto* key_shared = reinterpret_cast<__half*>(ptr);
    ptr += kN * HeadDim * sizeof(__half);
    auto* value_shared = reinterpret_cast<__half*>(ptr);
    ptr += kN * HeadDim * sizeof(__half);
    auto* scores = reinterpret_cast<float*>(ptr);
    ptr += GroupSize * kBlockM * kN * sizeof(float);
    auto* probabilities = reinterpret_cast<__half*>(ptr);
    ptr += GroupSize * kBlockM * kN * sizeof(__half);
    auto* output_accumulator = reinterpret_cast<float*>(ptr);
    ptr += GroupSize * kBlockM * HeadDim * sizeof(float);
    auto* mma_tiles = reinterpret_cast<float*>(ptr);
    ptr += GroupSize * kBlockM * 16 * sizeof(float);
    auto* running_max = reinterpret_cast<float*>(ptr);
    ptr += GroupSize * kBlockM * sizeof(float);
    auto* running_sum = reinterpret_cast<float*>(ptr);

    constexpr int kVectorsPerRow = HeadDim / 8;
    constexpr int kQueryVectors = GroupSize * kBlockM * kVectorsPerRow;
    auto* query_shared_vec = reinterpret_cast<uint4*>(query_shared);
    for (int vector_index = static_cast<int>(threadIdx.x);
         vector_index < kQueryVectors;
         vector_index += kThreads) {
        const int head_local = vector_index / (kBlockM * kVectorsPerRow);
        const int within_head = vector_index % (kBlockM * kVectorsPerRow);
        const int row = within_head / kVectorsPerRow;
        const int column_vector = within_head % kVectorsPerRow;
        const int query_position = query_begin + row;
        if (query_position < query_sequence) {
            const int query_head = first_query_head + head_local;
            const std::int64_t offset =
                ((static_cast<std::int64_t>(batch) * query_sequence
                  + query_position) * num_query_heads + query_head) * HeadDim
                + column_vector * 8;
            query_shared_vec[vector_index] =
                *reinterpret_cast<const uint4*>(query + offset);
        } else {
            query_shared_vec[vector_index] = make_uint4(0U, 0U, 0U, 0U);
        }
    }

    constexpr int kStateElements = GroupSize * kBlockM * HeadDim;
    for (int index = static_cast<int>(threadIdx.x); index < kStateElements; index += kThreads) {
        output_accumulator[index] = 0.0F;
    }
    for (int row = static_cast<int>(threadIdx.x);
         row < GroupSize * kBlockM;
         row += kThreads) {
        running_max[row] = -INFINITY;
        running_sum[row] = 0.0F;
    }
    __syncthreads();

    const int max_visible_key = causal
        ? min(key_value_sequence,
              query_position_offset + query_begin + kBlockM)
        : key_value_sequence;

    const int warp = static_cast<int>(threadIdx.x) / kWarpSize;
    const int lane = static_cast<int>(threadIdx.x) & (kWarpSize - 1);
    const int state_row_begin = warp * kBlockM;
    __half* warp_query = query_shared + warp * kBlockM * HeadDim;
    float* warp_scores = scores + warp * kBlockM * kN;
    __half* warp_probabilities = probabilities + warp * kBlockM * kN;
    float* warp_output = output_accumulator + warp * kBlockM * HeadDim;
    float* warp_mma_tile = mma_tiles + warp * kBlockM * 16;

    for (int key_begin = 0; key_begin < max_visible_key; key_begin += kN) {
        load_kv_rows_f16<HeadDim, kN>(
            key_shared,
            value_shared,
            key + kv_head_base,
            value + kv_head_base,
            key_begin,
            key_value_sequence,
            kv_row_stride);
        __syncthreads();

        using namespace nvcuda;
        #pragma unroll
        for (int subtile = 0; subtile < kN; subtile += 16) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __half,
                           wmma::row_major> query_fragment;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __half,
                           wmma::col_major> key_fragment;
            wmma::fragment<wmma::accumulator, 16, 16, 16, float>
                score_fragment;
            wmma::fill_fragment(score_fragment, 0.0F);
            #pragma unroll
            for (int dimension = 0; dimension < HeadDim; dimension += 16) {
                wmma::load_matrix_sync(
                    query_fragment, warp_query + dimension, HeadDim);
                wmma::load_matrix_sync(
                    key_fragment,
                    key_shared + subtile * HeadDim + dimension,
                    HeadDim);
                wmma::mma_sync(
                    score_fragment,
                    query_fragment,
                    key_fragment,
                    score_fragment);
            }
            wmma::store_matrix_sync(
                warp_scores + subtile,
                score_fragment,
                kN,
                wmma::mem_row_major);
        }
        __syncwarp();

        #pragma unroll
        for (int local_row = 0; local_row < kBlockM; ++local_row) {
            const int query_position = query_begin + local_row;
            const int key_position = key_begin + lane;
            float score = -INFINITY;
            const bool valid =
                query_position < query_sequence
                && key_position < key_value_sequence
                && (!causal
                    || key_position <= query_position_offset + query_position);
            if (valid) {
                score = warp_scores[local_row * kN + lane] * scale;
            }

            float tile_max = warp_reduce_max(score);
            tile_max = __shfl_sync(0xFFFFFFFFU, tile_max, 0);
            const int state_row = state_row_begin + local_row;
            const float old_max = running_max[state_row];
            const float new_max = fmaxf(old_max, tile_max);
            const float old_scale = old_max == -INFINITY
                ? 0.0F
                : __expf(old_max - new_max);
            const float probability = valid ? __expf(score - new_max) : 0.0F;
            float tile_sum = warp_reduce_sum(probability);
            tile_sum = __shfl_sync(0xFFFFFFFFU, tile_sum, 0);

            warp_probabilities[local_row * kN + lane] =
                __float2half_rn(probability);

            for (int dimension = lane;
                 dimension < HeadDim;
                 dimension += kWarpSize) {
                warp_output[local_row * HeadDim + dimension] *= old_scale;
            }
            if (lane == 0) {
                running_max[state_row] = new_max;
                running_sum[state_row] =
                    running_sum[state_row] * old_scale + tile_sum;
            }
        }
        __syncwarp();

        // Accumulate every 16-row probability subtile in the WMMA fragment
        // before spilling it once, instead of round-tripping each partial
        // result through shared memory.
        #pragma unroll
        for (int dimension = 0; dimension < HeadDim; dimension += 16) {
            wmma::fragment<wmma::accumulator, 16, 16, 16, float>
                output_fragment;
            wmma::fill_fragment(output_fragment, 0.0F);
            #pragma unroll
            for (int subtile = 0; subtile < kN; subtile += 16) {
                wmma::fragment<wmma::matrix_a, 16, 16, 16, __half,
                               wmma::row_major> probability_fragment;
                wmma::fragment<wmma::matrix_b, 16, 16, 16, __half,
                               wmma::row_major> value_fragment;
                wmma::load_matrix_sync(
                    probability_fragment,
                    warp_probabilities + subtile,
                    kN);
                wmma::load_matrix_sync(
                    value_fragment,
                    value_shared + subtile * HeadDim + dimension,
                    HeadDim);
                wmma::mma_sync(
                    output_fragment,
                    probability_fragment,
                    value_fragment,
                    output_fragment);
            }
            wmma::store_matrix_sync(
                warp_mma_tile,
                output_fragment,
                16,
                wmma::mem_row_major);
            __syncwarp();
            for (int index = lane; index < kBlockM * 16; index += kWarpSize) {
                const int local_row = index >> 4;
                const int column = index & 15;
                warp_output[local_row * HeadDim + dimension + column] +=
                    warp_mma_tile[index];
            }
            __syncwarp();
        }
        __syncthreads();
    }

    constexpr int kOutputElements = GroupSize * kBlockM * HeadDim;
    for (int index = static_cast<int>(threadIdx.x); index < kOutputElements; index += kThreads) {
        const int head_local = index / (kBlockM * HeadDim);
        const int within_head = index % (kBlockM * HeadDim);
        const int row = within_head / HeadDim;
        const int dimension = within_head % HeadDim;
        const int query_position = query_begin + row;
        if (query_position < query_sequence) {
            const int state_row = head_local * kBlockM + row;
            const float denominator = running_sum[state_row];
            const float result = denominator > 0.0F
                ? output_accumulator[index] / denominator
                : 0.0F;
            const int query_head = first_query_head + head_local;
            const std::int64_t output_offset =
                ((static_cast<std::int64_t>(batch) * query_sequence
                  + query_position) * num_query_heads + query_head) * HeadDim
                + dimension;
            output[output_offset] = __float2half_rn(result);
            if (logsumexp != nullptr && dimension == 0) {
                logsumexp[(static_cast<std::int64_t>(batch) * query_sequence
                           + query_position) * num_query_heads + query_head] =
                    denominator > 0.0F ? running_max[state_row] + __logf(denominator)
                                       : -INFINITY;
            }
        }
    }
}

/**
 * @brief Computes dynamic shared-memory usage of the grouped GQA kernel.
 *
 * The returned size includes Q, K, V, score, probability, float output,
 * temporary WMMA, running-maximum, and running-sum storage.
 */
template <int HeadDim, int GroupSize>
constexpr std::size_t grouped_f16_shared_bytes() {
    constexpr int kBlockM = 16;
    constexpr int kN = kGroupedBlockN;
    return static_cast<std::size_t>(GroupSize) * kBlockM * HeadDim * sizeof(__half)
         + static_cast<std::size_t>(kN) * HeadDim * sizeof(__half)
         + static_cast<std::size_t>(kN) * HeadDim * sizeof(__half)
         + static_cast<std::size_t>(GroupSize) * kBlockM * kN * sizeof(float)
         + static_cast<std::size_t>(GroupSize) * kBlockM * kN * sizeof(__half)
         + static_cast<std::size_t>(GroupSize) * kBlockM * HeadDim * sizeof(float)
         + static_cast<std::size_t>(GroupSize) * kBlockM * 16 * sizeof(float)
         + static_cast<std::size_t>(GroupSize) * kBlockM * sizeof(float)
         + static_cast<std::size_t>(GroupSize) * kBlockM * sizeof(float);
}

/**
 * @brief Launches the grouped D=128, 4:1 GQA specialization.
 *
 * Configures opt-in dynamic shared memory when the specialization exceeds the
 * legacy 48 KiB per-block limit. Launch errors are reported immediately through
 * CUDA_CHECK; execution remains asynchronous with respect to the host.
 */
template <int HeadDim, int GroupSize>
void launch_grouped_f16(
    Tensor& output,
    float* logsumexp,
    const Tensor& query,
    const Tensor& key,
    const Tensor& value,
    std::size_t batch_size,
    std::size_t query_sequence,
    std::size_t key_value_sequence,
    float scale,
    const FlashAttentionOptions& options,
    cudaStream_t stream
) {
    constexpr int kBlockM = 16;
    constexpr int kThreads = GroupSize * kWarpSize;
    const std::size_t query_tiles =
        (query_sequence + static_cast<std::size_t>(kBlockM) - 1U)
        / static_cast<std::size_t>(kBlockM);
    const std::size_t blocks =
        batch_size * options.num_kv_heads * query_tiles;
    if (blocks == 0U) {
        return;
    }
    if (blocks
        > static_cast<std::size_t>(std::numeric_limits<unsigned int>::max())) {
        throw std::overflow_error(
            "flash_gqa_attention_forward: CUDA grid is too large");
    }

    constexpr std::size_t shared_bytes =
        grouped_f16_shared_bytes<HeadDim, GroupSize>();
    if constexpr (shared_bytes > 48U * 1024U) {
        CUDA_CHECK(cudaFuncSetAttribute(
            flash_gqa_grouped_f16_tensor_core_kernel<HeadDim, GroupSize>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            static_cast<int>(shared_bytes)));
    }

    flash_gqa_grouped_f16_tensor_core_kernel<HeadDim, GroupSize>
        <<<static_cast<unsigned int>(blocks),
           kThreads,
           shared_bytes,
           stream>>>(
            static_cast<__half*>(output.raw_data()), logsumexp,
            static_cast<const __half*>(query.raw_data()),
            static_cast<const __half*>(key.raw_data()),
            static_cast<const __half*>(value.raw_data()),
            static_cast<int>(batch_size),
            static_cast<int>(query_sequence),
            static_cast<int>(key_value_sequence),
            static_cast<int>(options.num_query_heads),
            static_cast<int>(options.num_kv_heads),
            scale,
            options.causal,
            static_cast<int>(options.query_position_offset));
    CUDA_CHECK(cudaGetLastError());
}

/**
 * @brief Computes dynamic shared-memory usage of the generic tiled kernel.
 *
 * The returned size includes the query tile, one K/V tile, intermediate scores
 * and probabilities, the FP16 online output accumulator, WMMA scratch space,
 * and softmax statistics.
 *
 * @note With kBlockN == 32, HeadDim=128 at the dispatcher's BlockM=64 needs
 *       ~64.5 KiB (66048 B), which fits Volta's 96 KiB and Ampere/Hopper's
 *       larger per-block opt-in limits but exceeds Turing's (CC 7.5) 64 KiB
 *       per-block ceiling. That combination is only reached for GQA ratios
 *       other than 4:1 (the 4:1 case uses launch_grouped_f16 instead). The
 *       cudaFuncSetAttribute call below surfaces this as a CUDA_CHECK error
 *       at launch time rather than corrupting results, but if Turing must be
 *       supported for non-4:1 GQA with head_dim=128, use a smaller BlockM
 *       (e.g. 32) for that specialization instead.
 */
template <int HeadDim, int BlockM>
constexpr std::size_t tiled_f16_shared_bytes() {
    constexpr int kWarps = BlockM / 16;
    return static_cast<std::size_t>(BlockM) * HeadDim * sizeof(__half)
         + static_cast<std::size_t>(kBlockN) * HeadDim * sizeof(__half)
         + static_cast<std::size_t>(kBlockN) * HeadDim * sizeof(__half)
         + static_cast<std::size_t>(BlockM) * kBlockN * sizeof(float)
         + static_cast<std::size_t>(BlockM) * kBlockN * sizeof(__half)
         + static_cast<std::size_t>(BlockM) * HeadDim * sizeof(__half)
         + static_cast<std::size_t>(kWarps) * 16 * 16 * sizeof(float)
         + static_cast<std::size_t>(BlockM) * sizeof(float)
         + static_cast<std::size_t>(BlockM) * sizeof(float);
}

/**
 * @brief Launches the generic per-query-head Tensor Core kernel.
 *
 * One CUDA block is created for every (batch, query head, query tile) tuple.
 * The function only validates launch-size overflow; tensor contracts are checked
 * by flash_gqa_attention_forward before dispatch.
 */
template <int HeadDim, int BlockM>
void launch_tiled_f16(
    Tensor& output,
    float* logsumexp,
    const Tensor& query,
    const Tensor& key,
    const Tensor& value,
    std::size_t batch_size,
    std::size_t query_sequence,
    std::size_t key_value_sequence,
    float scale,
    const FlashAttentionOptions& options,
    cudaStream_t stream
) {
    constexpr int kThreads = (BlockM / 16) * kWarpSize;
    const std::size_t query_tiles =
        (query_sequence + static_cast<std::size_t>(BlockM) - 1U)
        / static_cast<std::size_t>(BlockM);
    const std::size_t blocks =
        batch_size * options.num_query_heads * query_tiles;
    if (blocks == 0U) {
        return;
    }
    if (blocks
        > static_cast<std::size_t>(std::numeric_limits<unsigned int>::max())) {
        throw std::overflow_error(
            "flash_gqa_attention_forward: CUDA grid is too large");
    }

    constexpr std::size_t shared_bytes =
        tiled_f16_shared_bytes<HeadDim, BlockM>();
    if constexpr (shared_bytes > 48U * 1024U) {
        CUDA_CHECK(cudaFuncSetAttribute(
            flash_gqa_f16_tensor_core_kernel<HeadDim, BlockM>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            static_cast<int>(shared_bytes)));
    }

    flash_gqa_f16_tensor_core_kernel<HeadDim, BlockM>
        <<<static_cast<unsigned int>(blocks),
           kThreads,
           shared_bytes,
           stream>>>(
            static_cast<__half*>(output.raw_data()), logsumexp,
            static_cast<const __half*>(query.raw_data()),
            static_cast<const __half*>(key.raw_data()),
            static_cast<const __half*>(value.raw_data()),
            static_cast<int>(batch_size),
            static_cast<int>(query_sequence),
            static_cast<int>(key_value_sequence),
            static_cast<int>(options.num_query_heads),
            static_cast<int>(options.num_kv_heads),
            scale,
            options.causal,
            static_cast<int>(options.query_position_offset));
    CUDA_CHECK(cudaGetLastError());
}

/**
 * @brief Selects the compile-time kernel specialization for a validated request.
 *
 * D=128 with exactly four query heads per K/V head uses the grouped kernel.
 * All other supported configurations use the generic per-head kernel.
 */
void launch_tiled_f16_dispatch(
    Tensor& output,
    float* logsumexp,
    const Tensor& query,
    const Tensor& key,
    const Tensor& value,
    std::size_t batch_size,
    std::size_t query_sequence,
    std::size_t key_value_sequence,
    float scale,
    const FlashAttentionOptions& options,
    cudaStream_t stream
) {
    switch (options.head_dim) {
        case 32:
            // Smaller BlockM improves occupancy on RTX 3050-class GPUs.
            launch_tiled_f16<32, 32>(
                output, logsumexp, query, key, value,
                batch_size, query_sequence, key_value_sequence,
                scale, options, stream);
            break;
        case 64:
            launch_tiled_f16<64, 32>(
                output, logsumexp, query, key, value,
                batch_size, query_sequence, key_value_sequence,
                scale, options, stream);
            break;
        case 128:
            if (options.num_query_heads / options.num_kv_heads == 4U) {
                // Preferred path: one CTA owns the whole GQA group of 4 heads.
                // K/V loaded once and reused by 4 warps.
                launch_grouped_f16<128, 4>(
                    output, logsumexp, query, key, value,
                    batch_size, query_sequence, key_value_sequence,
                    scale, options, stream);
            } else {
                launch_tiled_f16<128, 32>(
                    output, logsumexp, query, key, value,
                    batch_size, query_sequence, key_value_sequence,
                    scale, options, stream);
            }
            break;
        default:
            throw std::invalid_argument(
                "flash_gqa_attention_forward: Tensor Core path supports "
                "head_dim 32, 64, or 128");
    }
}

/// Returns true when a compiled Tensor Core specialization exists for head_dim.
bool is_supported_head_dim(std::size_t head_dim) {
    return head_dim == 32U || head_dim == 64U || head_dim == 128U;
}

/** @brief Verifies the rank-four tensor contract used by all attention operands. */
void validate_tensor(const Tensor& tensor, const char* name) {
    if (tensor.shape().size() != 4U) {
        throw std::invalid_argument(
            std::string("flash_gqa_attention_forward: ")
            + name + " must have rank 4");
    }
}

/**
 * @brief Validates head topology and requirements of the implemented backend.
 *
 * This checks logical configuration only. Shapes, dtype, and causal sequence
 * bounds are validated separately by the public entry point.
 */
void validate_options(const FlashAttentionOptions& options) {
    if (options.num_query_heads == 0U
        || options.num_kv_heads == 0U
        || options.head_dim == 0U) {
        throw std::invalid_argument(
            "flash_gqa_attention_forward: invalid head configuration");
    }
    if (options.num_query_heads % options.num_kv_heads != 0U) {
        throw std::invalid_argument(
            "flash_gqa_attention_forward: num_query_heads must be divisible "
            "by num_kv_heads");
    }
    if (!is_supported_head_dim(options.head_dim)) {
        throw std::invalid_argument(
            "flash_gqa_attention_forward: Tensor Core kernel supports "
            "head_dim 32, 64, or 128");
    }
    if (!options.use_tensor_cores) {
        throw std::invalid_argument(
            "flash_gqa_attention_forward: this implementation is the tiled "
            "Tensor Core path; use_tensor_cores must be true");
    }
}

/**
 * @brief Ensures every value converted to int by a kernel launch is representable.
 *
 * Tensor address arithmetic itself uses 64-bit offsets; sequence dimensions and
 * launch parameters are intentionally kept in 32-bit registers inside kernels.
 */
void validate_kernel_integer_ranges(
    std::size_t batch_size,
    std::size_t query_sequence,
    std::size_t key_value_sequence,
    const FlashAttentionOptions& options
) {
    constexpr auto kIntMax =
        static_cast<std::size_t>(std::numeric_limits<int>::max());
    if (batch_size > kIntMax
        || query_sequence > kIntMax
        || key_value_sequence > kIntMax
        || options.num_query_heads > kIntMax
        || options.num_kv_heads > kIntMax
        || options.query_position_offset > kIntMax) {
        throw std::overflow_error(
            "flash_gqa_attention_forward: tensor dimensions exceed the "
            "optimized kernel's 32-bit indexing range");
    }
}

} // namespace

/**
 * @brief Executes the FP16 forward pass of scaled dot-product GQA attention.
 *
 * The operation is equivalent to:
 * @f[
 * output = softmax(mask(query key^T * scale)) value
 * @f]
 * but computes attention in sequence tiles and never stores the complete score
 * or probability matrix in global memory.
 *
 * Tensor layouts:
 * @code
 * query:  [B, Q, Hq,  D]
 * key:    [B, K, Hkv, D]
 * value:  [B, K, Hkv, D]
 * output: [B, Q, Hq,  D]
 * @endcode
 *
 * Query head h uses K/V head floor(h / (Hq / Hkv)). For causal attention, a
 * query at local index q is allowed to read keys up to the absolute position
 * query_position_offset + q. Consequently, cached decoding normally uses
 * query_position_offset = K - Q.
 *
 * @param output Preallocated contiguous FP16 output tensor. Its shape must equal
 *        the query shape. The tensor is written asynchronously on @p stream.
 * @param query Contiguous FP16 query tensor [B, Q, Hq, D].
 * @param key Contiguous FP16 key tensor [B, K, Hkv, D].
 * @param value Contiguous FP16 value tensor with exactly the key shape.
 * @param stream CUDA stream used for the kernel launch.
 * @param options Attention topology, masking mode, score scale, and cache offset.
 *
 * @throws std::invalid_argument If rank, shape, dtype, head topology, supported
 *         head dimension, Tensor Core mode, or causal bounds are invalid.
 * @throws std::overflow_error If launch dimensions exceed 32-bit kernel limits
 *         or the one-dimensional CUDA grid limit.
 *
 * @note If options.attention_scale <= 0, the function uses 1/sqrt(head_dim).
 * @note The function checks launch errors but does not synchronize @p stream.
 * @note output must not alias query, key, or value.
 */
void flash_gqa_attention_forward(
    Tensor& output,
    const Tensor& query,
    const Tensor& key,
    const Tensor& value,
    cudaStream_t stream,
    const FlashAttentionOptions& options
) {
    validate_options(options);
    validate_tensor(output, "output");
    validate_tensor(query, "query");
    validate_tensor(key, "key");
    validate_tensor(value, "value");

    const auto& query_shape = query.shape();
    const auto& key_shape = key.shape();
    const auto& value_shape = value.shape();
    const auto& output_shape = output.shape();

    const std::size_t batch_size = query_shape[0];
    const std::size_t query_sequence = query_shape[1];
    const std::size_t key_value_sequence = key_shape[1];

    if (query_shape[2] != options.num_query_heads
        || query_shape[3] != options.head_dim) {
        throw std::invalid_argument(
            "flash_gqa_attention_forward: query shape mismatch");
    }
    if (key_shape[0] != batch_size
        || key_shape[2] != options.num_kv_heads
        || key_shape[3] != options.head_dim) {
        throw std::invalid_argument(
            "flash_gqa_attention_forward: key shape mismatch");
    }
    if (value_shape[0] != batch_size
        || value_shape[1] != key_value_sequence
        || value_shape[2] != options.num_kv_heads
        || value_shape[3] != options.head_dim) {
        throw std::invalid_argument(
            "flash_gqa_attention_forward: value shape must match key");
    }
    if (output_shape != query_shape) {
        throw std::invalid_argument(
            "flash_gqa_attention_forward: output shape must match query");
    }
    if (query.dtype() != Dtype::F16
        || key.dtype() != Dtype::F16
        || value.dtype() != Dtype::F16
        || output.dtype() != Dtype::F16) {
        throw std::invalid_argument(
            "flash_gqa_attention_forward: optimized Tensor Core path "
            "currently requires F16 tensors");
    }
    if (options.causal
        && options.query_position_offset + query_sequence
            > key_value_sequence) {
        throw std::invalid_argument(
            "flash_gqa_attention_forward: causal query range exceeds K/V");
    }

    validate_kernel_integer_ranges(
        batch_size,
        query_sequence,
        key_value_sequence,
        options);

    const float scale = options.attention_scale > 0.0F
        ? options.attention_scale
        : 1.0F / std::sqrt(static_cast<float>(options.head_dim));

    launch_tiled_f16_dispatch(
        output,
        nullptr,
        query,
        key,
        value,
        batch_size,
        query_sequence,
        key_value_sequence,
        scale,
        options,
        stream);
}

/**
 * @brief Executes the FP16 GQA forward pass and stores row-wise log-sum-exp.
 *
 * This training-oriented variant performs the same tiled scaled dot-product
 * attention as flash_gqa_attention_forward(), while additionally persisting
 * one FP32 statistic for every attention row:
 * @f[
 * LSE = m + \log\left(\sum_j \exp(s_j - m)\right),
 * \qquad m = \max_j s_j.
 * @f]
 *
 * The saved statistic allows a backward kernel to reconstruct normalized
 * probabilities without materializing the complete attention matrix or
 * repeating the online-softmax statistics pass.
 *
 * Tensor layouts:
 * @code
 * query:     [B, Q, Hq,  D]  FP16
 * key:       [B, K, Hkv, D]  FP16
 * value:     [B, K, Hkv, D]  FP16
 * output:    [B, Q, Hq,  D]  FP16
 * logsumexp: [B, Q, Hq]      FP32
 * @endcode
 *
 * @param output Preallocated contiguous FP16 output tensor with the same shape
 *        as @p query. It is written asynchronously on @p stream.
 * @param logsumexp Preallocated contiguous FP32 tensor [B, Q, Hq] receiving
 *        one log-sum-exp value per batch, query position, and query head.
 * @param query Contiguous FP16 query tensor [B, Q, Hq, D].
 * @param key Contiguous FP16 key tensor [B, K, Hkv, D].
 * @param value Contiguous FP16 value tensor with exactly the same shape as
 *        @p key.
 * @param stream CUDA stream used for the asynchronous kernel launch.
 * @param options Attention topology, causal-mask configuration, score scale,
 *        Tensor Core selection, and query-position offset.
 *
 * @throws std::invalid_argument If tensor ranks, shapes, dtypes, head topology,
 *         supported head dimension, Tensor Core mode, or causal bounds are
 *         invalid.
 * @throws std::overflow_error If a tensor or launch dimension exceeds the
 *         optimized kernel's supported integer range or CUDA grid limit.
 *
 * @note If options.attention_scale <= 0, the scale defaults to
 *       1/sqrt(options.head_dim).
 * @note The function reports launch errors but does not synchronize @p stream.
 * @note @p output, @p logsumexp, @p query, @p key, and @p value must refer to
 *       valid device allocations for the entire asynchronous operation.
 *
 * @see flash_gqa_attention_forward()
 */
void flash_gqa_attention_forward_with_lse(
    Tensor& output, Tensor& logsumexp, const Tensor& query, const Tensor& key,
    const Tensor& value, cudaStream_t stream, const FlashAttentionOptions& options
) {
    validate_options(options);
    validate_tensor(output, "output");
    validate_tensor(query, "query");
    validate_tensor(key, "key");
    validate_tensor(value, "value");

    const auto& q = query.shape();
    const auto& k = key.shape();
    if (q.size() != 4U || k.size() != 4U || value.shape() != k || output.shape() != q
        || logsumexp.dim() != 3 || logsumexp.size(0) != q[0]
        || logsumexp.size(1) != q[1] || logsumexp.size(2) != q[2]
        || q[2] != options.num_query_heads || k[2] != options.num_kv_heads
        || q[3] != options.head_dim || k[3] != options.head_dim
        || q[0] != k[0] || query.dtype() != Dtype::F16 || key.dtype() != Dtype::F16
        || value.dtype() != Dtype::F16 || output.dtype() != Dtype::F16
        || logsumexp.dtype() != Dtype::F32) {
        throw std::invalid_argument("flash_gqa_attention_forward_with_lse: invalid tensor contract");
    }
    if (options.causal && options.query_position_offset + q[1] > k[1]) {
        throw std::invalid_argument("flash_gqa_attention_forward_with_lse: causal query range exceeds K/V");
    }
    validate_kernel_integer_ranges(q[0], q[1], k[1], options);
    const float scale = options.attention_scale > 0.0F ? options.attention_scale
        : 1.0F / std::sqrt(static_cast<float>(options.head_dim));
    launch_tiled_f16_dispatch(output, static_cast<float*>(logsumexp.raw_data()), query, key, value,
        q[0], q[1], k[1], scale, options, stream);
}
