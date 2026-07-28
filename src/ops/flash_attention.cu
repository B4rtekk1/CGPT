#include "ops/flash_attention.h"

#include "core/cuda_check.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>

#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace {

constexpr int kWarpSize = 32;
constexpr int kBlockN = 16;
constexpr int kGroupedBlockN = 32;

__device__ __forceinline__ float warp_reduce_sum(float value) {
    #pragma unroll
    for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(0xFFFFFFFFU, value, offset);
    }
    return value;
}

__device__ __forceinline__ float warp_reduce_max(float value) {
    #pragma unroll
    for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
        value = fmaxf(value, __shfl_down_sync(0xFFFFFFFFU, value, offset));
    }
    return value;
}

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

    for (int vector_index = threadIdx.x;
         vector_index < kVectorCount;
         vector_index += blockDim.x) {
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

    for (int vector_index = threadIdx.x;
         vector_index < kVectorCount;
         vector_index += blockDim.x) {
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

template <int HeadDim, int BlockM>
__global__ void flash_gqa_f16_tensor_core_kernel(
    __half* __restrict__ output,
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
    unsigned char* ptr = shared_raw;

    auto* query_shared = reinterpret_cast<__half*>(ptr);
    ptr += BlockM * HeadDim * sizeof(__half);

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

    for (int index = threadIdx.x;
         index < BlockM * HeadDim;
         index += kThreads) {
        output_accumulator[index] = __float2half_rn(0.0F);
    }
    for (int row = threadIdx.x; row < BlockM; row += kThreads) {
        running_max[row] = -INFINITY;
        running_sum[row] = 0.0F;
    }
    __syncthreads();

    const int max_visible_key = causal
        ? min(key_value_sequence,
              query_position_offset + query_begin + BlockM)
        : key_value_sequence;

    const int warp = threadIdx.x / kWarpSize;
    const int lane = threadIdx.x & (kWarpSize - 1);
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
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __half,
                           wmma::row_major> query_fragment;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __half,
                           wmma::col_major> key_fragment;
            wmma::fragment<wmma::accumulator, 16, 16, 16, float>
                score_fragment;
            wmma::fill_fragment(score_fragment, 0.0F);

            const __half* warp_query =
                query_shared + warp_row_begin * HeadDim;
            #pragma unroll
            for (int dimension = 0;
                 dimension < HeadDim;
                 dimension += 16) {
                wmma::load_matrix_sync(
                    query_fragment, warp_query + dimension, HeadDim);
                wmma::load_matrix_sync(
                    key_fragment, key_shared + dimension, HeadDim);
                wmma::mma_sync(
                    score_fragment,
                    query_fragment,
                    key_fragment,
                    score_fragment);
            }
            wmma::store_matrix_sync(
                warp_scores,
                score_fragment,
                kBlockN,
                wmma::mem_row_major);
        }
        __syncwarp();

        #pragma unroll
        for (int local_row = 0; local_row < 16; ++local_row) {
            const int row = warp_row_begin + local_row;
            const int query_position = query_begin + row;

            float score = -INFINITY;
            if (lane < kBlockN) {
                const int key_position = key_begin + lane;
                const bool valid =
                    query_position < query_sequence
                    && key_position < key_value_sequence
                    && (!causal
                        || key_position
                            <= query_position_offset + query_position);
                if (valid) {
                    score = warp_scores[local_row * kBlockN + lane] * scale;
                }
            }

            float tile_max = warp_reduce_max(score);
            tile_max = __shfl_sync(0xFFFFFFFFU, tile_max, 0);

            const float old_max = running_max[row];
            const float new_max = fmaxf(old_max, tile_max);
            const float old_scale = old_max == -INFINITY
                ? 0.0F
                : __expf(old_max - new_max);
            const float probability =
                lane < kBlockN && score != -INFINITY
                ? __expf(score - new_max)
                : 0.0F;

            float tile_sum = warp_reduce_sum(probability);
            tile_sum = __shfl_sync(0xFFFFFFFFU, tile_sum, 0);

            if (lane < kBlockN) {
                warp_probabilities[local_row * kBlockN + lane] =
                    __float2half_rn(probability);
            }

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

        // P @ V. Again, every warp processes its own 16 query rows.
        {
            using namespace nvcuda;
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __half,
                           wmma::row_major> probability_fragment;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __half,
                           wmma::row_major> value_fragment;
            wmma::fragment<wmma::accumulator, 16, 16, 16, float>
                output_fragment;

            wmma::load_matrix_sync(
                probability_fragment, warp_probabilities, kBlockN);

            #pragma unroll
            for (int dimension = 0;
                 dimension < HeadDim;
                 dimension += 16) {
                wmma::fill_fragment(output_fragment, 0.0F);
                wmma::load_matrix_sync(
                    value_fragment, value_shared + dimension, HeadDim);
                wmma::mma_sync(
                    output_fragment,
                    probability_fragment,
                    value_fragment,
                    output_fragment);
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

    for (int index = threadIdx.x;
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
        }
    }
}


template <int HeadDim, int GroupSize>
__global__ __launch_bounds__(128, 1)
void flash_gqa_grouped_f16_tensor_core_kernel(
    __half* __restrict__ output,
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
    unsigned char* ptr = shared_raw;

    auto* query_shared = reinterpret_cast<__half*>(ptr);
    ptr += GroupSize * kBlockM * HeadDim * sizeof(__half);
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
    for (int vector_index = threadIdx.x;
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
    for (int index = threadIdx.x; index < kStateElements; index += kThreads) {
        output_accumulator[index] = 0.0F;
    }
    for (int row = threadIdx.x;
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

    const int warp = threadIdx.x / kWarpSize;
    const int lane = threadIdx.x & (kWarpSize - 1);
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

        #pragma unroll
        for (int subtile = 0; subtile < kN; subtile += 16) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __half,
                           wmma::row_major> probability_fragment;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __half,
                           wmma::row_major> value_fragment;
            wmma::fragment<wmma::accumulator, 16, 16, 16, float>
                output_fragment;
            wmma::load_matrix_sync(
                probability_fragment,
                warp_probabilities + subtile,
                kN);

            #pragma unroll
            for (int dimension = 0; dimension < HeadDim; dimension += 16) {
                wmma::fill_fragment(output_fragment, 0.0F);
                wmma::load_matrix_sync(
                    value_fragment,
                    value_shared + subtile * HeadDim + dimension,
                    HeadDim);
                wmma::mma_sync(
                    output_fragment,
                    probability_fragment,
                    value_fragment,
                    output_fragment);
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
        }
        __syncthreads();
    }

    constexpr int kOutputElements = GroupSize * kBlockM * HeadDim;
    for (int index = threadIdx.x; index < kOutputElements; index += kThreads) {
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
        }
    }
}

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

template <int HeadDim, int GroupSize>
void launch_grouped_f16(
    Tensor& output,
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
            static_cast<__half*>(output.raw_data()),
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

template <int HeadDim, int BlockM>
void launch_tiled_f16(
    Tensor& output,
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
            static_cast<__half*>(output.raw_data()),
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

void launch_tiled_f16_dispatch(
    Tensor& output,
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
            launch_tiled_f16<32, 64>(
                output, query, key, value,
                batch_size, query_sequence, key_value_sequence,
                scale, options, stream);
            break;
        case 64:
            launch_tiled_f16<64, 64>(
                output, query, key, value,
                batch_size, query_sequence, key_value_sequence,
                scale, options, stream);
            break;
        case 128:
            if (options.num_query_heads / options.num_kv_heads == 4U) {
                // One CTA owns all four query heads mapped to a KV head.
                // The 16x128 K/V tiles are loaded once and reused by 4 warps.
                launch_grouped_f16<128, 4>(
                    output, query, key, value,
                    batch_size, query_sequence, key_value_sequence,
                    scale, options, stream);
            } else {
                launch_tiled_f16<128, 64>(
                    output, query, key, value,
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

bool is_supported_head_dim(std::size_t head_dim) {
    return head_dim == 32U || head_dim == 64U || head_dim == 128U;
}

void validate_tensor(const Tensor& tensor, const char* name) {
    if (tensor.shape().size() != 4U) {
        throw std::invalid_argument(
            std::string("flash_gqa_attention_forward: ")
            + name + " must have rank 4");
    }
}

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

void validate_kernel_integer_ranges(
    std::size_t batch_size,
    std::size_t query_sequence,
    std::size_t key_value_sequence,
    const FlashAttentionOptions& options
) {
    constexpr std::size_t kIntMax =
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
