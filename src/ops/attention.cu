/**
 * @file attention_ampere_fa2_optimized.cu
 * @brief Ampere-specialized FP16 FlashAttention-2 style forward kernel for GQA.
 *
 * Fast path:
 * - native mma.sync.aligned.m16n8k16 Tensor Core instructions,
 * - ldmatrix shared-memory fragment loads,
 * - double-buffered cp.async K/V pipeline,
 * - four packed query heads per KV head for the common GQA ratio 4:1,
 * - persistent query fragments in registers on the packed path,
 * - FP32 online-softmax state and FP32 output accumulation,
 * - compile-time causal and LSE specializations,
 * - padded shared-memory strides to reduce bank conflicts.
 *
 * Tensor layouts:
 *   query:  [B, Q, Hq,  D]
 *   key:    [B, K, Hkv, D]
 *   value:  [B, K, Hkv, D]
 *   output: [B, Q, Hq,  D]
 */
#include "ops/attention.h"

#include "core/cuda_check.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace {

constexpr int kWarpSize = 32;
constexpr int kBlockN = 32;
constexpr int kRowsPerWarp = 16;
constexpr int kPackedQueryHeads = 4;
constexpr int kPackedThreads = kPackedQueryHeads * kWarpSize;
constexpr int kProbStride = kBlockN + 8;
constexpr float kLog2E = 1.4426950408889634074F;
constexpr float kLn2 = 0.6931471805599453094F;

/** Adds one 16-byte padding vector to every FP16 shared-memory row. */
template <int HeadDim>
inline constexpr int kSharedStride = HeadDim + 8;


__device__ __forceinline__ float fast_max(float a, float b) {
    return a > b ? a : b;
}

__device__ __forceinline__ unsigned shared_address(const void* pointer) {
    return static_cast<unsigned>(__cvta_generic_to_shared(pointer));
}

__device__ __forceinline__ void cp_async_wait_all() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("cp.async.wait_group 0;\n" ::: "memory");
#endif
}

/**
 * Copies a row-major FP16 tile from global memory to padded shared memory.
 * Invalid rows are zero-filled by cp.async itself.
 */
template <int Rows, int HeadDim, int DestinationStride>
__device__ __forceinline__ void load_rows_f16_async(
    __half* __restrict__ destination,
    const __half* __restrict__ source,
    int first_row,
    int valid_rows,
    std::int64_t source_row_stride
) {
    static_assert(HeadDim % 8 == 0);
    static_assert(DestinationStride >= HeadDim);
    static_assert(DestinationStride % 8 == 0);

    constexpr int kVectorsPerRow = HeadDim / 8;
    constexpr int kVectorCount = Rows * kVectorsPerRow;

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    for (int vector_index = static_cast<int>(threadIdx.x);
         vector_index < kVectorCount;
         vector_index += static_cast<int>(blockDim.x)) {
        const int row = vector_index / kVectorsPerRow;
        const int column_vector = vector_index % kVectorsPerRow;
        const int source_row = first_row + row;
        const bool valid = source_row < valid_rows;

        const std::int64_t source_offset = valid
            ? static_cast<std::int64_t>(source_row) * source_row_stride
                + column_vector * 8
            : 0;
        const __half* source_pointer = source + source_offset;
        __half* destination_pointer =
            destination + row * DestinationStride + column_vector * 8;

        const unsigned destination_address = shared_address(destination_pointer);
        const int source_bytes = valid ? 16 : 0;
        asm volatile(
            "cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::
            "r"(destination_address), "l"(source_pointer), "r"(source_bytes));
    }
    asm volatile("cp.async.commit_group;\n" ::);
#else
    for (int vector_index = static_cast<int>(threadIdx.x);
         vector_index < kVectorCount;
         vector_index += static_cast<int>(blockDim.x)) {
        const int row = vector_index / kVectorsPerRow;
        const int column_vector = vector_index % kVectorsPerRow;
        const int source_row = first_row + row;
        auto* destination_vector = reinterpret_cast<uint4*>(
            destination + row * DestinationStride + column_vector * 8);
        if (source_row < valid_rows) {
            const __half* source_pointer =
                source + static_cast<std::int64_t>(source_row) * source_row_stride
                + column_vector * 8;
            *destination_vector = *reinterpret_cast<const uint4*>(source_pointer);
        } else {
            *destination_vector = make_uint4(0U, 0U, 0U, 0U);
        }
    }
#endif
}

/** Copies matching K and V tiles into one stage of the shared-memory pipeline. */
template <int HeadDim, int DestinationStride>
__device__ __forceinline__ void load_kv_f16_async(
    __half* __restrict__ key_destination,
    __half* __restrict__ value_destination,
    const __half* __restrict__ key_source,
    const __half* __restrict__ value_source,
    int first_row,
    int valid_rows,
    std::int64_t source_row_stride
) {
    static_assert(HeadDim % 8 == 0);
    constexpr int kVectorsPerRow = HeadDim / 8;
    constexpr int kVectorCount = kBlockN * kVectorsPerRow;

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    for (int vector_index = static_cast<int>(threadIdx.x);
         vector_index < kVectorCount;
         vector_index += static_cast<int>(blockDim.x)) {
        const int row = vector_index / kVectorsPerRow;
        const int column_vector = vector_index % kVectorsPerRow;
        const int source_row = first_row + row;
        const bool valid = source_row < valid_rows;

        const std::int64_t source_offset = valid
            ? static_cast<std::int64_t>(source_row) * source_row_stride
                + column_vector * 8
            : 0;
        const __half* key_pointer = key_source + source_offset;
        const __half* value_pointer = value_source + source_offset;
        __half* key_shared_pointer =
            key_destination + row * DestinationStride + column_vector * 8;
        __half* value_shared_pointer =
            value_destination + row * DestinationStride + column_vector * 8;

        const unsigned key_address = shared_address(key_shared_pointer);
        const unsigned value_address = shared_address(value_shared_pointer);
        const int source_bytes = valid ? 16 : 0;

        asm volatile(
            "cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::
            "r"(key_address), "l"(key_pointer), "r"(source_bytes));
        asm volatile(
            "cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::
            "r"(value_address), "l"(value_pointer), "r"(source_bytes));
    }
    asm volatile("cp.async.commit_group;\n" ::);
#else
    for (int vector_index = static_cast<int>(threadIdx.x);
         vector_index < kVectorCount;
         vector_index += static_cast<int>(blockDim.x)) {
        const int row = vector_index / kVectorsPerRow;
        const int column_vector = vector_index % kVectorsPerRow;
        const int source_row = first_row + row;
        auto* key_destination_vector = reinterpret_cast<uint4*>(
            key_destination + row * DestinationStride + column_vector * 8);
        auto* value_destination_vector = reinterpret_cast<uint4*>(
            value_destination + row * DestinationStride + column_vector * 8);
        if (source_row < valid_rows) {
            const std::int64_t source_offset =
                static_cast<std::int64_t>(source_row) * source_row_stride
                + column_vector * 8;
            *key_destination_vector =
                *reinterpret_cast<const uint4*>(key_source + source_offset);
            *value_destination_vector =
                *reinterpret_cast<const uint4*>(value_source + source_offset);
        } else {
            const uint4 zero = make_uint4(0U, 0U, 0U, 0U);
            *key_destination_vector = zero;
            *value_destination_vector = zero;
        }
    }
#endif
}

struct Mma16816Accumulator {
    float x[4];
};

__device__ __forceinline__ void clear_accumulator(
    Mma16816Accumulator& accumulator
) {
    accumulator.x[0] = 0.0F;
    accumulator.x[1] = 0.0F;
    accumulator.x[2] = 0.0F;
    accumulator.x[3] = 0.0F;
}

/** Loads a row-major 16x16 A operand. */
__device__ __forceinline__ void ldmatrix_a_m16k16_row(
    unsigned (&fragment)[4],
    const __half* base,
    int leading_dimension,
    int lane
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
    fragment[0] = fragment[1] = fragment[2] = fragment[3] = 0U;
#endif
}

/** Loads a row-major K tile as the column-major MMA B operand. */
__device__ __forceinline__ void ldmatrix_b_k16n8_col(
    unsigned (&fragment)[2],
    const __half* base,
    int leading_dimension,
    int lane
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
    fragment[0] = fragment[1] = 0U;
#endif
}

/** Loads and transposes a row-major 16x8 V tile for the MMA B operand. */
__device__ __forceinline__ void ldmatrix_b_k16n8_row_transposed(
    unsigned (&fragment)[2],
    const __half* base,
    int leading_dimension,
    int lane
) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    const int source_row = lane & 15;
    const unsigned address = shared_address(
        base + source_row * leading_dimension);
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 "
        "{%0, %1}, [%2];\n"
        : "=r"(fragment[0]), "=r"(fragment[1])
        : "r"(address));
#else
    fragment[0] = fragment[1] = 0U;
#endif
}

__device__ __forceinline__ void mma_m16n8k16(
    Mma16816Accumulator& accumulator,
    const unsigned (&a)[4],
    const unsigned (&b)[2]
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

struct ScoreTile16x32 {
    Mma16816Accumulator part[4];
};

__device__ __forceinline__ float subgroup4_max(float value) {
    value = fast_max(value, __shfl_xor_sync(0xFFFFFFFFU, value, 1, 4));
    value = fast_max(value, __shfl_xor_sync(0xFFFFFFFFU, value, 2, 4));
    return value;
}

__device__ __forceinline__ float subgroup4_sum(float value) {
    value += __shfl_xor_sync(0xFFFFFFFFU, value, 1, 4);
    value += __shfl_xor_sync(0xFFFFFFFFU, value, 2, 4);
    return value;
}

template <int HeadDim>
__device__ __forceinline__ void compute_qk_from_shared(
    ScoreTile16x32& scores,
    const __half* query,
    const __half* key,
    int shared_stride,
    int lane
) {
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        clear_accumulator(scores.part[i]);
    }

    #pragma unroll
    for (int dimension = 0; dimension < HeadDim; dimension += 16) {
        unsigned query_fragment[4];
        unsigned key_fragment0[2];
        unsigned key_fragment1[2];
        unsigned key_fragment2[2];
        unsigned key_fragment3[2];

        ldmatrix_a_m16k16_row(
            query_fragment, query + dimension, shared_stride, lane);
        ldmatrix_b_k16n8_col(
            key_fragment0, key + dimension, shared_stride, lane);
        ldmatrix_b_k16n8_col(
            key_fragment1, key + 8 * shared_stride + dimension,
            shared_stride, lane);
        ldmatrix_b_k16n8_col(
            key_fragment2, key + 16 * shared_stride + dimension,
            shared_stride, lane);
        ldmatrix_b_k16n8_col(
            key_fragment3, key + 24 * shared_stride + dimension,
            shared_stride, lane);

        mma_m16n8k16(scores.part[0], query_fragment, key_fragment0);
        mma_m16n8k16(scores.part[1], query_fragment, key_fragment1);
        mma_m16n8k16(scores.part[2], query_fragment, key_fragment2);
        mma_m16n8k16(scores.part[3], query_fragment, key_fragment3);
    }
}

template <int HeadDim>
__device__ __forceinline__ void compute_qk_from_registers(
    ScoreTile16x32& scores,
    const unsigned (&query_fragments)[HeadDim / 16][4],
    const __half* key,
    int shared_stride,
    int lane
) {
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        clear_accumulator(scores.part[i]);
    }

    #pragma unroll
    for (int dimension_tile = 0;
         dimension_tile < HeadDim / 16;
         ++dimension_tile) {
        const int dimension = dimension_tile * 16;
        unsigned key_fragment0[2];
        unsigned key_fragment1[2];
        unsigned key_fragment2[2];
        unsigned key_fragment3[2];

        ldmatrix_b_k16n8_col(
            key_fragment0, key + dimension, shared_stride, lane);
        ldmatrix_b_k16n8_col(
            key_fragment1, key + 8 * shared_stride + dimension,
            shared_stride, lane);
        ldmatrix_b_k16n8_col(
            key_fragment2, key + 16 * shared_stride + dimension,
            shared_stride, lane);
        ldmatrix_b_k16n8_col(
            key_fragment3, key + 24 * shared_stride + dimension,
            shared_stride, lane);

        mma_m16n8k16(
            scores.part[0], query_fragments[dimension_tile], key_fragment0);
        mma_m16n8k16(
            scores.part[1], query_fragments[dimension_tile], key_fragment1);
        mma_m16n8k16(
            scores.part[2], query_fragments[dimension_tile], key_fragment2);
        mma_m16n8k16(
            scores.part[3], query_fragments[dimension_tile], key_fragment3);
    }
}

template <int HeadDim>
__device__ __forceinline__ void accumulate_pv(
    Mma16816Accumulator (&output)[HeadDim / 8],
    const __half* probabilities,
    const __half* value,
    int probability_stride,
    int value_stride,
    int lane
) {
    #pragma unroll
    for (int output_column = 0; output_column < HeadDim; output_column += 8) {
        unsigned value_fragment[2];
        #pragma unroll
        for (int key_subtile = 0; key_subtile < kBlockN; key_subtile += 16) {
            unsigned probability_fragment[4];
            ldmatrix_a_m16k16_row(
                probability_fragment,
                probabilities + key_subtile,
                probability_stride,
                lane);
            ldmatrix_b_k16n8_row_transposed(
                value_fragment,
                value + key_subtile * value_stride + output_column,
                value_stride,
                lane);
            mma_m16n8k16(
                output[output_column / 8],
                probability_fragment,
                value_fragment);
        }
    }
}

template <int HeadDim>
__device__ __forceinline__ void scale_output(
    Mma16816Accumulator (&output)[HeadDim / 8],
    float upper_scale,
    float lower_scale
) {
    #pragma unroll
    for (int i = 0; i < HeadDim / 8; ++i) {
        output[i].x[0] *= upper_scale;
        output[i].x[1] *= upper_scale;
        output[i].x[2] *= lower_scale;
        output[i].x[3] *= lower_scale;
    }
}

/**
 * Converts one register-resident 16x32 score tile into FP16 probabilities,
 * updates the online-softmax recurrence, and rescales the previous O state only
 * when a row maximum actually changed.
 */
template <int HeadDim, bool Causal>
__device__ __forceinline__ void softmax_tile_to_shared(
    const ScoreTile16x32& score_tile,
    Mma16816Accumulator (&output)[HeadDim / 8],
    __half* probabilities,
    int key_begin,
    int query_sequence,
    int key_value_sequence,
    int query_position_offset,
    int upper_query_position,
    int lower_query_position,
    float scale_log2,
    float& running_max_upper,
    float& running_max_lower,
    float& running_sum_upper,
    float& running_sum_lower,
    int lane
) {
    const int lane_group = lane >> 2;
    const int lane_in_group = lane & 3;
    const int pair_column = lane_in_group * 2;

    const bool upper_query_valid = upper_query_position < query_sequence;
    const bool lower_query_valid = lower_query_position < query_sequence;
    const int upper_causal_limit = query_position_offset + upper_query_position;
    const int lower_causal_limit = query_position_offset + lower_query_position;
    const bool complete_key_tile = key_begin + kBlockN <= key_value_sequence;

    bool upper_tile_unmasked = upper_query_valid && complete_key_tile;
    bool lower_tile_unmasked = lower_query_valid && complete_key_tile;
    if constexpr (Causal) {
        upper_tile_unmasked =
            upper_tile_unmasked
            && key_begin + kBlockN - 1 <= upper_causal_limit;
        lower_tile_unmasked =
            lower_tile_unmasked
            && key_begin + kBlockN - 1 <= lower_causal_limit;
    }

    float upper_scores[8];
    float lower_scores[8];

    #pragma unroll
    for (int part = 0; part < 4; ++part) {
        const int base_column =
            (part >> 1) * 16 + (part & 1) * 8 + pair_column;
        const int key0 = key_begin + base_column;
        const int key1 = key0 + 1;

        bool upper_valid0 = upper_tile_unmasked;
        bool upper_valid1 = upper_tile_unmasked;
        bool lower_valid0 = lower_tile_unmasked;
        bool lower_valid1 = lower_tile_unmasked;

        if (!upper_tile_unmasked) {
            upper_valid0 = upper_query_valid && key0 < key_value_sequence;
            upper_valid1 = upper_query_valid && key1 < key_value_sequence;
            if constexpr (Causal) {
                upper_valid0 = upper_valid0 && key0 <= upper_causal_limit;
                upper_valid1 = upper_valid1 && key1 <= upper_causal_limit;
            }
        }
        if (!lower_tile_unmasked) {
            lower_valid0 = lower_query_valid && key0 < key_value_sequence;
            lower_valid1 = lower_query_valid && key1 < key_value_sequence;
            if constexpr (Causal) {
                lower_valid0 = lower_valid0 && key0 <= lower_causal_limit;
                lower_valid1 = lower_valid1 && key1 <= lower_causal_limit;
            }
        }

        upper_scores[part * 2] = upper_valid0
            ? score_tile.part[part].x[0] * scale_log2
            : -INFINITY;
        upper_scores[part * 2 + 1] = upper_valid1
            ? score_tile.part[part].x[1] * scale_log2
            : -INFINITY;
        lower_scores[part * 2] = lower_valid0
            ? score_tile.part[part].x[2] * scale_log2
            : -INFINITY;
        lower_scores[part * 2 + 1] = lower_valid1
            ? score_tile.part[part].x[3] * scale_log2
            : -INFINITY;
    }

    float upper_local_max = -INFINITY;
    float lower_local_max = -INFINITY;
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        upper_local_max = fast_max(upper_local_max, upper_scores[i]);
        lower_local_max = fast_max(lower_local_max, lower_scores[i]);
    }

    const float upper_tile_max = subgroup4_max(upper_local_max);
    const float lower_tile_max = subgroup4_max(lower_local_max);
    const float upper_new_max = fast_max(running_max_upper, upper_tile_max);
    const float lower_new_max = fast_max(running_max_lower, lower_tile_max);

    const bool upper_has_history = running_max_upper != -INFINITY;
    const bool lower_has_history = running_max_lower != -INFINITY;
    const float upper_alpha = upper_has_history
        ? exp2f(running_max_upper - upper_new_max)
        : 0.0F;
    const float lower_alpha = lower_has_history
        ? exp2f(running_max_lower - lower_new_max)
        : 0.0F;

    const bool upper_requires_rescale = upper_has_history && upper_alpha != 1.0F;
    const bool lower_requires_rescale = lower_has_history && lower_alpha != 1.0F;
    if (upper_requires_rescale || lower_requires_rescale) {
        scale_output<HeadDim>(
            output,
            upper_requires_rescale ? upper_alpha : 1.0F,
            lower_requires_rescale ? lower_alpha : 1.0F);
    }

    float upper_local_sum = 0.0F;
    float lower_local_sum = 0.0F;
    #pragma unroll
    for (int part = 0; part < 4; ++part) {
        const int base_column =
            (part >> 1) * 16 + (part & 1) * 8 + pair_column;

        const float upper0 = upper_scores[part * 2] != -INFINITY
            ? exp2f(upper_scores[part * 2] - upper_new_max)
            : 0.0F;
        const float upper1 = upper_scores[part * 2 + 1] != -INFINITY
            ? exp2f(upper_scores[part * 2 + 1] - upper_new_max)
            : 0.0F;
        const float lower0 = lower_scores[part * 2] != -INFINITY
            ? exp2f(lower_scores[part * 2] - lower_new_max)
            : 0.0F;
        const float lower1 = lower_scores[part * 2 + 1] != -INFINITY
            ? exp2f(lower_scores[part * 2 + 1] - lower_new_max)
            : 0.0F;

        upper_local_sum += upper0 + upper1;
        lower_local_sum += lower0 + lower1;

        probabilities[lane_group * kProbStride + base_column] =
            __float2half_rn(upper0);
        probabilities[lane_group * kProbStride + base_column + 1] =
            __float2half_rn(upper1);
        probabilities[(lane_group + 8) * kProbStride + base_column] =
            __float2half_rn(lower0);
        probabilities[(lane_group + 8) * kProbStride + base_column + 1] =
            __float2half_rn(lower1);
    }

    const float upper_tile_sum = subgroup4_sum(upper_local_sum);
    const float lower_tile_sum = subgroup4_sum(lower_local_sum);
    running_sum_upper = running_sum_upper * upper_alpha + upper_tile_sum;
    running_sum_lower = running_sum_lower * lower_alpha + lower_tile_sum;
    running_max_upper = upper_new_max;
    running_max_lower = lower_new_max;
}

template <int HeadDim, bool ReturnLSE>
__device__ __forceinline__ void store_output_and_lse(
    __half* __restrict__ output,
    float* __restrict__ logsumexp,
    const Mma16816Accumulator (&output_fragments)[HeadDim / 8],
    std::int64_t output_head_base,
    std::int64_t output_row_stride,
    int batch,
    int query_sequence,
    int num_query_heads,
    int query_head,
    int upper_query_position,
    int lower_query_position,
    float running_max_upper,
    float running_max_lower,
    float running_sum_upper,
    float running_sum_lower,
    int lane
) {
    const int lane_in_group = lane & 3;
    const int output_pair_column = lane_in_group * 2;
    const float inverse_upper = running_sum_upper > 0.0F
        ? __frcp_rn(running_sum_upper)
        : 0.0F;
    const float inverse_lower = running_sum_lower > 0.0F
        ? __frcp_rn(running_sum_lower)
        : 0.0F;

    #pragma unroll
    for (int tile = 0; tile < HeadDim / 8; ++tile) {
        const int column = tile * 8 + output_pair_column;
        if (upper_query_position < query_sequence) {
            const std::int64_t base = output_head_base
                + static_cast<std::int64_t>(upper_query_position)
                    * output_row_stride;
            output[base + column] = __float2half_rn(
                output_fragments[tile].x[0] * inverse_upper);
            output[base + column + 1] = __float2half_rn(
                output_fragments[tile].x[1] * inverse_upper);
        }
        if (lower_query_position < query_sequence) {
            const std::int64_t base = output_head_base
                + static_cast<std::int64_t>(lower_query_position)
                    * output_row_stride;
            output[base + column] = __float2half_rn(
                output_fragments[tile].x[2] * inverse_lower);
            output[base + column + 1] = __float2half_rn(
                output_fragments[tile].x[3] * inverse_lower);
        }
    }

    if constexpr (ReturnLSE) {
        const int lane_in_row_group = lane & 3;
        if (lane_in_row_group == 0) {
            if (upper_query_position < query_sequence) {
                const std::int64_t index =
                    (static_cast<std::int64_t>(batch) * query_sequence
                     + upper_query_position) * num_query_heads
                    + query_head;
                logsumexp[index] = running_sum_upper > 0.0F
                    ? running_max_upper * kLn2 + __logf(running_sum_upper)
                    : -INFINITY;
            }
            if (lower_query_position < query_sequence) {
                const std::int64_t index =
                    (static_cast<std::int64_t>(batch) * query_sequence
                     + lower_query_position) * num_query_heads
                    + query_head;
                logsumexp[index] = running_sum_lower > 0.0F
                    ? running_max_lower * kLn2 + __logf(running_sum_lower)
                    : -INFINITY;
            }
        }
    }
}

/** Shared-memory footprint of the packed GQA4 path. */
template <int HeadDim>
inline constexpr std::size_t packed_shared_bytes() {
    constexpr int kStride = kSharedStride<HeadDim>;
    return static_cast<std::size_t>(kRowsPerWarp) * kStride * sizeof(__half)
        + 2U * static_cast<std::size_t>(kBlockN) * kStride * sizeof(__half)
        + 2U * static_cast<std::size_t>(kBlockN) * kStride * sizeof(__half)
        + static_cast<std::size_t>(kPackedQueryHeads)
            * kRowsPerWarp * kProbStride * sizeof(__half);
}

/**
 * Fast GQA=4 path. One CTA owns one KV head and four corresponding query heads.
 * K and V are loaded once and reused by all four warps.
 */
template <int HeadDim, bool Causal, bool ReturnLSE>
__global__ __launch_bounds__(kPackedThreads, 2)
void flash_gqa4_f16_ampere_kernel(
    __half* __restrict__ output,
    float* __restrict__ logsumexp,
    const __half* __restrict__ query,
    const __half* __restrict__ key,
    const __half* __restrict__ value,
    int query_sequence,
    int key_value_sequence,
    int num_query_heads,
    int num_kv_heads,
    float scale_log2,
    int query_position_offset
) {
    static_assert(HeadDim == 32 || HeadDim == 64 || HeadDim == 128);
    constexpr int kStride = kSharedStride<HeadDim>;

    const int query_tile = static_cast<int>(blockIdx.x);
    const int kv_head = static_cast<int>(blockIdx.y);
    const int batch = static_cast<int>(blockIdx.z);
    const int warp = static_cast<int>(threadIdx.x) / kWarpSize;
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int query_head = kv_head * kPackedQueryHeads + warp;
    const int query_begin = query_tile * kRowsPerWarp;

    const std::int64_t query_row_stride =
        static_cast<std::int64_t>(num_query_heads) * HeadDim;
    const std::int64_t kv_row_stride =
        static_cast<std::int64_t>(num_kv_heads) * HeadDim;
    const std::int64_t kv_head_base =
        (static_cast<std::int64_t>(batch) * key_value_sequence * num_kv_heads
         + kv_head) * HeadDim;
    const std::int64_t query_head_base =
        (static_cast<std::int64_t>(batch) * query_sequence * num_query_heads
         + query_head) * HeadDim;

    extern __shared__ __align__(16) unsigned char shared_raw[];
    auto* query_stage = reinterpret_cast<__half*>(shared_raw);
    auto* key_shared = query_stage + kRowsPerWarp * kStride;
    auto* value_shared = key_shared + 2 * kBlockN * kStride;
    auto* probabilities = value_shared + 2 * kBlockN * kStride;
    __half* warp_probabilities =
        probabilities + warp * kRowsPerWarp * kProbStride;

    unsigned query_fragments[HeadDim / 16][4];

    // Reuse one small Q stage. Each warp retains its own Q fragment in registers.
    #pragma unroll
    for (int packed_head = 0;
         packed_head < kPackedQueryHeads;
         ++packed_head) {
        const int staged_query_head =
            kv_head * kPackedQueryHeads + packed_head;
        const std::int64_t staged_query_base =
            (static_cast<std::int64_t>(batch)
                 * query_sequence * num_query_heads
             + staged_query_head) * HeadDim;

        load_rows_f16_async<kRowsPerWarp, HeadDim, kStride>(
            query_stage,
            query + staged_query_base,
            query_begin,
            query_sequence,
            query_row_stride);
        cp_async_wait_all();
        __syncthreads();

        if (warp == packed_head) {
            #pragma unroll
            for (int dimension_tile = 0;
                 dimension_tile < HeadDim / 16;
                 ++dimension_tile) {
                ldmatrix_a_m16k16_row(
                    query_fragments[dimension_tile],
                    query_stage + dimension_tile * 16,
                    kStride,
                    lane);
            }
        }
        __syncthreads();
    }

    Mma16816Accumulator output_fragments[HeadDim / 8];
    #pragma unroll
    for (int i = 0; i < HeadDim / 8; ++i) {
        clear_accumulator(output_fragments[i]);
    }

    const int lane_group = lane >> 2;
    const int upper_query_position = query_begin + lane_group;
    const int lower_query_position = upper_query_position + 8;

    float running_max_upper = -INFINITY;
    float running_max_lower = -INFINITY;
    float running_sum_upper = 0.0F;
    float running_sum_lower = 0.0F;

    const int max_visible_key = Causal
        ? min(
            key_value_sequence,
            query_position_offset + query_begin + kRowsPerWarp)
        : key_value_sequence;

    if (max_visible_key > 0) {
        load_kv_f16_async<HeadDim, kStride>(
            key_shared,
            value_shared,
            key + kv_head_base,
            value + kv_head_base,
            0,
            key_value_sequence,
            kv_row_stride);
    }

    for (int key_begin = 0, tile_index = 0;
         key_begin < max_visible_key;
         key_begin += kBlockN, ++tile_index) {
        const int stage = tile_index & 1;
        __half* current_key = key_shared + stage * kBlockN * kStride;
        __half* current_value = value_shared + stage * kBlockN * kStride;

        cp_async_wait_all();
        __syncthreads();

        const int next_key_begin = key_begin + kBlockN;
        if (next_key_begin < max_visible_key) {
            const int next_stage = stage ^ 1;
            load_kv_f16_async<HeadDim, kStride>(
                key_shared + next_stage * kBlockN * kStride,
                value_shared + next_stage * kBlockN * kStride,
                key + kv_head_base,
                value + kv_head_base,
                next_key_begin,
                key_value_sequence,
                kv_row_stride);
        }

        ScoreTile16x32 score_tile;
        compute_qk_from_registers<HeadDim>(
            score_tile,
            query_fragments,
            current_key,
            kStride,
            lane);

        softmax_tile_to_shared<HeadDim, Causal>(
            score_tile,
            output_fragments,
            warp_probabilities,
            key_begin,
            query_sequence,
            key_value_sequence,
            query_position_offset,
            upper_query_position,
            lower_query_position,
            scale_log2,
            running_max_upper,
            running_max_lower,
            running_sum_upper,
            running_sum_lower,
            lane);

        __syncwarp();
        accumulate_pv<HeadDim>(
            output_fragments,
            warp_probabilities,
            current_value,
            kProbStride,
            kStride,
            lane);
        __syncthreads();
    }

    store_output_and_lse<HeadDim, ReturnLSE>(
        output,
        logsumexp,
        output_fragments,
        query_head_base,
        query_row_stride,
        batch,
        query_sequence,
        num_query_heads,
        query_head,
        upper_query_position,
        lower_query_position,
        running_max_upper,
        running_max_lower,
        running_sum_upper,
        running_sum_lower,
        lane);
}

/** Shared-memory footprint of the generic per-query-head path. */
template <int HeadDim, int BlockM>
inline constexpr std::size_t generic_shared_bytes() {
    constexpr int kStride = kSharedStride<HeadDim>;
    constexpr int kWarps = BlockM / kRowsPerWarp;
    return static_cast<std::size_t>(BlockM) * kStride * sizeof(__half)
        + 2U * static_cast<std::size_t>(kBlockN) * kStride * sizeof(__half)
        + 2U * static_cast<std::size_t>(kBlockN) * kStride * sizeof(__half)
        + static_cast<std::size_t>(kWarps)
            * kRowsPerWarp * kProbStride * sizeof(__half);
}

/** Generic fallback for non-4:1 GQA ratios. */
template <int HeadDim, int BlockM, bool Causal, bool ReturnLSE>
__global__ __launch_bounds__(128, 1)
void flash_gqa_f16_generic_ampere_kernel(
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
    float scale_log2,
    int query_position_offset
) {
    static_assert(HeadDim == 32 || HeadDim == 64 || HeadDim == 128);
    static_assert(BlockM == 32 || BlockM == 64);
    constexpr int kStride = kSharedStride<HeadDim>;
    constexpr int kWarps = BlockM / kRowsPerWarp;

    const int query_tiles = (query_sequence + BlockM - 1) / BlockM;
    const int linear_block = static_cast<int>(blockIdx.x);
    const int query_tile = linear_block % query_tiles;
    const int query_head =
        (linear_block / query_tiles) % num_query_heads;
    const int batch =
        linear_block / (query_tiles * num_query_heads);
    if (batch >= batch_size) {
        return;
    }

    const int heads_per_kv = num_query_heads / num_kv_heads;
    const int kv_head = query_head / heads_per_kv;
    const int query_begin = query_tile * BlockM;
    const std::int64_t query_row_stride =
        static_cast<std::int64_t>(num_query_heads) * HeadDim;
    const std::int64_t kv_row_stride =
        static_cast<std::int64_t>(num_kv_heads) * HeadDim;
    const std::int64_t query_head_base =
        (static_cast<std::int64_t>(batch) * query_sequence * num_query_heads
         + query_head) * HeadDim;
    const std::int64_t kv_head_base =
        (static_cast<std::int64_t>(batch) * key_value_sequence * num_kv_heads
         + kv_head) * HeadDim;

    extern __shared__ __align__(16) unsigned char shared_raw[];
    auto* query_shared = reinterpret_cast<__half*>(shared_raw);
    auto* key_shared = query_shared + BlockM * kStride;
    auto* value_shared = key_shared + 2 * kBlockN * kStride;
    auto* probabilities = value_shared + 2 * kBlockN * kStride;

    load_rows_f16_async<BlockM, HeadDim, kStride>(
        query_shared,
        query + query_head_base,
        query_begin,
        query_sequence,
        query_row_stride);
    cp_async_wait_all();
    __syncthreads();

    const int warp = static_cast<int>(threadIdx.x) / kWarpSize;
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int lane_group = lane >> 2;
    const int warp_row_begin = warp * kRowsPerWarp;
    const int upper_query_position =
        query_begin + warp_row_begin + lane_group;
    const int lower_query_position = upper_query_position + 8;
    const __half* warp_query = query_shared + warp_row_begin * kStride;
    __half* warp_probabilities =
        probabilities + warp * kRowsPerWarp * kProbStride;

    Mma16816Accumulator output_fragments[HeadDim / 8];
    #pragma unroll
    for (int i = 0; i < HeadDim / 8; ++i) {
        clear_accumulator(output_fragments[i]);
    }

    float running_max_upper = -INFINITY;
    float running_max_lower = -INFINITY;
    float running_sum_upper = 0.0F;
    float running_sum_lower = 0.0F;

    const int max_visible_key = Causal
        ? min(
            key_value_sequence,
            query_position_offset + query_begin + BlockM)
        : key_value_sequence;

    if (max_visible_key > 0) {
        load_kv_f16_async<HeadDim, kStride>(
            key_shared,
            value_shared,
            key + kv_head_base,
            value + kv_head_base,
            0,
            key_value_sequence,
            kv_row_stride);
    }

    for (int key_begin = 0, tile_index = 0;
         key_begin < max_visible_key;
         key_begin += kBlockN, ++tile_index) {
        const int stage = tile_index & 1;
        __half* current_key = key_shared + stage * kBlockN * kStride;
        __half* current_value = value_shared + stage * kBlockN * kStride;

        cp_async_wait_all();
        __syncthreads();

        const int next_key_begin = key_begin + kBlockN;
        if (next_key_begin < max_visible_key) {
            const int next_stage = stage ^ 1;
            load_kv_f16_async<HeadDim, kStride>(
                key_shared + next_stage * kBlockN * kStride,
                value_shared + next_stage * kBlockN * kStride,
                key + kv_head_base,
                value + kv_head_base,
                next_key_begin,
                key_value_sequence,
                kv_row_stride);
        }

        ScoreTile16x32 score_tile;
        compute_qk_from_shared<HeadDim>(
            score_tile,
            warp_query,
            current_key,
            kStride,
            lane);

        softmax_tile_to_shared<HeadDim, Causal>(
            score_tile,
            output_fragments,
            warp_probabilities,
            key_begin,
            query_sequence,
            key_value_sequence,
            query_position_offset,
            upper_query_position,
            lower_query_position,
            scale_log2,
            running_max_upper,
            running_max_lower,
            running_sum_upper,
            running_sum_lower,
            lane);

        __syncwarp();
        accumulate_pv<HeadDim>(
            output_fragments,
            warp_probabilities,
            current_value,
            kProbStride,
            kStride,
            lane);
        __syncthreads();
    }

    store_output_and_lse<HeadDim, ReturnLSE>(
        output,
        logsumexp,
        output_fragments,
        query_head_base,
        query_row_stride,
        batch,
        query_sequence,
        num_query_heads,
        query_head,
        upper_query_position,
        lower_query_position,
        running_max_upper,
        running_max_lower,
        running_sum_upper,
        running_sum_lower,
        lane);
}

template <int HeadDim, bool Causal, bool ReturnLSE>
void launch_packed_gqa4(
    Tensor& output,
    float* logsumexp,
    const Tensor& query,
    const Tensor& key,
    const Tensor& value,
    std::size_t batch_size,
    std::size_t query_sequence,
    std::size_t key_value_sequence,
    float scale_log2,
    const FlashAttentionOptions& options,
    cudaStream_t stream
) {
    const unsigned int query_tiles = static_cast<unsigned int>(
        (query_sequence + kRowsPerWarp - 1U) / kRowsPerWarp);
    if (query_tiles == 0U || batch_size == 0U) {
        return;
    }

    constexpr std::size_t shared_bytes = packed_shared_bytes<HeadDim>();
    if constexpr (shared_bytes > 48U * 1024U) {
        CUDA_CHECK(cudaFuncSetAttribute(
            flash_gqa4_f16_ampere_kernel<HeadDim, Causal, ReturnLSE>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            static_cast<int>(shared_bytes)));
    }

    const dim3 grid(
        query_tiles,
        static_cast<unsigned int>(options.num_kv_heads),
        static_cast<unsigned int>(batch_size));

    flash_gqa4_f16_ampere_kernel<HeadDim, Causal, ReturnLSE>
        <<<grid, kPackedThreads, shared_bytes, stream>>>(
            static_cast<__half*>(output.raw_data()),
            logsumexp,
            static_cast<const __half*>(query.raw_data()),
            static_cast<const __half*>(key.raw_data()),
            static_cast<const __half*>(value.raw_data()),
            static_cast<int>(query_sequence),
            static_cast<int>(key_value_sequence),
            static_cast<int>(options.num_query_heads),
            static_cast<int>(options.num_kv_heads),
            scale_log2,
            static_cast<int>(options.query_position_offset));
    CUDA_CHECK(cudaGetLastError());
}

template <int HeadDim, int BlockM, bool Causal, bool ReturnLSE>
void launch_generic(
    Tensor& output,
    float* logsumexp,
    const Tensor& query,
    const Tensor& key,
    const Tensor& value,
    std::size_t batch_size,
    std::size_t query_sequence,
    std::size_t key_value_sequence,
    float scale_log2,
    const FlashAttentionOptions& options,
    cudaStream_t stream
) {
    constexpr int kWarps = BlockM / kRowsPerWarp;
    constexpr int kThreads = kWarps * kWarpSize;
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
        generic_shared_bytes<HeadDim, BlockM>();
    if constexpr (shared_bytes > 48U * 1024U) {
        CUDA_CHECK(cudaFuncSetAttribute(
            flash_gqa_f16_generic_ampere_kernel<
                HeadDim, BlockM, Causal, ReturnLSE>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            static_cast<int>(shared_bytes)));
    }

    flash_gqa_f16_generic_ampere_kernel<
        HeadDim, BlockM, Causal, ReturnLSE>
        <<<static_cast<unsigned int>(blocks),
           kThreads,
           shared_bytes,
           stream>>>(
            static_cast<__half*>(output.raw_data()),
            logsumexp,
            static_cast<const __half*>(query.raw_data()),
            static_cast<const __half*>(key.raw_data()),
            static_cast<const __half*>(value.raw_data()),
            static_cast<int>(batch_size),
            static_cast<int>(query_sequence),
            static_cast<int>(key_value_sequence),
            static_cast<int>(options.num_query_heads),
            static_cast<int>(options.num_kv_heads),
            scale_log2,
            static_cast<int>(options.query_position_offset));
    CUDA_CHECK(cudaGetLastError());
}

template <int HeadDim, bool Causal, bool ReturnLSE>
void launch_selected(
    Tensor& output,
    float* logsumexp,
    const Tensor& query,
    const Tensor& key,
    const Tensor& value,
    std::size_t batch_size,
    std::size_t query_sequence,
    std::size_t key_value_sequence,
    float scale_log2,
    const FlashAttentionOptions& options,
    cudaStream_t stream
) {
    const std::size_t heads_per_kv =
        options.num_query_heads / options.num_kv_heads;
    const bool packed_grid_supported =
        batch_size <= 65535U
        && options.num_kv_heads <= 65535U
        && ((query_sequence + kRowsPerWarp - 1U) / kRowsPerWarp)
            <= static_cast<std::size_t>(std::numeric_limits<int>::max());

    if (heads_per_kv == kPackedQueryHeads && packed_grid_supported) {
        launch_packed_gqa4<HeadDim, Causal, ReturnLSE>(
            output,
            logsumexp,
            query,
            key,
            value,
            batch_size,
            query_sequence,
            key_value_sequence,
            scale_log2,
            options,
            stream);
        return;
    }

    if constexpr (HeadDim == 128) {
        launch_generic<HeadDim, 32, Causal, ReturnLSE>(
            output,
            logsumexp,
            query,
            key,
            value,
            batch_size,
            query_sequence,
            key_value_sequence,
            scale_log2,
            options,
            stream);
    } else {
        launch_generic<HeadDim, 64, Causal, ReturnLSE>(
            output,
            logsumexp,
            query,
            key,
            value,
            batch_size,
            query_sequence,
            key_value_sequence,
            scale_log2,
            options,
            stream);
    }
}

template <int HeadDim>
void launch_head_dimension(
    Tensor& output,
    float* logsumexp,
    const Tensor& query,
    const Tensor& key,
    const Tensor& value,
    std::size_t batch_size,
    std::size_t query_sequence,
    std::size_t key_value_sequence,
    float scale_log2,
    const FlashAttentionOptions& options,
    cudaStream_t stream
) {
    if (options.causal) {
        if (logsumexp != nullptr) {
            launch_selected<HeadDim, true, true>(
                output, logsumexp, query, key, value,
                batch_size, query_sequence, key_value_sequence,
                scale_log2, options, stream);
        } else {
            launch_selected<HeadDim, true, false>(
                output, nullptr, query, key, value,
                batch_size, query_sequence, key_value_sequence,
                scale_log2, options, stream);
        }
    } else {
        if (logsumexp != nullptr) {
            launch_selected<HeadDim, false, true>(
                output, logsumexp, query, key, value,
                batch_size, query_sequence, key_value_sequence,
                scale_log2, options, stream);
        } else {
            launch_selected<HeadDim, false, false>(
                output, nullptr, query, key, value,
                batch_size, query_sequence, key_value_sequence,
                scale_log2, options, stream);
        }
    }
}

void launch_f16_dispatch(
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
    const float scale_log2 = scale * kLog2E;
    switch (options.head_dim) {
        case 32:
            launch_head_dimension<32>(
                output, logsumexp, query, key, value,
                batch_size, query_sequence, key_value_sequence,
                scale_log2, options, stream);
            break;
        case 64:
            launch_head_dimension<64>(
                output, logsumexp, query, key, value,
                batch_size, query_sequence, key_value_sequence,
                scale_log2, options, stream);
            break;
        case 128:
            launch_head_dimension<128>(
                output, logsumexp, query, key, value,
                batch_size, query_sequence, key_value_sequence,
                scale_log2, options, stream);
            break;
        default:
            throw std::invalid_argument(
                "flash_gqa_attention_forward: Tensor Core path supports "
                "head_dim 32, 64, or 128");
    }
}

void validate_ampere_device() {
    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    cudaDeviceProp properties{};
    CUDA_CHECK(cudaGetDeviceProperties(&properties, device));
    if (properties.major < 8) {
        throw std::runtime_error(
            "flash_gqa_attention_forward: mma.sync/ldmatrix/cp.async "
            "path requires SM80 or newer");
    }
}

bool is_supported_head_dim(std::size_t head_dim) {
    return head_dim == 32U || head_dim == 64U || head_dim == 128U;
}

void validate_rank_four(const Tensor& tensor, const char* name) {
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
            "flash_gqa_attention_forward: use_tensor_cores must be true");
    }
}

void validate_integer_ranges(
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

struct ValidatedShape {
    std::size_t batch_size;
    std::size_t query_sequence;
    std::size_t key_value_sequence;
};

ValidatedShape validate_forward_contract(
    Tensor& output,
    const Tensor& query,
    const Tensor& key,
    const Tensor& value,
    const FlashAttentionOptions& options
) {
    validate_options(options);
    validate_rank_four(output, "output");
    validate_rank_four(query, "query");
    validate_rank_four(key, "key");
    validate_rank_four(value, "value");

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
            "flash_gqa_attention_forward: optimized path requires F16 tensors");
    }
    if (options.causal
        && options.query_position_offset + query_sequence
            > key_value_sequence) {
        throw std::invalid_argument(
            "flash_gqa_attention_forward: causal query range exceeds K/V");
    }

    validate_integer_ranges(
        batch_size,
        query_sequence,
        key_value_sequence,
        options);
    return {batch_size, query_sequence, key_value_sequence};
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
    validate_ampere_device();
    const ValidatedShape shape =
        validate_forward_contract(output, query, key, value, options);

    const float scale = options.attention_scale > 0.0F
        ? options.attention_scale
        : 1.0F / std::sqrt(static_cast<float>(options.head_dim));

    launch_f16_dispatch(
        output,
        nullptr,
        query,
        key,
        value,
        shape.batch_size,
        shape.query_sequence,
        shape.key_value_sequence,
        scale,
        options,
        stream);
}

void flash_gqa_attention_forward_with_lse(
    Tensor& output,
    Tensor& logsumexp,
    const Tensor& query,
    const Tensor& key,
    const Tensor& value,
    cudaStream_t stream,
    const FlashAttentionOptions& options
) {
    validate_ampere_device();
    const ValidatedShape shape =
        validate_forward_contract(output, query, key, value, options);

    if (logsumexp.dim() != 3
        || logsumexp.size(0) != query.shape()[0]
        || logsumexp.size(1) != query.shape()[1]
        || logsumexp.size(2) != query.shape()[2]
        || logsumexp.dtype() != Dtype::F32) {
        throw std::invalid_argument(
            "flash_gqa_attention_forward_with_lse: logsumexp must be "
            "contiguous F32 [B, Q, Hq]");
    }

    const float scale = options.attention_scale > 0.0F
        ? options.attention_scale
        : 1.0F / std::sqrt(static_cast<float>(options.head_dim));

    launch_f16_dispatch(
        output,
        static_cast<float*>(logsumexp.raw_data()),
        query,
        key,
        value,
        shape.batch_size,
        shape.query_sequence,
        shape.key_value_sequence,
        scale,
        options,
        stream);
}
