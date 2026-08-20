#pragma once

#include <immintrin.h>

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>

#if defined(_MSC_VER)
#  define CGPT_FORCE_INLINE __forceinline
#  define CGPT_RESTRICT __restrict
#else
#  define CGPT_FORCE_INLINE inline __attribute__((always_inline))
#  define CGPT_RESTRICT __restrict__
#endif

namespace cgpt::cpu::simd {

constexpr std::size_t kVectorWidth = 8;
constexpr std::size_t kParallelThreshold = 32U * 1024U;
constexpr std::size_t kWorkTileElements = 16U * 1024U;

/**
 * @brief Fast AVX2 exponential approximation.
 *
 * This is a range-reduced minimax polynomial approximation. It is substantially
 * faster than eight scalar std::exp calls and is accurate enough for neural
 * network activations and softmax. Inputs are clamped to the finite FP32 exp
 * range.
 */
CGPT_FORCE_INLINE __m256 exp256_ps(__m256 x) noexcept {
    const __m256 exp_hi = _mm256_set1_ps(88.3762626647949f);
    const __m256 exp_lo = _mm256_set1_ps(-88.3762626647949f);
    const __m256 log2ef = _mm256_set1_ps(1.44269504088896341f);
    const __m256 half = _mm256_set1_ps(0.5f);
    const __m256 one = _mm256_set1_ps(1.0f);

    x = _mm256_min_ps(x, exp_hi);
    x = _mm256_max_ps(x, exp_lo);

    __m256 fx = _mm256_fmadd_ps(x, log2ef, half);

    __m256i emm0 = _mm256_cvttps_epi32(fx);
    __m256 tmp = _mm256_cvtepi32_ps(emm0);
    const __m256 floor_mask = _mm256_cmp_ps(tmp, fx, _CMP_GT_OS);
    fx = _mm256_sub_ps(tmp, _mm256_and_ps(floor_mask, one));

    const __m256 c1 = _mm256_set1_ps(0.693359375f);
    const __m256 c2 = _mm256_set1_ps(-2.12194440e-4f);
    x = _mm256_fnmadd_ps(fx, c1, x);
    x = _mm256_fnmadd_ps(fx, c2, x);

    const __m256 z = _mm256_mul_ps(x, x);

    __m256 y = _mm256_set1_ps(1.9875691500E-4f);
    y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(1.3981999507E-3f));
    y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(8.3334519073E-3f));
    y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(4.1665795894E-2f));
    y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(1.6666665459E-1f));
    y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(5.0000001201E-1f));
    y = _mm256_fmadd_ps(y, z, x);
    y = _mm256_add_ps(y, one);

    emm0 = _mm256_cvttps_epi32(fx);
    emm0 = _mm256_add_epi32(emm0, _mm256_set1_epi32(0x7f));
    emm0 = _mm256_slli_epi32(emm0, 23);
    const __m256 pow2n = _mm256_castsi256_ps(emm0);
    return _mm256_mul_ps(y, pow2n);
}

/** @brief One Newton-Raphson refinement of the AVX reciprocal estimate. */
CGPT_FORCE_INLINE __m256 reciprocal256_ps(const __m256 x) noexcept {
    __m256 reciprocal = _mm256_rcp_ps(x);
    reciprocal = _mm256_mul_ps(
        reciprocal,
        _mm256_fnmadd_ps(x, reciprocal, _mm256_set1_ps(2.0f))
    );
    return reciprocal;
}

/** @brief Computes SiLU(x) = x / (1 + exp(-x)). */
CGPT_FORCE_INLINE __m256 silu256_ps(const __m256 x) noexcept {
    const __m256 denominator = _mm256_add_ps(
        _mm256_set1_ps(1.0f),
        exp256_ps(_mm256_sub_ps(_mm256_setzero_ps(), x))
    );
    return _mm256_mul_ps(x, reciprocal256_ps(denominator));
}

CGPT_FORCE_INLINE float horizontal_sum_ps(const __m256 value) noexcept {
    const __m128 low = _mm256_castps256_ps128(value);
    const __m128 high = _mm256_extractf128_ps(value, 1);
    __m128 sum = _mm_add_ps(low, high);
    sum = _mm_hadd_ps(sum, sum);
    sum = _mm_hadd_ps(sum, sum);
    return _mm_cvtss_f32(sum);
}

CGPT_FORCE_INLINE float horizontal_max_ps(const __m256 value) noexcept {
    const __m128 low = _mm256_castps256_ps128(value);
    const __m128 high = _mm256_extractf128_ps(value, 1);
    __m128 maximum = _mm_max_ps(low, high);
    maximum = _mm_max_ps(maximum, _mm_movehl_ps(maximum, maximum));
    maximum = _mm_max_ss(maximum, _mm_shuffle_ps(maximum, maximum, 0x55));
    return _mm_cvtss_f32(maximum);
}

CGPT_FORCE_INLINE float bf16_to_float(const std::uint16_t value) noexcept {
    const std::uint32_t bits = static_cast<std::uint32_t>(value) << 16U;
    return std::bit_cast<float>(bits);
}

CGPT_FORCE_INLINE std::uint16_t float_to_bf16(const float value) noexcept {
    std::uint32_t bits = std::bit_cast<std::uint32_t>(value);
    const std::uint32_t rounding_bias = 0x7FFFU + ((bits >> 16U) & 1U);
    bits += rounding_bias;
    return static_cast<std::uint16_t>(bits >> 16U);
}

CGPT_FORCE_INLINE __m256 load_bf16x8(const std::uint16_t* const source) noexcept {
    const __m128i packed = _mm_loadu_si128(reinterpret_cast<const __m128i*>(source));
    __m256i widened = _mm256_cvtepu16_epi32(packed);
    widened = _mm256_slli_epi32(widened, 16);
    return _mm256_castsi256_ps(widened);
}

CGPT_FORCE_INLINE void store_bf16x8(
    std::uint16_t* const destination,
    const __m256 value
) noexcept {
    __m256i bits = _mm256_castps_si256(value);
    const __m256i lsb = _mm256_and_si256(
        _mm256_srli_epi32(bits, 16),
        _mm256_set1_epi32(1)
    );
    bits = _mm256_add_epi32(bits, _mm256_set1_epi32(0x7FFF));
    bits = _mm256_add_epi32(bits, lsb);
    bits = _mm256_srli_epi32(bits, 16);

    const __m128i low = _mm256_castsi256_si128(bits);
    const __m128i high = _mm256_extracti128_si256(bits, 1);
    const __m128i packed = _mm_packus_epi32(low, high);
    _mm_storeu_si128(reinterpret_cast<__m128i*>(destination), packed);
}

struct Float32Ops {
    using Storage = float;

    CGPT_FORCE_INLINE static __m256 load(const Storage* const source) noexcept {
        return _mm256_loadu_ps(source);
    }

    CGPT_FORCE_INLINE static void store(
        Storage* const destination,
        const __m256 value
    ) noexcept {
        _mm256_storeu_ps(destination, value);
    }

    CGPT_FORCE_INLINE static float load_scalar(const Storage value) noexcept {
        return value;
    }

    CGPT_FORCE_INLINE static Storage store_scalar(const float value) noexcept {
        return value;
    }
};

struct Float16Ops {
    using Storage = std::uint16_t;

    CGPT_FORCE_INLINE static __m256 load(const Storage* const source) noexcept {
        const __m128i packed = _mm_loadu_si128(reinterpret_cast<const __m128i*>(source));
        return _mm256_cvtph_ps(packed);
    }

    CGPT_FORCE_INLINE static void store(
        Storage* const destination,
        const __m256 value
    ) noexcept {
        const __m128i packed = _mm256_cvtps_ph(
            value,
            _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC
        );
        _mm_storeu_si128(reinterpret_cast<__m128i*>(destination), packed);
    }

    CGPT_FORCE_INLINE static float load_scalar(const Storage value) noexcept {
        return _cvtsh_ss(value);
    }

    CGPT_FORCE_INLINE static Storage store_scalar(const float value) noexcept {
        return static_cast<Storage>(
            _cvtss_sh(value, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC)
        );
    }
};

struct BFloat16Ops {
    using Storage = std::uint16_t;

    CGPT_FORCE_INLINE static __m256 load(const Storage* const source) noexcept {
        return load_bf16x8(source);
    }

    CGPT_FORCE_INLINE static void store(
        Storage* const destination,
        const __m256 value
    ) noexcept {
        store_bf16x8(destination, value);
    }

    CGPT_FORCE_INLINE static float load_scalar(const Storage value) noexcept {
        return bf16_to_float(value);
    }

    CGPT_FORCE_INLINE static Storage store_scalar(const float value) noexcept {
        return float_to_bf16(value);
    }
};

}