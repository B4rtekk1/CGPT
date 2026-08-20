/**
 * @file cross_entropy_cpu.cpp
 * @brief CPU implementation of mean categorical cross-entropy loss.
 *
 * The implementation accepts F32, F16, and BF16 logits, performs all reduction
 * arithmetic in F32, uses a numerically stable log-sum-exp formulation, and
 * parallelizes independent rows with OpenMP.
 */

#include "ops/cpu/cross_entropy_cpu.h"

#include <immintrin.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace {
    /**
     * @brief Loads one tensor element and converts it to F32.
     *
     * @param p Pointer to the beginning of the tensor storage.
     * @param t Tensor element type.
     * @param i Element index.
     * @return The requested element represented as F32.
     */
    float load1(const void *p, Dtype t, std::size_t i) {
        if (t == Dtype::F32) return static_cast<const float *>(p)[i];
        const auto h = static_cast<const std::uint16_t *>(p)[i];
        if (t == Dtype::F16) return _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(h)));
        std::uint32_t b = static_cast<std::uint32_t>(h) << 16;
        float x;
        std::memcpy(&x, &b, 4);
        return x;
    }

    /**
     * @brief Loads eight tensor elements and converts them to F32 lanes.
     *
     * @param p Pointer to the beginning of the tensor storage.
     * @param t Tensor element type.
     * @param i Index of the first element to load.
     * @return Eight values represented as an AVX F32 vector.
     */
    __m256 load8(const void *p, Dtype t, std::size_t i) {
        if (t == Dtype::F32)return _mm256_loadu_ps(static_cast<const float *>(p) + i);
        const auto *h = static_cast<const std::uint16_t *>(p) + i;
        if (t == Dtype::F16)return _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i *>(h)));
        return _mm256_castsi256_ps(
            _mm256_slli_epi32(_mm256_cvtepu16_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i *>(h))), 16));
    }

    /**
     * @brief Validates tensors and target storage used by cross-entropy.
     *
     * @param loss Scalar F32 CPU tensor receiving the mean loss.
     * @param logits Two-dimensional CPU floating-point tensor with shape
     *        `[n, vocabulary_size]`.
     * @param targets Pointer to @p n target token identifiers.
     * @param n Number of rows and target identifiers.
     *
     * @throws std::invalid_argument If any pointer, shape, device, or data-type
     *         requirement is not satisfied.
     */
    void validate_loss(const Tensor &loss, const Tensor &logits, const bpe::TokenId *targets, std::size_t n) {
        if (!targets || !n || logits.device_type() != DeviceType::CPU || logits.dim() != 2 || !
            is_floating_point(logits.dtype()) || logits.size(0) != n || !logits.size(1) || loss.device_type() !=
            DeviceType::CPU || loss.dtype() != Dtype::F32 || loss.shape() != std::vector<std::size_t>{1})
            throw
                    std::invalid_argument("CPU cross entropy: invalid tensors or targets");
    }

    /**
     * @brief Computes cross-entropy for one row of logits.
     *
     * The result is evaluated as
     * `log(sum(exp(logit_i - max_logit))) + max_logit - logit_target`,
     * which avoids overflow in the exponential calculation.
     *
     * @param data Pointer to the logits tensor storage.
     * @param t Logits element type.
     * @param base Index of the first logit in the selected row.
     * @param vocab Number of logits in the row.
     * @param target Target class identifier.
     * @return The row loss, or zero when @p target is outside the vocabulary.
     *
     * @note The maximum reduction uses AVX2, while the exponential sum is
     *       accumulated in scalar F32 arithmetic.
     */
    float row_loss(const void *data, Dtype t, std::size_t base, std::size_t vocab, bpe::TokenId target) {
        float maximum = -std::numeric_limits<float>::infinity();
        __m256 m = _mm256_set1_ps(-std::numeric_limits<float>::infinity());
        std::size_t i = 0;
        for (; i + 7 < vocab; i += 8)m = _mm256_max_ps(m, load8(data, t, base + i));
        alignas(32)float q[8];
        _mm256_store_ps(q, m);
        for (float x: q)maximum = std::max(maximum, x);
        for (; i < vocab; ++i)maximum = std::max(maximum, load1(data, t, base + i));
        float sum = 0;
        for (i = 0; i < vocab; ++i)sum += std::exp(load1(data, t, base + i) - maximum);
        if (target >= vocab)return 0;
        return std::log(sum) + maximum - load1(data, t, base + target);
    }
}

/**
 * @brief Computes mean categorical cross-entropy from unnormalized logits.
 *
 * The logits tensor must have shape `[n, vocabulary_size]`. One loss value is
 * calculated for each row and the arithmetic mean is written to the scalar F32
 * tensor @p loss. Rows are distributed across OpenMP worker threads, and the
 * final sum is combined with an OpenMP reduction.
 *
 * An out-of-range target contributes zero to the mean rather than raising an
 * exception or reading outside the logits tensor.
 *
 * @param loss Scalar CPU tensor with shape `[1]` and type F32.
 * @param logits CPU floating-point tensor with shape `[n, vocabulary_size]`.
 * @param targets Pointer to an array of @p n target token identifiers.
 * @param n Number of rows and target identifiers. Must be greater than zero.
 *
 * @throws std::invalid_argument If the input pointers, tensor devices, shapes,
 *         or data types are incompatible.
 *
 * @note The implementation requires AVX2. F16 execution additionally requires
 *       F16C support.
 */
void cross_entropy_forward_cpu(Tensor &loss, const Tensor &logits, const bpe::TokenId *targets, std::size_t n) {
    validate_loss(loss, logits, targets, n);
    const auto vocab = logits.size(1);
    const auto *p = logits.raw_data();
    float total = 0;
#pragma omp parallel for reduction(+:total) schedule(static)
    for (std::int64_t r = 0; r < static_cast<std::int64_t>(n); ++r)
        total += row_loss(p, logits.dtype(),
                          static_cast<std::size_t>(r) * vocab, vocab, targets[r]);
    *static_cast<float *>(loss.raw_data()) = total / static_cast<float>(n);
}
