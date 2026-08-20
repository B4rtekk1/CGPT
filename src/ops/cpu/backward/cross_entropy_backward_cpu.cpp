#include "ops/cpu/backward/cross_entropy_backward_cpu.h"

#include <immintrin.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {
    /**
     * @brief Loads one floating-point tensor element and converts it to F32.
     *
     * The source storage may use F32, IEEE F16, or BF16 representation.
     * Computation is performed in single precision regardless of the storage
     * format.
     *
     * @param p Pointer to the source tensor storage.
     * @param t Element type of the source tensor.
     * @param i Linear element index to read.
     * @return The selected element represented as an F32 value.
     *
     * @note Reading F16 values requires CPU support for F16C.
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
     * @brief Converts and stores one F32 value in tensor storage.
     *
     * F16 conversion uses F16C instructions. BF16 conversion applies
     * round-to-nearest-even rounding before truncating the lower 16 bits.
     *
     * @param p Pointer to the destination tensor storage.
     * @param t Destination element type.
     * @param i Linear element index to write.
     * @param x F32 value to convert and store.
     */
    void store1(void *p, Dtype t, std::size_t i, float x) {
        if (t == Dtype::F32) {
            static_cast<float *>(p)[i] = x;
            return;
        }
        if (t == Dtype::F16) {
            static_cast<std::uint16_t *>(p)[i] = static_cast<std::uint16_t>(_mm_cvtsi128_si32(
                _mm_cvtps_ph(_mm_set_ss(x), 0)));
            return;
        }
        std::uint32_t b;
        std::memcpy(&b, &x, 4);
        static_cast<std::uint16_t *>(p)[i] = static_cast<std::uint16_t>((b + 0x7fff + ((b >> 16) & 1)) >> 16);
    }

    /**
     * @brief Validates arguments for the fused cross-entropy forward/backward pass.
     *
     * @param loss Scalar F32 output tensor with shape `[1]`.
     * @param gradient Gradient tensor with the same shape and dtype as @p logits.
     * @param logits Two-dimensional CPU tensor with shape `[n, vocabulary_size]`.
     * @param targets Pointer to an array of @p n target token identifiers.
     * @param n Number of rows in @p logits and number of target identifiers.
     * @param scale Positive finite multiplier applied to the mean-loss gradient.
     *
     * @throws std::invalid_argument If a pointer is null, @p n is zero, tensor
     * devices, shapes, or dtypes are incompatible, the vocabulary is empty, or
     * @p scale is not finite and positive.
     */
    void validate(const Tensor &loss, const Tensor &gradient, const Tensor &logits,
                  const bpe::TokenId *targets, std::size_t n, float scale) {
        if (!targets || !n || logits.device_type() != DeviceType::CPU || logits.dim() != 2 ||
            !is_floating_point(logits.dtype()) || logits.size(0) != n || !logits.size(1) ||
            loss.device_type() != DeviceType::CPU || loss.dtype() != Dtype::F32 ||
            loss.shape() != std::vector<std::size_t>{1} || gradient.device_type() != DeviceType::CPU ||
            gradient.dtype() != logits.dtype() || gradient.shape() != logits.shape() ||
            !std::isfinite(scale) || scale <= 0.0F)
            throw std::invalid_argument("CPU cross entropy backward: invalid tensors, targets or scale");
    }
}

/**
 * @brief Computes mean cross-entropy loss and its logits gradient on the CPU.
 *
 * For every row, the function evaluates a numerically stable softmax using the
 * maximum-logit subtraction method. For a valid target index @f$y_r@f$, the
 * per-row loss is
 * @f[
 *     L_r = \log\left(\sum_j e^{z_{r,j}}\right) - z_{r,y_r},
 * @f]
 * and the logits gradient is
 * @f[
 *     \frac{\partial L}{\partial z_{r,j}} =
 *     \frac{scale}{n}\left(softmax(z_r)_j - \mathbf{1}[j=y_r]\right).
 * @f]
 *
 * The returned loss is the unscaled arithmetic mean of valid per-row losses.
 * Rows whose target identifier lies outside the vocabulary contribute zero to
 * the loss, while their gradient remains the scaled softmax distribution.
 *
 * Rows are processed independently with OpenMP, and intermediate arithmetic is
 * accumulated in F32 for F32, F16, and BF16 storage formats.
 *
 * @param loss Destination scalar F32 tensor with shape `[1]`.
 * @param gradient Destination tensor for gradients with respect to @p logits.
 * @param logits Input logits with shape `[n, vocabulary_size]`.
 * @param targets Pointer to @p n target token identifiers.
 * @param n Number of samples represented by @p logits.
 * @param scale Positive multiplier applied to the mean gradient.
 *
 * @throws std::invalid_argument If tensor devices, shapes, dtypes, targets, or
 * @p scale are invalid.
 *
 * @note F16 tensor support requires the F16C instruction set.
 */
void cross_entropy_forward_backward_cpu(
    Tensor &loss, Tensor &gradient, const Tensor &logits,
    const bpe::TokenId *targets, std::size_t n, float scale) {
    validate(loss, gradient, logits, targets, n, scale);
    const auto vocab = logits.size(1);
    const auto *p = logits.raw_data();
    auto *g = gradient.raw_data();
    float total = 0.0F;
#pragma omp parallel for reduction(+:total) schedule(static)
    for (std::int64_t r = 0; r < static_cast<std::int64_t>(n); ++r) {
        const auto base = static_cast<std::size_t>(r) * vocab;
        float maximum = -std::numeric_limits<float>::infinity();
        for (std::size_t i = 0; i < vocab; ++i) maximum = std::max(maximum, load1(p, logits.dtype(), base + i));
        float sum = 0.0F;
        for (std::size_t i = 0; i < vocab; ++i) sum += std::exp(load1(p, logits.dtype(), base + i) - maximum);
        const float inv = 1.0F / sum;
        const float factor = scale / static_cast<float>(n);
        if (targets[r] < vocab) total += std::log(sum) + maximum - load1(p, logits.dtype(), base + targets[r]);
        for (std::size_t i = 0; i < vocab; ++i) {
            float value = std::exp(load1(p, logits.dtype(), base + i) - maximum) * inv;
            if (targets[r] == i) value -= 1.0F;
            store1(g, logits.dtype(), base + i, value * factor);
        }
    }
    *static_cast<float *>(loss.raw_data()) = total / static_cast<float>(n);
}
