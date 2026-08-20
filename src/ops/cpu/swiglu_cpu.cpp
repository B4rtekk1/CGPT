/**
 * @file swiglu_cpu.cpp
 * @brief AVX2-optimized CPU implementation of the SwiGLU activation.
 *
 * The implementation supports F32, F16, and BF16 tensors, evaluates the main
 * vectorized path in eight-element AVX2 registers, and parallelizes independent
 * rows with OpenMP when processing fused gate/up tensors.
 */
#include "ops/cpu/swiglu_cpu.h"
#include "core/simd_math.h"
#include <immintrin.h>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>

namespace {
    /**
     * @brief Loads eight tensor elements and converts them to F32 lanes.
     *
     * @param p Pointer to the beginning of the source storage.
     * @param t Source tensor data type.
     * @param i Element offset from @p p.
     * @return AVX register containing eight F32 values.
     *
     * @note The function uses unaligned loads. F16 conversion requires F16C,
     *       while BF16 values are expanded by shifting their bit patterns.
     */
    inline __m256 load8(const void *p, Dtype t, std::size_t i) {
        if (t == Dtype::F32) return _mm256_loadu_ps(static_cast<const float *>(p) + i);
        if (t == Dtype::F16)
            return _mm256_cvtph_ps(
                _mm_loadu_si128(reinterpret_cast<const __m128i *>(static_cast<const std::uint16_t *>(p) + i)));
        return _mm256_castsi256_ps(_mm256_slli_epi32(
            _mm256_cvtepu16_epi32(
                _mm_loadu_si128(reinterpret_cast<const __m128i *>(static_cast<const std::uint16_t *>(p) + i))), 16));
    }

    /**
     * @brief Loads one tensor element and converts it to F32.
     *
     * @param p Pointer to the beginning of the source storage.
     * @param t Source tensor data type.
     * @param i Element offset from @p p.
     * @return Scalar value represented as F32.
     */
    inline float load1(const void *p, Dtype t, std::size_t i) {
        if (t == Dtype::F32)return static_cast<const float *>(p)[i];
        const auto h = static_cast<const std::uint16_t *>(p)[i];
        if (t == Dtype::F16)return _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(h)));
        std::uint32_t b = static_cast<std::uint32_t>(h) << 16;
        float x;
        std::memcpy(&x, &b, 4);
        return x;
    }

    /**
     * @brief Stores eight F32 lanes in the requested tensor format.
     *
     * BF16 conversion uses round-to-nearest-even before packing the upper
     * 16 bits of each F32 lane.
     *
     * @param p Pointer to the beginning of the destination storage.
     * @param t Destination tensor data type.
     * @param i Element offset from @p p.
     * @param v AVX register containing values to store.
     */
    inline void store8(void *p, Dtype t, std::size_t i, __m256 v) {
        if (t == Dtype::F32) {
            _mm256_storeu_ps(static_cast<float *>(p) + i, v);
            return;
        }
        if (t == Dtype::F16) {
            _mm_storeu_si128(reinterpret_cast<__m128i *>(static_cast<std::uint16_t *>(p) + i), _mm256_cvtps_ph(v, 0));
            return;
        }
        const __m256i b = _mm256_castps_si256(v), r = _mm256_add_epi32(
            b, _mm256_add_epi32(_mm256_set1_epi32(0x7fff),
                                _mm256_and_si256(_mm256_srli_epi32(b, 16), _mm256_set1_epi32(1))));
        const __m256i h = _mm256_srli_epi32(r, 16);
        _mm_storeu_si128(reinterpret_cast<__m128i *>(static_cast<std::uint16_t *>(p) + i),
                         _mm_packus_epi32(_mm256_castsi256_si128(h), _mm256_extracti128_si256(h, 1)));
    }

    /**
     * @brief Stores one F32 value in the requested tensor format.
     *
     * @param p Pointer to the beginning of the destination storage.
     * @param t Destination tensor data type.
     * @param i Element offset from @p p.
     * @param x Value to store.
     */
    inline void store1(void *p, Dtype t, std::size_t i, float x) {
        if (t == Dtype::F32) {
            static_cast<float *>(p)[i] = x;
            return;
        }
        std::uint32_t b;
        std::memcpy(&b, &x, 4);
        if (t == Dtype::BF16) {
            static_cast<std::uint16_t *>(p)[i] = static_cast<std::uint16_t>((b + 0x7fff + ((b >> 16) & 1)) >> 16);
            return;
        }
        static_cast<std::uint16_t *>(p)[i] = static_cast<std::uint16_t>(_mm_cvtsi128_si32(_mm_cvtps_ph(_mm_set_ss(x), 0)));
    }

    /**
     * @brief Validates tensors used by the unfused or fused SwiGLU path.
     *
     * @param o Output tensor.
     * @param a Gate tensor or fused gate/up tensor.
     * @param b Optional separate up-projection tensor.
     *
     * @throws std::invalid_argument If devices, shapes, or data types are
     *         incompatible, or if the output type is not floating point.
     */
    inline void check(Tensor &o, const Tensor &a, const Tensor *b = nullptr) {
        if (o.device_type() != DeviceType::CPU || a.device_type() != DeviceType::CPU || (
                b && b->device_type() != DeviceType::CPU) || o.shape() != a.shape() || (b && b->shape() != a.shape()) ||
            o.dtype() != a.dtype() || (b && b->dtype() != a.dtype()) || !is_floating_point(o.dtype()))
            throw
                    std::invalid_argument("CPU SwiGLU: matching CPU floating-point tensors required");
    }

    /**
     * @brief Applies SwiGLU using AVX2 vectorization and an OpenMP row loop.
     *
     * The operation computes `SiLU(gate) * up`. In fused mode, @p in stores
     * gate and up values concatenated along its final dimension. In unfused
     * mode, @p in is the gate tensor and @p up points to a separate tensor.
     *
     * @tparam Fused Whether gate and up values share one concatenated tensor.
     * @param[out] o Output tensor.
     * @param in Gate tensor or fused gate/up tensor.
     * @param up Separate up tensor for the unfused path; ignored in fused mode.
     *
     * @note The vectorized loop processes 32 elements per iteration as four
     *       independent eight-lane AVX2 operations. Remaining elements use a
     *       scalar fallback.
     */
    template<bool Fused>
    void apply(Tensor &o, const Tensor &in, const Tensor *up) {
        const auto w = Fused ? o.shape().back() : o.numel(), rows = Fused ? o.numel() / w : 1;
        auto *d = o.raw_data();
        const auto *p = in.raw_data();
#pragma omp parallel for schedule(static)
        for (std::int64_t r = 0; r < static_cast<std::int64_t>(rows); ++r) {
            const auto rr = static_cast<std::size_t>(r), stride = dtype_size(o.dtype());
            const auto *g = static_cast<const std::uint8_t *>(p) + (Fused ? rr * 2 * w : rr * w) * stride;
            const auto *u = Fused
                                ? g + w * stride
                                : static_cast<const std::uint8_t *>(up->raw_data()) + rr * w * stride;
            std::size_t i = 0;
            for (; i + 31 < w; i += 32)
                for (int k = 0; k < 4; ++k) {
                    auto x = load8(g, o.dtype(), k * 8 + i), y = load8(u, o.dtype(), k * 8 + i);
                    auto z = cgpt::cpu::simd::silu256_ps(x);
                    store8(d, o.dtype(), rr * w + i + k * 8, _mm256_mul_ps(z, y));
                }
            for (; i < w; ++i) {
                auto x = load1(g, o.dtype(), i), y = load1(u, o.dtype(), i);
                store1(d, o.dtype(), rr * w + i, x / (1 + std::exp(-x)) * y);
            }
        }
    }
}

/**
 * @brief Computes SwiGLU from separate gate and up-projection tensors.
 *
 * For every element, the function evaluates
 * `output = SiLU(gate) * up` using AVX2 where possible.
 *
 * @param[out] output Tensor receiving the activation result.
 * @param gate Gate-projection tensor.
 * @param up Up-projection tensor.
 *
 * @throws std::invalid_argument If tensors are not matching CPU-resident
 *         floating-point tensors.
 */
void swiglu_forward_cpu(Tensor &output, const Tensor &gate, const Tensor &up) {
    check(output, gate, &up);
    apply<false>(output, gate, &up);
}

/**
 * @brief Computes SwiGLU from a fused gate/up-projection tensor.
 *
 * The final dimension of @p gate_up must contain two equally sized contiguous
 * halves: gate values followed by up values. The final dimension of @p output
 * must equal half of the fused input width.
 *
 * @param[out] output Tensor receiving the activation result.
 * @param gate_up Fused tensor containing concatenated gate and up projections.
 *
 * @throws std::invalid_argument If devices, data types, ranks, or shapes are
 *         incompatible, or if the fused width is odd.
 */
void swiglu_forward_cpu(Tensor &output, const Tensor &gate_up) {
    if (output.device_type() != DeviceType::CPU || gate_up.device_type() != DeviceType::CPU || !
        is_floating_point(output.dtype()) || output.dtype() != gate_up.dtype() || gate_up.shape().empty() || gate_up.
        shape().back() % 2 || output.shape().size() != gate_up.shape().size() || output.shape().back() != gate_up.
        shape().back() / 2)
        throw std::invalid_argument("CPU SwiGLU: invalid fused tensors");
    for (std::size_t i = 0; i + 1 < output.shape().size(); ++i)
        if (output.shape()[i] != gate_up.shape()[i])
            throw
                    std::invalid_argument("CPU SwiGLU: shape mismatch");
    apply<true>(output, gate_up, nullptr);
}