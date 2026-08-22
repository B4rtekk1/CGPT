/**
 * @file linear_cpu.cpp
 * @brief AVX2-accelerated CPU implementation of a dense linear transformation.
 *
 * The implementation evaluates `output = input * weight^T + bias` over the
 * final dimension of an arbitrary-rank input tensor. F32, F16, and BF16 values
 * are accumulated in single precision and converted back to the source type.
 */

#include "ops/cpu/linear_cpu.h"

#include <immintrin.h>
#include <algorithm>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <functional>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <vector>

namespace {
    /** Persistent native-thread executor used by CPU decode GEMMs.
     * It deliberately replaces no OpenMP functionality outside this file. */
    class CpuThreadPool {
    public:
        static CpuThreadPool &instance() {
            static CpuThreadPool pool;
            return pool;
        }

        [[nodiscard]] std::size_t parallelism() const noexcept { return workers_.size(); }

        template <typename Function>
        void run(const std::size_t task_count, Function &&function) {
            if (task_count <= 1 || workers_.empty()) {
                function(0);
                return;
            }
            std::unique_lock lock(mutex_);
            task_ = std::forward<Function>(function);
            task_count_ = task_count;
            next_task_ = 0;
            active_workers_ = 0;
            work_ready_.notify_all();
            completed_.wait(lock, [&] { return next_task_ == task_count_ && active_workers_ == 0; });
            task_ = {};
        }

    private:
        CpuThreadPool() {
            const std::size_t hardware = std::thread::hardware_concurrency();
            const std::size_t count = hardware > 1 ? hardware - 1 : 0;
            workers_.reserve(count);
            for (std::size_t index = 0; index < count; ++index)
                workers_.emplace_back([this] { worker_loop(); });
        }

        ~CpuThreadPool() {
            {
                std::lock_guard lock(mutex_);
                stopping_ = true;
            }
            work_ready_.notify_all();
            for (auto &worker : workers_) worker.join();
        }

        void worker_loop() {
            std::unique_lock lock(mutex_);
            for (;;) {
                work_ready_.wait(lock, [&] { return stopping_ || next_task_ < task_count_; });
                if (stopping_) return;
                const std::size_t index = next_task_++;
                ++active_workers_;
                const auto current_task = task_;
                lock.unlock();
                current_task(index);
                lock.lock();
                --active_workers_;
                if (next_task_ == task_count_ && active_workers_ == 0) completed_.notify_one();
            }
        }

        std::vector<std::thread> workers_;
        std::mutex mutex_;
        std::condition_variable work_ready_, completed_;
        std::function<void(std::size_t)> task_;
        std::size_t task_count_ = 0, next_task_ = 0, active_workers_ = 0;
        bool stopping_ = false;
    };

    /**
     * @brief Loads eight consecutive tensor elements as an AVX2 float vector.
     *
     * F32 values are loaded directly, F16 values are converted with F16C, and
     * BF16 values are expanded into the upper half of F32 lanes.
     *
     * @param p Pointer to the beginning of the tensor storage.
     * @param t Data type of the stored elements.
     * @param i Index of the first element to load.
     * @return Eight values converted to single precision.
     */
    inline __m256 load8(const void *p, Dtype t, std::size_t i) {
        if (t == Dtype::F32) return _mm256_loadu_ps(static_cast<const float *>(p) + i);
        const auto *h = static_cast<const std::uint16_t *>(p) + i;
        if (t == Dtype::F16) return _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i *>(h)));
        return _mm256_castsi256_ps(_mm256_slli_epi32(
            _mm256_cvtepu16_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i *>(h))), 16));
    }

    /**
     * @brief Loads one tensor element and converts it to single precision.
     *
     * @param p Pointer to the beginning of the tensor storage.
     * @param t Data type of the stored elements.
     * @param i Zero-based element index.
     * @return Element value converted to `float`.
     */
    inline float load1(const void *p, Dtype t, std::size_t i) {
        if (t == Dtype::F32) return static_cast<const float *>(p)[i];
        const auto h = static_cast<const std::uint16_t *>(p)[i];
        if (t == Dtype::F16) return _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(h)));
        std::uint32_t bits = static_cast<std::uint32_t>(h) << 16;
        float x;
        std::memcpy(&x, &bits, 4);
        return x;
    }

    /**
     * @brief Stores one single-precision value in the requested tensor format.
     *
     * F16 conversion uses F16C. BF16 conversion applies round-to-nearest-even
     * before truncating the low 16 bits.
     *
     * @param p Pointer to the beginning of the tensor storage.
     * @param t Destination element data type.
     * @param i Zero-based destination element index.
     * @param x Value to store.
     */
    inline void store1(void *p, Dtype t, std::size_t i, float x) {
        if (t == Dtype::F32) {
            static_cast<float *>(p)[i] = x;
            return;
        }
        if (t == Dtype::F16) {
            static_cast<std::uint16_t *>(p)[i] = static_cast<std::uint16_t>(_mm_cvtsi128_si32(_mm_cvtps_ph(_mm_set_ss(x), 0)));
            return;
        }
        std::uint32_t bits;
        std::memcpy(&bits, &x, 4);
        static_cast<std::uint16_t *>(p)[i] = static_cast<std::uint16_t>((bits + 0x7fff + ((bits >> 16) & 1)) >> 16);
    }

    /**
     * @brief Validates tensor placement, data types, ranks, and dimensions.
     *
     * The expected layout is an input tensor ending in `I`, a row-major weight
     * matrix of shape `[O, I]`, an output tensor ending in `O`, and an optional
     * bias vector of shape `[O]`. All leading input and output dimensions must
     * match.
     *
     * @param out Preallocated output tensor.
     * @param in Input tensor with rank of at least one.
     * @param w Weight matrix.
     * @param bias Optional bias vector, or `nullptr` when bias is disabled.
     *
     * @throws std::invalid_argument If any tensor is incompatible with the
     *         required CPU linear-operation layout.
     */
    inline void validate(const Tensor &out, const Tensor &in, const Tensor &w, const Tensor *bias) {
        if (out.device_type() != DeviceType::CPU || in.device_type() != DeviceType::CPU || w.device_type() !=
            DeviceType::CPU ||
            (bias && bias->device_type() != DeviceType::CPU) || in.dim() < 1 || w.dim() != 2 || out.dim() != in.dim() ||
            !is_floating_point(in.dtype()) || in.dtype() != w.dtype() || out.dtype() != in.dtype() ||
            (bias && (bias->dim() != 1 || bias->dtype() != in.dtype() || bias->shape()[0] != w.shape()[0])))
            throw std::invalid_argument("CPU linear: incompatible CPU floating-point tensors");
        if (in.shape().back() != w.shape()[1] || out.shape().back() != w.shape()[0])
            throw std::invalid_argument(
                "CPU linear: shape mismatch");
        for (std::size_t i = 0; i + 1 < in.dim(); ++i)
            if (out.shape()[i] != in.shape()[i])
                throw std::invalid_argument(
                    "CPU linear: leading shape mismatch");
    }

    /**
     * @brief Executes the dense linear transformation with an optional bias.
     *
     * All leading input dimensions are flattened into independent rows. Each
     * OpenMP iteration processes one row. The vectorized path computes eight
     * output channels concurrently and accumulates eight input features per
     * AVX2 FMA operation. Remaining input and output elements use scalar FMA.
     *
     * @param out Preallocated output tensor ending in the output size `O`.
     * @param in Input tensor ending in the input size `I`.
     * @param w Row-major weight matrix with shape `[O, I]`.
     * @param bias Optional bias vector with shape `[O]`.
     *
     * @throws std::invalid_argument If validation of the supplied tensors fails.
     *
     * @note Accumulation is performed in F32 for F32, F16, and BF16 storage.
     * @note The function requires a build target supporting AVX2, FMA, and F16C.
     */
    void apply(Tensor &out, const Tensor &in, const Tensor &w, const Tensor *bias) {
        validate(out, in, w, bias);
        const auto rows = in.numel() / in.shape().back();
        const auto I = in.shape().back();
        const auto O = w.shape()[0];
        const auto t = in.dtype();
        const auto *x = in.raw_data();
        const auto *weights = w.raw_data();
        const auto *b = bias ? bias->raw_data() : nullptr;
        auto *y = out.raw_data();
        const auto compute_outputs = [&](const std::size_t row, const std::size_t first_output,
                                         const std::size_t end_output) {
            const std::size_t rb = row * I, ob = row * O;
            std::size_t o = first_output;
            for (; o + 7 < end_output; o += 8) {
                __m256 a0 = _mm256_setzero_ps(), a1 = a0, a2 = a0, a3 = a0, a4 = a0, a5 = a0, a6 = a0, a7 = a0;
                std::size_t i = 0;
                for (; i + 7 < I; i += 8) {
                    const auto vx = load8(x, t, rb + i);
                    a0 = _mm256_fmadd_ps(vx, load8(weights, t, (o + 0) * I + i), a0);
                    a1 = _mm256_fmadd_ps(vx, load8(weights, t, (o + 1) * I + i), a1);
                    a2 = _mm256_fmadd_ps(vx, load8(weights, t, (o + 2) * I + i), a2);
                    a3 = _mm256_fmadd_ps(vx, load8(weights, t, (o + 3) * I + i), a3);
                    a4 = _mm256_fmadd_ps(vx, load8(weights, t, (o + 4) * I + i), a4);
                    a5 = _mm256_fmadd_ps(vx, load8(weights, t, (o + 5) * I + i), a5);
                    a6 = _mm256_fmadd_ps(vx, load8(weights, t, (o + 6) * I + i), a6);
                    a7 = _mm256_fmadd_ps(vx, load8(weights, t, (o + 7) * I + i), a7);
                }
                for (int z = 0; z < 8; ++z) {
                    float sums[8];
                    __m256 v[8] = {a0, a1, a2, a3, a4, a5, a6, a7};
                    alignas(32) float q[8];
                    _mm256_store_ps(q, v[z]);
                    float s = 0;
                    for (float n: q)s += n;
                    for (std::size_t j = i; j < I; ++j)
                        s = std::fma(load1(x, t, rb + j),
                                     load1(weights, t, (o + z) * I + j), s);
                    if (b)s += load1(b, t, o + z);
                    sums[z] = s;
                    store1(y, t, ob + o + z, s);
                }
            }
            for (; o < end_output; ++o) {
                float s = b ? load1(b, t, o) : 0;
                for (std::size_t i = 0; i < I; ++i)s = std::fma(load1(x, t, rb + i), load1(weights, t, o * I + i), s);
                store1(y, t, ob + o, s);
            }
        };

        // Autoregressive decoding has one input row, so the former OpenMP
        // row parallelism could not use multiple cores.  Split sufficiently
        // wide output matrices into independent eight-channel blocks instead.
        // std::thread keeps the portable build free of the OpenMP runtime.
        const std::size_t vector_outputs = O / 8 * 8;
        const std::size_t workers = std::min<std::size_t>(CpuThreadPool::instance().parallelism(),
                                                          vector_outputs / 1024);
        if (rows == 1 && workers > 1) {
            const std::size_t output_blocks = vector_outputs / 8;
            CpuThreadPool::instance().run(workers, [&](const std::size_t worker) {
                const std::size_t begin = (output_blocks * worker / workers) * 8;
                const std::size_t end = (output_blocks * (worker + 1) / workers) * 8;
                compute_outputs(0, begin, end);
            });
            if (vector_outputs < O) compute_outputs(0, vector_outputs, O);
        } else {
            for (std::size_t row = 0; row < rows; ++row) compute_outputs(row, 0, O);
        }
    }
}

/**
 * @brief Applies a bias-free dense linear transformation on the CPU.
 *
 * Computes `o = i * w^T` over the final input dimension. The weight tensor is
 * interpreted as a row-major matrix of shape `[output_features, input_features]`.
 *
 * @param o Preallocated output tensor whose final dimension is the number of
 *          output features.
 * @param i Input tensor whose final dimension is the number of input features.
 * @param w Weight matrix with shape `[output_features, input_features]`.
 *
 * @throws std::invalid_argument If tensor placement, types, ranks, or shapes are
 *         incompatible.
 */
void linear_forward_cpu(Tensor &o, const Tensor &i, const Tensor &w) { apply(o, i, w, nullptr); }

/**
 * @brief Applies a biased dense linear transformation on the CPU.
 *
 * Computes `o = i * w^T + b` over the final input dimension. The bias is
 * broadcast across every flattened input row.
 *
 * @param o Preallocated output tensor whose final dimension is the number of
 *          output features.
 * @param i Input tensor whose final dimension is the number of input features.
 * @param w Weight matrix with shape `[output_features, input_features]`.
 * @param b Bias vector with shape `[output_features]`.
 *
 * @throws std::invalid_argument If tensor placement, types, ranks, or shapes are
 *         incompatible.
 */
void linear_forward_cpu(Tensor &o, const Tensor &i, const Tensor &w, const Tensor &b) { apply(o, i, w, &b); }
