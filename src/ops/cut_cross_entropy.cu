/**
 * @file cut_cross_entropy.cu
 * @brief CUDA implementation of cut cross-entropy with input and classifier gradients.
 *
 * This implementation avoids storing the complete logits or softmax matrix.
 * For every input row it stores only the maximum logit and the inverse softmax
 * denominator required by the backward kernels.
 */

#include "ops/cut_cross_entropy.h"

#include "core/cuda_check.h"

#include <math_constants.h>
#include <stdexcept>

namespace {

constexpr int kThreads = 256;

/**
 * @brief Reduces a value across all threads in a block by summation.
 * @param value Per-thread value to reduce.
 * @return The block-wide sum, returned consistently to every thread.
 */
__device__ float block_sum(float value) {
    __shared__ float partial[kThreads / 32];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x / 32;
#pragma unroll
    for (int offset = 16; offset; offset >>= 1) value += __shfl_down_sync(0xffffffff, value, offset);
    if (lane == 0) partial[warp] = value;
    __syncthreads();
    value = warp == 0 ? (lane < kThreads / 32 ? partial[lane] : 0.0f) : 0.0f;
    if (warp == 0) {
#pragma unroll
        for (int offset = 16; offset; offset >>= 1) value += __shfl_down_sync(0xffffffff, value, offset);
        if (lane == 0) partial[0] = value;
    }
    __syncthreads();
    return partial[0];
}

/**
 * @brief Reduces a value across all threads in a block by maximum.
 * @param value Per-thread value to reduce.
 * @return The block-wide maximum, returned consistently to every thread.
 */
__device__ float block_max(float value) {
    __shared__ float partial[kThreads / 32];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x / 32;
#pragma unroll
    for (int offset = 16; offset; offset >>= 1) value = fmaxf(value, __shfl_down_sync(0xffffffff, value, offset));
    if (lane == 0) partial[warp] = value;
    __syncthreads();
    value = warp == 0 ? (lane < kThreads / 32 ? partial[lane] : -CUDART_INF_F) : -CUDART_INF_F;
    if (warp == 0) {
#pragma unroll
        for (int offset = 16; offset; offset >>= 1) value = fmaxf(value, __shfl_down_sync(0xffffffff, value, offset));
        if (lane == 0) partial[0] = value;
    }
    __syncthreads();
    return partial[0];
}

/**
 * @brief Computes a dot product between an input vector and a classifier row.
 * @param input Input vector.
 * @param classifier Classifier row.
 * @param hidden Number of vector elements.
 * @return The dot product accumulated with fused multiply-add operations.
 */
__device__ float dot(const float* input, const float* classifier, std::size_t hidden) {
    float value = 0.0f;
    for (std::size_t h = 0; h < hidden; ++h) value = fmaf(input[h], classifier[h], value);
    return value;
}

/**
 * @brief Computes the mean cross-entropy loss and per-row softmax statistics.
 *
 * One block processes one input row. Only the maximum logit and inverse softmax
 * denominator are retained for use by the backward kernels.
 *
 * @param loss Device scalar receiving the accumulated mean loss.
 * @param statistics Device array containing two values per row: max logit and
 *        inverse softmax denominator.
 * @param input Input matrix with shape [rows, hidden].
 * @param classifier Classifier matrix with shape [vocabulary, hidden].
 * @param targets Device target-token array with length @p rows.
 * @param rows Number of input rows.
 * @param vocabulary Number of classifier rows/classes.
 * @param hidden Hidden/vector dimension.
 * @param inv_rows Reciprocal of @p rows.
 */
__global__ void cce_forward(float* loss, float* statistics, const float* input,
    const float* classifier, const bpe::TokenId* targets, std::size_t rows,
    std::size_t vocabulary, std::size_t hidden, float inv_rows) {
    const std::size_t row = blockIdx.x;
    if (row >= rows) return;
    const float* x = input + row * hidden;
    float local_max = -CUDART_INF_F;
    for (std::size_t token = threadIdx.x; token < vocabulary; token += kThreads)
        local_max = fmaxf(local_max, dot(x, classifier + token * hidden, hidden));
    const float maximum = block_max(local_max);
    float local_sum = 0.0f;
    for (std::size_t token = threadIdx.x; token < vocabulary; token += kThreads)
        local_sum += __expf(dot(x, classifier + token * hidden, hidden) - maximum);
    const float denominator = block_sum(local_sum);
    if (threadIdx.x == 0) {
        const auto target = static_cast<std::size_t>(targets[row]);
        statistics[2 * row] = maximum;
        statistics[2 * row + 1] = __frcp_rn(denominator);
        atomicAdd(loss, (__logf(denominator) + maximum - dot(x, classifier + target * hidden, hidden)) * inv_rows);
    }
}

/**
 * @brief Computes the gradient with respect to the input matrix.
 *
 * Each block computes one input-gradient coordinate and reconstructs the
 * required probabilities from the saved row statistics.
 */
__global__ void cce_input_backward(float* input_gradient, const float* statistics,
    const float* input, const float* classifier, const bpe::TokenId* targets,
    std::size_t rows, std::size_t vocabulary, std::size_t hidden, float inv_rows) {
    const std::size_t row = blockIdx.x;
    const std::size_t h = blockIdx.y;
    if (row >= rows || h >= hidden) return;
    const float* x = input + row * hidden;
    const float maximum = statistics[2 * row];
    const float inverse_sum = statistics[2 * row + 1];
    float value = 0.0f;
    for (std::size_t token = threadIdx.x; token < vocabulary; token += kThreads) {
        const float probability = __expf(dot(x, classifier + token * hidden, hidden) - maximum) * inverse_sum;
        value = fmaf(probability, classifier[token * hidden + h], value);
    }
    value = block_sum(value);
    if (threadIdx.x == 0)
        input_gradient[row * hidden + h] = (value - classifier[static_cast<std::size_t>(targets[row]) * hidden + h]) * inv_rows;
}

/**
 * @brief Computes the gradient with respect to the classifier matrix.
 *
 * Blocks cover classifier rows while atomic additions accumulate contributions
 * from all examples in the batch.
 */
__global__ void cce_classifier_backward(float* classifier_gradient, const float* statistics,
    const float* input, const float* classifier, const bpe::TokenId* targets,
    std::size_t rows, std::size_t vocabulary, std::size_t hidden, float inv_rows) {
    const std::size_t token = static_cast<std::size_t>(blockIdx.x) * kThreads + threadIdx.x;
    const std::size_t row = blockIdx.y;
    if (token >= vocabulary || row >= rows) return;
    const float* x = input + row * hidden;
    const float probability = __expf(dot(x, classifier + token * hidden, hidden) - statistics[2 * row]) * statistics[2 * row + 1];
    const float scale = (probability - (token == static_cast<std::size_t>(targets[row]) ? 1.0f : 0.0f)) * inv_rows;
    for (std::size_t h = 0; h < hidden; ++h) atomicAdd(classifier_gradient + token * hidden + h, scale * x[h]);
}

/**
 * @brief Validates tensors and target metadata for cut cross-entropy.
 * @throws std::invalid_argument If devices, dtypes, shapes, or target count
 *         are incompatible with the operation.
 */
void validate(const Tensor& loss, const Tensor& input_gradient, const Tensor& classifier_gradient,
    const Tensor& input, const Tensor& classifier, const bpe::TokenId* targets, std::size_t target_count) {
    if (targets == nullptr || input.device_type() != DeviceType::CUDA || classifier.device_type() != DeviceType::CUDA ||
        input.dtype() != Dtype::F32 || classifier.dtype() != Dtype::F32 || input.dim() != 2 || classifier.dim() != 2 ||
        input.size(0) == 0 || input.size(1) == 0 || classifier.size(0) == 0 || classifier.size(1) != input.size(1))
        throw std::invalid_argument("cut_cross_entropy: input and classifier must be non-empty CUDA F32 matrices");
    if (target_count != input.size(0) || input_gradient.device_type() != DeviceType::CUDA ||
        input_gradient.dtype() != Dtype::F32 || input_gradient.shape() != input.shape() ||
        classifier_gradient.device_type() != DeviceType::CUDA || classifier_gradient.dtype() != Dtype::F32 ||
        classifier_gradient.shape() != classifier.shape())
        throw std::invalid_argument("cut_cross_entropy: incompatible gradient or target count");
    if (loss.device_type() != DeviceType::CUDA || loss.dtype() != Dtype::F32 || loss.shape() != std::vector<std::size_t>{1})
        throw std::invalid_argument("cut_cross_entropy: loss must be a CUDA F32 tensor with shape [1]");
}
} // namespace

/**
 * @brief Computes cut cross-entropy and both required parameter gradients.
 *
 * The operation expects CUDA F32 tensors. The loss is the mean over rows and
 * is accumulated into the one-element CUDA tensor @p loss. Target IDs must be
 * stored in device memory.
 *
 * @param[out] loss One-element CUDA F32 tensor receiving the mean loss.
 * @param[out] input_gradient Gradient with respect to @p input.
 * @param[out] classifier_gradient Gradient with respect to @p classifier.
 * @param[in] input Input activations with shape [rows, hidden].
 * @param[in] classifier Classifier weights with shape [vocabulary, hidden].
 * @param[in] device_targets Device pointer to @p target_count target IDs.
 * @param target_count Number of target IDs; must equal the number of rows.
 * @param stream CUDA stream used for all asynchronous operations.
 * @throws std::invalid_argument If the supplied tensors or targets are invalid.
 * @throws cudaError_t If a CUDA allocation, memory operation, or kernel launch
 *         fails.
 */
void cut_cross_entropy_forward_backward(Tensor& loss, Tensor& input_gradient, Tensor& classifier_gradient,
    const Tensor& input, const Tensor& classifier, const bpe::TokenId* device_targets,
    std::size_t target_count, cudaStream_t stream) {
    validate(loss, input_gradient, classifier_gradient, input, classifier, device_targets, target_count);
    const std::size_t rows = input.size(0), vocabulary = classifier.size(0), hidden = input.size(1);
    float* statistics = nullptr;
    CUDA_CHECK(cudaMallocAsync(&statistics, 2 * rows * sizeof(float), stream));
    CUDA_CHECK(cudaMemsetAsync(loss.raw_data(), 0, sizeof(float), stream));
    CUDA_CHECK(cudaMemsetAsync(input_gradient.raw_data(), 0, input_gradient.nbytes(), stream));
    CUDA_CHECK(cudaMemsetAsync(classifier_gradient.raw_data(), 0, classifier_gradient.nbytes(), stream));
    const float inv_rows = 1.0f / static_cast<float>(rows);
    cce_forward<<<static_cast<unsigned>(rows), kThreads, 0, stream>>>(static_cast<float*>(loss.raw_data()), statistics,
        static_cast<const float*>(input.raw_data()), static_cast<const float*>(classifier.raw_data()), device_targets, rows, vocabulary, hidden, inv_rows);
    cce_input_backward<<<dim3(static_cast<unsigned>(rows), static_cast<unsigned>(hidden)), kThreads, 0, stream>>>(
        static_cast<float*>(input_gradient.raw_data()), statistics, static_cast<const float*>(input.raw_data()), static_cast<const float*>(classifier.raw_data()), device_targets, rows, vocabulary, hidden, inv_rows);
    cce_classifier_backward<<<dim3(static_cast<unsigned>((vocabulary + kThreads - 1) / kThreads), static_cast<unsigned>(rows)), kThreads, 0, stream>>>(
        static_cast<float*>(classifier_gradient.raw_data()), statistics, static_cast<const float*>(input.raw_data()), static_cast<const float*>(classifier.raw_data()), device_targets, rows, vocabulary, hidden, inv_rows);
    CUDA_CHECK(cudaFreeAsync(statistics, stream));
    CUDA_CHECK(cudaGetLastError());
}
