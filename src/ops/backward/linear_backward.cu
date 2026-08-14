/**
 * @file linear_backward.cu
 * @brief Cached cuBLASLt implementation of the row-major linear backward pass.
 *
 * For Y = X W^T + b this computes dX = dY W, dW = dY^T X and db = sum(dY).
 */

#include "ops/backward/linear_backward.h"
#include "core/device_buffer.h"
#include "core/device_guard.h"

#include <cublasLt.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace {

const char* device_type_name(const DeviceType type) {
    return type == DeviceType::CUDA ? "CUDA" : "CPU";
}

void require_same_device(const Tensor& left, const Tensor& right) {
    if (left.device_type() != right.device_type()) {
        throw std::invalid_argument("linear_backward: tensors must be on the same device, got " +
            std::string(device_type_name(left.device_type())) + " and " + device_type_name(right.device_type()));
    }
}

void set_row_major(cublasLtMatrixLayout_t layout) {
    constexpr cublasLtOrder_t order = CUBLASLT_ORDER_ROW;
    CUBLAS_CHECK(cublasLtMatrixLayoutSetAttribute(
        layout, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order)));
}

struct MatmulKey {
    std::uintptr_t handle;
    std::uintptr_t stream;
    std::size_t rows, columns, inner, workspace_bytes;
    Dtype dtype;
    ComputeType compute_type;
    cublasOperation_t transpose_left, transpose_right;

    bool operator==(const MatmulKey&) const = default;
};

struct MatmulKeyHash {
    std::size_t operator()(const MatmulKey& key) const noexcept {
        std::size_t seed = 0;
        const auto combine = [&seed](const std::size_t value) {
            seed ^= value + 0x9e3779b97f4a7c15ULL + (seed << 6U) + (seed >> 2U);
        };
        combine(key.handle); combine(key.stream); combine(key.rows); combine(key.columns);
        combine(key.inner); combine(key.workspace_bytes); combine(static_cast<std::size_t>(key.dtype));
        combine(static_cast<std::size_t>(key.compute_type));
        combine(static_cast<std::size_t>(key.transpose_left));
        combine(static_cast<std::size_t>(key.transpose_right));
        return seed;
    }
};

class MatmulPlan {
public:
    MatmulPlan(cublasLtHandle_t handle, const MatmulKey& key) {
        const bool tf32 = key.dtype == Dtype::F32 && key.compute_type == ComputeType::TF32 &&
                          key.rows >= 8 && key.columns >= 8 && key.inner >= 8;
        CUBLAS_CHECK(cublasLtMatmulDescCreate(
            &operation_, tf32 ? CUBLAS_COMPUTE_32F_FAST_TF32 : CUBLAS_COMPUTE_32F, CUDA_R_32F));
        CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(operation_, CUBLASLT_MATMUL_DESC_TRANSA,
            &key.transpose_left, sizeof(key.transpose_left)));
        CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(operation_, CUBLASLT_MATMUL_DESC_TRANSB,
            &key.transpose_right, sizeof(key.transpose_right)));

        const std::size_t left_rows = key.transpose_left == CUBLAS_OP_N ? key.rows : key.inner;
        const std::size_t left_columns = key.transpose_left == CUBLAS_OP_N ? key.inner : key.rows;
        const std::size_t right_rows = key.transpose_right == CUBLAS_OP_N ? key.inner : key.columns;
        const std::size_t right_columns = key.transpose_right == CUBLAS_OP_N ? key.columns : key.inner;
        const cudaDataType_t type = to_cuda_dtype(key.dtype);
        CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&left_layout_, type, left_rows, left_columns, left_columns));
        CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&right_layout_, type, right_rows, right_columns, right_columns));
        CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&output_layout_, type, key.rows, key.columns, key.columns));
        set_row_major(left_layout_); set_row_major(right_layout_); set_row_major(output_layout_);

        cublasLtMatmulPreference_t preference = nullptr;
        CUBLAS_CHECK(cublasLtMatmulPreferenceCreate(&preference));
        CUBLAS_CHECK(cublasLtMatmulPreferenceSetAttribute(preference,
            CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &key.workspace_bytes, sizeof(key.workspace_bytes)));
        cublasLtMatmulHeuristicResult_t result{};
        int count = 0;
        const cublasStatus_t status = cublasLtMatmulAlgoGetHeuristic(handle, operation_, left_layout_,
            right_layout_, output_layout_, output_layout_, preference, 1, &result, &count);
        CUBLAS_CHECK(cublasLtMatmulPreferenceDestroy(preference));
        if (status == CUBLAS_STATUS_SUCCESS && count != 0 && result.state == CUBLAS_STATUS_SUCCESS) {
            algorithm_ = result.algo;
            has_algorithm_ = true;
            if (result.workspaceSize != 0) workspace_.allocate(result.workspaceSize);
        } else if (status != CUBLAS_STATUS_SUCCESS && status != CUBLAS_STATUS_NOT_SUPPORTED) {
            CUBLAS_CHECK(status);
        }
    }

    ~MatmulPlan() {
        if (output_layout_) CUBLAS_CHECK(cublasLtMatrixLayoutDestroy(output_layout_));
        if (right_layout_) CUBLAS_CHECK(cublasLtMatrixLayoutDestroy(right_layout_));
        if (left_layout_) CUBLAS_CHECK(cublasLtMatrixLayoutDestroy(left_layout_));
        if (operation_) CUBLAS_CHECK(cublasLtMatmulDescDestroy(operation_));
    }
    MatmulPlan(const MatmulPlan&) = delete;
    MatmulPlan& operator=(const MatmulPlan&) = delete;

    void execute(cublasLtHandle_t handle, Tensor& output, const Tensor& left, const Tensor& right,
                 const bool accumulate, cudaStream_t stream) {
        constexpr float alpha = 1.0f;
        const float beta = accumulate ? 1.0f : 0.0f;
        cublasStatus_t status = cublasLtMatmul(handle, operation_, &alpha, left.raw_data(), left_layout_,
            right.raw_data(), right_layout_, &beta, output.raw_data(), output_layout_, output.raw_data(),
            output_layout_, has_algorithm_ ? &algorithm_ : nullptr, workspace_.data(), workspace_.bytes(), stream);
        if (status == CUBLAS_STATUS_NOT_SUPPORTED && has_algorithm_) {
            status = cublasLtMatmul(handle, operation_, &alpha, left.raw_data(), left_layout_, right.raw_data(),
                right_layout_, &beta, output.raw_data(), output_layout_, output.raw_data(), output_layout_,
                nullptr, nullptr, 0, stream);
        }
        CUBLAS_CHECK(status);
    }

private:
    cublasLtMatmulDesc_t operation_ = nullptr;
    cublasLtMatrixLayout_t left_layout_ = nullptr, right_layout_ = nullptr, output_layout_ = nullptr;
    cublasLtMatmulAlgo_t algorithm_{};
    DeviceBuffer workspace_;
    bool has_algorithm_ = false;
};

MatmulPlan& cached_plan(cublasLtHandle_t handle, const MatmulKey& key) {
    thread_local std::unordered_map<MatmulKey, std::unique_ptr<MatmulPlan>, MatmulKeyHash> plans;
    if (const auto found = plans.find(key); found != plans.end()) return *found->second;
    constexpr std::size_t max_cached_plans = 64;
    if (plans.size() == max_cached_plans) plans.clear();
    auto plan = std::make_unique<MatmulPlan>(handle, key);
    MatmulPlan& result = *plan;
    plans.emplace(key, std::move(plan));
    return result;
}

__device__ __forceinline__ float warp_sum(float value) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) value += __shfl_down_sync(0xffffffff, value, offset);
    return value;
}

template <typename T>
__global__ void reduce_bias_kernel(T* __restrict__ grad_bias, const T* __restrict__ grad_output,
                                   std::size_t rows, std::size_t columns, bool accumulate) {
    const std::size_t column = blockIdx.x;
    float sum = 0.0f;
    for (std::size_t row = threadIdx.x; row < rows; row += blockDim.x)
        sum += static_cast<float>(grad_output[row * columns + column]);
    sum = warp_sum(sum);
    __shared__ float warp_sums[8];
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int warp = static_cast<int>(threadIdx.x) >> 5;
    if (lane == 0) warp_sums[warp] = sum;
    __syncthreads();
    if (warp == 0) {
        sum = lane < (blockDim.x + 31) / 32 ? warp_sums[lane] : 0.0f;
        sum = warp_sum(sum);
        if (lane == 0) grad_bias[column] = static_cast<T>(accumulate ?
            static_cast<float>(grad_bias[column]) + sum : sum);
    }
}

void launch_bias_reduction(Tensor& grad_bias, const Tensor& grad_output, std::size_t rows,
                           bool accumulate, cudaStream_t stream) {
    constexpr unsigned threads = 256;
    const std::size_t columns = grad_bias.numel();
    if (columns > std::numeric_limits<unsigned>::max())
        throw std::invalid_argument("linear_backward: out_features exceeds CUDA grid.x range");
    const dim3 blocks(static_cast<unsigned>(columns));
    if (grad_bias.dtype() == Dtype::F16) reduce_bias_kernel<<<blocks, threads, 0, stream>>>(
        static_cast<__half*>(grad_bias.raw_data()), static_cast<const __half*>(grad_output.raw_data()), rows, columns, accumulate);
    else if (grad_bias.dtype() == Dtype::BF16) reduce_bias_kernel<<<blocks, threads, 0, stream>>>(
        static_cast<__nv_bfloat16*>(grad_bias.raw_data()), static_cast<const __nv_bfloat16*>(grad_output.raw_data()), rows, columns, accumulate);
    else reduce_bias_kernel<<<blocks, threads, 0, stream>>>(
        static_cast<float*>(grad_bias.raw_data()), static_cast<const float*>(grad_output.raw_data()), rows, columns, accumulate);
    CUDA_CHECK(cudaGetLastError());
}

void validate_linear_backward(const Tensor& grad_input, const Tensor& grad_weight, const Tensor& grad_bias,
                              const Tensor& grad_output, const Tensor& input, const Tensor& weight,
                              const LinearBackwardOptions& options) {
    if (input.dim() < 2) throw std::invalid_argument("linear_backward: input must have at least 2 dimensions");
    if (weight.dim() != 2) throw std::invalid_argument("linear_backward: weight must have shape [out_features, in_features]");
    if (grad_output.dim() != input.dim() || grad_input.shape() != input.shape())
        throw std::invalid_argument("linear_backward: grad_output and grad_input must match input rank and leading dimensions");
    const std::size_t input_features = input.size(input.dim() - 1), output_features = weight.size(0);
    if (weight.size(1) != input_features) throw std::invalid_argument("linear_backward: input feature size does not match weight");
    for (std::size_t axis = 0; axis + 1 < input.dim(); ++axis)
        if (grad_output.size(axis) != input.size(axis)) throw std::invalid_argument("linear_backward: grad_output leading dimensions must match input");
    if (grad_output.size(grad_output.dim() - 1) != output_features || grad_weight.dim() != 2 ||
        grad_weight.size(0) != output_features || grad_weight.size(1) != input_features ||
        grad_bias.dim() != 1 || grad_bias.size(0) != output_features)
        throw std::invalid_argument("linear_backward: invalid gradient shape");
    const Tensor* tensors[] = {&grad_input, &grad_weight, &grad_bias, &grad_output, &input, &weight};
    for (const Tensor* tensor : tensors) {
        if (tensor->device_type() != DeviceType::CUDA) throw std::invalid_argument("linear_backward: tensors must be on a CUDA device");
        if (!is_floating_point(tensor->dtype()) || tensor->dtype() != input.dtype())
            throw std::invalid_argument("linear_backward: all tensors must have the same floating dtype");
        require_same_device(input, *tensor);
    }
    if (!is_valid_compute_type(options.compute_type)) throw std::invalid_argument("linear_backward: invalid compute type");
}

} // namespace

void linear_backward(Tensor& grad_input, Tensor& grad_weight, Tensor& grad_bias, const Tensor& grad_output,
                     const Tensor& input, const Tensor& weight, const CublasLtContext& context,
                     cudaStream_t stream, const LinearBackwardOptions& options) {
    validate_linear_backward(grad_input, grad_weight, grad_bias, grad_output, input, weight, options);
    const std::size_t input_features = input.size(input.dim() - 1);
    const std::size_t output_features = weight.size(0);
    const std::size_t rows = input.numel() / input_features;
    DeviceGuard device_guard(context.device_index());
    const auto make_key = [&](std::size_t m, std::size_t n, std::size_t k, cublasOperation_t trans_a) {
        return MatmulKey{reinterpret_cast<std::uintptr_t>(context.handle()), reinterpret_cast<std::uintptr_t>(stream),
            m, n, k, options.workspace_bytes, input.dtype(), options.compute_type, trans_a, CUBLAS_OP_N};
    };
    cached_plan(context.handle(), make_key(rows, input_features, output_features, CUBLAS_OP_N)).execute(
        context.handle(), grad_input, grad_output, weight, options.accumulate_input, stream);
    cached_plan(context.handle(), make_key(output_features, input_features, rows, CUBLAS_OP_T)).execute(
        context.handle(), grad_weight, grad_output, input, options.accumulate_weight, stream);
    launch_bias_reduction(grad_bias, grad_output, rows, options.accumulate_bias, stream);
}
