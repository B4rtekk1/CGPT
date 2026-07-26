#include "ops/linear.h"
#include "core/device_buffer.h"
#include "core/device_guard.h"

#include <cublasLt.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace {

const char* device_type_name(const DeviceType device_type) {
    switch (device_type) {
    case DeviceType::CPU:
        return "CPU";
    case DeviceType::CUDA:
        return "CUDA";
    }
    return "unknown";
}

void require_same_device(const Tensor& a, const Tensor& b) {
    if (a.device_type() != b.device_type()) {
        throw std::invalid_argument(
            "Tensors must be on the same device, got " +
            std::string(device_type_name(a.device_type())) + " and " +
            std::string(device_type_name(b.device_type())));
    }
}

void set_row_major(cublasLtMatrixLayout_t layout) {
    const cublasLtOrder_t order = CUBLASLT_ORDER_ROW;
    CUBLAS_CHECK(cublasLtMatrixLayoutSetAttribute(
        layout, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order)));
}

template <typename T>
__global__ void add_bias_kernel(
    T* output,
    const T* bias,
    const std::size_t rows,
    const std::size_t columns
) {
    const std::size_t column =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (column >= columns) {
        return;
    }

    // A 2-D mapping avoids an integer modulo for every output element.
    for (std::size_t row = blockIdx.y; row < rows; row += gridDim.y) {
        const std::size_t index = row * columns + column;
        output[index] = static_cast<T>(
            static_cast<float>(output[index]) +
            static_cast<float>(bias[column]));
    }
}

void launch_bias_fallback(
    Tensor& output,
    const Tensor& bias,
    cudaStream_t stream
) {
    constexpr int threads = 256;
    const std::size_t columns = bias.numel();
    const std::size_t rows = output.numel() / columns;
    constexpr std::size_t max_grid_y = 65535;
    const dim3 blocks(
        static_cast<unsigned>((columns + threads - 1) / threads),
        static_cast<unsigned>(rows < max_grid_y ? rows : max_grid_y));

    if (output.dtype() == Dtype::F16) {
        add_bias_kernel<<<blocks, threads, 0, stream>>>(
            static_cast<__half*>(output.raw_data()),
            static_cast<const __half*>(bias.raw_data()),
            rows,
            columns);
    } else if (output.dtype() == Dtype::BF16) {
        add_bias_kernel<<<blocks, threads, 0, stream>>>(
            static_cast<__nv_bfloat16*>(output.raw_data()),
            static_cast<const __nv_bfloat16*>(bias.raw_data()),
            rows,
            columns);
    } else {
        add_bias_kernel<<<blocks, threads, 0, stream>>>(
            static_cast<float*>(output.raw_data()),
            static_cast<const float*>(bias.raw_data()),
            rows,
            columns);
    }
    CUDA_CHECK(cudaGetLastError());
}

struct PlanKey {
    std::uintptr_t handle;
    std::uintptr_t stream;
    std::size_t rows;
    std::size_t input_dim;
    std::size_t output_dim;
    std::size_t workspace_bytes;
    Dtype dtype;
    ComputeType compute_type;
    bool has_bias;

    bool operator==(const PlanKey&) const = default;
};

struct PlanKeyHash {
    std::size_t operator()(const PlanKey& key) const noexcept {
        std::size_t seed = 0;
        const auto combine = [&seed](const std::size_t value) {
            seed ^= value + 0x9e3779b97f4a7c15ULL + (seed << 6U) + (seed >> 2U);
        };
        combine(key.handle);
        combine(key.stream);
        combine(key.rows);
        combine(key.input_dim);
        combine(key.output_dim);
        combine(key.workspace_bytes);
        combine(static_cast<std::size_t>(key.dtype));
        combine(static_cast<std::size_t>(key.compute_type));
        combine(key.has_bias);
        return seed;
    }
};

class MatmulPlan {
public:
    MatmulPlan(
        cublasLtHandle_t handle,
        const PlanKey& key
    ) {
        const cudaDataType_t data_type = to_cuda_dtype(key.dtype);
        // Tensor Cores also accelerate irregular, model-sized matrices.
        // Requiring every dimension to be divisible by eight unnecessarily
        // sent M=15/31/127 to FP32. Very small matrices remain on FP32: they
        // are bandwidth/launch-bound and TF32 can be slower there.
        const bool use_tf32 =
            key.dtype == Dtype::F32 &&
            key.compute_type == ComputeType::TF32 &&
            key.rows >= 8 &&
            key.input_dim >= 8 &&
            key.output_dim >= 8;
        const cublasComputeType_t compute_type =
            use_tf32 ? CUBLAS_COMPUTE_32F_FAST_TF32 : CUBLAS_COMPUTE_32F;

        CUBLAS_CHECK(cublasLtMatmulDescCreate(
            &operation_, compute_type, CUDA_R_32F));

        const cublasOperation_t transpose_weight = CUBLAS_OP_T;
        CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(
            operation_,
            CUBLASLT_MATMUL_DESC_TRANSB,
            &transpose_weight,
            sizeof(transpose_weight)));

        if (use_tf32) {
            // Older cuBLASLt releases may reject TF32 for a few tiny or
            // irregular shapes. Keep a descriptor-only FP32 fallback without
            // penalizing the normal Tensor-Core path.
            CUBLAS_CHECK(cublasLtMatmulDescCreate(
                &fp32_fallback_operation_,
                CUBLAS_COMPUTE_32F,
                CUDA_R_32F));
            CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(
                fp32_fallback_operation_,
                CUBLASLT_MATMUL_DESC_TRANSB,
                &transpose_weight,
                sizeof(transpose_weight)));
        }

        if (key.has_bias) {
            CUBLAS_CHECK(cublasLtMatmulDescCreate(
                &unfused_operation_, compute_type, CUDA_R_32F));
            CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(
                unfused_operation_,
                CUBLASLT_MATMUL_DESC_TRANSB,
                &transpose_weight,
                sizeof(transpose_weight)));

            fused_bias_supported_ = true;
            const cublasLtEpilogue_t epilogue = CUBLASLT_EPILOGUE_BIAS;
            CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(
                operation_,
                CUBLASLT_MATMUL_DESC_EPILOGUE,
                &epilogue,
                sizeof(epilogue)));
            CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(
                operation_,
                CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE,
                &data_type,
                sizeof(data_type)));
        }

        CUBLAS_CHECK(cublasLtMatrixLayoutCreate(
            &input_layout_, data_type, key.rows, key.input_dim, key.input_dim));
        CUBLAS_CHECK(cublasLtMatrixLayoutCreate(
            &weight_layout_, data_type, key.output_dim, key.input_dim, key.input_dim));
        CUBLAS_CHECK(cublasLtMatrixLayoutCreate(
            &output_layout_, data_type, key.rows, key.output_dim, key.output_dim));

        set_row_major(input_layout_);
        set_row_major(weight_layout_);
        set_row_major(output_layout_);

        cublasLtMatmulPreference_t preference = nullptr;
        CUBLAS_CHECK(cublasLtMatmulPreferenceCreate(&preference));
        CUBLAS_CHECK(cublasLtMatmulPreferenceSetAttribute(
            preference,
            CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
            &key.workspace_bytes,
            sizeof(key.workspace_bytes)));

        cublasLtMatmulHeuristicResult_t heuristic{};
        int returned_results = 0;
        const cublasStatus_t heuristic_status = cublasLtMatmulAlgoGetHeuristic(
            handle,
            operation_,
            input_layout_,
            weight_layout_,
            output_layout_,
            output_layout_,
            preference,
            1,
            &heuristic,
            &returned_results);
        CUBLAS_CHECK(cublasLtMatmulPreferenceDestroy(preference));

        // Some cuBLASLt versions do not expose heuristic algorithms for very
        // small or unusual shapes even though the library's default algorithm
        // supports them. Keep that compatibility path; normal model-sized
        // GEMMs retain the explicitly selected and cached algorithm.
        if (heuristic_status == CUBLAS_STATUS_SUCCESS &&
            returned_results > 0 &&
            heuristic.state == CUBLAS_STATUS_SUCCESS) {
            algorithm_ = heuristic.algo;
            workspace_required_ = heuristic.workspaceSize;
            workspace_.allocate(workspace_required_);
            has_selected_algorithm_ = true;
        } else if (heuristic_status != CUBLAS_STATUS_NOT_SUPPORTED &&
                   heuristic_status != CUBLAS_STATUS_SUCCESS) {
            CUBLAS_CHECK(heuristic_status);
        }
    }

    ~MatmulPlan() {
        if (input_layout_ != nullptr) {
            cublasLtMatrixLayoutDestroy(input_layout_);
        }
        if (weight_layout_ != nullptr) {
            cublasLtMatrixLayoutDestroy(weight_layout_);
        }
        if (output_layout_ != nullptr) {
            cublasLtMatrixLayoutDestroy(output_layout_);
        }
        if (operation_ != nullptr) {
            cublasLtMatmulDescDestroy(operation_);
        }
        if (unfused_operation_ != nullptr) {
            cublasLtMatmulDescDestroy(unfused_operation_);
        }
        if (fp32_fallback_operation_ != nullptr) {
            cublasLtMatmulDescDestroy(fp32_fallback_operation_);
        }
    }

    MatmulPlan(const MatmulPlan&) = delete;
    MatmulPlan& operator=(const MatmulPlan&) = delete;

    void execute(
        cublasLtHandle_t handle,
        Tensor& output,
        const Tensor& input,
        const Tensor& weight,
        const Tensor* bias,
        cudaStream_t stream
    ) {
        if (bias != nullptr && fused_bias_supported_) {
            const void* bias_pointer = bias->raw_data();
            CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(
                operation_,
                CUBLASLT_MATMUL_DESC_BIAS_POINTER,
                &bias_pointer,
                sizeof(bias_pointer)));
        }

        constexpr float alpha = 1.0f;
        constexpr float beta = 0.0f;
        const auto launch = [&](
            const cublasLtMatmulAlgo_t* algorithm,
            cublasLtMatmulDesc_t operation
        ) {
            const bool uses_workspace =
                algorithm != nullptr && workspace_required_ != 0;
            return cublasLtMatmul(
                handle,
                operation,
                &alpha,
                input.raw_data(),
                input_layout_,
                weight.raw_data(),
                weight_layout_,
                &beta,
                output.raw_data(),
                output_layout_,
                output.raw_data(),
                output_layout_,
                algorithm,
                uses_workspace ? workspace_.data() : nullptr,
                uses_workspace ? workspace_required_ : 0,
                stream);
        };

        const cublasLtMatmulDesc_t active_operation =
            bias != nullptr && !fused_bias_supported_
                ? unfused_operation_
                : operation_;
        cublasStatus_t status =
            launch(
                has_selected_algorithm_ ? &algorithm_ : nullptr,
                active_operation);
        if (status == CUBLAS_STATUS_NOT_SUPPORTED && has_selected_algorithm_) {
            has_selected_algorithm_ = false;
            workspace_required_ = 0;
            status = launch(nullptr, active_operation);
        }
        bool needs_bias_fallback = bias != nullptr && !fused_bias_supported_;
        if (status == CUBLAS_STATUS_NOT_SUPPORTED &&
            bias != nullptr &&
            fused_bias_supported_) {
            // Some small/irregular shapes have no cuBLASLt kernel supporting
            // the bias epilogue. Preserve correctness with a separate kernel;
            // model-sized compatible shapes remain fused.
            fused_bias_supported_ = false;
            has_selected_algorithm_ = false;
            workspace_required_ = 0;
            status = launch(nullptr, unfused_operation_);
            needs_bias_fallback = true;
        }
        if (status == CUBLAS_STATUS_NOT_SUPPORTED &&
            fp32_fallback_operation_ != nullptr) {
            // Preserve compatibility with CUDA versions that cannot execute
            // an irregular TF32 operation. The fallback intentionally has no
            // epilogue, so bias is added by the coalesced 2-D kernel.
            has_selected_algorithm_ = false;
            workspace_required_ = 0;
            status = launch(nullptr, fp32_fallback_operation_);
            needs_bias_fallback = bias != nullptr;
        }
        CUBLAS_CHECK(status);
        if (needs_bias_fallback) {
            launch_bias_fallback(output, *bias, stream);
        }
    }

private:
    cublasLtMatmulDesc_t operation_ = nullptr;
    cublasLtMatmulDesc_t unfused_operation_ = nullptr;
    cublasLtMatmulDesc_t fp32_fallback_operation_ = nullptr;
    cublasLtMatrixLayout_t input_layout_ = nullptr;
    cublasLtMatrixLayout_t weight_layout_ = nullptr;
    cublasLtMatrixLayout_t output_layout_ = nullptr;
    cublasLtMatmulAlgo_t algorithm_{};
    DeviceBuffer workspace_;
    std::size_t workspace_required_ = 0;
    bool has_selected_algorithm_ = false;
    bool fused_bias_supported_ = false;
};

MatmulPlan& cached_plan(
    cublasLtHandle_t handle,
    const PlanKey& key
) {
    thread_local std::unordered_map<
        PlanKey,
        std::unique_ptr<MatmulPlan>,
        PlanKeyHash> plans;

    if (const auto found = plans.find(key); found != plans.end()) {
        return *found->second;
    }

    // Bound memory retained by workloads with highly dynamic shapes.
    constexpr std::size_t max_cached_plans = 64;
    if (plans.size() >= max_cached_plans) {
        plans.clear();
    }

    auto plan = std::make_unique<MatmulPlan>(handle, key);
    MatmulPlan& result = *plan;
    plans.emplace(key, std::move(plan));
    return result;
}

void validate_linear(
    const Tensor& output,
    const Tensor& input,
    const Tensor& weight,
    const Tensor* bias
) {
    if (input.dim() < 2) {
        throw std::invalid_argument(
            "linear_forward: input must have at least 2 dimensions");
    }
    if (weight.dim() != 2) {
        throw std::invalid_argument(
            "linear_forward: weight must have shape [out_features, in_features]");
    }
    if (output.dim() != input.dim()) {
        throw std::invalid_argument(
            "linear_forward: output must have the same number of dimensions as input");
    }
    if (!is_floating_point(input.dtype()) ||
        input.dtype() != weight.dtype() ||
        input.dtype() != output.dtype()) {
        throw std::invalid_argument(
            "linear_forward: input, weight, and output must have the same floating dtype");
    }

    require_same_device(input, weight);
    require_same_device(input, output);
    if (input.device_type() != DeviceType::CUDA) {
        throw std::invalid_argument(
            "linear_forward: tensors must be on a CUDA device");
    }

    const std::size_t input_dim = input.size(input.dim() - 1);
    const std::size_t output_dim = weight.size(0);
    if (input_dim != weight.size(1)) {
        throw std::invalid_argument(
            "linear_forward: input feature size does not match weight");
    }
    for (std::size_t axis = 0; axis + 1 < input.dim(); ++axis) {
        if (input.size(axis) != output.size(axis)) {
            throw std::invalid_argument(
                "linear_forward: output leading dimensions must match input");
        }
    }
    if (output.size(output.dim() - 1) != output_dim) {
        throw std::invalid_argument(
            "linear_forward: output last dimension must equal out_features");
    }

    if (bias != nullptr) {
        require_same_device(input, *bias);
        if (bias->dim() != 1 || bias->size(0) != output_dim) {
            throw std::invalid_argument(
                "linear_forward: bias must have shape [out_features]");
        }
        if (bias->dtype() != input.dtype()) {
            throw std::invalid_argument(
                "linear_forward: bias dtype must match input dtype");
        }
    }
}

void linear_forward_impl(
    Tensor& output,
    const Tensor& input,
    const Tensor& weight,
    const Tensor* bias,
    const CublasLtContext& context,
    cudaStream_t stream,
    const LinearOptions& options
) {
    validate_linear(output, input, weight, bias);
    if (!is_valid_compute_type(options.compute_type)) {
        throw std::invalid_argument("linear_forward: invalid compute type");
    }

    const std::size_t input_dim = input.size(input.dim() - 1);
    const std::size_t output_dim = weight.size(0);
    const std::size_t rows = input.numel() / input_dim;

    DeviceGuard device_guard(context.device_index());
    const PlanKey key{
        reinterpret_cast<std::uintptr_t>(context.handle()),
        reinterpret_cast<std::uintptr_t>(stream),
        rows,
        input_dim,
        output_dim,
        options.workspace_bytes,
        input.dtype(),
        options.compute_type,
        bias != nullptr
    };

    cached_plan(context.handle(), key).execute(
        context.handle(), output, input, weight, bias, stream);
}

} // namespace

void linear_forward(
    Tensor& output,
    const Tensor& input,
    const Tensor& weight,
    const CublasLtContext& cublas_context,
    cudaStream_t stream,
    const LinearOptions& options
) {
    linear_forward_impl(
        output, input, weight, nullptr, cublas_context, stream, options);
}

void linear_forward(
    Tensor& output,
    const Tensor& input,
    const Tensor& weight,
    const Tensor& bias,
    const CublasLtContext& cublas_context,
    cudaStream_t stream,
    const LinearOptions& options
) {
    linear_forward_impl(
        output, input, weight, &bias, cublas_context, stream, options);
}
