#include "core/transformer_block.h"
#include "core/cuda_check.h"
#include "ops/rmsnorm.h"
#include "ops/rope.h"
#include "ops/swiglu.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <stdexcept>
#include <string>
#include <vector>

#include "ops/flash_attention.h"

namespace {

    constexpr int kElementwiseThreads = 256;

    template <typename T>
    __global__ void add_kernel(
    T* __restrict__ output,
    const T* __restrict__ left,
    const T* __restrict__ right,
    const std::size_t count
    ) {
        const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
        const std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;

        for (std::size_t i = index; i < count; i += stride) {
            output[i] = left[i] + right[i];
        }
    }

    template <>
    __global__ void add_kernel<half>(
        half* __restrict__ output,
        const half* __restrict__ left,
        const half* __restrict__ right,
        const std::size_t count
    ) {
        const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
        const std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;

        for (std::size_t i = index; i < count; i += stride) {
            output[i] = __hadd(left[i], right[i]);
        }
    }

    template <>
    __global__ void add_kernel<__nv_bfloat16>(
        __nv_bfloat16* __restrict__ output,
        const __nv_bfloat16* __restrict__ left,
        const __nv_bfloat16* __restrict__ right,
        const std::size_t count
    ) {
        const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
        const std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
        for (std::size_t i = index; i < count; i += stride) {
            output[i] = __hadd(left[i], right[i]);
        }
    }

    void add_tensors(
        Tensor& output,
        const Tensor& left,
        const Tensor& right,
        cudaStream_t stream
    ) {
        if (output.numel() != left.numel() || output.numel() != right.numel()) {
            throw std::runtime_error("add_tensors: tensor size mismatch");
        }

        const std::size_t count = output.numel();
        const int blocks = static_cast<int>((count + kElementwiseThreads - 1) / kElementwiseThreads);

        if (output.dtype() == Dtype::F32) {
            add_kernel<float><<<blocks, kElementwiseThreads, 0, stream>>>(
                static_cast<float*>(output.raw_data()),
                static_cast<const float*>(left.raw_data()),
                static_cast<const float*>(right.raw_data()),
                count
            );
        } else if (output.dtype() == Dtype::F16) {
            add_kernel<half><<<blocks, kElementwiseThreads, 0, stream>>>(
                static_cast<half*>(output.raw_data()),
                static_cast<const half*>(left.raw_data()),
                static_cast<const half*>(right.raw_data()),
                count
            );
        } else if (output.dtype() == Dtype::BF16) {
            add_kernel<__nv_bfloat16><<<blocks, kElementwiseThreads, 0, stream>>>(
                static_cast<__nv_bfloat16*>(output.raw_data()),
                static_cast<const __nv_bfloat16*>(left.raw_data()),
                static_cast<const __nv_bfloat16*>(right.raw_data()),
                count
            );
        } else {
            throw std::runtime_error("add_tensors: unsupported data type");
        }
        CUDA_CHECK(cudaGetLastError());
    }

    void require_shape(const Tensor& tensor, const std::vector<std::size_t>& shape, const char* name) {
        if (tensor.shape() != shape) {
            throw std::runtime_error(std::string(name) + ": tensor shape mismatch");
        }
    }

    void validate_options(const TransformerBlockOptions& options) {
        if (options.hidden_size ==0 || options.intermediate_size ==0 || options.num_query_heads == 0 ||
            options.num_kv_heads ==0 || options.head_dim == 0) {
            throw std::invalid_argument("transformer block dimensions must be non-zero");
        }
    }
}

void transformer_block_forward(
    Tensor& output,
    const Tensor& input,
    const TransformerBlockWeights& weights,
    TransformerBlockWorkspace& workspace,
    const Tensor& cos_cache,
    const Tensor& sin_cache,
    const CublasLtContext& cublas_context,
    cudaStream_t stream,
    const TransformerBlockOptions& options,
    std::size_t position_offset
) {
    validate_options(options);

    if (input.dim() != 2 || output.shape() != input.shape()) {
        throw std::invalid_argument("transformer block input/output must be [batch * sequence, hidden]");
    }

    const std::size_t rows = input.size(0);
    const std::size_t hidden_size = input.size(1);
    if (hidden_size != options.hidden_size) {
        throw std::invalid_argument("transformer block hidden size mismatch");
    }

    if (workspace.query.dim() != 4) {
        throw std::invalid_argument("workspace.query must initially be [batch, sequence, query_heads, head_dim]");
    }
    const std::size_t batch = workspace.query.size(0);
    const std::size_t sequence = workspace.query.size(1);
    if (batch * sequence != rows) {
        throw std::invalid_argument("workspace batch * sequence must be equal input rows");
    }

    require_shape(workspace.query,
                  {batch, sequence, options.num_query_heads, options.head_dim}, "query");
    require_shape(workspace.key,
                  {batch, sequence, options.num_kv_heads, options.head_dim}, "key");
    require_shape(workspace.value,
                  {batch, sequence, options.num_kv_heads, options.head_dim}, "value");
    require_shape(workspace.attention_output,
                  {batch, sequence, options.num_query_heads, options.head_dim},
                  "attention_output");
    require_shape(workspace.norm, {rows, hidden_size}, "norm");
    require_shape(workspace.attention_projection, {rows, hidden_size},
                  "attention_projection");
    require_shape(workspace.gate, {rows, options.intermediate_size}, "gate");
    require_shape(workspace.up, {rows, options.intermediate_size}, "up");
    require_shape(workspace.activated, {rows, options.intermediate_size},
                  "activated");
    require_shape(workspace.ffn_output, {rows, hidden_size}, "ffn_output");

    rmsnorm_forward(workspace.norm, input, weights.attention_norm,
                    options.rms_epsilon, stream);

    workspace.query.reshape({rows, options.num_query_heads * options.head_dim});
    workspace.key.reshape({rows, options.num_kv_heads * options.head_dim});
    workspace.value.reshape({rows, options.num_kv_heads * options.head_dim});

    linear_forward(workspace.query, workspace.norm, weights.q_projection,
                   cublas_context, stream, options.linear_options);
    linear_forward(workspace.key, workspace.norm, weights.k_projection,
                   cublas_context, stream, options.linear_options);
    linear_forward(workspace.value, workspace.norm, weights.v_projection,
                   cublas_context, stream, options.linear_options);

    workspace.query.reshape({batch, sequence, options.num_query_heads,
                             options.head_dim});
    workspace.key.reshape({batch, sequence, options.num_kv_heads,
                           options.head_dim});
    workspace.value.reshape({batch, sequence, options.num_kv_heads,
                             options.head_dim});

    rope_forward(
        workspace.query,
        workspace.key,
        cos_cache,
        sin_cache,
        stream,
        RopeOptions{options.rotary_dim, position_offset}
    );

    FlashAttentionOptions flash_attention_options{};
    flash_attention_options.num_query_heads = options.num_query_heads;
    flash_attention_options.num_kv_heads = options.num_kv_heads;
    flash_attention_options.head_dim = options.head_dim;
    flash_attention_options.causal = options.causal;
    flash_attention_options.query_position_offset = position_offset;

    flash_gqa_attention_forward(
        workspace.attention_output,
        workspace.query,
        workspace.key,
        workspace.value,
        stream,
        flash_attention_options
    );

    workspace.attention_output.reshape({rows, hidden_size});
    linear_forward(workspace.attention_projection, workspace.attention_output,
                   weights.o_projection, cublas_context, stream,
                   options.linear_options);

    add_tensors(output, input, workspace.attention_projection, stream);

    rmsnorm_forward(workspace.norm, output, weights.ffn_norm,
                    options.rms_epsilon, stream);

    linear_forward(workspace.gate, workspace.norm, weights.gate_proj,
                   cublas_context, stream, options.linear_options);
    linear_forward(workspace.up, workspace.norm, weights.up_proj,
                   cublas_context, stream, options.linear_options);
    swiglu_forward(workspace.activated, workspace.gate, workspace.up, stream);
    linear_forward(workspace.ffn_output, workspace.activated, weights.down_proj,
                   cublas_context, stream, options.linear_options);

    add_tensors(output, output, workspace.ffn_output, stream);

    workspace.query.reshape({batch, sequence, options.num_query_heads,
                             options.head_dim});
    workspace.key.reshape({batch, sequence, options.num_kv_heads,
                           options.head_dim});
    workspace.value.reshape({batch, sequence, options.num_kv_heads,
                             options.head_dim});
    workspace.attention_output.reshape(
        {batch, sequence, options.num_query_heads, options.head_dim});

}
