#include "core/transformer_block.h"
#include "core/cuda_check.h"
#include "ops/rmsnorm.h"
#include "ops/rope.h"
#include "ops/swiglu.h"
#include "ops/backward/flash_attention_backward.h"
#include "ops/backward/linear_backward.h"
#include "ops/backward/rmsnorm_backward.h"
#include "ops/backward/rope_backward.h"
#include "ops/backward/swiglu_backward.h"

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

    __device__ __forceinline__ half2 packed_add(const half2 left, const half2 right) {
        return __hadd2(left, right);
    }

    __device__ __forceinline__ __nv_bfloat162 packed_add(
        const __nv_bfloat162 left, const __nv_bfloat162 right) {
        return __hadd2(left, right);
    }

    template <typename PackedT>
    __global__ void add_packed_kernel(
        PackedT* __restrict__ output,
        const PackedT* __restrict__ left,
        const PackedT* __restrict__ right,
        const std::size_t count
    ) {
        const std::size_t index =
            static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
        if (index < count) output[index] = packed_add(left[index], right[index]);
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
        } else if (output.dtype() == Dtype::F16 && count % 2 == 0) {
            const std::size_t packed_count = count / 2;
            const int packed_blocks = static_cast<int>(
                (packed_count + kElementwiseThreads - 1) / kElementwiseThreads);
            add_packed_kernel<half2><<<packed_blocks, kElementwiseThreads, 0, stream>>>(
                static_cast<half2*>(output.raw_data()),
                static_cast<const half2*>(left.raw_data()),
                static_cast<const half2*>(right.raw_data()), packed_count);
        } else if (output.dtype() == Dtype::BF16 && count % 2 == 0) {
            const std::size_t packed_count = count / 2;
            const int packed_blocks = static_cast<int>(
                (packed_count + kElementwiseThreads - 1) / kElementwiseThreads);
            add_packed_kernel<__nv_bfloat162><<<packed_blocks, kElementwiseThreads, 0, stream>>>(
                static_cast<__nv_bfloat162*>(output.raw_data()),
                static_cast<const __nv_bfloat162*>(left.raw_data()),
                static_cast<const __nv_bfloat162*>(right.raw_data()), packed_count);
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

    template <typename T>
    __global__ void add_inplace_kernel(T* output, const T* source, const std::size_t count) {
        const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
        if (index < count) output[index] = static_cast<T>(static_cast<float>(output[index]) + static_cast<float>(source[index]));
    }

    template <typename PackedT>
    __global__ void add_inplace_packed_kernel(
        PackedT* output, const PackedT* source, const std::size_t count) {
        const std::size_t index =
            static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
        if (index < count) output[index] = packed_add(output[index], source[index]);
    }

    void add_inplace(Tensor& destination, const Tensor& source, cudaStream_t stream) {
        if (destination.shape() != source.shape() || destination.dtype() != source.dtype() ||
            destination.device_type() != DeviceType::CUDA || source.device_type() != DeviceType::CUDA) {
            throw std::invalid_argument("transformer block backward: incompatible tensor addition");
        }
        const int blocks = static_cast<int>((destination.numel() + kElementwiseThreads - 1) / kElementwiseThreads);
        if (destination.dtype() == Dtype::F32) add_inplace_kernel<float><<<blocks, kElementwiseThreads, 0, stream>>>(static_cast<float*>(destination.raw_data()), static_cast<const float*>(source.raw_data()), destination.numel());
        else if (destination.dtype() == Dtype::F16 && destination.numel() % 2 == 0) {
            const std::size_t count = destination.numel() / 2;
            const int packed_blocks = static_cast<int>((count + kElementwiseThreads - 1) / kElementwiseThreads);
            add_inplace_packed_kernel<half2><<<packed_blocks, kElementwiseThreads, 0, stream>>>(static_cast<half2*>(destination.raw_data()), static_cast<const half2*>(source.raw_data()), count);
        } else if (destination.dtype() == Dtype::BF16 && destination.numel() % 2 == 0) {
            const std::size_t count = destination.numel() / 2;
            const int packed_blocks = static_cast<int>((count + kElementwiseThreads - 1) / kElementwiseThreads);
            add_inplace_packed_kernel<__nv_bfloat162><<<packed_blocks, kElementwiseThreads, 0, stream>>>(static_cast<__nv_bfloat162*>(destination.raw_data()), static_cast<const __nv_bfloat162*>(source.raw_data()), count);
        } else if (destination.dtype() == Dtype::F16) add_inplace_kernel<half><<<blocks, kElementwiseThreads, 0, stream>>>(static_cast<half*>(destination.raw_data()), static_cast<const half*>(source.raw_data()), destination.numel());
        else add_inplace_kernel<__nv_bfloat16><<<blocks, kElementwiseThreads, 0, stream>>>(static_cast<__nv_bfloat16*>(destination.raw_data()), static_cast<const __nv_bfloat16*>(source.raw_data()), destination.numel());
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

TransformerBlockBackwardWorkspace::TransformerBlockBackwardWorkspace(
    const TransformerBlockOptions& options, const std::size_t batch_size,
    const std::size_t sequence_length, const Dtype dtype)
    : grad_attention_norm_input({batch_size * sequence_length, options.hidden_size}, dtype),
      attention_norm_output({batch_size * sequence_length, options.hidden_size}, dtype),
      grad_ffn_norm_input({batch_size * sequence_length, options.hidden_size}, dtype),
      grad_residual({batch_size * sequence_length, options.hidden_size}, dtype),
      grad_attention_projection({batch_size * sequence_length, options.hidden_size}, dtype),
      grad_attention_output({batch_size, sequence_length, options.num_query_heads, options.head_dim}, dtype),
      grad_query({batch_size, sequence_length, options.num_query_heads, options.head_dim}, dtype),
      grad_key({batch_size, sequence_length, options.num_kv_heads, options.head_dim}, dtype),
      grad_value({batch_size, sequence_length, options.num_kv_heads, options.head_dim}, dtype),
      grad_query_pre_rope({batch_size, sequence_length, options.num_query_heads, options.head_dim}, dtype),
      grad_key_pre_rope({batch_size, sequence_length, options.num_kv_heads, options.head_dim}, dtype),
      grad_gate({batch_size * sequence_length, options.intermediate_size}, dtype),
      grad_up({batch_size * sequence_length, options.intermediate_size}, dtype),
      grad_activated({batch_size * sequence_length, options.intermediate_size}, dtype),
      hidden_bias({options.hidden_size}, dtype),
      intermediate_bias({options.intermediate_size}, dtype),
      kv_bias({options.num_kv_heads * options.head_dim}, dtype) {
    validate_options(options);
    if (batch_size == 0 || sequence_length == 0 || !is_floating_point(dtype))
        throw std::invalid_argument("transformer block backward workspace requires positive dimensions and a floating dtype");
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
    require_shape(workspace.attention_logsumexp,
                  {batch, sequence, options.num_query_heads},
                  "attention_logsumexp");
    if (workspace.attention_logsumexp.dtype() != Dtype::F32) {
        throw std::invalid_argument(
            "attention_logsumexp must use F32 storage");
    }
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

    flash_gqa_attention_forward_with_lse(
        workspace.attention_output,
        workspace.attention_logsumexp,
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

void transformer_block_backward(
    Tensor& grad_input, const Tensor& grad_output, const Tensor& input,
    const TransformerBlockWeights& weights, const TransformerBlockGradients gradients,
    const TransformerBlockWorkspace& forward_workspace, TransformerBlockBackwardWorkspace& backward,
    const Tensor& cos_cache, const Tensor& sin_cache, const CublasLtContext& cublas_context,
    cudaStream_t stream, const TransformerBlockOptions& options, const std::size_t position_offset) {
    validate_options(options);
    if (input.shape() != grad_output.shape() || grad_input.shape() != input.shape() || input.dim() != 2 ||
        input.device_type() != DeviceType::CUDA || grad_output.device_type() != DeviceType::CUDA ||
        grad_input.device_type() != DeviceType::CUDA || input.dtype() != grad_output.dtype() || input.dtype() != grad_input.dtype())
        throw std::invalid_argument("transformer block backward: invalid input or output gradient");
    const std::size_t rows = input.size(0), hidden = options.hidden_size;
    const std::size_t batch = forward_workspace.query.size(0), sequence = forward_workspace.query.size(1);
    if (input.size(1) != hidden || batch * sequence != rows || forward_workspace.norm.shape() != std::vector<std::size_t>{rows, hidden})
        throw std::invalid_argument("transformer block backward: forward workspace does not match input");

    LinearBackwardOptions linear_options{options.linear_options.compute_type, options.linear_options.workspace_bytes};
    linear_backward(backward.grad_activated, gradients.down_proj, backward.hidden_bias, grad_output,
        forward_workspace.activated, weights.down_proj, cublas_context, stream, linear_options);
    swiglu_backward(backward.grad_gate, backward.grad_up, backward.grad_activated,
        forward_workspace.gate, forward_workspace.up, stream);
    linear_backward(backward.grad_ffn_norm_input, gradients.gate_proj, backward.intermediate_bias,
        backward.grad_gate, forward_workspace.norm, weights.gate_proj, cublas_context, stream, linear_options);
    linear_options.accumulate_input = true;
    linear_backward(backward.grad_ffn_norm_input, gradients.up_proj, backward.intermediate_bias,
        backward.grad_up, forward_workspace.norm, weights.up_proj, cublas_context, stream, linear_options);
    linear_options.accumulate_input = false;

    CUDA_CHECK(cudaMemcpyAsync(backward.grad_attention_projection.raw_data(), input.raw_data(), input.nbytes(), cudaMemcpyDeviceToDevice, stream));
    add_inplace(backward.grad_attention_projection, forward_workspace.attention_projection, stream);
    rmsnorm_backward(backward.grad_residual, gradients.ffn_norm, backward.grad_ffn_norm_input,
        backward.grad_attention_projection, weights.ffn_norm, options.rms_epsilon, stream);

    CUDA_CHECK(cudaMemcpyAsync(grad_input.raw_data(), grad_output.raw_data(), grad_output.nbytes(), cudaMemcpyDeviceToDevice, stream));
    add_inplace(grad_input, backward.grad_residual, stream);
    CUDA_CHECK(cudaMemcpyAsync(backward.grad_attention_projection.raw_data(), backward.grad_residual.raw_data(), backward.grad_residual.nbytes(), cudaMemcpyDeviceToDevice, stream));
    add_inplace(backward.grad_attention_projection, grad_output, stream);

    backward.grad_attention_output.reshape({rows, hidden});
    auto& saved_attention_output = const_cast<Tensor&>(forward_workspace.attention_output);
    saved_attention_output.reshape({rows, hidden});
    linear_backward(backward.grad_attention_output, gradients.o_projection, backward.hidden_bias,
        backward.grad_attention_projection, saved_attention_output, weights.o_projection,
        cublas_context, stream, linear_options);
    saved_attention_output.reshape({batch, sequence, options.num_query_heads, options.head_dim});
    backward.grad_attention_output.reshape({batch, sequence, options.num_query_heads, options.head_dim});
    FlashAttentionOptions attention_options{};
    attention_options.num_query_heads = options.num_query_heads;
    attention_options.num_kv_heads = options.num_kv_heads;
    attention_options.head_dim = options.head_dim;
    attention_options.causal = options.causal;
    attention_options.query_position_offset = position_offset;
    flash_gqa_attention_backward_with_lse(
        backward.grad_query, backward.grad_key, backward.grad_value,
        backward.grad_attention_output, forward_workspace.attention_output,
        forward_workspace.attention_logsumexp, forward_workspace.query,
        forward_workspace.key, forward_workspace.value, stream, attention_options);
    rope_backward(backward.grad_query_pre_rope, backward.grad_key_pre_rope, backward.grad_query,
        backward.grad_key, cos_cache, sin_cache, stream, RopeOptions{options.rotary_dim, position_offset});

    rmsnorm_forward(backward.attention_norm_output, input, weights.attention_norm, options.rms_epsilon, stream);
    backward.grad_query_pre_rope.reshape({rows, options.num_query_heads * options.head_dim});
    backward.grad_key_pre_rope.reshape({rows, options.num_kv_heads * options.head_dim});
    backward.grad_value.reshape({rows, options.num_kv_heads * options.head_dim});
    linear_backward(backward.grad_attention_norm_input, gradients.q_projection, backward.hidden_bias,
        backward.grad_query_pre_rope, backward.attention_norm_output, weights.q_projection, cublas_context, stream, linear_options);
    linear_options.accumulate_input = true;
    linear_backward(backward.grad_attention_norm_input, gradients.k_projection, backward.kv_bias,
        backward.grad_key_pre_rope, backward.attention_norm_output, weights.k_projection, cublas_context, stream, linear_options);
    linear_backward(backward.grad_attention_norm_input, gradients.v_projection, backward.kv_bias,
        backward.grad_value, backward.attention_norm_output, weights.v_projection, cublas_context, stream, linear_options);
    linear_options.accumulate_input = false;
    backward.grad_query_pre_rope.reshape({batch, sequence, options.num_query_heads, options.head_dim});
    backward.grad_key_pre_rope.reshape({batch, sequence, options.num_kv_heads, options.head_dim});
    backward.grad_value.reshape({batch, sequence, options.num_kv_heads, options.head_dim});
    rmsnorm_backward(backward.grad_residual, gradients.attention_norm, backward.grad_attention_norm_input,
        input, weights.attention_norm, options.rms_epsilon, stream);
    add_inplace(grad_input, backward.grad_residual, stream);
}
