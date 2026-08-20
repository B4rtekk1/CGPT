/** @file transformer_block.cu CUDA implementation of a Transformer block and its backward pass. */

#include "core/transformer_block.h"
#include "core/cuda_check.h"
#include "ops/rmsnorm.h"
#include "ops/rope.h"
#include "ops/swiglu.h"
#include "ops/backward/attention_backward.h"
#include "ops/backward/linear_backward.h"
#include "ops/backward/rmsnorm_backward.h"
#include "ops/backward/rope_backward.h"
#include "ops/backward/swiglu_backward.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <stdexcept>
#include <optional>
#include <string>
#include <vector>

#include "ops/attention.h"

namespace {
    /**
     * @brief CUDA block size used by elementwise residual-addition kernels.
     */
    constexpr int kElementwiseThreads = 256;


    /**
     * @brief Adds two contiguous tensors elementwise.
     *
     * A grid-stride loop permits the kernel to process tensors larger than the
     * launched grid.
     *
     * @tparam T Scalar storage type.
     * @param output Destination device buffer.
     * @param left First source device buffer.
     * @param right Second source device buffer.
     * @param count Number of elements to add.
     */
    template<typename T>
    __global__ void add_kernel(
        T * __restrict__ output,
        const T * __restrict__ left,
        const T * __restrict__ right,
        const std::size_t count
    ) {
        const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
        const std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;

        for (std::size_t i = index; i < count; i += stride) {
            output[i] = left[i] + right[i];
        }
    }


    /**
     * @brief F16 specialization of the elementwise addition kernel.
     *
     * Uses native half-precision addition for each scalar element.
     */
    template<>
    __global__ void add_kernel<half>(
        half * __restrict__ output,
        const half * __restrict__ left,
        const half * __restrict__ right,
        const std::size_t count
    ) {
        const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
        const std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;

        for (std::size_t i = index; i < count; i += stride) {
            output[i] = __hadd(left[i], right[i]);
        }
    }


    /**
     * @brief BF16 specialization of the elementwise addition kernel.
     *
     * Uses native bfloat16 addition for each scalar element.
     */
    template<>
    __global__ void add_kernel<__nv_bfloat16>(
        __nv_bfloat16 * __restrict__ output,
        const __nv_bfloat16 * __restrict__ left,
        const __nv_bfloat16 * __restrict__ right,
        const std::size_t count
    ) {
        const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
        const std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
        for (std::size_t i = index; i < count; i += stride) {
            output[i] = __hadd(left[i], right[i]);
        }
    }


    /**
     * @brief Adds two packed pairs of F16 values.
     *
     * @param left First packed pair.
     * @param right Second packed pair.
     * @return Pairwise sum.
     */
    __device__ __forceinline__ half2 packed_add(const half2 left, const half2 right) {
        return __hadd2(left, right);
    }


    /**
     * @brief Adds two packed pairs of BF16 values.
     *
     * @param left First packed pair.
     * @param right Second packed pair.
     * @return Pairwise sum.
     */
    __device__ __forceinline__ __nv_bfloat162 packed_add(
        const __nv_bfloat162 left, const __nv_bfloat162 right) {
        return __hadd2(left, right);
    }


    /**
     * @brief Adds two tensors using packed two-element CUDA vector types.
     *
     * @tparam PackedT Either half2 or __nv_bfloat162.
     * @param output Destination packed buffer.
     * @param left First packed source buffer.
     * @param right Second packed source buffer.
     * @param count Number of packed elements.
     */
    template<typename PackedT>
    __global__ void add_packed_kernel(
        PackedT * __restrict__ output,
        const PackedT * __restrict__ left,
        const PackedT * __restrict__ right,
        const std::size_t count
    ) {
        const std::size_t index =
                static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
        if (index < count) output[index] = packed_add(left[index], right[index]);
    }


    /**
     * @brief Computes an out-of-place elementwise tensor sum.
     *
     * Even-sized F16 and BF16 tensors use packed two-element operations.
     * Scalar kernels handle F32 and odd-sized low-precision tensors.
     *
     * @param output Destination CUDA tensor.
     * @param left First source CUDA tensor.
     * @param right Second source CUDA tensor.
     * @param stream CUDA stream used for the kernel launch.
     *
     * @throws std::runtime_error If tensor element counts differ, the dtype is
     * unsupported, or CUDA reports a launch error.
     */
    void add_tensors(
        Tensor &output,
        const Tensor &left,
        const Tensor &right,
        cudaStream_t stream
    ) {
        if (output.numel() != left.numel() || output.numel() != right.numel()) {
            throw std::runtime_error("add_tensors: tensor size mismatch");
        }

        const std::size_t count = output.numel();
        const int blocks = static_cast<int>((count + kElementwiseThreads - 1) / kElementwiseThreads);

        if (output.dtype() == Dtype::F32) {
            add_kernel<float><<<blocks, kElementwiseThreads, 0, stream>>>(
                static_cast<float *>(output.raw_data()),
                static_cast<const float *>(left.raw_data()),
                static_cast<const float *>(right.raw_data()),
                count
            );
        } else if (output.dtype() == Dtype::F16 && count % 2 == 0) {
            const std::size_t packed_count = count / 2;
            const int packed_blocks = static_cast<int>(
                (packed_count + kElementwiseThreads - 1) / kElementwiseThreads);
            add_packed_kernel<half2><<<packed_blocks, kElementwiseThreads, 0, stream>>>(
                static_cast<half2 *>(output.raw_data()),
                static_cast<const half2 *>(left.raw_data()),
                static_cast<const half2 *>(right.raw_data()), packed_count);
        } else if (output.dtype() == Dtype::BF16 && count % 2 == 0) {
            const std::size_t packed_count = count / 2;
            const int packed_blocks = static_cast<int>(
                (packed_count + kElementwiseThreads - 1) / kElementwiseThreads);
            add_packed_kernel<__nv_bfloat162><<<packed_blocks, kElementwiseThreads, 0, stream>>>(
                static_cast<__nv_bfloat162 *>(output.raw_data()),
                static_cast<const __nv_bfloat162 *>(left.raw_data()),
                static_cast<const __nv_bfloat162 *>(right.raw_data()), packed_count);
        } else if (output.dtype() == Dtype::F16) {
            add_kernel<half><<<blocks, kElementwiseThreads, 0, stream>>>(
                static_cast<half *>(output.raw_data()),
                static_cast<const half *>(left.raw_data()),
                static_cast<const half *>(right.raw_data()),
                count
            );
        } else if (output.dtype() == Dtype::BF16) {
            add_kernel<__nv_bfloat16><<<blocks, kElementwiseThreads, 0, stream>>>(
                static_cast<__nv_bfloat16 *>(output.raw_data()),
                static_cast<const __nv_bfloat16 *>(left.raw_data()),
                static_cast<const __nv_bfloat16 *>(right.raw_data()),
                count
            );
        } else {
            throw std::runtime_error("add_tensors: unsupported data type");
        }
        CUDA_CHECK(cudaGetLastError());
    }


    /**
     * @brief Adds a source buffer into a destination buffer in place.
     *
     * @tparam T Scalar storage type.
     * @param output Destination buffer updated as `output += source`.
     * @param source Source buffer.
     * @param count Number of scalar elements.
     */
    template<typename T>
    __global__ void add_inplace_kernel(T *output, const T *source, const std::size_t count) {
        const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
        if (index < count) output[index] = static_cast<T>(
                               static_cast<float>(output[index]) + static_cast<float>(source[index]));
    }


    /**
     * @brief Performs packed in-place addition for F16 or BF16 pairs.
     *
     * @tparam PackedT Either half2 or __nv_bfloat162.
     * @param output Destination packed buffer.
     * @param source Source packed buffer.
     * @param count Number of packed elements.
     */
    template<typename PackedT>
    __global__ void add_inplace_packed_kernel(
        PackedT *output, const PackedT *source, const std::size_t count) {
        const std::size_t index =
                static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
        if (index < count) output[index] = packed_add(output[index], source[index]);
    }


    /**
     * @brief Accumulates one CUDA tensor into another.
     *
     * @param destination Tensor updated in place.
     * @param source Tensor added to @p destination.
     * @param stream CUDA stream used for execution.
     *
     * @throws std::invalid_argument If shape, dtype, or device placement differs.
     * @throws std::runtime_error If CUDA reports a kernel launch failure.
     */
    void add_inplace(Tensor &destination, const Tensor &source, cudaStream_t stream) {
        if (destination.shape() != source.shape() || destination.dtype() != source.dtype() ||
            destination.device_type() != DeviceType::CUDA || source.device_type() != DeviceType::CUDA) {
            throw std::invalid_argument("transformer block backward: incompatible tensor addition");
        }
        const int blocks = static_cast<int>((destination.numel() + kElementwiseThreads - 1) / kElementwiseThreads);
        if (destination.dtype() == Dtype::F32) add_inplace_kernel<float><<<blocks, kElementwiseThreads, 0, stream>>>(
            static_cast<float *>(destination.raw_data()), static_cast<const float *>(source.raw_data()),
            destination.numel());
        else if (destination.dtype() == Dtype::F16 && destination.numel() % 2 == 0) {
            const std::size_t count = destination.numel() / 2;
            const int packed_blocks = static_cast<int>((count + kElementwiseThreads - 1) / kElementwiseThreads);
            add_inplace_packed_kernel<half2><<<packed_blocks, kElementwiseThreads, 0, stream>>>(
                static_cast<half2 *>(destination.raw_data()), static_cast<const half2 *>(source.raw_data()), count);
        } else if (destination.dtype() == Dtype::BF16 && destination.numel() % 2 == 0) {
            const std::size_t count = destination.numel() / 2;
            const int packed_blocks = static_cast<int>((count + kElementwiseThreads - 1) / kElementwiseThreads);
            add_inplace_packed_kernel<__nv_bfloat162><<<packed_blocks, kElementwiseThreads, 0, stream>>>(
                static_cast<__nv_bfloat162 *>(destination.raw_data()),
                static_cast<const __nv_bfloat162 *>(source.raw_data()), count);
        } else if (destination.dtype() == Dtype::F16) add_inplace_kernel<half><<<blocks, kElementwiseThreads, 0, stream
                >>>(static_cast<half *>(destination.raw_data()), static_cast<const half *>(source.raw_data()),
                    destination.numel());
        else add_inplace_kernel<__nv_bfloat16><<<blocks, kElementwiseThreads, 0, stream>>>(
            static_cast<__nv_bfloat16 *>(destination.raw_data()), static_cast<const __nv_bfloat16 *>(source.raw_data()),
            destination.numel());
        CUDA_CHECK(cudaGetLastError());
    }


    /**
     * @brief Requires a tensor to have an exact shape.
     *
     * @param tensor Tensor to inspect.
     * @param shape Expected dimensions.
     * @param name Human-readable tensor name used in diagnostics.
     *
     * @throws std::runtime_error If the shape differs.
     */
    void require_shape(const Tensor &tensor, const std::vector<std::size_t> &shape, const char *name) {
        if (tensor.shape() != shape) {
            throw std::runtime_error(std::string(name) + ": tensor shape mismatch");
        }
    }


    /**
     * @brief Validates non-zero Transformer block dimensions.
     *
     * @param options Block configuration.
     *
     * @throws std::invalid_argument If any required model dimension is zero.
     */
    void validate_options(const TransformerBlockOptions &options) {
        if (options.hidden_size == 0 || options.intermediate_size == 0 || options.num_query_heads == 0 ||
            options.num_kv_heads == 0 || options.head_dim == 0) {
            throw std::invalid_argument("transformer block dimensions must be non-zero");
        }
    }
}


/**
 * @brief Allocates temporary CUDA tensors used by Transformer block backward.
 *
 * Workspace tensors cover gradients for the attention branch, per-head QK
 * normalization, RoPE, the SwiGLU feed-forward branch, residual paths, and
 * temporary bias reductions.
 *
 * @param options Transformer block dimensions and operator configuration.
 * @param batch_size Number of sequences.
 * @param sequence_length Number of tokens per sequence.
 * @param dtype Floating-point storage type used by workspace tensors.
 *
 * @throws std::invalid_argument If dimensions are zero or @p dtype is not
 * floating point.
 */
TransformerBlockBackwardWorkspace::TransformerBlockBackwardWorkspace(
    const TransformerBlockOptions &options, const std::size_t batch_size,
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
        throw std::invalid_argument(
            "transformer block backward workspace requires positive dimensions and a floating dtype");
}


/**
 * @brief Executes the CUDA forward pass of one Transformer block.
 *
 * The block applies attention RMSNorm, Q/K/V projections, per-head QK-Norm,
 * RoPE, grouped-query attention, output projection, and a SwiGLU feed-forward
 * branch. Attention and feed-forward branches use a parallel-residual layout:
 * @f[
 *     output = input + attention(input) + ffn(input).
 * @f]
 *
 * Query tensors use shape `[B, S, Hq, D]`; key and value tensors use
 * `[B, S, Hkv, D]`. The externally visible input and output are flattened to
 * `[B*S, hidden_size]`.
 *
 * Optional key/value cache tensors may be supplied together. Newly generated
 * K/V entries are appended at @p cache_length, and attention then observes the
 * active prefix of length `cache_length + S`.
 *
 * @param output Destination tensor with shape `[B*S, hidden_size]`.
 * @param input Input tensor with shape `[B*S, hidden_size]`.
 * @param weights Block parameters.
 * @param workspace Preallocated forward workspace whose tensor shapes encode
 * batch size and sequence length.
 * @param cos_cache RoPE cosine cache.
 * @param sin_cache RoPE sine cache.
 * @param cublas_context cuBLASLt execution context for linear projections.
 * @param stream CUDA stream used by all operations.
 * @param options Block dimensions and operator settings.
 * @param position_offset Absolute position of the first query token.
 * @param key_cache Optional key cache with shape `[B, capacity, Hkv, D]`.
 * @param value_cache Optional value cache matching @p key_cache.
 * @param cache_length Number of valid tokens already stored in the cache.
 *
 * @throws std::invalid_argument If tensor shapes, dtypes, options, workspace
 * state, or cache configuration are incompatible.
 * @throws std::runtime_error If a workspace shape check or CUDA operation fails.
 *
 * @note Both cache pointers must be null or both must be non-null.
 * @note Operations are enqueued asynchronously on @p stream.
 */
void transformer_block_forward(
    Tensor &output,
    const Tensor &input,
    const TransformerBlockWeights &weights,
    TransformerBlockWorkspace &workspace,
    const Tensor &cos_cache,
    const Tensor &sin_cache,
    const CublasLtContext &cublas_context,
    cudaStream_t stream,
    const TransformerBlockOptions &options,
    std::size_t position_offset,
    Tensor *key_cache,
    Tensor *value_cache,
    const std::size_t cache_length
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

    workspace.query.reshape({
        batch, sequence, options.num_query_heads,
        options.head_dim
    });
    workspace.key.reshape({
        batch, sequence, options.num_kv_heads,
        options.head_dim
    });
    workspace.value.reshape({
        batch, sequence, options.num_kv_heads,
        options.head_dim
    });

    // Normalize each attention head independently before positional rotation.
    // Keep the unnormalized projections because QK-Norm needs them in backward.
    CUDA_CHECK(cudaMemcpyAsync(workspace.query_pre_norm.raw_data(), workspace.query.raw_data(),
        workspace.query.nbytes(), cudaMemcpyDeviceToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(workspace.key_pre_norm.raw_data(), workspace.key.raw_data(),
        workspace.key.nbytes(), cudaMemcpyDeviceToDevice, stream));
    workspace.query.reshape({batch * sequence * options.num_query_heads, options.head_dim});
    workspace.query_pre_norm.reshape({batch * sequence * options.num_query_heads, options.head_dim});
    rmsnorm_forward(workspace.query, workspace.query_pre_norm, weights.q_norm,
                    options.rms_epsilon, stream);
    workspace.key.reshape({batch * sequence * options.num_kv_heads, options.head_dim});
    workspace.key_pre_norm.reshape({batch * sequence * options.num_kv_heads, options.head_dim});
    rmsnorm_forward(workspace.key, workspace.key_pre_norm, weights.k_norm,
                    options.rms_epsilon, stream);
    workspace.query_pre_norm.reshape({batch, sequence, options.num_query_heads, options.head_dim});
    workspace.key_pre_norm.reshape({batch, sequence, options.num_kv_heads, options.head_dim});
    workspace.query.reshape({batch, sequence, options.num_query_heads, options.head_dim});
    workspace.key.reshape({batch, sequence, options.num_kv_heads, options.head_dim});

    rope_forward(
        workspace.query,
        workspace.key,
        cos_cache,
        sin_cache,
        stream,
        RopeOptions{options.rotary_dim, position_offset}
    );

    const Tensor *attention_key = &workspace.key;
    const Tensor *attention_value = &workspace.value;
    std::optional<Tensor> active_key;
    std::optional<Tensor> active_value;
    if (key_cache != nullptr || value_cache != nullptr) {
        if (key_cache == nullptr || value_cache == nullptr ||
            key_cache->shape().size() != 4 || value_cache->shape() != key_cache->shape() ||
            key_cache->size(0) != batch || key_cache->size(1) < cache_length + sequence ||
            key_cache->size(2) != options.num_kv_heads || key_cache->size(3) != options.head_dim ||
            key_cache->dtype() != workspace.key.dtype() || value_cache->dtype() != workspace.value.dtype()) {
            throw std::invalid_argument("invalid transformer KV cache shape");
        }
        const std::size_t token_bytes = options.num_kv_heads * options.head_dim * sizeof(__half);
        for (std::size_t batch_index = 0; batch_index < batch; ++batch_index) {
            auto *key_destination = static_cast<std::byte *>(key_cache->raw_data()) +
                                    (batch_index * key_cache->size(1) + cache_length) * token_bytes;
            auto *value_destination = static_cast<std::byte *>(value_cache->raw_data()) +
                                      (batch_index * value_cache->size(1) + cache_length) * token_bytes;
            const auto *key_source = static_cast<const std::byte *>(workspace.key.raw_data()) +
                                     batch_index * sequence * token_bytes;
            const auto *value_source = static_cast<const std::byte *>(workspace.value.raw_data()) +
                                       batch_index * sequence * token_bytes;
            CUDA_CHECK(cudaMemcpyAsync(key_destination, key_source, sequence * token_bytes,
                cudaMemcpyDeviceToDevice, stream));
            CUDA_CHECK(cudaMemcpyAsync(value_destination, value_source, sequence * token_bytes,
                cudaMemcpyDeviceToDevice, stream));
        }
        active_key.emplace(Tensor::device_view(
            {batch, cache_length + sequence, options.num_kv_heads, options.head_dim},
            key_cache->raw_data(), key_cache->dtype()));
        active_value.emplace(Tensor::device_view(
            {batch, cache_length + sequence, options.num_kv_heads, options.head_dim},
            value_cache->raw_data(), value_cache->dtype()));
        attention_key = &*active_key;
        attention_value = &*active_value;
    }

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
        *attention_key,
        *attention_value,
        stream,
        flash_attention_options
    );

    workspace.attention_output.reshape({rows, hidden_size});
    linear_forward(workspace.attention_projection, workspace.attention_output,
                   weights.o_projection, cublas_context, stream,
                   options.linear_options);

    // Parallel residuals: the MLP sees the original block input, rather than
    // the attention-updated residual stream.  This shortens both branch
    // gradient paths and keeps their activations independently normalized.
    rmsnorm_forward(workspace.norm, input, weights.ffn_norm,
                    options.rms_epsilon, stream);

    linear_forward(workspace.gate, workspace.norm, weights.gate_proj,
                   cublas_context, stream, options.linear_options);
    linear_forward(workspace.up, workspace.norm, weights.up_proj,
                   cublas_context, stream, options.linear_options);
    swiglu_forward(workspace.activated, workspace.gate, workspace.up, stream);
    linear_forward(workspace.ffn_output, workspace.activated, weights.down_proj,
                   cublas_context, stream, options.linear_options);

    add_tensors(output, input, workspace.attention_projection, stream);
    add_tensors(output, output, workspace.ffn_output, stream);

    workspace.query.reshape({
        batch, sequence, options.num_query_heads,
        options.head_dim
    });
    workspace.key.reshape({
        batch, sequence, options.num_kv_heads,
        options.head_dim
    });
    workspace.value.reshape({
        batch, sequence, options.num_kv_heads,
        options.head_dim
    });
    workspace.attention_output.reshape(
        {batch, sequence, options.num_query_heads, options.head_dim});
}


/**
 * @brief Backpropagates through one non-cached Transformer block.
 *
 * Gradients are propagated through the parallel SwiGLU and attention branches,
 * the output projection, grouped-query attention, inverse RoPE, per-head
 * QK-Norm, Q/K/V projections, and both RMSNorm operations. Residual-path
 * gradients are accumulated into @p grad_input.
 *
 * @param grad_input Destination gradient with respect to @p input.
 * @param grad_output Upstream gradient with respect to the block output.
 * @param input Forward block input with shape `[B*S, hidden_size]`.
 * @param weights Forward block parameters.
 * @param gradients Destination parameter-gradient tensors.
 * @param forward_workspace Saved forward activations and attention metadata.
 * @param backward Preallocated backward workspace.
 * @param cos_cache RoPE cosine cache used during the forward pass.
 * @param sin_cache RoPE sine cache used during the forward pass.
 * @param cublas_context cuBLASLt execution context.
 * @param stream CUDA stream used by all backward operations.
 * @param options Block configuration.
 * @param position_offset Position offset used by forward RoPE and causal
 * attention.
 *
 * @throws std::invalid_argument If input gradients, workspace state, or options
 * are incompatible.
 *
 * @note This routine expects forward activations from transformer_block_forward()
 * without an external KV cache.
 * @note Operations are asynchronous with respect to the host.
 */
void transformer_block_backward(
    Tensor &grad_input, const Tensor &grad_output, const Tensor &input,
    const TransformerBlockWeights &weights, const TransformerBlockGradients gradients,
    const TransformerBlockWorkspace &forward_workspace, TransformerBlockBackwardWorkspace &backward,
    const Tensor &cos_cache, const Tensor &sin_cache, const CublasLtContext &cublas_context,
    cudaStream_t stream, const TransformerBlockOptions &options, const std::size_t position_offset) {
    validate_options(options);
    if (input.shape() != grad_output.shape() || grad_input.shape() != input.shape() || input.dim() != 2 ||
        input.device_type() != DeviceType::CUDA || grad_output.device_type() != DeviceType::CUDA ||
        grad_input.device_type() != DeviceType::CUDA || input.dtype() != grad_output.dtype() || input.dtype() !=
        grad_input.dtype())
        throw std::invalid_argument("transformer block backward: invalid input or output gradient");
    const std::size_t rows = input.size(0), hidden = options.hidden_size;
    const std::size_t batch = forward_workspace.query.size(0), sequence = forward_workspace.query.size(1);
    if (input.size(1) != hidden || batch * sequence != rows || forward_workspace.norm.shape() != std::vector<
            std::size_t>{rows, hidden})
        throw std::invalid_argument("transformer block backward: forward workspace does not match input");

    LinearBackwardOptions linear_options{options.linear_options.compute_type, options.linear_options.workspace_bytes};
    linear_backward(backward.grad_activated, gradients.down_proj, backward.hidden_bias, grad_output,
                    forward_workspace.activated, weights.down_proj, cublas_context, stream, linear_options);
    swiglu_backward(backward.grad_gate, backward.grad_up, backward.grad_activated,
                    forward_workspace.gate, forward_workspace.up, stream);
    linear_backward(backward.grad_ffn_norm_input, gradients.gate_proj, backward.intermediate_bias,
                    backward.grad_gate, forward_workspace.norm, weights.gate_proj, cublas_context, stream,
                    linear_options);
    linear_options.accumulate_input = true;
    linear_backward(backward.grad_ffn_norm_input, gradients.up_proj, backward.intermediate_bias,
                    backward.grad_up, forward_workspace.norm, weights.up_proj, cublas_context, stream, linear_options);
    linear_options.accumulate_input = false;

    rmsnorm_backward(backward.grad_residual, gradients.ffn_norm, backward.grad_ffn_norm_input,
                     input, weights.ffn_norm, options.rms_epsilon, stream);

    CUDA_CHECK(
        cudaMemcpyAsync(grad_input.raw_data(), grad_output.raw_data(), grad_output.nbytes(), cudaMemcpyDeviceToDevice,
            stream));
    add_inplace(grad_input, backward.grad_residual, stream);
    CUDA_CHECK(
        cudaMemcpyAsync(backward.grad_attention_projection.raw_data(), grad_output.raw_data(), grad_output.nbytes(),
            cudaMemcpyDeviceToDevice, stream));

    backward.grad_attention_output.reshape({rows, hidden});
    auto &saved_attention_output = const_cast<Tensor &>(forward_workspace.attention_output);
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

    // Backpropagate through the per-head QK normalizations. Reuse the original
    // gradient buffers after RoPE has consumed them.
    backward.grad_query_pre_rope.reshape({batch * sequence * options.num_query_heads, options.head_dim});
    backward.grad_query.reshape({batch * sequence * options.num_query_heads, options.head_dim});
    auto &saved_query_pre_norm = const_cast<Tensor &>(forward_workspace.query_pre_norm);
    saved_query_pre_norm.reshape({batch * sequence * options.num_query_heads, options.head_dim});
    rmsnorm_backward(backward.grad_query, gradients.q_norm, backward.grad_query_pre_rope,
                     saved_query_pre_norm, weights.q_norm, options.rms_epsilon, stream);
    backward.grad_key_pre_rope.reshape({batch * sequence * options.num_kv_heads, options.head_dim});
    backward.grad_key.reshape({batch * sequence * options.num_kv_heads, options.head_dim});
    auto &saved_key_pre_norm = const_cast<Tensor &>(forward_workspace.key_pre_norm);
    saved_key_pre_norm.reshape({batch * sequence * options.num_kv_heads, options.head_dim});
    rmsnorm_backward(backward.grad_key, gradients.k_norm, backward.grad_key_pre_rope,
                     saved_key_pre_norm, weights.k_norm, options.rms_epsilon, stream);
    saved_query_pre_norm.reshape({batch, sequence, options.num_query_heads, options.head_dim});
    saved_key_pre_norm.reshape({batch, sequence, options.num_kv_heads, options.head_dim});

    rmsnorm_forward(backward.attention_norm_output, input, weights.attention_norm, options.rms_epsilon, stream);
    backward.grad_query.reshape({rows, options.num_query_heads * options.head_dim});
    backward.grad_key.reshape({rows, options.num_kv_heads * options.head_dim});
    backward.grad_value.reshape({rows, options.num_kv_heads * options.head_dim});
    linear_backward(backward.grad_attention_norm_input, gradients.q_projection, backward.hidden_bias,
                    backward.grad_query, backward.attention_norm_output, weights.q_projection, cublas_context, stream,
                    linear_options);
    linear_options.accumulate_input = true;
    linear_backward(backward.grad_attention_norm_input, gradients.k_projection, backward.kv_bias,
                    backward.grad_key, backward.attention_norm_output, weights.k_projection, cublas_context, stream,
                    linear_options);
    linear_backward(backward.grad_attention_norm_input, gradients.v_projection, backward.kv_bias,
                    backward.grad_value, backward.attention_norm_output, weights.v_projection, cublas_context, stream,
                    linear_options);
    linear_options.accumulate_input = false;
    backward.grad_query_pre_rope.reshape({batch, sequence, options.num_query_heads, options.head_dim});
    backward.grad_key_pre_rope.reshape({batch, sequence, options.num_kv_heads, options.head_dim});
    backward.grad_value.reshape({batch, sequence, options.num_kv_heads, options.head_dim});
    backward.grad_query.reshape({batch, sequence, options.num_query_heads, options.head_dim});
    backward.grad_key.reshape({batch, sequence, options.num_kv_heads, options.head_dim});
    rmsnorm_backward(backward.grad_residual, gradients.attention_norm, backward.grad_attention_norm_input,
                     input, weights.attention_norm, options.rms_epsilon, stream);
    add_inplace(grad_input, backward.grad_residual, stream);
}
