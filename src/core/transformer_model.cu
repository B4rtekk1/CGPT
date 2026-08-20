#include "core/transformer_model.h"

#include "core/cuda_check.h"

#include "ops/linear.h"
#include "ops/rmsnorm.h"
#include "ops/backward/embedding_backward.h"
#include "ops/backward/linear_backward.h"
#include "ops/backward/rmsnorm_backward.h"

#include <limits>
#include <stdexcept>
#include <string>

namespace {

void validate_options(const TransformerModelOptions& options) {
    const auto& block = options.block_options;
    if (options.vocabulary_size == 0 || options.num_layers == 0 ||
        block.hidden_size == 0 || block.intermediate_size == 0 ||
        block.num_query_heads == 0 || block.num_kv_heads == 0 || block.head_dim == 0) {
        throw std::invalid_argument("transformer model dimensions must be non-zero");
    }
    if (block.num_query_heads * block.head_dim != block.hidden_size) {
        throw std::invalid_argument("transformer model query heads times head_dim must equal hidden_size");
    }
    if (block.num_kv_heads > block.num_query_heads ||
        block.num_query_heads % block.num_kv_heads != 0) {
        throw std::invalid_argument("transformer model has an invalid GQA head configuration");
    }
}

TransformerBlockWorkspace make_block_workspace(
    const TransformerModelOptions& options,
    const std::size_t batch_size,
    const std::size_t sequence_length,
    const Dtype dtype
) {
    const auto& block = options.block_options;
    const std::size_t rows = batch_size * sequence_length;
    return {
        Tensor({rows, block.hidden_size}, dtype),
        Tensor({batch_size, sequence_length, block.num_query_heads, block.head_dim}, dtype),
        Tensor({batch_size, sequence_length, block.num_kv_heads, block.head_dim}, dtype),
        Tensor({batch_size, sequence_length, block.num_kv_heads, block.head_dim}, dtype),
        Tensor({batch_size, sequence_length, block.num_query_heads, block.head_dim}, dtype),
        Tensor({batch_size, sequence_length, block.num_kv_heads, block.head_dim}, dtype),
        Tensor({batch_size, sequence_length, block.num_query_heads, block.head_dim}, dtype),
        Tensor({batch_size, sequence_length, block.num_query_heads}, Dtype::F32),
        Tensor({rows, block.hidden_size}, dtype),
        Tensor({rows, block.intermediate_size}, dtype),
        Tensor({rows, block.intermediate_size}, dtype),
        Tensor({rows, block.intermediate_size}, dtype),
        Tensor({rows, block.hidden_size}, dtype)
    };
}

void require_cuda_tensor(const Tensor& tensor, const std::vector<std::size_t>& shape,
                         const Dtype dtype, const char* name) {
    if (tensor.device_type() != DeviceType::CUDA || tensor.dtype() != dtype || tensor.shape() != shape) {
        throw std::invalid_argument(std::string("transformer model: invalid ") + name + " tensor");
    }
}

} // namespace

TransformerModelWorkspace::TransformerModelWorkspace(
    const TransformerModelOptions& options,
    const std::size_t batch_size,
    const std::size_t sequence_length,
    const Dtype dtype
) : hidden({batch_size * sequence_length, options.block_options.hidden_size}, dtype),
    layer_output({batch_size * sequence_length, options.block_options.hidden_size}, dtype),
    normalized({batch_size * sequence_length, options.block_options.hidden_size}, dtype) {
    validate_options(options);
    if (batch_size == 0 || sequence_length == 0 || !is_floating_point(dtype)) {
        throw std::invalid_argument("transformer model workspace requires positive dimensions and a floating dtype");
    }
    if (batch_size > std::numeric_limits<std::size_t>::max() / sequence_length) {
        throw std::overflow_error("transformer model workspace row count overflows");
    }
    layers.reserve(options.num_layers);
    for (std::size_t layer = 0; layer < options.num_layers; ++layer) {
        layers.push_back(make_block_workspace(options, batch_size, sequence_length, dtype));
        layer_inputs.emplace_back(std::vector<std::size_t>{batch_size * sequence_length, options.block_options.hidden_size}, dtype);
    }
}

TransformerModelBackwardWorkspace::TransformerModelBackwardWorkspace(
    const TransformerModelOptions& options, const std::size_t batch_size,
    const std::size_t sequence_length, const Dtype dtype)
    : grad_normalized({batch_size * sequence_length, options.block_options.hidden_size}, dtype),
      grad_hidden({batch_size * sequence_length, options.block_options.hidden_size}, dtype),
      grad_previous({batch_size * sequence_length, options.block_options.hidden_size}, dtype),
      lm_head_bias({options.vocabulary_size}, dtype) {
    validate_options(options);
    if (batch_size == 0 || sequence_length == 0 || !is_floating_point(dtype))
        throw std::invalid_argument("transformer model backward workspace requires positive dimensions and a floating dtype");
    layers.reserve(options.num_layers);
    for (std::size_t layer = 0; layer < options.num_layers; ++layer)
        layers.emplace_back(options.block_options, batch_size, sequence_length, dtype);
}

void transformer_model_forward(
    Tensor& logits, const bpe::TokenId* const device_token_ids,
    const std::size_t batch_size, const std::size_t sequence_length,
    const TransformerModelWeights& weights, TransformerModelWorkspace& workspace,
    const Tensor& cos_cache, const Tensor& sin_cache,
    const CublasLtContext& cublas_context, cudaStream_t stream,
    const TransformerModelOptions& options, const std::size_t position_offset
) {
    validate_options(options);
    if (device_token_ids == nullptr || batch_size == 0 || sequence_length == 0) {
        throw std::invalid_argument("transformer model requires token IDs and positive batch/sequence dimensions");
    }
    if (batch_size > std::numeric_limits<std::size_t>::max() / sequence_length) {
        throw std::overflow_error("transformer model row count overflows");
    }
    const std::size_t rows = batch_size * sequence_length;
    const auto& block = options.block_options;
    const Dtype dtype = weights.token_embedding.dtype();
    if (!is_floating_point(dtype) || weights.layers.size() != options.num_layers) {
        throw std::invalid_argument("transformer model weights have an invalid dtype or layer count");
    }
    require_cuda_tensor(weights.token_embedding, {options.vocabulary_size, block.hidden_size}, dtype, "token_embedding");
    require_cuda_tensor(weights.final_norm, {block.hidden_size}, dtype, "final_norm");
    require_cuda_tensor(weights.lm_head, {options.vocabulary_size, block.hidden_size}, dtype, "lm_head");
    require_cuda_tensor(workspace.hidden, {rows, block.hidden_size}, dtype, "workspace.hidden");
    require_cuda_tensor(workspace.layer_output, {rows, block.hidden_size}, dtype, "workspace.layer_output");
    require_cuda_tensor(workspace.normalized, {rows, block.hidden_size}, dtype, "workspace.normalized");
    require_cuda_tensor(logits, {rows, options.vocabulary_size}, dtype, "logits");
    if (workspace.layers.size() != options.num_layers) {
        throw std::invalid_argument("transformer model workspace layer count mismatch");
    }

    embedding_forward(workspace.hidden, device_token_ids, rows, weights.token_embedding, stream);
    for (std::size_t layer = 0; layer < options.num_layers; ++layer) {
        CUDA_CHECK(cudaMemcpyAsync(workspace.layer_inputs[layer].raw_data(), workspace.hidden.raw_data(),
            workspace.hidden.nbytes(), cudaMemcpyDeviceToDevice, stream));
        transformer_block_forward(workspace.layer_output, workspace.hidden, weights.layers[layer],
            workspace.layers[layer], cos_cache, sin_cache, cublas_context, stream, block, position_offset);
        std::swap(workspace.hidden, workspace.layer_output);
    }
    rmsnorm_forward(workspace.normalized, workspace.hidden, weights.final_norm,
                    block.rms_epsilon, stream);
    linear_forward(logits, workspace.normalized, weights.lm_head, cublas_context,
                   stream, block.linear_options);
}

void transformer_model_backward(
    const Tensor& grad_logits, const bpe::TokenId* const device_token_ids,
    const std::size_t batch_size, const std::size_t sequence_length,
    const TransformerModelWeights& weights, const TransformerModelGradients gradients,
    const TransformerModelWorkspace& forward_workspace, TransformerModelBackwardWorkspace& backward_workspace,
    const Tensor& cos_cache, const Tensor& sin_cache, const CublasLtContext& cublas_context,
    cudaStream_t stream, const TransformerModelOptions& options, const std::size_t position_offset) {
    validate_options(options);
    if (device_token_ids == nullptr || batch_size == 0 || sequence_length == 0 ||
        gradients.layers.size() != options.num_layers || forward_workspace.layers.size() != options.num_layers ||
        forward_workspace.layer_inputs.size() != options.num_layers || backward_workspace.layers.size() != options.num_layers)
        throw std::invalid_argument("transformer model backward: invalid model state or token IDs");
    const std::size_t rows = batch_size * sequence_length;
    const auto& block = options.block_options;
    const Dtype dtype = weights.token_embedding.dtype();
    if (grad_logits.device_type() != DeviceType::CUDA || grad_logits.dtype() != dtype ||
        grad_logits.shape() != std::vector<std::size_t>{rows, options.vocabulary_size} ||
        backward_workspace.grad_normalized.shape() != std::vector<std::size_t>{rows, block.hidden_size} ||
        forward_workspace.normalized.shape() != std::vector<std::size_t>{rows, block.hidden_size} ||
        forward_workspace.hidden.shape() != std::vector<std::size_t>{rows, block.hidden_size})
        throw std::invalid_argument("transformer model backward: invalid logits gradient or workspace shape");

    LinearBackwardOptions linear_options{block.linear_options.compute_type, block.linear_options.workspace_bytes};
    linear_backward(backward_workspace.grad_normalized, gradients.lm_head, backward_workspace.lm_head_bias,
        grad_logits, forward_workspace.normalized, weights.lm_head, cublas_context, stream, linear_options);
    rmsnorm_backward(backward_workspace.grad_hidden, gradients.final_norm, backward_workspace.grad_normalized,
        forward_workspace.hidden, weights.final_norm, block.rms_epsilon, stream);

    Tensor* current = &backward_workspace.grad_hidden;
    Tensor* previous = &backward_workspace.grad_previous;
    for (std::size_t reverse = options.num_layers; reverse-- > 0;) {
        transformer_block_backward(*previous, *current, forward_workspace.layer_inputs[reverse], weights.layers[reverse],
            gradients.layers[reverse], forward_workspace.layers[reverse], backward_workspace.layers[reverse],
            cos_cache, sin_cache, cublas_context, stream, block, position_offset);
        std::swap(current, previous);
    }
    // With tied embeddings, linear_backward has already written the output
    // classifier gradient to this tensor.  Preserve it while adding the
    // gradient from the input lookup; untied models retain the old overwrite
    // behaviour.
    embedding_backward(gradients.token_embedding, *current, device_token_ids, rows, stream,
        EmbeddingBackwardOptions{.accumulate_weight = &gradients.token_embedding == &gradients.lm_head});
}
