#include "core/transformer_model.h"

#include "ops/linear.h"
#include "ops/rmsnorm.h"

#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

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
    }
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
        transformer_block_forward(workspace.layer_output, workspace.hidden, weights.layers[layer],
            workspace.layers[layer], cos_cache, sin_cache, cublas_context, stream, block, position_offset);
        std::swap(workspace.hidden, workspace.layer_output);
    }
    rmsnorm_forward(workspace.normalized, workspace.hidden, weights.final_norm,
                    block.rms_epsilon, stream);
    linear_forward(logits, workspace.normalized, weights.lm_head, cublas_context,
                   stream, block.linear_options);
}
