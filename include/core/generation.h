#pragma once

#include "core/cublas_context.h"
#include "core/transformer_model.h"
#include "tokenizer/bpe_tokenizer.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

struct GenerationOptions {
    std::size_t max_new_tokens = 32;
    std::size_t max_context_tokens = 2048;
    float temperature = 1.0F;
    std::size_t top_k = 0;       // 0 disables top-k filtering.
    float top_p = 1.0F;          // 1 disables nucleus filtering.
    std::uint64_t seed = 0;      // 0 uses a non-deterministic seed.
    std::optional<bpe::TokenId> eos_token_id;
};

/**
 * Generates token IDs autoregressively from a decoder-only Transformer.
 *
 * This reference implementation recomputes the complete context on every
 * step. It is intentionally correct and simple; KV-cache acceleration can be
 * added without changing the public sampling options.
 */
[[nodiscard]] std::vector<bpe::TokenId> generate_tokens(
    const std::vector<bpe::TokenId>& prompt_tokens,
    const TransformerModelWeights& weights,
    const Tensor& cos_cache,
    const Tensor& sin_cache,
    const CublasLtContext& cublas_context,
    const TransformerModelOptions& model_options,
    const GenerationOptions& generation_options = {},
    cudaStream_t stream = nullptr
);

[[nodiscard]] std::string generate_text(
    const bpe::BpeTokenizer& tokenizer,
    std::string_view prompt,
    const TransformerModelWeights& weights,
    const Tensor& cos_cache,
    const Tensor& sin_cache,
    const CublasLtContext& cublas_context,
    const TransformerModelOptions& model_options,
    const GenerationOptions& generation_options = {},
    cudaStream_t stream = nullptr
);
