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

/**
 * @brief Controls autoregressive text generation and token sampling.
 */
struct GenerationOptions {
    /** @brief Maximum number of tokens to generate. */
    std::size_t max_new_tokens = 32;
    /** @brief Maximum number of prompt and generated tokens evaluated as context. */
    std::size_t max_context_tokens = 2048;
    /**
     * @brief Sampling temperature.
     *
     * Lower values make sampling more deterministic. A value of `1.0` leaves
     * logits unchanged.
     */
    float temperature = 1.0F;
    /** @brief Number of highest-logit tokens retained; zero disables top-k filtering. */
    std::size_t top_k = 0;       // 0 disables top-k filtering.
    /** @brief Cumulative probability threshold; `1.0` disables nucleus filtering. */
    float top_p = 1.0F;          // 1 disables nucleus filtering.
    /** @brief Penalizes tokens already present in the active context. 1 disables it. */
    float repetition_penalty = 1.0F;
    /** @brief Additive penalty applied once to every token present in the context. */
    float presence_penalty = 0.0F;
    /** @brief Additive penalty applied for every occurrence in the context. */
    float frequency_penalty = 0.0F;
    /** @brief Prevents a previously seen n-gram from being completed again. */
    std::size_t no_repeat_ngram_size = 0;
    /** @brief Keeps tokens whose probability is at least min_p * best probability. */
    float min_p = 0.0F;
    /** @brief Random seed; zero requests a non-deterministic seed. */
    std::uint64_t seed = 0;      // 0 uses a non-deterministic seed.
    /** @brief Optional end-of-sequence token that stops generation when sampled. */
    std::optional<bpe::TokenId> eos_token_id;
};

/** @brief Text returned together with the exact number of generated tokens. */
struct TextGenerationResult {
    std::string text;
    std::size_t prompt_tokens = 0;
    std::size_t generated_tokens = 0;
};

/** CPU autoregressive generation. All model and cache tensors must be CPU tensors. */
[[nodiscard]] std::vector<bpe::TokenId> generate_tokens_cpu(
    const std::vector<bpe::TokenId>& prompt_tokens,
    const TransformerModelWeights& weights,
    const Tensor& cos_cache, const Tensor& sin_cache,
    const TransformerModelOptions& model_options,
    const GenerationOptions& generation_options = {});

[[nodiscard]] TextGenerationResult generate_text_with_stats_cpu(
    const bpe::BpeTokenizer& tokenizer, std::string_view prompt,
    const TransformerModelWeights& weights,
    const Tensor& cos_cache, const Tensor& sin_cache,
    const TransformerModelOptions& model_options,
    const GenerationOptions& generation_options = {});

/**
 * @brief Generates token IDs autoregressively from a decoder-only Transformer.
 *
 * The implementation performs a prompt prefill followed by incremental
 * single-token decoding with a per-layer KV cache. If the prompt is longer
 * than `max_context_tokens`, its oldest tokens are truncated for inference
 * while the returned sequence remains complete.
 *
 * @param prompt_tokens Token IDs used as the initial prompt.
 * @param weights Transformer weights used for inference.
 * @param cos_cache Precomputed cosine rotary-embedding cache.
 * @param sin_cache Precomputed sine rotary-embedding cache.
 * @param cublas_context cuBLASLt context used by model operations.
 * @param model_options Transformer architecture and execution options.
 * @param generation_options Sampling and generation limits.
 * @param stream CUDA stream used for model execution; `nullptr` uses the
 *        default stream.
 * @return Prompt tokens followed by the generated token IDs.
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

/**
 * @brief Generates text autoregressively from a decoder-only Transformer.
 *
 * The prompt is tokenized before inference and the generated token IDs are
 * decoded back into a string using @p tokenizer.
 *
 * @param tokenizer Tokenizer used to encode the prompt and decode the result.
 * @param prompt Input text used as the initial prompt.
 * @param weights Transformer weights used for inference.
 * @param cos_cache Precomputed cosine rotary-embedding cache.
 * @param sin_cache Precomputed sine rotary-embedding cache.
 * @param cublas_context cuBLASLt context used by model operations.
 * @param model_options Transformer architecture and execution options.
 * @param generation_options Sampling and generation limits.
 * @param stream CUDA stream used for model execution; `nullptr` uses the
 *        default stream.
 * @return Generated text decoded from the generated token IDs.
 */
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

/** @brief Generates text and reports exact prompt/output token counts. */
[[nodiscard]] TextGenerationResult generate_text_with_stats(
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
