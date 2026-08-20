/** @file generation.cu CUDA autoregressive text generation and sampling implementation. */

#include "core/generation.h"
#include "core/kv_cache.h"

#include "core/cuda_check.h"
#include "ops/embedding.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <random>
#include <stdexcept>

namespace {
    /**
     * @brief Sampling candidate containing a token identifier and adjusted logit.
     *
     * Candidates are sorted in descending logit order before temperature, min-p,
     * top-p, and random sampling are applied.
     */
    struct Candidate {
        bpe::TokenId id;
        float logit;
    };

    /**
     * @brief Selects the next token from a vector of vocabulary logits.
     *
     * The function applies repetition, presence, and frequency penalties using the
     * complete generated context, optionally prevents repeated n-grams, removes
     * non-finite candidates, and then applies top-k, temperature, min-p, and
     * nucleus (top-p) filtering. A temperature of zero selects the highest-logit
     * candidate deterministically.
     *
     * @param logits Logits for one decoding position, indexed by token identifier.
     * @param context Tokens generated or supplied before the current position.
     * @param options Sampling and logits-processing configuration.
     * @param rng Random-number generator used by categorical sampling.
     * @return Identifier of the selected token.
     *
     * @throws std::invalid_argument If @p logits is empty or a sampling parameter
     * is outside its supported range.
     * @throws std::runtime_error If logits processing removes every candidate.
     *
     * @note Penalties are applied before top-k, min-p, and top-p filtering.
     * @note Token identifiers outside the logits range are ignored while building
     * frequency counts.
     */
    bpe::TokenId sample(const std::vector<float> &logits, const std::vector<bpe::TokenId> &context,
                        const GenerationOptions &options, std::mt19937_64 &rng) {
        if (logits.empty()) throw std::invalid_argument("cannot sample from empty logits");
        if (!std::isfinite(options.temperature) || options.temperature < 0.0F) {
            throw std::invalid_argument("temperature must be finite and non-negative");
        }
        if (!std::isfinite(options.top_p) || options.top_p <= 0.0F || options.top_p > 1.0F) {
            throw std::invalid_argument("top_p must be in (0, 1]");
        }
        if (!std::isfinite(options.repetition_penalty) || options.repetition_penalty <= 0.0F) {
            throw std::invalid_argument("repetition_penalty must be finite and positive");
        }
        if (!std::isfinite(options.presence_penalty) || !std::isfinite(options.frequency_penalty) ||
            options.presence_penalty < 0.0F || options.frequency_penalty < 0.0F) {
            throw std::invalid_argument("presence and frequency penalties must be finite and non-negative");
        }
        if (!std::isfinite(options.min_p) || options.min_p < 0.0F || options.min_p > 1.0F) {
            throw std::invalid_argument("min_p must be in [0, 1]");
        }

        std::vector<float> adjusted = logits;
        std::vector<std::size_t> counts(logits.size(), 0);
        for (const bpe::TokenId token: context) {
            const auto index = static_cast<std::size_t>(token);
            if (index < counts.size()) ++counts[index];
        }
        for (std::size_t index = 0; index < adjusted.size(); ++index) {
            if (counts[index] != 0) {
                if (adjusted[index] >= 0.0F) adjusted[index] /= options.repetition_penalty;
                else adjusted[index] *= options.repetition_penalty;
                adjusted[index] -= options.presence_penalty +
                        options.frequency_penalty * static_cast<float>(counts[index]);
            }
        }

        // This is the same constraint used by common decoder APIs: if the last
        // n-1 tokens match an earlier n-gram prefix, its original continuation is
        // not allowed again.
        if (options.no_repeat_ngram_size > 1 && context.size() >= options.no_repeat_ngram_size) {
            const std::size_t n = options.no_repeat_ngram_size;
            const std::size_t prefix_begin = context.size() - (n - 1);
            for (std::size_t start = 0; start + n <= context.size() - 1; ++start) {
                bool same_prefix = true;
                for (std::size_t offset = 0; offset + 1 < n; ++offset) {
                    if (context[start + offset] != context[prefix_begin + offset]) {
                        same_prefix = false;
                        break;
                    }
                }
                if (same_prefix) {
                    const auto token = static_cast<std::size_t>(context[start + n - 1]);
                    if (token < adjusted.size()) adjusted[token] = -std::numeric_limits<float>::infinity();
                }
            }
        }

        std::vector<Candidate> candidates;
        candidates.reserve(adjusted.size());
        for (std::size_t i = 0; i < adjusted.size(); ++i) {
            if (std::isfinite(adjusted[i])) candidates.push_back({static_cast<bpe::TokenId>(i), adjusted[i]});
        }
        if (candidates.empty()) throw std::runtime_error("no valid token remains after logits processing");

        std::ranges::sort(candidates.begin(), candidates.end(),
                          [](const Candidate &a, const Candidate &b) { return a.logit > b.logit; });
        if (options.top_k != 0 && options.top_k < candidates.size()) candidates.resize(options.top_k);

        const float temperature = options.temperature == 0.0F ? 1.0F : options.temperature;
        const float max_logit = candidates.front().logit;
        std::vector<float> probabilities(candidates.size());
        float total = 0.0F;
        for (std::size_t i = 0; i < candidates.size(); ++i) {
            probabilities[i] = std::exp((candidates[i].logit - max_logit) / temperature);
            total += probabilities[i];
        }

        if (options.temperature == 0.0F) return candidates.front().id;

        for (float &probability: probabilities) probability /= total;
        if (options.min_p > 0.0F && probabilities.size() > 1) {
            const float cutoff = options.min_p * probabilities.front();
            std::size_t keep = probabilities.size();
            for (std::size_t i = 1; i < probabilities.size(); ++i) {
                if (probabilities[i] < cutoff) {
                    keep = i;
                    break;
                }
            }
            candidates.resize(keep);
            probabilities.resize(keep);
            float kept_total = 0.0F;
            for (const float probability: probabilities) kept_total += probability;
            for (float &probability: probabilities) probability /= kept_total;
        }
        if (options.top_p < 1.0F) {
            float cumulative = 0.0F;
            std::size_t keep = probabilities.size();
            for (std::size_t i = 0; i < probabilities.size(); ++i) {
                cumulative += probabilities[i];
                if (cumulative >= options.top_p) {
                    keep = i + 1;
                    break;
                }
            }
            candidates.resize(keep);
            probabilities.resize(keep);
            float kept_total = 0.0F;
            for (const float probability: probabilities) kept_total += probability;
            for (float &probability: probabilities) probability /= kept_total;
        }

        std::discrete_distribution<std::size_t> distribution(probabilities.begin(), probabilities.end());
        return candidates[distribution(rng)].id;
    }
} // namespace

/**
 * @brief Generates autoregressive tokens with CUDA prefill and KV-cached decode.
 *
 * The prompt is truncated from the left when it exceeds
 * `options.max_context_tokens`. The retained context is processed in one
 * prefill pass, after which every layer's key and value tensors are copied into
 * a single-batch KV cache. Subsequent tokens are decoded one at a time with
 * transformer_model_forward_cached().
 *
 * Generation terminates when the requested token count is reached, an
 * configured end-of-sequence token is sampled, or the KV cache reaches its
 * maximum context length.
 *
 * @param prompt_tokens Tokenized prompt. At least one token is required.
 * @param weights CUDA-resident transformer model weights.
 * @param cos_cache RoPE cosine cache with at least
 * `options.max_context_tokens` positions.
 * @param sin_cache RoPE sine cache with at least
 * `options.max_context_tokens` positions.
 * @param cublas_context cuBLASLt context used by model linear operations.
 * @param model_options Transformer architecture configuration.
 * @param options Generation and sampling configuration.
 * @param stream CUDA stream used for uploads, model execution, and cache
 * transfers.
 * @return The original prompt tokens followed by all generated tokens.
 *
 * @throws std::invalid_argument If the prompt is empty, the context limit is
 * zero, or the RoPE caches are too short.
 * @throws std::runtime_error If sampling cannot select a valid token.
 * @throws CudaError If a CUDA allocation, transfer, synchronization, or model
 * operation reported through CUDA_CHECK fails.
 *
 * @note A seed value of zero initializes the generator from
 * std::random_device; any other value makes sampling reproducible.
 * @note The function synchronizes @p stream before reading logits on the host.
 */
std::vector<bpe::TokenId> generate_tokens(
    const std::vector<bpe::TokenId> &prompt_tokens,
    const TransformerModelWeights &weights,
    const Tensor &cos_cache,
    const Tensor &sin_cache,
    const CublasLtContext &cublas_context,
    const TransformerModelOptions &model_options,
    const GenerationOptions &options,
    cudaStream_t stream
) {
    if (prompt_tokens.empty()) throw std::invalid_argument("prompt must contain at least one token");
    if (options.max_new_tokens == 0) return prompt_tokens;
    if (options.max_context_tokens == 0) throw std::invalid_argument("max_context_tokens must be positive");
    if (cos_cache.size(0) < options.max_context_tokens || sin_cache.size(0) < options.max_context_tokens) {
        throw std::invalid_argument("RoPE caches are shorter than max_context_tokens");
    }

    std::mt19937_64 rng(options.seed == 0 ? std::random_device{}() : options.seed);
    std::vector<bpe::TokenId> result = prompt_tokens;
    const std::size_t begin = prompt_tokens.size() > options.max_context_tokens
                                  ? prompt_tokens.size() - options.max_context_tokens
                                  : 0;
    const std::vector<bpe::TokenId> context(prompt_tokens.begin() + static_cast<long long>(begin), prompt_tokens.end());

    KVCache kv_cache({
        model_options.num_layers, 1, options.max_context_tokens,
        model_options.block_options.num_kv_heads,
        model_options.block_options.head_dim
    });
    TransformerModelWorkspace prefill_workspace(
        model_options, 1, context.size(), weights.token_embedding.dtype());
    Tensor prefill_logits({context.size(), model_options.vocabulary_size}, weights.token_embedding.dtype());
    bpe::TokenId *device_ids = nullptr;
    CUDA_CHECK(cudaMalloc(&device_ids, std::max(context.size(), std::size_t{1}) * sizeof(bpe::TokenId)));
    try {
        embedding_upload_token_ids(device_ids, context, stream);
        transformer_model_forward(prefill_logits, device_ids, 1, context.size(), weights,
                                  prefill_workspace, cos_cache, sin_cache, cublas_context,
                                  stream, model_options);
        for (std::size_t layer = 0; layer < model_options.num_layers; ++layer) {
            kv_cache.write(layer, 1, 0, context.size(),
                           static_cast<const __half *>(prefill_workspace.layers[layer].key.raw_data()),
                           static_cast<const __half *>(prefill_workspace.layers[layer].value.raw_data()),
                           stream);
            kv_cache.set_sequence_length(layer, 0, context.size());
        }
        CUDA_CHECK(cudaStreamSynchronize(stream));
        std::vector<float> host_logits(prefill_logits.numel());
        prefill_logits.copy_to_host(host_logits);
        const std::size_t vocab = model_options.vocabulary_size;
        std::vector<float> last_logits(host_logits.end() - static_cast<long long>(vocab), host_logits.end());
        TransformerModelWorkspace decode_workspace(model_options, 1, 1, weights.token_embedding.dtype());
        Tensor decode_logits({1, model_options.vocabulary_size}, weights.token_embedding.dtype());
        std::vector<float> next_logits(vocab);
        std::size_t cache_length = context.size();
        for (std::size_t step = 0; step < options.max_new_tokens; ++step) {
            const bpe::TokenId next = sample(last_logits, result, options, rng);
            result.push_back(next);
            if (options.eos_token_id && next == *options.eos_token_id) break;
            if (step + 1 == options.max_new_tokens) break;
            if (cache_length >= options.max_context_tokens) break;
            CUDA_CHECK(cudaMemcpyAsync(device_ids, &next, sizeof(next),
                cudaMemcpyHostToDevice, stream));
            transformer_model_forward_cached(decode_logits, device_ids, weights, decode_workspace,
                                             kv_cache, cache_length, cos_cache, sin_cache,
                                             cublas_context, stream, model_options);
            CUDA_CHECK(cudaStreamSynchronize(stream));
            decode_logits.copy_to_host(next_logits);
            last_logits = next_logits;
            ++cache_length;
        }
    } catch (...) {
        cudaFree(device_ids);
        throw;
    }
    CUDA_CHECK(cudaFree(device_ids));
    return result;
}

/**
 * @brief Generates text from a UTF-8 prompt using the CUDA transformer model.
 *
 * This convenience overload delegates to generate_text_with_stats() and
 * returns only the decoded text.
 *
 * @param tokenizer Tokenizer used to encode the prompt and decode all tokens.
 * @param prompt Input text to continue.
 * @param weights CUDA-resident model weights.
 * @param cos_cache RoPE cosine cache.
 * @param sin_cache RoPE sine cache.
 * @param cublas_context cuBLASLt execution context.
 * @param model_options Transformer architecture configuration.
 * @param options Generation and sampling configuration.
 * @param stream CUDA stream used by model execution.
 * @return Decoded prompt and generated continuation.
 *
 * @throws std::invalid_argument If generation inputs or options are invalid.
 * @throws std::runtime_error If token sampling fails.
 * @throws CudaError If a CUDA operation fails.
 */
std::string generate_text(
    const bpe::BpeTokenizer &tokenizer, std::string_view prompt,
    const TransformerModelWeights &weights, const Tensor &cos_cache, const Tensor &sin_cache,
    const CublasLtContext &cublas_context, const TransformerModelOptions &model_options,
    const GenerationOptions &options, cudaStream_t stream
) {
    return generate_text_with_stats(tokenizer, prompt, weights, cos_cache, sin_cache,
                                    cublas_context, model_options, options, stream).text;
}

/**
 * @brief Generates text and reports prompt and completion token counts.
 *
 * If no explicit EOS identifier is supplied, the tokenizer is searched for
 * common special-token names in the following order: `<eos>`, `</s>`,
 * `<|endoftext|>`, and `<|end|>`. The prompt is encoded, passed to
 * generate_tokens(), and the complete token sequence is decoded.
 *
 * @param tokenizer Tokenizer used for encoding, special-token lookup, and
 * decoding.
 * @param prompt Input text to continue.
 * @param weights CUDA-resident model weights.
 * @param cos_cache RoPE cosine cache.
 * @param sin_cache RoPE sine cache.
 * @param cublas_context cuBLASLt execution context.
 * @param model_options Transformer architecture configuration.
 * @param options Generation and sampling configuration.
 * @param stream CUDA stream used by model execution.
 * @return Generated text together with prompt-token and completion-token
 * counts.
 *
 * @throws std::invalid_argument If generation inputs or options are invalid.
 * @throws std::runtime_error If token sampling fails.
 * @throws CudaError If a CUDA operation fails.
 */
TextGenerationResult generate_text_with_stats(
    const bpe::BpeTokenizer &tokenizer, std::string_view prompt,
    const TransformerModelWeights &weights, const Tensor &cos_cache, const Tensor &sin_cache,
    const CublasLtContext &cublas_context, const TransformerModelOptions &model_options,
    const GenerationOptions &options, cudaStream_t stream
) {
    GenerationOptions text_options = options;
    if (!text_options.eos_token_id) {
        // Tokenizers use different conventional names for the same semantic
        // end marker. Prefer the first one present and otherwise allow the
        // caller to generate until max_new_tokens.
        for (const std::string_view name: {"<eos>", "</s>", "<|endoftext|>", "<|end|>"}) {
            if (const auto token = tokenizer.special_token_id(name)) {
                text_options.eos_token_id = token;
                break;
            }
        }
    }
    const auto prompt_tokens = tokenizer.encode(prompt);
    const auto all_tokens = generate_tokens(prompt_tokens, weights, cos_cache, sin_cache,
                                            cublas_context, model_options, text_options, stream);
    return {
        tokenizer.decode(all_tokens), prompt_tokens.size(),
        all_tokens.size() - prompt_tokens.size()
    };
}
