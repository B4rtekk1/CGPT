#include "core/generation.h"

#include "core/cuda_check.h"
#include "ops/embedding.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <random>
#include <stdexcept>

namespace {

struct Candidate {
    bpe::TokenId id;
    float logit;
};

bpe::TokenId sample(const std::vector<float>& logits, const std::vector<bpe::TokenId>& context,
                    const GenerationOptions& options, std::mt19937_64& rng) {
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
    for (const bpe::TokenId token : context) {
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
              [](const Candidate& a, const Candidate& b) { return a.logit > b.logit; });
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

    for (float& probability : probabilities) probability /= total;
    if (options.min_p > 0.0F && probabilities.size() > 1) {
        const float cutoff = options.min_p * probabilities.front();
        std::size_t keep = probabilities.size();
        for (std::size_t i = 1; i < probabilities.size(); ++i) {
            if (probabilities[i] < cutoff) { keep = i; break; }
        }
        candidates.resize(keep);
        probabilities.resize(keep);
        float kept_total = 0.0F;
        for (const float probability : probabilities) kept_total += probability;
        for (float& probability : probabilities) probability /= kept_total;
    }
    if (options.top_p < 1.0F) {
        float cumulative = 0.0F;
        std::size_t keep = probabilities.size();
        for (std::size_t i = 0; i < probabilities.size(); ++i) {
            cumulative += probabilities[i];
            if (cumulative >= options.top_p) { keep = i + 1; break; }
        }
        candidates.resize(keep);
        probabilities.resize(keep);
        float kept_total = 0.0F;
        for (const float probability : probabilities) kept_total += probability;
        for (float& probability : probabilities) probability /= kept_total;
    }

    std::discrete_distribution<std::size_t> distribution(probabilities.begin(), probabilities.end());
    return candidates[distribution(rng)].id;
}

} // namespace

std::vector<bpe::TokenId> generate_tokens(
    const std::vector<bpe::TokenId>& prompt_tokens,
    const TransformerModelWeights& weights,
    const Tensor& cos_cache,
    const Tensor& sin_cache,
    const CublasLtContext& cublas_context,
    const TransformerModelOptions& model_options,
    const GenerationOptions& options,
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
    for (std::size_t step = 0; step < options.max_new_tokens; ++step) {
        const std::size_t begin = result.size() > options.max_context_tokens
            ? result.size() - options.max_context_tokens : 0;
        // Avoid debug-iterator arithmetic here: generation is also used from
        // the Debug build, where MSVC turns an invalid seek into a modal abort.
        const std::vector<bpe::TokenId> context(
            result.data() + begin, result.data() + result.size());
        Tensor logits({context.size(), model_options.vocabulary_size}, weights.token_embedding.dtype());
        TransformerModelWorkspace workspace(model_options, 1, context.size(), weights.token_embedding.dtype());
        bpe::TokenId* device_ids = nullptr;
        CUDA_CHECK(cudaMalloc(&device_ids, context.size() * sizeof(bpe::TokenId)));
        try {
            embedding_upload_token_ids(device_ids, context, stream);
            transformer_model_forward(logits, device_ids, 1, context.size(), weights, workspace,
                                     cos_cache, sin_cache, cublas_context, stream, model_options);
            CUDA_CHECK(cudaStreamSynchronize(stream));
            std::vector<float> host_logits(logits.numel());
            logits.copy_to_host(host_logits);
            if (host_logits.size() < model_options.vocabulary_size) {
                throw std::runtime_error("generation logits are smaller than the vocabulary");
            }
            const std::size_t last_row = host_logits.size() - model_options.vocabulary_size;
            std::vector<float> last_logits(
                host_logits.data() + last_row, host_logits.data() + host_logits.size());
            const bpe::TokenId next = sample(last_logits, context, options, rng);
            result.push_back(next);
            if (options.eos_token_id && next == *options.eos_token_id) break;
        } catch (...) {
            cudaFree(device_ids);
            throw;
        }
        CUDA_CHECK(cudaFree(device_ids));
    }
    return result;
}

std::string generate_text(
    const bpe::BpeTokenizer& tokenizer, std::string_view prompt,
    const TransformerModelWeights& weights, const Tensor& cos_cache, const Tensor& sin_cache,
    const CublasLtContext& cublas_context, const TransformerModelOptions& model_options,
    const GenerationOptions& options, cudaStream_t stream
) {
    GenerationOptions text_options = options;
    if (!text_options.eos_token_id) {
        // Tokenizers use different conventional names for the same semantic
        // end marker. Prefer the first one present and otherwise allow the
        // caller to generate until max_new_tokens.
        for (const std::string_view name : {"<eos>", "</s>", "<|endoftext|>", "<|end|>"}) {
            if (const auto token = tokenizer.special_token_id(name)) {
                text_options.eos_token_id = token;
                break;
            }
        }
    }
    return tokenizer.decode(generate_tokens(tokenizer.encode(prompt), weights, cos_cache, sin_cache,
                                            cublas_context, model_options, text_options, stream));
}
