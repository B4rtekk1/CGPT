/**
 * @file generation_cpu.cpp
 * @brief CPU implementation of autoregressive Transformer text generation.
 *
 * This translation unit performs complete Transformer forward passes on CPU,
 * applies configurable logits-processing and sampling rules, and exposes token-
 * and text-level generation entry points.
 */
#include "core/generation.h"
#include "ops/cpu/embedding_cpu.h"
#include "ops/cpu/linear_cpu.h"
#include "ops/cpu/rmsnorm_cpu.h"
#include "ops/cpu/rope_cpu.h"
#include "ops/cpu/attention_cpu.h"
#include "ops/cpu/swiglu_cpu.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <cstdint>
#include <limits>
#include <random>
#include <stdexcept>

namespace {
    /**
     * @brief Copies the complete contents of one CPU tensor into another.
     *
     * The tensors must have identical storage sizes and data types. This helper
     * performs a raw byte copy and therefore assumes compatible tensor layouts.
     *
     * @param dst Destination tensor.
     * @param src Source tensor.
     *
     * @throws std::invalid_argument If the tensors differ in size or type, or
     *         if either tensor is not CPU-resident.
     */
    void copy_tensor(Tensor &dst, const Tensor &src) {
        if (dst.nbytes() != src.nbytes() || dst.dtype() != src.dtype() ||
            dst.device_type() != DeviceType::CPU || src.device_type() != DeviceType::CPU) {
            throw std::invalid_argument("CPU generation: incompatible tensor copy");
        }
        std::memcpy(dst.raw_data(), src.raw_data(), src.nbytes());
    }

    /**
     * @brief Adds a tensor to another tensor in place.
     *
     * For F32 tensors the operation is performed directly on CPU memory. Other
     * supported floating-point formats are converted through temporary F32 host
     * buffers before the result is written back.
     *
     * @param[in,out] a Tensor receiving the element-wise sum.
     * @param b Tensor added to @p a.
     *
     * @throws std::invalid_argument If tensor shapes, data types, or devices do
     *         not match, or if either tensor is not CPU-resident.
     */
    void add(Tensor &a, const Tensor &b) {
        if (a.shape() != b.shape() || a.dtype() != b.dtype() ||
            a.device_type() != DeviceType::CPU || b.device_type() != DeviceType::CPU) {
            throw std::invalid_argument("CPU generation: incompatible residual tensors");
        }
        if (a.dtype() == Dtype::F32) {
            auto *dst = static_cast<float *>(a.raw_data());
            const auto *src = static_cast<const float *>(b.raw_data());
            for (std::size_t i = 0; i < a.numel(); ++i) dst[i] += src[i];
            return;
        }
        std::vector<float> av(a.numel()), bv(b.numel());
        a.copy_to_host(av);
        b.copy_to_host(bv);
        for (std::size_t i = 0; i < a.numel(); ++i) av[i] += bv[i];
        a.copy_from_host(av);
    }

    /** Reusable buffers and a per-layer CPU key/value cache for one decode stream. */
    struct CpuLayerWorkspace {
        Tensor attention_norm, q_flat, k_flat, value_flat, q_norm_input, k_norm_input;
        Tensor q_norm_output, k_norm_output, query, key, attention, attention_flat;
        Tensor attention_projection, ffn_norm, gate, up, activated, ffn_output;
        Tensor key_cache, value_cache;

        CpuLayerWorkspace(const TransformerBlockOptions &o, const std::size_t context, const Dtype dtype)
            : attention_norm({1, o.hidden_size}, dtype, DeviceType::CPU),
              q_flat({1, o.num_query_heads * o.head_dim}, dtype, DeviceType::CPU),
              k_flat({1, o.num_kv_heads * o.head_dim}, dtype, DeviceType::CPU),
              value_flat({1, o.num_kv_heads * o.head_dim}, dtype, DeviceType::CPU),
              q_norm_input({o.num_query_heads, o.head_dim}, dtype, DeviceType::CPU),
              k_norm_input({o.num_kv_heads, o.head_dim}, dtype, DeviceType::CPU),
              q_norm_output({o.num_query_heads, o.head_dim}, dtype, DeviceType::CPU),
              k_norm_output({o.num_kv_heads, o.head_dim}, dtype, DeviceType::CPU),
              query({1, 1, o.num_query_heads, o.head_dim}, dtype, DeviceType::CPU),
              key({1, 1, o.num_kv_heads, o.head_dim}, dtype, DeviceType::CPU),
              attention({1, 1, o.num_query_heads, o.head_dim}, dtype, DeviceType::CPU),
              attention_flat({1, o.hidden_size}, dtype, DeviceType::CPU),
              attention_projection({1, o.hidden_size}, dtype, DeviceType::CPU),
              ffn_norm({1, o.hidden_size}, dtype, DeviceType::CPU),
              gate({1, o.intermediate_size}, dtype, DeviceType::CPU),
              up({1, o.intermediate_size}, dtype, DeviceType::CPU),
              activated({1, o.intermediate_size}, dtype, DeviceType::CPU),
              ffn_output({1, o.hidden_size}, dtype, DeviceType::CPU),
              key_cache({1, context, o.num_kv_heads, o.head_dim}, dtype, DeviceType::CPU),
              value_cache({1, context, o.num_kv_heads, o.head_dim}, dtype, DeviceType::CPU) {}
    };

    struct CpuGenerationWorkspace {
        Tensor hidden, residual, normalized, logits;
        std::vector<CpuLayerWorkspace> layers;

        CpuGenerationWorkspace(const TransformerModelOptions &o, const std::size_t context, const Dtype dtype)
            : hidden({1, o.block_options.hidden_size}, dtype, DeviceType::CPU),
              residual({1, o.block_options.hidden_size}, dtype, DeviceType::CPU),
              normalized({1, o.block_options.hidden_size}, dtype, DeviceType::CPU),
              logits({1, o.vocabulary_size}, dtype, DeviceType::CPU) {
            layers.reserve(o.num_layers);
            for (std::size_t index = 0; index < o.num_layers; ++index)
                layers.emplace_back(o.block_options, context, dtype);
        }
    };

    /** Executes one token while appending its K/V vectors to every layer cache. */
    void forward_cached_token(CpuGenerationWorkspace &workspace, const bpe::TokenId token,
                              const std::size_t position, const TransformerModelWeights &weights,
                              const TransformerModelOptions &options, const Tensor &cosine, const Tensor &sine) {
        const auto &block = options.block_options;
        if (position >= workspace.layers.front().key_cache.shape()[1])
            throw std::out_of_range("CPU generation: KV cache capacity exceeded");
        embedding_forward_cpu(workspace.hidden, &token, 1, weights.token_embedding);
        for (std::size_t index = 0; index < weights.layers.size(); ++index) {
            const auto &weights_layer = weights.layers[index];
            auto &layer = workspace.layers[index];
            copy_tensor(workspace.residual, workspace.hidden);
            rmsnorm_forward_cpu(layer.attention_norm, workspace.residual, weights_layer.attention_norm, block.rms_epsilon);
            linear_forward_cpu(layer.q_flat, layer.attention_norm, weights_layer.q_projection);
            linear_forward_cpu(layer.k_flat, layer.attention_norm, weights_layer.k_projection);
            linear_forward_cpu(layer.value_flat, layer.attention_norm, weights_layer.v_projection);
            copy_tensor(layer.q_norm_input, layer.q_flat);
            copy_tensor(layer.k_norm_input, layer.k_flat);
            rmsnorm_forward_cpu(layer.q_norm_output, layer.q_norm_input, weights_layer.q_norm, block.rms_epsilon);
            rmsnorm_forward_cpu(layer.k_norm_output, layer.k_norm_input, weights_layer.k_norm, block.rms_epsilon);
            copy_tensor(layer.query, layer.q_norm_output);
            copy_tensor(layer.key, layer.k_norm_output);
            rope_forward_cpu(layer.query, layer.key, cosine, sine, {block.rotary_dim, position});

            const std::size_t token_bytes = layer.key.nbytes();
            auto *key_destination = static_cast<std::uint8_t *>(layer.key_cache.raw_data()) + position * token_bytes;
            auto *value_destination = static_cast<std::uint8_t *>(layer.value_cache.raw_data()) + position * token_bytes;
            std::memcpy(key_destination, layer.key.raw_data(), token_bytes);
            std::memcpy(value_destination, layer.value_flat.raw_data(), token_bytes);

            FlashAttentionOptions attention_options{block.num_query_heads, block.num_kv_heads, block.head_dim,
                                                     0.0F, block.causal, position};
            flash_gqa_attention_forward_cpu(layer.attention, layer.query, layer.key_cache, layer.value_cache,
                                            attention_options, position + 1);
            copy_tensor(layer.attention_flat, layer.attention);
            linear_forward_cpu(layer.attention_projection, layer.attention_flat, weights_layer.o_projection);
            rmsnorm_forward_cpu(layer.ffn_norm, workspace.residual, weights_layer.ffn_norm, block.rms_epsilon);
            linear_forward_cpu(layer.gate, layer.ffn_norm, weights_layer.gate_proj);
            linear_forward_cpu(layer.up, layer.ffn_norm, weights_layer.up_proj);
            swiglu_forward_cpu(layer.activated, layer.gate, layer.up);
            linear_forward_cpu(layer.ffn_output, layer.activated, weights_layer.down_proj);
            copy_tensor(workspace.hidden, workspace.residual);
            add(workspace.hidden, layer.attention_projection);
            add(workspace.hidden, layer.ffn_output);
        }
        rmsnorm_forward_cpu(workspace.normalized, workspace.hidden, weights.final_norm, block.rms_epsilon);
        linear_forward_cpu(workspace.logits, workspace.normalized, weights.lm_head);
    }

    /**
     * @brief Validates generation sampling parameters.
     *
     * @param o Sampling and logits-processing options to validate.
     *
     * @throws std::invalid_argument If any sampling parameter is non-finite or
     *         outside its supported range.
     */
    void validate_sampling(const GenerationOptions &o) {
        if (!std::isfinite(o.temperature) || o.temperature < 0.0F)
            throw std::invalid_argument("temperature must be finite and non-negative");
        if (!std::isfinite(o.top_p) || o.top_p <= 0.0F || o.top_p > 1.0F)
            throw std::invalid_argument("top_p must be in (0, 1]");
        if (!std::isfinite(o.repetition_penalty) || o.repetition_penalty <= 0.0F)
            throw std::invalid_argument("repetition_penalty must be finite and positive");
        if (!std::isfinite(o.presence_penalty) || !std::isfinite(o.frequency_penalty) ||
            o.presence_penalty < 0.0F || o.frequency_penalty < 0.0F)
            throw std::invalid_argument("presence and frequency penalties must be finite and non-negative");
        if (!std::isfinite(o.min_p) || o.min_p < 0.0F || o.min_p > 1.0F)
            throw std::invalid_argument("min_p must be in [0, 1]");
    }

    /**
     * @brief Selects the next token from a vector of vocabulary logits.
     *
     * The function applies repetition, presence, and frequency penalties,
     * optional no-repeat n-gram masking, top-k filtering, min-p filtering, and
     * nucleus sampling. A temperature of zero enables deterministic greedy
     * selection after logits processing.
     *
     * @param logits Logits for every token in the vocabulary.
     * @param ctx Complete generated-token context used by repetition penalties
     *        and no-repeat n-gram masking.
     * @param o Sampling configuration.
     * @param rng Random-number generator used for stochastic sampling.
     * @return Identifier of the selected token.
     *
     * @throws std::invalid_argument If @p logits is empty or sampling options
     *         are invalid.
     * @throws std::runtime_error If logits processing removes every token.
     */
    bpe::TokenId sample(const std::vector<float> &logits, const std::vector<bpe::TokenId> &ctx,
                        const GenerationOptions &o, std::mt19937_64 &rng) {
        if (logits.empty()) throw std::invalid_argument("cannot sample from empty logits");
        validate_sampling(o);
        std::vector<float> adjusted = logits;
        std::vector<std::size_t> counts(adjusted.size());
        for (const auto id: ctx) if (id < counts.size()) ++counts[id];
        for (std::size_t i = 0; i < adjusted.size(); ++i)
            if (counts[i]) {
                adjusted[i] = adjusted[i] >= 0.0F
                                  ? adjusted[i] / o.repetition_penalty
                                  : adjusted[i] * o.repetition_penalty;
                adjusted[i] -= o.presence_penalty + o.frequency_penalty * static_cast<float>(counts[i]);
            }

        if (o.no_repeat_ngram_size > 1 && ctx.size() >= o.no_repeat_ngram_size) {
            const std::size_t n = o.no_repeat_ngram_size;
            const std::size_t prefix_begin = ctx.size() - (n - 1);
            for (std::size_t start = 0; start + n <= ctx.size() - 1; ++start) {
                bool same_prefix = true;
                for (std::size_t offset = 0; offset + 1 < n; ++offset) {
                    if (ctx[start + offset] != ctx[prefix_begin + offset]) {
                        same_prefix = false;
                        break;
                    }
                }
                if (same_prefix && ctx[start + n - 1] < adjusted.size())
                    adjusted[ctx[start + n - 1]] = -std::numeric_limits<float>::infinity();
            }
        }

        std::vector<std::size_t> ids;
        ids.reserve(adjusted.size());
        for (std::size_t i = 0; i < adjusted.size(); ++i)
            if (std::isfinite(adjusted[i])) ids.push_back(i);
        if (ids.empty()) throw std::runtime_error("no valid token remains after logits processing");
        const auto before = [&](const auto left, const auto right) { return adjusted[left] > adjusted[right]; };
        if (o.temperature == 0.0F) {
            return static_cast<bpe::TokenId>(*std::max_element(ids.begin(), ids.end(),
                [&](const auto left, const auto right) { return adjusted[left] < adjusted[right]; }));
        }
        if (o.top_k != 0 && o.top_k < ids.size()) {
            std::nth_element(ids.begin(), ids.begin() + static_cast<std::ptrdiff_t>(o.top_k), ids.end(), before);
            ids.resize(o.top_k);
        }
        // Nucleus/min-p sampling requires probabilities in descending order.
        // With top-p=1 and min-p=0 their order is irrelevant, avoiding a full
        // sort of the 32k-token vocabulary on every decode step.
        if (o.top_p < 1.0F || o.min_p > 0.0F)
            std::ranges::sort(ids, before);

        const float max_logit = adjusted[*std::max_element(ids.begin(), ids.end(),
            [&](const auto left, const auto right) { return adjusted[left] < adjusted[right]; })];
        std::vector<float> probabilities(ids.size());
        float total = 0.0F;
        for (std::size_t i = 0; i < ids.size(); ++i) {
            probabilities[i] = std::exp((adjusted[ids[i]] - max_logit) / o.temperature);
            total += probabilities[i];
        }
        for (float &probability: probabilities) probability /= total;
        if (o.min_p > 0.0F && probabilities.size() > 1) {
            const auto best = std::distance(probabilities.begin(), std::max_element(probabilities.begin(), probabilities.end()));
            const float cutoff = o.min_p * probabilities[best];
            std::size_t keep = probabilities.size();
            for (std::size_t i = 1; i < probabilities.size(); ++i)
                if (probabilities[i] < cutoff) {
                    keep = i;
                    break;
                }
            ids.resize(keep);
            probabilities.resize(keep);
            float kept_total = 0.0F;
            for (const float p: probabilities) kept_total += p;
            for (float &p: probabilities) p /= kept_total;
        }
        if (o.top_p < 1.0F) {
            float cumulative = 0.0F;
            std::size_t keep = probabilities.size();
            for (std::size_t i = 0; i < probabilities.size(); ++i) {
                cumulative += probabilities[i];
                if (cumulative >= o.top_p) {
                    keep = i + 1;
                    break;
                }
            }
            ids.resize(keep);
            probabilities.resize(keep);
            float kept_total = 0.0F;
            for (const float p: probabilities) kept_total += p;
            for (float &p: probabilities) p /= kept_total;
        }
        std::discrete_distribution<std::size_t> distribution(probabilities.begin(), probabilities.end());
        return static_cast<bpe::TokenId>(ids[distribution(rng)]);
    }

    /**
     * @brief Executes a full CPU Transformer forward pass for a token sequence.
     *
     * Each decoder layer applies pre-normalized grouped-query attention with
     * rotary position embeddings, followed by a pre-normalized SwiGLU feed-
     * forward network and residual additions. The final normalized hidden states
     * are projected through the language-model head.
     *
     * @param[out] logits Output tensor with shape
     *        `[sequence_length, vocabulary_size]`.
     * @param ids Input token identifiers.
     * @param w CPU-resident Transformer weights.
     * @param o Transformer architecture options.
     * @param co Precomputed RoPE cosine cache.
     * @param si Precomputed RoPE sine cache.
     */
    void forward(Tensor &logits, const std::vector<bpe::TokenId> &ids, const TransformerModelWeights &w,
                 const TransformerModelOptions &o, const Tensor &co, const Tensor &si) {
        const auto S = ids.size(), H = o.block_options.hidden_size, Q = o.block_options.num_query_heads, KV = o.
                block_options.num_kv_heads, D = o.block_options.head_dim, I = o.block_options.intermediate_size;
        Tensor h({S, H}, w.token_embedding.dtype(), DeviceType::CPU);
        embedding_forward_cpu(h, ids.data(), S, w.token_embedding);
        for (const auto &l: w.layers) {
            Tensor residual(h.shape(), h.dtype(), DeviceType::CPU);
            copy_tensor(residual, h);
            Tensor n({S, H}, h.dtype(), DeviceType::CPU);
            rmsnorm_forward_cpu(n, residual, l.attention_norm, o.block_options.rms_epsilon);
            Tensor qf({S, Q * D}, h.dtype(), DeviceType::CPU), kf({S, KV * D}, h.dtype(), DeviceType::CPU), vf(
                {S, KV * D}, h.dtype(), DeviceType::CPU);
            linear_forward_cpu(qf, n, l.q_projection);
            linear_forward_cpu(kf, n, l.k_projection);
            linear_forward_cpu(vf, n, l.v_projection);
            Tensor qn({S * Q, D}, h.dtype(), DeviceType::CPU), kn({S * KV, D}, h.dtype(), DeviceType::CPU);
            copy_tensor(qn, qf);
            copy_tensor(kn, kf);
            Tensor qnw({D}, h.dtype(), DeviceType::CPU), knw({D}, h.dtype(), DeviceType::CPU);
            copy_tensor(qnw, l.q_norm);
            copy_tensor(knw, l.k_norm);
            Tensor qno(qn.shape(), h.dtype(), DeviceType::CPU), kno(kn.shape(), h.dtype(), DeviceType::CPU);
            rmsnorm_forward_cpu(qno, qn, qnw, o.block_options.rms_epsilon);
            rmsnorm_forward_cpu(kno, kn, knw, o.block_options.rms_epsilon);
            copy_tensor(qf, qno);
            copy_tensor(kf, kno);
            Tensor q({1, S, Q, D}, h.dtype(), DeviceType::CPU), k({1, S, KV, D}, h.dtype(), DeviceType::CPU), v(
                {1, S, KV, D}, h.dtype(), DeviceType::CPU);
            copy_tensor(q, qf);
            copy_tensor(k, kf);
            copy_tensor(v, vf);
            RopeOptions ro{o.block_options.rotary_dim, 0};
            rope_forward_cpu(q, k, co, si, ro);
            Tensor ao({1, S, Q, D}, h.dtype(), DeviceType::CPU);
            FlashAttentionOptions aoopt{Q, KV, D, 0, o.block_options.causal};
            flash_gqa_attention_forward_cpu(ao, q, k, v, aoopt);
            Tensor af({S, H}, h.dtype(), DeviceType::CPU);
            copy_tensor(af, ao);
            Tensor proj({S, H}, h.dtype(), DeviceType::CPU);
            linear_forward_cpu(proj, af, l.o_projection);
            Tensor fn({S, H}, h.dtype(), DeviceType::CPU);
            rmsnorm_forward_cpu(fn, residual, l.ffn_norm, o.block_options.rms_epsilon);
            Tensor gate({S, I}, h.dtype(), DeviceType::CPU), up({S, I}, h.dtype(), DeviceType::CPU);
            linear_forward_cpu(gate, fn, l.gate_proj);
            linear_forward_cpu(up, fn, l.up_proj);
            Tensor act({S, I}, h.dtype(), DeviceType::CPU);
            swiglu_forward_cpu(act, gate, up);
            Tensor down({S, H}, h.dtype(), DeviceType::CPU);
            linear_forward_cpu(down, act, l.down_proj);
            add(h, proj);
            add(h, down);
        }
        Tensor norm({S, H}, h.dtype(), DeviceType::CPU);
        rmsnorm_forward_cpu(norm, h, w.final_norm, o.block_options.rms_epsilon);
        linear_forward_cpu(logits, norm, w.lm_head);
    }
}

/**
 * @brief Generates token identifiers autoregressively on the CPU.
 *
 * On every generation step, the function retains at most
 * `GenerationOptions::max_context_tokens` recent tokens, recomputes model
 * logits for that context, and samples one new token. Generation stops after
 * the configured token limit or when the optional EOS token is produced.
 *
 * @param prompt Initial prompt token identifiers. Must not be empty.
 * @param w CPU-resident Transformer weights.
 * @param co Precomputed RoPE cosine cache with sufficient sequence capacity.
 * @param si Precomputed RoPE sine cache matching @p co.
 * @param o Transformer architecture options.
 * @param go Generation and sampling options.
 * @return The prompt followed by all generated token identifiers.
 *
 * @throws std::invalid_argument If the prompt is empty, the context limit is
 *         zero, RoPE caches are incompatible, model weights are not suitable
 *         for CPU execution, or sampling options are invalid.
 * @throws std::runtime_error If logits processing leaves no valid token.
 */
std::vector<bpe::TokenId> generate_tokens_cpu(
    const std::vector<bpe::TokenId> &prompt, const TransformerModelWeights &w,
    const Tensor &co, const Tensor &si, const TransformerModelOptions &o,
    const GenerationOptions &go) {
    if (prompt.empty()) throw std::invalid_argument("prompt must contain at least one token");
    if (!go.max_new_tokens) return prompt;
    if (!go.max_context_tokens) throw std::invalid_argument("max_context_tokens must be positive");
    if (co.device_type() != DeviceType::CPU || si.device_type() != DeviceType::CPU ||
        co.dim() != 2 || si.dim() != 2 || co.shape() != si.shape() ||
        co.size(0) < go.max_context_tokens) {
        throw std::invalid_argument("CPU generation: RoPE caches are incompatible or too short");
    }
    if (w.token_embedding.device_type() != DeviceType::CPU ||
        w.final_norm.device_type() != DeviceType::CPU ||
        w.lm_head.device_type() != DeviceType::CPU ||
        w.layers.size() != o.num_layers) {
        throw std::invalid_argument("CPU generation requires CPU-resident model weights");
    }
    validate_sampling(go);
    std::mt19937_64 rng(go.seed ? go.seed : std::random_device{}());
    std::vector<bpe::TokenId> result = prompt;
    CpuGenerationWorkspace workspace(o, go.max_context_tokens, w.token_embedding.dtype());
    std::size_t cached_tokens = 0;
    const auto rebuild_cache = [&] {
        const std::size_t begin = result.size() > go.max_context_tokens
                                      ? result.size() - go.max_context_tokens
                                      : 0;
        cached_tokens = 0;
        for (std::size_t index = begin; index < result.size(); ++index)
            forward_cached_token(workspace, result[index], cached_tokens++, w, o, co, si);
    };
    rebuild_cache();
    for (std::size_t step = 0; step < go.max_new_tokens; ++step) {
        std::vector<float> last(o.vocabulary_size);
        workspace.logits.copy_to_host(last);
        const auto next = sample(last, result, go, rng);
        result.push_back(next);
        if (go.eos_token_id && next == *go.eos_token_id) break;
        if (cached_tokens == go.max_context_tokens) {
            // Preserve the established sliding-window semantics once the
            // cache is full.  Rebuilding is rare and keeps RoPE positions
            // identical to the previous full-context implementation.
            rebuild_cache();
        } else {
            forward_cached_token(workspace, next, cached_tokens++, w, o, co, si);
        }
    }
    return result;
}

/**
 * @brief Generates decoded text and token-count statistics on the CPU.
 *
 * If no EOS identifier is supplied, the tokenizer is queried for common EOS
 * special-token spellings. The prompt is then encoded, passed to
 * generate_tokens_cpu(), and the complete token sequence is decoded.
 *
 * @param tok Tokenizer used to encode the prompt and decode generated tokens.
 * @param prompt Input text prompt.
 * @param w CPU-resident Transformer weights.
 * @param co Precomputed RoPE cosine cache.
 * @param si Precomputed RoPE sine cache.
 * @param o Transformer architecture options.
 * @param go Generation and sampling options.
 * @return Generated text together with prompt and generated token counts.
 *
 * @throws std::invalid_argument If token generation inputs or options are
 *         invalid.
 * @throws std::runtime_error If sampling cannot select a valid token.
 */
TextGenerationResult generate_text_with_stats_cpu(
    const bpe::BpeTokenizer &tok, std::string_view prompt,
    const TransformerModelWeights &w, const Tensor &co, const Tensor &si,
    const TransformerModelOptions &o, const GenerationOptions &go) {
    GenerationOptions text_options = go;
    if (!text_options.eos_token_id) {
        for (const std::string_view name: {"<eos>", "</s>", "<|endoftext|>", "<|end|>"}) {
            if (const auto token = tok.special_token_id(name)) {
                text_options.eos_token_id = token;
                break;
            }
        }
    }
    const auto prompt_tokens = tok.encode(prompt);
    const auto all_tokens = generate_tokens_cpu(
        prompt_tokens, w, co, si, o, text_options);
    return {
        tok.decode(all_tokens), prompt_tokens.size(),
        all_tokens.size() - prompt_tokens.size()
    };
}
