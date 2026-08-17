#pragma once

#include <array>
#include <cstdint>
#include <filesystem>
#include <iosfwd>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <optional>
#include <vector>

namespace bpe {
    /** @brief Integer identifier assigned to a vocabulary entry. */
    using TokenId = std::uint32_t;

    /** @brief Pretokenization strategy applied before byte-pair encoding. */
    enum class PretokenizerMode : std::uint8_t {
        /** @brief Treat the complete input as one piece. */
        None = 0,
        /** @brief Use GPT-style pretokenization. */
        GptLike = 1
    };

    /** @brief Algorithm used to train the tokenizer. */
    enum class TrainerMode : std::uint8_t {
        /** @brief Keep the training representation in memory. */
        ExactInMemory = 0,
        /** @brief Recompute statistics in bounded-memory streaming passes. */
        LowMemoryStreaming = 1
    };

    /** @brief Serialization format used by the tokenizer. */
    enum class TokenizerFormat : std::uint8_t {
        /** @brief Detect the format from the file. */
        Auto = 0,
        /** @brief Hugging Face tokenizer JSON format. */
        HuggingFaceJson = 1,
        /** @brief Native compact binary format. */
        Binary = 2
    };

    /** @brief Configuration for BPE vocabulary training. */
    struct TrainerConfig {
        /** @brief Target vocabulary size. */
        std::size_t vocab_size = 32'000;
        /** @brief Minimum frequency required for a pair merge. */
        std::size_t min_pair_frequency = 2;
        /** @brief Number of worker threads; zero selects the default. */
        std::size_t worker_count = 0;
        /** @brief Pretokenization mode. */
        PretokenizerMode pretokenizer = PretokenizerMode::GptLike;
        /** @brief Training memory strategy. */
        TrainerMode mode = TrainerMode::ExactInMemory;
        /** @brief Number of merges per low-memory pass. */
        std::size_t low_memory_merges_per_pass = 8;
        /** @brief Vocabulary limit for dense merge lookup tables. */
        std::size_t low_memory_dense_vocab_limit = 2'048;
        /** @brief Reserved special vocabulary entries. */
        std::vector<std::string> special_tokens = {
            "<unk>", "<bos>", "<eos>", "<pad>"
        };
    };

    /** @brief Controls optional encoding cache behavior. */
    struct EncodeOptions {
        /** @brief Maximum number of cached encoded pieces. */
        std::size_t cache_entries = 65'536;
        /** @brief Maximum cached input size in bytes. */
        std::size_t cache_max_input_bytes = 256;
    };

    /** @brief Flattened batch encoding result with per-input offsets. */
    struct EncodedBatch {
        /** @brief Concatenated token IDs for all inputs. */
        std::vector<TokenId> tokens;
        /** @brief Start offsets for each input in @ref tokens. */
        std::vector<std::size_t> offsets;
    };

    /** @brief One BPE merge from two token IDs into a new token ID. */
    struct MergeRule {
        /** @brief Left-hand token ID. */
        TokenId left = 0;
        /** @brief Right-hand token ID. */
        TokenId right = 0;
        /** @brief Resulting merged token ID. */
        TokenId merged = 0;
    };

    /** @brief Byte-pair encoding tokenizer with training and serialization support. */
    class BpeTokenizer {
    public:
        /** @brief Number of byte tokens in the base vocabulary. */
        static constexpr TokenId kByteVocabularySize = 256;

        /** @brief Trains a tokenizer from an in-memory document collection. */
        [[nodiscard]] static BpeTokenizer train(
            const std::vector<std::string>& documents,
            const TrainerConfig& config = {});

        /** @brief Trains a tokenizer from an input stream. */
        [[nodiscard]] static BpeTokenizer train(
            std::istream& input,
            const TrainerConfig& config,
            std::size_t block_size = 1 << 20);

        /** @brief Trains a tokenizer using streaming input processing. */
        [[nodiscard]] static BpeTokenizer train_streaming(
            std::istream& input,
            const TrainerConfig& config = {},
            std::size_t block_size = 1 << 20);

        /** @brief Trains a tokenizer using bounded-memory merge passes. */
        [[nodiscard]] static BpeTokenizer train_low_memory(
            std::istream& input,
            const TrainerConfig& config = {},
            std::size_t block_size = 32U << 20);

        /** @brief Encodes text into token IDs using default options. */
        [[nodiscard]] std::vector<TokenId> encode(std::string_view text) const;
        /** @brief Encodes text using explicit cache options. */
        [[nodiscard]] std::vector<TokenId> encode(
            std::string_view text,
            const EncodeOptions& options) const;

        /** @brief Encodes a vector of strings, optionally in parallel. */
        [[nodiscard]] std::vector<std::vector<TokenId>> encode_batch(
            const std::vector<std::string>& texts,
            std::size_t worker_count = 0,
            const EncodeOptions& options = {}) const;
        /** @brief Encodes a span of string views, optionally in parallel. */
        [[nodiscard]] std::vector<std::vector<TokenId>> encode_batch(
            std::span<const std::string_view> texts,
            std::size_t worker_count = 0,
            const EncodeOptions& options = {}) const;

        /** @brief Encodes inputs into one flat token array with offsets. */
        [[nodiscard]] EncodedBatch encode_batch_flat(
            std::span<const std::string_view> texts,
            std::size_t worker_count = 0,
            const EncodeOptions& options = {}) const;

        /** @brief Decodes token IDs back into text. */
        [[nodiscard]] std::string decode(std::span<const TokenId> ids) const;

        /** @brief Saves the tokenizer in the selected format. */
        void save(
            const std::filesystem::path& path,
            TokenizerFormat format = TokenizerFormat::HuggingFaceJson) const;
        /** @brief Saves the tokenizer in Hugging Face JSON format. */
        void save_huggingface_json(const std::filesystem::path& path) const;
        /** @brief Loads a tokenizer and optionally detects its format. */
        [[nodiscard]] static BpeTokenizer load(
            const std::filesystem::path& path,
            TokenizerFormat format = TokenizerFormat::Auto);

        /** @brief Returns the vocabulary size. */
        [[nodiscard]] std::size_t vocab_size() const noexcept;
        /** @brief Returns the ordered BPE merge rules. */
        [[nodiscard]] const std::vector<MergeRule>& merges() const noexcept;
        /** @brief Returns the configured special token strings. */
        [[nodiscard]] const std::vector<std::string>& special_tokens() const noexcept;
        /** @brief Looks up a special token ID by string. */
        [[nodiscard]] std::optional<TokenId> special_token_id(std::string_view token) const noexcept;
        /** @brief Returns the tokenizer pretokenization mode. */
        [[nodiscard]] PretokenizerMode pretokenizer_mode() const noexcept;

    private:
        explicit BpeTokenizer(
            std::vector<std::string> special_tokens,
            PretokenizerMode pretokenizer = PretokenizerMode::GptLike);

        void add_merge(TokenId left, TokenId right);
        void rebuild_fast_merge_lookup();
        void rebuild_token_bytes();
        void rebuild_special_matcher();
        [[nodiscard]] std::vector<TokenId> encode_with_dense_limit(
            std::string_view text,
            const EncodeOptions& options,
            unsigned dense_limit_bits) const;
        void append_encoded_bytes(
            std::string_view text,
            std::vector<TokenId>& output,
            const EncodeOptions& options,
            unsigned dense_limit_bits) const;
        void append_encoded_bytes_uncached(
            std::string_view text,
            std::vector<TokenId>& output,
            unsigned dense_limit_bits) const;
        void encode_piece_cached(
            std::string_view piece,
            std::vector<TokenId>& output,
            const EncodeOptions& options,
            unsigned dense_limit_bits) const;
        [[nodiscard]] TokenId lookup_merge(
            TokenId left,
            TokenId right,
            unsigned dense_limit_bits) const noexcept;
        [[nodiscard]] bool is_special(TokenId id) const noexcept;
        void save_binary(const std::filesystem::path& path) const;
        [[nodiscard]] static BpeTokenizer load_binary(
            const std::filesystem::path& path);
        [[nodiscard]] static BpeTokenizer load_huggingface_json(
            const std::filesystem::path& path);

        PretokenizerMode pretokenizer_mode_ = PretokenizerMode::GptLike;
        std::vector<std::string> special_tokens_;
        struct SpecialMatcherNode {
            std::array<std::int32_t, 256> next{};
            std::int32_t token_index = -1;

            SpecialMatcherNode() { next.fill(-1); }
        };
        std::vector<SpecialMatcherNode> special_matcher_;
        std::vector<MergeRule> merges_;
        std::vector<std::vector<std::uint8_t>> token_bytes_;
        std::unordered_map<std::uint64_t, TokenId> merge_lookup_;
        std::vector<TokenId> dense_merge_lookup_;
        unsigned dense_merge_bits_ = 0;
        std::vector<std::uint64_t> packed_merge_lookup_;
        std::size_t packed_merge_mask_ = 0;
        unsigned packed_merge_shift_ = 0;
        std::uint64_t cache_identity_ = 0;
    };
}
