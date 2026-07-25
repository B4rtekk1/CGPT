#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <iosfwd>
#include <limits>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace bpe {
    using TokenId = std::uint32_t;

    enum class PretokenizerMode : std::uint8_t {
        None = 0,
        GptLike = 1
    };

    enum class TrainerMode : std::uint8_t {
        ExactInMemory = 0,
        LowMemoryStreaming = 1
    };

    struct TrainerConfig {
        std::size_t vocab_size = 32'000;
        std::size_t min_pair_frequency = 2;
        std::size_t worker_count = 0;
        PretokenizerMode pretokenizer = PretokenizerMode::GptLike;
        TrainerMode mode = TrainerMode::ExactInMemory;
        // LowMemoryStreaming keeps only one input block, its encoded symbols,
        // and pair counters in RAM. More merges per pass reduce I/O but make
        // training an approximation of strictly sequential BPE. Set to 1 for
        // exact merge selection at the cost of one corpus pass per merge.
        std::size_t low_memory_merges_per_pass = 8;
        // Dense pair counters are used while vocab_size <= this value.
        // 2048 consumes 32 MiB with uint64 counters.
        std::size_t low_memory_dense_vocab_limit = 2'048;
        std::vector<std::string> special_tokens = {
            "<unk>", "<bos>", "<eos>", "<pad>"
        };
    };

    struct EncodeOptions {
        // Per-thread bounded cache. Set either value to zero to disable it.
        std::size_t cache_entries = 4'096;
        std::size_t cache_max_input_bytes = 256;
    };

    struct MergeRule {
        TokenId left = 0;
        TokenId right = 0;
        TokenId merged = 0;
    };

    class BpeTokenizer {
    public:
        static constexpr TokenId kByteVocabularySize = 256;

        [[nodiscard]] static BpeTokenizer train(
            const std::vector<std::string>& documents,
            const TrainerConfig& config = {});

        [[nodiscard]] static BpeTokenizer train(
            std::istream& input,
            const TrainerConfig& config,
            std::size_t block_size = 1 << 20);

        [[nodiscard]] static BpeTokenizer train_streaming(
            std::istream& input,
            const TrainerConfig& config = {},
            std::size_t block_size = 1 << 20);

        [[nodiscard]] static BpeTokenizer train_low_memory(
            std::istream& input,
            const TrainerConfig& config = {},
            std::size_t block_size = 32U << 20);

        [[nodiscard]] std::vector<TokenId> encode(std::string_view text) const;
        [[nodiscard]] std::vector<TokenId> encode(
            std::string_view text,
            const EncodeOptions& options) const;

        [[nodiscard]] std::vector<std::vector<TokenId>> encode_batch(
            const std::vector<std::string>& texts,
            std::size_t worker_count = 0,
            const EncodeOptions& options = {}) const;
        [[nodiscard]] std::vector<std::vector<TokenId>> encode_batch(
            std::span<const std::string_view> texts,
            std::size_t worker_count = 0,
            const EncodeOptions& options = {}) const;

        [[nodiscard]] std::string decode(std::span<const TokenId> ids) const;

        void save(const std::filesystem::path& path) const;
        [[nodiscard]] static BpeTokenizer load(const std::filesystem::path& path);

        [[nodiscard]] std::size_t vocab_size() const noexcept;
        [[nodiscard]] const std::vector<MergeRule>& merges() const noexcept;
        [[nodiscard]] const std::vector<std::string>& special_tokens() const noexcept;
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
        [[nodiscard]] TokenId lookup_merge(
            TokenId left,
            TokenId right,
            unsigned dense_limit_bits) const noexcept;
        [[nodiscard]] bool is_special(TokenId id) const noexcept;

        struct SpecialMatcherNode {
            static constexpr std::uint32_t kMissing =
                std::numeric_limits<std::uint32_t>::max();

            std::array<std::uint32_t, 256> transitions{};
            std::vector<std::uint32_t> outputs;
            std::uint32_t failure = 0;

            SpecialMatcherNode() { transitions.fill(kMissing); }
        };

        PretokenizerMode pretokenizer_mode_ = PretokenizerMode::GptLike;
        std::vector<std::string> special_tokens_;
        std::vector<SpecialMatcherNode> special_matcher_;
        std::vector<MergeRule> merges_;
        std::vector<std::vector<std::uint8_t>> token_bytes_;
        std::unordered_map<std::uint64_t, TokenId> merge_lookup_;
        // GigaToken-style two-level lookup used by the encoding hot path:
        // early token IDs use an L3-resident dense table, while the remaining
        // pairs use a packed, immutable, open-addressed table.
        std::vector<TokenId> dense_merge_lookup_;
        unsigned dense_merge_bits_ = 0;
        std::vector<std::uint64_t> packed_merge_lookup_;
        std::size_t packed_merge_mask_ = 0;
        unsigned packed_merge_shift_ = 0;
        std::uint64_t cache_identity_ = 0;
    };
}