#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace bpe {
    using TokenId = std::uint32_t;

    struct TrainerConfig {
        std::size_t vocab_size = 32'000;
        std::size_t min_pair_frequency = 2;
        std::size_t worker_count = 0; // 0 = std::thread::hardware_concurrency()
        std::vector<std::string> special_tokens = {
            "<unk>", "<bos>", "<eos>", "<pad>"
        };
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

        [[nodiscard]] std::vector<TokenId> encode(std::string_view text) const;
        [[nodiscard]] std::string decode(std::span<const TokenId> ids) const;

        void save(const std::filesystem::path& path) const;
        [[nodiscard]] static BpeTokenizer load(const std::filesystem::path& path);

        [[nodiscard]] std::size_t vocab_size() const noexcept;
        [[nodiscard]] const std::vector<MergeRule>& merges() const noexcept;
        [[nodiscard]] const std::vector<std::string>& special_tokens() const noexcept;

    private:
        explicit BpeTokenizer(std::vector<std::string> special_tokens);

        void add_merge(TokenId left, TokenId right);
        void rebuild_token_bytes();
        [[nodiscard]] bool is_special(TokenId id) const noexcept;

        std::vector<std::string> special_tokens_;
        std::vector<MergeRule> merges_;
        std::vector<std::vector<std::uint8_t>> token_bytes_;
    };

}