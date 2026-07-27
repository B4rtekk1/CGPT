#include "tokenizer/bpe_tokenizer.h"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {
    void require(const bool condition, const char *message) {
        if (!condition) {
            throw std::runtime_error(message);
        }
    }

    template<typename Function>
    void expect_throw(Function &&function, const char *message) {
        try {
            function();
        } catch (const std::exception &) {
            return;
        }
        throw std::runtime_error(message);
    }

    bool same_merges(const std::vector<bpe::MergeRule> &left,
                     const std::vector<bpe::MergeRule> &right) {
        if (left.size() != right.size()) {
            return false;
        }
        for (std::size_t index = 0; index < left.size(); ++index) {
            if (left[index].left != right[index].left ||
                left[index].right != right[index].right ||
                left[index].merged != right[index].merged) {
                return false;
            }
        }
        return true;
    }

    std::vector<bpe::TokenId> reference_encode_plain(
        const std::string_view text,
        const std::vector<bpe::MergeRule>& merges
    ) {
        std::vector<bpe::TokenId> tokens;
        tokens.reserve(text.size());
        for (const unsigned char byte : text) {
            tokens.push_back(byte);
        }

        for (const bpe::MergeRule& rule : merges) {
            std::vector<bpe::TokenId> next;
            next.reserve(tokens.size());
            for (std::size_t index = 0; index < tokens.size();) {
                if (index + 1 < tokens.size() &&
                    tokens[index] == rule.left &&
                    tokens[index + 1] == rule.right) {
                    next.push_back(rule.merged);
                    index += 2;
                } else {
                    next.push_back(tokens[index]);
                    ++index;
                }
            }
            tokens = std::move(next);
        }
        return tokens;
    }

    void test_batch_flat_and_cache_options(const bpe::BpeTokenizer& tokenizer) {
        const std::vector<std::string> owned = {
            "short", "a deliberately longer document for scheduling", "", "<bos>x<eos>"};
        std::vector<std::string_view> views;
        views.reserve(owned.size());
        for (const std::string& text : owned) views.emplace_back(text);

        const bpe::EncodeOptions uncached{.cache_entries = 0, .cache_max_input_bytes = 0};
        const auto from_owned = tokenizer.encode_batch(owned, 3, uncached);
        const auto from_views = tokenizer.encode_batch(views, 3, uncached);
        require(from_owned == from_views, "Both batch API overloads must agree");

        const bpe::EncodedBatch flat = tokenizer.encode_batch_flat(views, 3, uncached);
        require(flat.offsets.size() == views.size() + 1 && flat.offsets.front() == 0,
                "Flat batch offsets have an invalid shape");
        for (std::size_t index = 0; index < views.size(); ++index) {
            require(flat.offsets[index] <= flat.offsets[index + 1],
                    "Flat batch offsets are not monotonic");
            const std::span<const bpe::TokenId> item(
                flat.tokens.data() + flat.offsets[index],
                flat.offsets[index + 1] - flat.offsets[index]);
            require(std::vector<bpe::TokenId>(item.begin(), item.end()) == from_views[index],
                    "Flat batch tokens differ from regular batch encoding");
        }
        require(flat.offsets.back() == flat.tokens.size(),
                "Flat batch offsets do not cover all tokens");

        const std::span<const std::string_view> empty;
        const auto empty_flat = tokenizer.encode_batch_flat(empty);
        require(empty_flat.tokens.empty() && empty_flat.offsets == std::vector<std::size_t>{0},
                "Empty flat batch has invalid output");
    }

    void test_low_memory_and_pretokenizer_modes() {
        const std::string corpus = "alpha alpha beta beta 123456 punctuation! alpha beta";
        bpe::TrainerConfig low_memory;
        low_memory.vocab_size = 264;
        low_memory.min_pair_frequency = 2;
        low_memory.mode = bpe::TrainerMode::LowMemoryStreaming;
        low_memory.low_memory_merges_per_pass = 2;
        low_memory.low_memory_dense_vocab_limit = 512;

        std::istringstream stream(corpus);
        const bpe::BpeTokenizer tokenizer =
            bpe::BpeTokenizer::train(stream, low_memory, 5);
        const std::string text = "<bos>alpha 123! beta<eos>";
        require(tokenizer.decode(tokenizer.encode(text)) == text,
                "Low-memory tokenizer round-trip failed");
        require(tokenizer.merges().size() > 0,
                "Low-memory tokenizer did not train any merges");

        bpe::TrainerConfig none = low_memory;
        none.mode = bpe::TrainerMode::ExactInMemory;
        none.pretokenizer = bpe::PretokenizerMode::None;
        const bpe::BpeTokenizer no_pretokenizer =
            bpe::BpeTokenizer::train({"ab ab ab"}, none);
        require(no_pretokenizer.pretokenizer_mode() == bpe::PretokenizerMode::None,
                "Tokenizer did not retain its pretokenizer mode");
        require(no_pretokenizer.decode(no_pretokenizer.encode("ab ab")) == "ab ab",
                "Tokenizer without pretokenization did not round-trip");

        const auto binary_path = std::filesystem::temp_directory_path() /
                                 "cgpt_bpe_none_pretokenizer.bin";
        no_pretokenizer.save(binary_path, bpe::TokenizerFormat::Binary);
        const auto reloaded = bpe::BpeTokenizer::load(binary_path, bpe::TokenizerFormat::Binary);
        require(reloaded.pretokenizer_mode() == bpe::PretokenizerMode::None,
                "Binary model lost the pretokenizer mode");
        std::filesystem::remove(binary_path);

        expect_throw([&] {
            no_pretokenizer.save("ignored.json", bpe::TokenizerFormat::Auto);
        }, "Saving with Auto format was accepted");
        expect_throw([] {
            bpe::TrainerConfig invalid;
            invalid.mode = bpe::TrainerMode::LowMemoryStreaming;
            invalid.low_memory_merges_per_pass = 0;
            std::istringstream input("text");
            (void)bpe::BpeTokenizer::train_low_memory(input, invalid, 16);
        }, "Zero low-memory merges per pass was accepted");
    }
}

int main() {
    try {
        static_assert(sizeof(bpe::BpeTokenizer) < 4'096,
                      "BpeTokenizer must keep large lookup tables off the stack");
        const std::vector<std::string> corpus = {
            "Ala ma kota. Ala ma psa.",
            "kot kot kot pies pies",
            "Zażółć gęślą jaźń 🦀",
            "int main() { return 0; }"
        };

        bpe::TrainerConfig config;
        config.vocab_size = 300;
        config.worker_count = 4;

        const bpe::BpeTokenizer tokenizer = bpe::BpeTokenizer::train(corpus, config);
        const std::string text = "<bos>Zażółć kota 🦀<eos>";
        const std::vector<bpe::TokenId> ids = tokenizer.encode(text);

        require(tokenizer.decode(ids) == text, "BPE round-trip failed");
        require(tokenizer.vocab_size() > bpe::BpeTokenizer::kByteVocabularySize,
                "Training did not create any merge");
        require(ids.size() < text.size(), "Special tokens or merges were not applied");
        require(ids.size() >= 3 && ids.front() == bpe::BpeTokenizer::kByteVocabularySize + 1 &&
                    ids.back() == bpe::BpeTokenizer::kByteVocabularySize + 2,
                "Special tokens were not encoded as token IDs");
        require(tokenizer.encode("").empty(), "Empty input did not produce empty output");
        for (std::size_t encoded = 0; encoded < 19'683; ++encoded) {
            std::size_t value = encoded;
            std::string sample(9, 'a');
            for (char& character : sample) {
                character = static_cast<char>('a' + value % 3);
                value /= 3;
            }
            require(
                tokenizer.encode(sample) ==
                    reference_encode_plain(sample, tokenizer.merges()),
                "Optimized encoder differs from sequential BPE semantics");
        }
        std::uint64_t random_state = 0x9E37'79B9'7F4A'7C15ULL;
        for (std::size_t trial = 0; trial < 10'000; ++trial) {
            random_state = random_state * 6'364'136'223'846'793'005ULL + 1;
            const std::size_t length = 1 + (random_state >> 32U) % 32;
            std::string sample(length, '\0');
            for (char& character : sample) {
                random_state = random_state * 6'364'136'223'846'793'005ULL + 1;
                character = static_cast<char>(random_state >> 56U);
            }
            require(
                tokenizer.encode(sample) ==
                    reference_encode_plain(sample, tokenizer.merges()),
                "Short stack encoder differs from sequential BPE semantics");
        }
        bpe::EncodeOptions uncached;
        uncached.cache_entries = 0;
        uncached.cache_max_input_bytes = 0;
        for (std::size_t trial = 0; trial < 2'000; ++trial) {
            random_state = random_state * 6'364'136'223'846'793'005ULL + 1;
            const std::size_t length = 33 + (random_state >> 32U) % 224;
            std::string sample(length, '\0');
            for (char& character : sample) {
                random_state = random_state * 6'364'136'223'846'793'005ULL + 1;
                character = static_cast<char>('a' + (random_state >> 62U));
            }
            require(
                tokenizer.encode(sample, uncached) ==
                    reference_encode_plain(sample, tokenizer.merges()),
                "Heap encoder differs from sequential BPE semantics");
        }
        std::string long_sample(8'192, '\0');
        for (char& character : long_sample) {
            random_state = random_state * 6'364'136'223'846'793'005ULL + 1;
            character = static_cast<char>('a' + (random_state >> 62U));
        }
        require(
            tokenizer.encode(long_sample, uncached) ==
                reference_encode_plain(long_sample, tokenizer.merges()),
            "Rank-bucket encoder differs from sequential BPE semantics");

        const std::vector<std::string_view> batch_input = {
            "Ala ma kota", "Zażółć gęślą jaźń", "<bos>abc<eos>", ""};
        const auto batch = tokenizer.encode_batch(batch_input, 4);
        require(batch.size() == batch_input.size(), "Batch encoder changed item count");
        for (std::size_t i = 0; i < batch.size(); ++i) {
            require(batch[i] == tokenizer.encode(batch_input[i]),
                    "Batch encoder differs from scalar encoding");
        }
        test_batch_flat_and_cache_options(tokenizer);
        test_low_memory_and_pretokenizer_modes();

        std::istringstream stream(corpus[0] + corpus[1]);
        bpe::TrainerConfig streaming_config = config;
        streaming_config.worker_count = 1;
        const bpe::BpeTokenizer streaming =
            bpe::BpeTokenizer::train_streaming(stream, streaming_config, 7);
        require(streaming.decode(streaming.encode(text)) == text,
                "Streaming tokenizer round-trip failed");
        require(streaming.vocab_size() > bpe::BpeTokenizer::kByteVocabularySize,
                "Streaming training did not create any merge");
        expect_throw([&] {
            std::istringstream invalid_stream("text");
            (void)bpe::BpeTokenizer::train_streaming(invalid_stream, config, 0);
        }, "Zero streaming block size was accepted");

        bpe::TrainerConfig overlapping_config;
        overlapping_config.vocab_size = 260;
        overlapping_config.special_tokens = {"a", "ab", "abc", "bc"};
        const bpe::BpeTokenizer overlapping =
            bpe::BpeTokenizer::train({"ordinary training text"}, overlapping_config);
        require(
            overlapping.encode("abc") ==
                std::vector<bpe::TokenId>{bpe::BpeTokenizer::kByteVocabularySize + 2},
            "Longest overlapping special token was not selected");

        bpe::TrainerConfig single_thread_config = config;
        single_thread_config.worker_count = 1;
        const bpe::BpeTokenizer single_thread =
            bpe::BpeTokenizer::train(corpus, single_thread_config);
        require(same_merges(tokenizer.merges(), single_thread.merges()),
                "Training is not deterministic across worker counts");

        const auto model_path = std::filesystem::temp_directory_path() /
                                "cgpt_bpe_tokenizer_test.bin";
        tokenizer.save(model_path, bpe::TokenizerFormat::Binary);
        const bpe::BpeTokenizer loaded = bpe::BpeTokenizer::load(model_path);
        require(same_merges(loaded.merges(), tokenizer.merges()), "Loaded merges differ from saved merges");
        require(loaded.special_tokens() == tokenizer.special_tokens(),
                "Loaded special tokens differ from saved tokens");
        require(loaded.encode(text) == ids, "Loaded tokenizer produced different IDs");
        std::filesystem::remove(model_path);

        const auto huggingface_path = std::filesystem::temp_directory_path() /
                                      "cgpt_bpe_tokenizer_test.json";
        tokenizer.save(huggingface_path);
        const bpe::BpeTokenizer huggingface =
            bpe::BpeTokenizer::load(huggingface_path);
        require(same_merges(huggingface.merges(), tokenizer.merges()),
                "Hugging Face loaded merges differ from saved merges");
        require(huggingface.special_tokens() == tokenizer.special_tokens(),
                "Hugging Face loaded special tokens differ from saved tokens");
        require(huggingface.encode(text) == ids,
                "Hugging Face loaded tokenizer produced different IDs");
        std::filesystem::remove(huggingface_path);

        expect_throw([] {
            bpe::TrainerConfig invalid;
            invalid.vocab_size = 259;
            invalid.special_tokens = {""};
            (void)bpe::BpeTokenizer::train({"text"}, invalid);
        }, "Empty special token was accepted");
        expect_throw([] {
            bpe::TrainerConfig invalid;
            invalid.vocab_size = 255;
            (void)bpe::BpeTokenizer::train({"text"}, invalid);
        }, "Too-small vocabulary was accepted");
        expect_throw([] {
            bpe::TrainerConfig invalid;
            invalid.min_pair_frequency = 0;
            (void)bpe::BpeTokenizer::train({"text"}, invalid);
        }, "Zero pair frequency was accepted");
        expect_throw([] {
            (void)bpe::BpeTokenizer::train({"", ""});
        }, "Empty corpus was accepted");
        expect_throw([&] {
            const std::vector<bpe::TokenId> invalid_ids = {
                static_cast<bpe::TokenId>(tokenizer.vocab_size())};
            (void)tokenizer.decode(invalid_ids);
        }, "Invalid token ID was accepted");
        expect_throw([] {
            (void)bpe::BpeTokenizer::load("file-that-does-not-exist.bin");
        }, "Missing model file was accepted");

        const auto bad_magic_path = std::filesystem::temp_directory_path() /
                                    "cgpt_bpe_bad_magic.bin";
        {
            std::ofstream output(bad_magic_path, std::ios::binary);
            output << "not a tokenizer";
        }
        expect_throw([&] { (void)bpe::BpeTokenizer::load(bad_magic_path); },
                     "Invalid model magic was accepted");
        std::filesystem::remove(bad_magic_path);

        const auto truncated_path = std::filesystem::temp_directory_path() /
                                    "cgpt_bpe_truncated.bin";
        tokenizer.save(truncated_path, bpe::TokenizerFormat::Binary);
        {
            std::ifstream input(truncated_path, std::ios::binary);
            std::string bytes((std::istreambuf_iterator<char>(input)), {});
            bytes.resize(bytes.size() - 1);
            std::ofstream output(truncated_path, std::ios::binary | std::ios::trunc);
            output.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
        }
        expect_throw([&] { (void)bpe::BpeTokenizer::load(truncated_path); },
                     "Truncated model was accepted");
        std::filesystem::remove(truncated_path);

        std::cout << "BPE tokenizer tests passed.\n";
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "BPE tokenizer test failed: " << error.what() << '\n';
        return 1;
    }
}
