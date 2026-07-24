#include "tokenizer/bpe_tokenizer.h"
#include "utils/progress_bar.h"

#include <algorithm>
#include <array>
#include <fstream>
#include <limits>
#include <queue>
#include <set>
#include <stdexcept>
#include <thread>
#include <unordered_map>

namespace bpe {
    namespace {
        using PairKey = std::uint64_t;
        using PairCounts = std::unordered_map<PairKey, std::uint64_t>;

        [[nodiscard]] PairKey pair_key(const TokenId left, const TokenId right) noexcept {
            return (static_cast<PairKey>(left) << 32U) | static_cast<PairKey>(right);
        }

        [[nodiscard]] TokenId pair_left(const PairKey key) noexcept {
            return static_cast<TokenId>(key >> 32U);
        }

        [[nodiscard]] TokenId pair_right(const PairKey key) noexcept {
            return static_cast<TokenId>(key & 0xFFFF'FFFFULL);
        }

        [[nodiscard]] std::size_t worker_count_for(
            const std::size_t requested,
            const std::size_t work_items
        ) {
            if (work_items == 0) {
                return 1;
            }

            const std::size_t hardware = std::max<std::size_t>(
                1, std::thread::hardware_concurrency()
            );
            const std::size_t selected = requested == 0 ? hardware : requested;
            return std::max<std::size_t>(1, std::min(selected, work_items));
        }

        template<typename Function>
        void parallel_for_ranges(
            const std::size_t item_count,
            const std::size_t workers,
            Function &&function) {
            if (workers <= 1) {
                function(0, 0, item_count);
                return;
            }

            std::vector<std::thread> threads;
            threads.reserve(workers);

            for (std::size_t worker = 0; worker < workers; ++worker) {
                const std::size_t begin = item_count * worker / workers;
                const std::size_t end = item_count * (worker + 1) / workers;
                threads.emplace_back(function, worker, begin, end);
            }

            for (auto &thread: threads) {
                thread.join();
            }
        }

        [[nodiscard]] std::vector<TokenId> bytes_to_tokens(const std::string_view text) {
            std::vector<TokenId> result;
            result.reserve(text.size());

            for (const unsigned char byte: text) {
                result.push_back(static_cast<TokenId>(byte));
            }

            return result;
        }

        void write_u32(std::ostream &output, const std::uint32_t value) {
            output.write(reinterpret_cast<const char *>(&value), sizeof(value));
        }

        [[nodiscard]] std::uint32_t read_u32(std::istream &input) {
            std::uint32_t value = 0;
            input.read(reinterpret_cast<char *>(&value), sizeof(value));
            if (!input) {
                throw std::runtime_error("Unexpected end of tokenizer file");
            }
            return value;
        }
    }

    BpeTokenizer::BpeTokenizer(std::vector<std::string> special_tokens)
        : special_tokens_(std::move(special_tokens)) {
        rebuild_token_bytes();
    }

    BpeTokenizer BpeTokenizer::train(
        const std::vector<std::string> &documents,
        const TrainerConfig &config
    ) {
        const std::size_t minimum_vocab =
                kByteVocabularySize + config.special_tokens.size();

        if (config.vocab_size < minimum_vocab) {
            throw std::invalid_argument("vocab_size is smaller than byte and special-token vocabulary");
        }
        if (config.min_pair_frequency == 0) {
            throw std::invalid_argument("min_pair_frequency must be at least one");
        }

        BpeTokenizer tokenizer(config.special_tokens);
        std::vector<std::vector<TokenId> > corpus;
        corpus.reserve(documents.size());

        for (const std::string &document: documents) {
            if (!document.empty()) {
                corpus.push_back(bytes_to_tokens(document));
            }
        }

        if (corpus.empty()) {
            throw std::invalid_argument("Cannot train a BPE tokenizer on an empty corpus");
        }

        const std::size_t workers = worker_count_for(config.worker_count, corpus.size());
        PairCounts global_counts;

        std::vector<PairCounts> initial_counts(workers);
        parallel_for_ranges(corpus.size(), workers,
                            [&corpus, &initial_counts](const std::size_t worker,
                                                       const std::size_t begin,
                                                       const std::size_t end) {
                                PairCounts &counts = initial_counts[worker];
                                for (std::size_t index = begin; index < end; ++index) {
                                    const auto &sequence = corpus[index];
                                    for (std::size_t i = 1; i < sequence.size(); ++i) {
                                        ++counts[pair_key(sequence[i - 1], sequence[i])];
                                    }
                                }
                            });

        for (const PairCounts &counts: initial_counts) {
            for (const auto &[pair, count]: counts) {
                global_counts[pair] += count;
            }
        }

        // Keep the best candidates in a heap. Scanning global_counts for every
        // merge is needlessly expensive when the vocabulary is large.
        struct CandidateCompare {
            bool operator()(const std::pair<std::uint64_t, PairKey> &left,
                            const std::pair<std::uint64_t, PairKey> &right) const noexcept {
                if (left.first != right.first) {
                    return left.first < right.first;
                }
                // For equal frequencies, prefer the smaller pair key.
                return left.second > right.second;
            }
        };
        std::priority_queue<
            std::pair<std::uint64_t, PairKey>,
            std::vector<std::pair<std::uint64_t, PairKey>>,
            CandidateCompare> candidates;

        for (const auto &[pair, count]: global_counts) {
            candidates.emplace(count, pair);
        }

        const std::size_t requested_merges = config.vocab_size - tokenizer.vocab_size();
        ProgressBar progress(requested_merges, "BPE training");

        while (tokenizer.vocab_size() < config.vocab_size) {
            PairKey best_pair = 0;
            std::uint64_t best_count = 0;

            // Heap entries are immutable snapshots. Discard stale snapshots.
            while (!candidates.empty()) {
                const auto [count, pair] = candidates.top();
                candidates.pop();
                if (global_counts.contains(pair) && global_counts[pair] == count) {
                    best_pair = pair;
                    best_count = count;
                    break;
                }
            }

            if (best_count < config.min_pair_frequency) {
                break;
            }

            const TokenId left = pair_left(best_pair);
            const TokenId right = pair_right(best_pair);
            const auto merged = static_cast<TokenId>(tokenizer.vocab_size());
            tokenizer.add_merge(left, right);

            // Documents are independent. Update pair frequencies incrementally:
            // only the pairs touching a replacement can have changed.
            using PairDelta = std::unordered_map<PairKey, std::int64_t>;
            std::vector<PairDelta> local_deltas(workers);
            parallel_for_ranges(corpus.size(), workers,
                                [&corpus, &local_deltas, left, right, merged](const std::size_t worker,
                                                               const std::size_t begin,
                                                               const std::size_t end) {
                                    PairDelta &delta = local_deltas[worker];
                                    for (std::size_t index = begin; index < end; ++index) {
                                        const auto &source = corpus[index];
                                        std::vector<TokenId> replaced;
                                        bool changed = false;

                                        for (std::size_t i = 0; i < source.size();) {
                                            if (i + 1 < source.size() &&
                                                source[i] == left && source[i + 1] == right) {
                                                if (!changed) {
                                                    replaced.reserve(source.size());
                                                    replaced.insert(replaced.end(), source.begin(),
                                                                    source.begin() + static_cast<std::ptrdiff_t>(i));
                                                    changed = true;
                                                }
                                                const bool follows_merge =
                                                    i >= 2 && source[i - 2] == left &&
                                                    source[i - 1] == right;
                                                if (i > 0 && !follows_merge) {
                                                    --delta[pair_key(source[i - 1], source[i])];
                                                }
                                                --delta[pair_key(left, right)];
                                                if (i + 2 < source.size()) {
                                                    --delta[pair_key(right, source[i + 2])];
                                                }

                                                if (!replaced.empty()) {
                                                    ++delta[pair_key(replaced.back(), merged)];
                                                }
                                                if (i + 2 < source.size()) {
                                                    ++delta[pair_key(merged, source[i + 2])];
                                                }
                                                replaced.push_back(merged);
                                                i += 2;
                                            } else {
                                                if (changed) {
                                                    replaced.push_back(source[i]);
                                                }
                                                ++i;
                                            }
                                        }

                                        if (changed) {
                                            corpus[index] = std::move(replaced);
                                        }
                                    }
                                }
            );

            for (const PairDelta &delta: local_deltas) {
                for (const auto &[pair, change]: delta) {
                    if (change < 0) {
                        global_counts[pair] -= static_cast<std::uint64_t>(-change);
                    } else {
                        global_counts[pair] += static_cast<std::uint64_t>(change);
                    }
                    candidates.emplace(global_counts[pair], pair);
                }
            }

            progress.update(tokenizer.vocab_size() - kByteVocabularySize -
                            config.special_tokens.size());
        }
        progress.finish();
        return tokenizer;
    }

    std::vector<TokenId> BpeTokenizer::encode(const std::string_view text) const {
        if (text.empty()) {
            return {};
        }

        // Keep the token stream as an intrusive doubly-linked list. A merge
        // only changes the two pairs touching the merged node, so encoding no
        // longer allocates and rescans the complete stream for every rule.
        std::vector<TokenId> tokens = bytes_to_tokens(text);
        const std::size_t size = tokens.size();
        std::vector<std::size_t> next(size);
        std::vector<std::size_t> previous(size);
        constexpr std::size_t end = std::numeric_limits<std::size_t>::max();

        for (std::size_t i = 0; i < size; ++i) {
            next[i] = i + 1 < size ? i + 1 : end;
            previous[i] = i == 0 ? end : i - 1;
        }

        using Positions = std::set<std::size_t>;
        std::unordered_map<PairKey, Positions> positions;
        positions.reserve(size + merges_.size() * 2 + 1);

        for (std::size_t i = 0; i + 1 < size; ++i) {
            positions[pair_key(tokens[i], tokens[i + 1])].insert(i);
        }

        const auto remove_pair = [&positions, &tokens, &next](const std::size_t start) {
            if (start == end || next[start] == end) {
                return;
            }
            const PairKey key = pair_key(tokens[start], tokens[next[start]]);
            auto found = positions.find(key);
            if (found != positions.end()) {
                found->second.erase(start);
            }
        };

        const auto add_pair = [&positions, &tokens, &next](const std::size_t start) {
            if (start != end && next[start] != end) {
                positions[pair_key(tokens[start], tokens[next[start]])].insert(start);
            }
        };

        for (const MergeRule& rule : merges_) {
            const PairKey key = pair_key(rule.left, rule.right);
            auto found = positions.find(key);
            if (found == positions.end()) {
                continue;
            }

            // The set is ordered, matching the original left-to-right scan.
            // Snapshot the positions because updating neighbouring pairs can
            // insert into the unordered map and invalidate its iterators.
            const std::vector<std::size_t> candidates(
                found->second.begin(), found->second.end());
            for (const std::size_t left : candidates) {
                const std::size_t right = next[left];
                if (right == end || tokens[left] != rule.left ||
                    tokens[right] != rule.right) {
                    continue;
                }

                const std::size_t before = previous[left];
                const std::size_t after = next[right];
                remove_pair(before);
                remove_pair(left);
                remove_pair(right);

                tokens[left] = rule.merged;
                next[left] = after;
                if (after != end) {
                    previous[after] = left;
                }
                if (before != end) {
                    next[before] = left;
                }
                // Mark the consumed node detached so a stale candidate from
                // the snapshot cannot be processed a second time.
                next[right] = end;
                previous[right] = end;
                add_pair(before);
                add_pair(left);
            }
        }

        std::vector<TokenId> result;
        result.reserve(size);
        for (std::size_t index = 0; index != end; index = next[index]) {
            result.push_back(tokens[index]);
        }
        return result;
    }

std::string BpeTokenizer::decode(const std::span<const TokenId> ids) const {
    std::size_t decoded_size = 0;
    for (const TokenId id : ids) {
        if (is_special(id)) {
            decoded_size += special_tokens_[id - kByteVocabularySize].size();
        } else {
            if (id >= token_bytes_.size()) {
                throw std::out_of_range("Token ID is outside this tokenizer vocabulary");
            }
            decoded_size += token_bytes_[id].size();
        }
    }

    std::string text;
    text.reserve(decoded_size);

    for (const TokenId id : ids) {
        if (is_special(id)) {
            text += special_tokens_[id - kByteVocabularySize];
            continue;
        }

        if (id >= token_bytes_.size()) {
            throw std::out_of_range("Token ID is outside this tokenizer vocabulary");
        }

        const auto& bytes = token_bytes_[id];
        text.append(
            reinterpret_cast<const char*>(bytes.data()),
            bytes.size()
        );
    }

    return text;
}

void BpeTokenizer::save(const std::filesystem::path& path) const {
    std::ofstream output(path, std::ios::binary);
    if (!output) {
        throw std::runtime_error("Cannot open tokenizer output file");
    }

    constexpr std::array<char, 8> magic = {'B', 'P', 'E', 'T', 'O', 'K', '1', '\0'};
    output.write(magic.data(), static_cast<std::streamsize>(magic.size()));
    write_u32(output, static_cast<std::uint32_t>(special_tokens_.size()));
    write_u32(output, static_cast<std::uint32_t>(merges_.size()));

    for (const std::string& token : special_tokens_) {
        write_u32(output, static_cast<std::uint32_t>(token.size()));
        output.write(token.data(), static_cast<std::streamsize>(token.size()));
    }

    for (const MergeRule& merge : merges_) {
        write_u32(output, merge.left);
        write_u32(output, merge.right);
        write_u32(output, merge.merged);
    }

    if (!output) {
        throw std::runtime_error("Cannot write tokenizer file");
    }
}

BpeTokenizer BpeTokenizer::load(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw std::runtime_error("Cannot open tokenizer model file");
    }

    constexpr std::array<char, 8> expected_magic = {'B', 'P', 'E', 'T', 'O', 'K', '1', '\0'};
    std::array<char, 8> actual_magic{};
    input.read(actual_magic.data(), static_cast<std::streamsize>(actual_magic.size()));

    if (!input || actual_magic != expected_magic) {
        throw std::runtime_error("Invalid tokenizer file magic");
    }

    const std::uint32_t special_count = read_u32(input);
    const std::uint32_t merge_count = read_u32(input);

    if (special_count > 1'000'000 || merge_count > 10'000'000) {
        throw std::runtime_error("Tokenizer file declares unreasonable sizes");
    }

    std::vector<std::string> special_tokens;
    special_tokens.reserve(special_count);

    for (std::uint32_t i = 0; i < special_count; ++i) {
        const std::uint32_t size = read_u32(input);
        if (size > 1'000'000) {
            throw std::runtime_error("Tokenizer special token is too large");
        }

        std::string token(size, '\0');
        input.read(token.data(), static_cast<std::streamsize>(size));
        if (!input) {
            throw std::runtime_error("Unexpected end of tokenizer special token");
        }
        special_tokens.push_back(std::move(token));
    }

    BpeTokenizer tokenizer(std::move(special_tokens));
    for (std::uint32_t i = 0; i < merge_count; ++i) {
        const TokenId left = read_u32(input);
        const TokenId right = read_u32(input);
        const TokenId merged = read_u32(input);

        if (merged != tokenizer.vocab_size()) {
            throw std::runtime_error("Tokenizer merge IDs are not contiguous");
        }
        if (left >= merged || right >= merged) {
            throw std::runtime_error("Tokenizer merge references a future token");
        }

        tokenizer.add_merge(left, right);
    }

    return tokenizer;
}

std::size_t BpeTokenizer::vocab_size() const noexcept {
    return token_bytes_.size();
}

const std::vector<MergeRule>& BpeTokenizer::merges() const noexcept {
    return merges_;
}

const std::vector<std::string>& BpeTokenizer::special_tokens() const noexcept {
    return special_tokens_;
}

void BpeTokenizer::add_merge(const TokenId left, const TokenId right) {
    const auto merged = static_cast<TokenId>(token_bytes_.size());

    if (left >= merged || right >= merged) {
        throw std::invalid_argument("BPE merge references an unknown token");
    }

    std::vector<std::uint8_t> bytes = token_bytes_[left];
    bytes.insert(bytes.end(), token_bytes_[right].begin(), token_bytes_[right].end());

    merges_.push_back({left, right, merged});
    token_bytes_.push_back(std::move(bytes));
}

void BpeTokenizer::rebuild_token_bytes() {
    token_bytes_.clear();
    token_bytes_.reserve(kByteVocabularySize + special_tokens_.size() + merges_.size());

    for (TokenId byte = 0; byte < kByteVocabularySize; ++byte) {
        token_bytes_.push_back({static_cast<std::uint8_t>(byte)});
    }

    // Special token IDs deliberately do not have byte payloads.
    token_bytes_.resize(kByteVocabularySize + special_tokens_.size());
}

bool BpeTokenizer::is_special(const TokenId id) const noexcept {
    return id >= kByteVocabularySize &&
           id < kByteVocabularySize + special_tokens_.size();
}
}
