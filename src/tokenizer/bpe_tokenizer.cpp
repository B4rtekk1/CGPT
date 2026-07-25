#include "tokenizer/bpe_tokenizer.h"
#include "utils/progress_bar.h"

#include <algorithm>
#include <array>
#include <fstream>
#include <limits>
#include <queue>
#include <stdexcept>
#include <thread>
#include <unordered_map>

namespace bpe {
    namespace {
        using PairKey = std::uint64_t;
        using PairCounts = std::unordered_map<PairKey, std::uint64_t>;
        using Occurrence = std::uint64_t;

        [[nodiscard]] Occurrence occurrence_key(
            const std::size_t document, const std::size_t position) {
            if (document > 0xFFFF'FFFFULL || position > 0xFFFF'FFFFULL) {
                throw std::length_error("BPE corpus document or position exceeds 32-bit limit");
            }
            return (static_cast<Occurrence>(document) << 32U) |
                   static_cast<Occurrence>(position);
        }

        [[nodiscard]] std::size_t occurrence_document(const Occurrence value) noexcept {
            return static_cast<std::size_t>(value >> 32U);
        }

        [[nodiscard]] std::size_t occurrence_position(const Occurrence value) noexcept {
            return static_cast<std::size_t>(value & 0xFFFF'FFFFULL);
        }

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
        for (const std::string &token : special_tokens_) {
            if (token.empty()) {
                throw std::invalid_argument("special tokens must not be empty");
            }
        }
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
        if (config.vocab_size > std::numeric_limits<TokenId>::max()) {
            throw std::invalid_argument("vocab_size exceeds the TokenId limit");
        }
        for (const std::string &token : config.special_tokens) {
            if (token.empty()) {
                throw std::invalid_argument("special tokens must not be empty");
            }
        }

        const std::size_t requested_merges = config.vocab_size - minimum_vocab;
        ProgressBar progress(requested_merges, "BPE training");

        BpeTokenizer tokenizer(config.special_tokens);
        tokenizer.merges_.reserve(requested_merges);
        tokenizer.token_bytes_.reserve(config.vocab_size);
        tokenizer.merge_lookup_.reserve(requested_merges);
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
        std::unordered_map<PairKey, std::vector<Occurrence>> occurrences;
        occurrences.reserve(4096);
        std::size_t active_edges = 0;
        std::size_t occurrence_entries = 0;

        using Link = std::uint32_t;
        constexpr Link end = std::numeric_limits<Link>::max();
        std::vector<std::vector<Link>> next_positions(corpus.size());
        std::vector<std::vector<Link>> previous_positions(corpus.size());
        for (std::size_t document = 0; document < corpus.size(); ++document) {
            const std::size_t size = corpus[document].size();
            if (size >= end) {
                throw std::length_error("BPE document is too large for the 32-bit occurrence index");
            }
            active_edges += size > 0 ? size - 1 : 0;
            next_positions[document].resize(size);
            previous_positions[document].resize(size);
            for (std::size_t i = 0; i < size; ++i) {
                next_positions[document][i] = i + 1 < size ? static_cast<Link>(i + 1) : end;
                previous_positions[document][i] = i == 0 ? end : static_cast<Link>(i - 1);
            }
        }

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

        for (std::size_t document = 0; document < corpus.size(); ++document) {
            const auto &sequence = corpus[document];
            for (std::size_t i = 1; i < sequence.size(); ++i) {
                occurrences[pair_key(sequence[i - 1], sequence[i])].push_back(
                    occurrence_key(document, i - 1));
                ++occurrence_entries;
            }
        }

        struct CandidateCompare {
            bool operator()(const std::pair<std::uint64_t, PairKey> &left,
                            const std::pair<std::uint64_t, PairKey> &right) const noexcept {
                if (left.first != right.first) {
                    return left.first < right.first;
                }
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

        while (tokenizer.vocab_size() < config.vocab_size) {
            PairKey best_pair = 0;
            std::uint64_t best_count = 0;

            while (!candidates.empty()) {
                const auto [count, pair] = candidates.top();
                candidates.pop();
                const auto current = global_counts.find(pair);
                if (current != global_counts.end() && current->second == count) {
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

            if (auto found = occurrences.find(best_pair); found != occurrences.end()) {
                auto positions = std::move(found->second);
                occurrence_entries -= positions.size();
                std::ranges::sort(positions.begin(), positions.end());

                for (const Occurrence occurrence: positions) {
                    const std::size_t document = occurrence_document(occurrence);
                    const std::size_t position = occurrence_position(occurrence);
                    auto &sequence = corpus[document];
                    auto &next = next_positions[document];
                    auto &previous = previous_positions[document];

                    if (position >= sequence.size() || sequence[position] != left ||
                        next[position] == end || sequence[next[position]] != right) {
                        continue;
                    }

                    const std::size_t right_position = next[position];
                    const std::size_t before = previous[position];
                    const std::size_t after = next[right_position];

                    const auto remove_pair = [&](const std::size_t start) {
                        if (start != end && next[start] != end) {
                            const PairKey pair = pair_key(sequence[start], sequence[next[start]]);
                            const auto count = --global_counts[pair];
                            if (count != 0) {
                                candidates.emplace(count, pair);
                            }
                        }
                    };
                    const auto add_pair = [&](const std::size_t start) {
                        if (start != end && next[start] != end) {
                            const PairKey pair = pair_key(sequence[start], sequence[next[start]]);
                            const auto count = ++global_counts[pair];
                            occurrences[pair].push_back(occurrence_key(document, start));
                            ++occurrence_entries;
                            candidates.emplace(count, pair);
                        }
                    };

                    remove_pair(before);
                    remove_pair(position);
                    remove_pair(right_position);

                    sequence[position] = merged;
                    next[position] = after;
                    if (after != end) {
                        previous[after] = position;
                    }
                    if (before != end) {
                        next[before] = position;
                    }
                    next[right_position] = end;
                    previous[right_position] = end;

                    add_pair(before);
                    add_pair(position);
                    --active_edges;
                }

                constexpr std::size_t compact_slack = 1'000'000;
                if (occurrence_entries > active_edges + compact_slack) {
                    occurrences.clear();
                    occurrences.reserve(global_counts.size());
                    occurrence_entries = 0;
                    for (std::size_t document = 0; document < corpus.size(); ++document) {
                        const auto &sequence = corpus[document];
                        const auto &next = next_positions[document];
                        for (std::size_t start = 0; start < sequence.size(); ++start) {
                            if (next[start] != end) {
                                occurrences[pair_key(sequence[start], sequence[next[start]])].push_back(
                                    occurrence_key(document, start));
                                ++occurrence_entries;
                            }
                        }
                    }
                }
            }

            progress.update(tokenizer.vocab_size() - kByteVocabularySize -
                            config.special_tokens.size());
        }
        progress.finish();
        return tokenizer;
    }

    BpeTokenizer BpeTokenizer::train_streaming(
        std::istream &input,
        const TrainerConfig &config,
        const std::size_t block_size
    ) {
        if (block_size == 0) {
            throw std::invalid_argument("streaming training block_size must be at least one");
        }
        if (block_size > static_cast<std::size_t>(std::numeric_limits<std::streamsize>::max())) {
            throw std::invalid_argument("streaming training block_size exceeds stream limits");
        }

        std::vector<std::string> documents;
        std::string block(block_size, '\0');
        while (input.read(block.data(), static_cast<std::streamsize>(block.size())) ||
               input.gcount() != 0) {
            const auto bytes_read = input.gcount();
            documents.emplace_back(block.data(), static_cast<std::size_t>(bytes_read));
        }

        if (input.fail() && !input.eof()) {
            throw std::runtime_error("Failed while reading tokenizer training stream");
        }
        return train(documents, config);
    }

    BpeTokenizer BpeTokenizer::train(
        std::istream &input,
        const TrainerConfig &config,
        const std::size_t block_size
    ) {
        return train_streaming(input, config, block_size);
    }

    std::vector<TokenId> BpeTokenizer::encode(const std::string_view text) const {
        if (text.empty()) {
            return {};
        }

        std::vector<TokenId> result;
        std::vector<std::size_t> next_special(special_tokens_.size());
        for (std::size_t index = 0; index < special_tokens_.size(); ++index) {
            next_special[index] = text.find(special_tokens_[index]);
        }

        std::size_t cursor = 0;
        while (cursor < text.size()) {
            std::size_t matched_position = std::string_view::npos;
            std::size_t matched_size = 0;
            std::size_t matched_index = 0;
            for (std::size_t index = 0; index < special_tokens_.size(); ++index) {
                if (next_special[index] < cursor) {
                    next_special[index] =
                        text.find(special_tokens_[index], cursor);
                }
                if (next_special[index] < matched_position ||
                    (next_special[index] == matched_position &&
                     special_tokens_[index].size() > matched_size)) {
                    matched_position = next_special[index];
                    matched_size = special_tokens_[index].size();
                    matched_index = index;
                }
            }

            if (matched_position == std::string_view::npos) {
                append_encoded_bytes(text.substr(cursor), result);
                break;
            }

            append_encoded_bytes(
                text.substr(cursor, matched_position - cursor),
                result);
            result.push_back(kByteVocabularySize + static_cast<TokenId>(matched_index));
            cursor = matched_position + matched_size;
        }
        return result;
    }

    void BpeTokenizer::append_encoded_bytes(
        const std::string_view text,
        std::vector<TokenId>& output
    ) const {
        if (text.empty()) {
            return;
        }

        const auto encode_with_links = [&]<typename Link>() {
            constexpr Link end = std::numeric_limits<Link>::max();
            struct Node {
            TokenId token;
            Link next;
            Link previous;
            };
            std::vector<Node> nodes;
            nodes.reserve(text.size());
            for (std::size_t index = 0; index < text.size(); ++index) {
                nodes.push_back({
                    static_cast<TokenId>(
                        static_cast<unsigned char>(text[index])),
                    index + 1 < text.size() ? static_cast<Link>(index + 1) : end,
                    index == 0 ? end : static_cast<Link>(index - 1)
                });
            }

            const auto first_merged = static_cast<TokenId>(
                kByteVocabularySize + special_tokens_.size());
            std::vector<std::vector<Link>> positions(merges_.size());

            const auto add_candidate = [&](const Link left) {
                if (left == end || nodes[left].next == end) {
                    return;
                }
                const auto merge = merge_lookup_.find(
                    pair_key(nodes[left].token, nodes[nodes[left].next].token));
                if (merge != merge_lookup_.end()) {
                    positions[merge->second - first_merged].push_back(left);
                }
            };

            for (Link left = 0; left + 1 < nodes.size(); ++left) {
                add_candidate(left);
            }

            std::size_t token_count = nodes.size();
            for (std::size_t rank = 0; rank < merges_.size(); ++rank) {
                const MergeRule& rule = merges_[rank];
                // A pair can first appear only while its newest operand is being
                // created. Since every earlier rule is applied left-to-right,
                // these append-only position lists are already ordered.
                for (const Link left_index : positions[rank]) {
                    Node& left = nodes[left_index];
                    if (left.next == end) {
                        continue;
                    }
                    const Link right_index = left.next;
                    Node& right = nodes[right_index];
                    if (left.token != rule.left || right.token != rule.right) {
                        continue;
                    }

                    const Link before = left.previous;
                    const Link after = right.next;
                    left.token = rule.merged;
                    left.next = after;
                    if (after != end) {
                        nodes[after].previous = left_index;
                    }
                    right.next = end;
                    right.previous = end;
                    --token_count;

                    add_candidate(before);
                    add_candidate(left_index);
                }
            }

            output.reserve(output.size() + token_count);
            for (Link index = 0; index != end; index = nodes[index].next) {
                output.push_back(nodes[index].token);
            }
        };

        if (text.size() < std::numeric_limits<std::uint32_t>::max()) {
            encode_with_links.template operator()<std::uint32_t>();
        } else {
            encode_with_links.template operator()<std::size_t>();
        }
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
    if (special_tokens_.size() > std::numeric_limits<std::uint32_t>::max() ||
        merges_.size() > std::numeric_limits<std::uint32_t>::max()) {
        throw std::length_error("Tokenizer is too large to serialize");
    }
    output.write(magic.data(), static_cast<std::streamsize>(magic.size()));
    write_u32(output, static_cast<std::uint32_t>(special_tokens_.size()));
    write_u32(output, static_cast<std::uint32_t>(merges_.size()));

    for (const std::string& token : special_tokens_) {
        if (token.size() > std::numeric_limits<std::uint32_t>::max()) {
            throw std::length_error("Tokenizer special token is too large to serialize");
        }
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
    if (static_cast<std::uint64_t>(kByteVocabularySize) + special_count + merge_count >
        std::numeric_limits<TokenId>::max()) {
        throw std::runtime_error("Tokenizer file vocabulary exceeds TokenId limit");
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
    tokenizer.merges_.reserve(merge_count);
    tokenizer.token_bytes_.reserve(
        kByteVocabularySize + special_count + merge_count);
    tokenizer.merge_lookup_.reserve(merge_count);
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

    char trailing = 0;
    input.read(&trailing, 1);
    if (input.gcount() != 0) {
        throw std::runtime_error("Tokenizer file contains trailing data");
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
    const PairKey key = pair_key(left, right);
    if (merge_lookup_.contains(key)) {
        throw std::invalid_argument("BPE merge pair is duplicated");
    }

    std::vector<std::uint8_t> bytes = token_bytes_[left];
    bytes.insert(bytes.end(), token_bytes_[right].begin(), token_bytes_[right].end());

    merges_.push_back({left, right, merged});
    token_bytes_.push_back(std::move(bytes));
    merge_lookup_.emplace(key, merged);
}

void BpeTokenizer::rebuild_token_bytes() {
    token_bytes_.clear();
    token_bytes_.reserve(kByteVocabularySize + special_tokens_.size() + merges_.size());

    for (TokenId byte = 0; byte < kByteVocabularySize; ++byte) {
        token_bytes_.push_back({static_cast<std::uint8_t>(byte)});
    }

    // Special token IDs deliberately do not have byte payloads.
    token_bytes_.resize(kByteVocabularySize + special_tokens_.size());
    merge_lookup_.clear();
}

bool BpeTokenizer::is_special(const TokenId id) const noexcept {
    return id >= kByteVocabularySize &&
           id < kByteVocabularySize + special_tokens_.size();
}
}
