#include "tokenizer/bpe_tokenizer.h"
#include "utils/progress_bar.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <fstream>
#include <limits>
#include <queue>
#include <stdexcept>
#include <thread>
#include <unordered_map>
#include <utility>

namespace bpe {
    namespace {
        using PairKey = std::uint64_t;
        using PairCounts = std::unordered_map<PairKey, std::uint64_t>;
        using Occurrence = std::uint64_t;

        [[nodiscard]] std::uint64_t next_cache_identity() noexcept {
            static std::atomic_uint64_t next{1};
            return next.fetch_add(1, std::memory_order_relaxed);
        }

        class CandidateHeap {
        public:
            void reserve(const std::size_t count) {
                entries_.reserve(count);
                indices_.reserve(count);
            }

            void insert(const PairKey pair, const std::uint64_t count) {
                indices_.emplace(pair, entries_.size());
                entries_.push_back({count, pair});
                sift_up(entries_.size() - 1);
            }

            void set(const PairKey pair, const std::uint64_t count) {
                const auto found = indices_.find(pair);
                if (found == indices_.end()) {
                    insert(pair, count);
                    return;
                }

                const std::size_t index = found->second;
                entries_[index].count = count;
                if (index != 0 && better(entries_[index], entries_[parent(index)])) {
                    sift_up(index);
                } else {
                    sift_down(index);
                }
            }

            void erase(const PairKey pair) {
                const auto found = indices_.find(pair);
                if (found == indices_.end()) {
                    return;
                }

                const std::size_t index = found->second;
                const std::size_t last = entries_.size() - 1;
                indices_.erase(found);
                if (index == last) {
                    entries_.pop_back();
                    return;
                }

                entries_[index] = entries_.back();
                entries_.pop_back();
                indices_[entries_[index].pair] = index;
                if (index != 0 && better(entries_[index], entries_[parent(index)])) {
                    sift_up(index);
                } else {
                    sift_down(index);
                }
            }

            [[nodiscard]] bool empty() const noexcept { return entries_.empty(); }
            [[nodiscard]] PairKey top_pair() const noexcept { return entries_.front().pair; }
            [[nodiscard]] std::uint64_t top_count() const noexcept { return entries_.front().count; }

        private:
            struct Entry {
                std::uint64_t count;
                PairKey pair;
            };

            [[nodiscard]] static bool better(const Entry &left, const Entry &right) noexcept {
                return left.count != right.count ? left.count > right.count : left.pair < right.pair;
            }
            [[nodiscard]] static std::size_t parent(const std::size_t index) noexcept {
                return (index - 1) / 2;
            }

            void swap_entries(const std::size_t left, const std::size_t right) {
                std::swap(entries_[left], entries_[right]);
                indices_[entries_[left].pair] = left;
                indices_[entries_[right].pair] = right;
            }
            void sift_up(std::size_t index) {
                while (index != 0 && better(entries_[index], entries_[parent(index)])) {
                    const std::size_t next = parent(index);
                    swap_entries(index, next);
                    index = next;
                }
            }
            void sift_down(std::size_t index) {
                while (true) {
                    const std::size_t left = index * 2 + 1;
                    if (left >= entries_.size()) {
                        return;
                    }
                    const std::size_t right = left + 1;
                    std::size_t best = left;
                    if (right < entries_.size() && better(entries_[right], entries_[left])) {
                        best = right;
                    }
                    if (!better(entries_[best], entries_[index])) {
                        return;
                    }
                    swap_entries(index, best);
                    index = best;
                }
            }

            std::vector<Entry> entries_;
            std::unordered_map<PairKey, std::size_t> indices_;
        };

        struct OccurrenceList {
            std::vector<Occurrence> values;
            std::size_t sorted_size = 0;

            void append(const Occurrence occurrence) {
                values.push_back(occurrence);
            }

            void mark_sorted() noexcept {
                sorted_size = values.size();
            }

            void sort() {
                if (sorted_size == values.size()) {
                    return;
                }

                auto middle = values.begin() + static_cast<std::ptrdiff_t>(sorted_size);
                std::ranges::sort(middle, values.end());
                if (sorted_size != 0) {
                    std::inplace_merge(values.begin(), middle, values.end());
                }
                mark_sorted();
            }
        };

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

        enum class ByteClass : std::uint8_t {
            Whitespace,
            Letter,
            Digit,
            Punctuation
        };

        [[nodiscard]] constexpr std::array<ByteClass, 256> make_byte_classes() noexcept {
            std::array<ByteClass, 256> classes{};
            classes.fill(ByteClass::Punctuation);

            for (std::size_t byte = 0x80; byte < 256; ++byte) {
                classes[byte] = ByteClass::Letter;
            }
            for (unsigned char byte = 'A'; byte <= 'Z'; ++byte) {
                classes[byte] = ByteClass::Letter;
            }
            for (unsigned char byte = 'a'; byte <= 'z'; ++byte) {
                classes[byte] = ByteClass::Letter;
            }
            classes[static_cast<unsigned char>('_')] = ByteClass::Letter;
            for (unsigned char byte = '0'; byte <= '9'; ++byte) {
                classes[byte] = ByteClass::Digit;
            }
            classes[static_cast<unsigned char>(' ')] = ByteClass::Whitespace;
            classes[static_cast<unsigned char>('\t')] = ByteClass::Whitespace;
            classes[static_cast<unsigned char>('\n')] = ByteClass::Whitespace;
            classes[static_cast<unsigned char>('\r')] = ByteClass::Whitespace;
            classes[static_cast<unsigned char>('\f')] = ByteClass::Whitespace;
            classes[static_cast<unsigned char>('\v')] = ByteClass::Whitespace;
            return classes;
        }

        alignas(64) constexpr auto kByteClasses = make_byte_classes();

        [[nodiscard]] inline ByteClass classify_byte(const unsigned char byte) noexcept {
            return kByteClasses[byte];
        }

        template<typename Function>
        void for_each_pretoken(
            const std::string_view text,
            const PretokenizerMode mode,
            Function&& function) {
            if (text.empty()) {
                return;
            }
            if (mode == PretokenizerMode::None) {
                function(text);
                return;
            }

            std::size_t position = 0;
            while (position < text.size()) {
                std::size_t begin = position;
                unsigned char byte = static_cast<unsigned char>(text[position]);
                ByteClass kind = classify_byte(byte);

                // GPT-style patterns usually attach one ordinary space to the
                // following word/number/punctuation piece. This preserves the
                // useful distinction between "word" and " word" tokens.
                if (byte == ' ' && position + 1 < text.size() &&
                    classify_byte(static_cast<unsigned char>(text[position + 1])) !=
                        ByteClass::Whitespace) {
                    ++position;
                    kind = classify_byte(static_cast<unsigned char>(text[position]));
                }

                if (kind == ByteClass::Digit) {
                    std::size_t digits = 0;
                    while (position < text.size() && digits < 3 &&
                           classify_byte(static_cast<unsigned char>(text[position])) ==
                               ByteClass::Digit) {
                        ++position;
                        ++digits;
                    }
                } else {
                    while (position < text.size() &&
                           classify_byte(static_cast<unsigned char>(text[position])) == kind) {
                        ++position;
                    }
                }

                function(text.substr(begin, position - begin));
            }
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

    BpeTokenizer::BpeTokenizer(
        std::vector<std::string> special_tokens,
        const PretokenizerMode pretokenizer)
        : pretokenizer_mode_(pretokenizer),
          special_tokens_(std::move(special_tokens)),
          cache_identity_(next_cache_identity()) {
        for (const std::string &token : special_tokens_) {
            if (token.empty()) {
                throw std::invalid_argument("special tokens must not be empty");
            }
        }
        rebuild_special_matcher();
        rebuild_token_bytes();
    }

    void BpeTokenizer::rebuild_special_matcher() {
        special_matcher_.clear();
        special_matcher_.emplace_back();

        for (std::size_t token_index = 0; token_index < special_tokens_.size(); ++token_index) {
            std::uint32_t state = 0;
            for (const unsigned char byte : special_tokens_[token_index]) {
                std::uint32_t transition = special_matcher_[state].transitions[byte];
                if (transition == SpecialMatcherNode::kMissing) {
                    transition = static_cast<std::uint32_t>(special_matcher_.size());
                    special_matcher_[state].transitions[byte] = transition;
                    special_matcher_.emplace_back();
                }
                state = transition;
            }
            special_matcher_[state].outputs.push_back(
                static_cast<std::uint32_t>(token_index));
        }

        std::queue<std::uint32_t> pending;
        for (std::size_t byte = 0; byte < 256; ++byte) {
            std::uint32_t& next = special_matcher_[0].transitions[byte];
            if (next != SpecialMatcherNode::kMissing) {
                pending.push(next);
            } else {
                next = 0;
            }
        }

        while (!pending.empty()) {
            const std::uint32_t state = pending.front();
            pending.pop();

            for (std::size_t byte = 0; byte < 256; ++byte) {
                std::uint32_t& next = special_matcher_[state].transitions[byte];
                if (next == SpecialMatcherNode::kMissing) {
                    next = special_matcher_[special_matcher_[state].failure]
                               .transitions[byte];
                    continue;
                }

                special_matcher_[next].failure =
                    special_matcher_[special_matcher_[state].failure]
                        .transitions[byte];

                const auto& inherited =
                    special_matcher_[special_matcher_[next].failure].outputs;
                special_matcher_[next].outputs.insert(
                    special_matcher_[next].outputs.end(),
                    inherited.begin(), inherited.end());
                pending.push(next);
            }
        }
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

        BpeTokenizer tokenizer(config.special_tokens, config.pretokenizer);
        tokenizer.merges_.reserve(requested_merges);
        tokenizer.token_bytes_.reserve(config.vocab_size);
        tokenizer.merge_lookup_.reserve(requested_merges);
        std::vector<std::vector<TokenId> > corpus;
        corpus.reserve(documents.size());

        for (const std::string &document: documents) {
            for_each_pretoken(document, config.pretokenizer,
                [&corpus](const std::string_view piece) {
                    if (!piece.empty()) {
                        corpus.push_back(bytes_to_tokens(piece));
                    }
                });
        }

        if (corpus.empty()) {
            throw std::invalid_argument("Cannot train a BPE tokenizer on an empty corpus");
        }

        const std::size_t workers = worker_count_for(config.worker_count, corpus.size());
        PairCounts global_counts;
        std::unordered_map<PairKey, OccurrenceList> occurrences;

        occurrences.reserve(65'536);
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
                occurrences[pair_key(sequence[i - 1], sequence[i])].append(
                    occurrence_key(document, i - 1));
                ++occurrence_entries;
            }
        }
        for (auto& [pair, list] : occurrences) {
            list.mark_sorted();
        }

        CandidateHeap candidates;
        candidates.reserve(global_counts.size());

        for (const auto &[pair, count]: global_counts) {
            candidates.insert(pair, count);
        }

        struct MergeUpdates {
            std::vector<std::pair<PairKey, std::int64_t>> count_deltas;
            std::vector<std::pair<PairKey, Occurrence>> new_occurrences;
            std::size_t merged_edges = 0;
        };
        std::vector<MergeUpdates> updates(workers);

        while (tokenizer.vocab_size() < config.vocab_size) {
            PairKey best_pair = 0;
            std::uint64_t best_count = 0;

            if (!candidates.empty()) {
                best_pair = candidates.top_pair();
                best_count = candidates.top_count();
            }

            if (best_count < config.min_pair_frequency) {
                break;
            }

            const TokenId left = pair_left(best_pair);
            const TokenId right = pair_right(best_pair);
            const auto merged = static_cast<TokenId>(tokenizer.vocab_size());
            tokenizer.add_merge(left, right);

            if (auto found = occurrences.find(best_pair); found != occurrences.end()) {
                OccurrenceList positions = std::move(found->second);
                occurrences.erase(found);
                occurrence_entries -= positions.values.size();
                positions.sort();

                const std::size_t merge_workers = worker_count_for(
                    workers, std::min(corpus.size(), positions.values.size()));
                for (std::size_t w = 0; w < merge_workers; ++w) {
                    updates[w].count_deltas.clear();
                    updates[w].new_occurrences.clear();
                    updates[w].merged_edges = 0;
                }
                parallel_for_ranges(corpus.size(), merge_workers,
                    [&corpus, &next_positions, &previous_positions, &positions,
                     &updates, left, right, merged](const std::size_t worker,
                                                      const std::size_t begin,
                                                      const std::size_t finish) {
                        MergeUpdates& local = updates[worker];
                        const auto first = std::lower_bound(
                            positions.values.begin(), positions.values.end(),
                            occurrence_key(begin, 0));
                        const auto last = finish == corpus.size()
                            ? positions.values.end()
                            : std::lower_bound(
                                positions.values.begin(), positions.values.end(),
                                occurrence_key(finish, 0));

                        for (auto current = first; current != last; ++current) {
                            const Occurrence occurrence = *current;
                            const std::size_t document = occurrence_document(occurrence);
                            const std::size_t position = occurrence_position(occurrence);
                            auto& sequence = corpus[document];
                            auto& next = next_positions[document];
                            auto& previous = previous_positions[document];

                            if (position >= sequence.size() || sequence[position] != left ||
                                next[position] == end || sequence[next[position]] != right) {
                                continue;
                            }

                            const std::size_t right_position = next[position];
                            const std::size_t before = previous[position];
                            const std::size_t after = next[right_position];
                            const auto remove_pair = [&](const std::size_t start) {
                                if (start != end && next[start] != end) {
                                    local.count_deltas.emplace_back(
                                        pair_key(sequence[start], sequence[next[start]]), -1);
                                }
                            };
                            const auto add_pair = [&](const std::size_t start) {
                                if (start != end && next[start] != end) {
                                    const PairKey pair = pair_key(
                                        sequence[start], sequence[next[start]]);
                                    local.count_deltas.emplace_back(pair, 1);
                                    local.new_occurrences.emplace_back(
                                        pair, occurrence_key(document, start));
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
                            ++local.merged_edges;
                        }
                    });

                std::unordered_map<PairKey, std::int64_t> total_deltas;
                for (std::size_t w = 0; w < merge_workers; ++w) {
                    MergeUpdates& local = updates[w];
                    active_edges -= local.merged_edges;

                    std::ranges::sort(local.count_deltas,
                        [](const auto& a, const auto& b) { return a.first < b.first; });

                    for (std::size_t i = 0; i < local.count_deltas.size();) {
                        const PairKey pair = local.count_deltas[i].first;
                        std::int64_t sum = 0;
                        std::size_t j = i;
                        while (j < local.count_deltas.size() && local.count_deltas[j].first == pair) {
                            sum += local.count_deltas[j].second;
                            ++j;
                        }
                        total_deltas[pair] += sum;
                        i = j;
                    }
                }
                for (const auto& [pair, delta] : total_deltas) {
                    if (delta == 0) {
                        continue;
                    }
                    auto found = global_counts.find(pair);
                    const std::int64_t previous = found == global_counts.end()
                        ? 0
                        : static_cast<std::int64_t>(found->second);
                    const std::int64_t updated = previous + delta;
                    if (updated == 0) {
                        if (found != global_counts.end()) {
                            global_counts.erase(found);
                        }
                        candidates.erase(pair);
                    } else {
                        const auto count = static_cast<std::uint64_t>(updated);
                        if (found == global_counts.end()) {
                            global_counts.emplace(pair, count);
                        } else {
                            found->second = count;
                        }
                        candidates.set(pair, count);
                    }
                }
                for (std::size_t w = 0; w < merge_workers; ++w) {
                    for (const auto& [pair, occurrence] : updates[w].new_occurrences) {
                        occurrences[pair].append(occurrence);
                        ++occurrence_entries;
                    }
                }

                const std::size_t compact_slack =
                    std::max<std::size_t>(1'000'000, active_edges / 8);
                if (occurrence_entries > active_edges + compact_slack) {
                    const std::size_t rebuild_workers = worker_count_for(workers, corpus.size());
                    std::vector<std::unordered_map<PairKey, std::vector<Occurrence>>>
                        local_occurrences(rebuild_workers);
                    parallel_for_ranges(corpus.size(), rebuild_workers,
                        [&corpus, &next_positions, &local_occurrences](
                            const std::size_t worker,
                            const std::size_t begin,
                            const std::size_t finish) {
                            auto& local = local_occurrences[worker];
                            for (std::size_t document = begin; document < finish; ++document) {
                                const auto &sequence = corpus[document];
                                const auto &next = next_positions[document];
                                for (std::size_t start = 0; start < sequence.size(); ++start) {
                                    if (next[start] != end) {
                                        local[pair_key(sequence[start], sequence[next[start]])]
                                            .push_back(occurrence_key(document, start));
                                    }
                                }
                            }
                        });

                    occurrences.clear();
                    occurrences.reserve(global_counts.size());
                    occurrence_entries = 0;
                    for (auto& local : local_occurrences) {
                        for (auto& [pair, values] : local) {
                            OccurrenceList& list = occurrences[pair];
                            occurrence_entries += values.size();
                            for (const Occurrence value : values) {
                                list.append(value);
                            }
                        }
                    }
                    for (auto& [pair, list] : occurrences) {
                        list.mark_sorted();
                    }
                }
            }

            progress.update(tokenizer.vocab_size() - kByteVocabularySize -
                            config.special_tokens.size());
        }
        progress.finish();
        tokenizer.rebuild_fast_merge_lookup();
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
        return encode(text, EncodeOptions{});
    }

    std::vector<TokenId> BpeTokenizer::encode(
        const std::string_view text,
        const EncodeOptions& options) const {
        // A 1024x1024 working window occupies 4 MiB and is substantially more
        // cache-friendly on mainstream desktop CPUs than the full 16 MiB table.
        // The packed fallback still handles every other pair exactly.
        constexpr unsigned single_dense_bits = 10;
        return encode_with_dense_limit(
            text, options, std::min(dense_merge_bits_, single_dense_bits));
    }

    std::vector<TokenId> BpeTokenizer::encode_with_dense_limit(
        const std::string_view text,
        const EncodeOptions& options,
        const unsigned dense_limit_bits) const {
        if (text.empty()) {
            return {};
        }

        // Special tokens are very rare in ordinary corpora. Walking the
        // Aho-Corasick DFA for every byte adds a dependent table load to the
        // entire input. A small number of std::string_view::find calls is much
        // faster in the common no-match case because the CRT implementation is
        // vectorized. We repeat the search only after an actual special token.
        std::vector<TokenId> result;
        // Natural-language corpora usually average well above 2 bytes/token.
        // Avoid reserving four bytes of TokenId storage for every input byte.
        result.reserve(text.size() / 2 + 16);

        std::size_t cursor = 0;
        while (cursor < text.size()) {
            std::size_t best_start = std::string_view::npos;
            std::size_t best_length = 0;
            std::size_t best_index = 0;

            for (std::size_t token_index = 0;
                 token_index < special_tokens_.size(); ++token_index) {
                const std::string& token = special_tokens_[token_index];
                const std::size_t found = text.find(token, cursor);
                if (found < best_start ||
                    (found == best_start && token.size() > best_length) ||
                    (found == best_start && token.size() == best_length &&
                     token_index < best_index)) {
                    best_start = found;
                    best_length = token.size();
                    best_index = token_index;
                }
            }

            if (best_start == std::string_view::npos) {
                append_encoded_bytes(
                    text.substr(cursor), result, options, dense_limit_bits);
                break;
            }

            append_encoded_bytes(
                text.substr(cursor, best_start - cursor),
                result,
                options,
                dense_limit_bits);
            result.push_back(
                kByteVocabularySize + static_cast<TokenId>(best_index));
            cursor = best_start + best_length;
        }

        return result;
    }

    std::vector<std::vector<TokenId>> BpeTokenizer::encode_batch(
        const std::vector<std::string>& texts,
        const std::size_t worker_count,
        const EncodeOptions& options) const {
        std::vector<std::string_view> views;
        views.reserve(texts.size());
        for (const std::string& text : texts) {
            views.emplace_back(text);
        }
        return encode_batch(views, worker_count, options);
    }

    std::vector<std::vector<TokenId>> BpeTokenizer::encode_batch(
        const std::span<const std::string_view> texts,
        const std::size_t worker_count,
        const EncodeOptions& options) const {
        std::vector<std::vector<TokenId>> result(texts.size());
        if (texts.empty()) return result;

        const std::size_t workers = worker_count_for(worker_count, texts.size());
        if (workers == 1) {
            for (std::size_t index = 0; index < texts.size(); ++index) {
                result[index] = encode(texts[index], options);
            }
            return result;
        }

        std::vector<std::thread> threads;
        threads.reserve(workers);
        constexpr unsigned batch_dense_bits = 10;
        const unsigned dense_limit_bits =
            std::min(dense_merge_bits_, batch_dense_bits);

        // Split by input bytes rather than document count. Real datasets often
        // contain highly uneven documents, which otherwise leaves some workers idle.
        std::vector<std::uint64_t> byte_prefix(texts.size() + 1, 0);
        for (std::size_t index = 0; index < texts.size(); ++index) {
            byte_prefix[index + 1] = byte_prefix[index] + texts[index].size();
        }

        const std::uint64_t total_bytes = byte_prefix.back();
        for (std::size_t worker = 0; worker < workers; ++worker) {
            const std::uint64_t target_begin = total_bytes * worker / workers;
            const std::uint64_t target_end = total_bytes * (worker + 1) / workers;
            const std::size_t begin = static_cast<std::size_t>(
                std::lower_bound(byte_prefix.begin(), byte_prefix.end(), target_begin) -
                byte_prefix.begin());
            const std::size_t end = worker + 1 == workers
                ? texts.size()
                : static_cast<std::size_t>(
                    std::lower_bound(byte_prefix.begin(), byte_prefix.end(), target_end) -
                    byte_prefix.begin());

            threads.emplace_back(
                [this, &texts, &result, &options, dense_limit_bits, begin, end] {
                    for (std::size_t index = begin; index < end; ++index) {
                        result[index] = encode_with_dense_limit(
                            texts[index], options, dense_limit_bits);
                    }
                });
        }
        for (auto& thread : threads) thread.join();
        return result;
    }

    void BpeTokenizer::append_encoded_bytes(
        const std::string_view text,
        std::vector<TokenId>& output,
        const EncodeOptions& options,
        const unsigned dense_limit_bits) const {
        if (text.empty()) return;

        if (pretokenizer_mode_ != PretokenizerMode::None) {
            for_each_pretoken(text, pretokenizer_mode_,
                [this, &output, &options, dense_limit_bits](const std::string_view piece) {
                    if (piece.empty()) return;
                    if (options.cache_entries == 0 ||
                        options.cache_max_input_bytes == 0 ||
                        piece.size() > options.cache_max_input_bytes) {
                        append_encoded_bytes_uncached(piece, output, dense_limit_bits);
                        return;
                    }

                    // Re-enter with pretokenization disabled through the local
                    // cache path below by duplicating only the inexpensive cache
                    // dispatch in append_encoded_pretoken.
                    struct ThreadCache {
                        struct StringHash {
                            using is_transparent = void;
                            [[nodiscard]] std::size_t operator()(
                                const std::string_view value) const noexcept {
                                return std::hash<std::string_view>{}(value);
                            }
                        };
                        const BpeTokenizer* owner = nullptr;
                        std::uint64_t identity = 0;
                        std::size_t merge_count = 0;
                        std::size_t capacity = 0;
                        std::unordered_map<std::string, std::vector<TokenId>,
                            StringHash, std::equal_to<>> entries;
                    };
                    thread_local ThreadCache cache;
                    if (cache.owner != this || cache.identity != cache_identity_ ||
                        cache.merge_count != merges_.size() ||
                        cache.capacity != options.cache_entries) {
                        cache.owner = this;
                        cache.identity = cache_identity_;
                        cache.merge_count = merges_.size();
                        cache.capacity = options.cache_entries;
                        cache.entries.clear();
                        cache.entries.reserve(options.cache_entries);
                    }
                    const auto found = cache.entries.find(piece);
                    if (found != cache.entries.end()) {
                        output.insert(output.end(), found->second.begin(), found->second.end());
                        return;
                    }
                    std::vector<TokenId> encoded;
                    encoded.reserve(piece.size());
                    append_encoded_bytes_uncached(piece, encoded, dense_limit_bits);
                    output.insert(output.end(), encoded.begin(), encoded.end());
                    if (cache.entries.size() >= options.cache_entries) {
                        // Incremental eviction avoids periodic full-cache latency spikes.
                        cache.entries.erase(cache.entries.begin());
                    }
                    cache.entries.emplace(std::string(piece), std::move(encoded));
                });
            return;
        }

        if (options.cache_entries == 0 || options.cache_max_input_bytes == 0 ||
            text.size() > options.cache_max_input_bytes) {
            append_encoded_bytes_uncached(text, output, dense_limit_bits);
            return;
        }

        struct ThreadCache {
            struct StringHash {
                using is_transparent = void;

                [[nodiscard]] std::size_t operator()(
                    const std::string_view value) const noexcept {
                    return std::hash<std::string_view>{}(value);
                }
            };

            const BpeTokenizer* owner = nullptr;
            std::uint64_t identity = 0;
            std::size_t merge_count = 0;
            std::size_t capacity = 0;
            std::unordered_map<
                std::string,
                std::vector<TokenId>,
                StringHash,
                std::equal_to<>> entries;
        };
        thread_local ThreadCache cache;

        if (cache.owner != this || cache.identity != cache_identity_ ||
            cache.merge_count != merges_.size() ||
            cache.capacity != options.cache_entries) {
            cache.owner = this;
            cache.identity = cache_identity_;
            cache.merge_count = merges_.size();
            cache.capacity = options.cache_entries;
            cache.entries.clear();
            cache.entries.reserve(options.cache_entries);
        }

        const auto found = cache.entries.find(text);
        if (found != cache.entries.end()) {
            output.insert(output.end(), found->second.begin(), found->second.end());
            return;
        }

        std::vector<TokenId> encoded;
        encoded.reserve(text.size());
        append_encoded_bytes_uncached(text, encoded, dense_limit_bits);
        output.insert(output.end(), encoded.begin(), encoded.end());

        if (cache.entries.size() >= options.cache_entries) {
            cache.entries.erase(cache.entries.begin());
        }
        cache.entries.emplace(std::string(text), std::move(encoded));
    }

    TokenId BpeTokenizer::lookup_merge(
        const TokenId left,
        const TokenId right,
        const unsigned dense_limit_bits) const noexcept {
        constexpr TokenId missing = std::numeric_limits<TokenId>::max();
        if (!dense_merge_lookup_.empty() &&
            ((left | right) >> dense_limit_bits) == 0) {
            return dense_merge_lookup_[
                (static_cast<std::size_t>(left) << dense_merge_bits_) | right];
        }

        if (!packed_merge_lookup_.empty()) {
            constexpr unsigned id_bits = 21;
            constexpr std::uint64_t id_mask = (std::uint64_t{1} << id_bits) - 1;
            if ((left | right) <= id_mask) {
                const std::uint64_t key =
                    (static_cast<std::uint64_t>(left) << id_bits) | right;
                std::size_t index = static_cast<std::size_t>(
                    key * 0x9E37'79B9'7F4A'7C15ULL >> packed_merge_shift_);
                while (true) {
                    const std::uint64_t slot = packed_merge_lookup_[index];
                    if (slot >> id_bits == key) {
                        return static_cast<TokenId>(slot & id_mask);
                    }
                    if (slot == std::numeric_limits<std::uint64_t>::max()) {
                        return missing;
                    }
                    index = (index + 1) & packed_merge_mask_;
                }
            }
        }

        const auto found = merge_lookup_.find(pair_key(left, right));
        return found == merge_lookup_.end() ? missing : found->second;
    }

    void BpeTokenizer::append_encoded_bytes_uncached(
        const std::string_view text,
        std::vector<TokenId>& output,
        const unsigned dense_limit_bits) const {
        if (text.empty()) return;

        const auto find_merge = [this, dense_limit_bits](
                                    const TokenId left,
                                    const TokenId right) noexcept {
            return lookup_merge(left, right, dense_limit_bits);
        };

        constexpr std::size_t short_merge_max = 32;
        if (text.size() <= short_merge_max) {
            constexpr TokenId missing = std::numeric_limits<TokenId>::max();
            std::array<TokenId, short_merge_max> symbols{};
            std::array<TokenId, short_merge_max> ranks{};
            std::array<std::uint8_t, short_merge_max> next{};
            std::array<std::uint8_t, short_merge_max> previous{};
            const std::size_t size = text.size();
            for (std::size_t i = 0; i < size; ++i) {
                symbols[i] = static_cast<unsigned char>(text[i]);
                next[i] = static_cast<std::uint8_t>(i + 1);
                previous[i] = static_cast<std::uint8_t>(i - 1);
                ranks[i] = missing;
            }
            for (std::size_t i = 0; i + 1 < size; ++i) {
                ranks[i] = find_merge(symbols[i], symbols[i + 1]);
            }

            while (true) {
                TokenId best_merge = missing;
                std::size_t best_position = 0;
                for (std::size_t i = 0; i + 1 < size; ++i) {
                    if (ranks[i] < best_merge) {
                        best_merge = ranks[i];
                        best_position = i;
                    }
                }
                if (best_merge == missing) {
                    break;
                }

                symbols[best_position] = best_merge;
                const std::size_t removed = next[best_position];
                const std::size_t new_right = next[removed];
                next[best_position] = static_cast<std::uint8_t>(new_right);
                ranks[removed] = missing;

                if (new_right < size) {
                    previous[new_right] = static_cast<std::uint8_t>(best_position);
                    ranks[best_position] =
                        find_merge(symbols[best_position], symbols[new_right]);
                } else {
                    ranks[best_position] = missing;
                }

                const std::size_t left = previous[best_position];
                if (left < size) {
                    ranks[left] = find_merge(symbols[left], symbols[best_position]);
                }
            }

            for (std::size_t index = 0; index < size; index = next[index]) {
                output.push_back(symbols[index]);
            }
            return;
        }

        const auto encode_with_links = [&]<typename Link>() {
            constexpr Link end = std::numeric_limits<Link>::max();

            struct Node {
                TokenId token;
                Link next;
                Link previous;
            };

            thread_local std::vector<Node> nodes;
            nodes.clear();
            nodes.reserve(text.size());
            for (std::size_t index = 0; index < text.size(); ++index) {
                nodes.push_back({
                    static_cast<TokenId>(static_cast<unsigned char>(text[index])),
                    index + 1 < text.size() ? static_cast<Link>(index + 1) : end,
                    index == 0 ? end : static_cast<Link>(index - 1)
                });
            }

            constexpr std::size_t rank_bucket_min_size = 4'096;
            if (text.size() >= rank_bucket_min_size) {
                const auto first_merged = static_cast<TokenId>(
                    kByteVocabularySize + special_tokens_.size());
                struct RankBucket {
                    std::vector<Link> values;
                    std::size_t sorted_size = 0;

                    void normalize() {
                        if (sorted_size == values.size()) {
                            return;
                        }
                        auto middle = values.begin() +
                            static_cast<std::ptrdiff_t>(sorted_size);
                        std::ranges::sort(middle, values.end());
                        if (sorted_size != 0) {
                            std::inplace_merge(values.begin(), middle, values.end());
                        }
                        sorted_size = values.size();
                    }

                    void clear() {
                        values.clear();
                        sorted_size = 0;
                    }
                };
                thread_local std::vector<RankBucket> positions;
                if (positions.size() < merges_.size()) {
                    positions.resize(merges_.size());
                }

                thread_local std::vector<std::size_t> rank_heap;
                rank_heap.clear();
                const auto add_rank_candidate = [&](const Link left) {
                    if (left == end || nodes[left].next == end) {
                        return;
                    }

                    constexpr TokenId missing = std::numeric_limits<TokenId>::max();
                    const TokenId merge = find_merge(
                        nodes[left].token, nodes[nodes[left].next].token);
                    if (merge == missing) {
                        return;
                    }

                    const std::size_t rank = merge - first_merged;
                    auto& bucket = positions[rank];
                    if (bucket.values.empty()) {
                        rank_heap.push_back(rank);
                        std::push_heap(
                            rank_heap.begin(), rank_heap.end(), std::greater<>());
                    }
                    bucket.values.push_back(left);
                };

                for (Link left = 0; left + 1 < nodes.size(); ++left) {
                    add_rank_candidate(left);
                }
                for (auto& bucket : positions) {
                    bucket.sorted_size = bucket.values.size();
                }

                std::size_t token_count = nodes.size();
                while (!rank_heap.empty()) {
                    std::pop_heap(
                        rank_heap.begin(), rank_heap.end(), std::greater<>());
                    const std::size_t rank = rank_heap.back();
                    rank_heap.pop_back();

                    auto& bucket = positions[rank];
                    bucket.normalize();
                    bucket.values.erase(
                        std::unique(bucket.values.begin(), bucket.values.end()),
                        bucket.values.end());
                    const MergeRule& rule = merges_[rank];
                    for (const Link left_index : bucket.values) {
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

                        add_rank_candidate(before);
                        add_rank_candidate(left_index);
                    }
                    bucket.clear();
                }

                output.reserve(output.size() + token_count);
                for (Link index = 0; index != end; index = nodes[index].next) {
                    output.push_back(nodes[index].token);
                }
                return;
            }

            // A packed pair is 8 bytes for the common uint32_t-link path.
            // Lexicographic ordering gives merge rank first and source
            // position second, exactly matching sequential BPE semantics.
            using Candidate = std::pair<TokenId, Link>;
            thread_local std::vector<Candidate> candidate_heap;
            candidate_heap.clear();

            const auto add_candidate = [&](const Link left) {
                if (left == end || nodes[left].next == end) {
                    return;
                }

                constexpr TokenId missing = std::numeric_limits<TokenId>::max();
                const TokenId merge = find_merge(
                    nodes[left].token, nodes[nodes[left].next].token);
                if (merge == missing) {
                    return;
                }
                candidate_heap.emplace_back(merge, left);
            };

            for (Link left = 0; left + 1 < nodes.size(); ++left) {
                add_candidate(left);
            }
            std::make_heap(
                candidate_heap.begin(), candidate_heap.end(), std::greater<>());

            std::size_t token_count = nodes.size();
            while (!candidate_heap.empty()) {
                std::pop_heap(
                    candidate_heap.begin(), candidate_heap.end(), std::greater<>());
                const auto [expected_merge, left_index] = candidate_heap.back();
                candidate_heap.pop_back();

                Node& left = nodes[left_index];
                if (left.next == end) {
                    continue;
                }

                const Link right_index = left.next;
                Node& right = nodes[right_index];
                if (find_merge(left.token, right.token) != expected_merge) {
                    continue;
                }

                const Link before = left.previous;
                const Link after = right.next;
                left.token = expected_merge;
                left.next = after;
                if (after != end) {
                    nodes[after].previous = left_index;
                }
                right.next = end;
                right.previous = end;
                --token_count;

                const std::size_t old_heap_size = candidate_heap.size();
                add_candidate(before);
                if (candidate_heap.size() != old_heap_size) {
                    std::push_heap(
                        candidate_heap.begin(), candidate_heap.end(), std::greater<>());
                }
                const std::size_t after_left_size = candidate_heap.size();
                add_candidate(left_index);
                if (candidate_heap.size() != after_left_size) {
                    std::push_heap(
                        candidate_heap.begin(), candidate_heap.end(), std::greater<>());
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

    constexpr std::array<char, 8> magic = {'B', 'P', 'E', 'T', 'O', 'K', '2', '\0'};
    if (special_tokens_.size() > std::numeric_limits<std::uint32_t>::max() ||
        merges_.size() > std::numeric_limits<std::uint32_t>::max()) {
        throw std::length_error("Tokenizer is too large to serialize");
    }
    output.write(magic.data(), static_cast<std::streamsize>(magic.size()));
    write_u32(output, static_cast<std::uint32_t>(special_tokens_.size()));
    write_u32(output, static_cast<std::uint32_t>(merges_.size()));
    write_u32(output, static_cast<std::uint32_t>(pretokenizer_mode_));

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

    constexpr std::array<char, 8> magic_v1 = {'B', 'P', 'E', 'T', 'O', 'K', '1', '\0'};
    constexpr std::array<char, 8> magic_v2 = {'B', 'P', 'E', 'T', 'O', 'K', '2', '\0'};
    std::array<char, 8> actual_magic{};
    input.read(actual_magic.data(), static_cast<std::streamsize>(actual_magic.size()));

    if (!input || (actual_magic != magic_v1 && actual_magic != magic_v2)) {
        throw std::runtime_error("Invalid tokenizer file magic");
    }

    const std::uint32_t special_count = read_u32(input);
    const std::uint32_t merge_count = read_u32(input);
    PretokenizerMode pretokenizer = PretokenizerMode::None;
    if (actual_magic == magic_v2) {
        const std::uint32_t raw_mode = read_u32(input);
        if (raw_mode > static_cast<std::uint32_t>(PretokenizerMode::GptLike)) {
            throw std::runtime_error("Tokenizer file declares an unknown pretokenizer");
        }
        pretokenizer = static_cast<PretokenizerMode>(raw_mode);
    }

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

    BpeTokenizer tokenizer(std::move(special_tokens), pretokenizer);
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
    tokenizer.rebuild_fast_merge_lookup();

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

PretokenizerMode BpeTokenizer::pretokenizer_mode() const noexcept {
    return pretokenizer_mode_;
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

void BpeTokenizer::rebuild_fast_merge_lookup() {
    constexpr unsigned id_bits = 21;
    constexpr TokenId id_limit = TokenId{1} << id_bits;
    constexpr TokenId missing = std::numeric_limits<TokenId>::max();
    constexpr std::uint64_t empty = std::numeric_limits<std::uint64_t>::max();

    packed_merge_lookup_.clear();
    packed_merge_mask_ = 0;
    packed_merge_shift_ = 0;
    dense_merge_lookup_.clear();
    dense_merge_bits_ = 0;
    if (merges_.empty() || token_bytes_.size() > id_limit) {
        return;
    }

    // A physically compact 1024 x 1024 table occupies exactly 4 MiB. Keeping
    // the row stride equal to the checked ID range is important for cache locality.
    constexpr unsigned dense_bits = 10;
    constexpr std::size_t dense_side = std::size_t{1} << dense_bits;
    dense_merge_lookup_.assign(dense_side * dense_side, missing);
    dense_merge_bits_ = dense_bits;
    for (const MergeRule& merge : merges_) {
        if (((merge.left | merge.right) >> dense_bits) == 0) {
            dense_merge_lookup_[
                (static_cast<std::size_t>(merge.left) << dense_bits) |
                merge.right] = merge.merged;
        }
    }

    const std::size_t slot_count =
        std::max<std::size_t>(64, std::bit_ceil(merges_.size() * 2));
    packed_merge_lookup_.assign(slot_count, empty);
    packed_merge_mask_ = slot_count - 1;
    packed_merge_shift_ = 64U - std::countr_zero(slot_count);

    for (const MergeRule& merge : merges_) {
        if (merge.left >= id_limit || merge.right >= id_limit ||
            merge.merged >= id_limit) {
            dense_merge_lookup_.clear();
            dense_merge_bits_ = 0;
            packed_merge_lookup_.clear();
            packed_merge_mask_ = 0;
            packed_merge_shift_ = 0;
            return;
        }

        const std::uint64_t key =
            (static_cast<std::uint64_t>(merge.left) << id_bits) | merge.right;
        std::size_t index = static_cast<std::size_t>(
            key * 0x9E37'79B9'7F4A'7C15ULL >> packed_merge_shift_);
        std::size_t displacement = 0;
        while (packed_merge_lookup_[index] != empty) {
            index = (index + 1) & packed_merge_mask_;
            if (++displacement > 64) {
                packed_merge_lookup_.clear();
                packed_merge_mask_ = 0;
                packed_merge_shift_ = 0;
                return;
            }
        }
        packed_merge_lookup_[index] = (key << id_bits) | merge.merged;
    }
}

void BpeTokenizer::rebuild_token_bytes() {
    token_bytes_.clear();
    token_bytes_.reserve(kByteVocabularySize + special_tokens_.size() + merges_.size());
    for (TokenId byte = 0; byte < kByteVocabularySize; ++byte) {
        token_bytes_.push_back({static_cast<std::uint8_t>(byte)});
    }
    token_bytes_.resize(kByteVocabularySize + special_tokens_.size());
    merge_lookup_.clear();
    dense_merge_lookup_.clear();
    dense_merge_bits_ = 0;
    packed_merge_lookup_.clear();
    packed_merge_mask_ = 0;
    packed_merge_shift_ = 0;
}

bool BpeTokenizer::is_special(const TokenId id) const noexcept {
    return id >= kByteVocabularySize &&
           id < kByteVocabularySize + special_tokens_.size();
}
}