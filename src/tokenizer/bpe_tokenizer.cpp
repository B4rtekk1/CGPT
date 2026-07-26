#include "tokenizer/bpe_tokenizer.h"
#include "utils/progress_bar.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <fstream>
#include <ranges>
#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#define BPE_HAVE_X64_SIMD 1
#endif

#ifdef BPE_HAVE_X64_SIMD
#if defined(_MSC_VER) && !defined(__clang__)
#define BPE_PREFETCH(addr, rw, locality) \
    _mm_prefetch(reinterpret_cast<const char*>(addr), _MM_HINT_T0)
#else
#define BPE_PREFETCH(addr, rw, locality) \
    __builtin_prefetch((addr), (rw), (locality))
#endif
#endif
#include <limits>
#include <numeric>
#include <iterator>
#include <map>
#include <stdexcept>
#include <thread>
#include <unordered_map>
#include <utility>

namespace bpe {
    namespace {
        using PairKey = std::uint64_t;
        using PairCounts = std::unordered_map<PairKey, std::uint64_t>;
        using Occurrence = std::uint32_t;

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
                entries_.push_back({.count = count, .pair = pair});
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

                auto unsorted_values = values | std::views::drop(sorted_size);
                std::ranges::sort(unsorted_values);
                if (sorted_size != 0) {
                    std::ranges::inplace_merge(
                        values | std::views::take(sorted_size), unsorted_values);
                }
                mark_sorted();
            }
        };

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
                result.push_back(byte);
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

        [[nodiscard]] ByteClass classify_byte(const unsigned char byte) noexcept {
            return kByteClasses[byte];
        }

#ifdef BPE_HAVE_X64_SIMD
#define BPE_HAVE_SSE2 1
#endif
#if defined(BPE_HAVE_X64_SIMD) && defined(__AVX2__)
#define BPE_HAVE_AVX2 1
#endif

#ifdef BPE_HAVE_AVX2
        [[nodiscard]] inline __m256i classify_vector_avx2(const __m256i bytes) noexcept {
            const __m256i zero = _mm256_setzero_si256();
            const __m256i high_bit = _mm256_cmpgt_epi8(zero, bytes); // unsigned >= 0x80

            const __m256i bx = _mm256_xor_si256(bytes, _mm256_set1_epi8(static_cast<char>(0x80)));
            const auto in_range = [&](const unsigned char lo, const unsigned char hi) noexcept {
                const __m256i lo_x = _mm256_set1_epi8(
                    static_cast<char>(static_cast<unsigned char>(lo - 1) ^ 0x80U));
                const __m256i hi_x = _mm256_set1_epi8(
                    static_cast<char>(static_cast<unsigned char>(hi + 1) ^ 0x80U));
                return _mm256_and_si256(
                    _mm256_cmpgt_epi8(bx, lo_x), _mm256_cmpgt_epi8(hi_x, bx));
            };

            const __m256i ws = _mm256_or_si256(
                in_range(0x09, 0x0D), _mm256_cmpeq_epi8(bytes, _mm256_set1_epi8(0x20)));
            const __m256i digit = in_range('0', '9');
            __m256i letter = _mm256_or_si256(in_range('A', 'Z'), in_range('a', 'z'));
            letter = _mm256_or_si256(letter, _mm256_cmpeq_epi8(bytes, _mm256_set1_epi8('_')));
            letter = _mm256_or_si256(letter, high_bit);

            __m256i cls = _mm256_set1_epi8(static_cast<char>(ByteClass::Punctuation));
            cls = _mm256_or_si256(
                _mm256_andnot_si256(letter, cls),
                _mm256_and_si256(letter, _mm256_set1_epi8(static_cast<char>(ByteClass::Letter))));
            cls = _mm256_or_si256(
                _mm256_andnot_si256(digit, cls),
                _mm256_and_si256(digit, _mm256_set1_epi8(static_cast<char>(ByteClass::Digit))));
            cls = _mm256_or_si256(
                _mm256_andnot_si256(ws, cls),
                _mm256_and_si256(ws, _mm256_set1_epi8(static_cast<char>(ByteClass::Whitespace))));
            return cls;
        }
#endif

#ifdef BPE_HAVE_SSE2
        [[nodiscard]] __m128i classify_vector_sse2(const __m128i bytes) noexcept {
            const __m128i zero = _mm_setzero_si128();
            const __m128i high_bit = _mm_cmplt_epi8(bytes, zero); // unsigned >= 0x80

            const __m128i bx = _mm_xor_si128(bytes, _mm_set1_epi8(static_cast<char>(0x80)));
            const auto in_range = [&](const unsigned char lo, const unsigned char hi) noexcept {
                const __m128i lo_x = _mm_set1_epi8(
                    static_cast<char>(static_cast<unsigned char>(lo - 1) ^ 0x80U));
                const __m128i hi_x = _mm_set1_epi8(
                    static_cast<char>(static_cast<unsigned char>(hi + 1) ^ 0x80U));
                return _mm_and_si128(_mm_cmpgt_epi8(bx, lo_x), _mm_cmpgt_epi8(hi_x, bx));
            };

            const __m128i ws = _mm_or_si128(
                in_range(0x09, 0x0D), _mm_cmpeq_epi8(bytes, _mm_set1_epi8(0x20)));
            const __m128i digit = in_range('0', '9');
            __m128i letter = _mm_or_si128(in_range('A', 'Z'), in_range('a', 'z'));
            letter = _mm_or_si128(letter, _mm_cmpeq_epi8(bytes, _mm_set1_epi8('_')));
            letter = _mm_or_si128(letter, high_bit);

            __m128i cls = _mm_set1_epi8(static_cast<char>(ByteClass::Punctuation));
            cls = _mm_or_si128(
                _mm_andnot_si128(letter, cls),
                _mm_and_si128(letter, _mm_set1_epi8(static_cast<char>(ByteClass::Letter))));
            cls = _mm_or_si128(
                _mm_andnot_si128(digit, cls),
                _mm_and_si128(digit, _mm_set1_epi8(static_cast<char>(ByteClass::Digit))));
            cls = _mm_or_si128(
                _mm_andnot_si128(ws, cls),
                _mm_and_si128(ws, _mm_set1_epi8(static_cast<char>(ByteClass::Whitespace))));
            return cls;
        }
#endif

        [[nodiscard]] std::size_t scan_class_run(
            const std::string_view text,
            std::size_t position,
            const ByteClass byte_class) noexcept {
#ifdef BPE_HAVE_AVX2
            const __m256i expected32 = _mm256_set1_epi8(
                static_cast<char>(byte_class));
            while (position + 32 <= text.size()) {
                const __m256i bytes = _mm256_loadu_si256(
                    reinterpret_cast<const __m256i*>(text.data() + position));
                const auto matches = static_cast<std::uint32_t>(
                    _mm256_movemask_epi8(_mm256_cmpeq_epi8(
                        classify_vector_avx2(bytes), expected32)));
                if (matches != 0xFFFF'FFFFU) {
                    return position + std::countr_one(matches);
                }
                position += 32;
            }
#endif
#ifdef BPE_HAVE_SSE2
            const __m128i expected16 = _mm_set1_epi8(
                static_cast<char>(byte_class));
            while (position + 16 <= text.size()) {
                const __m128i bytes = _mm_loadu_si128(
                    reinterpret_cast<const __m128i*>(text.data() + position));
                const auto matches = static_cast<std::uint32_t>(
                    _mm_movemask_epi8(_mm_cmpeq_epi8(
                        classify_vector_sse2(bytes), expected16)));
                if (matches != 0xFFFFU) {
                    return position + std::countr_one(matches);
                }
                position += 16;
            }
#endif
            while (position < text.size() &&
                   classify_byte(static_cast<unsigned char>(text[position])) ==
                       byte_class) {
                ++position;
            }
            return position;
        }

        template<typename Function>
        void for_each_pretoken(
            const std::string_view text,
            const PretokenizerMode mode,
            Function&& function) {
            if (text.empty()) return;
            if (mode == PretokenizerMode::None) {
                function(text);
                return;
            }

            std::size_t position = 0;
            while (position < text.size()) {
                const std::size_t begin = position;
                ByteClass byte_class = classify_byte(
                    static_cast<unsigned char>(text[position]));
                if (text[position] == ' ' &&
                    (position == 0 ||
                     classify_byte(static_cast<unsigned char>(text[position - 1])) !=
                         ByteClass::Whitespace) &&
                    position + 1 < text.size()) {
                    const ByteClass following = classify_byte(
                        static_cast<unsigned char>(text[position + 1]));
                    if (following != ByteClass::Whitespace) {
                        byte_class = following;
                        ++position;
                    }
                }

                if (byte_class == ByteClass::Digit) {
                    const std::size_t limit = std::min(position + 3, text.size());
                    while (position < limit &&
                           classify_byte(static_cast<unsigned char>(text[position])) ==
                               ByteClass::Digit) {
                        ++position;
                    }
                } else {
                    position = scan_class_run(text, position, byte_class);
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

        struct JsonValue {
            enum class Type { Null, Boolean, Number, String, Array, Object };
            Type type = Type::Null;
            bool boolean = false;
            std::uint64_t number = 0;
            std::string string;
            std::vector<JsonValue> array;
            std::map<std::string, JsonValue, std::less<>> object;

            [[nodiscard]] const JsonValue& member(
                const std::string_view name) const {
                if (type != Type::Object) {
                    throw std::runtime_error("Expected a JSON object");
                }
                const auto found = object.find(name);
                if (found == object.end()) {
                    throw std::runtime_error(
                        "Missing tokenizer.json field: " + std::string(name));
                }
                return found->second;
            }
        };

        class JsonParser {
        public:
            explicit JsonParser(const std::string_view input) : input_(input) {}

            [[nodiscard]] JsonValue parse() {
                JsonValue value = parse_value();
                skip_space();
                if (position_ != input_.size()) {
                    fail("trailing data");
                }
                return value;
            }

        private:
            [[noreturn]] void fail(const std::string_view reason) const {
                throw std::runtime_error(
                    "Invalid tokenizer.json at byte " +
                    std::to_string(position_) + ": " + std::string(reason));
            }

            void skip_space() {
                while (position_ < input_.size() &&
                       (input_[position_] == ' ' || input_[position_] == '\t' ||
                        input_[position_] == '\r' || input_[position_] == '\n')) {
                    ++position_;
                }
            }

            [[nodiscard]] bool consume(const char expected) {
                skip_space();
                if (position_ < input_.size() && input_[position_] == expected) {
                    ++position_;
                    return true;
                }
                return false;
            }

            void expect(const char expected) {
                if (!consume(expected)) {
                    fail(std::string("expected '") + expected + "'");
                }
            }

            void expect_literal(const std::string_view literal) {
                if (input_.substr(position_, literal.size()) != literal) {
                    fail("invalid literal");
                }
                position_ += literal.size();
            }

            static void append_utf8(std::string& output, const std::uint32_t cp) {
                if (cp <= 0x7F) {
                    output.push_back(static_cast<char>(cp));
                } else if (cp <= 0x7FF) {
                    output.push_back(static_cast<char>(0xC0 | (cp >> 6)));
                    output.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
                } else if (cp <= 0xFFFF) {
                    output.push_back(static_cast<char>(0xE0 | (cp >> 12)));
                    output.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
                    output.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
                } else {
                    output.push_back(static_cast<char>(0xF0 | (cp >> 18)));
                    output.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3F)));
                    output.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
                    output.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
                }
            }

            [[nodiscard]] std::uint32_t parse_hex4() {
                if (position_ + 4 > input_.size()) fail("short Unicode escape");
                std::uint32_t value = 0;
                for (unsigned index = 0; index < 4; ++index) {
                    const char character = input_[position_++];
                    value <<= 4;
                    if (character >= '0' && character <= '9') {
                        value += character - '0';
                    } else if (character >= 'a' && character <= 'f') {
                        value += character - 'a' + 10;
                    } else if (character >= 'A' && character <= 'F') {
                        value += character - 'A' + 10;
                    } else {
                        fail("invalid Unicode escape");
                    }
                }
                return value;
            }

            [[nodiscard]] std::string parse_string() {
                skip_space();
                if (position_ >= input_.size() || input_[position_++] != '"') {
                    fail("expected string");
                }
                std::string result;
                while (position_ < input_.size()) {
                    const unsigned char character = input_[position_++];
                    if (character == '"') return result;
                    if (character < 0x20) fail("control character in string");
                    if (character != '\\') {
                        result.push_back(static_cast<char>(character));
                        continue;
                    }
                    if (position_ >= input_.size()) fail("short escape");
                    switch (input_[position_++]) {
                        case '"': result.push_back('"'); break;
                        case '\\': result.push_back('\\'); break;
                        case '/': result.push_back('/'); break;
                        case 'b': result.push_back('\b'); break;
                        case 'f': result.push_back('\f'); break;
                        case 'n': result.push_back('\n'); break;
                        case 'r': result.push_back('\r'); break;
                        case 't': result.push_back('\t'); break;
                        case 'u': {
                            std::uint32_t cp = parse_hex4();
                            if (cp >= 0xD800 && cp <= 0xDBFF) {
                                if (position_ + 2 > input_.size() ||
                                    input_[position_] != '\\' ||
                                    input_[position_ + 1] != 'u') {
                                    fail("missing low surrogate");
                                }
                                position_ += 2;
                                const std::uint32_t low = parse_hex4();
                                if (low < 0xDC00 || low > 0xDFFF) {
                                    fail("invalid low surrogate");
                                }
                                cp = 0x10000 + ((cp - 0xD800) << 10) +
                                     (low - 0xDC00);
                            } else if (cp >= 0xDC00 && cp <= 0xDFFF) {
                                fail("unexpected low surrogate");
                            }
                            append_utf8(result, cp);
                            break;
                        }
                        default: fail("invalid escape");
                    }
                }
                fail("unterminated string");
            }

            [[nodiscard]] JsonValue parse_value() {
                skip_space();
                if (position_ >= input_.size()) fail("unexpected end of input");
                JsonValue result;
                switch (input_[position_]) {
                    case '{': {
                        result.type = JsonValue::Type::Object;
                        ++position_;
                        if (consume('}')) return result;
                        do {
                            std::string key = parse_string();
                            expect(':');
                            if (!result.object.emplace(
                                    std::move(key), parse_value()).second) {
                                fail("duplicate object key");
                            }
                        } while (consume(','));
                        expect('}');
                        return result;
                    }
                    case '[':
                        result.type = JsonValue::Type::Array;
                        ++position_;
                        if (consume(']')) return result;
                        do {
                            result.array.push_back(parse_value());
                        } while (consume(','));
                        expect(']');
                        return result;
                    case '"':
                        result.type = JsonValue::Type::String;
                        result.string = parse_string();
                        return result;
                    case 't':
                        expect_literal("true");
                        result.type = JsonValue::Type::Boolean;
                        result.boolean = true;
                        return result;
                    case 'f':
                        expect_literal("false");
                        result.type = JsonValue::Type::Boolean;
                        return result;
                    case 'n':
                        expect_literal("null");
                        return result;
                    default:
                        if (input_[position_] < '0' || input_[position_] > '9') {
                            fail("expected value");
                        }
                        result.type = JsonValue::Type::Number;
                        do {
                            const auto digit =
                                static_cast<unsigned>(input_[position_] - '0');
                            if (result.number >
                                (std::numeric_limits<std::uint64_t>::max() - digit) /
                                    10) {
                                fail("number is too large");
                            }
                            result.number = result.number * 10 + digit;
                            ++position_;
                        } while (position_ < input_.size() &&
                                 input_[position_] >= '0' &&
                                 input_[position_] <= '9');
                        return result;
                }
            }

            std::string_view input_;
            std::size_t position_ = 0;
        };

        [[nodiscard]] std::array<std::string, 256> bytelevel_alphabet() {
            std::array<std::string, 256> result;
            std::uint32_t replacement = 0;
            for (std::uint32_t byte = 0; byte < 256; ++byte) {
                const bool direct =
                    (byte >= 33 && byte <= 126) ||
                    (byte >= 161 && byte <= 172) || byte >= 174;
                std::uint32_t cp = direct ? byte : 256 + replacement++;
                if (cp <= 0x7F) {
                    result[byte].push_back(static_cast<char>(cp));
                } else {
                    result[byte].push_back(static_cast<char>(0xC0 | (cp >> 6)));
                    result[byte].push_back(static_cast<char>(0x80 | (cp & 0x3F)));
                }
            }
            return result;
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
        rebuild_token_bytes();
    }

    BpeTokenizer BpeTokenizer::train(
        const std::vector<std::string> &documents,
        const TrainerConfig &config
    ) {
        const std::size_t minimum_vocab =
            kByteVocabularySize + config.special_tokens.size();

        if (config.vocab_size < minimum_vocab) {
            throw std::invalid_argument(
                "vocab_size is smaller than byte and special-token vocabulary");
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

        std::size_t input_bytes = 0;
        for (const auto& document : documents) input_bytes += document.size();
        if (input_bytes >= std::numeric_limits<Occurrence>::max()) {
            throw std::length_error(
                "ExactInMemory supports less than 4 GiB of pretokenized input");
        }

        std::vector<TokenId> corpus;
        corpus.reserve(input_bytes);
        std::vector<Occurrence> piece_offsets;
        piece_offsets.reserve(documents.size() * 8 + 1);
        piece_offsets.push_back(0);

        for (const std::string &document : documents) {
            for_each_pretoken(document, config.pretokenizer,
                [&corpus, &piece_offsets](const std::string_view piece) {
                    if (piece.empty()) return;
                    if (corpus.size() + piece.size() >=
                        std::numeric_limits<Occurrence>::max()) {
                        throw std::length_error(
                            "ExactInMemory corpus exceeds 32-bit position space");
                    }
                    for (const unsigned char byte : piece) {
                        corpus.push_back(byte);
                    }
                    piece_offsets.push_back(static_cast<Occurrence>(corpus.size()));
                });
        }

        if (corpus.empty() || piece_offsets.size() <= 1) {
            throw std::invalid_argument("Cannot train a BPE tokenizer on an empty corpus");
        }

        using Link = Occurrence;
        constexpr Link link_end = std::numeric_limits<Link>::max();
        std::vector<Link> next_positions(corpus.size());
        std::vector<Link> previous_positions(corpus.size());
        std::size_t active_edges = 0;

        const std::size_t piece_count = piece_offsets.size() - 1;
        for (std::size_t piece = 0; piece < piece_count; ++piece) {
            const std::size_t begin = piece_offsets[piece];
            const std::size_t finish = piece_offsets[piece + 1];
            if (finish > begin) active_edges += finish - begin - 1;
            for (std::size_t i = begin; i < finish; ++i) {
                next_positions[i] = i + 1 < finish ? static_cast<Link>(i + 1) : link_end;
                previous_positions[i] = i == begin ? link_end : static_cast<Link>(i - 1);
            }
        }

        const std::size_t workers = worker_count_for(config.worker_count, piece_count);
        PairCounts global_counts;

        {
            std::vector<PairCounts> initial_counts(workers);
            parallel_for_ranges(piece_count, workers,
                [&corpus, &piece_offsets, &initial_counts](
                    const std::size_t worker,
                    const std::size_t begin_piece,
                    const std::size_t end_piece) {
                    PairCounts &counts = initial_counts[worker];
                    for (std::size_t piece = begin_piece; piece < end_piece; ++piece) {
                        const std::size_t begin = piece_offsets[piece];
                        const std::size_t finish = piece_offsets[piece + 1];
                        for (std::size_t i = begin + 1; i < finish; ++i) {
                            ++counts[pair_key(corpus[i - 1], corpus[i])];
                        }
                    }
                });

            std::size_t unique_hint = 0;
            for (const auto& counts : initial_counts) unique_hint += counts.size();
            global_counts.reserve(unique_hint / 2 + 1);
            for (const PairCounts &counts : initial_counts) {
                for (const auto &[pair, count] : counts) {
                    global_counts[pair] += count;
                }
            }
        }

        std::unordered_map<PairKey, OccurrenceList> occurrences;
        occurrences.reserve(global_counts.size());
        for (const auto& [pair, count] : global_counts) {
            auto [entry, inserted] = occurrences.try_emplace(pair);
            (void)inserted;
            entry->second.values.reserve(count);
        }
        std::size_t occurrence_entries = 0;
        for (std::size_t piece = 0; piece < piece_count; ++piece) {
            const std::size_t begin = piece_offsets[piece];
            const std::size_t finish = piece_offsets[piece + 1];
            for (std::size_t i = begin + 1; i < finish; ++i) {
                const PairKey pair = pair_key(corpus[i - 1], corpus[i]);
                occurrences.find(pair)->second.append(static_cast<Occurrence>(i - 1));
                ++occurrence_entries;
            }
        }
        for (auto &list: occurrences | std::views::values) list.mark_sorted();

        CandidateHeap candidates;
        candidates.reserve(global_counts.size());
        for (const auto &[pair, count] : global_counts) candidates.insert(pair, count);

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
            if (best_count < config.min_pair_frequency) break;

            const TokenId left = pair_left(best_pair);
            const TokenId right = pair_right(best_pair);
            const auto merged = static_cast<TokenId>(tokenizer.vocab_size());
            tokenizer.add_merge(left, right);

            if (auto found = occurrences.find(best_pair); found != occurrences.end()) {
                OccurrenceList positions = std::move(found->second);
                occurrences.erase(found);
                occurrence_entries -= positions.values.size();
                positions.sort();

                constexpr std::size_t min_occurrences_per_parallel_merge = 4'096;
                const std::size_t merge_workers =
                    positions.values.size() < min_occurrences_per_parallel_merge
                        ? 1
                        : worker_count_for(
                              workers, std::min(piece_count, positions.values.size()));
                for (std::size_t w = 0; w < merge_workers; ++w) {
                    updates[w].count_deltas.clear();
                    updates[w].new_occurrences.clear();
                    updates[w].merged_edges = 0;
                }

                parallel_for_ranges(piece_count, merge_workers,
                    [&corpus, &next_positions, &previous_positions, &piece_offsets,
                     &positions, &updates, left, right, merged](
                        const std::size_t worker,
                        const std::size_t begin_piece,
                        const std::size_t end_piece) {
                        MergeUpdates& local = updates[worker];
                        const Occurrence begin_position = piece_offsets[begin_piece];
                        const Occurrence end_position = piece_offsets[end_piece];
                        const auto first = std::ranges::lower_bound(
                            positions.values.begin(), positions.values.end(), begin_position);
                        const auto last = std::ranges::lower_bound(
                            positions.values.begin(), positions.values.end(), end_position);

                        for (auto current = first; current != last; ++current) {
                            constexpr std::ptrdiff_t prefetch_ahead = 8;
#ifdef BPE_HAVE_X64_SIMD
                            if (const auto ahead = current + prefetch_ahead; ahead < last) {
                                const std::size_t future_position = *ahead;
                                BPE_PREFETCH(&corpus[future_position], 1, 1);
                                BPE_PREFETCH(&next_positions[future_position], 0, 1);
                                BPE_PREFETCH(&previous_positions[future_position], 0, 1);
                            }
#endif
                            const std::size_t position = *current;
                            if (corpus[position] != left ||
                                next_positions[position] == link_end ||
                                corpus[next_positions[position]] != right) {
                                continue;
                            }

                            const std::size_t right_position = next_positions[position];
                            const Link before = previous_positions[position];
                            const Link after = next_positions[right_position];
                            const auto remove_pair = [&](const Link start) {
                                if (start != link_end && next_positions[start] != link_end) {
                                    local.count_deltas.emplace_back(
                                        pair_key(corpus[start], corpus[next_positions[start]]), -1);
                                }
                            };
                            const auto add_pair = [&](const Link start) {
                                if (start != link_end && next_positions[start] != link_end) {
                                    const PairKey pair = pair_key(
                                        corpus[start], corpus[next_positions[start]]);
                                    local.count_deltas.emplace_back(pair, 1);
                                    local.new_occurrences.emplace_back(pair, start);
                                }
                            };

                            remove_pair(before);
                            remove_pair(static_cast<Link>(position));
                            remove_pair(static_cast<Link>(right_position));

                            corpus[position] = merged;
                            next_positions[position] = after;
                            if (after != link_end) previous_positions[after] = static_cast<Link>(position);
                            if (before != link_end) next_positions[before] = static_cast<Link>(position);
                            next_positions[right_position] = link_end;
                            previous_positions[right_position] = link_end;

                            add_pair(before);
                            add_pair(static_cast<Link>(position));
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
                        while (j < local.count_deltas.size() &&
                               local.count_deltas[j].first == pair) {
                            sum += local.count_deltas[j].second;
                            ++j;
                        }
                        total_deltas[pair] += sum;
                        i = j;
                    }
                }

                for (const auto& [pair, delta] : total_deltas) {
                    if (delta == 0) continue;
                    auto count_it = global_counts.find(pair);
                    const std::int64_t previous = count_it == global_counts.end()
                        ? 0 : static_cast<std::int64_t>(count_it->second);
                    const std::int64_t updated = previous + delta;
                    if (updated <= 0) {
                        if (count_it != global_counts.end()) global_counts.erase(count_it);
                        candidates.erase(pair);
                    } else {
                        const auto count = static_cast<std::uint64_t>(updated);
                        if (count_it == global_counts.end()) global_counts.emplace(pair, count);
                        else count_it->second = count;
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
                    const std::size_t rebuild_workers = worker_count_for(workers, piece_count);
                    std::vector<std::unordered_map<PairKey, std::vector<Occurrence>>>
                        local_occurrences(rebuild_workers);
                    parallel_for_ranges(piece_count, rebuild_workers,
                        [&corpus, &next_positions, &piece_offsets, &local_occurrences](
                            const std::size_t worker,
                            const std::size_t begin_piece,
                            const std::size_t end_piece) {
                            auto& local = local_occurrences[worker];
                            for (std::size_t piece = begin_piece; piece < end_piece; ++piece) {
                                const std::size_t begin = piece_offsets[piece];
                                const std::size_t finish = piece_offsets[piece + 1];
                                for (std::size_t start = begin; start < finish; ++start) {
                                    if (next_positions[start] != link_end) {
                                        local[pair_key(corpus[start], corpus[next_positions[start]])]
                                            .push_back(static_cast<Occurrence>(start));
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
                            list.values.insert(list.values.end(),
                                std::make_move_iterator(values.begin()),
                                std::make_move_iterator(values.end()));
                        }
                    }
                    for (auto &list: occurrences | std::views::values) {
                        std::ranges::sort(list.values);
                        list.mark_sorted();
                    }
                }
            }

            progress.update(tokenizer.vocab_size() - minimum_vocab);
        }

        progress.finish();
        tokenizer.rebuild_fast_merge_lookup();
        return tokenizer;
    }

    BpeTokenizer BpeTokenizer::train_low_memory(
        std::istream& input,
        const TrainerConfig& config,
        const std::size_t block_size
    ) {
        if (block_size == 0) {
            throw std::invalid_argument("low-memory block_size must be at least one");
        }
        if (block_size > static_cast<std::size_t>(
                std::numeric_limits<std::streamsize>::max())) {
            throw std::invalid_argument("low-memory block_size exceeds stream limits");
        }
        if (config.low_memory_merges_per_pass == 0) {
            throw std::invalid_argument(
                "low_memory_merges_per_pass must be at least one");
        }

        const std::size_t minimum_vocab =
            kByteVocabularySize + config.special_tokens.size();
        if (config.vocab_size < minimum_vocab) {
            throw std::invalid_argument(
                "vocab_size is smaller than byte and special-token vocabulary");
        }

        const std::streampos origin = input.tellg();
        if (origin == std::streampos(-1)) {
            throw std::invalid_argument(
                "LowMemoryStreaming requires a seekable input stream");
        }

        BpeTokenizer tokenizer(config.special_tokens, config.pretokenizer);
        tokenizer.merges_.reserve(config.vocab_size - minimum_vocab);
        tokenizer.token_bytes_.reserve(config.vocab_size);
        tokenizer.merge_lookup_.reserve(config.vocab_size - minimum_vocab);

        ProgressBar progress(
            config.vocab_size - minimum_vocab, "Low-memory BPE training");
        std::string block(block_size, '\0');
        std::vector<TokenId> encoded;
        encoded.reserve(block_size / 2 + 64);

        while (tokenizer.vocab_size() < config.vocab_size) {
            input.clear();
            input.seekg(origin);
            if (!input) {
                throw std::runtime_error("Cannot rewind tokenizer training stream");
            }

            const std::size_t current_vocab = tokenizer.vocab_size();
            const bool use_dense =
                current_vocab <= config.low_memory_dense_vocab_limit &&
                current_vocab <= 4'096;
            std::vector<std::uint64_t> dense_counts;
            PairCounts sparse_counts;
            if (use_dense) {
                dense_counts.assign(current_vocab * current_vocab, 0);
            } else {
                sparse_counts.reserve(std::min<std::size_t>(
                    current_vocab * 32, 2'000'000));
            }

            std::uint64_t observed_pairs = 0;
            while (input.read(block.data(), static_cast<std::streamsize>(block.size())) ||
                   input.gcount() != 0) {
                const auto bytes_read =
                    static_cast<std::size_t>(input.gcount());
                const std::string_view view(block.data(), bytes_read);

                for_each_pretoken(view, config.pretokenizer,
                    [&](const std::string_view piece) {
                        if (piece.empty()) return;
                        encoded.clear();
                        tokenizer.append_encoded_bytes_uncached(piece, encoded, 10);
                        for (std::size_t i = 1; i < encoded.size(); ++i) {
                            const TokenId left = encoded[i - 1];
                            const TokenId right = encoded[i];
                            if (use_dense && left < current_vocab && right < current_vocab) {
                                ++dense_counts[
                                    static_cast<std::size_t>(left) * current_vocab + right];
                            } else {
                                ++sparse_counts[pair_key(left, right)];
                            }
                            ++observed_pairs;
                        }
                    });
            }
            if (input.fail() && !input.eof()) {
                throw std::runtime_error(
                    "Failed while reading low-memory tokenizer training stream");
            }
            if (observed_pairs == 0) break;

            struct RankedPair {
                std::uint64_t count;
                PairKey pair;
            };
            const auto worse = [](const RankedPair& a, const RankedPair& b) {
                return a.count > b.count ||
                    (a.count == b.count && a.pair < b.pair);
            };
            std::vector<RankedPair> best;
            const std::size_t remaining = config.vocab_size - tokenizer.vocab_size();
            const std::size_t wanted = std::min(
                config.low_memory_merges_per_pass, remaining);
            best.reserve(wanted);

            const auto consider = [&](const PairKey pair, const std::uint64_t count) {
                if (count < config.min_pair_frequency ||
                    tokenizer.merge_lookup_.contains(pair)) {
                    return;
                }
                if (best.size() < wanted) {
                    best.push_back({.count = count, .pair = pair});
                    std::ranges::push_heap(best.begin(), best.end(), worse);
                    return;
                }
                const RankedPair candidate{.count = count, .pair = pair};
                const RankedPair& minimum = best.front();
                if (candidate.count > minimum.count ||
                    (candidate.count == minimum.count && candidate.pair < minimum.pair)) {
                    std::ranges::pop_heap(best.begin(), best.end(), worse);
                    best.back() = candidate;
                    std::ranges::push_heap(best.begin(), best.end(), worse);
                }
            };

            if (use_dense) {
                for (TokenId left = 0; left < current_vocab; ++left) {
                    const std::size_t row =
                        static_cast<std::size_t>(left) * current_vocab;
                    for (TokenId right = 0; right < current_vocab; ++right) {
                        const std::uint64_t count = dense_counts[row + right];
                        if (count != 0) consider(pair_key(left, right), count);
                    }
                }
            }
            for (const auto& [pair, count] : sparse_counts) {
                consider(pair, count);
            }
            if (best.empty()) break;

            std::ranges::sort(best, [](const RankedPair& a, const RankedPair& b) {
                return a.count != b.count ? a.count > b.count : a.pair < b.pair;
            });
            std::size_t added = 0;
            for (const RankedPair& candidate : best) {
                if (tokenizer.vocab_size() >= config.vocab_size) break;
                if (tokenizer.merge_lookup_.contains(candidate.pair)) continue;
                tokenizer.add_merge(
                    pair_left(candidate.pair), pair_right(candidate.pair));
                ++added;
                progress.update(tokenizer.vocab_size() - minimum_vocab);
            }
            if (added == 0) break;
            tokenizer.rebuild_fast_merge_lookup();
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
        if (config.mode == TrainerMode::LowMemoryStreaming) {
            return train_low_memory(input, config, block_size);
        }
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
        std::vector<TokenId> result;
        result.reserve(text.size());

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
        constexpr unsigned batch_dense_bits = 10;
        const unsigned dense_limit_bits =
            std::min(dense_merge_bits_, batch_dense_bits);

        std::size_t smallest = texts.front().size();
        std::size_t largest = smallest;
        for (const std::string_view text : texts.subspan(1)) {
            smallest = std::min(smallest, text.size());
            largest = std::max(largest, text.size());
        }

        const bool use_size_order =
            texts.size() >= workers * 2 &&
            (smallest == 0 ? largest != 0 : largest / smallest >= 2);
        std::vector<std::size_t> jobs;
        if (use_size_order) {
            jobs.resize(texts.size());
            std::iota(jobs.begin(), jobs.end(), 0);
            std::ranges::sort(jobs.begin(), jobs.end(),
                [&texts](const std::size_t a, const std::size_t b) {
                    return texts[a].size() > texts[b].size();
                });
        }

        std::atomic_size_t next_job{0};
        const auto worker = [this, &texts, &result, &options, &jobs,
                             &next_job, dense_limit_bits, use_size_order] {
            const std::size_t chunk_size = use_size_order ? 1 : 8;
            while (true) {
                const std::size_t begin =
                    next_job.fetch_add(chunk_size, std::memory_order_relaxed);
                if (begin >= texts.size()) break;
                const std::size_t end = std::min(begin + chunk_size, texts.size());
                for (std::size_t job = begin; job < end; ++job) {
                    const std::size_t index = use_size_order ? jobs[job] : job;
                    result[index] = encode_with_dense_limit(
                        texts[index], options, dense_limit_bits);
                }
            }
        };

        if (workers == 1) {
            worker();
            return result;
        }

        std::vector<std::thread> threads;
        threads.reserve(workers);
        for (std::size_t i = 0; i < workers; ++i) threads.emplace_back(worker);
        for (auto& thread : threads) thread.join();
        return result;
    }

    EncodedBatch BpeTokenizer::encode_batch_flat(
        const std::span<const std::string_view> texts,
        const std::size_t worker_count,
        const EncodeOptions& options) const {
        EncodedBatch flat;
        flat.offsets.resize(texts.size() + 1, 0);
        if (texts.empty()) return flat;

        auto per_document = encode_batch(texts, worker_count, options);
        for (std::size_t i = 0; i < per_document.size(); ++i) {
            flat.offsets[i + 1] = flat.offsets[i] + per_document[i].size();
        }
        flat.tokens.resize(flat.offsets.back());

        const std::size_t workers = worker_count_for(worker_count, texts.size());
        std::atomic_size_t next_document{0};
        const auto gather = [&] {
            while (true) {
                const std::size_t i = next_document.fetch_add(1, std::memory_order_relaxed);
                if (i >= per_document.size()) break;
                std::ranges::copy(per_document[i].begin(), per_document[i].end(),
                          flat.tokens.begin() + static_cast<std::ptrdiff_t>(flat.offsets[i]));
                std::vector<TokenId>().swap(per_document[i]);
            }
        };
        if (workers == 1) {
            gather();
        } else {
            std::vector<std::thread> threads;
            threads.reserve(workers);
            for (std::size_t i = 0; i < workers; ++i) threads.emplace_back(gather);
            for (auto& thread : threads) thread.join();
        }
        return flat;
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
                    encode_piece_cached(piece, output, options, dense_limit_bits);
                });
            return;
        }

        encode_piece_cached(text, output, options, dense_limit_bits);
    }

    void BpeTokenizer::encode_piece_cached(
        const std::string_view piece,
        std::vector<TokenId>& output,
        const EncodeOptions& options,
        const unsigned dense_limit_bits) const {
        if (piece.empty()) return;
        if (options.cache_entries == 0 || options.cache_max_input_bytes == 0 ||
            piece.size() > options.cache_max_input_bytes) {
            append_encoded_bytes_uncached(piece, output, dense_limit_bits);
            return;
        }

        constexpr std::size_t inline_tokens = 4;
        struct ShortEntry {
            std::uint64_t key_low = 0;
            std::uint64_t key_high_and_count = 0;
            std::array<TokenId, inline_tokens> tokens{};
        };
        static_assert(sizeof(ShortEntry) == 32);

        struct Entry {
            std::uint64_t hash = 0;
            std::string key;
            std::array<TokenId, inline_tokens> tokens{};
            std::vector<TokenId> spill;
            std::uint32_t token_count = 0;
            bool occupied = false;

            void emit(std::vector<TokenId>& destination) const {
                const std::size_t count = token_count;
                destination.reserve(destination.size() + count);
                if (count <= inline_tokens) {
                    destination.insert(destination.end(), tokens.begin(), tokens.begin() + count);
                } else {
                    destination.insert(destination.end(), spill.begin(), spill.end());
                }
            }

            void assign(const std::uint64_t new_hash,
                        const std::string_view new_key,
                        const std::span<const TokenId> encoded) {
                hash = new_hash;
                key.assign(new_key);
                token_count = static_cast<std::uint32_t>(encoded.size());
                if (encoded.size() <= inline_tokens) {
                    std::ranges::copy(encoded.begin(), encoded.end(), tokens.begin());
                    spill.clear();
                } else {
                    spill.assign(encoded.begin(), encoded.end());
                }
                occupied = true;
            }
        };

        struct ThreadCache {
            const BpeTokenizer* owner = nullptr;
            std::uint64_t identity = 0;
            std::size_t merge_count = 0;
            std::size_t capacity = 0;
            std::size_t short_mask = 0;
            std::size_t long_mask = 0;
            std::vector<ShortEntry> short_table;
            std::vector<Entry> long_table;
        };
        thread_local ThreadCache cache;

        const std::size_t requested = std::max<std::size_t>(16, options.cache_entries);
        const std::size_t capacity = std::bit_ceil(requested * 2);
        if (cache.owner != this || cache.identity != cache_identity_ ||
            cache.merge_count != merges_.size() || cache.capacity != capacity) {
            cache.owner = this;
            cache.identity = cache_identity_;
            cache.merge_count = merges_.size();
            cache.capacity = capacity;
            cache.short_mask = capacity - 1;
            cache.short_table.clear();
            cache.short_table.resize(capacity);

            const std::size_t long_capacity = std::bit_ceil(
                std::max<std::size_t>(16, requested / 8) * 2);
            cache.long_mask = long_capacity - 1;
            cache.long_table.clear();
            cache.long_table.resize(long_capacity);
        }

        ShortEntry* short_victim = nullptr;
        std::uint64_t short_key_low = 0;
        std::uint64_t short_key_high = 0;
        if (piece.size() <= 15) {
            const std::size_t low_size = std::min<std::size_t>(8, piece.size());
            std::memcpy(&short_key_low, piece.data(), low_size);
            if (piece.size() > 8) {
                std::memcpy(
                    &short_key_high, piece.data() + 8, piece.size() - 8);
            }
            short_key_high |= piece.size() << 56U;

            std::uint64_t packed_hash =
                short_key_low ^ std::rotl(short_key_high, 23);
            packed_hash *= 0x9E37'79B9'7F4A'7C15ULL;
            std::size_t short_index =
                packed_hash & cache.short_mask;
            constexpr std::uint64_t key_mask = (std::uint64_t{1} << 60U) - 1;
            constexpr std::size_t short_max_probe = 8;
            for (std::size_t probe = 0; probe < short_max_probe; ++probe) {
                ShortEntry& entry = cache.short_table[short_index];
                const std::uint64_t stored_key_high =
                    entry.key_high_and_count & key_mask;
                if (entry.key_low == short_key_low &&
                    stored_key_high == short_key_high) {
                    const std::size_t count = entry.key_high_and_count >> 60U;
                    output.insert(
                        output.end(), entry.tokens.begin(),
                        entry.tokens.begin() + static_cast<std::ptrdiff_t>(count));
                    return;
                }
                if (stored_key_high == 0) {
                    short_victim = &entry;
                    break;
                }
                if (short_victim == nullptr) short_victim = &entry;
                short_index = (short_index + 1) & cache.short_mask;
            }
        }

        std::uint64_t hash = 1469598103934665603ULL;
        if (piece.size() > 15) {
            for (const unsigned char byte : piece) {
                hash ^= byte;
                hash *= 1099511628211ULL;
            }
            hash ^= piece.size() *
                0x9E3779B97F4A7C15ULL;
        }

        std::size_t index = hash & cache.long_mask;
        Entry* victim = nullptr;
        if (piece.size() > 15) {
            for (std::size_t probe = 0; probe < 8; ++probe) {
                Entry& entry = cache.long_table[index];
                if (!entry.occupied) {
                    victim = &entry;
                    break;
                }
                if (entry.hash == hash && entry.key == piece) {
                    entry.emit(output);
                    return;
                }
                if (victim == nullptr) victim = &entry;
                index = (index + 1) & cache.long_mask;
            }
        }

        thread_local std::vector<TokenId> encoded;
        encoded.clear();
        encoded.reserve(piece.size());
        append_encoded_bytes_uncached(piece, encoded, dense_limit_bits);
        output.insert(output.end(), encoded.begin(), encoded.end());

        if (short_victim != nullptr && encoded.size() <= inline_tokens) {
            short_victim->key_low = short_key_low;
            short_victim->key_high_and_count =
                short_key_high |
                (encoded.size() << 60U);
            std::ranges::copy(
                encoded.begin(), encoded.end(), short_victim->tokens.begin());
        } else if (victim != nullptr) {
            victim->assign(hash, piece, encoded);
        }
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
                auto index = key * 0x9E37'79B9'7F4A'7C15ULL >> packed_merge_shift_;
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
                        std::ranges::push_heap(
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
                    std::ranges::pop_heap(
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
            encode_with_links.operator()<std::uint32_t>();
        } else {
            encode_with_links.operator()<std::size_t>();
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

void BpeTokenizer::save(
    const std::filesystem::path& path,
    const TokenizerFormat format) const {
    switch (format) {
        case TokenizerFormat::HuggingFaceJson:
            save_huggingface_json(path);
            return;
        case TokenizerFormat::Binary:
            save_binary(path);
            return;
        case TokenizerFormat::Auto:
            throw std::invalid_argument(
                "TokenizerFormat::Auto is valid only when loading");
    }
    throw std::invalid_argument("Unknown tokenizer output format");
}

void BpeTokenizer::save_binary(const std::filesystem::path& path) const {
    std::ofstream output(path, std::ios::binary);
    if (!output) {
        throw std::runtime_error("Cannot open tokenizer output file");
    }

    constexpr std::array<char, 8> magic = {'B', 'P', 'E', 'T', 'O', 'K', '2', '\0'};
    if (special_tokens_.size() > std::numeric_limits<std::uint32_t>::max() ||
        merges_.size() > std::numeric_limits<std::uint32_t>::max()) {
        throw std::length_error("Tokenizer is too large to serialize");
    }
    output.write(magic.data(), magic.size());
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

    for (const auto&[left, right, merged] : merges_) {
        write_u32(output, left);
        write_u32(output, right);
        write_u32(output, merged);
    }

    if (!output) {
        throw std::runtime_error("Cannot write tokenizer file");
    }
}

void BpeTokenizer::save_huggingface_json(
    const std::filesystem::path& path) const {
    if (pretokenizer_mode_ != PretokenizerMode::GptLike) {
        throw std::runtime_error(
            "Hugging Face export currently supports only GptLike pretokenization");
    }

    std::array<std::uint32_t, 256> byte_to_codepoint{};
    std::uint32_t replacement = 0;
    for (std::uint32_t byte = 0; byte < 256; ++byte) {
        const bool direct =
            (byte >= 33 && byte <= 126) ||
            (byte >= 161 && byte <= 172) ||
            (byte >= 174);
        byte_to_codepoint[byte] =
            direct ? byte : 256 + replacement++;
    }

    const auto append_utf8 = [](std::string& output, const std::uint32_t cp) {
        if (cp <= 0x7F) {
            output.push_back(static_cast<char>(cp));
        } else if (cp <= 0x7FF) {
            output.push_back(static_cast<char>(0xC0 | (cp >> 6)));
            output.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
        } else {
            output.push_back(static_cast<char>(0xE0 | (cp >> 12)));
            output.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
            output.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
        }
    };
    const auto bytelevel_token =
        [&byte_to_codepoint, &append_utf8](
            const std::vector<std::uint8_t>& bytes) {
            std::string result;
            result.reserve(bytes.size() * 2);
            for (const std::uint8_t byte : bytes) {
                append_utf8(result, byte_to_codepoint[byte]);
            }
            return result;
        };
    const auto json_string = [](std::ostream& output, const std::string_view text) {
        static constexpr char hex[] = "0123456789abcdef";
        output.put('"');
        for (const unsigned char byte : text) {
            switch (byte) {
                case '"': output << "\\\""; break;
                case '\\': output << "\\\\"; break;
                case '\b': output << "\\b"; break;
                case '\f': output << "\\f"; break;
                case '\n': output << "\\n"; break;
                case '\r': output << "\\r"; break;
                case '\t': output << "\\t"; break;
                default:
                    if (byte < 0x20) {
                        output << "\\u00" << hex[byte >> 4] << hex[byte & 0x0F];
                    } else {
                        output.put(static_cast<char>(byte));
                    }
            }
        }
        output.put('"');
    };

    std::ofstream output(path, std::ios::binary);
    if (!output) {
        throw std::runtime_error(
            "Cannot open Hugging Face tokenizer output file");
    }

    output << "{\n"
              "  \"version\": \"1.0\",\n"
              "  \"truncation\": null,\n"
              "  \"padding\": null,\n"
              "  \"added_tokens\": [";
    for (std::size_t index = 0; index < special_tokens_.size(); ++index) {
        if (index != 0) output << ',';
        output << "\n    {\"id\":" << (kByteVocabularySize + index)
               << ",\"content\":";
        json_string(output, special_tokens_[index]);
        output << ",\"single_word\":false,\"lstrip\":false,\"rstrip\":false,"
                  "\"normalized\":false,\"special\":true}";
    }
    if (!special_tokens_.empty()) output << '\n';


    output << "  ],\n"
              "  \"normalizer\": null,\n"
              "  \"pre_tokenizer\": {\n"
              "    \"type\":\"Sequence\",\n"
              "    \"pretokenizers\":[\n"
              "      {\"type\":\"Split\",\"pattern\":{\"Regex\":"
              "\"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\\\r\\\\n\\\\p{L}\\\\p{N}]?"
              "\\\\p{L}+|\\\\p{N}{1,3}| ?[^\\\\s\\\\p{L}\\\\p{N}]+"
              "[\\\\r\\\\n]*|\\\\s*[\\\\r\\\\n]|\\\\s+(?!\\\\S)|\\\\s+\"},"
              "\"behavior\":\"Isolated\",\"invert\":false},\n"
              "      {\"type\":\"ByteLevel\",\"add_prefix_space\":false,"
              "\"trim_offsets\":true,\"use_regex\":false}\n"
              "    ]\n"
              "  },\n"
              "  \"post_processor\": null,\n"
              "  \"decoder\": {\"type\":\"ByteLevel\",\"add_prefix_space\":false,"
              "\"trim_offsets\":true,\"use_regex\":true},\n"
              "  \"model\": {\n"
              "    \"type\":\"BPE\",\"dropout\":null,\"unk_token\":null,"
              "\"continuing_subword_prefix\":\"\",\"end_of_word_suffix\":\"\","
              "\"fuse_unk\":false,\"byte_fallback\":false,\"ignore_merges\":false,\n"
              "    \"vocab\": {";

    bool first = true;
    std::unordered_map<std::string, TokenId> exported_tokens;
    exported_tokens.reserve(token_bytes_.size());
    for (TokenId id = 0; id < token_bytes_.size(); ++id) {
        if (is_special(id)) continue;
        const std::string token = bytelevel_token(token_bytes_[id]);
        if (!exported_tokens.emplace(token, id).second) {
            throw std::runtime_error(
                "Cannot export duplicate token byte sequence to tokenizer.json");
        }
        if (!first) output << ',';
        output << "\n      ";
        json_string(output, token);
        output << ':' << id;
        first = false;
    }
    if (!first) output << '\n';
    output << "    },\n"
              "    \"merges\": [";
    for (std::size_t index = 0; index < merges_.size(); ++index) {
        const MergeRule& merge = merges_[index];
        if (index != 0) output << ',';
        output << "\n      [";
        json_string(output, bytelevel_token(token_bytes_[merge.left]));
        output << ',';
        json_string(output, bytelevel_token(token_bytes_[merge.right]));
        output << ']';
    }
    if (!merges_.empty()) output << '\n';
    output << "    ]\n"
              "  }\n"
              "}\n";

    if (!output) {
        throw std::runtime_error("Cannot write Hugging Face tokenizer file");
    }
}

BpeTokenizer BpeTokenizer::load(
    const std::filesystem::path& path,
    TokenizerFormat format) {
    if (format == TokenizerFormat::Auto) {
        std::ifstream probe(path, std::ios::binary);
        if (!probe) {
            throw std::runtime_error("Cannot open tokenizer model file");
        }
        char first = 0;
        do {
            if (!probe.get(first)) {
                throw std::runtime_error("Tokenizer model file is empty");
            }
        } while (first == ' ' || first == '\t' || first == '\r' || first == '\n');
        format = first == '{'
                     ? TokenizerFormat::HuggingFaceJson
                     : TokenizerFormat::Binary;
    }
    switch (format) {
        case TokenizerFormat::HuggingFaceJson:
            return load_huggingface_json(path);
        case TokenizerFormat::Binary:
            return load_binary(path);
        case TokenizerFormat::Auto:
            break;
    }
    throw std::invalid_argument("Unknown tokenizer input format");
}

BpeTokenizer BpeTokenizer::load_binary(const std::filesystem::path& path) {
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
    auto pretokenizer = PretokenizerMode::None;
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
        input.read(token.data(), size);
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

BpeTokenizer BpeTokenizer::load_huggingface_json(
    const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw std::runtime_error("Cannot open Hugging Face tokenizer model file");
    }
    const std::string json{
        std::istreambuf_iterator<char>(input),
        std::istreambuf_iterator<char>()};
    const JsonValue root = JsonParser(json).parse();
    const JsonValue& model = root.member("model");
    const JsonValue& type = model.member("type");
    if (type.type != JsonValue::Type::String || type.string != "BPE") {
        throw std::runtime_error("tokenizer.json model is not BPE");
    }

    const JsonValue& added = root.member("added_tokens");
    if (added.type != JsonValue::Type::Array) {
        throw std::runtime_error("tokenizer.json added_tokens must be an array");
    }
    std::vector<std::pair<TokenId, std::string>> specials;
    for (const JsonValue& entry : added.array) {
        const JsonValue& special = entry.member("special");
        if (special.type != JsonValue::Type::Boolean || !special.boolean) continue;
        const JsonValue& id = entry.member("id");
        const JsonValue& content = entry.member("content");
        if (id.type != JsonValue::Type::Number ||
            id.number > std::numeric_limits<TokenId>::max() ||
            content.type != JsonValue::Type::String) {
            throw std::runtime_error("Invalid special token in tokenizer.json");
        }
        specials.emplace_back(
            static_cast<TokenId>(id.number), content.string);
    }
    std::ranges::sort(specials);
    std::vector<std::string> special_tokens;
    special_tokens.reserve(specials.size());
    for (std::size_t index = 0; index < specials.size(); ++index) {
        if (specials[index].first != kByteVocabularySize + index) {
            throw std::runtime_error(
                "tokenizer.json special token IDs must immediately follow "
                "the 256 byte tokens");
        }
        special_tokens.push_back(std::move(specials[index].second));
    }

    const JsonValue& vocab = model.member("vocab");
    if (vocab.type != JsonValue::Type::Object) {
        throw std::runtime_error("tokenizer.json vocab must be an object");
    }
    std::unordered_map<std::string, TokenId> token_ids;
    token_ids.reserve(vocab.object.size());
    for (const auto& [token, id] : vocab.object) {
        if (id.type != JsonValue::Type::Number ||
            id.number > std::numeric_limits<TokenId>::max()) {
            throw std::runtime_error("Invalid vocabulary ID in tokenizer.json");
        }
        token_ids.emplace(token, static_cast<TokenId>(id.number));
    }
    const auto alphabet = bytelevel_alphabet();
    for (TokenId byte = 0; byte < kByteVocabularySize; ++byte) {
        const auto found = token_ids.find(alphabet[byte]);
        if (found == token_ids.end() || found->second != byte) {
            throw std::runtime_error(
                "tokenizer.json does not use CGPT-compatible byte token IDs");
        }
    }

    BpeTokenizer tokenizer(std::move(special_tokens), PretokenizerMode::GptLike);
    const JsonValue& merges = model.member("merges");
    if (merges.type != JsonValue::Type::Array) {
        throw std::runtime_error("tokenizer.json merges must be an array");
    }
    tokenizer.merges_.reserve(merges.array.size());
    tokenizer.token_bytes_.reserve(tokenizer.vocab_size() + merges.array.size());
    tokenizer.merge_lookup_.reserve(merges.array.size());
    for (const JsonValue& entry : merges.array) {
        if (entry.type != JsonValue::Type::Array || entry.array.size() != 2 ||
            entry.array[0].type != JsonValue::Type::String ||
            entry.array[1].type != JsonValue::Type::String) {
            throw std::runtime_error(
                "tokenizer.json merge must be a two-string array");
        }
        const auto left = token_ids.find(entry.array[0].string);
        const auto right = token_ids.find(entry.array[1].string);
        const auto merged =
            token_ids.find(entry.array[0].string + entry.array[1].string);
        if (left == token_ids.end() || right == token_ids.end() ||
            merged == token_ids.end()) {
            throw std::runtime_error(
                "tokenizer.json merge references an unknown token");
        }
        if (merged->second != tokenizer.vocab_size()) {
            throw std::runtime_error(
                "tokenizer.json merge IDs are not contiguous in rank order");
        }
        tokenizer.add_merge(left->second, right->second);
    }
    tokenizer.rebuild_fast_merge_lookup();
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
    merges_.push_back({.left = left, .right = right, .merged = merged});
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

    const auto packed_merge_count = static_cast<std::size_t>(
        std::ranges::count_if(merges_, [](const MergeRule& merge) {
            return ((merge.left | merge.right) >> dense_bits) != 0;
        }));
    if (packed_merge_count == 0) {
        return;
    }

    const std::size_t slot_count = std::max<std::size_t>(
        64, std::bit_ceil(packed_merge_count * 2));
    packed_merge_lookup_.assign(slot_count, empty);
    packed_merge_mask_ = slot_count - 1;
    packed_merge_shift_ = 64U - std::countr_zero(slot_count);

    for (const MergeRule& merge : merges_) {
        if (((merge.left | merge.right) >> dense_bits) == 0) {
            continue;
        }
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
        auto index = key * 0x9E37'79B9'7F4A'7C15ULL >> packed_merge_shift_;
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
