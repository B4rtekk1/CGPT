#include "tokenizer/bpe_tokenizer.h"

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

int main(int argc, char** argv) {
    constexpr auto input_path = "data/tokenizer_100mb.txt";

    std::ifstream input(input_path, std::ios::binary);
    if (!input) {
        std::cerr << "Cannot open " << input_path << '\n';
        std::cerr << "Generate it first with: python scripts/download_100mb.py\n";
        return 77;
    }

    const std::string text{
        std::istreambuf_iterator<char>(input),
        std::istreambuf_iterator<char>()};

    if (text.empty()) {
        std::cerr << "Input file is empty\n";
        return 1;
    }

    std::cout << "Input size: "
              << static_cast<double>(text.size()) / (1024.0 * 1024.0)
              << " MiB\n";

    constexpr std::size_t default_training_size = 8 * 1024 * 1024;
    const bool full_training = argc > 1 &&
                                std::string_view(argv[1]) == "--full-training";
    const std::size_t training_size = full_training
                                          ? text.size()
                                          : std::min(text.size(), default_training_size);

    std::cout << "Training size: "
              << static_cast<double>(training_size) / (1024.0 * 1024.0)
              << " MiB\n";

    bpe::TrainerConfig config;
    config.vocab_size = 32000;
    const std::size_t worker_count = std::max<std::size_t>(
        1, std::thread::hardware_concurrency());
    config.worker_count = worker_count;

    constexpr std::size_t blocks_per_worker = 100;
    const std::size_t block_count = std::min(
        training_size, worker_count * blocks_per_worker);

    std::vector<std::string> corpus(block_count);
    std::vector<std::thread> block_workers;
    block_workers.reserve(worker_count);
    for (std::size_t worker = 0; worker < worker_count; ++worker) {
        const std::size_t begin_block = block_count * worker / worker_count;
        const std::size_t end_block = block_count * (worker + 1) / worker_count;

        block_workers.emplace_back(
            [&, begin_block, end_block] {
                for (std::size_t block = begin_block; block < end_block; ++block) {
                    const std::size_t begin = training_size * block / block_count;
                    const std::size_t end = training_size * (block + 1) / block_count;
                    corpus[block].assign(text.data() + begin, end - begin);
                }
            });
    }
    for (std::thread& worker : block_workers) {
        worker.join();
    }

    std::cout << "Training blocks: " << corpus.size()
              << " (" << blocks_per_worker << " per worker)\n";

    const auto train_start = std::chrono::steady_clock::now();
    const bpe::BpeTokenizer tokenizer = bpe::BpeTokenizer::train(corpus, config);
    const auto train_end = std::chrono::steady_clock::now();

    const auto encode_start = std::chrono::steady_clock::now();
    const std::vector<bpe::TokenId> ids = tokenizer.encode(text);
    const auto encode_end = std::chrono::steady_clock::now();

    const std::string decoded = tokenizer.decode(ids);

    if (decoded != text) {
        std::cerr << "Round-trip failure: decode(encode(text)) != text\n";
        return 1;
    }

    // GigaToken gets its file-scale throughput by handing independent
    // documents to per-thread workers and keeping the input as borrowed
    // slices. Exercise the equivalent zero-copy batch API separately from
    // the exact single-sequence benchmark above.
    const std::size_t encode_block_count = std::min(
        text.size(), worker_count * blocks_per_worker);
    std::vector<std::string_view> encode_blocks;
    encode_blocks.reserve(encode_block_count);
    for (std::size_t block = 0; block < encode_block_count; ++block) {
        const std::size_t begin = text.size() * block / encode_block_count;
        const std::size_t end = text.size() * (block + 1) / encode_block_count;
        encode_blocks.emplace_back(text.data() + begin, end - begin);
    }

    const auto batch_encode_start = std::chrono::steady_clock::now();
    const auto batch_ids = tokenizer.encode_batch(encode_blocks, worker_count);
    const auto batch_encode_end = std::chrono::steady_clock::now();
    std::size_t batch_token_count = 0;
    for (std::size_t block = 0; block < batch_ids.size(); ++block) {
        batch_token_count += batch_ids[block].size();
        if (tokenizer.decode(batch_ids[block]) != encode_blocks[block]) {
            std::cerr << "Batch round-trip failure in block " << block << '\n';
            return 1;
        }
    }

    if (tokenizer.vocab_size() <= bpe::BpeTokenizer::kByteVocabularySize ||
        tokenizer.vocab_size() > config.vocab_size) {
        std::cerr << "Unexpected vocabulary size: " << tokenizer.vocab_size() << '\n';
        return 1;
    }

    const auto model_path = std::filesystem::temp_directory_path() /
                            "cgpt_bpe_tokenizer_100mb_test.bin";
    tokenizer.save(model_path);
    const bpe::BpeTokenizer loaded = bpe::BpeTokenizer::load(model_path);
    std::filesystem::remove(model_path);
    if (loaded.decode(loaded.encode(text)) != text) {
        std::cerr << "Serialized tokenizer round-trip failure\n";
        return 1;
    }

    const double train_seconds =
        std::chrono::duration<double>(train_end - train_start).count();
    const double encode_seconds =
        std::chrono::duration<double>(encode_end - encode_start).count();
    const double batch_encode_seconds =
        std::chrono::duration<double>(
            batch_encode_end - batch_encode_start).count();

    std::cout << "Vocabulary size: " << tokenizer.vocab_size() << '\n';
    std::cout << "Token count: " << ids.size() << '\n';
    std::cout << "Compression ratio: "
              << static_cast<double>(text.size()) / static_cast<double>(ids.size())
              << " bytes/token\n";
    std::cout << "Training time: " << train_seconds << " s\n";
    std::cout << "Encoding time: " << encode_seconds << " s\n";
    std::cout << "Batch token count: " << batch_token_count << '\n';
    std::cout << "Batch encoding time: " << batch_encode_seconds << " s\n";

    if (encode_seconds > 0.0) {
        std::cout << "Encoding throughput: "
                  << text.size() / encode_seconds / (1024.0 * 1024.0)
                  << " MiB/s\n";
    }
    if (batch_encode_seconds > 0.0) {
        std::cout << "Batch encoding throughput: "
                  << text.size() / batch_encode_seconds / (1024.0 * 1024.0)
                  << " MiB/s\n";
    }

    std::cout << "100 MiB tokenizer test passed.\n";
    return 0;
}
