#include "tokenizer/bpe_tokenizer.h"

#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <iterator>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

int main(int argc, char** argv) {
    constexpr const char* input_path = "data/tokenizer_100mb.txt";

    std::ifstream input(input_path, std::ios::binary);
    if (!input) {
        std::cerr << "Cannot open " << input_path << '\n';
        std::cerr << "Generate it first with: python scripts/download_100mb.py\n";
        return 1;
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
    config.vocab_size = 2048;
    const std::size_t worker_count = std::max<std::size_t>(
        1, std::thread::hardware_concurrency());
    config.worker_count = worker_count;

    // Keep 100 independent blocks per worker. BpeTokenizer::train() processes
    // these documents in parallel using config.worker_count workers.
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

    const double train_seconds =
        std::chrono::duration<double>(train_end - train_start).count();
    const double encode_seconds =
        std::chrono::duration<double>(encode_end - encode_start).count();

    std::cout << "Vocabulary size: " << tokenizer.vocab_size() << '\n';
    std::cout << "Token count: " << ids.size() << '\n';
    std::cout << "Training time: " << train_seconds << " s\n";
    std::cout << "Encoding time: " << encode_seconds << " s\n";

    if (encode_seconds > 0.0) {
        std::cout << "Encoding throughput: "
                  << text.size() / encode_seconds / (1024.0 * 1024.0)
                  << " MiB/s\n";
    }

    std::cout << "100 MiB tokenizer test passed.\n";
    return 0;
}
