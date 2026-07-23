#include "tokenizer/bpe_tokenizer.h"

#include <cassert>
#include <filesystem>
#include <iostream>

int main() {
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
    const std::string text = "Zażółć kota 🦀";
    const std::vector<bpe::TokenId> ids = tokenizer.encode(text);

    assert(tokenizer.decode(ids) == text);
    assert(tokenizer.vocab_size() > bpe::BpeTokenizer::kByteVocabularySize);

    const std::filesystem::path model_path = "bpe_tokenizer_test.bin";
    tokenizer.save(model_path);
    const bpe::BpeTokenizer loaded = bpe::BpeTokenizer::load(model_path);

    assert(loaded.decode(loaded.encode(text)) == text);
    std::filesystem::remove(model_path);

    std::cout << "BPE tokenizer test passed.\n";
}
