#include "data/dataset_loader.h"

#include <iostream>
#include <stdexcept>
#include <vector>

namespace {
void require(const bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
}

int main() {
    try {
        data::TokenDataset dataset({0, 1, 2, 3, 4, 5, 6, 7, 8, 9});
        data::DataLoaderConfig config;
        config.batch_size = 2;
        config.sequence_length = 3;
        config.shuffle = false;
        config.drop_last = false;
        data::DatasetLoader loader(std::move(dataset), config);

        require(loader.sample_count() == 3, "Unexpected sample count");
        require(loader.batch_count() == 2, "Unexpected batch count");
        data::Batch batch;
        require(loader.next(batch), "First batch missing");
        require(batch.batch_size == 2 && batch.sequence_length == 3, "Wrong first batch shape");
        require(batch.input_ids == std::vector<bpe::TokenId>({0, 1, 2, 3, 4, 5}), "Wrong first inputs");
        require(batch.target_ids == std::vector<bpe::TokenId>({1, 2, 3, 4, 5, 6}), "Wrong first targets");
        require(loader.next(batch) && batch.batch_size == 1, "Partial batch missing or invalid");
        require(batch.input_ids == std::vector<bpe::TokenId>({6, 7, 8}), "Wrong partial inputs");
        require(!loader.next(batch), "Batch produced past epoch end");

        loader.reset();
        require(loader.epoch() == 1 && loader.next(batch), "Reset did not start an epoch");
        config.drop_last = true;
        data::DatasetLoader dropped(data::TokenDataset({0, 1, 2, 3, 4, 5, 6, 7}), config);
        require(dropped.batch_count() == 1 && dropped.next(batch) && !dropped.next(batch),
                "drop_last did not discard the partial batch");
        std::cout << "Dataset loader tests passed.\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Dataset loader test failed: " << error.what() << '\n';
        return 1;
    }
}
