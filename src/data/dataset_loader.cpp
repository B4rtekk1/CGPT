#include "data/dataset_loader.h"

#include "core/cuda_check.h"

#include <algorithm>
#include <fstream>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>

namespace data {
namespace {
void validate_config(const DataLoaderConfig& config) {
    if (config.batch_size == 0 || config.sequence_length == 0) {
        throw std::invalid_argument("DatasetLoader batch_size and sequence_length must be positive");
    }
    if (config.sequence_length > std::numeric_limits<std::size_t>::max() / config.batch_size) {
        throw std::overflow_error("DatasetLoader batch token count overflows size_t");
    }
}
} // namespace

TokenDataset::TokenDataset(std::vector<bpe::TokenId> tokens) : tokens_(std::move(tokens)) {}

TokenDataset TokenDataset::load(const std::filesystem::path& path) {
    std::error_code error;
    const std::uintmax_t bytes = std::filesystem::file_size(path, error);
    if (error) throw std::runtime_error("Unable to stat token dataset: " + path.string());
    if (bytes % sizeof(bpe::TokenId) != 0) {
        throw std::invalid_argument("Token dataset byte count is not divisible by uint32");
    }
    if (bytes / sizeof(bpe::TokenId) > std::numeric_limits<std::size_t>::max()) {
        throw std::overflow_error("Token dataset is too large for this process");
    }
    if (bytes > static_cast<std::uintmax_t>(std::numeric_limits<std::streamsize>::max())) {
        throw std::overflow_error("Token dataset is too large for one read operation");
    }

    std::vector<bpe::TokenId> tokens(static_cast<std::size_t>(bytes / sizeof(bpe::TokenId)));
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("Unable to open token dataset: " + path.string());
    if (!tokens.empty()) {
        input.read(reinterpret_cast<char*>(tokens.data()), static_cast<std::streamsize>(bytes));
        if (!input) throw std::runtime_error("Unable to read complete token dataset: " + path.string());
    }
    return TokenDataset(std::move(tokens));
}

std::span<const bpe::TokenId> TokenDataset::tokens() const noexcept { return tokens_; }
std::size_t TokenDataset::size() const noexcept { return tokens_.size(); }
bool TokenDataset::empty() const noexcept { return tokens_.empty(); }

std::size_t Batch::token_count() const noexcept { return batch_size * sequence_length; }
bool Batch::empty() const noexcept { return token_count() == 0; }

void upload_batch(const Batch& batch, DeviceBatch& destination, const cudaStream_t stream) {
    const std::size_t count = batch.token_count();
    if (batch.input_ids.size() != count || batch.target_ids.size() != count) {
        throw std::invalid_argument("Batch buffers do not match its declared shape");
    }
    const std::size_t bytes = count * sizeof(bpe::TokenId);
    if (destination.input_ids.bytes() < bytes) destination.input_ids.allocate(bytes);
    if (destination.target_ids.bytes() < bytes) destination.target_ids.allocate(bytes);
    if (bytes != 0) {
        CUDA_CHECK(cudaMemcpyAsync(destination.input_ids.data(), batch.input_ids.data(), bytes,
                                   cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(destination.target_ids.data(), batch.target_ids.data(), bytes,
                                   cudaMemcpyHostToDevice, stream));
    }
    destination.batch_size = batch.batch_size;
    destination.sequence_length = batch.sequence_length;
}

DatasetLoader::DatasetLoader(std::shared_ptr<const TokenDataset> dataset, const DataLoaderConfig &config)
    : dataset_(std::move(dataset)), config_(config) {
    initialize();
}

DatasetLoader::DatasetLoader(TokenDataset dataset, const DataLoaderConfig &config)
    : DatasetLoader(std::make_shared<TokenDataset>(std::move(dataset)), config) {}

void DatasetLoader::initialize() {
    if (!dataset_) throw std::invalid_argument("DatasetLoader dataset cannot be null");
    validate_config(config_);
    const std::size_t samples = dataset_->size() > 0
        ? (dataset_->size() - 1) / config_.sequence_length : 0;
    sample_indices_.resize(samples);
    std::iota(sample_indices_.begin(), sample_indices_.end(), 0);
    if (config_.shuffle) shuffle_indices();
}

bool DatasetLoader::has_next() const noexcept {
    return next_sample_ < sample_indices_.size() &&
        (!config_.drop_last || sample_indices_.size() - next_sample_ >= config_.batch_size);
}

std::size_t DatasetLoader::sample_count() const noexcept { return sample_indices_.size(); }
std::size_t DatasetLoader::batch_count() const noexcept {
    return config_.drop_last ? sample_indices_.size() / config_.batch_size
                             : sample_indices_.size() / config_.batch_size +
                                   (sample_indices_.size() % config_.batch_size != 0);
}
std::size_t DatasetLoader::epoch() const noexcept { return epoch_; }

bool DatasetLoader::next(Batch& batch) {
    if (!has_next()) return false;
    const std::size_t current_batch = std::min(config_.batch_size, sample_indices_.size() - next_sample_);
    const std::size_t token_count = current_batch * config_.sequence_length;
    batch.input_ids.resize(token_count);
    batch.target_ids.resize(token_count);
    const auto tokens = dataset_->tokens();
    for (std::size_t item = 0; item < current_batch; ++item) {
        const std::size_t start = sample_indices_[next_sample_ + item] * config_.sequence_length;
        const std::size_t destination = item * config_.sequence_length;
        std::copy_n(tokens.data() + start, config_.sequence_length, batch.input_ids.data() + destination);
        std::copy_n(tokens.data() + start + 1, config_.sequence_length, batch.target_ids.data() + destination);
    }
    next_sample_ += current_batch;
    batch.batch_size = current_batch;
    batch.sequence_length = config_.sequence_length;
    return true;
}

std::optional<Batch> DatasetLoader::next() {
    Batch batch;
    if (!next(batch)) return std::nullopt;
    return batch;
}

void DatasetLoader::reset() {
    next_sample_ = 0;
    ++epoch_;
    if (config_.shuffle) shuffle_indices();
}

void DatasetLoader::shuffle_indices() {
    std::mt19937_64 generator(config_.seed + epoch_);
    std::ranges::shuffle(sample_indices_, generator);
}

} // namespace data
