/**
 * @file dataset_loader.cpp
 * @brief Token-dataset loading, batching, shuffling, and CUDA upload support.
 */

#include "data/dataset_loader.h"

#include "core/cuda_check.h"

#include <algorithm>
#include <cstring>
#include <fstream>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <utility>

namespace data {
    namespace {
        /** @brief Validates dimensions that determine the number of tokens per batch. */
        void validate_config(const DataLoaderConfig &config) {
            if (config.batch_size == 0 || config.sequence_length == 0) {
                throw std::invalid_argument("DatasetLoader batch_size and sequence_length must be positive");
            }
            if (config.sequence_length > std::numeric_limits<std::size_t>::max() / config.batch_size) {
                throw std::overflow_error("DatasetLoader batch token count overflows size_t");
            }
        }
    } // namespace

    /** @brief Takes ownership of an in-memory token sequence. */
    TokenDataset::TokenDataset(std::vector<bpe::TokenId> tokens) : tokens_(std::move(tokens)) {
    }

    /**
     * @brief Loads a binary uint32 token dataset from disk.
     * @param path Path to the binary token file.
     * @return Dataset containing all decoded token IDs.
     * @throws std::runtime_error If the file cannot be inspected, opened, or read.
     * @throws std::invalid_argument If the file size is not token-aligned.
     * @throws std::overflow_error If the file is too large for the process.
     */
    TokenDataset TokenDataset::load(const std::filesystem::path &path) {
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
            input.read(reinterpret_cast<char *>(tokens.data()), static_cast<std::streamsize>(bytes));
            if (!input) throw std::runtime_error("Unable to read complete token dataset: " + path.string());
        }
        return TokenDataset(std::move(tokens));
    }

    /** @brief Returns a non-owning view of the stored token IDs. */
    std::span<const bpe::TokenId> TokenDataset::tokens() const noexcept { return tokens_; }
    /** @brief Returns the number of stored token IDs. */
    std::size_t TokenDataset::size() const noexcept { return tokens_.size(); }
    /** @brief Indicates whether the dataset contains no tokens. */
    bool TokenDataset::empty() const noexcept { return tokens_.empty(); }

    /** @brief Waits until a previous asynchronous upload no longer uses host buffers. */
    Batch::~Batch() {
        wait_until_reusable();
        if (upload_complete_ != nullptr) static_cast<void>(cudaEventDestroy(upload_complete_));
    }

    Batch::Batch(const Batch &other) {
        other.wait_until_reusable();
        input_ids = other.input_ids;
        target_ids = other.target_ids;
        batch_size = other.batch_size;
        sequence_length = other.sequence_length;
    }

    Batch &Batch::operator=(const Batch &other) {
        if (this != &other) {
            wait_until_reusable();
            other.wait_until_reusable();
            input_ids = other.input_ids;
            target_ids = other.target_ids;
            batch_size = other.batch_size;
            sequence_length = other.sequence_length;
        }
        return *this;
    }

    Batch::Batch(Batch &&other) noexcept
        : input_ids(std::move(other.input_ids)), target_ids(std::move(other.target_ids)),
          batch_size(other.batch_size), sequence_length(other.sequence_length),
          upload_complete_(std::exchange(other.upload_complete_, nullptr)) {
    }

    Batch &Batch::operator=(Batch &&other) noexcept {
        if (this != &other) {
            wait_until_reusable();
            if (upload_complete_ != nullptr) static_cast<void>(cudaEventDestroy(upload_complete_));
            input_ids = std::move(other.input_ids);
            target_ids = std::move(other.target_ids);
            batch_size = other.batch_size;
            sequence_length = other.sequence_length;
            upload_complete_ = std::exchange(other.upload_complete_, nullptr);
        }
        return *this;
    }

    /** @brief Synchronizes the upload-completion event when one exists. */
    void Batch::wait_until_reusable() const {
        if (upload_complete_ != nullptr)
            CUDA_CHECK(cudaEventSynchronize(upload_complete_));
    }

    /** @brief Records an event marking host-buffer reuse as safe. */
    void Batch::mark_upload_complete(const cudaStream_t stream) const {
        if (upload_complete_ == nullptr)
            CUDA_CHECK(cudaEventCreateWithFlags(&upload_complete_, cudaEventDisableTiming));
        CUDA_CHECK(cudaEventRecord(upload_complete_, stream));
    }

    /** @brief Returns the number of tokens in the batch. */
    std::size_t Batch::token_count() const noexcept { return batch_size * sequence_length; }
    /** @brief Indicates whether the declared batch shape contains zero tokens. */
    bool Batch::empty() const noexcept { return token_count() == 0; }

    /**
     * @brief Asynchronously uploads a host batch to device buffers.
     * @param batch Host batch whose buffers remain protected until upload completes.
     * @param destination Device-side destination buffers.
     * @param stream CUDA stream used for the copies.
     * @throws std::invalid_argument If batch buffer sizes do not match its shape.
     */
    void upload_batch(const Batch &batch, DeviceBatch &destination, const cudaStream_t stream) {
        batch.wait_until_reusable();
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
        batch.mark_upload_complete(stream);
        destination.batch_size = batch.batch_size;
        destination.sequence_length = batch.sequence_length;
    }

    /** @brief Creates a loader sharing ownership of a token dataset. */
    DatasetLoader::DatasetLoader(std::shared_ptr<const TokenDataset> dataset, const DataLoaderConfig &config)
        : dataset_(std::move(dataset)), config_(config) {
        initialize();
    }

    /** @brief Creates a loader by taking ownership of a token dataset. */
    DatasetLoader::DatasetLoader(TokenDataset dataset, const DataLoaderConfig &config)
        : DatasetLoader(std::make_shared<TokenDataset>(std::move(dataset)), config) {
    }

    /** @brief Validates configuration and initializes epoch sampling state. */
    void DatasetLoader::initialize() {
        if (!dataset_) throw std::invalid_argument("DatasetLoader dataset cannot be null");
        if (config_.sample_stride == 0) config_.sample_stride = config_.sequence_length;
        validate_config(config_);
        sample_count_ = dataset_->size() > 0
                            ? (dataset_->size() - 1) / config_.sample_stride
                            : 0;
        sequential_ = !config_.shuffle;
        if (sequential_) {
            // The common validation/evaluation path does not need an index array:
            // sample i always starts at i * sequence_length.
            sample_indices_.clear();
        } else {
            sample_indices_.resize(sample_count_);
            std::iota(sample_indices_.begin(), sample_indices_.end(), 0);
            shuffle_indices();
        }
    }

    /** @brief Indicates whether another complete or partial batch is available. */
    bool DatasetLoader::has_next() const noexcept {
        return next_sample_ < sample_count_ &&
               (!config_.drop_last || sample_count_ - next_sample_ >= config_.batch_size);
    }

    /** @brief Returns the number of sequence samples in the dataset. */
    std::size_t DatasetLoader::sample_count() const noexcept { return sample_count_; }
    /** @brief Returns the number of batches in the current epoch. */
    std::size_t DatasetLoader::batch_count() const noexcept {
        return config_.drop_last
                   ? sample_count_ / config_.batch_size
                   : sample_count_ / config_.batch_size +
                     (sample_count_ % config_.batch_size != 0);
    }

    /** @brief Returns the current zero-based epoch counter. */
    std::size_t DatasetLoader::epoch() const noexcept { return epoch_; }

    /**
     * @brief Fills a reusable batch with the next sequence samples.
     * @param batch Destination batch, resized as necessary.
     * @return `true` when a batch was produced, otherwise `false` at epoch end.
     */
    bool DatasetLoader::next(Batch &batch) {
        if (!has_next()) return false;
        batch.wait_until_reusable();
        const std::size_t current_batch = std::min(config_.batch_size, sample_count_ - next_sample_);
        const std::size_t token_count = current_batch * config_.sequence_length;
        // Avoid touching the buffers when the steady-state batch shape is unchanged.
        if (batch.input_ids.size() != token_count) batch.input_ids.resize(token_count);
        if (batch.target_ids.size() != token_count) batch.target_ids.resize(token_count);
        const auto tokens = dataset_->tokens();

        if (sequential_) {
                const std::size_t start = next_sample_ * config_.sample_stride;
            // Both arrays are trivially copyable and non-overlapping.  This is the
            // hot path for non-shuffled loaders and lets the standard library use
            // the platform's bulk-copy implementation.
            std::memcpy(batch.input_ids.data(), tokens.data() + start,
                        token_count * sizeof(bpe::TokenId));
            std::memcpy(batch.target_ids.data(), tokens.data() + start + 1,
                        token_count * sizeof(bpe::TokenId));
        } else {
            for (std::size_t item = 0; item < current_batch; ++item) {
                const std::size_t start = sample_indices_[next_sample_ + item] * config_.sample_stride;
                const std::size_t destination = item * config_.sequence_length;
                const std::size_t bytes = config_.sequence_length * sizeof(bpe::TokenId);
                std::memcpy(batch.input_ids.data() + destination, tokens.data() + start, bytes);
                std::memcpy(batch.target_ids.data() + destination, tokens.data() + start + 1, bytes);
            }
        }
        next_sample_ += current_batch;
        batch.batch_size = current_batch;
        batch.sequence_length = config_.sequence_length;
        return true;
    }

    /** @brief Returns the next batch by value, if one is available. */
    std::optional<Batch> DatasetLoader::next() {
        Batch batch;
        if (!next(batch)) return std::nullopt;
        return batch;
    }

    /** @brief Starts the next epoch and reshuffles indices when configured. */
    void DatasetLoader::reset() {
        next_sample_ = 0;
        ++epoch_;
        if (!sequential_) shuffle_indices();
    }

    /** @brief Deterministically shuffles sample indices using seed plus epoch. */
    void DatasetLoader::shuffle_indices() {
        std::mt19937_64 generator(config_.seed + epoch_);
        std::ranges::shuffle(sample_indices_, generator);
    }
} // namespace data
