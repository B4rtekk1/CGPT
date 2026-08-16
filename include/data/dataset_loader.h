#pragma once

#include "core/device_buffer.h"
#include "tokenizer/bpe_tokenizer.h"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <span>
#include <vector>

namespace data {

/** Immutable, contiguous token corpus used by the training data loader. */
class TokenDataset {
public:
    TokenDataset() = default;
    explicit TokenDataset(std::vector<bpe::TokenId> tokens);

    /**
     * Loads a raw little-endian uint32 token file.  This is the deliberately
     * simple format used by most training pipelines after tokenization: there
     * is no per-record parsing in the hot path.
     */
    [[nodiscard]] static TokenDataset load(const std::filesystem::path& path);

    [[nodiscard]] std::span<const bpe::TokenId> tokens() const noexcept;
    [[nodiscard]] std::size_t size() const noexcept;
    [[nodiscard]] bool empty() const noexcept;

private:
    std::vector<bpe::TokenId> tokens_;
};

struct Batch {
    std::vector<bpe::TokenId> input_ids;  // [batch_size, sequence_length]
    std::vector<bpe::TokenId> target_ids; // input_ids shifted by one token
    std::size_t batch_size = 0;
    std::size_t sequence_length = 0;

    [[nodiscard]] std::size_t token_count() const noexcept;
    [[nodiscard]] bool empty() const noexcept;
};

/** Reusable GPU storage for a host Batch. */
struct DeviceBatch {
    DeviceBuffer input_ids;
    DeviceBuffer target_ids;
    std::size_t batch_size = 0;
    std::size_t sequence_length = 0;
};

/** Uploads a batch with cudaMemcpyAsync; the destination buffers grow only when needed. */
void upload_batch(const Batch& batch, DeviceBatch& destination, cudaStream_t stream = nullptr);

struct DataLoaderConfig {
    std::size_t batch_size = 1;
    std::size_t sequence_length = 1;
    bool shuffle = true;
    bool drop_last = true;
    std::uint64_t seed = 0;
};

/**
 * Batches fixed-size, non-overlapping autoregressive examples from a token
 * corpus.  It stores only sample indices, so shuffling does not duplicate the
 * dataset and a supplied Batch can be reused across all iterations.
 */
class DatasetLoader {
public:
    explicit DatasetLoader(std::shared_ptr<const TokenDataset> dataset,
                           const DataLoaderConfig &config = {});

    DatasetLoader(TokenDataset dataset, const DataLoaderConfig &config);

    [[nodiscard]] bool has_next() const noexcept;
    [[nodiscard]] std::size_t sample_count() const noexcept;
    [[nodiscard]] std::size_t batch_count() const noexcept;
    [[nodiscard]] std::size_t epoch() const noexcept;

    /** Fills @p batch and returns false once the epoch has been consumed. */
    bool next(Batch& batch);
    /** Allocating convenience overload for callers that do not reuse a Batch. */
    [[nodiscard]] std::optional<Batch> next();
    /** Starts a new epoch; shuffle order is deterministically advanced. */
    void reset();

private:
    void initialize();
    void shuffle_indices();

    std::shared_ptr<const TokenDataset> dataset_;
    DataLoaderConfig config_;
    std::vector<std::size_t> sample_indices_;
    std::size_t next_sample_ = 0;
    std::size_t epoch_ = 0;
};

using DataLoader = DatasetLoader;

} // namespace data
