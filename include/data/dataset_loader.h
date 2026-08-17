#pragma once

#include "core/device_buffer.h"
#include "tokenizer/bpe_tokenizer.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <new>
#include <optional>
#include <span>
#include <vector>

namespace data {

/**
 * @brief Immutable, contiguous token corpus used by the training data loader.
 */
class TokenDataset {
public:
    /** @brief Constructs an empty dataset. */
    TokenDataset() = default;
    /** @brief Takes ownership of a token vector. */
    explicit TokenDataset(std::vector<bpe::TokenId> tokens);

    /**
     * @brief Loads a raw little-endian uint32 token file.
     *
     * This is the deliberately
     * simple format used by most training pipelines after tokenization: there
     * is no per-record parsing in the hot path.
     *
     * @param path File containing contiguous token IDs.
     * @return Loaded token dataset.
     */
    [[nodiscard]] static TokenDataset load(const std::filesystem::path& path);

    /** @brief Returns a read-only view of all token IDs. */
    [[nodiscard]] std::span<const bpe::TokenId> tokens() const noexcept;
    /** @brief Returns the number of tokens in the dataset. */
    [[nodiscard]] std::size_t size() const noexcept;
    /** @brief Checks whether the dataset contains no tokens. */
    [[nodiscard]] bool empty() const noexcept;

private:
    std::vector<bpe::TokenId> tokens_;
};

/**
 * @brief STL allocator backed by CUDA page-locked host memory.
 * @tparam T Element type allocated by the allocator.
 */
template <typename T>
class CudaHostAllocator {
public:
    /** @brief Required allocator value type. */
    using value_type = T;

    /** @brief Constructs an empty allocator. */
    CudaHostAllocator() noexcept = default;
    /** @brief Constructs an allocator compatible with another value type. */
    template <typename U> CudaHostAllocator(const CudaHostAllocator<U>&) noexcept {}

    /**
     * @brief Allocates page-locked host memory for a number of elements.
     * @param count Number of elements to allocate.
     * @return Pointer to the allocated pinned memory.
     */
    [[nodiscard]] T* allocate(std::size_t count) {
        if (count > static_cast<std::size_t>(-1) / sizeof(T)) throw std::bad_array_new_length();
        void* pointer = nullptr;
        CUDA_CHECK(cudaHostAlloc(&pointer, count * sizeof(T), cudaHostAllocDefault));
        return static_cast<T*>(pointer);
    }

    /**
     * @brief Releases page-locked host memory.
     * @param pointer Memory previously returned by allocate.
     */
    void deallocate(T* pointer, std::size_t) noexcept {
        if (pointer != nullptr) static_cast<void>(cudaFreeHost(pointer));
    }

    template <typename U>
    friend constexpr bool operator==(const CudaHostAllocator&, const CudaHostAllocator<U>&) noexcept { return true; }
};

using HostTokenVector = std::vector<bpe::TokenId, CudaHostAllocator<bpe::TokenId>>;

template <typename Allocator>
[[nodiscard]] bool operator==(const HostTokenVector& left,
                              const std::vector<bpe::TokenId, Allocator>& right) noexcept {
    return left.size() == right.size() &&
        std::equal(left.begin(), left.end(), right.begin());
}

template <typename Allocator>
[[nodiscard]] bool operator==(const std::vector<bpe::TokenId, Allocator>& left,
                              const HostTokenVector& right) noexcept {
    return right == left;
}

/** Forward declaration of the GPU-side training batch. */
struct DeviceBatch;

/**
 * @brief Host-side training batch stored in page-locked memory for async H2D copies.
 */
struct Batch {
    /** @brief Input token IDs with shape `[batch_size, sequence_length]`. */
    HostTokenVector input_ids;  // [batch_size, sequence_length]
    /** @brief Target IDs shifted by one token relative to input_ids. */
    HostTokenVector target_ids; // input_ids shifted by one token
    /** @brief Number of sequences in the batch. */
    std::size_t batch_size = 0;
    /** @brief Number of tokens in each sequence. */
    std::size_t sequence_length = 0;

    /** @brief Constructs an empty batch. */
    Batch() = default;
    /** @brief Releases the CUDA event associated with asynchronous uploads. */
    ~Batch();
    /** @brief Creates a deep copy of a batch. */
    Batch(const Batch& other);
    /** @brief Replaces this batch with a deep copy of another batch. */
    Batch& operator=(const Batch& other);
    /** @brief Transfers batch ownership from another batch. */
    Batch(Batch&& other) noexcept;
    /** @brief Transfers batch ownership from another batch. */
    Batch& operator=(Batch&& other) noexcept;

    /** @brief Returns the total number of tokens in the input batch. */
    [[nodiscard]] std::size_t token_count() const noexcept;
    /** @brief Checks whether the batch contains no samples or tokens. */
    [[nodiscard]] bool empty() const noexcept;

    /** @brief Waits until the last asynchronous upload no longer reads this batch. */
    void wait_until_reusable() const;

private:
    friend void upload_batch(const Batch&, DeviceBatch&, cudaStream_t);
    void mark_upload_complete(cudaStream_t stream) const;
    mutable cudaEvent_t upload_complete_ = nullptr;
};

/** @brief Reusable GPU storage for a host-side Batch. */
struct DeviceBatch {
    /** @brief Device buffer containing input token IDs. */
    DeviceBuffer input_ids;
    /** @brief Device buffer containing target token IDs. */
    DeviceBuffer target_ids;
    /** @brief Number of sequences in the batch. */
    std::size_t batch_size = 0;
    /** @brief Number of tokens in each sequence. */
    std::size_t sequence_length = 0;
};

/**
 * @brief Uploads a batch with cudaMemcpyAsync.
 *
 * Destination buffers grow only when needed. The source batch remains in use
 * until the asynchronous copy has completed.
 *
 * @param batch Host batch to upload.
 * @param destination Reusable device-side destination storage.
 * @param stream CUDA stream used for the asynchronous copies.
 */
void upload_batch(const Batch& batch, DeviceBatch& destination, cudaStream_t stream = nullptr);

/** @brief Configuration of a DatasetLoader. */
struct DataLoaderConfig {
    /** @brief Number of sequences returned by each batch. */
    std::size_t batch_size = 1;
    /** @brief Number of tokens in each input sequence. */
    std::size_t sequence_length = 1;
    /** @brief Whether samples are shuffled at epoch boundaries. */
    bool shuffle = true;
    /** @brief Whether an incomplete final batch is discarded. */
    bool drop_last = true;
    /** @brief Seed used to advance the shuffle order. */
    std::uint64_t seed = 0;
};

/**
 * @brief Batches fixed-size, non-overlapping autoregressive examples from a token
 * corpus.  It stores only sample indices, so shuffling does not duplicate the
 * dataset and a supplied Batch can be reused across all iterations.
 */
class DatasetLoader {
public:
    /**
     * @brief Creates a loader that shares ownership of a token dataset.
     * @param dataset Dataset to batch.
     * @param config Loader and batching configuration.
     */
    explicit DatasetLoader(std::shared_ptr<const TokenDataset> dataset,
                           const DataLoaderConfig &config = {});

    /**
     * @brief Creates a loader by taking a dataset value.
     * @param dataset Dataset to batch.
     * @param config Loader and batching configuration.
     */
    DatasetLoader(TokenDataset dataset, const DataLoaderConfig &config);

    /** @brief Returns whether another batch is available in the current epoch. */
    [[nodiscard]] bool has_next() const noexcept;
    /** @brief Returns the number of training samples in the dataset. */
    [[nodiscard]] std::size_t sample_count() const noexcept;
    /** @brief Returns the number of batches in the current epoch. */
    [[nodiscard]] std::size_t batch_count() const noexcept;
    /** @brief Returns the zero-based epoch counter. */
    [[nodiscard]] std::size_t epoch() const noexcept;

    /**
     * @brief Fills @p batch with the next sample batch.
     * @return `true` when a batch was produced; `false` after the epoch is consumed.
     */
    bool next(Batch& batch);
    /** @brief Allocating convenience overload for callers that do not reuse a Batch. */
    [[nodiscard]] std::optional<Batch> next();
    /** @brief Starts a new epoch and deterministically advances shuffle order. */
    void reset();

private:
    void initialize();
    void shuffle_indices();

    std::shared_ptr<const TokenDataset> dataset_;
    DataLoaderConfig config_;
    std::vector<std::size_t> sample_indices_;
    std::size_t sample_count_ = 0;
    bool sequential_ = true;
    std::size_t next_sample_ = 0;
    std::size_t epoch_ = 0;
};

using DataLoader = DatasetLoader;

} // namespace data