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

/** STL allocator backed by CUDA page-locked host memory. */
template <typename T>
class CudaHostAllocator {
public:
    using value_type = T;

    CudaHostAllocator() noexcept = default;
    template <typename U> CudaHostAllocator(const CudaHostAllocator<U>&) noexcept {}

    [[nodiscard]] T* allocate(std::size_t count) {
        if (count > static_cast<std::size_t>(-1) / sizeof(T)) throw std::bad_array_new_length();
        void* pointer = nullptr;
        CUDA_CHECK(cudaHostAlloc(&pointer, count * sizeof(T), cudaHostAllocDefault));
        return static_cast<T*>(pointer);
    }

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

struct DeviceBatch;

/** Host-side training batch stored in page-locked memory for async H2D copies. */
struct Batch {
    HostTokenVector input_ids;  // [batch_size, sequence_length]
    HostTokenVector target_ids; // input_ids shifted by one token
    std::size_t batch_size = 0;
    std::size_t sequence_length = 0;

    Batch() = default;
    ~Batch();
    Batch(const Batch& other);
    Batch& operator=(const Batch& other);
    Batch(Batch&& other) noexcept;
    Batch& operator=(Batch&& other) noexcept;

    [[nodiscard]] std::size_t token_count() const noexcept;
    [[nodiscard]] bool empty() const noexcept;

    /** Waits until the last async upload no longer reads this batch. */
    void wait_until_reusable() const;

private:
    friend void upload_batch(const Batch&, DeviceBatch&, cudaStream_t);
    void mark_upload_complete(cudaStream_t stream) const;
    mutable cudaEvent_t upload_complete_ = nullptr;
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
    std::size_t sample_count_ = 0;
    bool sequential_ = true;
    std::size_t next_sample_ = 0;
    std::size_t epoch_ = 0;
};

using DataLoader = DatasetLoader;

} // namespace data
