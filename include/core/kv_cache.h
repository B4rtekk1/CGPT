#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstddef>

/** @brief Configuration used to allocate a transformer key/value cache. */
struct KVCacheConfig {
    /** @brief Number of transformer layers represented in the cache. */
    std::size_t num_layers = 0;
    /** @brief Maximum batch size supported by the cache. */
    std::size_t max_batch_size = 0;
    /** @brief Maximum number of tokens stored for each sequence. */
    std::size_t max_sequence_length = 0;
    /** @brief Number of key/value attention heads. */
    std::size_t num_kv_heads = 0;
    /** @brief Size of one attention head in elements. */
    std::size_t head_dim = 0;
};

/**
 * @brief GPU-resident key/value cache for autoregressive transformer inference.
 *
 * Keys and values are stored as CUDA half-precision values. Sequence lengths
 * are tracked independently for every layer and batch entry.
 */
class KVCache {
public:
    /** @brief Allocates a cache according to the supplied configuration. */
    explicit KVCache(const KVCacheConfig& config);
    /** @brief Releases all cache allocations. */
    ~KVCache();

    /** @brief Copy construction is disabled because the cache owns resources. */
    KVCache(const KVCache&) = delete;
    /** @brief Copy assignment is disabled because the cache owns resources. */
    KVCache& operator=(const KVCache&) = delete;

    /** @brief Transfers ownership from another key/value cache. */
    KVCache(KVCache&& other) noexcept;
    /** @brief Transfers ownership from another key/value cache. */
    KVCache& operator=(KVCache&& other) noexcept;

    /** @brief Returns the cache allocation configuration. */
    [[nodiscard]] const KVCacheConfig& config() const noexcept { return config_; }

    /** @brief Returns the raw device pointer to key storage. */
    [[nodiscard]] __half* keys() noexcept { return keys_; }
    /** @brief Returns the raw device pointer to value storage. */
    [[nodiscard]] __half* values() noexcept { return values_; }
    /** @brief Returns a const pointer to key storage. */
    [[nodiscard]] const __half* keys() const noexcept { return keys_; }
    /** @brief Returns a const pointer to value storage. */
    [[nodiscard]] const __half* values() const noexcept { return values_; }

    /**
     * @brief Returns the beginning of one layer/batch key sequence.
     * @param layer Transformer layer index.
     * @param batch Batch entry index.
     */
    [[nodiscard]] __half* key_sequence(std::size_t layer, std::size_t batch);
    /** @copydoc key_sequence(std::size_t, std::size_t) */
    [[nodiscard]] const __half* key_sequence(std::size_t layer, std::size_t batch) const;

    /**
     * @brief Returns the beginning of one layer/batch value sequence.
     * @param layer Transformer layer index.
     * @param batch Batch entry index.
     */
    [[nodiscard]] __half* value_sequence(std::size_t layer, std::size_t batch);
    /** @copydoc value_sequence(std::size_t, std::size_t) */
    [[nodiscard]] const __half* value_sequence(std::size_t layer, std::size_t batch) const;

    /**
     * @brief Returns the number of valid cached tokens.
     * @param layer Transformer layer index.
     * @param batch Batch entry index.
     */
    [[nodiscard]] std::size_t sequence_length(std::size_t layer, std::size_t batch) const;
    /**
     * @brief Sets the number of valid cached tokens.
     * @param layer Transformer layer index.
     * @param batch Batch entry index.
     * @param length New sequence length.
     */
    void set_sequence_length(std::size_t layer, std::size_t batch, std::size_t length);
    /**
     * @brief Advances a sequence length by a number of tokens.
     * @param layer Transformer layer index.
     * @param batch Batch entry index.
     * @param token_count Number of tokens to add.
     */
    void advance_sequence_length(
        std::size_t layer,
        std::size_t batch,
        std::size_t token_count = 1
    );

    /** @brief Clears key/value contents on the selected CUDA stream. */
    void clear(cudaStream_t stream = nullptr);

    /** @brief Resets all sequence lengths to zero. */
    void reset_lengths() noexcept;

    /**
     * @brief Writes a token range into a layer's cache sequences.
     * @param layer Transformer layer index.
     * @param batch_size Number of batch entries in the source arrays.
     * @param token_offset Destination token offset.
     * @param token_count Number of tokens to write per batch entry.
     * @param key Source device keys.
     * @param value Source device values.
     * @param stream CUDA stream used for the copy.
     */
    void write(
        std::size_t layer,
        std::size_t batch_size,
        std::size_t token_offset,
        std::size_t token_count,
        const __half* key,
        const __half* value,
        cudaStream_t stream = nullptr
    );

    /**
     * @brief Appends tokens after the current end of each cached sequence.
     * @param layer Transformer layer index.
     * @param batch_size Number of batch entries in the source arrays.
     * @param token_count Number of tokens to append per batch entry.
     * @param key Source device keys.
     * @param value Source device values.
     * @param stream CUDA stream used for the copy.
     */
    void append(
        std::size_t layer,
        std::size_t batch_size,
        std::size_t token_count,
        const __half* key,
        const __half* value,
        cudaStream_t stream = nullptr
    );

private:
    KVCacheConfig config_{};
    __half* keys_ = nullptr;
    __half* values_ = nullptr;
    std::size_t* host_lengths_ = nullptr;

    [[nodiscard]] std::size_t elements_per_token() const noexcept;
    [[nodiscard]] std::size_t elements_per_sequence() const noexcept;
    [[nodiscard]] std::size_t total_elements() const noexcept;
    [[nodiscard]] std::size_t length_index(
        std::size_t layer,
        std::size_t batch
    ) const noexcept;

    void validate_config() const;
    void validate_layer(std::size_t layer) const;
    void validate_batch_size(std::size_t batch_size) const;
    void release() noexcept;
};