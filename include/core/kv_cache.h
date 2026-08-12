#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstddef>

struct KVCacheConfig {
    std::size_t num_layers = 0;
    std::size_t max_batch_size = 0;
    std::size_t max_sequence_length = 0;
    std::size_t num_kv_heads = 0;
    std::size_t head_dim = 0;
};

class KVCache {
public:
    explicit KVCache(const KVCacheConfig& config);
    ~KVCache();

    KVCache(const KVCache&) = delete;
    KVCache& operator=(const KVCache&) = delete;

    KVCache(KVCache&& other) noexcept;
    KVCache& operator=(KVCache&& other) noexcept;

    [[nodiscard]] const KVCacheConfig& config() const noexcept { return config_; }

    [[nodiscard]] __half* keys() noexcept { return keys_; }
    [[nodiscard]] __half* values() noexcept { return values_; }
    [[nodiscard]] const __half* keys() const noexcept { return keys_; }
    [[nodiscard]] const __half* values() const noexcept { return values_; }

    [[nodiscard]] __half* key_sequence(std::size_t layer, std::size_t batch);
    [[nodiscard]] const __half* key_sequence(std::size_t layer, std::size_t batch) const;

    [[nodiscard]] __half* value_sequence(std::size_t layer, std::size_t batch);
    [[nodiscard]] const __half* value_sequence(std::size_t layer, std::size_t batch) const;

    [[nodiscard]] std::size_t sequence_length(std::size_t layer, std::size_t batch) const;
    void set_sequence_length(std::size_t layer, std::size_t batch, std::size_t length);
    void advance_sequence_length(
        std::size_t layer,
        std::size_t batch,
        std::size_t token_count = 1
    );

    void clear(cudaStream_t stream = nullptr);

    void reset_lengths() noexcept;

    void write(
        std::size_t layer,
        std::size_t batch_size,
        std::size_t token_offset,
        std::size_t token_count,
        const __half* key,
        const __half* value,
        cudaStream_t stream = nullptr
    );

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
