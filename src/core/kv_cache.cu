/** @file kv_cache.cu CUDA implementation of the autoregressive key-value cache. */

#include "core/kv_cache.h"
#include "core/cuda_check.h"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <utility>

namespace {

/**
 * @brief Validates a single batch index against a KV-cache configuration.
 *
 * @param config Cache configuration containing the maximum batch size.
 * @param batch Batch index to validate.
 *
 * @throws std::out_of_range If @p batch is outside the configured range.
 */
void validate_batch_index(const KVCacheConfig& config, const std::size_t batch) {
    if (batch >= config.max_batch_size) {
        throw std::out_of_range("KVCache batch index is out of range");
    }
}

}

/**
 * @brief Allocates an FP16 CUDA key-value cache.
 *
 * Two equally sized device buffers are allocated for keys and values using the
 * logical layout `[layer, batch, token, kv_head, head_dim]`. Per-layer,
 * per-batch sequence lengths are stored in a zero-initialized host array.
 *
 * @param config Cache dimensions.
 *
 * @throws std::invalid_argument If any configured dimension is zero.
 * @throws std::overflow_error If the required allocation size overflows.
 * @throws CudaError If CUDA allocation or initial clearing fails.
 *
 * @note Construction provides cleanup on partial allocation failure.
 */
KVCache::KVCache(const KVCacheConfig& config) : config_(config) {
    validate_config();

    const std::size_t bytes = total_elements() * sizeof(__half);
    CUDA_CHECK(cudaMalloc(&keys_, bytes));

    try {
        CUDA_CHECK(cudaMalloc(&values_, bytes));
        host_lengths_ = new std::size_t[config_.num_layers * config_.max_batch_size]{};
        clear();
    } catch (...) {
        release();
        throw;
    }
}

/**
 * @brief Releases CUDA buffers and host-side sequence-length storage.
 */
KVCache::~KVCache() {
    release();
}

/**
 * @brief Move-constructs a cache by transferring resource ownership.
 *
 * @param other Cache whose device buffers and host metadata are transferred.
 *
 * @post @p other no longer owns allocated resources.
 */
KVCache::KVCache(KVCache&& other) noexcept
    : config_(other.config_),
      keys_(std::exchange(other.keys_, nullptr)),
      values_(std::exchange(other.values_, nullptr)),
      host_lengths_(std::exchange(other.host_lengths_, nullptr)) {
    other.config_ = {};
}

/**
 * @brief Move-assigns a cache by releasing current resources and taking ownership.
 *
 * @param other Cache whose resources are transferred.
 * @return Reference to this cache.
 *
 * @post @p other no longer owns allocated resources.
 */
KVCache& KVCache::operator=(KVCache&& other) noexcept {
    if (this != &other) {
        release();
        config_ = other.config_;
        keys_ = std::exchange(other.keys_, nullptr);
        values_ = std::exchange(other.values_, nullptr);
        host_lengths_ = std::exchange(other.host_lengths_, nullptr);
        other.config_ = {};
    }
    return *this;
}

/**
 * @brief Returns a mutable pointer to one key sequence in device memory.
 *
 * @param layer Transformer-layer index.
 * @param batch Batch-item index.
 * @return Device pointer to the first `[kv_head, head_dim]` element at token
 * position zero.
 *
 * @throws std::out_of_range If @p layer or @p batch is invalid.
 */
__half* KVCache::key_sequence(const std::size_t layer, const std::size_t batch) {
    validate_layer(layer);
    validate_batch_index(config_, batch);
    return keys_ + (layer * config_.max_batch_size + batch) * elements_per_sequence();
}

/**
 * @brief Returns a read-only pointer to one key sequence in device memory.
 *
 * @param layer Transformer-layer index.
 * @param batch Batch-item index.
 * @return Const device pointer to the sequence start.
 *
 * @throws std::out_of_range If @p layer or @p batch is invalid.
 */
const __half* KVCache::key_sequence(
    const std::size_t layer,
    const std::size_t batch
) const {
    validate_layer(layer);
    validate_batch_index(config_, batch);
    return keys_ + (layer * config_.max_batch_size + batch) * elements_per_sequence();
}

/**
 * @brief Returns a mutable pointer to one value sequence in device memory.
 *
 * @param layer Transformer-layer index.
 * @param batch Batch-item index.
 * @return Device pointer to the sequence start.
 *
 * @throws std::out_of_range If @p layer or @p batch is invalid.
 */
__half* KVCache::value_sequence(const std::size_t layer, const std::size_t batch) {
    validate_layer(layer);
    validate_batch_index(config_, batch);
    return values_ + (layer * config_.max_batch_size + batch) * elements_per_sequence();
}

/**
 * @brief Returns a read-only pointer to one value sequence in device memory.
 *
 * @param layer Transformer-layer index.
 * @param batch Batch-item index.
 * @return Const device pointer to the sequence start.
 *
 * @throws std::out_of_range If @p layer or @p batch is invalid.
 */
const __half* KVCache::value_sequence(
    const std::size_t layer,
    const std::size_t batch
) const {
    validate_layer(layer);
    validate_batch_index(config_, batch);
    return values_ + (layer * config_.max_batch_size + batch) * elements_per_sequence();
}

/**
 * @brief Returns the host-tracked cached length for one layer and batch item.
 *
 * @param layer Transformer-layer index.
 * @param batch Batch-item index.
 * @return Number of valid cached token positions.
 *
 * @throws std::out_of_range If @p layer or @p batch is invalid.
 */
std::size_t KVCache::sequence_length(
    const std::size_t layer,
    const std::size_t batch
) const {
    validate_layer(layer);
    validate_batch_index(config_, batch);
    return host_lengths_[length_index(layer, batch)];
}

/**
 * @brief Sets the host-tracked cached length for one sequence.
 *
 * This method updates metadata only and does not initialize or copy device
 * memory.
 *
 * @param layer Transformer-layer index.
 * @param batch Batch-item index.
 * @param length New number of valid cached token positions.
 *
 * @throws std::out_of_range If an index is invalid or @p length exceeds the
 * configured sequence capacity.
 */
void KVCache::set_sequence_length(
    const std::size_t layer,
    const std::size_t batch,
    const std::size_t length
) {
    validate_layer(layer);
    validate_batch_index(config_, batch);
    if (length > config_.max_sequence_length) {
        throw std::out_of_range("KVCache sequence length is out of range");
    }
    host_lengths_[length_index(layer, batch)] = length;
}

/**
 * @brief Increments a sequence's host-tracked length.
 *
 * @param layer Transformer-layer index.
 * @param batch Batch-item index.
 * @param token_count Number of newly valid token positions.
 *
 * @throws std::out_of_range If an index is invalid or the increment exceeds
 * the remaining cache capacity.
 */
void KVCache::advance_sequence_length(
    const std::size_t layer,
    const std::size_t batch,
    const std::size_t token_count
) {
    const std::size_t current_length = sequence_length(layer, batch);
    if (token_count > config_.max_sequence_length - current_length) {
        throw std::out_of_range("KVCache sequence length is out of range");
    }
    set_sequence_length(layer, batch, current_length + token_count);
}

/**
 * @brief Asynchronously zeroes all key/value storage and resets all lengths.
 *
 * @param stream CUDA stream on which both memset operations are enqueued.
 *
 * @throws CudaError If either asynchronous CUDA memset fails.
 *
 * @note Host length metadata is reset immediately. Device clearing may still
 * be pending when this function returns.
 */
void KVCache::clear(cudaStream_t stream) {
    const std::size_t bytes = total_elements() * sizeof(__half);
    CUDA_CHECK(cudaMemsetAsync(keys_, 0, bytes, stream));
    CUDA_CHECK(cudaMemsetAsync(values_, 0, bytes, stream));
    reset_lengths();
}

/**
 * @brief Resets every host-side sequence length to zero.
 *
 * Device key and value storage is left unchanged.
 */
void KVCache::reset_lengths() noexcept {
    if (host_lengths_ != nullptr) {
        std::fill_n(host_lengths_, config_.num_layers * config_.max_batch_size, 0);
    }
}

/**
 * @brief Copies a rectangular token range into one layer of the cache.
 *
 * Input keys and values are expected in contiguous
 * `[batch_size, token_count, num_kv_heads, head_dim]` FP16 layout. Two
 * cudaMemcpy2DAsync operations copy each batch row into the configured cache
 * pitch at @p token_offset.
 *
 * @param layer Destination transformer-layer index.
 * @param batch_size Number of batch rows to copy.
 * @param token_offset First destination token position.
 * @param token_count Number of token positions copied per batch item.
 * @param key Device pointer to contiguous source keys.
 * @param value Device pointer to contiguous source values.
 * @param stream CUDA stream used for asynchronous copies.
 *
 * @throws std::out_of_range If layer, batch size, or token range is invalid.
 * @throws std::invalid_argument If a source pointer is null.
 * @throws CudaError If a CUDA copy cannot be enqueued.
 *
 * @note This operation does not modify host-tracked sequence lengths.
 */
void KVCache::write(
    const std::size_t layer,
    const std::size_t batch_size,
    const std::size_t token_offset,
    const std::size_t token_count,
    const __half* key,
    const __half* value,
    cudaStream_t stream
) {
    validate_layer(layer);
    validate_batch_size(batch_size);

    if (token_count == 0 || token_offset > config_.max_sequence_length ||
        token_count > config_.max_sequence_length - token_offset) {
        throw std::out_of_range("KVCache write token range is out of bounds");
    }
    if (key == nullptr || value == nullptr) {
        throw std::invalid_argument("KVCache input key/value pointer is null");
    }

    // A batch item occupies one contiguous row in both layouts.  Delegating
    // the strided bulk copy to the CUDA copy engine avoids a launch plus
    // integer division/modulo for every cache element.
    const std::size_t row_bytes = token_count * elements_per_token() * sizeof(__half);
    const std::size_t cache_pitch = elements_per_sequence() * sizeof(__half);
    __half* const key_destination =
        key_sequence(layer, 0) + token_offset * elements_per_token();
    __half* const value_destination =
        value_sequence(layer, 0) + token_offset * elements_per_token();

    CUDA_CHECK(cudaMemcpy2DAsync(
        key_destination, cache_pitch, key, row_bytes, row_bytes, batch_size,
        cudaMemcpyDeviceToDevice, stream));
    CUDA_CHECK(cudaMemcpy2DAsync(
        value_destination, cache_pitch, value, row_bytes, row_bytes, batch_size,
        cudaMemcpyDeviceToDevice, stream));
}

/**
 * @brief Appends tokens to the valid end of each batch sequence.
 *
 * Input keys and values use contiguous
 * `[batch_size, token_count, num_kv_heads, head_dim]` FP16 layout. When every
 * batch item has the same current length, strided 2D copies are used. Otherwise
 * one asynchronous copy per batch item and per K/V buffer is enqueued.
 * Sequence lengths are advanced after all copies have been submitted.
 *
 * @param layer Destination transformer-layer index.
 * @param batch_size Number of batch rows to append.
 * @param token_count Number of tokens appended to every row.
 * @param key Device pointer to source keys.
 * @param value Device pointer to source values.
 * @param stream CUDA stream used for asynchronous copies.
 *
 * @throws std::out_of_range If layer, batch size, or resulting sequence length
 * is invalid.
 * @throws std::invalid_argument If @p token_count is zero or a source pointer
 * is null.
 * @throws CudaError If a CUDA copy cannot be enqueued.
 *
 * @note Host lengths are updated before the asynchronous copies necessarily
 * complete; users must preserve stream ordering before consuming the data.
 */
void KVCache::append(
    const std::size_t layer,
    const std::size_t batch_size,
    const std::size_t token_count,
    const __half* key,
    const __half* value,
    cudaStream_t stream
) {
    validate_layer(layer);
    validate_batch_size(batch_size);

    if (token_count == 0) {
        throw std::invalid_argument("token_count must be positive");
    }
    if (key == nullptr || value == nullptr) {
        throw std::invalid_argument("KVCache input key/value pointer is null");
    }

    for (std::size_t batch = 0; batch < batch_size; ++batch) {
        const std::size_t sequence_length = host_lengths_[length_index(layer, batch)];
        if (token_count > config_.max_sequence_length - sequence_length) {
            throw std::out_of_range("KVCache append exceeds max_sequence_length");
        }
    }

    const std::size_t row_bytes = token_count * elements_per_token() * sizeof(__half);
    const std::size_t first_length = host_lengths_[length_index(layer, 0)];
    bool uniform_lengths = true;
    for (std::size_t batch = 1; batch < batch_size; ++batch) {
        uniform_lengths = uniform_lengths &&
            host_lengths_[length_index(layer, batch)] == first_length;
    }

    if (uniform_lengths) {
        const std::size_t cache_pitch = elements_per_sequence() * sizeof(__half);
        CUDA_CHECK(cudaMemcpy2DAsync(
            key_sequence(layer, 0) + first_length * elements_per_token(),
            cache_pitch, key, row_bytes, row_bytes, batch_size,
            cudaMemcpyDeviceToDevice, stream));
        CUDA_CHECK(cudaMemcpy2DAsync(
            value_sequence(layer, 0) + first_length * elements_per_token(),
            cache_pitch, value, row_bytes, row_bytes, batch_size,
            cudaMemcpyDeviceToDevice, stream));
    } else {
        for (std::size_t batch = 0; batch < batch_size; ++batch) {
            const std::size_t offset = host_lengths_[length_index(layer, batch)];
            const std::size_t source_offset = batch * token_count * elements_per_token();
            CUDA_CHECK(cudaMemcpyAsync(
                key_sequence(layer, batch) + offset * elements_per_token(),
                key + source_offset, row_bytes, cudaMemcpyDeviceToDevice, stream));
            CUDA_CHECK(cudaMemcpyAsync(
                value_sequence(layer, batch) + offset * elements_per_token(),
                value + source_offset, row_bytes, cudaMemcpyDeviceToDevice, stream));
        }
    }

    for (std::size_t batch = 0; batch < batch_size; ++batch) {
        host_lengths_[length_index(layer, batch)] += token_count;
    }
}

/**
 * @brief Returns the number of FP16 elements stored for one token.
 *
 * @return `num_kv_heads * head_dim`.
 */
std::size_t KVCache::elements_per_token() const noexcept {
    return config_.num_kv_heads * config_.head_dim;
}

/**
 * @brief Returns the capacity of one layer/batch sequence in elements.
 *
 * @return `max_sequence_length * elements_per_token()`.
 */
std::size_t KVCache::elements_per_sequence() const noexcept {
    return config_.max_sequence_length * elements_per_token();
}

/**
 * @brief Returns the element capacity of either the full key or value buffer.
 *
 * @return Product of layer, batch, sequence, KV-head, and head dimensions.
 */
std::size_t KVCache::total_elements() const noexcept {
    return config_.num_layers * config_.max_batch_size * elements_per_sequence();
}

/**
 * @brief Computes the flattened host-length index for a layer/batch pair.
 *
 * @param layer Transformer-layer index.
 * @param batch Batch-item index.
 * @return Flattened metadata index.
 */
std::size_t KVCache::length_index(
    const std::size_t layer,
    const std::size_t batch
) const noexcept {
    return layer * config_.max_batch_size + batch;
}

/**
 * @brief Validates cache dimensions and allocation-size arithmetic.
 *
 * @throws std::invalid_argument If any dimension is zero.
 * @throws std::overflow_error If the element or byte count overflows
 * std::size_t.
 */
void KVCache::validate_config() const {
    if (config_.num_layers == 0 || config_.max_batch_size == 0 ||
        config_.max_sequence_length == 0 || config_.num_kv_heads == 0 ||
        config_.head_dim == 0) {
        throw std::invalid_argument("All KVCacheConfig dimensions must be positive");
    }

    std::size_t elements = 1;
    for (const std::size_t dimension : {
             config_.num_layers,
             config_.max_batch_size,
             config_.max_sequence_length,
             config_.num_kv_heads,
             config_.head_dim}) {
        if (elements > std::numeric_limits<std::size_t>::max() / dimension) {
            throw std::overflow_error("KVCache allocation size overflow");
        }
        elements *= dimension;
    }

    if (elements > std::numeric_limits<std::size_t>::max() / sizeof(__half)) {
        throw std::overflow_error("KVCache allocation size overflow");
    }
}

/**
 * @brief Validates a transformer-layer index.
 *
 * @param layer Layer index to check.
 * @throws std::out_of_range If @p layer is invalid.
 */
void KVCache::validate_layer(const std::size_t layer) const {
    if (layer >= config_.num_layers) {
        throw std::out_of_range("KVCache layer index is out of range");
    }
}

/**
 * @brief Validates a nonzero active batch size.
 *
 * @param batch_size Number of active batch items.
 * @throws std::out_of_range If @p batch_size is zero or exceeds capacity.
 */
void KVCache::validate_batch_size(const std::size_t batch_size) const {
    if (batch_size == 0 || batch_size > config_.max_batch_size) {
        throw std::out_of_range("KVCache batch_size is out of range");
    }
}

/**
 * @brief Releases all owned CUDA and host resources.
 *
 * The function is idempotent and leaves every owned pointer null.
 */
void KVCache::release() noexcept {
    if (keys_ != nullptr) {
        cudaFree(keys_);
        keys_ = nullptr;
    }
    if (values_ != nullptr) {
        cudaFree(values_);
        values_ = nullptr;
    }
    delete[] host_lengths_;
    host_lengths_ = nullptr;
}
