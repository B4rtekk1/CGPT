#include "core/kv_cache.h"
#include "core/cuda_check.h"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <utility>

namespace {

void validate_batch_index(const KVCacheConfig& config, const std::size_t batch) {
    if (batch >= config.max_batch_size) {
        throw std::out_of_range("KVCache batch index is out of range");
    }
}

}

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

KVCache::~KVCache() {
    release();
}

KVCache::KVCache(KVCache&& other) noexcept
    : config_(other.config_),
      keys_(std::exchange(other.keys_, nullptr)),
      values_(std::exchange(other.values_, nullptr)),
      host_lengths_(std::exchange(other.host_lengths_, nullptr)) {
    other.config_ = {};
}

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

__half* KVCache::key_sequence(const std::size_t layer, const std::size_t batch) {
    validate_layer(layer);
    validate_batch_index(config_, batch);
    return keys_ + (layer * config_.max_batch_size + batch) * elements_per_sequence();
}

const __half* KVCache::key_sequence(
    const std::size_t layer,
    const std::size_t batch
) const {
    validate_layer(layer);
    validate_batch_index(config_, batch);
    return keys_ + (layer * config_.max_batch_size + batch) * elements_per_sequence();
}

__half* KVCache::value_sequence(const std::size_t layer, const std::size_t batch) {
    validate_layer(layer);
    validate_batch_index(config_, batch);
    return values_ + (layer * config_.max_batch_size + batch) * elements_per_sequence();
}

const __half* KVCache::value_sequence(
    const std::size_t layer,
    const std::size_t batch
) const {
    validate_layer(layer);
    validate_batch_index(config_, batch);
    return values_ + (layer * config_.max_batch_size + batch) * elements_per_sequence();
}

std::size_t KVCache::sequence_length(
    const std::size_t layer,
    const std::size_t batch
) const {
    validate_layer(layer);
    validate_batch_index(config_, batch);
    return host_lengths_[length_index(layer, batch)];
}

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

void KVCache::clear(cudaStream_t stream) {
    const std::size_t bytes = total_elements() * sizeof(__half);
    CUDA_CHECK(cudaMemsetAsync(keys_, 0, bytes, stream));
    CUDA_CHECK(cudaMemsetAsync(values_, 0, bytes, stream));
    reset_lengths();
}

void KVCache::reset_lengths() noexcept {
    if (host_lengths_ != nullptr) {
        std::fill_n(host_lengths_, config_.num_layers * config_.max_batch_size, 0);
    }
}

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

std::size_t KVCache::elements_per_token() const noexcept {
    return config_.num_kv_heads * config_.head_dim;
}

std::size_t KVCache::elements_per_sequence() const noexcept {
    return config_.max_sequence_length * elements_per_token();
}

std::size_t KVCache::total_elements() const noexcept {
    return config_.num_layers * config_.max_batch_size * elements_per_sequence();
}

std::size_t KVCache::length_index(
    const std::size_t layer,
    const std::size_t batch
) const noexcept {
    return layer * config_.max_batch_size + batch;
}

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

void KVCache::validate_layer(const std::size_t layer) const {
    if (layer >= config_.num_layers) {
        throw std::out_of_range("KVCache layer index is out of range");
    }
}

void KVCache::validate_batch_size(const std::size_t batch_size) const {
    if (batch_size == 0 || batch_size > config_.max_batch_size) {
        throw std::out_of_range("KVCache batch_size is out of range");
    }
}

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
