#include "core/kv_cache.h"
#include "core/cuda_check.h"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <utility>

namespace {

constexpr int kThreadsPerBlock = 256;
constexpr int kMaxBlocks = 4096;

__global__ void write_kv_kernel(
    __half* __restrict__ cache_keys,
    __half* __restrict__ cache_values,
    const __half* __restrict__ input_keys,
    const __half* __restrict__ input_values,
    const std::size_t layer,
    const std::size_t batch_size,
    const std::size_t token_offset,
    const std::size_t token_count,
    const std::size_t max_batch_size,
    const std::size_t max_sequence_length,
    const std::size_t elements_per_token
) {
    const std::size_t input_elements = batch_size * token_count * elements_per_token;

    for (std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < input_elements;
         index += static_cast<std::size_t>(blockDim.x) * gridDim.x) {
        const std::size_t element = index % elements_per_token;
        const std::size_t token_linear = index / elements_per_token;
        const std::size_t token = token_linear % token_count;
        const std::size_t batch = token_linear / token_count;

        const std::size_t cache_index =
            (((layer * max_batch_size + batch) * max_sequence_length + token_offset + token) *
             elements_per_token) +
            element;

        cache_keys[cache_index] = input_keys[index];
        cache_values[cache_index] = input_values[index];
    }
}

__global__ void append_kv_kernel(
    __half* __restrict__ cache_keys,
    __half* __restrict__ cache_values,
    const __half* __restrict__ input_keys,
    const __half* __restrict__ input_values,
    const std::size_t* __restrict__ sequence_lengths,
    const std::size_t layer,
    const std::size_t batch_size,
    const std::size_t token_count,
    const std::size_t max_batch_size,
    const std::size_t max_sequence_length,
    const std::size_t elements_per_token
) {
    const std::size_t input_elements = batch_size * token_count * elements_per_token;

    for (std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < input_elements;
         index += static_cast<std::size_t>(blockDim.x) * gridDim.x) {
        const std::size_t element = index % elements_per_token;
        const std::size_t token_linear = index / elements_per_token;
        const std::size_t token = token_linear % token_count;
        const std::size_t batch = token_linear / token_count;
        const std::size_t destination_token = sequence_lengths[batch] + token;

        const std::size_t cache_index =
            (((layer * max_batch_size + batch) * max_sequence_length + destination_token) *
             elements_per_token) +
            element;

        cache_keys[cache_index] = input_keys[index];
        cache_values[cache_index] = input_values[index];
    }
}

int launch_blocks(const std::size_t element_count) {
    const std::size_t required =
        (element_count + kThreadsPerBlock - 1) / kThreadsPerBlock;
    return static_cast<int>(std::min<std::size_t>(required, kMaxBlocks));
}

void validate_batch_index(const KVCacheConfig& config, const std::size_t batch) {
    if (batch >= config.max_batch_size) {
        throw std::out_of_range("KVCache batch index is out of range");
    }
}

} // namespace

KVCache::KVCache(const KVCacheConfig& config) : config_(config) {
    validate_config();

    const std::size_t bytes = total_elements() * sizeof(__half);
    CUDA_CHECK(cudaMalloc(&keys_, bytes));

    try {
        CUDA_CHECK(cudaMalloc(&values_, bytes));
        host_lengths_ = new std::size_t[config_.num_layers * config_.max_batch_size]{};
        CUDA_CHECK(cudaMalloc(
            &device_lengths_,
            config_.num_layers * config_.max_batch_size * sizeof(std::size_t)));
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
      host_lengths_(std::exchange(other.host_lengths_, nullptr)),
      device_lengths_(std::exchange(other.device_lengths_, nullptr)) {
    other.config_ = {};
}

KVCache& KVCache::operator=(KVCache&& other) noexcept {
    if (this != &other) {
        release();
        config_ = other.config_;
        keys_ = std::exchange(other.keys_, nullptr);
        values_ = std::exchange(other.values_, nullptr);
        host_lengths_ = std::exchange(other.host_lengths_, nullptr);
        device_lengths_ = std::exchange(other.device_lengths_, nullptr);
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
    CUDA_CHECK(cudaMemsetAsync(
        device_lengths_,
        0,
        config_.num_layers * config_.max_batch_size * sizeof(std::size_t),
        stream));
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

    const std::size_t per_token = elements_per_token();
    const std::size_t element_count = batch_size * token_count * per_token;

    write_kv_kernel<<<launch_blocks(element_count), kThreadsPerBlock, 0, stream>>>(
        keys_,
        values_,
        key,
        value,
        layer,
        batch_size,
        token_offset,
        token_count,
        config_.max_batch_size,
        config_.max_sequence_length,
        per_token);
    CUDA_CHECK(cudaGetLastError());
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

    CUDA_CHECK(cudaMemcpyAsync(
        device_lengths_,
        host_lengths_ + length_index(layer, 0),
        batch_size * sizeof(std::size_t),
        cudaMemcpyHostToDevice,
        stream));

    const std::size_t per_token = elements_per_token();
    const std::size_t element_count = batch_size * token_count * per_token;

    append_kv_kernel<<<launch_blocks(element_count), kThreadsPerBlock, 0, stream>>>(
        keys_,
        values_,
        key,
        value,
        device_lengths_,
        layer,
        batch_size,
        token_count,
        config_.max_batch_size,
        config_.max_sequence_length,
        per_token);
    CUDA_CHECK(cudaGetLastError());

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
    if (device_lengths_ != nullptr) {
        cudaFree(device_lengths_);
        device_lengths_ = nullptr;
    }
    delete[] host_lengths_;
    host_lengths_ = nullptr;
}
