#pragma once

#include <cuda_runtime_api.h>
#include <algorithm>
#include <cstddef>

namespace cuda_autotune {

/** Returns a warp-aligned block size; zero-valued public options mean auto. */
inline int embedding_block_size(const std::size_t token_count,
                               const std::size_t hidden_size) {
    cudaDeviceProp properties{};
    int device = 0;
    if (cudaGetDevice(&device) != cudaSuccess ||
        cudaGetDeviceProperties(&properties, device) != cudaSuccess) {
        return 128;
    }
    int block = hidden_size <= 128 ? 128 : hidden_size <= 1024 ? 256 : 512;
    if (token_count < static_cast<std::size_t>(properties.multiProcessorCount)) {
        block = std::min(block, 256);
    }
    if (properties.major >= 9 && hidden_size >= 2048) {
        block = 512;
    }
    return std::clamp(block, 32, 1024);
}

/** Selects the compile-time attention tile best suited to the active GPU. */
inline int attention_block_m() {
    cudaDeviceProp properties{};
    int device = 0;
    if (cudaGetDevice(&device) != cudaSuccess ||
        cudaGetDeviceProperties(&properties, device) != cudaSuccess) {
        return 32;
    }
    // Hopper has substantially more shared memory and register throughput;
    // the larger tile reduces grid and global-memory traffic.  Keep the
    // existing conservative tile on other architectures.
    return properties.major >= 9 ? 64 : 32;
}

} // namespace cuda_autotune
