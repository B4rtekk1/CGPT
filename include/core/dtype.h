#pragma once

#include <cublasLt.h>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string_view>

enum class Dtype: std::uint8_t {
    F16,
    BF16,
    F32,
    I32
};

[[nodiscard]] constexpr bool is_valid_dtype(const Dtype dtype) noexcept {
    switch (dtype) {
        case Dtype::F16:
        case Dtype::BF16:
        case Dtype::F32:
        case Dtype::I32:
            return true;
    }
    return false;
}

constexpr std::size_t dtype_size(Dtype dtype) {
    switch (dtype) {
        case Dtype::F16:
        case Dtype::BF16:
            return 2;
        case Dtype::F32:
        case Dtype::I32:
            return 4;
    }
    throw std::invalid_argument("Dtype not supported");
}

[[nodiscard]] constexpr bool is_floating_point(const Dtype dtype) noexcept {
    return dtype == Dtype::F16 ||
           dtype == Dtype::BF16 ||
           dtype == Dtype::F32;
}

[[nodiscard]] constexpr bool is_integral(const Dtype dtype) noexcept {
    return dtype == Dtype::I32;
}

[[nodiscard]] constexpr std::string_view dtype_name(const Dtype dtype) {
    switch (dtype) {
        case Dtype::F16:  return "F16";
        case Dtype::BF16: return "BF16";
        case Dtype::F32:  return "F32";
        case Dtype::I32:  return "I32";
    }
    throw std::invalid_argument("Dtype has no name");
}

[[nodiscard]] constexpr Dtype dtype_from_name(const std::string_view name) {
    if (name == "F16") return Dtype::F16;
    if (name == "BF16") return Dtype::BF16;
    if (name == "F32") return Dtype::F32;
    if (name == "I32") return Dtype::I32;
    throw std::invalid_argument("Unknown dtype name");
}

[[nodiscard]] constexpr std::size_t dtype_bytes(
    const std::size_t element_count,
    const Dtype dtype
) {
    if (element_count > std::numeric_limits<std::size_t>::max() / dtype_size(dtype)) {
        throw std::invalid_argument("Dtype byte size overflows");
    }
    return element_count * dtype_size(dtype);
}

[[nodiscard]] constexpr cudaDataType_t to_cuda_dtype(const Dtype dtype) {
    switch (dtype) {
        case Dtype::F16:  return CUDA_R_16F;
        case Dtype::BF16: return CUDA_R_16BF;
        case Dtype::F32:  return CUDA_R_32F;
        case Dtype::I32:  return CUDA_R_32I;
    }

    throw std::invalid_argument("Dtype has no CUDA mapping");
}

// Backwards-compatible spelling retained for existing callers.
[[deprecated("use to_cuda_dtype")]]
[[nodiscard]] constexpr cudaDataType_t to_cuda_Dtype(const Dtype dtype) {
    return to_cuda_dtype(dtype);
}

enum class ComputeType {
    F32,
    TF32
};

[[nodiscard]] constexpr bool is_valid_compute_type(const ComputeType type) noexcept {
    return type == ComputeType::F32 || type == ComputeType::TF32;
}

[[nodiscard]] constexpr std::string_view compute_type_name(const ComputeType type) {
    switch (type) {
        case ComputeType::F32:  return "F32";
        case ComputeType::TF32: return "TF32";
    }
    throw std::invalid_argument("ComputeType has no name");
}

[[nodiscard]] constexpr ComputeType compute_type_from_name(const std::string_view name) {
    if (name == "F32") return ComputeType::F32;
    if (name == "TF32") return ComputeType::TF32;
    throw std::invalid_argument("Unknown compute type name");
}

[[nodiscard]] constexpr cublasComputeType_t to_cublas_compute_type(
    const ComputeType type
) {
    switch (type) {
        case ComputeType::F32:
            return CUBLAS_COMPUTE_32F;

        case ComputeType::TF32:
            return CUBLAS_COMPUTE_32F_FAST_TF32;
    }

    throw std::invalid_argument("Unknown ComputeType");
}
