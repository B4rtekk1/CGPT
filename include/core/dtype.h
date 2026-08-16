#pragma once

#include <cublasLt.h>

#include <limits>
#include <stdexcept>
#include <string_view>

/**
 * @brief Element data types supported by the tensor and cuBLAS layers.
 */
enum class Dtype: std::uint8_t {
    /** @brief IEEE 754 half-precision floating point. */
    F16,
    /** @brief Brain floating point 16-bit floating point. */
    BF16,
    /** @brief IEEE 754 single-precision floating point. */
    F32,
    /** @brief Signed 32-bit integer. */
    I32
};

/**
 * @brief Checks whether a data type is recognized by the library.
 * @param dtype Data type to validate.
 * @return `true` if @p dtype is supported; otherwise `false`.
 */
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

/**
 * @brief Returns the storage size of one element of a data type.
 * @param dtype Data type whose element size is requested.
 * @return Number of bytes occupied by one element.
 * @throws std::invalid_argument If @p dtype is not supported.
 */
constexpr std::size_t dtype_size(const Dtype dtype) {
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

/**
 * @brief Checks whether a data type represents floating-point values.
 * @param dtype Data type to inspect.
 * @return `true` for F16, BF16, and F32; otherwise `false`.
 */
[[nodiscard]] constexpr bool is_floating_point(const Dtype dtype) noexcept {
    return dtype == Dtype::F16 ||
           dtype == Dtype::BF16 ||
           dtype == Dtype::F32;
}

/**
 * @brief Checks whether a data type represents integral values.
 * @param dtype Data type to inspect.
 * @return `true` for I32; otherwise `false`.
 */
[[nodiscard]] constexpr bool is_integral(const Dtype dtype) noexcept {
    return dtype == Dtype::I32;
}

/**
 * @brief Returns the canonical name of a data type.
 * @param dtype Data type to convert to text.
 * @return String view containing the type name.
 * @throws std::invalid_argument If @p dtype has no recognized name.
 */
[[nodiscard]] constexpr std::string_view dtype_name(const Dtype dtype) {
    switch (dtype) {
        case Dtype::F16:  return "F16";
        case Dtype::BF16: return "BF16";
        case Dtype::F32:  return "F32";
        case Dtype::I32:  return "I32";
    }
    throw std::invalid_argument("Dtype has no name");
}

/**
 * @brief Parses a canonical data type name.
 * @param name Type name such as `F16`, `BF16`, `F32`, or `I32`.
 * @return The corresponding @ref Dtype value.
 * @throws std::invalid_argument If @p name is unknown.
 */
[[nodiscard]] constexpr Dtype dtype_from_name(const std::string_view name) {
    if (name == "F16") return Dtype::F16;
    if (name == "BF16") return Dtype::BF16;
    if (name == "F32") return Dtype::F32;
    if (name == "I32") return Dtype::I32;
    throw std::invalid_argument("Unknown dtype name");
}

/**
 * @brief Calculates the number of bytes required for typed elements.
 * @param element_count Number of elements.
 * @param dtype Data type of the elements.
 * @return `element_count * dtype_size(dtype)`.
 * @throws std::invalid_argument If the result would overflow `std::size_t` or
 *         if @p dtype is unsupported.
 */
[[nodiscard]] constexpr std::size_t dtype_bytes(
    const std::size_t element_count,
    const Dtype dtype
) {
    if (element_count > std::numeric_limits<std::size_t>::max() / dtype_size(dtype)) {
        throw std::invalid_argument("Dtype byte size overflows");
    }
    return element_count * dtype_size(dtype);
}

/**
 * @brief Converts a library data type to its CUDA data type representation.
 * @param dtype Data type to convert.
 * @return Corresponding CUDA Runtime data type.
 * @throws std::invalid_argument If @p dtype has no CUDA mapping.
 */
[[nodiscard]] constexpr cudaDataType_t to_cuda_dtype(const Dtype dtype) {
    switch (dtype) {
        case Dtype::F16:  return CUDA_R_16F;
        case Dtype::BF16: return CUDA_R_16BF;
        case Dtype::F32:  return CUDA_R_32F;
        case Dtype::I32:  return CUDA_R_32I;
    }

    throw std::invalid_argument("Dtype has no CUDA mapping");
}

/**
 * @brief Matrix-computation modes supported by the cuBLAS layer.
 */
enum class ComputeType {
    /** @brief Standard 32-bit floating-point computation. */
    F32,
    /** @brief Tensor Core-oriented TF32 computation using 32-bit inputs. */
    TF32
};

/**
 * @brief Checks whether a compute type is supported.
 * @param type Compute type to validate.
 * @return `true` if @p type is supported; otherwise `false`.
 */
[[nodiscard]] constexpr bool is_valid_compute_type(const ComputeType type) noexcept {
    return type == ComputeType::F32 || type == ComputeType::TF32;
}

/**
 * @brief Returns the canonical name of a compute type.
 * @param type Compute type to convert to text.
 * @return String view containing the compute type name.
 * @throws std::invalid_argument If @p type has no recognized name.
 */
[[nodiscard]] constexpr std::string_view compute_type_name(const ComputeType type) {
    switch (type) {
        case ComputeType::F32:  return "F32";
        case ComputeType::TF32: return "TF32";
    }
    throw std::invalid_argument("ComputeType has no name");
}

/**
 * @brief Parses a canonical compute type name.
 * @param name Compute type name, either `F32` or `TF32`.
 * @return The corresponding @ref ComputeType value.
 * @throws std::invalid_argument If @p name is unknown.
 */
[[nodiscard]] constexpr ComputeType compute_type_from_name(const std::string_view name) {
    if (name == "F32") return ComputeType::F32;
    if (name == "TF32") return ComputeType::TF32;
    throw std::invalid_argument("Unknown compute type name");
}

/**
 * @brief Converts a compute type to its cuBLAS computation mode.
 * @param type Compute type to convert.
 * @return Corresponding cuBLAS compute type.
 * @throws std::invalid_argument If @p type has no cuBLAS mapping.
 */
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