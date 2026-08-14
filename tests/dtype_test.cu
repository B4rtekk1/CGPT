#include "core/dtype.h"

#include <cassert>
#include <limits>
#include <stdexcept>

namespace {

template <typename Exception, typename Function>
void expects_throw(Function&& function) {
    bool thrown = false;
    try {
        function();
    } catch (const Exception&) {
        thrown = true;
    }
    assert(thrown);
}

void test_dtype_properties() {
    static_assert(dtype_size(Dtype::F16) == 2);
    static_assert(dtype_size(Dtype::BF16) == 2);
    static_assert(dtype_size(Dtype::F32) == 4);
    static_assert(dtype_size(Dtype::I32) == 4);

    assert(is_valid_dtype(Dtype::F16));
    assert(is_floating_point(Dtype::BF16));
    assert(is_floating_point(Dtype::F32));
    assert(!is_floating_point(Dtype::I32));
    assert(is_integral(Dtype::I32));
    assert(!is_integral(Dtype::F32));
}

void test_dtype_names_and_cuda_mappings() {
    assert(dtype_name(Dtype::F16) == "F16");
    assert(dtype_name(Dtype::BF16) == "BF16");
    assert(dtype_name(Dtype::F32) == "F32");
    assert(dtype_name(Dtype::I32) == "I32");
    assert(dtype_from_name("BF16") == Dtype::BF16);
    assert(dtype_from_name("I32") == Dtype::I32);

    assert(to_cuda_dtype(Dtype::F16) == CUDA_R_16F);
    assert(to_cuda_dtype(Dtype::BF16) == CUDA_R_16BF);
    assert(to_cuda_dtype(Dtype::F32) == CUDA_R_32F);
    assert(to_cuda_dtype(Dtype::I32) == CUDA_R_32I);
    assert(to_cuda_dtype(Dtype::F32) == CUDA_R_32F);
}

void test_checked_byte_size() {
    assert(dtype_bytes(0, Dtype::F32) == 0);
    assert(dtype_bytes(10, Dtype::F16) == 20);
    assert(dtype_bytes(3, Dtype::I32) == 12);

    const auto max_count = std::numeric_limits<std::size_t>::max() / dtype_size(Dtype::F32);
    assert(dtype_bytes(max_count, Dtype::F32) == max_count * sizeof(std::uint32_t));
    expects_throw<std::invalid_argument>([&] { (void)dtype_bytes(max_count + 1, Dtype::F32); });
    expects_throw<std::invalid_argument>([] {
        dtype_size(static_cast<Dtype>(std::numeric_limits<std::uint8_t>::max()));
    });
}

void test_compute_types() {
    assert(is_valid_compute_type(ComputeType::F32));
    assert(is_valid_compute_type(ComputeType::TF32));
    assert(compute_type_name(ComputeType::TF32) == "TF32");
    assert(compute_type_from_name("F32") == ComputeType::F32);
    assert(to_cublas_compute_type(ComputeType::F32) == CUBLAS_COMPUTE_32F);
    assert(to_cublas_compute_type(ComputeType::TF32) == CUBLAS_COMPUTE_32F_FAST_TF32);
    expects_throw<std::invalid_argument>([] { (void)dtype_from_name("float32"); });
    expects_throw<std::invalid_argument>([] { (void)compute_type_from_name("FP16"); });
}

} // namespace

int main() {
    test_dtype_properties();
    test_dtype_names_and_cuda_mappings();
    test_checked_byte_size();
    test_compute_types();
    return 0;
}
