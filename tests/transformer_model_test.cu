#include "core/cuda_check.h"
#include "core/transformer_model.h"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

constexpr std::size_t kBatch = 1;
constexpr std::size_t kSequence = 2;
constexpr std::size_t kVocabulary = 8;
constexpr std::size_t kHidden = 64;
constexpr std::size_t kIntermediate = 64;
constexpr std::size_t kQueryHeads = 2;
constexpr std::size_t kKvHeads = 1;
constexpr std::size_t kHeadDim = 32;

std::vector<float> values(const std::size_t count, const float phase) {
    std::vector<float> result(count);
    for (std::size_t index = 0; index < count; ++index) {
        result[index] = 0.1F * std::sin(phase + 0.17F * static_cast<float>(index));
    }
    return result;
}

void expect_invalid_argument(const auto& function) {
    try {
        function();
    } catch (const std::invalid_argument&) {
        return;
    }
    throw std::runtime_error("Transformer model test: invalid input was accepted");
}

struct Fixture {
    TransformerModelOptions options{kVocabulary, 1,
        {kHidden, kIntermediate, kQueryHeads, kKvHeads, kHeadDim, kHeadDim,
         1.0e-5F, true, {ComputeType::F32}}};
    Tensor embedding{{kVocabulary, kHidden}, Dtype::F16};
    Tensor attention_norm{{kHidden}, Dtype::F16};
    Tensor q{{kHidden, kHidden}, Dtype::F16};
    Tensor k{{kKvHeads * kHeadDim, kHidden}, Dtype::F16};
    Tensor v{{kKvHeads * kHeadDim, kHidden}, Dtype::F16};
    Tensor o{{kHidden, kHidden}, Dtype::F16};
    Tensor ffn_norm{{kHidden}, Dtype::F16};
    Tensor gate{{kIntermediate, kHidden}, Dtype::F16};
    Tensor up{{kIntermediate, kHidden}, Dtype::F16};
    Tensor down{{kHidden, kIntermediate}, Dtype::F16};
    Tensor final_norm{{kHidden}, Dtype::F16};
    Tensor lm_head{{kVocabulary, kHidden}, Dtype::F16};
    Tensor cosine{{kSequence, kHeadDim / 2}, Dtype::F16};
    Tensor sine{{kSequence, kHeadDim / 2}, Dtype::F16};
    Tensor logits{{kBatch * kSequence, kVocabulary}, Dtype::F16};
    TransformerModelWorkspace workspace{options, kBatch, kSequence, Dtype::F16};
    std::vector<TransformerBlockWeights> layers;

    Fixture() {
        embedding.copy_from_host(values(embedding.numel(), 0.1F));
        attention_norm.copy_from_host(values(attention_norm.numel(), 0.2F));
        q.copy_from_host(values(q.numel(), 0.3F));
        k.copy_from_host(values(k.numel(), 0.4F));
        v.copy_from_host(values(v.numel(), 0.5F));
        o.copy_from_host(values(o.numel(), 0.6F));
        ffn_norm.copy_from_host(values(ffn_norm.numel(), 0.7F));
        gate.copy_from_host(values(gate.numel(), 0.8F));
        up.copy_from_host(values(up.numel(), 0.9F));
        down.copy_from_host(values(down.numel(), 1.0F));
        final_norm.copy_from_host(values(final_norm.numel(), 1.1F));
        lm_head.copy_from_host(values(lm_head.numel(), 1.2F));
        cosine.copy_from_host(std::vector<float>(cosine.numel(), 1.0F));
        sine.copy_from_host(std::vector<float>(sine.numel(), 0.0F));
        layers.push_back({attention_norm, q, k, v, o, ffn_norm, gate, up, down});
    }

    [[nodiscard]] TransformerModelWeights weights() const {
        return {embedding, layers, final_norm, lm_head};
    }
};

void test_forward() {
    Fixture fixture;
    bpe::TokenId* device_ids = nullptr;
    CUDA_CHECK(cudaMalloc(&device_ids, kSequence * sizeof(bpe::TokenId)));
    try {
        const std::vector<bpe::TokenId> ids{1, 3};
        embedding_upload_token_ids(device_ids, ids);
        const CublasLtContext context;
        transformer_model_forward(fixture.logits, device_ids, kBatch, kSequence,
            fixture.weights(), fixture.workspace, fixture.cosine, fixture.sine,
            context, nullptr, fixture.options);
        CUDA_CHECK(cudaDeviceSynchronize());
        std::vector<float> actual(fixture.logits.numel());
        fixture.logits.copy_to_host(actual);
        for (const float value : actual) {
            if (!std::isfinite(value)) {
                throw std::runtime_error("Transformer model test: logits are not finite");
            }
        }
    } catch (...) {
        cudaFree(device_ids);
        throw;
    }
    CUDA_CHECK(cudaFree(device_ids));
}

void test_validation() {
    Fixture fixture;
    const CublasLtContext context;
    expect_invalid_argument([&] {
        transformer_model_forward(fixture.logits, nullptr, kBatch, kSequence,
            fixture.weights(), fixture.workspace, fixture.cosine, fixture.sine,
            context, nullptr, fixture.options);
    });
}

} // namespace

int main() {
    try {
        test_forward();
        test_validation();
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return EXIT_FAILURE;
    }
    std::cout << "Transformer model tests passed.\n";
    return EXIT_SUCCESS;
}
