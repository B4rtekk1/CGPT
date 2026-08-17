#pragma once

#include "core/transformer_model.h"
#include "core/weight_initialization.h"
#include "tokenizer/bpe_tokenizer.h"

#include <cstddef>
#include <filesystem>

/** Export the trained model as a Hugging Face-style safetensors bundle. */
void export_huggingface_model(
    const std::filesystem::path& output_directory,
    const bpe::BpeTokenizer& tokenizer,
    const TransformerModelWeights& weights,
    const TransformerModelOptions& options,
    std::size_t max_position_embeddings);

/** Load model.safetensors from a Hugging Face-style bundle into model weights. */
void load_huggingface_model(
    const std::filesystem::path& input_directory,
    const MutableTransformerModelWeights& weights);
