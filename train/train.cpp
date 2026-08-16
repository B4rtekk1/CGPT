#include "core/cuda_check.h"
#include "core/cuda_graph.h"
#include "core/generation.h"
#include "core/transformer_model.h"
#include "core/weight_initialization.h"
#include "data/dataset_loader.h"
#include "ops/cross_entropy.h"
#include "ops/embedding.h"
#include "optim/adamw.h"
#include "tokenizer/bpe_tokenizer.h"
#include "utils/progress_bar.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {
constexpr std::size_t kMiB = 1024 * 1024;

struct Arguments {
    std::filesystem::path input = "data/tokenizer_100mb.txt";
    std::filesystem::path tokenizer_output = "train/tokenizer_100mb.json";
    std::size_t vocab_size = 32'000;
    std::size_t batch_size = 16;
    std::size_t sequence_length = 1024;
    std::size_t epochs = 10;
    std::size_t max_steps = 0; // 0 means all batches from every requested epoch.
    float learning_rate = 1.0e-4F;
    std::string prompt = "The";
    std::size_t generate_tokens = 64;
};

class Stream {
public:
    Stream() { CUDA_CHECK(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking)); }
    Stream(const Stream&) = delete;
    Stream& operator=(const Stream&) = delete;
    ~Stream() noexcept {
        if (stream_ != nullptr) static_cast<void>(cudaStreamDestroy(stream_));
    }
    [[nodiscard]] cudaStream_t get() const noexcept { return stream_; }

private:
    cudaStream_t stream_ = nullptr;
};

Arguments parse_arguments(int argc, char** argv) {
    Arguments args;
    for (int i = 1; i < argc; ++i) {
        const std::string option = argv[i];
        const auto value = [&]() -> std::string {
            if (++i == argc) throw std::invalid_argument("Missing value for " + option);
            return argv[i];
        };
        if (option == "--input") args.input = value();
        else if (option == "--tokenizer") args.tokenizer_output = value();
        else if (option == "--vocab-size") args.vocab_size = std::stoull(value());
        else if (option == "--batch-size") args.batch_size = std::stoull(value());
        else if (option == "--sequence-length") args.sequence_length = std::stoull(value());
        else if (option == "--epochs") args.epochs = std::stoull(value());
        else if (option == "--max-steps") args.max_steps = std::stoull(value());
        else if (option == "--learning-rate") args.learning_rate = std::stof(value());
        else if (option == "--prompt") args.prompt = value();
        else if (option == "--generate-tokens") args.generate_tokens = std::stoull(value());
        else if (option == "--help") {
            std::cout << "Usage: cgpt_train [--input PATH] [--tokenizer PATH] [--vocab-size N] "
                         "[--batch-size N] [--sequence-length N] [--epochs N] [--max-steps N] "
                         "[--learning-rate N] [--prompt TEXT] [--generate-tokens N]\n";
            std::exit(EXIT_SUCCESS);
        } else throw std::invalid_argument("Unknown option: " + option);
    }
    if (args.vocab_size < 512 || args.batch_size == 0 || args.sequence_length < 2 || args.epochs == 0 ||
        !std::isfinite(args.learning_rate) || args.learning_rate <= 0.0F)
        throw std::invalid_argument("Invalid training dimensions");
    return args;
}

struct LayerStorage {
    Tensor attention_norm, q, k, v, o, ffn_norm, gate, up, down;
    Tensor g_attention_norm, g_q, g_k, g_v, g_o, g_ffn_norm, g_gate, g_up, g_down;
    explicit LayerStorage(const TransformerBlockOptions& options)
        : attention_norm({options.hidden_size}, Dtype::F16), q({options.hidden_size, options.hidden_size}, Dtype::F16),
          k({options.num_kv_heads * options.head_dim, options.hidden_size}, Dtype::F16),
          v({options.num_kv_heads * options.head_dim, options.hidden_size}, Dtype::F16),
          o({options.hidden_size, options.hidden_size}, Dtype::F16), ffn_norm({options.hidden_size}, Dtype::F16),
          gate({options.intermediate_size, options.hidden_size}, Dtype::F16), up({options.intermediate_size, options.hidden_size}, Dtype::F16),
          down({options.hidden_size, options.intermediate_size}, Dtype::F16),
          g_attention_norm({options.hidden_size}, Dtype::F16), g_q({options.hidden_size, options.hidden_size}, Dtype::F16),
          g_k({options.num_kv_heads * options.head_dim, options.hidden_size}, Dtype::F16),
          g_v({options.num_kv_heads * options.head_dim, options.hidden_size}, Dtype::F16),
          g_o({options.hidden_size, options.hidden_size}, Dtype::F16), g_ffn_norm({options.hidden_size}, Dtype::F16),
          g_gate({options.intermediate_size, options.hidden_size}, Dtype::F16), g_up({options.intermediate_size, options.hidden_size}, Dtype::F16),
          g_down({options.hidden_size, options.intermediate_size}, Dtype::F16) {}
};

struct ModelStorage {
    TransformerModelOptions options;
    Tensor embedding, final_norm, lm_head, g_embedding, g_final_norm, g_lm_head;
    std::vector<LayerStorage> layers;
    std::vector<TransformerBlockWeights> weight_layers;
    std::vector<MutableTransformerBlockWeights> mutable_layers;
    std::vector<TransformerBlockGradients> gradient_layers;

    explicit ModelStorage(std::size_t vocab)
        : options{vocab, 2, {256, 768, 4, 4, 64, 64, 1.0e-5F, true, {ComputeType::F32}}},
          embedding({vocab, 256}, Dtype::F16), final_norm({256}, Dtype::F16), lm_head({vocab, 256}, Dtype::F16),
          g_embedding({vocab, 256}, Dtype::F16), g_final_norm({256}, Dtype::F16), g_lm_head({vocab, 256}, Dtype::F16) {
        layers.reserve(options.num_layers);
        for (std::size_t i = 0; i < options.num_layers; ++i) layers.emplace_back(options.block_options);
        for (auto& layer : layers) {
            weight_layers.push_back({layer.attention_norm, layer.q, layer.k, layer.v, layer.o, layer.ffn_norm, layer.gate, layer.up, layer.down});
            mutable_layers.push_back({layer.attention_norm, layer.q, layer.k, layer.v, layer.o, layer.ffn_norm, layer.gate, layer.up, layer.down});
            gradient_layers.push_back({layer.g_attention_norm, layer.g_q, layer.g_k, layer.g_v, layer.g_o, layer.g_ffn_norm, layer.g_gate, layer.g_up, layer.g_down});
        }
        initialize_transformer_weights({embedding, mutable_layers, final_norm, lm_head}, options);
    }
    [[nodiscard]] TransformerModelWeights weights() const { return {embedding, weight_layers, final_norm, lm_head}; }
    [[nodiscard]] TransformerModelGradients gradients() { return {g_embedding, gradient_layers, g_final_norm, g_lm_head}; }
};

void append_parameters(ModelStorage& model, std::vector<std::pair<Tensor*, Tensor*>>& result) {
    result.clear();
    result.reserve(3 + model.layers.size() * 9);
    result.emplace_back(&model.embedding, &model.g_embedding);
    result.emplace_back(&model.final_norm, &model.g_final_norm);
    result.emplace_back(&model.lm_head, &model.g_lm_head);
    for (auto& l : model.layers) {
        result.emplace_back(&l.attention_norm, &l.g_attention_norm);
        result.emplace_back(&l.q, &l.g_q);
        result.emplace_back(&l.k, &l.g_k);
        result.emplace_back(&l.v, &l.g_v);
        result.emplace_back(&l.o, &l.g_o);
        result.emplace_back(&l.ffn_norm, &l.g_ffn_norm);
        result.emplace_back(&l.gate, &l.g_gate);
        result.emplace_back(&l.up, &l.g_up);
        result.emplace_back(&l.down, &l.g_down);
    }
}

std::pair<Tensor, Tensor> rotary_cache(std::size_t sequence_length, std::size_t rotary_dim) {
    const std::size_t cache_size = sequence_length * (rotary_dim / 2);
    std::vector<float> cosine(cache_size);
    std::vector<float> sine(cache_size);
    for (std::size_t pos = 0; pos < sequence_length; ++pos) for (std::size_t i = 0; i < rotary_dim / 2; ++i) {
        const float theta = static_cast<float>(pos) / std::pow(10000.0F, 2.0F * static_cast<float>(i) / static_cast<float>(rotary_dim));
        cosine[pos * (rotary_dim / 2) + i] = std::cos(theta); sine[pos * (rotary_dim / 2) + i] = std::sin(theta);
    }
    Tensor cos_cache({sequence_length, rotary_dim / 2}, Dtype::F16), sin_cache({sequence_length, rotary_dim / 2}, Dtype::F16);
    cos_cache.copy_from_host(cosine); sin_cache.copy_from_host(sine);
    return {std::move(cos_cache), std::move(sin_cache)};
}
}

int main(int argc, char** argv) {
    try {
        const Arguments args = parse_arguments(argc, argv);
        if (!std::filesystem::exists(args.input)) throw std::runtime_error("Missing input: " + args.input.string());
        if (std::filesystem::file_size(args.input) < 100 * kMiB) throw std::runtime_error("Input must contain at least 100 MiB");
        std::ifstream input(args.input, std::ios::binary);
        std::string text(100 * kMiB, '\0'); input.read(text.data(), static_cast<std::streamsize>(text.size()));

        bpe::TrainerConfig tokenizer_config; tokenizer_config.vocab_size = args.vocab_size;
        std::cout << "Training BPE tokenizer on 100 MiB...\n";
        const bpe::BpeTokenizer tokenizer = bpe::BpeTokenizer::train(std::vector<std::string>{text}, tokenizer_config);
        if (!args.tokenizer_output.parent_path().empty())
            std::filesystem::create_directories(args.tokenizer_output.parent_path());
        tokenizer.save(args.tokenizer_output);

        ProgressBar tokenization_bar(text.size(), "Tokenizing", static_cast<double>(kMiB), "MiB/s");
        const std::size_t worker_count = std::max<std::size_t>(
            1, std::thread::hardware_concurrency());
        const std::size_t chunk_size = std::max<std::size_t>(
            1, (text.size() + worker_count - 1) / worker_count);
        std::vector<std::string> chunks;
        chunks.reserve(worker_count);
        for (std::size_t begin = 0; begin < text.size();) {
            std::size_t end = std::min(text.size(), begin + chunk_size);
            if (end < text.size()) {
                const std::size_t newline = text.find('\n', end);
                if (newline != std::string::npos) end = newline + 1;
            }
            chunks.emplace_back(text.substr(begin, end - begin));
            begin = end;
        }
        const std::vector<std::vector<bpe::TokenId>> encoded_chunks =
            tokenizer.encode_batch(chunks, worker_count);
        std::vector<bpe::TokenId> tokens;
        for (const auto& chunk : encoded_chunks)
            tokens.insert(tokens.end(), chunk.begin(), chunk.end());
        tokenization_bar.update(text.size()); tokenization_bar.finish();
        std::cout << "Vocabulary: " << tokenizer.vocab_size() << ", tokens: " << tokens.size() << '\n';

        data::DataLoaderConfig loader_config{args.batch_size, args.sequence_length, true, true, 42};
        data::DatasetLoader loader(std::make_shared<const data::TokenDataset>(tokens), loader_config);
        if (loader.batch_count() == 0) throw std::runtime_error("Not enough tokens for one batch");
        const std::size_t total_steps = args.max_steps ? std::min(args.max_steps, loader.batch_count() * args.epochs) : loader.batch_count() * args.epochs;
        ModelStorage model(tokenizer.vocab_size());
        auto [cos_cache, sin_cache] = rotary_cache(args.sequence_length, model.options.block_options.rotary_dim);
        TransformerModelWorkspace forward_workspace(model.options, args.batch_size, args.sequence_length, Dtype::F16);
        TransformerModelBackwardWorkspace backward_workspace(model.options, args.batch_size, args.sequence_length, Dtype::F16);
        // CCE currently accepts only F32 activations and weights.  This training
        // configuration uses F16 plus cuBLASLt for lm_head, which is faster and
        // avoids materializing/converting a second F32 model copy.
        Tensor logits({args.batch_size * args.sequence_length, tokenizer.vocab_size()}, Dtype::F16);
        Tensor loss({1}, Dtype::F32), grad_logits({args.batch_size * args.sequence_length, tokenizer.vocab_size()}, Dtype::F16);
        data::DeviceBatch device_batch;
        std::vector<std::pair<Tensor*, Tensor*>> parameters; append_parameters(model, parameters);
        std::vector<AdamWState> optimizer; for (const auto& [parameter, gradient] : parameters) optimizer.push_back(AdamWState::for_parameter(*parameter));
        std::vector<AdamWBatchEntry> optimizer_entries;
        optimizer_entries.reserve(parameters.size());
        for (std::size_t i = 0; i < parameters.size(); ++i)
            optimizer_entries.push_back({parameters[i].first, parameters[i].second, &optimizer[i]});
        AdamWWorkspace optimizer_workspace;
        AdamWOptions optimizer_options;
        optimizer_options.learning_rate = args.learning_rate;
        optimizer_options.weight_decay = 0.01F;
        optimizer_options.max_grad_norm = 1.0F;
        const CublasLtContext cublas;
        const Stream stream;
        const cudaStream_t cuda_stream = stream.get();
        CudaGraph training_graph;
        ProgressBar progress(total_steps, "Training");
        std::array<data::Batch, 2> host_batches;
        std::size_t host_batch_index = 0;
        data::Batch& captured_batch = host_batches[host_batch_index];
        std::size_t step = 0;

        // All batches have the configured fixed shape (drop_last=true).  Allocate
        // the device buffers and tune cuBLASLt plans before capture, so graph
        // replay contains only steady-state GPU work.
        if (!loader.next(captured_batch)) throw std::runtime_error("Not enough tokens for one batch");
        data::upload_batch(captured_batch, device_batch, cuda_stream);
        const auto* const captured_inputs = static_cast<const bpe::TokenId*>(device_batch.input_ids.data());
        const auto* const captured_targets = static_cast<const bpe::TokenId*>(device_batch.target_ids.data());
        const auto enqueue_model_step = [&] {
            transformer_model_forward(logits, captured_inputs, args.batch_size, args.sequence_length,
                model.weights(), forward_workspace, cos_cache, sin_cache, cublas, cuda_stream, model.options);
            cross_entropy_forward_backward(loss, grad_logits, logits, captured_targets, captured_batch.token_count(), cuda_stream);
            transformer_model_backward(grad_logits, captured_inputs, args.batch_size, args.sequence_length,
                model.weights(), model.gradients(), forward_workspace, backward_workspace, cos_cache, sin_cache,
                cublas, cuda_stream, model.options);
        };
        enqueue_model_step();
        CUDA_CHECK(cudaStreamSynchronize(cuda_stream));
        training_graph.capture(cuda_stream, enqueue_model_step);
        training_graph.upload(cuda_stream);

        for (std::size_t epoch = 0; epoch < args.epochs && step < total_steps; ++epoch) {
            do {
                if (device_batch.input_ids.data() != captured_inputs || device_batch.target_ids.data() != captured_targets)
                    throw std::runtime_error("Training batch storage changed after CUDA graph capture");
                training_graph.launch(cuda_stream);
                adamw_step_many_async(optimizer_entries, optimizer_options, optimizer_workspace, cuda_stream);
                ++step; progress.update(step);
                if (step % 50 == 0 || step == total_steps) {
                    if (!adamw_check(optimizer_workspace, cuda_stream))
                        throw std::runtime_error("Non-finite gradient at step " + std::to_string(step));
                    std::vector<float> host_loss(1); loss.copy_to_host(host_loss);
                    std::cout << " loss=" << host_loss[0] << std::flush;
                }
                if (step < total_steps) {
                    host_batch_index ^= 1U;
                    if (!loader.next(host_batches[host_batch_index])) break;
                    data::upload_batch(host_batches[host_batch_index], device_batch, cuda_stream);
                }
            } while (step < total_steps);
            if (epoch + 1 < args.epochs && step < total_steps) {
                loader.reset();
                host_batch_index ^= 1U;
                if (!loader.next(host_batches[host_batch_index])) break;
                data::upload_batch(host_batches[host_batch_index], device_batch, cuda_stream);
            }
        }
        CUDA_CHECK(cudaStreamSynchronize(cuda_stream));
        progress.finish();
        GenerationOptions generation_options;
        generation_options.max_new_tokens = args.generate_tokens;
        generation_options.max_context_tokens = args.sequence_length;
        generation_options.temperature = 0.8F;
        generation_options.top_k = 40;
        generation_options.seed = 42;
        std::cout << "\nGenerated text:\n" << generate_text(tokenizer, args.prompt, model.weights(), cos_cache,
            sin_cache, cublas, model.options, generation_options) << '\n';
        std::cout << "Completed " << step << " optimizer steps. Tokenizer saved to " << args.tokenizer_output << '\n';
        return EXIT_SUCCESS;
    } catch (const std::exception& error) { std::cerr << "Training failed: " << error.what() << '\n'; return EXIT_FAILURE; }
}
