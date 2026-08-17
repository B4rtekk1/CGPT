#include "core/cuda_check.h"
#include "core/cuda_graph.h"
#include "core/generation.h"
#include "core/huggingface_export.h"
#include "core/transformer_model.h"
#include "core/weight_initialization.h"
#include "data/dataset_loader.h"
#include "ops/cross_entropy.h"
#include "ops/embedding.h"
#include "optim/adamw.h"
#include "optim/lr_scheduler.h"
#include "tokenizer/bpe_tokenizer.h"
#include "utils/progress_bar.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

namespace {
constexpr std::size_t kMiB = 1024 * 1024;
constexpr std::size_t kTokenizerTrainingBytes = std::size_t{1} << 30;

struct Arguments {
    std::filesystem::path input = "data/tokenizer_100mb.txt";
    std::filesystem::path tokenizer_output = "train/tokenizer_100mb.json";
    std::filesystem::path output_directory = "train/huggingface";
    std::optional<std::filesystem::path> load_directory;
    std::size_t vocab_size = 32'000;
    std::size_t batch_size = 16;
    std::size_t sequence_length = 1024;
    std::size_t epochs = 10;
    std::size_t max_steps = 0; // 0 means all batches from every requested epoch.
    std::size_t validation_interval = 500;
    std::size_t validation_batches = 64;
    float learning_rate = 1.0e-4F;
    float min_learning_rate = 0.0F;
    std::size_t warmup_steps = 0;
    float validation_fraction = 0.1F;
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
        else if (option == "--output-dir") args.output_directory = value();
        else if (option == "--load-dir") args.load_directory = value();
        else if (option == "--vocab-size") args.vocab_size = std::stoull(value());
        else if (option == "--batch-size") args.batch_size = std::stoull(value());
        else if (option == "--sequence-length") args.sequence_length = std::stoull(value());
        else if (option == "--epochs") args.epochs = std::stoull(value());
        else if (option == "--max-steps") args.max_steps = std::stoull(value());
        else if (option == "--validation-interval") args.validation_interval = std::stoull(value());
        else if (option == "--validation-batches") args.validation_batches = std::stoull(value());
        else if (option == "--learning-rate") args.learning_rate = std::stof(value());
        else if (option == "--min-learning-rate") args.min_learning_rate = std::stof(value());
        else if (option == "--warmup-steps") args.warmup_steps = std::stoull(value());
        else if (option == "--validation-fraction") args.validation_fraction = std::stof(value());
        else if (option == "--prompt") args.prompt = value();
        else if (option == "--generate-tokens") args.generate_tokens = std::stoull(value());
        else if (option == "--help") {
            std::cout << "Usage: cgpt_train [--input PATH] [--tokenizer PATH] [--vocab-size N] "
                         "[--output-dir PATH] "
                         "[--load-dir PATH] "
                         "[--batch-size N] [--sequence-length N] [--epochs N] [--max-steps N] "
                         "[--learning-rate N] [--min-learning-rate N] [--warmup-steps N] "
                         "[--validation-fraction N] [--validation-interval N] "
                         "[--validation-batches N] [--prompt TEXT] [--generate-tokens N]\n"
                         "Input may be plain text or FineWeb JSONL (the `text` field is used).\n";
            std::exit(EXIT_SUCCESS);
        } else throw std::invalid_argument("Unknown option: " + option);
    }
    if (args.vocab_size < 512 || args.batch_size == 0 || args.sequence_length < 2 || args.epochs == 0 ||
        !std::isfinite(args.learning_rate) || args.learning_rate <= 0.0F ||
        !std::isfinite(args.min_learning_rate) || args.min_learning_rate < 0.0F ||
        args.min_learning_rate > args.learning_rate ||
        args.validation_interval == 0 || args.validation_batches == 0)
        throw std::invalid_argument("Invalid training dimensions");
    if (!std::isfinite(args.validation_fraction) || args.validation_fraction <= 0.0F ||
        args.validation_fraction >= 1.0F)
        throw std::invalid_argument("validation-fraction must be between 0 and 1");
    return args;
}

/**
 * @brief Decodes the JSON string value belonging to the FineWeb `text` field.
 *
 * FineWeb-Edu is distributed as JSONL.  Keeping this small reader local to
 * the training executable avoids making the binary token dataset loader aware
 * of document formats while still handling escaped quotes, control characters,
 * and BMP Unicode escapes emitted by JSON writers.
 */
std::string json_text_field(std::string_view line) {
    const std::size_t key = line.find("\"text\"");
    if (key == std::string_view::npos) return {};
    std::size_t cursor = key + 6;
    while (cursor < line.size() && std::isspace(static_cast<unsigned char>(line[cursor]))) ++cursor;
    if (cursor >= line.size() || line[cursor++] != ':') return {};
    while (cursor < line.size() && std::isspace(static_cast<unsigned char>(line[cursor]))) ++cursor;
    if (cursor >= line.size() || line[cursor++] != '\"') return {};

    std::string result;
    result.reserve(line.size());
    while (cursor < line.size()) {
        const char character = line[cursor++];
        if (character == '\"') return result;
        if (character != '\\') {
            result.push_back(character);
            continue;
        }
        if (cursor >= line.size()) throw std::runtime_error("Invalid JSONL text field: dangling escape");
        switch (const char escaped = line[cursor++]) {
            case '\"': result.push_back('\"'); break;
            case '\\': result.push_back('\\'); break;
            case '/': result.push_back('/'); break;
            case 'b': result.push_back('\b'); break;
            case 'f': result.push_back('\f'); break;
            case 'n': result.push_back('\n'); break;
            case 'r': result.push_back('\r'); break;
            case 't': result.push_back('\t'); break;
            case 'u': {
                if (cursor + 4 > line.size()) throw std::runtime_error("Invalid JSONL Unicode escape");
                unsigned value = 0;
                for (int digit = 0; digit < 4; ++digit) {
                    const char hex = line[cursor++];
                    const int number = std::isdigit(static_cast<unsigned char>(hex)) ? hex - '0' :
                        (hex >= 'a' && hex <= 'f') ? hex - 'a' + 10 :
                        (hex >= 'A' && hex <= 'F') ? hex - 'A' + 10 : -1;
                    if (number < 0) throw std::runtime_error("Invalid JSONL Unicode escape");
                    value = value * 16U + static_cast<unsigned>(number);
                }
                if (value <= 0x7fU) result.push_back(static_cast<char>(value));
                else if (value <= 0x7ffU) {
                    result.push_back(static_cast<char>(0xc0U | (value >> 6U)));
                    result.push_back(static_cast<char>(0x80U | (value & 0x3fU)));
                } else {
                    result.push_back(static_cast<char>(0xe0U | (value >> 12U)));
                    result.push_back(static_cast<char>(0x80U | ((value >> 6U) & 0x3fU)));
                    result.push_back(static_cast<char>(0x80U | (value & 0x3fU)));
                }
                break;
            }
            default: throw std::runtime_error("Unsupported JSONL escape in text field");
        }
    }
    throw std::runtime_error("Invalid JSONL text field: unterminated string");
}

std::string load_training_text(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("Unable to open input: " + path.string());
    const bool jsonl = path.extension() == ".jsonl";
    if (!jsonl) {
        input.seekg(0, std::ios::end);
        const std::streampos end = input.tellg();
        if (end <= 0) throw std::runtime_error("Input is empty: " + path.string());
        if (static_cast<std::uintmax_t>(end) > std::numeric_limits<std::size_t>::max() ||
            static_cast<std::uintmax_t>(end) > static_cast<std::uintmax_t>(std::numeric_limits<std::streamsize>::max()))
            throw std::overflow_error("Input is too large to load into memory: " + path.string());
        std::string text(static_cast<std::size_t>(end), '\0');
        input.seekg(0, std::ios::beg);
        input.read(text.data(), static_cast<std::streamsize>(text.size()));
        if (!input) throw std::runtime_error("Unable to read input: " + path.string());
        return text;
    }

    std::string text;
    std::error_code error;
    const std::uintmax_t file_bytes = std::filesystem::file_size(path, error);
    if (!error && file_bytes <= std::numeric_limits<std::size_t>::max())
        text.reserve(static_cast<std::size_t>(file_bytes));
    std::string line;
    while (std::getline(input, line)) {
        const std::string document = json_text_field(line);
        if (document.empty()) continue;
        if (!text.empty()) text.push_back('\n');
        text.append(document);
    }
    if (text.empty()) throw std::runtime_error("JSONL input contains no text fields: " + path.string());
    return text;
}

struct LayerStorage {
    Tensor attention_norm, q, k, q_norm, k_norm, v, o, ffn_norm, gate, up, down;
    Tensor g_attention_norm, g_q, g_k, g_q_norm, g_k_norm, g_v, g_o, g_ffn_norm, g_gate, g_up, g_down;
    explicit LayerStorage(const TransformerBlockOptions& options)
        : attention_norm({options.hidden_size}, Dtype::F16), q({options.hidden_size, options.hidden_size}, Dtype::F16),
          k({options.num_kv_heads * options.head_dim, options.hidden_size}, Dtype::F16),
          q_norm({options.head_dim}, Dtype::F16), k_norm({options.head_dim}, Dtype::F16),
          v({options.num_kv_heads * options.head_dim, options.hidden_size}, Dtype::F16),
          o({options.hidden_size, options.hidden_size}, Dtype::F16), ffn_norm({options.hidden_size}, Dtype::F16),
          gate({options.intermediate_size, options.hidden_size}, Dtype::F16), up({options.intermediate_size, options.hidden_size}, Dtype::F16),
          down({options.hidden_size, options.intermediate_size}, Dtype::F16),
          g_attention_norm({options.hidden_size}, Dtype::F16), g_q({options.hidden_size, options.hidden_size}, Dtype::F16),
          g_k({options.num_kv_heads * options.head_dim, options.hidden_size}, Dtype::F16),
          g_q_norm({options.head_dim}, Dtype::F16), g_k_norm({options.head_dim}, Dtype::F16),
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
        : options{vocab, 2, {256, 768, 4, 1, 64, 64, 1.0e-5F, true, {ComputeType::F32}}},
          embedding({vocab, 256}, Dtype::F16), final_norm({256}, Dtype::F16), lm_head({vocab, 256}, Dtype::F16),
          g_embedding({vocab, 256}, Dtype::F16), g_final_norm({256}, Dtype::F16), g_lm_head({vocab, 256}, Dtype::F16) {
        layers.reserve(options.num_layers);
        for (std::size_t i = 0; i < options.num_layers; ++i) layers.emplace_back(options.block_options);
        for (auto& layer : layers) {
            weight_layers.push_back({layer.attention_norm, layer.q, layer.k, layer.q_norm, layer.k_norm, layer.v, layer.o, layer.ffn_norm, layer.gate, layer.up, layer.down});
            mutable_layers.push_back({layer.attention_norm, layer.q, layer.k, layer.q_norm, layer.k_norm, layer.v, layer.o, layer.ffn_norm, layer.gate, layer.up, layer.down});
            gradient_layers.push_back({layer.g_attention_norm, layer.g_q, layer.g_k, layer.g_q_norm, layer.g_k_norm, layer.g_v, layer.g_o, layer.g_ffn_norm, layer.g_gate, layer.g_up, layer.g_down});
        }
        initialize_transformer_weights({embedding, mutable_layers, final_norm, lm_head}, options);
    }
    [[nodiscard]] TransformerModelWeights weights() const { return {embedding, weight_layers, final_norm, lm_head}; }
    [[nodiscard]] TransformerModelGradients gradients() { return {g_embedding, gradient_layers, g_final_norm, g_lm_head}; }
};

void append_parameters(ModelStorage& model, std::vector<std::pair<Tensor*, Tensor*>>& result) {
    result.clear();
    result.reserve(3 + model.layers.size() * 11);
    result.emplace_back(&model.embedding, &model.g_embedding);
    result.emplace_back(&model.final_norm, &model.g_final_norm);
    result.emplace_back(&model.lm_head, &model.g_lm_head);
    for (auto& l : model.layers) {
        result.emplace_back(&l.attention_norm, &l.g_attention_norm);
        result.emplace_back(&l.q, &l.g_q);
        result.emplace_back(&l.k, &l.g_k);
        result.emplace_back(&l.q_norm, &l.g_q_norm);
        result.emplace_back(&l.k_norm, &l.g_k_norm);
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
        std::string text = load_training_text(args.input);

        bpe::BpeTokenizer tokenizer = [&] {
            if (args.load_directory) {
                std::cout << "Loading tokenizer from " << *args.load_directory << "...\n";
                return bpe::BpeTokenizer::load(*args.load_directory / "tokenizer.json");
            }
            bpe::TrainerConfig tokenizer_config; tokenizer_config.vocab_size = args.vocab_size;
            const std::size_t tokenizer_bytes = std::min(text.size(), kTokenizerTrainingBytes);
            // Keep the tokenizer corpus bounded while retaining the complete
            // input in `text` for the subsequent full-corpus tokenization.
            std::string tokenizer_text(text.data(), tokenizer_bytes);
            std::cout << "Training BPE tokenizer on "
                      << static_cast<double>(tokenizer_bytes) / static_cast<double>(kMiB)
                      << " MiB (full input will be tokenized)...\n";
            return bpe::BpeTokenizer::train(
                std::vector<std::string>{std::move(tokenizer_text)}, tokenizer_config);
        }();
        if (!args.tokenizer_output.parent_path().empty())
            std::filesystem::create_directories(args.tokenizer_output.parent_path());
        tokenizer.save(args.tokenizer_output);

        const std::size_t worker_count = std::max<std::size_t>(
            1, std::thread::hardware_concurrency());
        constexpr std::size_t blocks_per_worker = 100;
        const std::size_t block_count = std::min(
            text.size(), worker_count * blocks_per_worker);
        std::vector<std::string_view> blocks;
        blocks.reserve(block_count);
        for (std::size_t block = 0; block < block_count; ++block) {
            const std::size_t begin = text.size() * block / block_count;
            const std::size_t end = text.size() * (block + 1) / block_count;
            blocks.emplace_back(text.data() + begin, end - begin);
        }

        ProgressBar tokenization_bar(text.size(), "Tokenizing", static_cast<double>(kMiB), "MiB/s");
        const std::vector<std::vector<bpe::TokenId>> encoded_chunks =
            tokenizer.encode_batch(blocks, worker_count);
        std::vector<bpe::TokenId> tokens;
        for (const auto& chunk : encoded_chunks)
            tokens.insert(tokens.end(), chunk.begin(), chunk.end());
        tokenization_bar.update(text.size()); tokenization_bar.finish();
        std::cout << "Vocabulary: " << tokenizer.vocab_size() << ", tokens: " << tokens.size() << '\n';

        if (args.batch_size > (std::numeric_limits<std::size_t>::max() - 1) / args.sequence_length)
            throw std::overflow_error("Batch token count overflows size_t");
        const std::size_t minimum_split_tokens = args.batch_size * args.sequence_length + 1;
        if (minimum_split_tokens > tokens.size() / 2)
            throw std::runtime_error("Not enough tokens for one training and validation batch");
        const std::size_t requested_validation_tokens = static_cast<std::size_t>(
            static_cast<double>(tokens.size()) * args.validation_fraction);
        const std::size_t validation_tokens = std::clamp(
            requested_validation_tokens, minimum_split_tokens, tokens.size() - minimum_split_tokens);
        const auto split = tokens.begin() + static_cast<std::ptrdiff_t>(tokens.size() - validation_tokens);
        std::vector<bpe::TokenId> training_tokens(tokens.begin(), split);
        std::vector<bpe::TokenId> validation_tokens_data(split, tokens.end());

        data::DataLoaderConfig loader_config{args.batch_size, args.sequence_length, true, true, 42};
        data::DatasetLoader loader(
            std::make_shared<const data::TokenDataset>(std::move(training_tokens)), loader_config);
        data::DataLoaderConfig validation_loader_config{args.batch_size, args.sequence_length, false, true, 42};
        data::DatasetLoader validation_loader(
            std::make_shared<const data::TokenDataset>(std::move(validation_tokens_data)), validation_loader_config);
        if (loader.batch_count() == 0 || validation_loader.batch_count() == 0)
            throw std::runtime_error("Not enough tokens for one training and validation batch");
        const std::size_t total_steps = args.max_steps ? std::min(args.max_steps, loader.batch_count() * args.epochs) : loader.batch_count() * args.epochs;
        LearningRateScheduler scheduler({args.learning_rate, args.min_learning_rate, args.warmup_steps, total_steps});
        ModelStorage model(tokenizer.vocab_size());
        if (args.load_directory) {
            load_huggingface_model(*args.load_directory,
                {model.embedding, model.mutable_layers, model.final_norm, model.lm_head});
            std::cout << "Loaded model weights from " << *args.load_directory << '\n';
        }
        auto [cos_cache, sin_cache] = rotary_cache(args.sequence_length, model.options.block_options.rotary_dim);
        TransformerModelWorkspace forward_workspace(model.options, args.batch_size, args.sequence_length, Dtype::F16);
        TransformerModelBackwardWorkspace backward_workspace(model.options, args.batch_size, args.sequence_length, Dtype::F16);
        // CCE currently accepts only F32 activations and weights.  This training
        // configuration uses F16 plus cuBLASLt for lm_head, which is faster and
        // avoids materializing/converting a second F32 model copy.
        Tensor logits({args.batch_size * args.sequence_length, tokenizer.vocab_size()}, Dtype::F16);
        Tensor loss({1}, Dtype::F32), grad_logits({args.batch_size * args.sequence_length, tokenizer.vocab_size()}, Dtype::F16);
        Tensor validation_logits({args.batch_size * args.sequence_length, tokenizer.vocab_size()}, Dtype::F16);
        Tensor validation_loss({1}, Dtype::F32);
        data::DeviceBatch device_batch;
        data::DeviceBatch validation_device_batch;
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
        double loss_sum = 0.0;
        std::size_t loss_count = 0;
        std::vector<float> host_loss(1);
        std::vector<float> host_validation_loss(1);
        std::optional<double> last_validation_loss;
        std::optional<double> last_average_loss;
        const auto update_training_metrics = [&] {
            std::string suffix = " | avg_loss=";
            suffix += last_average_loss ? std::to_string(*last_average_loss) : "n/a";
            suffix += " | validation_loss=";
            suffix += last_validation_loss ? std::to_string(*last_validation_loss) : "n/a";
            progress.set_suffix(std::move(suffix));
        };
        const auto save_validation_checkpoint = [&] {
            const std::filesystem::path checkpoint_directory =
                args.output_directory / "checkpoints" / ("step-" + std::to_string(step));
            export_huggingface_model(checkpoint_directory, tokenizer, model.weights(), model.options,
                                     args.sequence_length);
            std::cout << "\nSaved validation checkpoint at step " << step << " to "
                      << checkpoint_directory << '\n';
        };
        const auto run_validation = [&] {
            validation_loader.reset();
            double validation_loss_sum = 0.0;
            std::size_t validation_loss_count = 0;
            data::Batch validation_batch;
            while (validation_loss_count < args.validation_batches && validation_loader.next(validation_batch)) {
                data::upload_batch(validation_batch, validation_device_batch, cuda_stream);
                transformer_model_forward(validation_logits,
                    static_cast<const bpe::TokenId*>(validation_device_batch.input_ids.data()), args.batch_size,
                    args.sequence_length, model.weights(), forward_workspace, cos_cache, sin_cache, cublas,
                    cuda_stream, model.options);
                cross_entropy_forward(validation_loss, validation_logits,
                    static_cast<const bpe::TokenId*>(validation_device_batch.target_ids.data()),
                    validation_batch.token_count(), cuda_stream);
                CUDA_CHECK(cudaStreamSynchronize(cuda_stream));
                validation_loss.copy_to_host(host_validation_loss);
                validation_loss_sum += host_validation_loss[0];
                ++validation_loss_count;
            }
            last_validation_loss = validation_loss_sum / static_cast<double>(validation_loss_count);
            update_training_metrics();
            save_validation_checkpoint();
        };
        update_training_metrics();

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
                optimizer_options.learning_rate = scheduler.learning_rate(step);
                adamw_step_many_async(optimizer_entries, optimizer_options, optimizer_workspace, cuda_stream);
                ++step; progress.update(step);
                loss.copy_to_host(host_loss);
                loss_sum += host_loss[0];
                ++loss_count;
                if (step % 50 == 0 || step == total_steps) {
                    if (!adamw_check(optimizer_workspace, cuda_stream))
                        throw std::runtime_error("Non-finite gradient at step " + std::to_string(step));
                    last_average_loss = loss_sum / static_cast<double>(loss_count);
                    loss_sum = 0.0;
                    loss_count = 0;
                    update_training_metrics();
                }
                if (step % args.validation_interval == 0 || step == total_steps)
                    run_validation();
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
        export_huggingface_model(args.output_directory, tokenizer, model.weights(), model.options,
                                 args.sequence_length);
        GenerationOptions generation_options;
        generation_options.max_new_tokens = args.generate_tokens;
        generation_options.max_context_tokens = args.sequence_length;
        generation_options.temperature = 0.8F;
        generation_options.top_k = 40;
        generation_options.seed = 42;
        std::cout << "\nGenerated text:\n" << generate_text(tokenizer, args.prompt, model.weights(), cos_cache,
            sin_cache, cublas, model.options, generation_options) << '\n';
        std::cout << "Completed " << step << " optimizer steps. Hugging Face model saved to "
                  << args.output_directory << '\n';
        return EXIT_SUCCESS;
    } catch (const std::exception& error) { std::cerr << "Training failed: " << error.what() << '\n'; return EXIT_FAILURE; }
}
