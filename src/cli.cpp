#include "core/generation.h"
#include "core/huggingface_export.h"
#include "core/cuda_check.h"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

struct Arguments {
    std::filesystem::path model;
    std::filesystem::path tokenizer;
    std::string prompt;
    std::size_t max_new_tokens = 64;
    std::size_t max_context_tokens = 0;
    std::size_t top_k = 0;
    float top_p = 1.0F;
    float temperature = 0.8F;
    std::uint64_t seed = 0;
};

[[noreturn]] void usage(const std::string& error = {}) {
    if (!error.empty()) std::cerr << "Error: " << error << "\n\n";
    std::cerr << "Usage: cgpt_cli --model <model.safetensors|directory> [options]\n"
              << "  --tokenizer <tokenizer.json>   tokenizer path (default: next to the model)\n"
              << "  --prompt <text>                generate once; without it, start the REPL\n"
              << "  --max-new-tokens N             number of new tokens (default: 64)\n"
              << "  --context N                    maximum context length\n"
              << "  --temperature X                sampling temperature (default: 0.8)\n"
              << "  --top-k N                      top-k sampling (0 = disabled)\n"
              << "  --top-p X                      nucleus sampling (default: 1.0)\n"
              << "  --seed N                       random seed\n";
    if (error.empty()) std::exit(0);
    throw std::invalid_argument("invalid arguments");
}

template <typename T>
T value_after(const std::string& option, int& index, int argc, char** argv) {
    if (index + 1 >= argc) usage("missing value for " + option);
    try { return static_cast<T>(std::stoull(argv[++index])); }
    catch (...) { usage("invalid value for " + option); }
}

float float_after(const std::string& option, int& index, int argc, char** argv) {
    if (index + 1 >= argc) usage("missing value for " + option);
    try { return std::stof(argv[++index]); }
    catch (...) { usage("invalid value for " + option); }
}

Arguments parse_arguments(int argc, char** argv) {
    Arguments result;
    for (int i = 1; i < argc; ++i) {
        const std::string option = argv[i];
        if (option == "--help" || option == "-h") usage();
        else if (option == "--model") { if (++i >= argc) usage("missing model path"); result.model = argv[i]; }
        else if (option == "--tokenizer") { if (++i >= argc) usage("missing tokenizer path"); result.tokenizer = argv[i]; }
        else if (option == "--prompt") { if (++i >= argc) usage("missing prompt text"); result.prompt = argv[i]; }
        else if (option == "--max-new-tokens") result.max_new_tokens = value_after<std::size_t>(option, i, argc, argv);
        else if (option == "--context") result.max_context_tokens = value_after<std::size_t>(option, i, argc, argv);
        else if (option == "--top-k") result.top_k = value_after<std::size_t>(option, i, argc, argv);
        else if (option == "--seed") result.seed = value_after<std::uint64_t>(option, i, argc, argv);
        else if (option == "--temperature") result.temperature = float_after(option, i, argc, argv);
        else if (option == "--top-p") result.top_p = float_after(option, i, argc, argv);
        else usage("unknown option: " + option);
    }
    if (result.model.empty()) usage("--model is required");
    return result;
}

std::string read_file(const std::filesystem::path& path) {
    std::ifstream input(path);
    if (!input) throw std::runtime_error("cannot open " + path.string());
    return {std::istreambuf_iterator<char>(input), {}};
}

std::size_t json_size(const std::string& json, const std::string& key) {
    const auto marker = "\"" + key + "\"";
    const auto start = json.find(marker);
    if (start == std::string::npos) throw std::runtime_error("missing field in config.json: " + key);
    const auto colon = json.find(':', start + marker.size());
    if (colon == std::string::npos) throw std::runtime_error("invalid config.json");
    return std::stoull(json.substr(colon + 1));
}

float json_float(const std::string& json, const std::string& key) {
    const auto marker = "\"" + key + "\"";
    const auto start = json.find(marker);
    if (start == std::string::npos) throw std::runtime_error("missing field in config.json: " + key);
    const auto colon = json.find(':', start + marker.size());
    return std::stof(json.substr(colon + 1));
}

struct LayerStorage {
    Tensor attention_norm, q, k, q_norm, k_norm, v, o, ffn_norm, gate, up, down;
    explicit LayerStorage(const TransformerBlockOptions& b)
        : attention_norm({b.hidden_size}, Dtype::F16), q({b.num_query_heads * b.head_dim, b.hidden_size}, Dtype::F16),
          k({b.num_kv_heads * b.head_dim, b.hidden_size}, Dtype::F16), q_norm({b.head_dim}, Dtype::F16),
          k_norm({b.head_dim}, Dtype::F16), v({b.num_kv_heads * b.head_dim, b.hidden_size}, Dtype::F16),
          o({b.hidden_size, b.num_query_heads * b.head_dim}, Dtype::F16), ffn_norm({b.hidden_size}, Dtype::F16),
          gate({b.intermediate_size, b.hidden_size}, Dtype::F16), up({b.intermediate_size, b.hidden_size}, Dtype::F16),
          down({b.hidden_size, b.intermediate_size}, Dtype::F16) {}
};

struct ModelStorage {
    TransformerModelOptions options;
    Tensor embedding, final_norm, lm_head;
    std::vector<LayerStorage> storage;
    std::vector<TransformerBlockWeights> layers;
    std::vector<MutableTransformerBlockWeights> mutable_layers;

    explicit ModelStorage(TransformerModelOptions o) : options(o),
        embedding({o.vocabulary_size, o.block_options.hidden_size}, Dtype::F16),
        final_norm({o.block_options.hidden_size}, Dtype::F16),
        lm_head({o.vocabulary_size, o.block_options.hidden_size}, Dtype::F16) {
        for (std::size_t i = 0; i < o.num_layers; ++i) storage.emplace_back(o.block_options);
        for (auto& l : storage) {
            layers.push_back({l.attention_norm,l.q,l.k,l.q_norm,l.k_norm,l.v,l.o,l.ffn_norm,l.gate,l.up,l.down});
            mutable_layers.push_back({l.attention_norm,l.q,l.k,l.q_norm,l.k_norm,l.v,l.o,l.ffn_norm,l.gate,l.up,l.down});
        }
    }
    MutableTransformerModelWeights mutable_weights() { return {embedding, mutable_layers, final_norm, lm_head}; }
    TransformerModelWeights weights() const { return {embedding, layers, final_norm, lm_head}; }
};

std::pair<Tensor, Tensor> rotary_cache(std::size_t length, std::size_t dim) {
    std::vector<float> cosines(length * dim / 2), sines(cosines.size());
    for (std::size_t p = 0; p < length; ++p) for (std::size_t i = 0; i < dim / 2; ++i) {
        const float theta = static_cast<float>(p) / std::pow(10000.0F, 2.0F * static_cast<float>(i) / dim);
        cosines[p * dim / 2 + i] = std::cos(theta); sines[p * dim / 2 + i] = std::sin(theta);
    }
    Tensor c({length, dim / 2}, Dtype::F16), s({length, dim / 2}, Dtype::F16);
    c.copy_from_host(cosines); s.copy_from_host(sines); return {std::move(c), std::move(s)};
}

std::string generate(const std::string& prompt, const Arguments& args, const bpe::BpeTokenizer& tokenizer,
                     ModelStorage& model, const Tensor& cosine, const Tensor& sine, const CublasLtContext& cublas) {
    GenerationOptions options;
    options.max_new_tokens = args.max_new_tokens;
    options.max_context_tokens = args.max_context_tokens;
    options.temperature = args.temperature; options.top_k = args.top_k; options.top_p = args.top_p; options.seed = args.seed;
    return generate_text(tokenizer, prompt, model.weights(), cosine, sine, cublas, model.options, options);
}

} // namespace

int main(int argc, char** argv) {
    try {
        const Arguments args = parse_arguments(argc, argv);
        const std::filesystem::path directory = std::filesystem::is_directory(args.model) ? args.model : args.model.parent_path();
        const auto config_path = directory / "config.json";
        const auto tokenizer_path = args.tokenizer.empty() ? directory / "tokenizer.json" : args.tokenizer;
        const std::string config = read_file(config_path);
        const auto block = TransformerBlockOptions{
            json_size(config,"hidden_size"), json_size(config,"intermediate_size"), json_size(config,"num_attention_heads"),
            json_size(config,"num_key_value_heads"), json_size(config,"head_dim"), json_size(config,"head_dim"),
            json_float(config,"rms_norm_eps"), true, {ComputeType::F32}};
        const std::size_t context = json_size(config, "max_position_embeddings");
        ModelStorage model({json_size(config,"vocab_size"), json_size(config,"num_hidden_layers"), block});
        load_huggingface_model(directory, model.mutable_weights());
        const auto tokenizer = bpe::BpeTokenizer::load(tokenizer_path);
        Arguments runtime_args = args;
        if (runtime_args.max_context_tokens == 0) runtime_args.max_context_tokens = context;
        if (runtime_args.max_context_tokens > context) throw std::runtime_error("--context exceeds max_position_embeddings");
        const auto [cosine, sine] = rotary_cache(runtime_args.max_context_tokens, block.rotary_dim);
        const CublasLtContext cublas;
        std::cout << "Model loaded. Enter a prompt (type 'exit' or 'quit' to stop).\n";
        if (!args.prompt.empty()) { std::cout << generate(args.prompt, runtime_args, tokenizer, model, cosine, sine, cublas) << '\n'; return 0; }
        for (std::string prompt; std::cout << "> " && std::getline(std::cin, prompt); ) {
            if (prompt == "exit" || prompt == "quit") break;
            try { std::cout << generate(prompt, runtime_args, tokenizer, model, cosine, sine, cublas) << "\n"; }
            catch (const std::exception& error) { std::cerr << "Generation error: " << error.what() << "\n"; }
        }
        return 0;
    } catch (const std::exception& error) { std::cerr << "Error: " << error.what() << '\n'; return 1; }
}
