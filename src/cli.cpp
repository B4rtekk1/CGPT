/** @file Command-line interface for model loading and text generation. */

#include "core/generation.h"
#include "core/huggingface_export.h"
#include "core/cuda_check.h"

#include <cuda_runtime.h>

#include <cmath>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <iterator>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {
    namespace ansi {
        constexpr std::string_view reset = "\033[0m";
        constexpr std::string_view cyan = "\033[36m";
        constexpr std::string_view blue = "\033[94m";
        constexpr std::string_view magenta = "\033[95m";
        constexpr std::string_view green = "\033[32m";
        constexpr std::string_view yellow = "\033[33m";
        constexpr std::string_view dim = "\033[2m";
    }

    void print_banner(std::ostream &out) {
        constexpr std::string_view c[] = {
            "  ###### ", " ##      ", " ##      ", " ##      ", "  ###### "
        };
        constexpr std::string_view g[] = {
            "  ###### ", " ##      ", " ##  ### ", " ##   ## ", "  ###### "
        };
        constexpr std::string_view p[] = {
            " ####### ", " ##   ## ", " ####### ", " ##      ", " ##      "
        };
        constexpr std::string_view t[] = {
            " ####### ", "    ##   ", "    ##   ", "    ##   ", "    ##   "
        };

        for (std::size_t row = 0; row < std::size(c); ++row) {
            out << ansi::cyan << c[row]
                << ansi::blue << "  " << g[row]
                << ansi::magenta << "  " << p[row]
                << ansi::yellow << "  " << t[row]
                << ansi::reset << '\n';
        }
        constexpr std::string_view subtitle = "Fast, local text generation from the command line";
        out << ansi::dim << subtitle << ansi::reset << "\n\n";
    }

    struct Arguments {
        std::filesystem::path model;
        std::filesystem::path tokenizer;
        std::string prompt;
        std::size_t max_new_tokens = 128;
        std::size_t max_context_tokens = 0;
        std::size_t top_k = 0;
        float top_p = 1.0F;
        float min_p = 0.0F;
        float temperature = 0.8F;
        float repetition_penalty = 1.0F;
        float presence_penalty = 0.0F;
        float frequency_penalty = 0.0F;
        std::size_t no_repeat_ngram_size = 0;
        std::uint64_t seed = 0;
        // CPU is the safest default for machines without a compatible CUDA
        // runtime or GPU. CUDA remains available explicitly via --device cuda.
        std::string device = "cpu";
        bool cpu = false;
    };

    /**
     * Selects the execution device without turning an unavailable CUDA
     * runtime into a fatal error for the automatic mode.
     *
     * cudaGetDeviceCount can return errors such as cudaErrorNoDevice,
     * cudaErrorInsufficientDriver, or cudaErrorInitializationError when the
     * CUDA toolkit/runtime is present but no usable GPU is available. Those
     * are all valid reasons to use the CPU when the user selected `auto`.
     */
    bool select_cpu_device(const std::string &requested_device) {
        if (requested_device == "cpu") return true;
        if (requested_device == "cuda") return false;

        int device_count = 0;
        const cudaError_t status = cudaGetDeviceCount(&device_count);
        if (status == cudaSuccess && device_count > 0) return false;

        // cudaGetDeviceCount leaves the runtime error pending. Clear it before
        // constructing CPU tensors, otherwise a later CUDA API call can report
        // this stale detection failure instead of the actual operation result.
        const cudaError_t cleared_status = cudaGetLastError();
        (void)cleared_status;
        return true;
    }

    [[noreturn]] void usage(const std::string &error = {}) {
        print_banner(std::cerr);
        if (!error.empty()) std::cerr << "Error: " << error << "\n\n";
        std::cerr << ansi::blue << "Usage" << ansi::reset
                << ": cgpt_cli [--model <model.safetensors|directory>] [options]\n"
                << "  --model <...>                  model path (default: outputdg/step-58874 next to executable)\n"
                << "  --tokenizer <tokenizer.json>   tokenizer path (default: next to the model)\n"
                << "  --prompt <text>                generate once; without it, start the REPL\n"
                << "  --max-new-tokens N             number of new tokens (default: 128)\n"
                << "  --context N                    maximum context length\n"
                << "  --temperature X                sampling temperature (default: 0.8)\n"
                << "  --top-k N                      top-k sampling (0 = disabled)\n"
                << "  --top-p X                      nucleus sampling (default: 1.0)\n"
                << "  --min-p X                      minimum probability sampling (default: 0)\n"
                << "  --repetition-penalty X        repetition penalty (default: 1.0)\n"
                << "  --presence-penalty X          one-time token penalty (default: 0)\n"
                << "  --frequency-penalty X         per-occurrence token penalty (default: 0)\n"
                << "  --no-repeat-ngram N            block repeated n-grams (default: 0)\n"
                << "  --seed N                       random seed\n"
                << "  --device auto|cpu|cuda         execution device (default: cpu)\n"
                << "\nREPL commands:\n"
                << "  /set <parameter> <value>       change generation parameter\n"
                << "  /params                        show current generation parameters\n"
                << "  /help                          show this help\n"
                << "  /exit                           leave the REPL\n";
        if (error.empty()) std::exit(0);
        throw std::invalid_argument("invalid arguments");
    }

    template<typename T>
    T value_after(const std::string &option, int &index, int argc, char **argv) {
        if (index + 1 >= argc) usage("missing value for " + option);
        try { return static_cast<T>(std::stoull(argv[++index])); } catch (...) { usage("invalid value for " + option); }
    }

    float float_after(const std::string &option, int &index, int argc, char **argv) {
        if (index + 1 >= argc) usage("missing value for " + option);
        try { return std::stof(argv[++index]); } catch (...) { usage("invalid value for " + option); }
    }

    void print_parameters(const Arguments &args) {
        std::cout << "Current parameters:\n"
                  << "  max-new-tokens: " << args.max_new_tokens << '\n'
                  << "  context: " << args.max_context_tokens << '\n'
                  << "  temperature: " << args.temperature << '\n'
                  << "  top-k: " << args.top_k << '\n'
                  << "  top-p: " << args.top_p << '\n'
                  << "  min-p: " << args.min_p << '\n'
                  << "  repetition-penalty: " << args.repetition_penalty << '\n'
                  << "  presence-penalty: " << args.presence_penalty << '\n'
                  << "  frequency-penalty: " << args.frequency_penalty << '\n'
                  << "  no-repeat-ngram: " << args.no_repeat_ngram_size << '\n'
                  << "  seed: " << args.seed << '\n';
    }

    bool handle_repl_command(const std::string &line, Arguments &args, const std::size_t model_context) {
        if (line == "/help") {
            std::cout << "REPL commands:\n"
                      << "  /set <parameter> <value>  change a generation parameter\n"
                      << "  /params                   show current parameters\n"
                      << "  /help                     show this help\n"
                      << "  /exit                     leave the REPL\n";
            return true;
        }
        if (line == "/params") {
            print_parameters(args);
            return true;
        }
        if (line == "/exit" || line == "/quit") return false;
        if (line.rfind("/set ", 0) != 0) return true;

        std::istringstream input(line.substr(5));
        std::string parameter, value;
        if (!(input >> parameter >> value)) {
            std::cerr << "Usage: /set <parameter> <value>\n";
            return true;
        }
        try {
            if (parameter == "max-new-tokens") args.max_new_tokens = std::stoull(value);
            else if (parameter == "context") {
                const auto context = std::stoull(value);
                if (context == 0 || context > model_context) throw std::out_of_range("context");
                args.max_context_tokens = context;
            } else if (parameter == "top-k") args.top_k = std::stoull(value);
            else if (parameter == "seed") args.seed = std::stoull(value);
            else if (parameter == "no-repeat-ngram") args.no_repeat_ngram_size = std::stoull(value);
            else if (parameter == "temperature") args.temperature = std::stof(value);
            else if (parameter == "top-p") args.top_p = std::stof(value);
            else if (parameter == "min-p") args.min_p = std::stof(value);
            else if (parameter == "repetition-penalty") args.repetition_penalty = std::stof(value);
            else if (parameter == "presence-penalty") args.presence_penalty = std::stof(value);
            else if (parameter == "frequency-penalty") args.frequency_penalty = std::stof(value);
            else {
                std::cerr << "Unknown parameter: " << parameter << '\n';
                return true;
            }
            std::cout << "Updated " << parameter << " = " << value << '\n';
        } catch (...) {
            std::cerr << "Invalid value for " << parameter << ": " << value << '\n';
        }
        return true;
    }

    Arguments parse_arguments(int argc, char **argv) {
        Arguments result;
        for (int i = 1; i < argc; ++i) {
            const std::string option = argv[i];
            if (option == "--help" || option == "-h") usage();
            else if (option == "--model") {
                if (++i >= argc) usage("missing model path");
                result.model = argv[i];
            } else if (option == "--tokenizer") {
                if (++i >= argc) usage("missing tokenizer path");
                result.tokenizer = argv[i];
            } else if (option == "--prompt") {
                if (++i >= argc) usage("missing prompt text");
                result.prompt = argv[i];
            } else if (option == "--max-new-tokens") result.max_new_tokens = value_after<std::size_t>(
                                                         option, i, argc, argv);
            else if (option == "--context") result.max_context_tokens = value_after<std::size_t>(option, i, argc, argv);
            else if (option == "--top-k") result.top_k = value_after<std::size_t>(option, i, argc, argv);
            else if (option == "--seed") result.seed = value_after<std::uint64_t>(option, i, argc, argv);
            else if (option == "--device") {
                if (++i >= argc) usage("missing device");
                result.device = argv[i];
                if (result.device != "auto" && result.device != "cpu" && result.device != "cuda")
                    usage("device must be auto, cpu or cuda");
            } else if (option == "--temperature") result.temperature = float_after(option, i, argc, argv);
            else if (option == "--top-p") result.top_p = float_after(option, i, argc, argv);
            else if (option == "--min-p") result.min_p = float_after(option, i, argc, argv);
            else if (option == "--repetition-penalty") result.repetition_penalty = float_after(option, i, argc, argv);
            else if (option == "--presence-penalty") result.presence_penalty = float_after(option, i, argc, argv);
            else if (option == "--frequency-penalty") result.frequency_penalty = float_after(option, i, argc, argv);
            else if (option == "--no-repeat-ngram") result.no_repeat_ngram_size = value_after<std::size_t>(
                                                        option, i, argc, argv);
            else usage("unknown option: " + option);
        }
        return result;
    }

    std::string read_file(const std::filesystem::path &path) {
        std::ifstream input(path);
        if (!input) throw std::runtime_error("cannot open " + path.string());
        return {std::istreambuf_iterator<char>(input), {}};
    }

    std::size_t json_size(const std::string &json, const std::string &key) {
        const auto marker = "\"" + key + "\"";
        const auto start = json.find(marker);
        if (start == std::string::npos) throw std::runtime_error("missing field in config.json: " + key);
        const auto colon = json.find(':', start + marker.size());
        if (colon == std::string::npos) throw std::runtime_error("invalid config.json");
        return std::stoull(json.substr(colon + 1));
    }

    float json_float(const std::string &json, const std::string &key) {
        const auto marker = "\"" + key + "\"";
        const auto start = json.find(marker);
        if (start == std::string::npos) throw std::runtime_error("missing field in config.json: " + key);
        const auto colon = json.find(':', start + marker.size());
        return std::stof(json.substr(colon + 1));
    }

    struct LayerStorage {
        Tensor attention_norm, q, k, q_norm, k_norm, v, o, ffn_norm, gate, up, down;

        explicit LayerStorage(const TransformerBlockOptions &b, DeviceType device)
            : attention_norm({b.hidden_size}, Dtype::F16, device),
              q({b.num_query_heads * b.head_dim, b.hidden_size}, Dtype::F16, device),
              k({b.num_kv_heads * b.head_dim, b.hidden_size}, Dtype::F16, device),
              q_norm({b.head_dim}, Dtype::F16, device),
              k_norm({b.head_dim}, Dtype::F16, device),
              v({b.num_kv_heads * b.head_dim, b.hidden_size}, Dtype::F16, device),
              o({b.hidden_size, b.num_query_heads * b.head_dim}, Dtype::F16, device),
              ffn_norm({b.hidden_size}, Dtype::F16, device),
              gate({b.intermediate_size, b.hidden_size}, Dtype::F16, device),
              up({b.intermediate_size, b.hidden_size}, Dtype::F16, device),
              down({b.hidden_size, b.intermediate_size}, Dtype::F16, device) {
        }
    };

    struct ModelStorage {
        TransformerModelOptions options;
        Tensor embedding, final_norm, lm_head;
        std::vector<LayerStorage> storage;
        std::vector<TransformerBlockWeights> layers;
        std::vector<MutableTransformerBlockWeights> mutable_layers;

        explicit ModelStorage(TransformerModelOptions o, DeviceType device) : options(o),
                                                                              embedding({
                                                                                  o.vocabulary_size,
                                                                                  o.block_options.hidden_size
                                                                              }, Dtype::F16, device),
                                                                              final_norm({o.block_options.hidden_size},
                                                                                  Dtype::F16, device),
                                                                              lm_head({
                                                                                  o.vocabulary_size,
                                                                                  o.block_options.hidden_size
                                                                              }, Dtype::F16, device) {
            for (std::size_t i = 0; i < o.num_layers; ++i) storage.emplace_back(o.block_options, device);
            for (auto &l: storage) {
                layers.push_back({
                    l.attention_norm, l.q, l.k, l.q_norm, l.k_norm, l.v, l.o, l.ffn_norm, l.gate, l.up, l.down
                });
                mutable_layers.push_back({
                    l.attention_norm, l.q, l.k, l.q_norm, l.k_norm, l.v, l.o, l.ffn_norm, l.gate, l.up, l.down
                });
            }
        }

        MutableTransformerModelWeights mutable_weights() { return {embedding, mutable_layers, final_norm, lm_head}; }
        [[nodiscard]] TransformerModelWeights weights() const { return {embedding, layers, final_norm, lm_head}; }
    };

    std::pair<Tensor, Tensor> rotary_cache(std::size_t length, std::size_t dim, DeviceType device) {
        std::vector<float> cosines(length * dim / 2), sines(cosines.size());
        for (std::size_t p = 0; p < length; ++p)
            for (std::size_t i = 0; i < dim / 2; ++i) {
                const float theta = static_cast<float>(p) / std::pow(10000.0F, 2.0F * static_cast<float>(i) / static_cast<float>(dim));
                cosines[p * dim / 2 + i] = std::cos(theta);
                sines[p * dim / 2 + i] = std::sin(theta);
            }
        Tensor c({length, dim / 2}, Dtype::F16, device), s({length, dim / 2}, Dtype::F16, device);
        c.copy_from_host(cosines);
        s.copy_from_host(sines);
        return {std::move(c), std::move(s)};
    }

    TextGenerationResult generate(const std::string &prompt, const Arguments &args,
                                  const bpe::BpeTokenizer &tokenizer, ModelStorage &model,
                                  const Tensor &cosine, const Tensor &sine, const CublasLtContext *cublas) {
        GenerationOptions options;
        options.max_new_tokens = args.max_new_tokens;
        options.max_context_tokens = args.max_context_tokens;
        options.temperature = args.temperature;
        options.top_k = args.top_k;
        options.top_p = args.top_p;
        options.seed = args.seed;
        options.min_p = args.min_p;
        options.repetition_penalty = args.repetition_penalty;
        options.presence_penalty = args.presence_penalty;
        options.frequency_penalty = args.frequency_penalty;
        options.no_repeat_ngram_size = args.no_repeat_ngram_size;
        if (args.cpu) return generate_text_with_stats_cpu(tokenizer, prompt, model.weights(), cosine, sine,
                                                          model.options, options);
        return generate_text_with_stats(tokenizer, prompt, model.weights(), cosine, sine,
                                        *cublas, model.options, options);
    }

    void print_generation(const std::string &prompt, const Arguments &args,
                          const bpe::BpeTokenizer &tokenizer, ModelStorage &model,
                          const Tensor &cosine, const Tensor &sine, const CublasLtContext *cublas) {
        const auto start = std::chrono::steady_clock::now();
        const auto result = generate(prompt, args, tokenizer, model, cosine, sine, cublas);
        const auto elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        std::cout << ansi::green << result.text << ansi::reset << '\n';
        const double tokens_per_second = elapsed > 0.0
                                             ? static_cast<double>(result.generated_tokens) / elapsed
                                             : 0.0;
        std::cerr << ansi::dim << "[generated " << result.generated_tokens << " tokens in "
                << std::fixed << std::setprecision(2) << elapsed << " s | "
                << tokens_per_second << " tok/s]" << ansi::reset << "\n";
    }
} // namespace

int main(int argc, char **argv) {
    try {
        const Arguments args = parse_arguments(argc, argv);
        print_banner(std::cout);
        const auto executable_directory = std::filesystem::absolute(std::filesystem::path(argv[0])).parent_path();
        const auto model_path = args.model.empty()
                                    ? executable_directory / "outputdg" / "step-58874"
                                    : args.model;
        const std::filesystem::path directory = std::filesystem::is_directory(model_path)
                                                    ? model_path
                                                    : model_path.parent_path();
        const auto config_path = directory / "config.json";
        const auto tokenizer_path = args.tokenizer.empty() ? directory / "tokenizer.json" : args.tokenizer;
        const std::string config = read_file(config_path);
        const auto block = TransformerBlockOptions{
            json_size(config, "hidden_size"), json_size(config, "intermediate_size"),
            json_size(config, "num_attention_heads"),
            json_size(config, "num_key_value_heads"), json_size(config, "head_dim"), json_size(config, "head_dim"),
            json_float(config, "rms_norm_eps"), true, {ComputeType::F32}
        };
        const std::size_t context = json_size(config, "max_position_embeddings");
        Arguments runtime_args = args;
        runtime_args.cpu = select_cpu_device(runtime_args.device);
        if (runtime_args.device == "auto" && runtime_args.cpu)
            std::cerr << ansi::yellow
                      << "CUDA unavailable; falling back to CPU."
                      << ansi::reset << '\n';
        if (runtime_args.max_context_tokens == 0) runtime_args.max_context_tokens = context;
        if (runtime_args.max_context_tokens > context) throw std::runtime_error(
            "--context exceeds max_position_embeddings");
        const DeviceType device = runtime_args.cpu ? DeviceType::CPU : DeviceType::CUDA;
        std::cout << ansi::yellow << "Loading model..." << ansi::reset << '\n';
        ModelStorage model({json_size(config, "vocab_size"), json_size(config, "num_hidden_layers"), block}, device);
        load_huggingface_model(directory, model.mutable_weights());
        const auto tokenizer = bpe::BpeTokenizer::load(tokenizer_path);
        const auto [cosine, sine] = rotary_cache(runtime_args.max_context_tokens, block.rotary_dim, device);
        std::unique_ptr<CublasLtContext> cublas;
        if (!runtime_args.cpu) cublas = std::make_unique<CublasLtContext>();
        std::cout << ansi::green << "[OK] Model loaded" << ansi::reset
                << "  " << ansi::dim << "device: " << (runtime_args.cpu ? "CPU" : "CUDA")
                << ", context: " << runtime_args.max_context_tokens << " tokens" << ansi::reset << '\n'
                << ansi::dim << "Type 'exit' or 'quit' to stop.\n" << ansi::reset;
        if (!args.prompt.empty()) {
            print_generation(args.prompt, runtime_args, tokenizer, model, cosine, sine, cublas.get());
            return 0;
        }
        for (std::string prompt; std::cout << ansi::blue << "\nYou > " << ansi::reset
                                           && std::getline(std::cin, prompt);) {
            if (!prompt.empty() && prompt.front() == '/') {
                if (!handle_repl_command(prompt, runtime_args, context)) break;
                continue;
            }
            if (prompt == "exit" || prompt == "quit") break;
            try { print_generation(prompt, runtime_args, tokenizer, model, cosine, sine, cublas.get()); } catch (const
                std::exception &error) { std::cerr << "Generation error: " << error.what() << "\n"; }
        }
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }
}
