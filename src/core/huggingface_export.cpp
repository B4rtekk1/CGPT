#include "core/huggingface_export.h"

#include "core/cuda_check.h"

#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <string>
#include <string_view>
#include <stdexcept>
#include <vector>

namespace {
struct ExportTensor {
    std::string name;
    const Tensor* tensor;
    std::vector<std::uint8_t> bytes;
    std::size_t offset = 0;
};

std::string json_string(const std::string_view value) {
    std::ostringstream result;
    result << '"';
    for (const unsigned char character : value) {
        switch (character) {
            case '"': result << "\\\""; break;
            case '\\': result << "\\\\"; break;
            case '\n': result << "\\n"; break;
            case '\r': result << "\\r"; break;
            case '\t': result << "\\t"; break;
            default: result << character; break;
        }
    }
    result << '"';
    return result.str();
}

std::string safetensors_dtype(const Dtype dtype) {
    switch (dtype) {
        case Dtype::F16: return "F16";
        case Dtype::BF16: return "BF16";
        case Dtype::F32: return "F32";
        default: throw std::invalid_argument("Unsupported dtype for safetensors export");
    }
}

void copy_tensor_bytes(ExportTensor& output) {
    output.bytes.resize(output.tensor->nbytes());
    if (output.tensor->device_type() == DeviceType::CUDA) {
        CUDA_CHECK(cudaMemcpy(output.bytes.data(), output.tensor->raw_data(), output.bytes.size(),
                              cudaMemcpyDeviceToHost));
    } else {
        std::memcpy(output.bytes.data(), output.tensor->raw_data(), output.bytes.size());
    }
}

void add_tensor(std::vector<ExportTensor>& tensors, std::string name, const Tensor& tensor) {
    tensors.push_back({std::move(name), &tensor});
}

std::string shape_json(const Tensor& tensor) {
    std::ostringstream result;
    result << '[';
    for (std::size_t index = 0; index < tensor.shape().size(); ++index) {
        if (index != 0) result << ',';
        result << tensor.shape()[index];
    }
    result << ']';
    return result.str();
}

void write_config(const std::filesystem::path& path, const TransformerModelOptions& options,
                  const std::size_t max_position_embeddings) {
    const auto& block = options.block_options;
    std::ofstream output(path);
    if (!output) throw std::runtime_error("Cannot create " + path.string());
    output << "{\n"
           << "  \"architectures\": [\"CGPTForCausalLM\"],\n"
           << "  \"model_type\": \"cgpt\",\n"
           << "  \"torch_dtype\": \"float16\",\n"
           << "  \"vocab_size\": " << options.vocabulary_size << ",\n"
           << "  \"hidden_size\": " << block.hidden_size << ",\n"
           << "  \"intermediate_size\": " << block.intermediate_size << ",\n"
           << "  \"num_hidden_layers\": " << options.num_layers << ",\n"
           << "  \"num_attention_heads\": " << block.num_query_heads << ",\n"
           << "  \"num_key_value_heads\": " << block.num_kv_heads << ",\n"
           << "  \"head_dim\": " << block.head_dim << ",\n"
           << "  \"max_position_embeddings\": " << max_position_embeddings << ",\n"
           << "  \"rms_norm_eps\": " << std::setprecision(9) << block.rms_epsilon << ",\n"
           << "  \"rope_theta\": 10000.0,\n"
           << "  \"use_parallel_residual\": true,\n"
           << "  \"tie_word_embeddings\": true\n"
           << "}\n";
}

void write_safetensors(const std::filesystem::path& path, std::vector<ExportTensor>& tensors) {
    std::size_t data_size = 0;
    for (ExportTensor& tensor : tensors) {
        copy_tensor_bytes(tensor);
        tensor.offset = data_size;
        if (tensor.bytes.size() > std::numeric_limits<std::size_t>::max() - data_size)
            throw std::overflow_error("Safetensors data size overflow");
        data_size += tensor.bytes.size();
    }

    std::ostringstream header;
    header << '{';
    for (std::size_t index = 0; index < tensors.size(); ++index) {
        if (index != 0) header << ',';
        const ExportTensor& tensor = tensors[index];
        header << json_string(tensor.name) << ":{\"dtype\":"
               << json_string(safetensors_dtype(tensor.tensor->dtype()))
               << ",\"shape\":" << shape_json(*tensor.tensor)
               << ",\"data_offsets\":[" << tensor.offset << ','
               << tensor.offset + tensor.bytes.size() << "]}";
    }
    header << '}';
    std::string header_bytes = header.str();
    while (header_bytes.size() % 8 != 0) header_bytes.push_back(' ');

    std::ofstream output(path, std::ios::binary);
    if (!output) throw std::runtime_error("Cannot create " + path.string());
    const std::uint64_t header_size = header_bytes.size();
    output.write(reinterpret_cast<const char*>(&header_size), sizeof(header_size));
    output.write(header_bytes.data(), static_cast<std::streamsize>(header_bytes.size()));
    for (const ExportTensor& tensor : tensors)
        output.write(reinterpret_cast<const char*>(tensor.bytes.data()),
                     static_cast<std::streamsize>(tensor.bytes.size()));
    if (!output) throw std::runtime_error("Failed writing " + path.string());
}

std::size_t parse_number(const std::string& text, std::size_t& position) {
    while (position < text.size() && (text[position] == ' ' || text[position] == '\n' || text[position] == '\r' || text[position] == '\t' || text[position] == '[' || text[position] == ',')) ++position;
    const std::size_t begin = position;
    while (position < text.size() && text[position] >= '0' && text[position] <= '9') ++position;
    if (begin == position) throw std::runtime_error("Invalid safetensors number");
    return std::stoull(text.substr(begin, position - begin));
}

std::string field_string(const std::string& object, const std::string& field) {
    const std::string marker = "\"" + field + "\":\"";
    const std::size_t begin = object.find(marker);
    if (begin == std::string::npos) throw std::runtime_error("Missing safetensors field: " + field);
    const std::size_t value_begin = begin + marker.size();
    const std::size_t value_end = object.find('"', value_begin);
    if (value_end == std::string::npos) throw std::runtime_error("Invalid safetensors string field");
    return object.substr(value_begin, value_end - value_begin);
}

std::string tensor_object(const std::string& header, const std::string& name) {
    const std::string marker = json_string(name) + ":{";
    const std::size_t begin = header.find(marker);
    if (begin == std::string::npos) throw std::runtime_error("Tensor missing from safetensors: " + name);
    const std::size_t end = header.find("},\"", begin);
    return header.substr(begin, end == std::string::npos ? std::string::npos : end - begin + 1);
}

void load_one_tensor(const std::string& header, std::ifstream& input, const std::size_t data_begin,
                     const std::string& name, Tensor& tensor) {
    const std::string object = tensor_object(header, name);
    if (field_string(object, "dtype") != safetensors_dtype(tensor.dtype()))
        throw std::runtime_error("Dtype mismatch for tensor: " + name);
    const std::size_t shape_begin = object.find("\"shape\":[");
    if (shape_begin == std::string::npos) throw std::runtime_error("Missing shape for tensor: " + name);
    std::size_t cursor = shape_begin + 9;
    for (const std::size_t dimension : tensor.shape()) {
        if (parse_number(object, cursor) != dimension)
            throw std::runtime_error("Shape mismatch for tensor: " + name);
    }
    const std::size_t offsets_begin = object.find("\"data_offsets\":[");
    if (offsets_begin == std::string::npos) throw std::runtime_error("Missing offsets for tensor: " + name);
    // `"data_offsets":[` is 16 bytes long. Starting at the first value is
    // important for zero offsets: advancing one byte too far lands on `]`
    // and produces the misleading "Invalid safetensors number" error.
    cursor = offsets_begin + 16;
    const std::size_t start = parse_number(object, cursor);
    const std::size_t end = parse_number(object, cursor);
    if (end < start || end - start != tensor.nbytes())
        throw std::runtime_error("Byte-size mismatch for tensor: " + name);
    std::vector<std::uint8_t> bytes(end - start);
    input.seekg(static_cast<std::streamoff>(data_begin + start));
    input.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    if (!input) throw std::runtime_error("Cannot read tensor: " + name);
    tensor.copy_raw_from_host(bytes);
}
} // namespace

void export_huggingface_model(const std::filesystem::path& output_directory,
                              const bpe::BpeTokenizer& tokenizer,
                              const TransformerModelWeights& weights,
                              const TransformerModelOptions& options,
                              const std::size_t max_position_embeddings) {
    std::filesystem::create_directories(output_directory);
    write_config(output_directory / "config.json", options, max_position_embeddings);
    tokenizer.save(output_directory / "tokenizer.json");
    std::ofstream tokenizer_config(output_directory / "tokenizer_config.json");
    tokenizer_config << "{\n  \"model_max_length\": " << max_position_embeddings
                     << ",\n  \"tokenizer_class\": \"PreTrainedTokenizerFast\"\n}\n";

    std::vector<ExportTensor> tensors;
    add_tensor(tensors, "model.embed_tokens.weight", weights.token_embedding);
    add_tensor(tensors, "model.norm.weight", weights.final_norm);
    add_tensor(tensors, "lm_head.weight", weights.lm_head);
    for (std::size_t layer = 0; layer < weights.layers.size(); ++layer) {
        const auto& block = weights.layers[layer];
        const std::string prefix = "model.layers." + std::to_string(layer) + ".";
        add_tensor(tensors, prefix + "input_layernorm.weight", block.attention_norm);
        add_tensor(tensors, prefix + "self_attn.q_proj.weight", block.q_projection);
        add_tensor(tensors, prefix + "self_attn.k_proj.weight", block.k_projection);
        add_tensor(tensors, prefix + "self_attn.q_norm.weight", block.q_norm);
        add_tensor(tensors, prefix + "self_attn.k_norm.weight", block.k_norm);
        add_tensor(tensors, prefix + "self_attn.v_proj.weight", block.v_projection);
        add_tensor(tensors, prefix + "self_attn.o_proj.weight", block.o_projection);
        add_tensor(tensors, prefix + "post_attention_layernorm.weight", block.ffn_norm);
        add_tensor(tensors, prefix + "mlp.gate_proj.weight", block.gate_proj);
        add_tensor(tensors, prefix + "mlp.up_proj.weight", block.up_proj);
        add_tensor(tensors, prefix + "mlp.down_proj.weight", block.down_proj);
    }
    write_safetensors(output_directory / "model.safetensors", tensors);
}

void load_huggingface_model(const std::filesystem::path& input_directory,
                            const MutableTransformerModelWeights& weights) {
    std::ifstream input(input_directory / "model.safetensors", std::ios::binary);
    if (!input) throw std::runtime_error("Cannot open " + (input_directory / "model.safetensors").string());
    std::uint64_t header_size = 0;
    input.read(reinterpret_cast<char*>(&header_size), sizeof(header_size));
    if (!input || header_size > 256U * 1024U * 1024U) throw std::runtime_error("Invalid safetensors header");
    std::string header(static_cast<std::size_t>(header_size), '\0');
    input.read(header.data(), static_cast<std::streamsize>(header.size()));
    if (!input) throw std::runtime_error("Cannot read safetensors header");
    const std::size_t data_begin = sizeof(header_size) + header.size();
    const auto load = [&](const std::string& name, Tensor& tensor) {
        load_one_tensor(header, input, data_begin, name, tensor);
    };
    load("model.embed_tokens.weight", weights.token_embedding);
    load("model.norm.weight", weights.final_norm);
    load("lm_head.weight", weights.lm_head);
    for (std::size_t layer = 0; layer < weights.layers.size(); ++layer) {
        auto& block = weights.layers[layer];
        const std::string prefix = "model.layers." + std::to_string(layer) + ".";
        load(prefix + "input_layernorm.weight", block.attention_norm);
        load(prefix + "self_attn.q_proj.weight", block.q_projection);
        load(prefix + "self_attn.k_proj.weight", block.k_projection);
        load(prefix + "self_attn.q_norm.weight", block.q_norm);
        load(prefix + "self_attn.k_norm.weight", block.k_norm);
        load(prefix + "self_attn.v_proj.weight", block.v_projection);
        load(prefix + "self_attn.o_proj.weight", block.o_projection);
        load(prefix + "post_attention_layernorm.weight", block.ffn_norm);
        load(prefix + "mlp.gate_proj.weight", block.gate_proj);
        load(prefix + "mlp.up_proj.weight", block.up_proj);
        load(prefix + "mlp.down_proj.weight", block.down_proj);
    }
}
