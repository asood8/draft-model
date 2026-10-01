#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

// Reading the weights file written by python/specdraft/export.py. The file is
// memory-mapped and never copied: the OS pages weights in as the forward pass touches
// them, which is also what makes the process start instantly.

namespace specdraft {

enum class Format : uint32_t { fp32 = 0, q4 = 1, q8 = 2 };

const char* format_name(Format format);

// Bytes one row of `cols` values occupies in each format.
size_t row_bytes(Format format, uint32_t cols);

struct ModelConfig {
    uint32_t vocab_size = 0;
    uint32_t hidden_size = 0;
    uint32_t intermediate_size = 0;
    uint32_t num_hidden_layers = 0;
    uint32_t num_attention_heads = 0;
    uint32_t num_key_value_heads = 0;
    uint32_t head_dim = 0;  // from the file, not hidden_size / heads
    uint32_t vocab_limit = 0;  // the tokenizer's size; rows past it are padding
    uint32_t tie_word_embeddings = 0;
    double rms_norm_eps = 0.0;
    double rope_theta = 0.0;

    uint32_t q_dim() const { return num_attention_heads * head_dim; }
    uint32_t kv_dim() const { return num_key_value_heads * head_dim; }
    uint32_t group_size() const { return num_attention_heads / num_key_value_heads; }
};

struct Tensor {
    const void* data = nullptr;
    Format format = Format::fp32;
    uint32_t rows = 0;
    uint32_t cols = 1;  // 1 for vectors
    size_t nbytes = 0;

    size_t stride() const { return row_bytes(format, cols); }
    const void* row(uint32_t index) const {
        return static_cast<const uint8_t*>(data) + static_cast<size_t>(index) * stride();
    }
};

class ModelFile {
public:
    ModelFile() = default;
    ~ModelFile();
    ModelFile(const ModelFile&) = delete;
    ModelFile& operator=(const ModelFile&) = delete;
    ModelFile(ModelFile&& other) noexcept;
    ModelFile& operator=(ModelFile&& other) noexcept;

    // Throws std::runtime_error if the file is missing, truncated or the wrong version.
    static ModelFile open(const std::string& path);

    const ModelConfig& config() const { return config_; }
    bool has(const std::string& name) const { return tensors_.count(name) != 0; }
    const Tensor& tensor(const std::string& name) const;  // throws if absent
    std::vector<std::string> names() const;
    size_t size_bytes() const { return size_; }

private:
    void close();

    void* mapping_ = nullptr;
    size_t size_ = 0;
#if defined(_WIN32)
    void* file_handle_ = nullptr;
    void* map_handle_ = nullptr;
#else
    int fd_ = -1;
#endif
    ModelConfig config_;
    std::unordered_map<std::string, Tensor> tensors_;
};

}  // namespace specdraft
