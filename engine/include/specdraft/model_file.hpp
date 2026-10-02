#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include "specdraft/quant.hpp"  // QK, and the block layouts the formats describe

// Reading the weights file written by python/specdraft/export.py. The file is
// memory-mapped and never copied: the OS pages weights in as the forward pass touches
// them, which is also what makes the process start instantly.

namespace specdraft {

// i32 is not a weight format: it carries the vocabulary map of a trimmed output layer.
enum class Format : uint32_t { fp32 = 0, q4 = 1, q8 = 2, i32 = 3 };

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
    // Rows in a trimmed output layer, or 0 when it spans the whole vocabulary. A trimmed draft
    // can only propose the tokens it kept, which costs acceptance but saves a quarter of its
    // bytes per step; the acceptance rule needs no change, since it only reads q where the draft
    // actually proposed.
    uint32_t output_vocab = 0;
    uint32_t tie_word_embeddings = 0;
    double rms_norm_eps = 0.0;
    double rope_theta = 0.0;

    uint32_t q_dim() const { return num_attention_heads * head_dim; }
    uint32_t kv_dim() const { return num_key_value_heads * head_dim; }
    uint32_t group_size() const { return num_attention_heads / num_key_value_heads; }
};

// A quantized tensor is stored as two regions rather than one array of blocks: every row's scales
// first, then every row's quantized bytes. The kernels need eight consecutive scales to convert in
// one instruction, which interleaved blocks cannot give them. `data` points at the first region, or
// at the values themselves for fp32 and i32; `qs` points at the second, and is null for those.
struct Tensor {
    const void* data = nullptr;
    const void* qs = nullptr;
    Format format = Format::fp32;
    uint32_t rows = 0;
    uint32_t cols = 1;  // 1 for vectors
    size_t nbytes = 0;

    bool quantized() const { return format == Format::q4 || format == Format::q8; }
    uint32_t blocks_per_row() const { return cols / QK; }
    size_t qs_bytes_per_row() const { return format == Format::q4 ? blocks_per_row() * (QK / 2)
                                                                 : blocks_per_row() * QK; }

    size_t stride() const { return row_bytes(format, cols); }
    // For fp32 and i32 only; a quantized row needs the two accessors below.
    const void* row(uint32_t index) const {
        return static_cast<const uint8_t*>(data) + static_cast<size_t>(index) * stride();
    }
    const uint16_t* row_scales(uint32_t index) const {
        return static_cast<const uint16_t*>(data) + static_cast<size_t>(index) * blocks_per_row();
    }
    const void* row_qs(uint32_t index) const {
        return static_cast<const uint8_t*>(qs) + static_cast<size_t>(index) * qs_bytes_per_row();
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
