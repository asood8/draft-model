#include "specdraft/model.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>

namespace specdraft {
namespace {

using Clock = std::chrono::steady_clock;

// RMSNorm with the reduction in float32, as Qwen3 does it. Safe with out == x.
void rms_norm(const float* x, const float* weight, uint32_t n, float eps, float* out) {
    float sum = 0.0f;
    for (uint32_t i = 0; i < n; ++i) {
        sum += x[i] * x[i];
    }
    const float scale = 1.0f / std::sqrt(sum / static_cast<float>(n) + eps);
    for (uint32_t i = 0; i < n; ++i) {
        out[i] = x[i] * scale * weight[i];
    }
}

// Rotate the two halves of one head, matching Hugging Face's layout: element i pairs with
// i + head_dim/2, and the angle for pair i is position / theta^(2i/head_dim).
void rope(float* vec, uint32_t head_dim, int position, float theta) {
    const uint32_t half = head_dim / 2;
    for (uint32_t i = 0; i < half; ++i) {
        const float inv_freq =
            1.0f / std::pow(theta, (2.0f * static_cast<float>(i)) / static_cast<float>(head_dim));
        const float angle = static_cast<float>(position) * inv_freq;
        const float c = std::cos(angle);
        const float s = std::sin(angle);
        const float a = vec[i];
        const float b = vec[i + half];
        vec[i] = a * c - b * s;
        vec[i + half] = b * c + a * s;
    }
}

void softmax(float* values, int n) {
    float largest = values[0];
    for (int i = 1; i < n; ++i) {
        largest = std::max(largest, values[i]);
    }
    float total = 0.0f;
    for (int i = 0; i < n; ++i) {
        values[i] = std::exp(values[i] - largest);
        total += values[i];
    }
    for (int i = 0; i < n; ++i) {
        values[i] /= total;
    }
}

const float* as_floats(const Tensor& tensor, const char* name) {
    if (tensor.format != Format::fp32) {
        throw std::runtime_error(std::string(name) + " must be fp32");
    }
    return static_cast<const float*>(tensor.data);
}

void require_quantizable(const Tensor& tensor, const char* name) {
    if (tensor.format != Format::fp32 && tensor.cols % QK != 0) {
        throw std::runtime_error(std::string(name) +
                                 " has a row length that is not a multiple of 32");
    }
}

}  // namespace

Model::Model(ModelFile file, EngineOptions options)
    : file_(std::move(file)), options_(options) {
    if (options_.max_positions <= 0) {
        throw std::runtime_error("max_positions must be positive");
    }
    const ModelConfig& c = config();
    if (c.head_dim % 2 != 0) {
        throw std::runtime_error("head_dim must be even for rotary embeddings");
    }
    if (c.num_key_value_heads == 0 || c.num_attention_heads % c.num_key_value_heads != 0) {
        throw std::runtime_error("num_attention_heads must be a multiple of num_key_value_heads");
    }
    if (c.vocab_limit > c.vocab_size) {
        throw std::runtime_error("vocab_limit exceeds vocab_size");
    }

    pool_ = std::make_unique<ThreadPool>(options_.threads, options_.cores);

    embedding_ = file_.tensor("token_embd");
    output_ = file_.has("output") ? file_.tensor("output") : embedding_;
    output_norm_ = as_floats(file_.tensor("output_norm"), "output_norm");
    require_quantizable(embedding_, "token_embd");

    layers_.resize(c.num_hidden_layers);
    for (uint32_t i = 0; i < c.num_hidden_layers; ++i) {
        const std::string prefix = "blk." + std::to_string(i) + ".";
        Layer& layer = layers_[i];
        layer.attn_norm = as_floats(file_.tensor(prefix + "attn_norm"), "attn_norm");
        layer.q_norm = as_floats(file_.tensor(prefix + "q_norm"), "q_norm");
        layer.k_norm = as_floats(file_.tensor(prefix + "k_norm"), "k_norm");
        layer.ffn_norm = as_floats(file_.tensor(prefix + "ffn_norm"), "ffn_norm");
        layer.qkv = file_.tensor(prefix + "qkv");
        layer.attn_out = file_.tensor(prefix + "attn_out");
        layer.gate_up = file_.tensor(prefix + "gate_up");
        layer.ffn_down = file_.tensor(prefix + "ffn_down");
        require_quantizable(layer.qkv, "qkv");
        require_quantizable(layer.attn_out, "attn_out");
        require_quantizable(layer.gate_up, "gate_up");
        require_quantizable(layer.ffn_down, "ffn_down");
    }

    x_.resize(c.hidden_size);
    xb_.resize(c.hidden_size);
    xb2_.resize(c.hidden_size);
    qkv_.resize(c.q_dim() + 2 * c.kv_dim());
    att_.resize(c.q_dim());
    scores_.resize(static_cast<size_t>(c.num_attention_heads) * options_.max_positions);
    mlp_.resize(2 * c.intermediate_size);
    kv_scratch_.resize(static_cast<size_t>(pool_->size()) * 2 * c.head_dim);
    activations_.resize(std::max({c.hidden_size, c.q_dim(), c.intermediate_size}) / QK);

    const size_t cache_values = static_cast<size_t>(c.num_hidden_layers) * c.num_key_value_heads *
                                options_.max_positions * c.head_dim;
    key_cache_.assign(cache_values, 0);
    value_cache_.assign(cache_values, 0);
}

Model::~Model() = default;

int Model::threads() const {
    return pool_->size();
}

const char* Model::stage_name(Stage stage) {
    switch (stage) {
        case kEmbed:
            return "embed";
        case kQkv:
            return "qkv";
        case kAttention:
            return "attention";
        case kAttnOut:
            return "attn_out";
        case kGateUp:
            return "gate_up";
        case kFfnDown:
            return "ffn_down";
        case kOutput:
            return "output";
        case kStageCount:
            break;
    }
    return "?";
}

void Model::set_timing(bool enabled) {
    timing_ = enabled;
}

void Model::reset_timings() {
    stage_seconds_.fill(0.0);
    timed_tokens_ = 0;
}

size_t Model::weight_bytes_per_token() const {
    const ModelConfig& c = config();
    size_t total = 0;
    for (const Layer& layer : layers_) {
        total += layer.qkv.nbytes + layer.attn_out.nbytes + layer.gate_up.nbytes +
                 layer.ffn_down.nbytes;
    }
    // The embedding is only read a row at a time, but the output layer reads every row it
    // scores, which is vocab_limit of them.
    total += row_bytes(output_.format, output_.cols) * c.vocab_limit;
    return total;
}

size_t Model::kv_bytes_per_token() const {
    const ModelConfig& c = config();
    // One K and one V row per layer per kv head, read once for every position in the context.
    return static_cast<size_t>(c.num_hidden_layers) * c.num_key_value_heads * c.head_dim * 2 *
           sizeof(uint16_t);
}

void Model::set_pos(int position) {
    if (position < 0 || position > options_.max_positions) {
        throw std::runtime_error("position out of range");
    }
    pos_ = position;
}

size_t Model::cache_index(int layer, uint32_t kv_head, int position) const {
    const ModelConfig& c = config();
    return ((static_cast<size_t>(layer) * c.num_key_value_heads + kv_head) * options_.max_positions +
            position) *
           c.head_dim;
}

void Model::embed(int32_t token, float* out) const {
    const ModelConfig& c = config();
    if (token < 0 || static_cast<uint32_t>(token) >= c.vocab_size) {
        throw std::runtime_error("token id out of range");
    }
    const void* row = embedding_.row(static_cast<uint32_t>(token));
    switch (embedding_.format) {
        case Format::fp32:
            std::memcpy(out, row, c.hidden_size * sizeof(float));
            break;
        case Format::q4:
            dequantize_q4(static_cast<const BlockQ4*>(row), c.hidden_size, out);
            break;
        case Format::q8:
            dequantize_q8(static_cast<const BlockQ8*>(row), c.hidden_size, out);
            break;
    }
}

void Model::matvec(const Tensor& weight, const float* x, uint32_t n_in, float* out,
                   uint32_t n_out) {
    if (weight.cols != n_in || weight.rows < n_out) {
        throw std::runtime_error("matvec shape mismatch");
    }
    const int rows = static_cast<int>(n_out);

    if (weight.format == Format::fp32) {
        const float* data = static_cast<const float*>(weight.data);
        pool_->run(rows, [&](int begin, int end, int) {
            for (int r = begin; r < end; ++r) {
                const float* row = data + static_cast<size_t>(r) * n_in;
                float sum = 0.0f;
                for (uint32_t i = 0; i < n_in; ++i) {
                    sum += row[i] * x[i];
                }
                out[r] = sum;
            }
        });
        return;
    }

    // Quantize the activation vector once, on this thread, then every row is an integer dot
    // product that any worker can do independently.
    const int nblocks = static_cast<int>(n_in / QK);
    quantize_a8(x, static_cast<int>(n_in), activations_.data());
    const BlockA8* activations = activations_.data();
    const bool four_bit = weight.format == Format::q4;

    const auto body = [&](int begin, int end, int) {
        if (four_bit) {
            for (int r = begin; r < end; ++r) {
                out[r] = dot_q4_a8(static_cast<const BlockQ4*>(weight.row(r)), activations, nblocks);
            }
        } else {
            for (int r = begin; r < end; ++r) {
                out[r] = dot_q8_a8(static_cast<const BlockQ8*>(weight.row(r)), activations, nblocks);
            }
        }
    };

    if (options_.dynamic_schedule) {
        const int chunk = std::max(1, rows / (pool_->size() * 8));
        pool_->run_dynamic(rows, chunk, body);
    } else {
        pool_->run(rows, body);
    }
}

void Model::attention(int layer_index, int position) {
    const ModelConfig& c = config();
    const uint32_t head_dim = c.head_dim;
    const uint32_t group = c.group_size();
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    const int length = position + 1;
    const size_t stride = static_cast<size_t>(options_.max_positions);

    // One job per key/value head, so every cached K and V row is read once and shared by the
    // group of query heads that use it, instead of once per query head.
    pool_->run(static_cast<int>(c.num_key_value_heads), [&](int begin, int end, int worker) {
        float* keys_buffer = kv_scratch_.data() + static_cast<size_t>(worker) * 2 * head_dim;
        float* values_buffer = keys_buffer + head_dim;

        for (int kv = begin; kv < end; ++kv) {
            const uint32_t first_head = static_cast<uint32_t>(kv) * group;

            for (int t = 0; t < length; ++t) {
                const uint16_t* cached = key_cache_.data() + cache_index(layer_index, kv, t);
                for (uint32_t d = 0; d < head_dim; ++d) {
                    keys_buffer[d] = fp16_to_fp32(cached[d]);
                }
                for (uint32_t g = 0; g < group; ++g) {
                    const uint32_t h = first_head + g;
                    const float* q = qkv_.data() + static_cast<size_t>(h) * head_dim;
                    float dot = 0.0f;
                    for (uint32_t d = 0; d < head_dim; ++d) {
                        dot += q[d] * keys_buffer[d];
                    }
                    scores_[static_cast<size_t>(h) * stride + t] = dot * scale;
                }
            }

            for (uint32_t g = 0; g < group; ++g) {
                const uint32_t h = first_head + g;
                softmax(scores_.data() + static_cast<size_t>(h) * stride, length);
                float* out = att_.data() + static_cast<size_t>(h) * head_dim;
                std::fill(out, out + head_dim, 0.0f);
            }

            for (int t = 0; t < length; ++t) {
                const uint16_t* cached = value_cache_.data() + cache_index(layer_index, kv, t);
                for (uint32_t d = 0; d < head_dim; ++d) {
                    values_buffer[d] = fp16_to_fp32(cached[d]);
                }
                for (uint32_t g = 0; g < group; ++g) {
                    const uint32_t h = first_head + g;
                    const float weight = scores_[static_cast<size_t>(h) * stride + t];
                    float* out = att_.data() + static_cast<size_t>(h) * head_dim;
                    for (uint32_t d = 0; d < head_dim; ++d) {
                        out[d] += weight * values_buffer[d];
                    }
                }
            }
        }
    });
}

void Model::forward_one(int32_t token, int position, float* logits_out, float* capture_out, int k,
                        int token_index) {
    const ModelConfig& c = config();
    const float eps = static_cast<float>(c.rms_norm_eps);
    const float theta = static_cast<float>(c.rope_theta);
    const uint32_t head_dim = c.head_dim;
    const uint32_t q_dim = c.q_dim();
    const uint32_t kv_dim = c.kv_dim();

    Clock::time_point mark = timing_ ? Clock::now() : Clock::time_point{};
    const auto tick = [&](Stage stage) {
        if (!timing_) {
            return;
        }
        const Clock::time_point now = Clock::now();
        stage_seconds_[stage] += std::chrono::duration<double>(now - mark).count();
        mark = now;
    };

    embed(token, x_.data());
    tick(kEmbed);

    for (uint32_t l = 0; l < layers_.size(); ++l) {
        const Layer& layer = layers_[l];

        rms_norm(x_.data(), layer.attn_norm, c.hidden_size, eps, xb_.data());
        matvec(layer.qkv, xb_.data(), c.hidden_size, qkv_.data(), q_dim + 2 * kv_dim);

        float* q = qkv_.data();
        float* keys = q + q_dim;
        float* values = keys + kv_dim;

        // Qwen3 normalizes queries and keys per head, before the rotation.
        for (uint32_t h = 0; h < c.num_attention_heads; ++h) {
            float* head = q + static_cast<size_t>(h) * head_dim;
            rms_norm(head, layer.q_norm, head_dim, eps, head);
            rope(head, head_dim, position, theta);
        }
        for (uint32_t h = 0; h < c.num_key_value_heads; ++h) {
            float* head = keys + static_cast<size_t>(h) * head_dim;
            rms_norm(head, layer.k_norm, head_dim, eps, head);
            rope(head, head_dim, position, theta);
        }

        for (uint32_t h = 0; h < c.num_key_value_heads; ++h) {
            uint16_t* key_slot = key_cache_.data() + cache_index(static_cast<int>(l), h, position);
            uint16_t* value_slot =
                value_cache_.data() + cache_index(static_cast<int>(l), h, position);
            for (uint32_t d = 0; d < head_dim; ++d) {
                key_slot[d] = fp32_to_fp16(keys[static_cast<size_t>(h) * head_dim + d]);
                value_slot[d] = fp32_to_fp16(values[static_cast<size_t>(h) * head_dim + d]);
            }
        }
        tick(kQkv);

        attention(static_cast<int>(l), position);
        tick(kAttention);

        matvec(layer.attn_out, att_.data(), q_dim, xb2_.data(), c.hidden_size);
        for (uint32_t i = 0; i < c.hidden_size; ++i) {
            x_[i] += xb2_[i];
        }
        tick(kAttnOut);

        rms_norm(x_.data(), layer.ffn_norm, c.hidden_size, eps, xb_.data());
        matvec(layer.gate_up, xb_.data(), c.hidden_size, mlp_.data(), 2 * c.intermediate_size);
        for (uint32_t i = 0; i < c.intermediate_size; ++i) {
            const float gate = mlp_[i];
            mlp_[i] = gate / (1.0f + std::exp(-gate)) * mlp_[c.intermediate_size + i];  // SwiGLU
        }
        tick(kGateUp);

        matvec(layer.ffn_down, mlp_.data(), c.intermediate_size, xb2_.data(), c.hidden_size);
        for (uint32_t i = 0; i < c.hidden_size; ++i) {
            x_[i] += xb2_[i];
        }
        tick(kFfnDown);

        if (capture_out != nullptr) {
            float* destination =
                capture_out + (static_cast<size_t>(l) * k + token_index) * c.hidden_size;
            std::memcpy(destination, x_.data(), c.hidden_size * sizeof(float));
        }
    }

    if (logits_out != nullptr) {
        rms_norm(x_.data(), output_norm_, c.hidden_size, eps, xb_.data());
        matvec(output_, xb_.data(), c.hidden_size, logits_out, c.vocab_limit);
        tick(kOutput);
    }
    if (timing_) {
        ++timed_tokens_;
    }
}

void Model::forward(const int32_t* tokens, int k, float* logits_out, bool all_logits,
                    float* capture_out) {
    if (k <= 0) {
        throw std::runtime_error("need at least one token");
    }
    if (pos_ + k > options_.max_positions) {
        throw std::runtime_error("sequence longer than max_positions");
    }

    const uint32_t vocab_limit = config().vocab_limit;
    for (int j = 0; j < k; ++j) {
        float* row = nullptr;
        if (logits_out != nullptr) {
            if (all_logits) {
                row = logits_out + static_cast<size_t>(j) * vocab_limit;
            } else if (j == k - 1) {
                row = logits_out;
            }
        }
        forward_one(tokens[j], pos_ + j, row, capture_out, k, j);
    }
    pos_ += k;
}

}  // namespace specdraft
