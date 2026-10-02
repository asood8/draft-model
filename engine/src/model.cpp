#include "specdraft/model.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>

#include "specdraft/simd.hpp"

namespace specdraft {
namespace {

using Clock = std::chrono::steady_clock;

// RMSNorm with the reduction in float32, as Qwen3 does it. Safe with out == x.
void rms_norm(const float* x, const float* weight, uint32_t n, float eps, float* out) {
    const float scale = 1.0f / std::sqrt(sum_of_squares(x, n) / static_cast<float>(n) + eps);
    scale_and_weight(x, weight, scale, n, out);
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

Model::Model(ModelFile file, EngineOptions options) : file_(std::move(file)), options_(options) {
    if (options_.max_positions <= 0) {
        throw std::runtime_error("max_positions must be positive");
    }
    if (options_.max_batch <= 0) {
        throw std::runtime_error("max_batch must be positive");
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
    if (file_.has("output_map")) {
        const Tensor& map = file_.tensor("output_map");
        if (map.format != Format::i32) {
            throw std::runtime_error("output_map must be i32");
        }
        if (map.rows != c.output_vocab || c.output_vocab == 0) {
            throw std::runtime_error("output_map does not match output_vocab");
        }
        if (output_.rows < c.output_vocab) {
            throw std::runtime_error("the output layer has fewer rows than output_vocab");
        }
        output_map_ = static_cast<const int32_t*>(map.data);
        for (uint32_t i = 0; i < c.output_vocab; ++i) {
            const int32_t token = output_map_[i];
            if (token < 0 || static_cast<uint32_t>(token) >= c.vocab_limit) {
                throw std::runtime_error("output_map refers to a token outside the vocabulary");
            }
        }
    } else if (c.output_vocab != 0) {
        throw std::runtime_error("output_vocab is set but no output_map was written");
    }
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

    const size_t batch = static_cast<size_t>(options_.max_batch);
    qkv_stride_ = c.q_dim() + 2 * c.kv_dim();
    mlp_stride_ = 2 * c.intermediate_size;
    blocks_stride_ = std::max({c.hidden_size, c.q_dim(), c.intermediate_size}) / QK;

    x_.resize(batch * c.hidden_size);
    xb_.resize(batch * c.hidden_size);
    xb2_.resize(batch * c.hidden_size);
    qkv_.resize(batch * qkv_stride_);
    att_.resize(batch * c.q_dim());
    mlp_.resize(batch * mlp_stride_);
    scores_.resize(batch * c.num_attention_heads * options_.max_positions);
    act_scales_.resize(batch * blocks_stride_);
    act_qs_.resize(batch * blocks_stride_ * QK);
    act_bias_.resize(batch * blocks_stride_ * 8);
    kv_scratch_.resize(static_cast<size_t>(pool_->size()) * 2 * c.head_dim);
    row_scratch_.resize(static_cast<size_t>(pool_->size()) * batch);

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
    // scores -- which is where trimming the vocabulary saves its bytes.
    total += row_bytes(output_.format, output_.cols) * logit_count();
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

size_t Model::score_index(int token, uint32_t head) const {
    const ModelConfig& c = config();
    return (static_cast<size_t>(token) * c.num_attention_heads + head) * options_.max_positions;
}

void Model::embed(int32_t token, float* out) const {
    const ModelConfig& c = config();
    if (token < 0 || static_cast<uint32_t>(token) >= c.vocab_size) {
        throw std::runtime_error("token id out of range");
    }
    const uint32_t id = static_cast<uint32_t>(token);
    switch (embedding_.format) {
        case Format::fp32:
            std::memcpy(out, embedding_.row(id), c.hidden_size * sizeof(float));
            break;
        case Format::q4:
            dequantize_q4_soa(embedding_.row_scales(id),
                              static_cast<const uint8_t*>(embedding_.row_qs(id)), c.hidden_size, out);
            break;
        case Format::q8:
            dequantize_q8_soa(embedding_.row_scales(id),
                              static_cast<const int8_t*>(embedding_.row_qs(id)), c.hidden_size, out);
            break;
    }
}

void Model::matmul(const Tensor& weight, const float* in, uint32_t in_stride, uint32_t n_in,
                   float* out, uint32_t out_stride, uint32_t n_out, int batch) {
    if (weight.cols != n_in || weight.rows < n_out) {
        throw std::runtime_error("matmul shape mismatch");
    }
    const int rows = static_cast<int>(n_out);
    const int tokens = batch;

    if (weight.format == Format::fp32) {
        const float* data = static_cast<const float*>(weight.data);
        pool_->run(rows, [&](int begin, int end, int) {
            for (int r = begin; r < end; ++r) {
                const float* row = data + static_cast<size_t>(r) * n_in;
                for (int t = 0; t < tokens; ++t) {
                    out[static_cast<size_t>(t) * out_stride + r] =
                        dot_f32(row, in + static_cast<size_t>(t) * in_stride, n_in);
                }
            }
        });
        return;
    }

    // Quantize each token's activation vector once, here, so the row loop below is pure
    // integer work that any worker can do independently. The zero point comes from the *weight*
    // format, since the bias it folds into the accumulator corrects for the weights being stored
    // unsigned -- 8 for q4's nibbles, 128 for q8's bytes.
    const int nblocks = static_cast<int>(n_in / QK);
    const bool four_bit = weight.format == Format::q4;
    const int zero_point = four_bit ? 8 : 128;
    for (int t = 0; t < tokens; ++t) {
        const size_t block = static_cast<size_t>(t) * nblocks;
        quantize_a8_soa(in + static_cast<size_t>(t) * in_stride, static_cast<int>(n_in), zero_point,
                        act_scales_.data() + block, act_qs_.data() + block * QK,
                        act_bias_.data() + block * 8);
    }
    const float* x_scales = act_scales_.data();
    const int8_t* x_qs = act_qs_.data();
    const int32_t* x_bias = act_bias_.data();
    const int batch_width = options_.max_batch;

    const auto body = [&](int begin, int end, int worker) {
        // One pass over each weight row serves every token in the batch, which is what makes
        // verifying γ+1 guesses cheaper than γ+1 separate steps.
        float* results = row_scratch_.data() + static_cast<size_t>(worker) * batch_width;
        for (int r = begin; r < end; ++r) {
            const uint32_t row = static_cast<uint32_t>(r);
            if (four_bit) {
                dot_q4_soa_multi(weight.row_scales(row),
                                 static_cast<const uint8_t*>(weight.row_qs(row)), x_scales, x_qs,
                                 x_bias, nblocks, tokens, results);
            } else {
                dot_q8_soa_multi(weight.row_scales(row),
                                 static_cast<const int8_t*>(weight.row_qs(row)), x_scales, x_qs,
                                 x_bias, nblocks, tokens, results);
            }
            for (int t = 0; t < tokens; ++t) {
                out[static_cast<size_t>(t) * out_stride + r] = results[t];
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

void Model::attention(int layer_index, int base, int batch) {
    const ModelConfig& c = config();
    const uint32_t head_dim = c.head_dim;
    const uint32_t group = c.group_size();
    const uint32_t q_dim = c.q_dim();
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    const int total = base + batch;  // every position now in the cache

    // One job per key/value head: each cached K and V row is read once and shared by the group
    // of query heads that use it *and* by every token in the batch.
    pool_->run(static_cast<int>(c.num_key_value_heads), [&](int begin, int end, int worker) {
        float* keys_buffer = kv_scratch_.data() + static_cast<size_t>(worker) * 2 * head_dim;
        float* values_buffer = keys_buffer + head_dim;

        for (int kv = begin; kv < end; ++kv) {
            const uint32_t first_head = static_cast<uint32_t>(kv) * group;

            for (int t = 0; t < total; ++t) {
                fp16_to_fp32_many(key_cache_.data() + cache_index(layer_index, kv, t), keys_buffer,
                                  head_dim);
                // Token j sits at position base + j, so it may attend to t only if t <= base + j.
                const int first_token = std::max(0, t - base);
                for (int j = first_token; j < batch; ++j) {
                    for (uint32_t g = 0; g < group; ++g) {
                        const uint32_t h = first_head + g;
                        const float* q =
                            qkv_.data() + static_cast<size_t>(j) * qkv_stride_ + h * head_dim;
                        scores_[score_index(j, h) + t] = dot_f32(q, keys_buffer, head_dim) * scale;
                    }
                }
            }

            for (int j = 0; j < batch; ++j) {
                for (uint32_t g = 0; g < group; ++g) {
                    const uint32_t h = first_head + g;
                    softmax(scores_.data() + score_index(j, h), base + j + 1);
                    float* out = att_.data() + static_cast<size_t>(j) * q_dim + h * head_dim;
                    std::fill(out, out + head_dim, 0.0f);
                }
            }

            for (int t = 0; t < total; ++t) {
                fp16_to_fp32_many(value_cache_.data() + cache_index(layer_index, kv, t),
                                  values_buffer, head_dim);
                const int first_token = std::max(0, t - base);
                for (int j = first_token; j < batch; ++j) {
                    for (uint32_t g = 0; g < group; ++g) {
                        const uint32_t h = first_head + g;
                        accumulate_scaled(
                            att_.data() + static_cast<size_t>(j) * q_dim + h * head_dim,
                            values_buffer, scores_[score_index(j, h) + t], head_dim);
                    }
                }
            }
        }
    });
}

void Model::forward_batch(const int32_t* tokens, int batch, int base, float* logits_out,
                          bool all_logits, int single_token, float* capture_out,
                          int capture_offset, int capture_total) {
    const ModelConfig& c = config();
    const float eps = static_cast<float>(c.rms_norm_eps);
    const float theta = static_cast<float>(c.rope_theta);
    const uint32_t head_dim = c.head_dim;
    const uint32_t hidden = c.hidden_size;
    const uint32_t q_dim = c.q_dim();
    const uint32_t kv_dim = c.kv_dim();
    const uint32_t inter = c.intermediate_size;

    Clock::time_point mark = timing_ ? Clock::now() : Clock::time_point{};
    const auto tick = [&](Stage stage) {
        if (!timing_) {
            return;
        }
        const Clock::time_point now = Clock::now();
        stage_seconds_[stage] += std::chrono::duration<double>(now - mark).count();
        mark = now;
    };

    for (int j = 0; j < batch; ++j) {
        embed(tokens[j], x_.data() + static_cast<size_t>(j) * hidden);
    }
    tick(kEmbed);

    for (uint32_t l = 0; l < layers_.size(); ++l) {
        const Layer& layer = layers_[l];

        for (int j = 0; j < batch; ++j) {
            rms_norm(x_.data() + static_cast<size_t>(j) * hidden, layer.attn_norm, hidden, eps,
                     xb_.data() + static_cast<size_t>(j) * hidden);
        }
        matmul(layer.qkv, xb_.data(), hidden, hidden, qkv_.data(), qkv_stride_, q_dim + 2 * kv_dim,
               batch);

        for (int j = 0; j < batch; ++j) {
            float* q = qkv_.data() + static_cast<size_t>(j) * qkv_stride_;
            float* keys = q + q_dim;
            float* values = keys + kv_dim;
            const int position = base + j;

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
                fp32_to_fp16_many(keys + static_cast<size_t>(h) * head_dim,
                                  key_cache_.data() + cache_index(static_cast<int>(l), h, position),
                                  head_dim);
                fp32_to_fp16_many(
                    values + static_cast<size_t>(h) * head_dim,
                    value_cache_.data() + cache_index(static_cast<int>(l), h, position), head_dim);
            }
        }
        tick(kQkv);

        attention(static_cast<int>(l), base, batch);
        tick(kAttention);

        matmul(layer.attn_out, att_.data(), q_dim, q_dim, xb2_.data(), hidden, hidden, batch);
        for (int j = 0; j < batch; ++j) {
            add_in_place(x_.data() + static_cast<size_t>(j) * hidden,
                         xb2_.data() + static_cast<size_t>(j) * hidden, hidden);
        }
        tick(kAttnOut);

        for (int j = 0; j < batch; ++j) {
            rms_norm(x_.data() + static_cast<size_t>(j) * hidden, layer.ffn_norm, hidden, eps,
                     xb_.data() + static_cast<size_t>(j) * hidden);
        }
        matmul(layer.gate_up, xb_.data(), hidden, hidden, mlp_.data(), mlp_stride_, 2 * inter,
               batch);
        for (int j = 0; j < batch; ++j) {
            float* row = mlp_.data() + static_cast<size_t>(j) * mlp_stride_;
            for (uint32_t i = 0; i < inter; ++i) {
                const float gate = row[i];
                row[i] = gate / (1.0f + std::exp(-gate)) * row[inter + i];  // SwiGLU
            }
        }
        tick(kGateUp);

        matmul(layer.ffn_down, mlp_.data(), mlp_stride_, inter, xb2_.data(), hidden, hidden, batch);
        for (int j = 0; j < batch; ++j) {
            add_in_place(x_.data() + static_cast<size_t>(j) * hidden,
                         xb2_.data() + static_cast<size_t>(j) * hidden, hidden);
        }
        tick(kFfnDown);

        if (capture_out != nullptr) {
            for (int j = 0; j < batch; ++j) {
                float* destination =
                    capture_out + (static_cast<size_t>(l) * capture_total + capture_offset + j) *
                                      hidden;
                std::memcpy(destination, x_.data() + static_cast<size_t>(j) * hidden,
                            hidden * sizeof(float));
            }
        }
    }

    if (logits_out != nullptr && (all_logits || single_token >= 0)) {
        const int first = all_logits ? 0 : single_token;
        const int count = all_logits ? batch : 1;
        for (int j = 0; j < count; ++j) {
            rms_norm(x_.data() + static_cast<size_t>(first + j) * hidden, output_norm_, hidden, eps,
                     xb_.data() + static_cast<size_t>(j) * hidden);
        }
        // A trimmed output layer scores only the tokens it kept, so a pass writes that many
        // logits per token rather than one per vocabulary entry.
        const uint32_t width = logit_count();
        matmul(output_, xb_.data(), hidden, hidden, logits_out, width, width, count);
        tick(kOutput);
    }
    if (timing_) {
        timed_tokens_ += static_cast<uint64_t>(batch);
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

    // Longer sequences are split into chunks that share one pass over the weights each.
    const uint32_t width = logit_count();  // fewer than the vocabulary when the output is trimmed
    int done = 0;
    while (done < k) {
        const int batch = std::min(k - done, options_.max_batch);
        const bool last_chunk = done + batch == k;
        float* chunk_logits = nullptr;
        int single_token = -1;
        if (logits_out != nullptr) {
            if (all_logits) {
                chunk_logits = logits_out + static_cast<size_t>(done) * width;
            } else if (last_chunk) {
                chunk_logits = logits_out;
                single_token = batch - 1;
            }
        }
        forward_batch(tokens + done, batch, pos_ + done, chunk_logits, all_logits && chunk_logits,
                      single_token, capture_out, done, k);
        done += batch;
    }
    pos_ += k;
}

}  // namespace specdraft
