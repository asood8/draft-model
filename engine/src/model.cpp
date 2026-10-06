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
    act_offsets_.resize(batch * blocks_stride_);
    kv_scratch_.resize(static_cast<size_t>(pool_->size()) * 2 * c.head_dim);
    // One partial sum per chunk of positions, per token, per head: attention adds these up in chunk
    // order so that the result does not depend on which worker ran which chunk. Sized for the
    // longest sequence this model can be asked for, since that decides the chunk count.
    constexpr int kChunkPositions = 256;  // must match kChunk in Model::attention
    const size_t max_chunks =
        (static_cast<size_t>(options_.max_positions) + kChunkPositions - 1) / kChunkPositions;
    partials_.resize(max_chunks * batch * c.q_dim());
    // One cache line per worker, not max_batch floats. Each worker writes its own slice once per
    // weight row, so a stride below 64 bytes puts several workers in one line and every row write
    // invalidates it for the others: at max_batch=2, which is what gamma=1 asks for, all six
    // workers shared a single line. The same mistake in the benchmark harness made one layout look
    // 8x slower than another (plan section 13).
    constexpr size_t kLineFloats = 16;  // 64 bytes
    row_scratch_stride_ = static_cast<uint32_t>(((batch + kLineFloats - 1) / kLineFloats) *
                                               kLineFloats);
    row_scratch_.resize(static_cast<size_t>(pool_->size()) * row_scratch_stride_);

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
    // format, since the offset it precomputes corrects for the weights being stored unsigned --
    // 8 for q4's nibbles, 128 for q8's bytes.
    const int nblocks = static_cast<int>(n_in / QK);
    const bool four_bit = weight.format == Format::q4;
    const int zero_point = four_bit ? 8 : 128;

    // One flat index over (token, block), so a wide ffn_down input splits as evenly across workers as
    // a batch of tokens does, and a slice that spans two tokens becomes one call per token.
    const auto quantize_range = [&](int begin, int end, int) {
        while (begin < end) {
            const int t = begin / nblocks;
            const size_t base = static_cast<size_t>(t) * nblocks;
            quantize_a8_soa_blocks(in + static_cast<size_t>(t) * in_stride, nblocks,
                                  begin - t * nblocks, std::min(nblocks, end - t * nblocks),
                                  zero_point, act_scales_.data() + base,
                                  act_qs_.data() + base * QK, act_offsets_.data() + base, four_bit);
            begin = (t + 1) * nblocks;
        }
    };
    // Below the threshold the dispatch costs more than the work: a forward pass makes about 180 of
    // these calls per token, and a parallel job is 1-5 us depending on how quiet the machine is. The
    // wide ones -- ffn_down takes a 9728-value input, 304 blocks -- are where the time actually is.
    constexpr int kParallelBlocks = 64;
    if (tokens * nblocks >= kParallelBlocks && pool_->size() > 1) {
        pool_->run(tokens * nblocks, quantize_range);
    } else {
        quantize_range(0, tokens * nblocks, 0);
    }
    const float* x_scales = act_scales_.data();
    const int8_t* x_qs = act_qs_.data();
    const int32_t* x_offsets = act_offsets_.data();
    const size_t batch_width = row_scratch_stride_;

    const auto body = [&](int begin, int end, int worker) {
        // One pass over each weight row serves every token in the batch, which is what makes
        // verifying γ+1 guesses cheaper than γ+1 separate steps.
        float* results = row_scratch_.data() + static_cast<size_t>(worker) * batch_width;
        for (int r = begin; r < end; ++r) {
            const uint32_t row = static_cast<uint32_t>(r);
            if (four_bit) {
                dot_q4_soa_multi(weight.row_scales(row),
                                 static_cast<const uint8_t*>(weight.row_qs(row)), x_scales, x_qs,
                                 x_offsets, nblocks, tokens, results);
            } else {
                dot_q8_soa_multi(weight.row_scales(row),
                                 static_cast<const int8_t*>(weight.row_qs(row)), x_scales, x_qs,
                                 x_offsets, nblocks, tokens, results);
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
    const int heads = static_cast<int>(c.num_attention_heads);
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    const int total = base + batch;  // every position now in the cache
    const int kv_heads = static_cast<int>(c.num_key_value_heads);

    // Positions are cut into fixed-size chunks so that there is work for every worker. The natural
    // unit is the key/value head, since its cached rows are read once and shared by the group of
    // query heads that use them -- but there are 8 of those against 6 workers, which cannot
    // balance: two workers take two heads and the makespan is two heads' work however many cores
    // idle. Measured, splitting positions as well is worth about 1.35x (plan §10.1).
    //
    // The size is a constant rather than something derived from the sequence length, and that is
    // the whole trick. The value accumulation sums chunk by chunk, so the chunk boundaries decide
    // the order the additions happen in; fixed boundaries mean a token at a given position sums the
    // same chunks in the same order whether it arrived in a one-token pass or a k-token one, which
    // is the invariant greedy speculative decoding rests on (§10.3).
    // 256 rather than something smaller because the split costs three extra barriers a layer, and
    // just past a boundary it buys nothing: decoding from position 128 with a chunk of 128 puts one
    // position in the second chunk and pays the barriers anyway, which measured as attention going
    // from 1.5 ms a token to 2.9. Everything up to a chunk's worth of context takes the single-region
    // path below instead, and that is the only threshold available -- the two paths agree exactly
    // when every position lies in the first chunk, and not otherwise.
    constexpr int kChunk = 256;
    const int chunks = (total + kChunk - 1) / kChunk;

    // Scores for one key/value head over a stretch of positions. Nothing is shared: this writes its
    // own stretch of its own heads' rows.
    const auto score_range = [&](int kv, int lo, int hi, float* row_scores) {
        const uint32_t first_head = static_cast<uint32_t>(kv) * group;
        for (int t = lo; t < hi; ++t) {
            const uint16_t* key = key_cache_.data() + cache_index(layer_index, kv, t);
            // Token j sits at position base + j, so it may attend to t only if t <= base + j.
            const int first_token = std::max(0, t - base);
            for (int j = first_token; j < batch; ++j) {
                // One pass over the cached row for the whole group of query heads that share it.
                // Their queries are adjacent, which is what makes one call enough.
                dot_f16_f32_group(key,
                                  qkv_.data() + static_cast<size_t>(j) * qkv_stride_ +
                                      first_head * head_dim,
                                  head_dim, static_cast<int>(group), head_dim, row_scores);
                for (uint32_t g = 0; g < group; ++g) {
                    scores_[score_index(j, first_head + g) + t] = row_scores[g] * scale;
                }
            }
        }
    };

    // The weighted sum of values over a stretch of positions, into `dest`, which is laid out as
    // batch tokens of q_dim each. Element for element the same operation as a scalar accumulate.
    const auto value_range = [&](int kv, int lo, int hi, float* row_scores, float* dest) {
        const uint32_t first_head = static_cast<uint32_t>(kv) * group;
        for (int t = lo; t < hi; ++t) {
            const uint16_t* value = value_cache_.data() + cache_index(layer_index, kv, t);
            const int first_token = std::max(0, t - base);
            for (int j = first_token; j < batch; ++j) {
                for (uint32_t g = 0; g < group; ++g) {
                    row_scores[g] = scores_[score_index(j, first_head + g) + t];
                }
                accumulate_scaled_f16_group(
                    dest + static_cast<size_t>(j) * q_dim + first_head * head_dim, head_dim, value,
                    row_scores, static_cast<int>(group), head_dim);
            }
        }
    };

    const auto softmax_rows = [&](int begin, int end) {
        for (int job = begin; job < end; ++job) {
            softmax_in_place(
                scores_.data() + score_index(job / heads, static_cast<uint32_t>(job % heads)),
                static_cast<uint32_t>(base + job / heads + 1));
        }
    };

    if (chunks == 1) {
        // Short context: one job per key/value head does the whole thing, and the pass costs one
        // barrier rather than four. Worth keeping separate -- on a small model the barriers are the
        // cost, and this is the shape every draft model runs in early in a sequence.
        pool_->run(kv_heads, [&](int begin, int end, int worker) {
            float* row_scores = kv_scratch_.data() + static_cast<size_t>(worker) * 2 * head_dim;
            for (int kv = begin; kv < end; ++kv) {
                const uint32_t first_head = static_cast<uint32_t>(kv) * group;
                score_range(kv, 0, total, row_scores);
                for (int j = 0; j < batch; ++j) {
                    for (uint32_t g = 0; g < group; ++g) {
                        softmax_in_place(scores_.data() + score_index(j, first_head + g),
                                         static_cast<uint32_t>(base + j + 1));
                    }
                    float* out = att_.data() + static_cast<size_t>(j) * q_dim + first_head * head_dim;
                    std::fill(out, out + group * head_dim, 0.0f);
                }
                value_range(kv, 0, total, row_scores, att_.data());
            }
        });
        return;
    }

    pool_->run(kv_heads * chunks, [&](int begin, int end, int worker) {
        float* row_scores = kv_scratch_.data() + static_cast<size_t>(worker) * 2 * head_dim;
        for (int job = begin; job < end; ++job) {
            const int lo = (job % chunks) * kChunk;
            score_range(job / chunks, lo, std::min(total, lo + kChunk), row_scores);
        }
    });

    // One softmax per (token, head), which needs every chunk's scores and so cannot start earlier.
    pool_->run(batch * heads, [&](int begin, int end, int) { softmax_rows(begin, end); });

    // Values, into one partial sum per chunk. A chunk's partial is a sum from zero over its own
    // positions, so which worker ran it does not enter the result -- only the chunk index does, and
    // that is fixed.
    const size_t partial_stride = static_cast<size_t>(batch) * q_dim;
    pool_->run(kv_heads * chunks, [&](int begin, int end, int worker) {
        float* row_scores = kv_scratch_.data() + static_cast<size_t>(worker) * 2 * head_dim;
        for (int job = begin; job < end; ++job) {
            const int kv = job / chunks;
            const int chunk = job % chunks;
            const int lo = chunk * kChunk;
            const uint32_t first_head = static_cast<uint32_t>(kv) * group;
            float* dest = partials_.data() + static_cast<size_t>(chunk) * partial_stride;

            // Every (chunk, token) slot this job owns starts at zero, including the ones no position
            // reaches: a token early in the batch does not see the later chunks, and adding their
            // zeros back in costs nothing and keeps the sum the same shape for all of them.
            for (int j = 0; j < batch; ++j) {
                float* slot = dest + static_cast<size_t>(j) * q_dim + first_head * head_dim;
                std::fill(slot, slot + group * head_dim, 0.0f);
            }
            value_range(kv, lo, std::min(total, lo + kChunk), row_scores, dest);
        }
    });

    // Add the chunks up, in chunk order, which is what makes this deterministic.
    pool_->run(batch * heads, [&](int begin, int end, int) {
        for (int job = begin; job < end; ++job) {
            const size_t offset = static_cast<size_t>(job / heads) * q_dim +
                                  static_cast<size_t>(job % heads) * head_dim;
            float* out = att_.data() + offset;
            std::copy(partials_.data() + offset, partials_.data() + offset + head_dim, out);
            for (int chunk = 1; chunk < chunks; ++chunk) {
                add_in_place(
                    out, partials_.data() + static_cast<size_t>(chunk) * partial_stride + offset,
                    head_dim);
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
