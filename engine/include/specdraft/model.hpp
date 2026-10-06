#pragma once

#include <array>
#include <cstdint>
#include <memory>
#include <vector>

#include "specdraft/model_file.hpp"
#include "specdraft/quant.hpp"
#include "specdraft/threadpool.hpp"

// The forward pass.
//
// Tokens are processed one at a time, even when several are passed at once, so a k-token
// pass is identical to k single-token passes by construction. That is the property
// bit-exact greedy speculative decoding needs. Work is split across threads by output row,
// and each row is computed start to finish by one worker, so the result does not depend on
// the thread count or the schedule either.

namespace specdraft {

struct EngineOptions {
    int max_positions = 2048;
    int threads = 0;  // 0 means as many as the core selection offers
    CoreSelection cores = CoreSelection::performance;
    // Hand out row chunks from a shared counter instead of a fixed split. On a hybrid CPU a
    // fixed split makes the performance cores wait for the efficiency cores.
    bool dynamic_schedule = false;
    // How many tokens may share one pass over the weights. Verification needs γ+1 and prompt
    // processing wants as many as fit; longer sequences are split into chunks of this size.
    int max_batch = 16;
};

class Model {
public:
    explicit Model(ModelFile file, EngineOptions options = {});
    ~Model();

    const ModelConfig& config() const { return file_.config(); }
    int max_positions() const { return options_.max_positions; }
    int threads() const;
    const char* core_selection() const { return core_selection_name(options_.cores); }
    bool dynamic_schedule() const { return options_.dynamic_schedule; }
    int max_batch() const { return options_.max_batch; }
    int pos() const { return pos_; }

    // How many logits one scored token produces: the whole vocabulary, or just the tokens a
    // trimmed output layer kept.
    uint32_t logit_count() const { return output_map_ == nullptr ? config().vocab_limit : config().output_vocab; }
    // Which token a logit refers to. The identity unless the output layer was trimmed.
    int32_t token_for_logit(uint32_t index) const {
        return output_map_ == nullptr ? static_cast<int32_t>(index) : output_map_[index];
    }
    bool trimmed_vocabulary() const { return output_map_ != nullptr; }

    // Rolling back rejected draft tokens: attention only reads up to the counter, and stale
    // entries beyond it get overwritten by the next round.
    void set_pos(int position);
    void reset() { pos_ = 0; }

    // Run k tokens starting at the current position, then advance it by k.
    //   logits_out: vocab_limit floats per scored token, k of them when all_logits is set,
    //               otherwise just the last token's. May be null.
    //   capture_out: when not null, receives num_hidden_layers * k * hidden_size floats,
    //               layer-major, for comparing against the PyTorch twin layer by layer.
    void forward(const int32_t* tokens, int k, float* logits_out, bool all_logits,
                 float* capture_out = nullptr);

    // Per-stage timing, off by default. Switching it on costs one clock read per stage.
    enum Stage { kEmbed, kQkv, kAttention, kAttnOut, kGateUp, kFfnDown, kOutput, kStageCount };
    static const char* stage_name(Stage stage);
    void set_timing(bool enabled);
    void reset_timings();
    double stage_seconds(Stage stage) const { return stage_seconds_[stage]; }
    uint64_t timed_tokens() const { return timed_tokens_; }

    // Bytes the weights contribute to one token, for comparing against the bandwidth
    // ceiling. The KV cache adds more as the context grows; see kv_bytes_per_token().
    size_t weight_bytes_per_token() const;
    size_t kv_bytes_per_token() const;

private:
    struct Layer {
        const float* attn_norm = nullptr;
        const float* q_norm = nullptr;
        const float* k_norm = nullptr;
        const float* ffn_norm = nullptr;
        Tensor qkv, attn_out, gate_up, ffn_down;
    };

    // One pass over the weights for `batch` tokens starting at position `base`.
    void forward_batch(const int32_t* tokens, int batch, int base, float* logits_out,
                       bool all_logits, int single_token, float* capture_out, int capture_offset,
                       int capture_total);
    void attention(int layer_index, int base, int batch);
    // `batch` activation vectors against one weight matrix. Strides are in floats, so a caller
    // can feed vectors that sit inside a wider buffer.
    void matmul(const Tensor& weight, const float* in, uint32_t in_stride, uint32_t n_in,
                float* out, uint32_t out_stride, uint32_t n_out, int batch);
    void embed(int32_t token, float* out) const;
    size_t cache_index(int layer, uint32_t kv_head, int position) const;
    size_t score_index(int token, uint32_t head) const;

    ModelFile file_;
    EngineOptions options_;
    std::unique_ptr<ThreadPool> pool_;
    int pos_ = 0;

    std::vector<Layer> layers_;
    Tensor embedding_;  // token_embd, also the output layer when weights are tied
    Tensor output_;
    const int32_t* output_map_ = nullptr;  // null unless the output layer was trimmed
    const float* output_norm_ = nullptr;

    // All sized for max_batch tokens; strides are the per-token widths below.
    std::vector<float> x_, xb_, xb2_, qkv_, att_, scores_, mlp_, kv_scratch_, row_scratch_;
    // Attention's per-chunk partial sums; see Model::attention for why the chunking is fixed.
    std::vector<float> partials_;
    uint32_t qkv_stride_ = 0, mlp_stride_ = 0, blocks_stride_ = 0;
    // row_scratch_ holds one worker's results for the row it is on, and its stride is padded to a
    // whole cache line: workers writing inside one line bounce it between cores on every row.
    uint32_t row_scratch_stride_ = 0;
    // Quantized activations in the split layout the kernels read: one scale a block, the bytes, and
    // one offset a block carrying the weight format's zero point times that block's activation sum.
    std::vector<float> act_scales_;
    std::vector<int8_t> act_qs_;
    std::vector<int32_t> act_offsets_;
    std::vector<uint16_t> key_cache_, value_cache_;  // [layer][kv_head][position][head_dim], fp16

    bool timing_ = false;
    std::array<double, kStageCount> stage_seconds_{};
    uint64_t timed_tokens_ = 0;
};

}  // namespace specdraft
