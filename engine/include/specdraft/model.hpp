#pragma once

#include <cstdint>
#include <vector>

#include "specdraft/model_file.hpp"
#include "specdraft/quant.hpp"

// The forward pass. Milestone 2 of the plan: correctness first, so this is plain scalar
// code with no threads, using the scalar or AVX-VNNI dot products from quant.hpp.
//
// Tokens are processed one at a time, even when several are passed at once. That makes a
// k-token pass identical to k single-token passes by construction, which is the property
// bit-exact greedy speculative decoding needs. Milestone 3 and 4 make the k-token path
// fast by unpacking each weight block once and reusing it, while keeping that per-token
// summation order.

namespace specdraft {

class Model {
public:
    explicit Model(ModelFile file, int max_positions = 2048);

    const ModelConfig& config() const { return file_.config(); }
    int max_positions() const { return max_positions_; }
    int pos() const { return pos_; }

    // Rolling back rejected draft tokens: attention only reads up to the counter, and
    // stale entries beyond it get overwritten by the next round.
    void set_pos(int position);
    void reset() { pos_ = 0; }

    // Run k tokens starting at the current position, then advance it by k.
    //   logits_out: vocab_limit floats per scored token, k of them when all_logits is set,
    //               otherwise just the last token's. May be null when no logits are wanted.
    //   capture_out: when not null, receives num_hidden_layers * k * hidden_size floats,
    //               layer-major, for comparing against the PyTorch twin layer by layer.
    void forward(const int32_t* tokens, int k, float* logits_out, bool all_logits,
                 float* capture_out = nullptr);

private:
    struct Layer {
        const float* attn_norm = nullptr;
        const float* q_norm = nullptr;
        const float* k_norm = nullptr;
        const float* ffn_norm = nullptr;
        Tensor qkv, attn_out, gate_up, ffn_down;
    };

    void forward_one(int32_t token, int position, float* logits_out, float* capture_out, int k,
                     int token_index);
    void attention(int layer_index, int position);
    void matvec(const Tensor& weight, const float* x, uint32_t n_in, float* out, uint32_t n_out);
    void embed(int32_t token, float* out) const;
    size_t cache_index(int layer, uint32_t kv_head, int position) const;

    ModelFile file_;
    int max_positions_;
    int pos_ = 0;

    std::vector<Layer> layers_;
    Tensor embedding_;  // token_embd, also the output layer when weights are tied
    Tensor output_;
    const float* output_norm_ = nullptr;

    std::vector<float> x_, xb_, xb2_, qkv_, att_, scores_, mlp_;
    std::vector<BlockA8> activations_;
    std::vector<uint16_t> key_cache_, value_cache_;  // [layer][kv_head][position][head_dim], fp16
};

}  // namespace specdraft
