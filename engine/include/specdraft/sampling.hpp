#pragma once

#include <cstdint>
#include <vector>

// Sampling and the speculative acceptance rule, in C++.
//
// python/specdraft/sampling.py is the oracle this is tested against: the emitted token must
// follow the target's distribution exactly, which is the whole point of the scheme. Two
// conditions make that true, and both are the caller's responsibility:
//   * p is the *warped* target distribution, after temperature, top-k and top-p;
//   * q is *exactly* the distribution the draft sampled from.

namespace specdraft {

struct SamplingConfig {
    float temperature = 1.0f;  // 0 means greedy
    int top_k = 0;             // 0 means no limit
    float top_p = 1.0f;        // 1 means no limit

    bool greedy() const { return temperature == 0.0f; }
    void validate() const;
};

// xoshiro256++. Small, fast, and reproducible from a seed, so a decode can be replayed.
class Rng {
public:
    explicit Rng(uint64_t seed);
    uint64_t next_u64();
    float next_float();  // in [0, 1)

private:
    uint64_t state_[4];
};

// First index of the largest value. Vectorized, because greedy decoding spends real time
// here: two scans of a 152k-element row per guess.
int argmax(const float* values, int n);
// The scalar scan it replaced, kept so the two can be compared in one process -- the only
// kind of comparison this machine supports -- and so a test can pin them to the same answer.
int argmax_scalar(const float* values, int n);

// Logits in, probabilities out, in the same buffer. `scratch` is reused across calls so the
// nucleus search does not allocate per token.
void warp_to_probs(float* values, int n, const SamplingConfig& config, std::vector<int>& scratch);

int sample_from_probs(const float* probs, int n, Rng& rng);

struct Verdict {
    int accepted = 0;   // how many guesses survived
    int next_token = 0;  // the target's token: a correction, or a bonus if all were accepted
};

// Verify gamma guesses at once.
//   p: gamma + 1 rows of `vocab` values, `stride` apart. Warped probabilities, or raw logits
//      when greedy.
//   q: gamma rows, the distributions the draft actually sampled from (ignored when greedy).
Verdict accept_or_resample(const float* p, int p_stride, const float* q, int q_stride,
                           const int32_t* guesses, int gamma, int vocab, bool greedy, Rng& rng,
                           std::vector<float>& scratch);

}  // namespace specdraft
