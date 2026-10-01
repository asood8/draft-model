#include "specdraft/sampling.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>

namespace specdraft {
namespace {

uint64_t rotl(uint64_t x, int k) {
    return (x << k) | (x >> (64 - k));
}

// SplitMix64, used to spread a single seed over xoshiro's four words.
uint64_t splitmix64(uint64_t& x) {
    x += 0x9E3779B97F4A7C15ULL;
    uint64_t z = x;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

void softmax_in_place(float* values, int n) {
    float largest = values[0];
    for (int i = 1; i < n; ++i) {
        largest = std::max(largest, values[i]);
    }
    float total = 0.0f;
    for (int i = 0; i < n; ++i) {
        values[i] = std::exp(values[i] - largest);
        total += values[i];
    }
    const float inverse = 1.0f / total;
    for (int i = 0; i < n; ++i) {
        values[i] *= inverse;
    }
}

// The smallest set of most likely tokens whose mass reaches top_p, found without sorting the
// whole vocabulary: take the top 64, then 128, 256 ... until their mass is enough. Typical
// distributions need one or two rounds over a 152k vocabulary, where a full sort would cost
// milliseconds per token and show up as per-round overhead in the speedup model.
void keep_nucleus(float* probs, int n, float top_p, std::vector<int>& scratch) {
    scratch.resize(static_cast<size_t>(n));
    std::iota(scratch.begin(), scratch.end(), 0);
    const auto by_probability = [probs](int a, int b) { return probs[a] > probs[b]; };

    int kept = std::min(n, 64);
    while (true) {
        std::nth_element(scratch.begin(), scratch.begin() + kept, scratch.end(), by_probability);
        float mass = 0.0f;
        for (int i = 0; i < kept; ++i) {
            mass += probs[scratch[i]];
        }
        if (mass >= top_p || kept >= n) {
            break;
        }
        kept = std::min(n, kept * 2);
    }

    // Within the candidates, keep the shortest prefix that reaches top_p. The most likely
    // token is always kept, however peaked the distribution is.
    std::sort(scratch.begin(), scratch.begin() + kept, by_probability);
    float mass_before = 0.0f;
    int survivors = 0;
    for (int i = 0; i < kept; ++i) {
        if (i > 0 && mass_before >= top_p) {
            break;
        }
        mass_before += probs[scratch[i]];
        ++survivors;
    }

    // Zero everything outside the nucleus, then renormalize.
    std::vector<int>& order = scratch;
    float total = 0.0f;
    for (int i = 0; i < survivors; ++i) {
        total += probs[order[i]];
    }
    std::vector<float> keep_values(static_cast<size_t>(survivors));
    for (int i = 0; i < survivors; ++i) {
        keep_values[static_cast<size_t>(i)] = probs[order[i]] / total;
    }
    std::fill(probs, probs + n, 0.0f);
    for (int i = 0; i < survivors; ++i) {
        probs[order[i]] = keep_values[static_cast<size_t>(i)];
    }
}

}  // namespace

void SamplingConfig::validate() const {
    if (temperature < 0.0f) {
        throw std::invalid_argument("temperature must be >= 0");
    }
    if (top_k < 0) {
        throw std::invalid_argument("top_k must be >= 0");
    }
    if (!(top_p > 0.0f && top_p <= 1.0f)) {
        throw std::invalid_argument("top_p must be in (0, 1]");
    }
}

Rng::Rng(uint64_t seed) {
    uint64_t spread = seed;
    for (uint64_t& word : state_) {
        word = splitmix64(spread);
    }
}

uint64_t Rng::next_u64() {
    const uint64_t result = rotl(state_[0] + state_[3], 23) + state_[0];
    const uint64_t t = state_[1] << 17;
    state_[2] ^= state_[0];
    state_[3] ^= state_[1];
    state_[1] ^= state_[2];
    state_[0] ^= state_[3];
    state_[2] ^= t;
    state_[3] = rotl(state_[3], 45);
    return result;
}

float Rng::next_float() {
    // 24 bits is all a float can hold, so take the top 24 of a 64-bit draw.
    return static_cast<float>(next_u64() >> 40) * (1.0f / 16777216.0f);
}

int argmax(const float* values, int n) {
    int best = 0;
    for (int i = 1; i < n; ++i) {
        if (values[i] > values[best]) {
            best = i;
        }
    }
    return best;
}

void warp_to_probs(float* values, int n, const SamplingConfig& config, std::vector<int>& scratch) {
    config.validate();
    if (config.greedy()) {
        throw std::invalid_argument("greedy decoding has no distribution; compare argmax instead");
    }

    const float inverse_temperature = 1.0f / config.temperature;
    for (int i = 0; i < n; ++i) {
        values[i] *= inverse_temperature;
    }

    if (config.top_k > 0 && config.top_k < n) {
        scratch.resize(static_cast<size_t>(n));
        std::iota(scratch.begin(), scratch.end(), 0);
        std::nth_element(scratch.begin(), scratch.begin() + config.top_k, scratch.end(),
                         [values](int a, int b) { return values[a] > values[b]; });
        const float cutoff = values[scratch[static_cast<size_t>(config.top_k)]];
        for (int i = 0; i < n; ++i) {
            if (values[i] <= cutoff) {
                values[i] = -std::numeric_limits<float>::infinity();
            }
        }
        // Ties at the cutoff can leave fewer than top_k survivors; that matches taking the
        // strictly-better tokens, which is what a threshold comparison means.
    }

    softmax_in_place(values, n);

    if (config.top_p < 1.0f) {
        keep_nucleus(values, n, config.top_p, scratch);
    }
}

int sample_from_probs(const float* probs, int n, Rng& rng) {
    const float target = rng.next_float();
    float cumulative = 0.0f;
    for (int i = 0; i < n; ++i) {
        cumulative += probs[i];
        if (target < cumulative) {
            return i;
        }
    }
    // Floating-point round-off can leave the draw just past the end; fall back to the last
    // token that carries any mass.
    for (int i = n - 1; i >= 0; --i) {
        if (probs[i] > 0.0f) {
            return i;
        }
    }
    return 0;
}

Verdict accept_or_resample(const float* p, int p_stride, const float* q, int q_stride,
                           const int32_t* guesses, int gamma, int vocab, bool greedy, Rng& rng,
                           std::vector<float>& scratch) {
    Verdict verdict;
    int accepted = 0;
    for (; accepted < gamma; ++accepted) {
        const int32_t guess = guesses[accepted];
        if (guess < 0 || guess >= vocab) {
            throw std::runtime_error("draft proposed a token outside the vocabulary");
        }
        const float* p_row = p + static_cast<size_t>(accepted) * p_stride;
        if (greedy) {
            if (guess != argmax(p_row, vocab)) {
                break;
            }
        } else {
            const float* q_row = q + static_cast<size_t>(accepted) * q_stride;
            // u * q < p is u < min(1, p/q) without a division, so q = 0 cannot blow up.
            if (!(rng.next_float() * q_row[guess] < p_row[guess])) {
                break;
            }
        }
    }
    verdict.accepted = accepted;

    const float* p_row = p + static_cast<size_t>(accepted) * p_stride;
    if (greedy) {
        verdict.next_token = argmax(p_row, vocab);
        return verdict;
    }
    if (accepted == gamma) {  // every guess accepted: take the bonus token
        verdict.next_token = sample_from_probs(p_row, vocab, rng);
        return verdict;
    }

    // A rejection at this position: resample from the renormalized max(0, p - q). Some mass is
    // always left, because a rejection needs p(x) < q(x) somewhere and both sum to one.
    const float* q_row = q + static_cast<size_t>(accepted) * q_stride;
    scratch.resize(static_cast<size_t>(vocab));
    float total = 0.0f;
    for (int i = 0; i < vocab; ++i) {
        const float residual = p_row[i] - q_row[i];
        scratch[static_cast<size_t>(i)] = residual > 0.0f ? residual : 0.0f;
        total += scratch[static_cast<size_t>(i)];
    }
    if (total <= 0.0f) {  // only reachable through float round-off
        verdict.next_token = sample_from_probs(p_row, vocab, rng);
        return verdict;
    }
    const float inverse = 1.0f / total;
    for (int i = 0; i < vocab; ++i) {
        scratch[static_cast<size_t>(i)] *= inverse;
    }
    verdict.next_token = sample_from_probs(scratch.data(), vocab, rng);
    return verdict;
}

}  // namespace specdraft
