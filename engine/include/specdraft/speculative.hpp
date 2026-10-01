#pragma once

#include <cstdint>
#include <vector>

#include "specdraft/model.hpp"
#include "specdraft/sampling.hpp"

// The decoding loops, in C++ so that Python overhead never lands in a timing.
//
// The cache rule is the one from plan §10.2: `seq` is the whole token sequence, each model's
// cache holds some prefix of it, and before a forward pass a model is fed exactly the tokens
// its cache is missing. After verification both caches are rewound to "everything but the
// newest token", which is one token for the target and one or two for the draft, since after a
// round where every guess was accepted the draft never saw its own last guess or the bonus
// token.

namespace specdraft {

struct DecodeStats {
    int emitted = 0;
    int rounds = 0;
    int accepted = 0;    // accepted draft tokens
    int rejections = 0;  // rounds that ended in a rejection
    int target_forwards = 0;  // verification passes, prefill excluded
    int draft_forwards = 0;
    double seconds = 0.0;
    std::vector<int> accepted_lengths;

    // τ: the number this project is trying to raise.
    double tokens_per_target_forward() const {
        return target_forwards > 0 ? static_cast<double>(emitted) / target_forwards : 0.0;
    }
    // Each round shows accepts until the first rejection, so the maximum-likelihood estimate
    // for a truncated geometric is accepts / (accepts + rejections).
    double alpha() const {
        const int trials = accepted + rejections;
        return trials > 0 ? static_cast<double>(accepted) / trials : 0.0;
    }
    double tokens_per_second() const {
        return seconds > 0.0 ? emitted / seconds : 0.0;
    }
};

struct GenerateOptions {
    int max_new_tokens = 64;
    int gamma = 4;
    SamplingConfig sampling;
    std::vector<int32_t> stop;
    uint64_t seed = 0;
};

// Ordinary token-at-a-time decoding: the baseline, and what greedy speculative decoding has to
// reproduce exactly.
std::vector<int32_t> generate_plain(Model& model, const std::vector<int32_t>& prompt,
                                    const GenerateOptions& options, DecodeStats* stats);

// Draft gamma tokens, verify them in one target pass, repeat. The output has exactly the
// distribution plain decoding from `target` would have.
//
// A draft with a smaller vocabulary than the target is allowed: its missing tokens get
// probability zero, which the acceptance rule handles without any change, since it only ever
// reads q at tokens the draft actually proposed.
std::vector<int32_t> generate_speculative(Model& target, Model& draft,
                                          const std::vector<int32_t>& prompt,
                                          const GenerateOptions& options, DecodeStats* stats);

}  // namespace specdraft
