#pragma once

#include <cstdint>
#include <memory>
#include <string>
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
//
// Where the guesses come from is a separate question from how they are checked, so the loop
// takes a Drafter. A small model is one; copying from earlier in the prompt is another.

namespace specdraft {

struct DecodeStats {
    int emitted = 0;
    int rounds = 0;
    int proposed = 0;    // guesses offered, which varies when the drafter stops early
    int accepted = 0;    // guesses that survived
    int rejections = 0;  // rounds that ended in a rejection
    int target_forwards = 0;  // verification passes, prefill excluded
    int draft_forwards = 0;
    double seconds = 0.0;
    // The prompt pass, separated out because the per-round overhead term o is measured as whatever
    // wall time the models did not spend computing, divided by rounds -- and a prefill is neither a
    // round nor free. Charging its share to the rounds inflated o by most of its value.
    double prefill_seconds = 0.0;
    double prefill_model_seconds = 0.0;  // what the models' own timers attributed to that pass
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
    double tokens_per_second() const { return seconds > 0.0 ? emitted / seconds : 0.0; }
    // Wall time spent in the round loop, which is what o should be derived from.
    double rounds_seconds() const { return seconds - prefill_seconds; }
};

struct GenerateOptions {
    int max_new_tokens = 64;
    int gamma = 4;
    SamplingConfig sampling;
    std::vector<int32_t> stop;
    uint64_t seed = 0;
    // Stop drafting once the draft's own top probability falls below this. Every extra guess
    // costs a draft step and makes the verification pass wider, so giving up on a position the
    // draft is unsure about can pay (plan §8.3). 0 disables it.
    float confidence_threshold = 0.0f;
};

// Where guesses come from. A drafter fills `guesses` and the matching rows of `q`, which hold
// the distribution each guess was actually drawn from, over the target's vocabulary.
class Drafter {
public:
    virtual ~Drafter() = default;
    virtual const char* name() const = 0;
    // Returns how many guesses were produced, which may be fewer than gamma, or zero.
    virtual int propose(const std::vector<int32_t>& seq, int gamma, int32_t* guesses, float* q,
                        int q_stride, DecodeStats& stats, Rng& rng) = 0;
    // Called after each round so a drafter holding a cache can roll it back.
    virtual void rewind(int kept) = 0;
    virtual void reset() = 0;
    virtual void prefill(const std::vector<int32_t>& prompt) = 0;
    // What this drafter's model has spent computing so far, by its own stage timers. Zero for a
    // drafter without a model, and zero unless timing is on; it exists so the per-round overhead can
    // be measured with the prompt pass taken out of both sides.
    virtual double model_seconds() const { return 0.0; }
};

// Sum of a model's stage timers: the compute it accounts for, against which anything else is overhead.
double total_stage_seconds(const Model& model);

// Guesses from a small model: the usual arrangement.
class ModelDrafter : public Drafter {
public:
    ModelDrafter(Model& model, const SamplingConfig& sampling, float confidence_threshold);
    const char* name() const override { return "model"; }
    int propose(const std::vector<int32_t>& seq, int gamma, int32_t* guesses, float* q,
                int q_stride, DecodeStats& stats, Rng& rng) override;
    void rewind(int kept) override;
    void reset() override;
    void prefill(const std::vector<int32_t>& prompt) override;
    double model_seconds() const override;

private:
    Model& model_;
    SamplingConfig sampling_;
    float confidence_threshold_;
    int vocab_;   // the target's vocabulary, which q is expressed over
    int width_;   // logits the draft produces: fewer when its output layer was trimmed
    std::vector<float> trimmed_;  // scratch for a trimmed draft's own logits
    std::vector<int> scratch_;
};

// Guesses copied from earlier in the text: find the most recent place the last few tokens
// appeared and copy what followed. Costs no model work at all, so c is effectively zero, which
// makes it hard to beat where text repeats, such as summarization (plan §10.4).
class PromptLookupDrafter : public Drafter {
public:
    PromptLookupDrafter(int vocab, int max_ngram = 3, int min_ngram = 1);
    const char* name() const override { return "prompt_lookup"; }
    int propose(const std::vector<int32_t>& seq, int gamma, int32_t* guesses, float* q,
                int q_stride, DecodeStats& stats, Rng& rng) override;
    void rewind(int) override {}
    void reset() override {}
    void prefill(const std::vector<int32_t>&) override {}

private:
    int vocab_;
    int max_ngram_;
    int min_ngram_;
};

// Ordinary token-at-a-time decoding: the baseline, and what greedy speculative decoding has to
// reproduce exactly.
std::vector<int32_t> generate_plain(Model& model, const std::vector<int32_t>& prompt,
                                    const GenerateOptions& options, DecodeStats* stats);

// Draft, verify in one target pass, repeat. The output has exactly the distribution plain
// decoding from `target` would have, whatever the drafter does.
std::vector<int32_t> generate_with_drafter(Model& target, Drafter& drafter,
                                           const std::vector<int32_t>& prompt,
                                           const GenerateOptions& options, DecodeStats* stats);

// Convenience wrapper for the usual case: a draft model.
//
// A draft with a smaller vocabulary than the target is allowed: its missing tokens get
// probability zero, which the acceptance rule handles without any change, since it only ever
// reads q at tokens the draft actually proposed.
std::vector<int32_t> generate_speculative(Model& target, Model& draft,
                                          const std::vector<int32_t>& prompt,
                                          const GenerateOptions& options, DecodeStats* stats);

std::vector<int32_t> generate_prompt_lookup(Model& target, const std::vector<int32_t>& prompt,
                                            const GenerateOptions& options, int max_ngram,
                                            DecodeStats* stats);

}  // namespace specdraft
