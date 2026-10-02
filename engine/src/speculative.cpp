#include "specdraft/speculative.hpp"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <stdexcept>

namespace specdraft {
namespace {

using Clock = std::chrono::steady_clock;

bool is_stop(const std::vector<int32_t>& stop, int32_t token) {
    return std::find(stop.begin(), stop.end(), token) != stop.end();
}

void check(const GenerateOptions& options) {
    if (options.max_new_tokens < 0) {
        throw std::invalid_argument("max_new_tokens must be >= 0");
    }
    if (options.gamma < 1) {
        throw std::invalid_argument("gamma must be >= 1");
    }
    if (options.confidence_threshold < 0.0f || options.confidence_threshold > 1.0f) {
        throw std::invalid_argument("confidence_threshold must be in [0, 1]");
    }
    options.sampling.validate();
}

// Tokens a model's cache has not seen yet: one in the usual case, two for the draft after a
// round where every guess was accepted.
std::vector<int32_t> pending_for(const Model& model, const std::vector<int32_t>& seq) {
    const int have = model.pos();
    if (have > static_cast<int>(seq.size())) {
        throw std::runtime_error("cache holds more tokens than the sequence");
    }
    return std::vector<int32_t>(seq.begin() + have, seq.end());
}

float max_probability(const float* probs, int n) {
    float best = 0.0f;
    for (int i = 0; i < n; ++i) {
        best = std::max(best, probs[i]);
    }
    return best;
}

}  // namespace

// ------------------------------------------------------------------------ model drafter

ModelDrafter::ModelDrafter(Model& model, const SamplingConfig& sampling, float confidence_threshold)
    : model_(model),
      sampling_(sampling),
      confidence_threshold_(confidence_threshold),
      vocab_(static_cast<int>(model.config().vocab_limit)),
      width_(static_cast<int>(model.logit_count())) {
    if (model.trimmed_vocabulary()) {
        trimmed_.assign(static_cast<size_t>(width_), 0.0f);
    }
}

double total_stage_seconds(const Model& model) {
    double total = 0.0;
    for (int stage = 0; stage < Model::kStageCount; ++stage) {
        total += model.stage_seconds(static_cast<Model::Stage>(stage));
    }
    return total;
}

double ModelDrafter::model_seconds() const {
    return total_stage_seconds(model_);
}

void ModelDrafter::reset() {
    model_.reset();
}

void ModelDrafter::prefill(const std::vector<int32_t>& prompt) {
    if (prompt.size() > 1) {
        model_.forward(prompt.data(), static_cast<int>(prompt.size()) - 1, nullptr, false);
    }
}

void ModelDrafter::rewind(int kept) {
    model_.set_pos(std::min(model_.pos(), kept));
}

int ModelDrafter::propose(const std::vector<int32_t>& seq, int gamma, int32_t* guesses, float* q,
                          int q_stride, DecodeStats& stats, Rng& rng) {
    const bool greedy = sampling_.greedy();
    const bool check_confidence = confidence_threshold_ > 0.0f;
    const SamplingConfig plain_softmax;  // temperature 1, no filtering
    const bool trimmed = !trimmed_.empty();

    // A trimmed draft writes one logit per kept token, so its output goes to a scratch buffer and
    // is scattered into the target's vocabulary afterwards. An untrimmed draft writes straight
    // into the row, which is the common case and copies nothing.
    const auto destination = [&](int j) {
        return trimmed ? trimmed_.data() : q + static_cast<size_t>(j) * q_stride;
    };

    // Feed whatever the draft's cache is missing, then walk forward one token at a time.
    const std::vector<int32_t> pending = pending_for(model_, seq);
    model_.forward(pending.data(), static_cast<int>(pending.size()), destination(0), false);
    ++stats.draft_forwards;

    int produced = 0;
    for (int j = 0; j < gamma; ++j) {
        float* row = q + static_cast<size_t>(j) * q_stride;
        float* values = trimmed ? trimmed_.data() : row;
        if (!trimmed && vocab_ < q_stride) {
            std::fill(row + vocab_, row + q_stride, 0.0f);
        }

        if (greedy && check_confidence) {
            // The threshold is about how sure the draft is, which needs probabilities even when
            // the guess itself only needs an argmax. Costs one softmax per step, and only when
            // the feature is switched on.
            warp_to_probs(values, width_, plain_softmax, scratch_);
        } else if (!greedy) {
            warp_to_probs(values, width_, sampling_, scratch_);
        }
        if (check_confidence && max_probability(values, width_) < confidence_threshold_) {
            break;
        }

        const int chosen = greedy ? argmax(values, width_) : sample_from_probs(values, width_, rng);
        guesses[j] = model_.token_for_logit(static_cast<uint32_t>(chosen));
        if (trimmed) {
            // The acceptance rule reads q over the target's vocabulary, so place each kept
            // token's probability at its own id and leave the dropped tokens at zero.
            std::fill(row, row + q_stride, 0.0f);
            if (!greedy) {
                for (int i = 0; i < width_; ++i) {
                    row[model_.token_for_logit(static_cast<uint32_t>(i))] = trimmed_[static_cast<size_t>(i)];
                }
            }
        }
        ++produced;

        if (j + 1 < gamma) {
            model_.forward(&guesses[j], 1, destination(j + 1), false);
            ++stats.draft_forwards;
        }
    }
    stats.proposed += produced;
    return produced;
}

// ----------------------------------------------------------------- prompt lookup drafter

PromptLookupDrafter::PromptLookupDrafter(int vocab, int max_ngram, int min_ngram)
    : vocab_(vocab), max_ngram_(std::max(1, max_ngram)), min_ngram_(std::max(1, min_ngram)) {
    if (min_ngram_ > max_ngram_) {
        throw std::invalid_argument("min_ngram must not exceed max_ngram");
    }
}

int PromptLookupDrafter::propose(const std::vector<int32_t>& seq, int gamma, int32_t* guesses,
                                 float* q, int q_stride, DecodeStats& stats, Rng&) {
    const int length = static_cast<int>(seq.size());
    for (int size = std::min(max_ngram_, length - 1); size >= min_ngram_; --size) {
        const int32_t* suffix = seq.data() + (length - size);
        // Most recent match first: nearby text is the better predictor.
        for (int start = length - size - 1; start >= 0; --start) {
            if (!std::equal(suffix, suffix + size, seq.data() + start)) {
                continue;
            }
            const int available = length - (start + size);
            const int take = std::min(gamma, available);
            if (take <= 0) {
                continue;
            }
            for (int j = 0; j < take; ++j) {
                guesses[j] = seq[static_cast<size_t>(start + size + j)];
                // A copied token carries no distribution, so treat it as drawn from a point
                // mass. The acceptance rule then takes it with probability p(token) and
                // otherwise resamples from p with that token removed, which still leaves the
                // output exactly p.
                float* row = q + static_cast<size_t>(j) * q_stride;
                std::fill(row, row + vocab_, 0.0f);
                row[guesses[j]] = 1.0f;
            }
            stats.proposed += take;
            return take;
        }
    }
    return 0;  // nothing to copy: the round becomes an ordinary single-token step
}

// ------------------------------------------------------------------------------- loops

std::vector<int32_t> generate_plain(Model& model, const std::vector<int32_t>& prompt,
                                    const GenerateOptions& options, DecodeStats* stats) {
    check(options);
    if (prompt.empty()) {
        throw std::invalid_argument("prompt must not be empty");
    }
    const int vocab = static_cast<int>(model.config().vocab_limit);
    const bool greedy = options.sampling.greedy();

    std::vector<int32_t> seq = prompt;
    std::vector<float> logits(static_cast<size_t>(vocab));
    std::vector<int> scratch;
    Rng rng(options.seed);
    DecodeStats local;

    model.reset();
    const Clock::time_point started = Clock::now();
    if (seq.size() > 1) {  // prefill everything but the newest token
        model.forward(seq.data(), static_cast<int>(seq.size()) - 1, nullptr, false);
    }

    while (local.emitted < options.max_new_tokens) {
        const std::vector<int32_t> pending = pending_for(model, seq);
        model.forward(pending.data(), static_cast<int>(pending.size()), logits.data(), false);
        ++local.target_forwards;

        int32_t token;
        if (greedy) {
            token = argmax(logits.data(), vocab);
        } else {
            warp_to_probs(logits.data(), vocab, options.sampling, scratch);
            token = sample_from_probs(logits.data(), vocab, rng);
        }
        seq.push_back(token);
        ++local.emitted;
        if (is_stop(options.stop, token)) {
            break;
        }
    }
    local.seconds = std::chrono::duration<double>(Clock::now() - started).count();

    if (stats != nullptr) {
        *stats = local;
    }
    return std::vector<int32_t>(seq.begin() + static_cast<long>(prompt.size()), seq.end());
}

std::vector<int32_t> generate_with_drafter(Model& target, Drafter& drafter,
                                           const std::vector<int32_t>& prompt,
                                           const GenerateOptions& options, DecodeStats* stats) {
    check(options);
    if (prompt.empty()) {
        throw std::invalid_argument("prompt must not be empty");
    }
    if (target.max_batch() < options.gamma + 1) {
        throw std::invalid_argument("the target's max_batch is smaller than gamma + 1");
    }
    const int vocab = static_cast<int>(target.config().vocab_limit);
    const int gamma = options.gamma;
    const bool greedy = options.sampling.greedy();

    std::vector<int32_t> seq = prompt;
    std::vector<float> target_rows(static_cast<size_t>(gamma + 1) * vocab);
    std::vector<float> draft_rows(static_cast<size_t>(gamma) * vocab, 0.0f);
    std::vector<int32_t> guesses(static_cast<size_t>(gamma));
    std::vector<int32_t> verify(static_cast<size_t>(gamma) + 1);
    std::vector<int> scratch;
    std::vector<float> residual;
    Rng rng(options.seed);
    DecodeStats local;
    local.accepted_lengths.assign(static_cast<size_t>(gamma) + 1, 0);

    target.reset();
    drafter.reset();
    const Clock::time_point started = Clock::now();
    if (seq.size() > 1) {  // after this, every round has the same shapes
        target.forward(seq.data(), static_cast<int>(seq.size()) - 1, nullptr, false);
        drafter.prefill(seq);
        local.prefill_seconds = std::chrono::duration<double>(Clock::now() - started).count();
        // Read the models' own timers here so the caller can tell the prefill's compute apart from
        // the rounds'. Zero unless the caller enabled timing, which is exactly when it matters.
        local.prefill_model_seconds = total_stage_seconds(target) + drafter.model_seconds();
    }

    bool finished = false;
    while (local.emitted < options.max_new_tokens && !finished) {
        const int proposed =
            drafter.propose(seq, gamma, guesses.data(), draft_rows.data(), vocab, local, rng);
        if (proposed < 0 || proposed > gamma) {
            throw std::runtime_error("a drafter proposed an impossible number of tokens");
        }

        // One target pass over the newest real token plus every guess. With no guesses this is
        // an ordinary decoding step, which is what a drafter with nothing to say should cost.
        verify[0] = seq.back();
        std::copy(guesses.begin(), guesses.begin() + proposed, verify.begin() + 1);
        target.forward(verify.data(), proposed + 1, target_rows.data(), true);
        ++local.target_forwards;
        if (!greedy) {
            for (int j = 0; j <= proposed; ++j) {
                warp_to_probs(target_rows.data() + static_cast<size_t>(j) * vocab, vocab,
                              options.sampling, scratch);
            }
        }

        const Verdict verdict =
            accept_or_resample(target_rows.data(), vocab, draft_rows.data(), vocab, guesses.data(),
                               proposed, vocab, greedy, rng, residual);
        ++local.rounds;
        local.accepted += verdict.accepted;
        local.accepted_lengths[static_cast<size_t>(verdict.accepted)] += 1;
        if (verdict.accepted < proposed) {
            ++local.rejections;
        }

        for (int j = 0; j <= verdict.accepted; ++j) {
            if (local.emitted >= options.max_new_tokens) {
                break;
            }
            const int32_t token =
                j < verdict.accepted ? guesses[static_cast<size_t>(j)] : verdict.next_token;
            seq.push_back(token);
            ++local.emitted;
            if (is_stop(options.stop, token)) {
                finished = true;
                break;
            }
        }

        // Both caches go back to holding everything but the newest token.
        const int keep = static_cast<int>(seq.size()) - 1;
        target.set_pos(keep);
        drafter.rewind(keep);
    }
    local.seconds = std::chrono::duration<double>(Clock::now() - started).count();

    if (stats != nullptr) {
        *stats = local;
    }
    return std::vector<int32_t>(seq.begin() + static_cast<long>(prompt.size()), seq.end());
}

std::vector<int32_t> generate_speculative(Model& target, Model& draft,
                                          const std::vector<int32_t>& prompt,
                                          const GenerateOptions& options, DecodeStats* stats) {
    if (draft.config().vocab_limit > target.config().vocab_limit) {
        throw std::invalid_argument("the draft's vocabulary is larger than the target's");
    }
    if (draft.logit_count() > target.config().vocab_limit) {
        throw std::invalid_argument("the draft scores more tokens than the target has");
    }
    ModelDrafter drafter(draft, options.sampling, options.confidence_threshold);
    return generate_with_drafter(target, drafter, prompt, options, stats);
}

std::vector<int32_t> generate_prompt_lookup(Model& target, const std::vector<int32_t>& prompt,
                                            const GenerateOptions& options, int max_ngram,
                                            DecodeStats* stats) {
    PromptLookupDrafter drafter(static_cast<int>(target.config().vocab_limit), max_ngram);
    return generate_with_drafter(target, drafter, prompt, options, stats);
}

}  // namespace specdraft
