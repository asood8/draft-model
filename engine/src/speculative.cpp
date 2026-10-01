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

}  // namespace

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

std::vector<int32_t> generate_speculative(Model& target, Model& draft,
                                          const std::vector<int32_t>& prompt,
                                          const GenerateOptions& options, DecodeStats* stats) {
    check(options);
    if (prompt.empty()) {
        throw std::invalid_argument("prompt must not be empty");
    }
    const int vocab = static_cast<int>(target.config().vocab_limit);
    const int draft_vocab = static_cast<int>(draft.config().vocab_limit);
    if (draft_vocab > vocab) {
        throw std::invalid_argument("the draft's vocabulary is larger than the target's");
    }
    if (target.max_batch() < options.gamma + 1) {
        throw std::invalid_argument("the target's max_batch is smaller than gamma + 1");
    }
    const int gamma = options.gamma;
    const bool greedy = options.sampling.greedy();

    std::vector<int32_t> seq = prompt;
    // One row per verified position, and one per guess. The draft's rows are padded with zeros
    // past its own vocabulary, so a trimmed draft needs no special case.
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
    draft.reset();
    const Clock::time_point started = Clock::now();
    if (seq.size() > 1) {  // after this, every round has the same shapes
        const int prefill = static_cast<int>(seq.size()) - 1;
        target.forward(seq.data(), prefill, nullptr, false);
        draft.forward(seq.data(), prefill, nullptr, false);
    }

    bool finished = false;
    while (local.emitted < options.max_new_tokens && !finished) {
        // -- draft gamma tokens, one at a time -------------------------------------------
        {
            const std::vector<int32_t> pending = pending_for(draft, seq);
            float* row = draft_rows.data();
            draft.forward(pending.data(), static_cast<int>(pending.size()), row, false);
            ++local.draft_forwards;
            for (int j = 0; j < gamma; ++j) {
                float* current = draft_rows.data() + static_cast<size_t>(j) * vocab;
                if (greedy) {
                    guesses[static_cast<size_t>(j)] = argmax(current, draft_vocab);
                } else {
                    warp_to_probs(current, draft_vocab, options.sampling, scratch);
                    guesses[static_cast<size_t>(j)] = sample_from_probs(current, draft_vocab, rng);
                }
                if (j + 1 < gamma) {
                    float* next = draft_rows.data() + static_cast<size_t>(j + 1) * vocab;
                    draft.forward(&guesses[static_cast<size_t>(j)], 1, next, false);
                    ++local.draft_forwards;
                }
            }
        }

        // -- verify all of them in one target pass ----------------------------------------
        verify[0] = seq.back();
        std::copy(guesses.begin(), guesses.end(), verify.begin() + 1);
        target.forward(verify.data(), gamma + 1, target_rows.data(), true);
        ++local.target_forwards;
        if (!greedy) {
            for (int j = 0; j <= gamma; ++j) {
                warp_to_probs(target_rows.data() + static_cast<size_t>(j) * vocab, vocab,
                              options.sampling, scratch);
            }
        }

        const Verdict verdict =
            accept_or_resample(target_rows.data(), vocab, draft_rows.data(), vocab, guesses.data(),
                               gamma, vocab, greedy, rng, residual);
        ++local.rounds;
        local.accepted += verdict.accepted;
        local.accepted_lengths[static_cast<size_t>(verdict.accepted)] += 1;
        if (verdict.accepted < gamma) {
            ++local.rejections;
        }

        // -- commit, then rewind both caches to "everything but the newest token" ----------
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
        const int keep = static_cast<int>(seq.size()) - 1;
        target.set_pos(keep);
        draft.set_pos(std::min(draft.pos(), keep));
    }
    local.seconds = std::chrono::duration<double>(Clock::now() - started).count();

    if (stats != nullptr) {
        *stats = local;
    }
    return std::vector<int32_t>(seq.begin() + static_cast<long>(prompt.size()), seq.end());
}

}  // namespace specdraft
