"""Trimming the draft's output vocabulary (plan §8.2).

The claim being tested is that this is free in correctness terms: the draft can only propose the
tokens it kept, which costs acceptance, but what comes out of the decoder still follows the target
exactly, because the acceptance rule reads q only where the draft proposed and resamples from the
residual everywhere else.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

torch = pytest.importorskip("torch")
stats_module = pytest.importorskip("scipy.stats")

from specdraft import _engine as cpp  # noqa: E402
from specdraft.export import read_model, write_model  # noqa: E402
from specdraft.prune import choose_vocabulary, token_frequencies  # noqa: E402
from specdraft.reference import TINY_CONFIG, perturbed_copy, random_reference  # noqa: E402
from specdraft.sampling import SamplingConfig, warp_probs  # noqa: E402
from specdraft.train import TrainedSequence  # noqa: E402

SMALL = dataclasses.replace(TINY_CONFIG, vocab_size=64, num_hidden_layers=1)
KEEP = 32


# ------------------------------------------------------------------- choosing the tokens


def test_frequencies_count_response_positions():
    sequences = [
        TrainedSequence([1, 1, 1, 5, 5, 7], response_start=3),  # 5, 5, 7 are generated
        TrainedSequence([2, 5, 9], response_start=1),  # 5, 9
    ]
    counts = token_frequencies(sequences, vocab_size=10)
    assert counts[5].item() == 3
    assert counts[7].item() == 1 and counts[9].item() == 1
    assert counts[1].item() == 0, "prompt tokens are read, not generated"

    everything = token_frequencies(sequences, vocab_size=10, response_only=False)
    assert everything[1].item() == 3


def test_choosing_keeps_the_frequent_tokens_and_reports_coverage():
    counts = torch.zeros(10, dtype=torch.long)
    counts[3] = 50
    counts[7] = 30
    counts[1] = 20
    ids, covered = choose_vocabulary(counts, keep=2)
    assert ids.tolist() == [3, 7]
    assert covered == pytest.approx(0.8)  # the dropped token held a fifth of the mass


def test_tokens_can_be_forced_to_survive():
    counts = torch.zeros(10, dtype=torch.long)
    counts[4] = 100
    ids, _ = choose_vocabulary(counts, keep=1, always_keep=[0, 9])
    assert ids.tolist() == [0, 4, 9]  # the stop token must never be dropped, for instance


def test_impossible_sizes_are_rejected():
    counts = torch.ones(10, dtype=torch.long)
    for keep in (0, 11):
        with pytest.raises(ValueError):
            choose_vocabulary(counts, keep=keep)


# ---------------------------------------------------------------------- the file and engine


@pytest.fixture(scope="module")
def trimmed_pair(tmp_path_factory):
    """A full target, and a draft whose output layer covers only half the vocabulary."""
    directory = tmp_path_factory.mktemp("trim")
    target_reference = random_reference(SMALL, seed=0)
    draft_reference = perturbed_copy(target_reference, sigma=0.02, seed=1)
    kept = np.arange(0, SMALL.vocab_size, 2, dtype=np.int32)  # every other token

    target_path = directory / "target.sdm"
    write_model(target_path, target_reference.state_dict(), SMALL, weight_format="q8")
    draft_path = directory / "draft-trimmed.sdm"
    write_model(
        draft_path, draft_reference.state_dict(), SMALL, weight_format="q8", output_map=kept
    )
    return target_path, draft_path, kept


def test_the_file_records_the_map(trimmed_pair):
    _, draft_path, kept = trimmed_pair
    written = read_model(draft_path)
    assert written.metadata["output_vocab"] == len(kept)
    assert written.entries["output_map"].format == "i32"
    assert np.array_equal(written.tensor("output_map"), kept)
    # The embedding stays whole: the draft is still fed tokens the target chose.
    assert written.entries["token_embd"].shape == (SMALL.vocab_size, SMALL.hidden_size)
    assert written.entries["output"].shape == (len(kept), SMALL.hidden_size)


def test_the_engine_scores_only_the_kept_tokens(trimmed_pair):
    _, draft_path, kept = trimmed_pair
    draft = cpp.Model(str(draft_path), max_positions=64, max_batch=5)
    assert draft.trimmed_vocabulary
    assert draft.logit_count == len(kept)
    assert draft.config["output_vocab"] == len(kept)
    assert draft.token_for_logit(0) == 0 and draft.token_for_logit(1) == 2

    logits = draft.forward([1, 2, 3], all_logits=True)
    assert logits.shape == (3, len(kept))


def test_a_trimmed_draft_reads_fewer_bytes(trimmed_pair):
    """The entire point: the output layer is a quarter of a draft's bytes per step."""
    target_path, draft_path, _ = trimmed_pair
    full = cpp.Model(str(target_path), max_positions=16)
    trimmed = cpp.Model(str(draft_path), max_positions=16)
    assert trimmed.weight_bytes_per_token < full.weight_bytes_per_token


def test_an_untrimmed_model_is_unaffected(trimmed_pair):
    target_path, _, _ = trimmed_pair
    target = cpp.Model(str(target_path), max_positions=16)
    assert not target.trimmed_vocabulary
    assert target.logit_count == SMALL.vocab_size
    assert target.token_for_logit(5) == 5


def test_a_bad_index_is_refused(trimmed_pair):
    _, draft_path, kept = trimmed_pair
    draft = cpp.Model(str(draft_path), max_positions=16)
    with pytest.raises(Exception):
        draft.token_for_logit(len(kept))


def test_a_map_outside_the_vocabulary_is_rejected(tmp_path):
    model = random_reference(SMALL, seed=2)
    with pytest.raises(ValueError):
        write_model(tmp_path / "bad.sdm", model.state_dict(), SMALL, weight_format="q8",
                    output_map=np.array([0, SMALL.vocab_size], dtype=np.int32))
    with pytest.raises(ValueError):
        write_model(tmp_path / "dup.sdm", model.state_dict(), SMALL, weight_format="q8",
                    output_map=np.array([3, 3], dtype=np.int32))


# ------------------------------------------------------------------- still exactly correct


def test_greedy_decoding_with_a_trimmed_draft_matches_plain(trimmed_pair):
    target_path, draft_path, _ = trimmed_pair
    target = cpp.Model(str(target_path), max_positions=96, max_batch=5)
    draft = cpp.Model(str(draft_path), max_positions=96, max_batch=5)
    prompt = [1, 3, 5, 7]

    expected, _ = cpp.generate_plain(target, prompt, max_new_tokens=24)
    got, stats = cpp.generate_speculative(target, draft, prompt, max_new_tokens=24, gamma=4)

    assert got == expected, "trimming the draft must not change what the target produces"
    assert stats["proposed"] > 0, "the draft should still be proposing something"


def test_the_trimmed_draft_only_ever_proposes_kept_tokens(trimmed_pair):
    """Acceptance can fall, but a dropped token must never be guessed."""
    target_path, draft_path, kept = trimmed_pair
    target = cpp.Model(str(target_path), max_positions=96, max_batch=5)
    draft = cpp.Model(str(draft_path), max_positions=96, max_batch=5)
    allowed = set(kept.tolist())

    # Whatever the draft contributes has to come from its own vocabulary; tokens outside it can
    # only arrive from the target. Running greedily makes the draft's guesses deterministic.
    tokens, stats = cpp.generate_speculative(target, draft, [2, 4, 6], max_new_tokens=20, gamma=4)
    from_draft = stats["accepted"]
    assert from_draft <= sum(1 for token in tokens if token in allowed)


@pytest.mark.slow
def test_sampling_with_a_trimmed_draft_is_still_exact(tmp_path):
    """Chi-square over every two-token continuation: the distribution must be untouched."""
    tiny = dataclasses.replace(SMALL, vocab_size=8)
    target_reference = random_reference(tiny, seed=5)
    draft_reference = perturbed_copy(target_reference, sigma=0.05, seed=6)
    target_path = tmp_path / "t.sdm"
    draft_path = tmp_path / "d.sdm"
    write_model(target_path, target_reference.state_dict(), tiny, weight_format="q8")
    write_model(
        draft_path, draft_reference.state_dict(), tiny, weight_format="q8",
        output_map=np.arange(0, tiny.vocab_size, 2, dtype=np.int32),  # half the tokens dropped
    )
    target = cpp.Model(str(target_path), max_positions=64, max_batch=5)
    draft = cpp.Model(str(draft_path), max_positions=64, max_batch=5)
    assert draft.logit_count == tiny.vocab_size // 2
    vocab = tiny.vocab_size
    prompt = [1, 2, 3]
    config = SamplingConfig(temperature=1.0)

    first = warp_probs(
        torch.from_numpy(target.forward(np.array(prompt, dtype=np.int32))[0]), config
    ).numpy()
    exact = np.empty((vocab, vocab))
    for a in range(vocab):
        target.reset()
        row = target.forward(np.array(prompt + [a], dtype=np.int32))[0]
        exact[a] = first[a] * warp_probs(torch.from_numpy(row), config).numpy()

    trials = 4000
    counts = np.zeros((vocab, vocab))
    for trial in range(trials):
        got, _ = cpp.generate_speculative(
            target, draft, prompt, max_new_tokens=2, gamma=3, temperature=1.0, seed=trial
        )
        counts[got[0], got[1]] += 1

    expected = (exact * trials).flatten()
    keep = expected > 5
    observed = counts.flatten()[keep]
    scaled = expected[keep] * (observed.sum() / expected[keep].sum())
    pvalue = float(stats_module.chisquare(observed, scaled).pvalue)
    assert pvalue > 1e-3, f"a trimmed draft changed the distribution (p={pvalue:.2e})"
