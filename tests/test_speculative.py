"""Speculative decoding must be indistinguishable from plain decoding.

Greedy runs have to reproduce plain greedy token for token, and sampling runs have to
produce the target's distribution, which is checked here by enumerating every two-token
continuation of a tiny model exactly. Both tests run on random models, so no weights are
needed, and both later become the oracle for the C++ engine (plan §10.3).
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

torch = pytest.importorskip("torch")
stats = pytest.importorskip("scipy.stats")

from specdraft.offline import score_sequence, simulate  # noqa: E402
from specdraft.reference import (  # noqa: E402
    TINY_CONFIG,
    perturbed_copy,
    random_reference,
)
from specdraft.sampling import GREEDY, SamplingConfig, warp_probs  # noqa: E402
from specdraft.speculative import plain_generate, speculative_generate  # noqa: E402

SAMPLING = SamplingConfig(temperature=1.0)
# One layer and a tiny vocabulary, so every two-token continuation can be enumerated.
SMALL_CONFIG = dataclasses.replace(TINY_CONFIG, vocab_size=8, num_hidden_layers=1)


@pytest.fixture(scope="module")
def pair():
    """A target and a draft that agrees with it often but not always."""
    target = random_reference(TINY_CONFIG, seed=0)
    return target, perturbed_copy(target, sigma=0.02, seed=1)


def prompt_tokens(n: int, vocab: int, seed: int) -> "torch.Tensor":
    generator = torch.Generator().manual_seed(seed)
    return torch.randint(0, vocab, (n,), generator=generator)


# ------------------------------------------------------------------------ correctness


@pytest.mark.parametrize("gamma", [1, 2, 3, 5, 8])
def test_greedy_matches_plain_decoding(pair, gamma):
    target, draft = pair
    prompt = prompt_tokens(5, TINY_CONFIG.vocab_size, seed=2)

    expected, plain_stats = plain_generate(target, prompt, 40)
    got, spec_stats = speculative_generate(target, draft, prompt, 40, gamma=gamma)

    assert got == expected
    assert spec_stats.emitted == plain_stats.emitted == 40
    assert spec_stats.target_forwards <= plain_stats.target_forwards
    assert spec_stats.tokens_per_target_forward >= 1.0


@pytest.mark.parametrize("gamma", [1, 4])
def test_a_draft_identical_to_the_target_accepts_everything(pair, gamma):
    target, _ = pair
    prompt = prompt_tokens(4, TINY_CONFIG.vocab_size, seed=3)
    tokens = (gamma + 1) * 6  # a whole number of rounds

    got, spec_stats = speculative_generate(target, target, prompt, tokens, gamma=gamma)

    assert got == plain_generate(target, prompt, tokens)[0]
    assert set(spec_stats.accepted_lengths) == {gamma}
    assert spec_stats.tokens_per_target_forward == pytest.approx(gamma + 1)
    assert spec_stats.alpha == 1.0


@pytest.mark.slow
def test_sampling_reproduces_the_targets_distribution():
    """Chi-square over all 64 two-token continuations, against exact probabilities."""
    target = random_reference(SMALL_CONFIG, seed=4)
    draft = perturbed_copy(target, sigma=0.05, seed=5)
    vocab = SMALL_CONFIG.vocab_size
    prompt = prompt_tokens(3, vocab, seed=6)

    first = warp_probs(target.forward(prompt, only_last_logits=True)[-1], SAMPLING)
    exact = torch.empty(vocab, vocab)
    for a in range(vocab):
        extended = torch.cat([prompt, torch.tensor([a])])
        second = warp_probs(target.forward(extended, only_last_logits=True)[-1], SAMPLING)
        exact[a] = first[a] * second

    trials = 3_000
    generator = torch.Generator().manual_seed(7)
    counts = torch.zeros(vocab, vocab)
    for _ in range(trials):
        got, _ = speculative_generate(
            target, draft, prompt, 2, gamma=3, config=SAMPLING, generator=generator
        )
        counts[got[0], got[1]] += 1

    expected = (exact * trials).flatten().numpy()
    keep = expected > 5.0
    pvalue = stats.chisquare(counts.flatten().numpy()[keep], expected[keep]).pvalue
    assert pvalue > 1e-3, f"distribution differs from the target (p={pvalue:.2e})"


def test_stop_token_ends_generation_immediately(pair):
    target, draft = pair
    prompt = prompt_tokens(4, TINY_CONFIG.vocab_size, seed=8)
    reference, _ = plain_generate(target, prompt, 20)
    stop = {reference[6]}

    got, _ = speculative_generate(target, draft, prompt, 20, gamma=4, stop=stop)

    assert got[-1] in stop
    assert all(token not in stop for token in got[:-1])
    assert got == reference[: len(got)]


@pytest.mark.parametrize("gamma", [1, 3, 7])
@pytest.mark.parametrize("max_new_tokens", [1, 5, 13])
def test_never_emits_more_than_asked(pair, gamma, max_new_tokens):
    target, draft = pair
    prompt = prompt_tokens(3, TINY_CONFIG.vocab_size, seed=9)
    got, spec_stats = speculative_generate(
        target, draft, prompt, max_new_tokens, gamma=gamma
    )
    assert len(got) == max_new_tokens == spec_stats.emitted


def test_rejects_bad_gamma(pair):
    target, draft = pair
    with pytest.raises(ValueError):
        speculative_generate(target, draft, prompt_tokens(2, 64, seed=10), 4, gamma=0)


def test_single_token_prompt_works(pair):
    """There is nothing to prefill, so the first round has to handle it."""
    target, draft = pair
    prompt = torch.tensor([5])
    got, _ = speculative_generate(target, draft, prompt, 10, gamma=3)
    assert got == plain_generate(target, prompt, 10)[0]


# --------------------------------------------------- the offline simulator is exact


def test_offline_simulation_matches_online_decoding_exactly(pair):
    """Plan §11.6: for greedy, the simulator is not an estimate but the same computation."""
    target, draft = pair
    prompt = prompt_tokens(6, TINY_CONFIG.vocab_size, seed=11)
    gamma = 4

    generated, online = speculative_generate(target, draft, prompt, 60, gamma=gamma)

    tokens = torch.cat([prompt, torch.tensor(generated)])
    metrics = score_sequence(target, draft, tokens, response_start=len(prompt), config=GREEDY)
    assert metrics.is_greedy_reference, "speculative greedy output must be the target's own"
    offline_tau, lengths = simulate(metrics.accept_prob(greedy=True), gamma=gamma)

    assert offline_tau == pytest.approx(online.tokens_per_target_forward)
    assert int(lengths.sum()) == online.rounds
    assert np.array_equal(
        lengths, np.bincount(online.accepted_lengths, minlength=gamma + 1)
    )
