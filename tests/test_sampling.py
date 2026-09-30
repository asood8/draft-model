"""The acceptance rule must leave the target's distribution untouched.

These are the tests from plan §10.3: statistics on toy distributions, which is the only
way to catch a rule that looks right but shifts probabilities slightly. This module is also
the oracle the C++ implementation will be compared against.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
stats = pytest.importorskip("scipy.stats")

from specdraft.sampling import (  # noqa: E402
    SamplingConfig,
    accept_or_resample,
    expected_acceptance,
    residual_probs,
    warp_probs,
)

SIGNIFICANCE = 1e-3  # a correct rule fails this once in a thousand runs


def random_distribution(size: int, seed: int, sharpness: float = 2.0) -> "torch.Tensor":
    generator = torch.Generator().manual_seed(seed)
    return torch.softmax(torch.randn(size, generator=generator) * sharpness, dim=-1)


def chi_square_pvalue(counts, expected_probs, trials) -> float:
    expected = expected_probs.numpy() * trials
    keep = expected > 5.0  # the test needs a reasonable expected count per cell
    return float(stats.chisquare(counts[keep], expected[keep]).pvalue)


# ------------------------------------------------------------------------------ warps


def test_temperature_scales_logits():
    logits = torch.tensor([1.0, 2.0, 3.0])
    cold = warp_probs(logits, SamplingConfig(temperature=0.5))
    warm = warp_probs(logits, SamplingConfig(temperature=2.0))
    assert cold[2] > warm[2] and cold.sum() == pytest.approx(1.0)


def test_top_k_keeps_exactly_k():
    probs = warp_probs(torch.arange(10.0), SamplingConfig(top_k=3))
    assert int((probs > 0).sum()) == 3
    assert probs.sum() == pytest.approx(1.0)
    assert set(probs.topk(3).indices.tolist()) == {9, 8, 7}


def test_top_p_keeps_the_shortest_prefix_reaching_p():
    # Probabilities 0.5, 0.3, 0.15, 0.05: 0.5 alone is short of 0.7, so two are kept.
    logits = torch.log(torch.tensor([0.5, 0.3, 0.15, 0.05]))
    probs = warp_probs(logits, SamplingConfig(top_p=0.7))
    assert int((probs > 0).sum()) == 2
    assert probs[:2].tolist() == pytest.approx([0.625, 0.375], abs=1e-6)


def test_top_p_always_keeps_the_most_likely_token():
    logits = torch.log(torch.tensor([0.9, 0.05, 0.05]))
    probs = warp_probs(logits, SamplingConfig(top_p=0.1))
    assert probs[0] == pytest.approx(1.0)


def test_greedy_has_no_distribution():
    with pytest.raises(ValueError):
        warp_probs(torch.zeros(4), SamplingConfig(temperature=0.0))


def test_invalid_configs_rejected():
    for kwargs in ({"temperature": -1.0}, {"top_p": 0.0}, {"top_p": 1.5}, {"top_k": -1}):
        with pytest.raises(ValueError):
            SamplingConfig(**kwargs)


# -------------------------------------------------------------------- acceptance rule


@pytest.mark.parametrize("gamma", [1, 3])
@pytest.mark.slow
def test_first_emitted_token_follows_the_target(gamma):
    """The point of the whole scheme: sampling through the draft changes nothing."""
    vocab, trials = 8, 6_000
    p = random_distribution(vocab, seed=1)
    q = random_distribution(vocab, seed=2)
    p_rows = p.expand(gamma + 1, vocab).contiguous()
    q_rows = q.expand(gamma, vocab).contiguous()

    generator = torch.Generator().manual_seed(3)
    counts = torch.zeros(vocab)
    for _ in range(trials):
        guesses = torch.multinomial(q, gamma, replacement=True, generator=generator)
        accepted, next_token = accept_or_resample(p_rows, q_rows, guesses, generator=generator)
        first = int(accepted[0]) if len(accepted) else next_token
        counts[first] += 1

    assert chi_square_pvalue(counts.numpy(), p, trials) > SIGNIFICANCE


@pytest.mark.slow
def test_rejection_resamples_from_the_residual():
    """Force a rejection every round, so the emitted token must follow max(0, p - q)."""
    vocab, trials = 8, 6_000
    p = random_distribution(vocab, seed=4)
    q = torch.zeros(vocab)
    q[0] = 1.0  # the draft always guesses token 0 ...
    p = p.clone()
    p[0] = 0.0
    p = p / p.sum()  # ... which the target never wants, so every guess is rejected

    expected = residual_probs(p, q)
    generator = torch.Generator().manual_seed(5)
    counts = torch.zeros(vocab)
    for _ in range(trials):
        accepted, next_token = accept_or_resample(
            p.expand(2, vocab).contiguous(),
            q.expand(1, vocab).contiguous(),
            torch.zeros(1, dtype=torch.long),
            generator=generator,
        )
        assert len(accepted) == 0
        counts[next_token] += 1

    assert chi_square_pvalue(counts.numpy(), expected, trials) > SIGNIFICANCE


def test_identical_models_accept_everything():
    p = random_distribution(16, seed=6)
    gamma = 5
    generator = torch.Generator().manual_seed(7)
    for _ in range(200):
        guesses = torch.multinomial(p, gamma, replacement=True, generator=generator)
        accepted, _ = accept_or_resample(
            p.expand(gamma + 1, 16).contiguous(),
            p.expand(gamma, 16).contiguous(),
            guesses,
            generator=generator,
        )
        assert len(accepted) == gamma


@pytest.mark.slow
def test_acceptance_rate_equals_one_minus_tvd():
    """Σ min(p, q) is what the offline metric measures, so it must match the rule."""
    vocab, trials, gamma = 8, 8_000, 1
    p = random_distribution(vocab, seed=8)
    q = random_distribution(vocab, seed=9)
    predicted = float(expected_acceptance(p, q))

    generator = torch.Generator().manual_seed(10)
    accepted_count = 0
    for _ in range(trials):
        guesses = torch.multinomial(q, gamma, replacement=True, generator=generator)
        accepted, _ = accept_or_resample(
            p.expand(gamma + 1, vocab).contiguous(),
            q.expand(gamma, vocab).contiguous(),
            guesses,
            generator=generator,
        )
        accepted_count += len(accepted)

    observed = accepted_count / trials
    assert abs(observed - predicted) < 4.0 * (0.25 / trials) ** 0.5  # ~4 standard errors


def test_greedy_accepts_matching_tokens_and_takes_the_targets_pick():
    scores = torch.tensor([[0.0, 5.0, 1.0], [3.0, 0.0, 0.0], [0.0, 0.0, 7.0]])  # argmaxes 1, 0, 2
    accepted, next_token = accept_or_resample(
        scores, torch.zeros(2, 3), torch.tensor([1, 0]), greedy=True
    )
    assert accepted.tolist() == [1, 0] and next_token == 2  # all accepted -> bonus token

    accepted, next_token = accept_or_resample(
        scores, torch.zeros(2, 3), torch.tensor([1, 2]), greedy=True
    )
    assert accepted.tolist() == [1] and next_token == 0  # rejected at position 1 -> correction


def test_shape_mismatch_is_rejected():
    with pytest.raises(ValueError):
        accept_or_resample(torch.zeros(2, 4), torch.zeros(2, 4), torch.zeros(2, dtype=torch.long))
