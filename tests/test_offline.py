"""The offline metrics and the round simulator (plan §11.6 and the model of §3)."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from specdraft.offline import (  # noqa: E402
    predicted_speedup,
    score_sequence,
    simulate,
    simulate_tokens_per_step,
    tau_from_alpha,
)
from specdraft.reference import TINY_CONFIG, perturbed_copy, random_reference  # noqa: E402
from specdraft.sampling import GREEDY, SamplingConfig  # noqa: E402

SAMPLING = SamplingConfig(temperature=1.0)


# ------------------------------------------------------------------------- simulator


@pytest.mark.parametrize("gamma", [1, 2, 4, 8])
def test_every_guess_accepted_gives_gamma_plus_one(gamma):
    accept = np.ones(12 * (gamma + 1))
    tau, lengths = simulate(accept, gamma=gamma)
    assert tau == pytest.approx(gamma + 1)
    assert lengths[gamma] == len(accept) // (gamma + 1)


@pytest.mark.parametrize("gamma", [1, 4])
def test_no_guess_accepted_gives_one(gamma):
    tau, lengths = simulate(np.zeros(50), gamma=gamma)
    assert tau == pytest.approx(1.0)
    assert lengths[0] == 50


def test_alternating_acceptance_is_hand_checkable():
    # Accept, reject, accept, reject ...: every round emits the accepted token plus one
    # from the target, so two tokens per target pass whatever γ is.
    accept = np.tile([1.0, 0.0], 25)
    assert simulate_tokens_per_step(accept, gamma=4) == pytest.approx(2.0)


@pytest.mark.parametrize("alpha", [0.3, 0.6, 0.8])
@pytest.mark.parametrize("gamma", [2, 4])
@pytest.mark.slow
def test_constant_acceptance_matches_the_formula(alpha, gamma):
    """A constant per-token rate is exactly the i.i.d. assumption behind τ(γ)."""
    tau = simulate_tokens_per_step(np.full(20_000, alpha), gamma=gamma, draws=4, seed=0)
    assert tau == pytest.approx(tau_from_alpha(alpha, gamma), rel=0.02)


def test_greedy_input_needs_only_one_draw():
    """0/1 acceptance is deterministic, so extra draws cannot change the answer."""
    accept = (np.arange(100) % 3 != 0).astype(float)
    assert simulate_tokens_per_step(accept, gamma=3, draws=1) == simulate_tokens_per_step(
        accept, gamma=3, draws=16
    )


def test_confidence_threshold_can_stop_drafting():
    accept = np.ones(60)
    confidence = np.full(60, 0.4)
    # Nothing clears the bar, so no guesses are made and the target emits every token.
    assert simulate_tokens_per_step(
        accept, gamma=4, draft_confidence=confidence, confidence_threshold=0.9
    ) == pytest.approx(1.0)
    # With a bar the draft clears, it behaves as if there were no threshold.
    assert simulate_tokens_per_step(
        accept, gamma=4, draft_confidence=confidence, confidence_threshold=0.2
    ) == pytest.approx(5.0)


def test_simulator_rejects_bad_input():
    with pytest.raises(ValueError):
        simulate(np.ones(10), gamma=0)
    with pytest.raises(ValueError):
        simulate(np.array([]), gamma=2)


# --------------------------------------------------------------------- speedup model


def test_tau_matches_closed_form():
    assert tau_from_alpha(0.7, 4) == pytest.approx(2.7731, abs=1e-4)
    assert tau_from_alpha(1.0, 3) == 4.0
    assert tau_from_alpha(0.0, 5) == 1.0


def test_predicted_speedup_reproduces_the_plan_examples():
    """The two worked examples in plan §3, with α = 0.7 and c = 0.15."""
    assert predicted_speedup(tau_from_alpha(0.7, 4), 4, c=0.15, v=1.5) == pytest.approx(1.32, abs=0.01)
    assert predicted_speedup(tau_from_alpha(0.7, 2), 2, c=0.15, v=1.2) == pytest.approx(1.46, abs=0.01)


def test_cheaper_verification_moves_the_best_gamma_up():
    """Why v(k) matters: when extra tokens are dear, fewer guesses per round win."""
    def best(v_slope: float) -> int:
        speedups = {
            gamma: predicted_speedup(
                tau_from_alpha(0.7, gamma), gamma, c=0.15, v=1.0 + v_slope * gamma
            )
            for gamma in range(1, 9)
        }
        return max(speedups, key=speedups.__getitem__)

    assert best(0.2) < best(0.05) <= best(0.0)


# ------------------------------------------------------------------------- scoring


@pytest.fixture(scope="module")
def pair():
    target = random_reference(TINY_CONFIG, seed=0)
    return target, perturbed_copy(target, sigma=0.02, seed=1)


def tokens(n: int, seed: int) -> "torch.Tensor":
    generator = torch.Generator().manual_seed(seed)
    return torch.randint(0, TINY_CONFIG.vocab_size, (n,), generator=generator)


@pytest.mark.parametrize("config", [GREEDY, SAMPLING])
def test_scoring_shapes_and_ranges(pair, config):
    target, draft = pair
    sequence = tokens(30, seed=2)
    metrics = score_sequence(target, draft, sequence, response_start=10, config=config)

    assert len(metrics) == 20
    assert metrics.tokens.tolist() == sequence[10:].tolist()
    assert 0.0 <= metrics.sampling_alpha <= 1.0
    assert ((metrics.one_minus_tvd >= 0) & (metrics.one_minus_tvd <= 1)).all()
    assert ((metrics.coupled_accept >= 0) & (metrics.coupled_accept <= 1)).all()
    assert ((metrics.draft_confidence > 0) & (metrics.draft_confidence <= 1)).all()


@pytest.mark.parametrize("config", [GREEDY, SAMPLING])
def test_a_draft_identical_to_the_target_scores_perfectly(pair, config):
    target, _ = pair
    sequence = tokens(24, seed=3)
    metrics = score_sequence(target, target, sequence, response_start=8, config=config)

    assert metrics.greedy_match.all()
    assert metrics.one_minus_tvd == pytest.approx(np.ones(len(metrics)), abs=1e-5)
    assert metrics.coupled_accept == pytest.approx(np.ones(len(metrics)), abs=1e-5)


def test_greedy_reference_check_spots_the_wrong_kind_of_text(pair):
    """The greedy simulation is only exact on the target's own greedy continuation."""
    from specdraft.speculative import plain_generate

    target, draft = pair
    prompt = tokens(6, seed=20)
    random_text = torch.cat([prompt, tokens(10, seed=21)])
    assert not score_sequence(target, draft, random_text, len(prompt)).is_greedy_reference

    greedy_text = torch.cat([prompt, torch.tensor(plain_generate(target, prompt, 10)[0])])
    assert score_sequence(target, draft, greedy_text, len(prompt)).is_greedy_reference


def test_chunking_does_not_change_the_scores(pair):
    target, draft = pair
    sequence = tokens(40, seed=4)
    whole = score_sequence(target, draft, sequence, 5, config=SAMPLING, chunk=1000)
    pieces = score_sequence(target, draft, sequence, 5, config=SAMPLING, chunk=7)
    assert np.array_equal(whole.greedy_match, pieces.greedy_match)
    assert whole.one_minus_tvd == pytest.approx(pieces.one_minus_tvd)


def test_scoring_rejects_bad_response_start(pair):
    target, draft = pair
    with pytest.raises(ValueError):
        score_sequence(target, draft, tokens(10, seed=5), response_start=0)
    with pytest.raises(ValueError):
        score_sequence(target, draft, tokens(10, seed=5), response_start=10)


# ------------------------------------------------- pooling, token classes, gamma sweeps


def test_concatenating_pools_every_column(pair):
    from specdraft.offline import concatenate

    target, draft = pair
    parts = [
        score_sequence(target, draft, tokens(20, seed=30), 5, config=SAMPLING),
        score_sequence(target, draft, tokens(14, seed=31), 4, config=SAMPLING),
    ]
    pooled = concatenate(parts)
    assert len(pooled) == len(parts[0]) + len(parts[1])
    assert pooled.one_minus_tvd[0] == parts[0].one_minus_tvd[0]
    assert pooled.tokens[-1] == parts[1].tokens[-1]
    with pytest.raises(ValueError):
        concatenate([])


@pytest.mark.parametrize(
    "text,expected",
    [
        (" ", "whitespace"),
        ("\n\n", "whitespace"),
        (" 42", "digit"),
        ("3.5", "digit"),
        (".", "punctuation"),
        (" ,", "punctuation"),
        (" Paris", "word_start_capital"),
        (" paris", "word_start"),
        ("ing", "word_continuation"),
        ("日本", "cjk"),
        ("", "other"),
    ],
)
def test_token_classes(text, expected):
    from specdraft.offline import classify_token

    assert classify_token(text) == expected


def test_special_tokens_are_their_own_class():
    from specdraft.offline import classify_token

    assert classify_token("<|im_end|>", is_special=True) == "special"


def test_acceptance_by_class_splits_the_blame():
    """A class that is frequent *and* poorly accepted is where a draft needs work."""
    import numpy as np

    from specdraft.offline import PositionMetrics, acceptance_by_class

    metrics = PositionMetrics(
        greedy_match=np.array([True, True, False, False, True, False]),
        one_minus_tvd=np.zeros(6),
        coupled_accept=np.zeros(6),
        draft_confidence=np.ones(6),
        target_match=np.ones(6, dtype=bool),
        tokens=np.arange(6),
    )
    classes = ["word_start", "word_start", "digit", "digit", "word_start", "digit"]
    profile = acceptance_by_class(metrics, classes, greedy=True)

    assert profile["word_start"]["acceptance"] == pytest.approx(1.0)
    assert profile["digit"]["acceptance"] == pytest.approx(0.0)
    assert profile["word_start"]["share"] == pytest.approx(0.5)
    assert list(profile) == ["word_start", "digit"]  # most frequent first


def test_gamma_sweep_answers_every_gamma_from_one_pass(pair):
    from specdraft.offline import gamma_sweep

    target, draft = pair
    metrics = score_sequence(target, draft, tokens(60, seed=32), 10, config=GREEDY)
    sweep = gamma_sweep(metrics, [1, 2, 4, 8], greedy=True)

    assert set(sweep) == {1, 2, 4, 8}
    assert all(1.0 <= value <= gamma + 1 for gamma, value in sweep.items())
    # More guesses per round can never emit fewer tokens per pass.
    assert sweep[1] <= sweep[2] <= sweep[4] <= sweep[8]


def test_a_confidence_threshold_can_only_reduce_tokens_per_pass(pair):
    from specdraft.offline import gamma_sweep

    target, draft = pair
    metrics = score_sequence(target, draft, tokens(60, seed=33), 10, config=GREEDY)
    without = gamma_sweep(metrics, [4], greedy=True)[4]
    with_threshold = gamma_sweep(metrics, [4], greedy=True, confidence_threshold=1.01)[4]
    assert with_threshold == pytest.approx(1.0)  # nothing clears it, so no guesses are made
    assert with_threshold <= without


def test_best_gamma_uses_the_measured_curve():
    from specdraft.offline import best_gamma

    tokens_per_step = {1: 1.8, 2: 2.5, 4: 3.4, 8: 4.0}
    # A flat verification cost favours many guesses; a steeply rising one favours few, which is
    # the whole reason v(k) is measured rather than assumed.
    flat = best_gamma(tokens_per_step, c=0.1, v=1.0)
    steep = best_gamma(tokens_per_step, c=0.1, v={2: 1.1, 3: 1.4, 5: 2.4, 9: 5.0})
    assert flat[0] >= steep[0]
    assert flat[1] > steep[1]
