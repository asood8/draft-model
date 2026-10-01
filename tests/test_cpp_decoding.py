"""The decoding loop inside the engine (plan §10.2).

The Python implementation in ``specdraft.sampling`` and ``specdraft.speculative`` is the
oracle. Greedy decoding is deterministic, so the C++ loop must reproduce it token for token.
Sampling uses the engine's own generator, so its output cannot match draw for draw; what must
hold is the distribution, checked here by enumerating every two-token continuation of a tiny
model exactly, the same way the Python loop was checked.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

torch = pytest.importorskip("torch")
stats_module = pytest.importorskip("scipy.stats")

from specdraft import _engine as cpp  # noqa: E402
from specdraft.engine import EngineModel  # noqa: E402
from specdraft.export import write_model  # noqa: E402
from specdraft.reference import TINY_CONFIG, perturbed_copy, random_reference  # noqa: E402
from specdraft.sampling import SamplingConfig, warp_probs  # noqa: E402
from specdraft.speculative import plain_generate, speculative_generate  # noqa: E402

MAX_POSITIONS = 128
GAMMA_VALUES = [1, 2, 4, 6]
SMALL_CONFIG = dataclasses.replace(TINY_CONFIG, vocab_size=8, num_hidden_layers=1)


@pytest.fixture(scope="module")
def paths(tmp_path_factory):
    directory = tmp_path_factory.mktemp("cpp_decoding")
    target = random_reference(TINY_CONFIG, seed=0)
    draft = perturbed_copy(target, sigma=0.02, seed=1)
    out = {}
    for name, model in (("target", target), ("draft", draft)):
        path = directory / f"{name}.sdm"
        write_model(path, model.state_dict(), model.config, weight_format="q8")
        out[name] = path
    return out


def open_pair(paths, gamma: int = 8):
    target = cpp.Model(str(paths["target"]), max_positions=MAX_POSITIONS, max_batch=gamma + 1)
    draft = cpp.Model(str(paths["draft"]), max_positions=MAX_POSITIONS, max_batch=gamma + 1)
    return target, draft


def chi_square_pvalue(observed: np.ndarray, expected: np.ndarray) -> float:
    """Chi-square over the cells with enough expected mass to be meaningful.

    Dropping the sparse cells means the two totals no longer agree, which scipy refuses, so
    the expected counts are rescaled to the observations that remain.
    """
    keep = expected > 5
    observed, expected = observed[keep], expected[keep]
    expected = expected * (observed.sum() / expected.sum())
    return float(stats_module.chisquare(observed, expected).pvalue)


def prompt_tokens(n: int, vocab: int, seed: int) -> list[int]:
    return np.random.default_rng(seed).integers(0, vocab, size=n).tolist()


# ------------------------------------------------------------------ warps and the rule


@pytest.mark.parametrize(
    "config",
    [
        SamplingConfig(temperature=1.0),
        SamplingConfig(temperature=0.7),
        SamplingConfig(temperature=1.0, top_k=3),
        SamplingConfig(temperature=1.0, top_p=0.7),
        SamplingConfig(temperature=0.7, top_k=20, top_p=0.8),  # Qwen's recommended settings
    ],
)
def test_warps_match_the_python_oracle(config):
    logits = np.random.default_rng(3).standard_normal(200).astype(np.float32) * 3.0
    mine = cpp.warp_to_probs(logits, config.temperature, config.top_k, config.top_p)
    theirs = warp_probs(torch.from_numpy(logits), config).numpy()
    assert mine.sum() == pytest.approx(1.0, abs=1e-5)
    np.testing.assert_allclose(mine, theirs, atol=1e-6)


def test_acceptance_rule_greedy_matches_the_oracle():
    scores = np.array([[0.0, 5.0, 1.0], [3.0, 0.0, 0.0], [0.0, 0.0, 7.0]], dtype=np.float32)
    zeros = np.zeros((2, 3), dtype=np.float32)
    assert cpp.accept_or_resample(scores, zeros, [1, 0], greedy=True) == (2, 2)
    assert cpp.accept_or_resample(scores, zeros, [1, 2], greedy=True) == (1, 0)


@pytest.mark.slow
def test_first_emitted_token_follows_the_target():
    """The engine's own acceptance rule, checked statistically like the Python one."""
    rng = np.random.default_rng(4)
    vocab, gamma, trials = 8, 3, 6000
    p = np.exp(rng.standard_normal(vocab) * 2).astype(np.float32)
    p /= p.sum()
    q = np.exp(rng.standard_normal(vocab) * 2).astype(np.float32)
    q /= q.sum()
    p_rows = np.repeat(p[None, :], gamma + 1, axis=0)
    q_rows = np.repeat(q[None, :], gamma, axis=0)

    counts = np.zeros(vocab)
    for trial in range(trials):
        guesses = rng.choice(vocab, size=gamma, p=q).tolist()
        accepted, next_token = cpp.accept_or_resample(p_rows, q_rows, guesses, False, trial)
        counts[guesses[0] if accepted else next_token] += 1

    pvalue = chi_square_pvalue(counts, p.astype(np.float64) * trials)
    assert pvalue > 1e-3, f"emitted token does not follow p (p={pvalue:.2e})"


# --------------------------------------------------------------- the loop, end to end


@pytest.mark.parametrize("gamma", GAMMA_VALUES)
def test_cpp_greedy_speculative_matches_cpp_plain(paths, gamma):
    target, draft = open_pair(paths, gamma)
    prompt = prompt_tokens(5, TINY_CONFIG.vocab_size, seed=5)

    expected, plain_stats = cpp.generate_plain(target, prompt, max_new_tokens=40)
    got, spec_stats = cpp.generate_speculative(
        target, draft, prompt, max_new_tokens=40, gamma=gamma
    )

    assert got == expected
    assert spec_stats["emitted"] == plain_stats["emitted"] == 40
    assert spec_stats["target_forwards"] < plain_stats["target_forwards"]
    assert spec_stats["tokens_per_target_forward"] > 1.0
    assert sum(spec_stats["accepted_lengths"]) == spec_stats["rounds"]
    assert 0.0 <= spec_stats["alpha"] <= 1.0
    assert spec_stats["tokens_per_second"] > 0.0


@pytest.mark.parametrize("gamma", GAMMA_VALUES)
def test_cpp_loop_matches_the_python_loop_token_for_token(paths, gamma):
    """Same models, same rule: the two implementations must not diverge."""
    target, draft = open_pair(paths, gamma)
    prompt = prompt_tokens(6, TINY_CONFIG.vocab_size, seed=6)
    from_cpp, _ = cpp.generate_speculative(target, draft, prompt, max_new_tokens=32, gamma=gamma)

    py_target = EngineModel(paths["target"], max_positions=MAX_POSITIONS)
    py_draft = EngineModel(paths["draft"], max_positions=MAX_POSITIONS)
    from_python, _ = speculative_generate(
        py_target, py_draft, torch.tensor(prompt), 32, gamma=gamma
    )
    assert from_cpp == from_python
    assert from_cpp == plain_generate(py_target, torch.tensor(prompt), 32)[0]


def test_a_draft_identical_to_the_target_accepts_everything(paths):
    target = cpp.Model(str(paths["target"]), max_positions=MAX_POSITIONS, max_batch=5)
    same = cpp.Model(str(paths["target"]), max_positions=MAX_POSITIONS, max_batch=5)
    prompt = prompt_tokens(4, TINY_CONFIG.vocab_size, seed=7)

    got, spec_stats = cpp.generate_speculative(target, same, prompt, max_new_tokens=24, gamma=4)

    assert got == cpp.generate_plain(target, prompt, max_new_tokens=24)[0]
    assert spec_stats["alpha"] == 1.0
    assert spec_stats["tokens_per_target_forward"] == pytest.approx(5.0, rel=0.2)


def test_stop_token_ends_generation(paths):
    target, draft = open_pair(paths)
    prompt = prompt_tokens(4, TINY_CONFIG.vocab_size, seed=8)
    reference, _ = cpp.generate_plain(target, prompt, max_new_tokens=20)
    stop = [reference[5]]

    got, _ = cpp.generate_speculative(
        target, draft, prompt, max_new_tokens=20, gamma=4, stop=stop
    )
    assert got[-1] in stop
    assert all(token not in stop for token in got[:-1])


@pytest.mark.parametrize("max_new_tokens", [1, 7, 15])
def test_token_budget_is_respected(paths, max_new_tokens):
    target, draft = open_pair(paths)
    prompt = prompt_tokens(3, TINY_CONFIG.vocab_size, seed=9)
    got, _ = cpp.generate_speculative(
        target, draft, prompt, max_new_tokens=max_new_tokens, gamma=4
    )
    assert len(got) == max_new_tokens


def test_seeded_sampling_is_reproducible(paths):
    target, draft = open_pair(paths)
    prompt = prompt_tokens(4, TINY_CONFIG.vocab_size, seed=10)
    kwargs = {"max_new_tokens": 16, "gamma": 3, "temperature": 1.0}
    first, _ = cpp.generate_speculative(target, draft, prompt, seed=1234, **kwargs)
    again, _ = cpp.generate_speculative(target, draft, prompt, seed=1234, **kwargs)
    different, _ = cpp.generate_speculative(target, draft, prompt, seed=5678, **kwargs)
    assert first == again
    assert first != different  # a different seed should explore elsewhere


def test_bad_arguments_are_rejected(paths):
    target, draft = open_pair(paths)
    with pytest.raises(Exception):
        cpp.generate_speculative(target, draft, [], max_new_tokens=4)
    with pytest.raises(Exception):
        cpp.generate_speculative(target, draft, [1, 2], max_new_tokens=4, gamma=0)
    with pytest.raises(Exception):
        cpp.generate_plain(target, [1, 2], max_new_tokens=4, top_p=0.0)
    # gamma + 1 must fit in the target's batch, or the weights cannot be shared in one pass.
    narrow = cpp.Model(str(paths["target"]), max_positions=MAX_POSITIONS, max_batch=2)
    with pytest.raises(Exception):
        cpp.generate_speculative(narrow, draft, [1, 2], max_new_tokens=4, gamma=4)


@pytest.mark.slow
def test_sampling_reproduces_the_targets_distribution(tmp_path):
    """Chi-square over all 64 two-token continuations, against exact probabilities."""
    reference = random_reference(SMALL_CONFIG, seed=11)
    draft_reference = perturbed_copy(reference, sigma=0.05, seed=12)
    vocab = SMALL_CONFIG.vocab_size
    paths = {}
    for name, model in (("target", reference), ("draft", draft_reference)):
        path = tmp_path / f"{name}.sdm"
        write_model(path, model.state_dict(), model.config, weight_format="q8")
        paths[name] = path

    target = cpp.Model(str(paths["target"]), max_positions=64, max_batch=5)
    draft = cpp.Model(str(paths["draft"]), max_positions=64, max_batch=5)
    prompt = prompt_tokens(3, vocab, seed=13)

    # Exact joint probabilities, from the engine's own logits so quantization is accounted for.
    config = SamplingConfig(temperature=1.0)
    first_logits = target.forward(np.array(prompt, dtype=np.int32))[0]
    first = warp_probs(torch.from_numpy(first_logits), config).numpy()
    exact = np.empty((vocab, vocab))
    for a in range(vocab):
        extended = np.array(prompt + [a], dtype=np.int32)
        target.reset()
        second_logits = target.forward(extended)[0]
        second = warp_probs(torch.from_numpy(second_logits), config).numpy()
        exact[a] = first[a] * second

    trials = 3000
    counts = np.zeros((vocab, vocab))
    for trial in range(trials):
        got, _ = cpp.generate_speculative(
            target, draft, prompt, max_new_tokens=2, gamma=3, temperature=1.0, seed=trial
        )
        counts[got[0], got[1]] += 1

    pvalue = chi_square_pvalue(counts.flatten(), (exact * trials).flatten())
    assert pvalue > 1e-3, f"the engine's sampling does not match the target (p={pvalue:.2e})"


# --------------------------------------------- prompt lookup and early stopping


def test_prompt_lookup_matches_plain_decoding(paths):
    """Copied guesses carry no distribution, so they are treated as a point mass. The output
    must still be exactly what plain decoding produces."""
    target = cpp.Model(str(paths["target"]), max_positions=MAX_POSITIONS, max_batch=5)
    # A prompt with repeated text, which is where copying pays.
    phrase = prompt_tokens(6, TINY_CONFIG.vocab_size, seed=20)
    prompt = phrase + phrase + phrase[:3]

    expected, plain_stats = cpp.generate_plain(target, prompt, max_new_tokens=30)
    got, lookup_stats = cpp.generate_prompt_lookup(
        target, prompt, max_new_tokens=30, gamma=4, max_ngram=3
    )

    assert got == expected
    assert lookup_stats["draft_forwards"] == 0, "copying must cost no model work at all"
    assert lookup_stats["proposed"] > 0, "a repeating prompt should give it something to copy"
    assert lookup_stats["target_forwards"] <= plain_stats["target_forwards"]


def test_prompt_lookup_falls_back_when_there_is_nothing_to_copy(paths):
    """With no repeated n-gram the drafter proposes nothing, and each round must then behave
    exactly like an ordinary decoding step."""
    target = cpp.Model(str(paths["target"]), max_positions=MAX_POSITIONS, max_batch=5)
    prompt = [3]

    got, stats = cpp.generate_prompt_lookup(
        target, prompt, max_new_tokens=6, gamma=4, max_ngram=8
    )
    assert got == cpp.generate_plain(target, prompt, max_new_tokens=6)[0]
    assert stats["emitted"] == 6
    assert stats["tokens_per_target_forward"] <= 2.0


@pytest.mark.slow
def test_prompt_lookup_sampling_is_still_exact(tmp_path):
    """The point-mass treatment has to leave the distribution alone, not just the greedy path."""
    reference = random_reference(SMALL_CONFIG, seed=21)
    vocab = SMALL_CONFIG.vocab_size
    path = tmp_path / "target.sdm"
    write_model(path, reference.state_dict(), reference.config, weight_format="q8")
    target = cpp.Model(str(path), max_positions=64, max_batch=5)

    phrase = prompt_tokens(4, vocab, seed=22)
    prompt = phrase + phrase
    config = SamplingConfig(temperature=1.0)
    first = warp_probs(torch.from_numpy(target.forward(np.array(prompt, dtype=np.int32))[0]), config).numpy()
    exact = np.empty((vocab, vocab))
    for a in range(vocab):
        target.reset()
        row = target.forward(np.array(prompt + [a], dtype=np.int32))[0]
        exact[a] = first[a] * warp_probs(torch.from_numpy(row), config).numpy()

    trials = 3000
    counts = np.zeros((vocab, vocab))
    for trial in range(trials):
        got, _ = cpp.generate_prompt_lookup(
            target, prompt, max_new_tokens=2, gamma=3, temperature=1.0, seed=trial
        )
        counts[got[0], got[1]] += 1

    pvalue = chi_square_pvalue(counts.flatten(), (exact * trials).flatten())
    assert pvalue > 1e-3, f"prompt lookup changed the distribution (p={pvalue:.2e})"


@pytest.mark.parametrize("threshold", [0.0, 0.3, 0.95])
def test_confidence_threshold_keeps_output_exact(paths, threshold):
    """Giving up early changes how many guesses are offered, never what comes out."""
    target, draft = open_pair(paths)
    prompt = prompt_tokens(5, TINY_CONFIG.vocab_size, seed=23)
    expected, _ = cpp.generate_plain(target, prompt, max_new_tokens=24)

    got, stats = cpp.generate_speculative(
        target, draft, prompt, max_new_tokens=24, gamma=6, confidence_threshold=threshold
    )
    assert got == expected
    assert stats["proposed"] <= 6 * stats["rounds"]


def test_a_high_threshold_stops_drafting_altogether(paths):
    target, draft = open_pair(paths)
    prompt = prompt_tokens(4, TINY_CONFIG.vocab_size, seed=24)
    # Nothing can clear a threshold above one, so no guesses should be offered.
    _, stats = cpp.generate_speculative(
        target, draft, prompt, max_new_tokens=8, gamma=4, confidence_threshold=1.0
    )
    assert stats["proposed"] == 0
    assert stats["tokens_per_target_forward"] == pytest.approx(1.0)


def test_invalid_threshold_is_rejected(paths):
    target, draft = open_pair(paths)
    with pytest.raises(Exception):
        cpp.generate_speculative(target, draft, [1, 2], max_new_tokens=4, confidence_threshold=1.5)
