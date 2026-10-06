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


def test_the_rule_asks_for_each_row_once_and_only_as_far_as_it_reads():
    """The target's rows are warped on demand, so the rule's requests are the contract.

    Each row has to be asked for exactly once and in order: a second request would warp probabilities
    rather than logits, and a skipped one would leave the rule reading raw logits as if they were a
    distribution. The rule stops at the first rejection, so it asks for rows 0 through `accepted`,
    and for one more only when every guess survived and a bonus token is drawn.
    """
    vocab = 8
    certain = np.full((3, vocab), 0.0, dtype=np.float32)
    certain[:, 1] = 1.0  # the target is sure of token 1 everywhere
    q = np.zeros((2, vocab), dtype=np.float32)
    q[:, 1] = 1.0

    accepted, _, rows = cpp.accept_or_resample_rows_read(certain, q, [1, 1], seed=4)
    assert accepted == 2  # both guesses match what the target wanted
    assert rows == [0, 1, 2]  # and the bonus row after them

    accepted, _, rows = cpp.accept_or_resample_rows_read(certain, q, [1, 7], seed=4)
    assert accepted == 1  # the second guess cannot be accepted: the target gives it no mass
    assert rows == [0, 1]  # so the row after the rejection is never asked for

    accepted, _, rows = cpp.accept_or_resample_rows_read(certain, q, [7, 1], seed=4)
    assert accepted == 0
    assert rows == [0]


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


# ------------------------------------------------------- the float kernels underneath


def test_the_vectorized_exp_is_accurate_to_under_one_ulp():
    """Both softmaxes run on this, so its error lands in the probabilities and in the logits.

    The tolerances it has to meet are 1e-6 against the Python sampling oracle and 1e-5 against the
    PyTorch twin; a float32 ulp is 1.2e-7, and this stays inside that over the whole range.
    """
    rng = np.random.default_rng(0)
    values = np.concatenate([rng.uniform(-87.0, 87.0, 50_000), rng.uniform(-1.0, 1.0, 10_000),
                             [0.0, 1.0, -1.0]]).astype(np.float32)
    mine = cpp.vector_exp(values).astype(np.float64)
    reference = np.exp(values.astype(np.float64))
    relative = np.abs(mine - reference) / reference
    assert relative.max() < 1.2e-7, f"worst relative error {relative.max():.3e}"
    assert cpp.vector_exp(np.array([0.0], dtype=np.float32))[0] == 1.0


def test_the_vectorized_exp_returns_zero_rather_than_a_denormal():
    """Top-k marks the tokens it drops with -inf, and they have to come out of the softmax as zero.

    A denormal would do no numerical harm and would still be wrong: "exactly k tokens carry mass" is
    what top-k means, and the test for it counts nonzero entries. The clamp is at -88, and just
    above it the result is zero as well, because 2^n is assembled from the exponent field and n
    rounds to -127 there. In a softmax both are terms 1e-38 the size of the largest.
    """
    values = np.array([-np.inf, -1000.0, -89.0, -88.0], dtype=np.float32)
    assert list(cpp.vector_exp(values)) == [0.0, 0.0, 0.0, 0.0]

    normal = np.array([-87.0, -80.0, -1.0], dtype=np.float32)  # all representable as normals
    assert (cpp.vector_exp(normal) > 0.0).all()
    np.testing.assert_allclose(cpp.vector_exp(normal), np.exp(normal.astype(np.float64)),
                               rtol=1.2e-7)


def test_the_vectorized_softmax_matches_a_float64_one_over_a_whole_vocabulary():
    """Summed in float32 the total came to 1.00003 over 151,936 entries, a 3e-5 bias on every row."""
    rng = np.random.default_rng(1)
    logits = (rng.standard_normal(151_936) * 4.0).astype(np.float32)
    mine = cpp.vector_softmax(logits).astype(np.float64)
    exponentials = np.exp(logits.astype(np.float64) - logits.max())
    reference = exponentials / exponentials.sum()
    assert np.abs(mine - reference).max() < 1e-7
    assert mine.sum() == pytest.approx(1.0, abs=1e-6)

    masked = logits.copy()
    masked[:-50] = -np.inf  # what top-k leaves behind
    kept = cpp.vector_softmax(masked)
    assert int((kept > 0).sum()) == 50
    assert float(kept.sum()) == pytest.approx(1.0, abs=1e-6)


@pytest.mark.parametrize("group", [1, 2, 4, 8])
@pytest.mark.parametrize("n", [8, 16, 128, 129, 136])
def test_attentions_grouped_kernels_are_bit_identical_to_the_per_head_ones(group, n):
    """A cached row is read by every query head in its group, so it is converted once for all of them.

    That is a performance change and nothing else, and this is what says so: the same numbers to the
    last bit as converting the row into a buffer and calling dot_f32 and accumulate_scaled per head,
    which is what the engine did before. Two accumulators in sixteen-element steps in both, and
    fp16 to fp32 is exact, so there is no tolerance to argue about.
    """
    out = cpp.check_group_kernels(n=n, group=group, seed=n * 31 + group)
    assert np.array_equal(np.array(out["grouped_dots"]), np.array(out["reference_dots"]))
    assert np.array_equal(np.array(out["grouped_sum"]), np.array(out["reference_sum"]))


# ------------------------------------------------------------------------------ argmax


@pytest.mark.parametrize("n", [1, 2, 7, 8, 9, 15, 16, 17, 31, 32, 33, 1023, 151936])
def test_the_two_argmax_scans_agree(n):
    """The vectorized scan has to choose the same index as the scalar one it replaced.

    This is not a numerical approximation anywhere: the index is a token id, so a disagreement is a
    different word, and the tie-break decides it whenever two logits land on the same float -- which
    they do, a quantized output projection producing plenty of exact ties. Lengths either side of a
    multiple of eight are where a vectorized scan gets its tail wrong, so all of them are covered.
    """
    rng = np.random.default_rng(n)
    rows = [
        rng.standard_normal(n).astype(np.float32),
        np.zeros(n, dtype=np.float32),  # every value tied, so the first index must win
        rng.integers(0, 3, size=n).astype(np.float32),  # ties in bulk
        np.full(n, -np.inf, dtype=np.float32),
    ]
    tail = rng.standard_normal(n).astype(np.float32)
    tail[n - 1] = np.inf  # the largest value in the last lane the tail loop touches
    rows.append(tail)
    for row in rows:
        assert cpp.argmax(row) == cpp.argmax_scalar(row) == int(np.argmax(row))


def test_a_nan_logit_is_skipped_rather_than_latched():
    """The one place the vectorized scan differs from the scalar one, deliberately.

    ``x > NaN`` is false for every x, so the scalar scan's running best could never move off index 0
    once a NaN sat there: one NaN in the first position swallowed the whole row, while a NaN anywhere
    else was ignored. The vectorized lanes start at -inf, so a NaN never wins a lane and the scan
    looks past it wherever it is. Neither matches numpy, which calls NaN the maximum. A NaN logit is a
    bug upstream either way; what is pinned here is that the engine ignores one unless there is
    nothing else to choose.
    """
    row = np.arange(32, dtype=np.float32)
    row[0] = np.nan
    assert cpp.argmax(row) == 31
    assert cpp.argmax_scalar(row) == 0  # what it used to answer
    every = np.full(16, np.nan, dtype=np.float32)
    assert cpp.argmax(every) == cpp.argmax_scalar(every) == 0


@pytest.mark.parametrize("variant", ["vector", "scalar"])
def test_the_argmax_benchmark_finds_what_it_planted(variant):
    """A benchmark whose answer nobody checks can happily time a scan that stops early."""
    result = cpp.bench_argmax(n=4096, iters=2, variant=variant)
    assert result["best"] == result["planted"]
    assert result["seconds"] > 0.0
    assert result["elements_per_second"] > 0.0


def test_the_argmax_benchmark_rejects_an_unknown_variant():
    with pytest.raises(ValueError):
        cpp.bench_argmax(n=64, iters=1, variant="sideways")


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


def test_stats_separate_the_prompt_pass_from_the_rounds(paths):
    """The per-round overhead term o is wall time the models did not spend computing, over rounds.

    A prompt pass is neither a round nor free, so it has to come out of both sides or it inflates o.
    Measured on the 4B it happens not to -- a prefill's wall time equals its model time to the
    microsecond, so the two corrections cancel -- but that is a fact about this engine rather than an
    identity, and the arithmetic below is what notices if it stops holding.
    """
    target, draft = open_pair(paths, gamma=3)
    for one in (target, draft):
        one.set_timing(True)
        one.reset_timings()

    _, stats = cpp.generate_speculative(target, draft, [3, 4, 5, 6, 7], max_new_tokens=8, gamma=3)

    assert stats["prefill_seconds"] > 0.0  # a five-token prompt does prefill
    assert stats["prefill_model_seconds"] > 0.0  # and timing was on, so it is attributed
    assert stats["prefill_model_seconds"] <= stats["prefill_seconds"] + 1e-9
    assert stats["rounds_seconds"] == pytest.approx(stats["seconds"] - stats["prefill_seconds"],
                                                    rel=1e-12)
    # What is left for the rounds must be positive, or o comes out negative and nonsensical.
    model_total = target.timings()["total"] + draft.timings()["total"]
    assert model_total - stats["prefill_model_seconds"] > 0.0
    assert stats["rounds_seconds"] > 0.0


def test_plain_decoding_reports_its_prompt_pass_too(paths):
    """The baseline has to be measured the same way as the thing it is a baseline for.

    Without this, a benchmark comparing decode-only speed against the baseline's whole-call speed
    reported speculative decoding at 9.91x on a summarization prompt, where the honest figure is near
    1: the baseline's prefill was being counted as decoding and the other method's was not.
    """
    target, _ = open_pair(paths, gamma=2)
    target.set_timing(True)
    target.reset_timings()
    _, stats = cpp.generate_plain(target, [3, 4, 5, 6, 7], max_new_tokens=6)
    assert stats["prefill_seconds"] > 0.0
    assert stats["prefill_model_seconds"] > 0.0
    assert stats["prefill_seconds"] < stats["seconds"]
    assert stats["rounds_seconds"] == pytest.approx(stats["seconds"] - stats["prefill_seconds"],
                                                    rel=1e-12)


def test_a_prompt_of_one_token_has_no_prefill(paths):
    """With nothing to prefill the fields stay zero rather than picking up the first round."""
    target, draft = open_pair(paths, gamma=2)
    _, stats = cpp.generate_speculative(target, draft, [3], max_new_tokens=4, gamma=2)
    assert stats["prefill_seconds"] == 0.0
    assert stats["prefill_model_seconds"] == 0.0
    assert stats["rounds_seconds"] == stats["seconds"]


def test_the_round_loop_accounts_for_every_section_of_its_own_time(paths):
    """o used to be one lump of wall time the models could not explain, and guesses at what was in it
    were wrong three times over (plan section 10.4), so the loop reports its own sections instead.

    The four add up to the round loop exactly, by construction: three timed sections plus whatever is
    left. That is what makes the leftover worth reading -- a residual that is merely small has nowhere
    for a mistake to hide, while a lump measured as a subtraction has room for all of them.
    """
    target, draft = open_pair(paths, gamma=3)
    for one in (target, draft):
        one.set_timing(True)
        one.reset_timings()

    _, stats = cpp.generate_speculative(target, draft, [3, 4, 5, 6, 7], max_new_tokens=12, gamma=3)

    for key in ("propose_seconds", "draft_forward_seconds", "verify_seconds", "accept_seconds"):
        assert stats[key] > 0.0, key
    # The draft's forward calls happen inside propose, so one bounds the other and the difference is
    # the draft's own sampling.
    assert stats["draft_forward_seconds"] <= stats["propose_seconds"]
    assert stats["draft_sampling_seconds"] == pytest.approx(
        stats["propose_seconds"] - stats["draft_forward_seconds"], rel=1e-12
    )
    # The sections and the leftover are the whole round loop, to the nanosecond.
    parts = (stats["propose_seconds"] + stats["verify_seconds"] + stats["accept_seconds"]
             + stats["bookkeeping_seconds"])
    assert parts == pytest.approx(stats["rounds_seconds"], rel=1e-12)
    # Appending tokens, the stop check and rewinding two caches, against two models' forward passes:
    # a leftover that stops being small means something untimed grew.
    assert 0.0 <= stats["bookkeeping_seconds"] < 0.5 * stats["rounds_seconds"]
    # And the models' compute has to fit inside the forward calls that were timed around it.
    model_total = target.timings()["total"] + draft.timings()["total"]
    inside_rounds = model_total - stats["prefill_model_seconds"]
    assert inside_rounds <= stats["draft_forward_seconds"] + stats["verify_seconds"] + 1e-9


def test_a_drafter_that_runs_no_model_reports_no_forward_time(paths):
    """Prompt lookup copies tokens, so all of propose is sampling and none of it is a forward pass."""
    target, _ = open_pair(paths, gamma=3)
    _, stats = cpp.generate_prompt_lookup(target, [3, 4, 5, 3, 4, 5], max_new_tokens=8, gamma=3)
    assert stats["draft_forward_seconds"] == 0.0
    assert stats["draft_sampling_seconds"] == stats["propose_seconds"]
    assert stats["verify_seconds"] > 0.0
