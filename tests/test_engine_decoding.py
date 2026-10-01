"""Speculative decoding running on the C++ engine.

The same round loop, acceptance rule and metrics that were tested against the PyTorch
reference now drive the engine, so the correctness properties carry over: greedy
speculative decoding must reproduce plain greedy decoding from the engine exactly, and the
offline simulator must predict the engine's tokens-per-step exactly.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from specdraft.engine import EngineModel  # noqa: E402
from specdraft.export import write_model  # noqa: E402
from specdraft.offline import score_sequence, simulate  # noqa: E402
from specdraft.reference import TINY_CONFIG, perturbed_copy, random_reference  # noqa: E402
from specdraft.sampling import GREEDY, SamplingConfig  # noqa: E402
from specdraft.speculative import plain_generate, speculative_generate  # noqa: E402

MAX_POSITIONS = 96


@pytest.fixture(scope="module")
def models(tmp_path_factory):
    """An engine target and an engine draft that agrees with it often but not always."""
    directory = tmp_path_factory.mktemp("engine_decoding")
    target_reference = random_reference(TINY_CONFIG, seed=0)
    draft_reference = perturbed_copy(target_reference, sigma=0.02, seed=1)

    paths = {}
    for name, model in (("target", target_reference), ("draft", draft_reference)):
        path = directory / f"{name}.sdm"
        write_model(path, model.state_dict(), model.config, weight_format="q8")
        paths[name] = path
    return (
        EngineModel(paths["target"], max_positions=MAX_POSITIONS),
        EngineModel(paths["draft"], max_positions=MAX_POSITIONS),
    )


def prompt(n: int, seed: int) -> "torch.Tensor":
    generator = torch.Generator().manual_seed(seed)
    return torch.randint(0, TINY_CONFIG.vocab_size, (n,), generator=generator)


def test_engine_model_reports_its_config(models):
    target, _ = models
    assert target.config["hidden_size"] == TINY_CONFIG.hidden_size
    assert target.vocab_limit == TINY_CONFIG.vocab_size
    assert "target.sdm" in repr(target)


def test_cache_handle_tracks_and_rewinds_the_engine(models):
    target, _ = models
    cache = target.new_cache(MAX_POSITIONS)
    assert cache.pos == 0
    target.forward(prompt(5, seed=2), cache=cache)
    assert cache.pos == 5 == target.pos
    cache.rewind_to(3)
    assert cache.pos == 3
    cache.rewind_to(9)  # never forward
    assert cache.pos == 3


def test_asking_for_too_many_positions_is_refused(models):
    target, _ = models
    with pytest.raises(ValueError):
        target.new_cache(MAX_POSITIONS + 1)


@pytest.mark.parametrize("gamma", [1, 2, 4, 6])
def test_greedy_speculative_decoding_on_the_engine_is_exact(models, gamma):
    """Bit-exact verification in the engine makes this an equality, not an approximation."""
    target, draft = models
    ids = prompt(5, seed=3)

    expected, plain_stats = plain_generate(target, ids, 40)
    got, spec_stats = speculative_generate(target, draft, ids, 40, gamma=gamma)

    assert got == expected
    assert spec_stats.target_forwards < plain_stats.target_forwards
    assert spec_stats.tokens_per_target_forward > 1.0


def test_a_draft_identical_to_the_target_accepts_everything(models):
    """Each role needs its own instance: an EngineModel owns one KV cache."""
    target, _ = models
    twin_of_target = EngineModel(target.path, max_positions=MAX_POSITIONS)
    ids = prompt(4, seed=4)

    got, stats = speculative_generate(target, twin_of_target, ids, 25, gamma=4)

    assert got == plain_generate(target, ids, 25)[0]
    assert stats.alpha == 1.0


def test_one_instance_cannot_play_both_roles(models):
    target, _ = models
    with pytest.raises(ValueError, match="one KV cache"):
        speculative_generate(target, target, prompt(4, seed=9), 5, gamma=2)


def test_sampling_runs_and_respects_the_token_budget(models):
    target, draft = models
    ids = prompt(4, seed=5)
    generator = torch.Generator().manual_seed(6)
    got, stats = speculative_generate(
        target, draft, ids, 20, gamma=3, config=SamplingConfig(temperature=1.0), generator=generator
    )
    assert len(got) == 20 == stats.emitted
    assert all(0 <= token < TINY_CONFIG.vocab_size for token in got)


def test_offline_simulation_matches_the_engine_exactly(models):
    """The engine scores its own text, so no twin is involved and nothing is approximated."""
    target, draft = models
    ids = prompt(6, seed=7)
    gamma = 4

    generated, online = speculative_generate(target, draft, ids, 50, gamma=gamma)

    tokens = torch.cat([ids, torch.tensor(generated)])
    metrics = score_sequence(target, draft, tokens, response_start=len(ids), config=GREEDY)
    assert metrics.is_greedy_reference
    offline_tau, lengths = simulate(metrics.accept_prob(greedy=True), gamma=gamma)

    assert offline_tau == pytest.approx(online.tokens_per_target_forward)
    assert np.array_equal(lengths, np.bincount(online.accepted_lengths, minlength=gamma + 1))


def test_hidden_states_are_not_available(models):
    target, _ = models
    with pytest.raises(NotImplementedError):
        target.forward(prompt(2, seed=8), hidden_only=True)
