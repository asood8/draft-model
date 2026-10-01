"""Layer pruning (plan §8.1).

The sharpest test available here is to build a model with a layer that provably does nothing —
its output projections zeroed, so it returns its input untouched — and check two things: that the
influence score notices, and that removing it leaves the model's output *identical*. Both together
show the scoring and the surgery agree about what a layer contributes.
"""

from __future__ import annotations

import dataclasses

import pytest

torch = pytest.importorskip("torch")

from specdraft.prune import (  # noqa: E402
    block_influence,
    bytes_per_token,
    choose_layers_to_keep,
    describe_pruning,
    prune_layers,
    prune_to_size,
)
from specdraft.reference import TINY_CONFIG, Qwen3Reference, random_reference  # noqa: E402

CONFIG = dataclasses.replace(TINY_CONFIG, num_hidden_layers=6)


def calibration(count: int = 3, length: int = 12, seed: int = 0) -> list["torch.Tensor"]:
    generator = torch.Generator().manual_seed(seed)
    return [
        torch.randint(0, CONFIG.vocab_size, (length,), generator=generator) for _ in range(count)
    ]


def model_with_dead_layers(dead: set[int], seed: int = 0) -> Qwen3Reference:
    """A model where the named layers return their input unchanged."""
    model = random_reference(CONFIG, seed=seed)
    state = model.state_dict()
    for index in dead:
        # Zeroing what each sub-block writes back into the residual stream makes the layer a
        # no-op, whatever its other weights are.
        state[f"model.layers.{index}.self_attn.o_proj.weight"] = torch.zeros_like(
            state[f"model.layers.{index}.self_attn.o_proj.weight"]
        )
        state[f"model.layers.{index}.mlp.down_proj.weight"] = torch.zeros_like(
            state[f"model.layers.{index}.mlp.down_proj.weight"]
        )
    return Qwen3Reference(CONFIG, state)


# ------------------------------------------------------------------------------ scoring


def test_influence_scores_one_value_per_layer():
    scores = block_influence(random_reference(CONFIG, seed=1), calibration())
    assert len(scores) == CONFIG.num_hidden_layers
    assert all(score >= 0 and score == score for score in scores)  # finite, non-negative


def test_a_layer_that_does_nothing_scores_zero():
    model = model_with_dead_layers({1, 3})
    scores = block_influence(model, calibration())
    assert scores[1] == pytest.approx(0.0, abs=1e-6)
    assert scores[3] == pytest.approx(0.0, abs=1e-6)
    assert max(scores) > 1e-4, "the live layers should score above the dead ones"


def test_empty_calibration_is_rejected():
    with pytest.raises(ValueError):
        block_influence(random_reference(CONFIG, seed=2), [])


# ------------------------------------------------------------------------------ choosing


def test_the_least_influential_layers_go():
    scores = [0.5, 0.0, 0.4, 0.0, 0.3, 0.2]
    assert choose_layers_to_keep(scores, keep=4) == [0, 2, 4, 5]


def test_the_last_layer_is_protected():
    scores = [0.9, 0.8, 0.7, 0.6, 0.5, 0.0]  # the final layer scores worst
    assert 5 in choose_layers_to_keep(scores, keep=3, protect_last=True)
    assert 5 not in choose_layers_to_keep(scores, keep=3, protect_last=False)


def test_keeping_everything_changes_nothing():
    scores = [0.1] * 6
    assert choose_layers_to_keep(scores, keep=6) == list(range(6))


def test_impossible_sizes_are_rejected():
    with pytest.raises(ValueError):
        choose_layers_to_keep([0.1, 0.2], keep=0)
    with pytest.raises(ValueError):
        choose_layers_to_keep([0.1, 0.2], keep=3)


# ------------------------------------------------------------------------------- surgery


def test_removing_dead_layers_leaves_the_output_identical():
    """If the dropped layers really did nothing, the pruned model must agree exactly."""
    model = model_with_dead_layers({2, 4})
    tokens = torch.tensor([3, 8, 1, 9, 5])
    with torch.no_grad():
        before = model.forward(tokens)

    pruned = prune_layers(model, [0, 1, 3, 5])
    assert pruned.config.num_hidden_layers == 4
    with torch.no_grad():
        after = pruned.forward(tokens)
    torch.testing.assert_close(after, before, atol=1e-5, rtol=1e-4)


def test_pruning_keeps_the_right_weights():
    model = random_reference(CONFIG, seed=3)
    pruned = prune_layers(model, [1, 4])
    torch.testing.assert_close(pruned.layers[0].q_proj, model.layers[1].q_proj)
    torch.testing.assert_close(pruned.layers[1].q_proj, model.layers[4].q_proj)
    torch.testing.assert_close(pruned.final_norm, model.final_norm)
    assert pruned.lm_head is pruned.embed_tokens  # ties survive


def test_pruned_models_still_decode():
    from specdraft.speculative import plain_generate

    model = prune_layers(random_reference(CONFIG, seed=4), [0, 2, 5])
    generated, stats = plain_generate(model, torch.tensor([1, 2, 3]), 6)
    assert len(generated) == 6 == stats.emitted


def test_bad_indices_are_rejected():
    model = random_reference(CONFIG, seed=5)
    for indices in ([], [2, 1], [0, 0], [0, CONFIG.num_hidden_layers]):
        with pytest.raises(ValueError):
            prune_layers(model, indices)


def test_scoring_and_surgery_together_find_the_dead_layers():
    model = model_with_dead_layers({0, 3})
    pruned, kept, scores = prune_to_size(model, keep=4, sequences=calibration())
    assert 0 not in kept and 3 not in kept, f"kept {kept} with scores {scores}"
    assert pruned.config.num_hidden_layers == 4

    tokens = torch.tensor([4, 7, 2])
    with torch.no_grad():
        torch.testing.assert_close(pruned.forward(tokens), model.forward(tokens),
                                   atol=1e-5, rtol=1e-4)


# -------------------------------------------------------------------------- what it buys


def test_bytes_per_token_falls_with_the_layer_count():
    half = dataclasses.replace(CONFIG, num_hidden_layers=3)
    assert bytes_per_token(half) < bytes_per_token(CONFIG)

    summary = describe_pruning(CONFIG, half)
    assert summary["layers_after"] == 3
    assert 0.0 < summary["bytes_ratio"] < 1.0


def test_the_output_layer_limits_what_pruning_can_save():
    """With a 152k-token vocabulary the output layer is a large fixed cost, which is why
    vocabulary trimming is the companion to pruning rather than an alternative."""
    big_vocab = dataclasses.replace(CONFIG, vocab_size=151_936, num_hidden_layers=6)
    one_layer = dataclasses.replace(big_vocab, num_hidden_layers=1)
    assert describe_pruning(big_vocab, one_layer)["bytes_ratio"] > 0.3


def test_saving_a_pruned_model_round_trips(tmp_path):
    from specdraft.prune import save_pruned

    model = prune_layers(random_reference(CONFIG, seed=6), [0, 1, 2])
    save_pruned(tmp_path / "pruned", model, extra={"kept": [0, 1, 2]})
    reloaded = Qwen3Reference.from_pretrained(tmp_path / "pruned")

    assert reloaded.config.num_hidden_layers == 3
    tokens = torch.tensor([2, 5, 8])
    with torch.no_grad():
        torch.testing.assert_close(reloaded.forward(tokens), model.forward(tokens))
