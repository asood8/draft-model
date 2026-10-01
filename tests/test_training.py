"""The distillation loop, end to end on tiny models (plan §11.5).

The test that matters is the last one: train a small draft toward a target and check that the
*acceptance* metric rises, not merely that the loss falls. Acceptance is what the engine
converts into speed, and a pipeline that reduces its loss without moving acceptance would be
quietly useless.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from specdraft.losses import LOSSES  # noqa: E402
from specdraft.reference import TINY_CONFIG, perturbed_copy, random_reference  # noqa: E402
from specdraft.train import (  # noqa: E402
    TrainConfig,
    TrainedSequence,
    build_optimizer,
    learning_rate_at,
    load_checkpoint,
    save_checkpoint,
    teacher_from_model,
    train,
    validate,
)
from specdraft.trainable import TrainableDraft  # noqa: E402


@pytest.fixture(scope="module")
def target():
    return random_reference(TINY_CONFIG, seed=0)


def make_sequences(count: int, length: int, seed: int) -> list[TrainedSequence]:
    generator = torch.Generator().manual_seed(seed)
    out = []
    for i in range(count):
        tokens = torch.randint(0, TINY_CONFIG.vocab_size, (length,), generator=generator)
        out.append(TrainedSequence(tokens.tolist(), response_start=length // 3, source="test"))
    return out


# ------------------------------------------------------------------ the trainable wrapper


def test_gradients_reach_every_weight(target):
    student = TrainableDraft.from_reference(target)
    tokens = torch.tensor([1, 2, 3, 4])
    student.logits(tokens).sum().backward()

    missing = [name for name, p in student.weights.items() if p.requires_grad and p.grad is None]
    assert not missing, f"no gradient reached {missing[:3]}"
    trainable, total = student.parameter_count()
    assert trainable == total


def test_freezing_the_embedding_leaves_it_alone(target):
    student = TrainableDraft.from_reference(target, freeze_embeddings=True)
    trainable, total = student.parameter_count()
    assert trainable < total
    assert not student.weights["model|embed_tokens|weight"].requires_grad

    before = student.weights["model|embed_tokens|weight"].detach().clone()
    student.logits(torch.tensor([1, 2, 3])).sum().backward()
    assert student.weights["model|embed_tokens|weight"].grad is None
    torch.testing.assert_close(student.weights["model|embed_tokens|weight"], before)


def test_the_wrapper_computes_what_the_reference_computes(target):
    """Training must not be happening to a different model than the engine will run."""
    student = TrainableDraft.from_reference(target)
    tokens = torch.tensor([5, 9, 2, 7])
    with torch.no_grad():
        torch.testing.assert_close(student.logits(tokens), target.forward(tokens))
        torch.testing.assert_close(student.as_reference().forward(tokens), target.forward(tokens))


# ----------------------------------------------------------------------- the schedule


def test_learning_rate_warms_up_then_decays():
    config = TrainConfig(learning_rate=1e-3, warmup_fraction=0.1)
    values = [learning_rate_at(step, 100, config) for step in range(100)]
    assert values[0] < values[9] <= config.learning_rate
    assert values[9] > values[50] > values[99]
    assert values[99] < config.learning_rate * 0.05


def test_bad_config_is_rejected():
    with pytest.raises(ValueError):
        TrainConfig(loss="nonsense")
    with pytest.raises(ValueError):
        TrainConfig(tokens_per_step=0)
    with pytest.raises(ValueError):
        TrainedSequence([1, 2, 3], response_start=0)
    with pytest.raises(ValueError):
        TrainedSequence([1, 2, 3], response_start=3)


# ---------------------------------------------------------------------- the loop runs


@pytest.mark.parametrize("loss", LOSSES)
def test_every_loss_runs_and_updates_the_draft(target, loss):
    student = TrainableDraft.from_reference(perturbed_copy(target, sigma=0.05, seed=1))
    before = student.weights["model|layers|0|mlp|gate_proj|weight"].detach().clone()
    config = TrainConfig(
        loss=loss, learning_rate=1e-3, tokens_per_step=64, max_response_tokens=256, loss_chunk=16
    )

    state = train(student, teacher_from_model(target), make_sequences(8, 24, seed=2), config)

    assert state.step > 0
    assert state.response_tokens >= 256 or state.sequences == 8
    assert state.history and all("loss" in entry for entry in state.history)
    assert not torch.equal(before, student.weights["model|layers|0|mlp|gate_proj|weight"])


def test_the_token_budget_is_respected(target):
    student = TrainableDraft.from_reference(target)
    config = TrainConfig(tokens_per_step=32, max_response_tokens=96, learning_rate=0.0)
    state = train(student, teacher_from_model(target), make_sequences(100, 20, seed=3), config)
    assert state.response_tokens >= 96
    assert state.response_tokens < 96 + 20  # stops as soon as the budget is met


def test_validation_reports_acceptance(target):
    student = TrainableDraft.from_reference(target)
    metrics = validate(
        student, teacher_from_model(target), make_sequences(3, 20, seed=4), TrainConfig()
    )
    # A copy of the target agrees with it everywhere, so both numbers are at their maximum.
    assert metrics["greedy_top1_match"] == pytest.approx(1.0)
    assert metrics["sampling_acceptance"] == pytest.approx(1.0, abs=1e-4)
    assert metrics["positions"] > 0


def test_checkpoint_round_trip(tmp_path, target):
    student = TrainableDraft.from_reference(perturbed_copy(target, sigma=0.05, seed=5))
    config = TrainConfig(tokens_per_step=32, max_response_tokens=64, learning_rate=1e-3)
    optimizer = build_optimizer(student, config)
    state = train(
        student, teacher_from_model(target), make_sequences(4, 24, seed=6), config,
        optimizer=optimizer, checkpoint_dir=tmp_path,
    )
    assert (tmp_path / "checkpoint.pt").is_file()
    assert (tmp_path / "state.json").is_file()

    fresh = TrainableDraft.from_reference(target)
    restored = load_checkpoint(tmp_path, fresh)
    assert restored.step == state.step
    assert restored.response_tokens == state.response_tokens
    for key, parameter in student.weights.items():
        torch.testing.assert_close(parameter.detach(), fresh.weights[key].detach())


# ----------------------------------------------------------- does it actually help?


@pytest.mark.slow
@pytest.mark.parametrize("loss", ["fkl", "tvd"])
def test_training_raises_acceptance(target, loss):
    """The point of the exercise: the draft should come to agree with the target more often."""
    drifted = perturbed_copy(target, sigma=0.08, seed=7)
    student = TrainableDraft.from_reference(drifted)
    teacher = teacher_from_model(target)

    # Train on the target's own greedy continuations, which is the text the draft will have to
    # predict while decoding.
    from specdraft.speculative import plain_generate

    sequences = []
    for seed in range(16):
        prompt = torch.randint(
            0, TINY_CONFIG.vocab_size, (6,), generator=torch.Generator().manual_seed(seed)
        )
        generated, _ = plain_generate(target, prompt, 18)
        sequences.append(
            TrainedSequence(prompt.tolist() + generated, response_start=len(prompt))
        )
    held_out, training = sequences[:6], sequences[6:]

    config = TrainConfig(
        loss=loss, learning_rate=3e-3, tokens_per_step=64, max_response_tokens=4096,
        loss_chunk=32,
    )
    before = validate(student, teacher, held_out, config)
    train(student, teacher, training * 12, config)
    after = validate(student, teacher, held_out, config)

    assert after["sampling_acceptance"] > before["sampling_acceptance"], (
        f"{loss}: acceptance fell from {before['sampling_acceptance']:.3f} to "
        f"{after['sampling_acceptance']:.3f}"
    )
    # Top-1 agreement is a discrete statistic over a few hundred positions of a toy model, and
    # these losses optimize the whole distribution rather than the argmax, so it is allowed to
    # wobble; it must not collapse. On real models it is reported alongside, not asserted here.
    assert after["greedy_top1_match"] > before["greedy_top1_match"] - 0.2
