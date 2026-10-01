"""The distillation losses (plan §11.4).

The one that has to be exactly right is TVD: it equals one minus the expected acceptance rate
when sampling, so it is the direct link between what training minimizes and what the engine
measures. That identity is asserted here against the acceptance metric itself.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from specdraft.losses import (  # noqa: E402
    LOSSES,
    chunked_distill_loss,
    distill_loss,
    per_position_loss,
)
from specdraft.sampling import expected_acceptance  # noqa: E402


def logits(rows: int, vocab: int, seed: int, scale: float = 2.0) -> "torch.Tensor":
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(rows, vocab, generator=generator) * scale


def test_tvd_is_one_minus_expected_acceptance():
    """The identity the whole sampling story rests on."""
    p_logits, q_logits = logits(16, 64, seed=1), logits(16, 64, seed=2)
    tvd = per_position_loss(q_logits, p_logits, kind="tvd")

    p = torch.softmax(p_logits, dim=-1)
    q = torch.softmax(q_logits, dim=-1)
    acceptance = expected_acceptance(p, q)

    torch.testing.assert_close(tvd, 1.0 - acceptance, atol=1e-6, rtol=1e-5)


def test_identical_models_give_zero_divergence():
    same = logits(8, 32, seed=3)
    for kind in ("fkl", "rkl", "tvd", "jsd"):
        value = distill_loss(same.clone(), same.clone(), kind=kind)
        assert value.item() == pytest.approx(0.0, abs=1e-6), kind


def test_every_loss_is_non_negative():
    p_logits, q_logits = logits(8, 32, seed=4), logits(8, 32, seed=5)
    for kind in ("fkl", "rkl", "tvd", "jsd"):
        assert distill_loss(q_logits, p_logits, kind=kind).item() >= 0.0, kind


def test_forward_and_reverse_kl_differ_and_match_torch():
    p_logits, q_logits = logits(8, 32, seed=6), logits(8, 32, seed=7)
    forward = distill_loss(q_logits, p_logits, kind="fkl")
    reverse = distill_loss(q_logits, p_logits, kind="rkl")
    assert forward.item() != pytest.approx(reverse.item())

    expected = torch.nn.functional.kl_div(
        torch.log_softmax(q_logits, -1), torch.log_softmax(p_logits, -1),
        reduction="none", log_target=True,
    ).sum(-1).mean()
    torch.testing.assert_close(forward, expected, atol=1e-6, rtol=1e-5)


def test_jsd_is_symmetric_and_bounded():
    a, b = logits(8, 32, seed=8), logits(8, 32, seed=9)
    one = distill_loss(a, b, kind="jsd")
    other = distill_loss(b, a, kind="jsd")
    torch.testing.assert_close(one, other, atol=1e-6, rtol=1e-5)
    assert one.item() <= math_log_two() + 1e-6


def math_log_two() -> float:
    import math

    return math.log(2.0)


def test_sft_is_cross_entropy_on_the_text():
    draft = logits(8, 32, seed=10)
    labels = torch.randint(0, 32, (8,), generator=torch.Generator().manual_seed(11))
    value = distill_loss(draft, kind="sft", labels=labels)
    expected = torch.nn.functional.cross_entropy(draft.float(), labels)
    torch.testing.assert_close(value, expected, atol=1e-6, rtol=1e-5)


def test_temperature_flattens_the_divergence():
    p_logits, q_logits = logits(8, 32, seed=12), logits(8, 32, seed=13)
    sharp = distill_loss(q_logits, p_logits, kind="tvd", temperature=0.5)
    flat = distill_loss(q_logits, p_logits, kind="tvd", temperature=4.0)
    assert flat.item() < sharp.item(), "a higher temperature should make the two look closer"


@pytest.mark.parametrize("kind", LOSSES)
def test_bad_arguments_are_rejected(kind):
    draft = logits(4, 16, seed=14)
    with pytest.raises(ValueError):
        per_position_loss(draft, kind="nope")
    with pytest.raises(ValueError):
        per_position_loss(draft, logits(4, 16, seed=15), kind=kind, temperature=0.0)
    if kind == "sft":
        with pytest.raises(ValueError):
            per_position_loss(draft, kind="sft")  # no labels
    else:
        with pytest.raises(ValueError):
            per_position_loss(draft, kind=kind)  # no teacher


# ------------------------------------------------------------------ the chunked version


@pytest.mark.parametrize("kind", LOSSES)
@pytest.mark.parametrize("chunk", [1, 7, 64, 1000])
def test_chunking_gives_the_same_value(kind, chunk):
    generator = torch.Generator().manual_seed(16)
    hidden = torch.randn(23, 8, generator=generator)
    head = torch.randn(40, 8, generator=generator)
    teacher = logits(23, 40, seed=17)
    labels = torch.randint(0, 40, (23,), generator=generator)

    whole = distill_loss(hidden @ head.T, teacher, kind=kind, labels=labels)
    pieces = chunked_distill_loss(
        hidden, head, teacher, kind=kind, labels=labels, chunk=chunk, recompute=False
    )
    torch.testing.assert_close(pieces, whole, atol=1e-5, rtol=1e-4)


@pytest.mark.parametrize("kind", LOSSES)
def test_recomputing_gives_the_same_gradients(kind):
    """Rebuilding the logits during the backward pass must not change the answer."""
    generator = torch.Generator().manual_seed(18)
    base = torch.randn(16, 8, generator=generator)
    head = torch.randn(32, 8, generator=generator)
    teacher = logits(16, 32, seed=19)
    labels = torch.randint(0, 32, (16,), generator=generator)

    gradients = []
    for recompute in (False, True):
        hidden = base.clone().requires_grad_(True)
        loss = chunked_distill_loss(
            hidden, head, teacher, kind=kind, labels=labels, chunk=5, recompute=recompute
        )
        loss.backward()
        gradients.append(hidden.grad.clone())
    torch.testing.assert_close(gradients[0], gradients[1], atol=1e-6, rtol=1e-5)
    assert gradients[0].abs().sum() > 0, "the loss should actually depend on the student"


def test_vocab_limit_drops_the_padded_rows():
    """Qwen3 pads its embedding past the tokenizer, and those rows must never be trained."""
    generator = torch.Generator().manual_seed(20)
    hidden = torch.randn(6, 8, generator=generator)
    head = torch.randn(40, 8, generator=generator)
    teacher = logits(6, 40, seed=21)

    limited = chunked_distill_loss(hidden, head, teacher, kind="fkl", vocab_limit=32,
                                   recompute=False)
    expected = distill_loss((hidden @ head.T)[:, :32], teacher[:, :32], kind="fkl")
    torch.testing.assert_close(limited, expected, atol=1e-5, rtol=1e-4)


def test_mismatched_shapes_are_rejected():
    hidden = torch.randn(4, 8)
    head = torch.randn(16, 8)
    with pytest.raises(ValueError):
        chunked_distill_loss(hidden, head, logits(5, 16, seed=22), kind="fkl")
    with pytest.raises(ValueError):
        chunked_distill_loss(hidden[:0], head, kind="sft", labels=torch.zeros(0, dtype=torch.long))
