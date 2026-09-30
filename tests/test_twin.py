"""The twin must behave like the engine: quantized weights, quantized activations, fp16 KV."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from specdraft.quant_torch import fake_quantize  # noqa: E402
from specdraft.reference import TINY_CONFIG, random_reference  # noqa: E402
from specdraft.twin import QuantizedTwin  # noqa: E402


@pytest.fixture(scope="module")
def model():
    return random_reference(TINY_CONFIG, seed=0)


def tokens(n: int, seed: int) -> "torch.Tensor":
    generator = torch.Generator().manual_seed(seed)
    return torch.randint(0, TINY_CONFIG.vocab_size, (n,), generator=generator)


def relative_error(twin, reference, ids) -> float:
    a, b = twin.forward(ids), reference.forward(ids)
    return ((a - b).norm() / b.norm()).item()


def test_weights_are_actually_quantized(model):
    twin = QuantizedTwin.from_reference(model, weight_format="q4", activation_format=None)
    q_proj = twin.layers[0].q_proj
    assert torch.equal(fake_quantize(q_proj, "q4"), q_proj)  # already on the grid
    assert not torch.equal(q_proj, model.layers[0].q_proj)
    # Norm weights are not quantized, in the twin or the engine.
    assert torch.equal(twin.layers[0].q_norm, model.layers[0].q_norm)
    assert torch.equal(twin.final_norm, model.final_norm)


def test_eight_bit_is_closer_than_four_bit(model):
    ids = tokens(12, seed=1)
    q8 = relative_error(
        QuantizedTwin.from_reference(model, weight_format="q8", activation_format=None), model, ids
    )
    q4 = relative_error(
        QuantizedTwin.from_reference(model, weight_format="q4", activation_format=None), model, ids
    )
    assert q8 < q4, f"q8 {q8:.4f} should beat q4 {q4:.4f}"
    assert q8 < 0.05


def test_activation_quantization_adds_error_but_stays_close(model):
    ids = tokens(12, seed=2)
    weights_only = relative_error(
        QuantizedTwin.from_reference(model, weight_format="q8", activation_format=None), model, ids
    )
    full = relative_error(
        QuantizedTwin.from_reference(model, weight_format="q8", activation_format="a8"), model, ids
    )
    assert weights_only <= full < 0.2


def test_output_layer_can_keep_more_precision(model):
    """4-bit body with an 8-bit output layer: the middle option from the plan."""
    mixed = QuantizedTwin.from_reference(model, weight_format="q4", output_format="q8")
    assert torch.equal(fake_quantize(mixed.embed_tokens, "q8"), mixed.embed_tokens)
    assert torch.equal(fake_quantize(mixed.layers[0].q_proj, "q4"), mixed.layers[0].q_proj)
    assert mixed.lm_head is mixed.embed_tokens  # tied, so one matrix serves both


def test_cache_is_fp16_by_default(model):
    twin = QuantizedTwin.from_reference(model)
    assert twin.new_cache(16).keys.dtype == torch.float16
    assert twin.new_cache(16, dtype=torch.float32).keys.dtype == torch.float32


def test_twin_is_deterministic_and_usable_for_decoding(model):
    from specdraft.speculative import plain_generate, speculative_generate

    twin = QuantizedTwin.from_reference(model)
    prompt = tokens(4, seed=3)
    assert twin.forward(prompt).equal(twin.forward(prompt))

    # A quantized target with a quantized draft still decodes exactly like plain decoding
    # from that same quantized target: quantization changes the distribution, not the rule.
    draft = QuantizedTwin.from_reference(random_reference(TINY_CONFIG, seed=4))
    expected, _ = plain_generate(twin, prompt, 20)
    got, _ = speculative_generate(twin, draft, prompt, 20, gamma=3)
    assert got == expected


def test_disabling_quantization_reproduces_the_reference(model):
    twin = QuantizedTwin.from_reference(
        model, weight_format=None, activation_format=None, kv_dtype=torch.float32
    )
    ids = tokens(8, seed=5)
    assert torch.equal(twin.forward(ids), model.forward(ids))


def test_describe_mentions_every_choice(model):
    text = QuantizedTwin.from_reference(model, weight_format="q4", output_format="q8").describe()
    assert "q4" in text and "q8" in text and "a8" in text and "float16" in text
