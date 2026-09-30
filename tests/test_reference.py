"""Reference-model tests that need no downloads: they run on a tiny random Qwen3.

The invariants checked here are the ones the C++ engine and the speculative decoder rely
on: feeding k tokens at once must equal feeding them one at a time, and rolling the cache
back must leave no trace of the tokens that were discarded.
"""

from __future__ import annotations

import dataclasses

import pytest

torch = pytest.importorskip("torch")

from specdraft.reference import (  # noqa: E402  (import after the torch check)
    TINY_CONFIG,
    random_reference,
    trim_vocab,
)

CLOSE = {"atol": 2e-5, "rtol": 1e-4}  # fp32 matmuls take different paths at different shapes


@pytest.fixture(scope="module")
def model():
    return random_reference(TINY_CONFIG, seed=0)


def tokens(n: int, seed: int = 0) -> "torch.Tensor":
    generator = torch.Generator().manual_seed(seed)
    return torch.randint(0, TINY_CONFIG.vocab_size, (n,), generator=generator)


def test_config_head_dim_comes_from_the_config():
    cfg = TINY_CONFIG
    assert cfg.head_dim != cfg.hidden_size // cfg.num_attention_heads
    assert cfg.group_size == cfg.num_attention_heads // cfg.num_key_value_heads == 2


def test_forward_shapes(model):
    ids = tokens(5)
    assert model.forward(ids).shape == (5, TINY_CONFIG.vocab_size)
    assert model.forward(ids, only_last_logits=True).shape == (1, TINY_CONFIG.vocab_size)


def test_rejects_batched_input(model):
    with pytest.raises(ValueError):
        model.forward(tokens(4).unsqueeze(0))


def test_incremental_decoding_matches_one_pass(model):
    ids = tokens(12, seed=1)
    full = model.forward(ids)

    cache = model.new_cache(32)
    step_logits = [
        model.forward(ids[i : i + 1], cache=cache, only_last_logits=True)[0] for i in range(len(ids))
    ]
    torch.testing.assert_close(torch.stack(step_logits), full, **CLOSE)
    assert cache.pos == len(ids)


@pytest.mark.parametrize("chunk", [1, 2, 3, 5])
def test_chunked_prefill_matches_one_pass(model, chunk):
    """Verification and prompt processing feed several tokens at once; same answer."""
    ids = tokens(10, seed=2)
    full = model.forward(ids)

    cache = model.new_cache(32)
    pieces = [
        model.forward(ids[i : i + chunk], cache=cache) for i in range(0, len(ids), chunk)
    ]
    torch.testing.assert_close(torch.cat(pieces), full, **CLOSE)


def test_rewinding_the_cache_discards_rejected_tokens(model):
    """What the speculative decoder does after a rejection: move the counter back."""
    accepted, rejected = tokens(6, seed=3), tokens(4, seed=4)
    replacement = tokens(4, seed=5)

    cache = model.new_cache(32)
    model.forward(accepted, cache=cache)
    model.forward(rejected, cache=cache)  # guesses that turn out to be wrong
    cache.rewind_to(len(accepted))
    after_rewind = model.forward(replacement, cache=cache)

    clean = model.new_cache(32)
    model.forward(accepted, cache=clean)
    expected = model.forward(replacement, cache=clean)

    torch.testing.assert_close(after_rewind, expected, **CLOSE)


def test_causal_mask_hides_later_tokens(model):
    """Logits at position i must not depend on tokens after i."""
    ids = tokens(8, seed=6)
    changed = ids.clone()
    changed[5:] = (changed[5:] + 1) % TINY_CONFIG.vocab_size

    torch.testing.assert_close(model.forward(ids)[:5], model.forward(changed)[:5], **CLOSE)


def test_fp16_cache_is_close_but_not_identical(model):
    """The engine keeps its KV cache in fp16; the twin will do the same."""
    ids = tokens(10, seed=7)
    exact = model.forward(ids)

    cache = model.new_cache(32, dtype=torch.float16)
    rounded = model.forward(ids, cache=cache)

    assert cache.keys.dtype == torch.float16
    torch.testing.assert_close(rounded, exact, atol=5e-3, rtol=5e-3)


def test_capture_returns_one_hidden_state_per_layer(model):
    """This is the hook that pinpoints which engine layer has a bug."""
    captured: list[torch.Tensor] = []
    model.forward(tokens(4), capture=captured)
    assert len(captured) == TINY_CONFIG.num_hidden_layers
    assert all(h.shape == (4, TINY_CONFIG.hidden_size) for h in captured)


def test_greedy_generate_is_deterministic_and_stops(model):
    prompt = tokens(3, seed=8)
    first = model.greedy_generate(prompt, max_new_tokens=6)
    assert first == model.greedy_generate(prompt, max_new_tokens=6)
    assert len(first) == 6

    # A random model often repeats a token, so stop on the first one: that is unambiguous.
    stopped = model.greedy_generate(prompt, max_new_tokens=6, stop={first[0]})
    assert stopped == first[:1]


def test_trim_vocab_drops_padded_rows(model):
    logits = model.forward(tokens(2))
    trimmed = trim_vocab(logits, TINY_CONFIG.vocab_size - 3)
    assert trimmed.shape == (2, TINY_CONFIG.vocab_size - 3)


def test_untied_lm_head_is_used_when_present():
    untied = dataclasses.replace(TINY_CONFIG, tie_word_embeddings=False)
    model = random_reference(untied, seed=9)
    assert model.lm_head is not model.embed_tokens
