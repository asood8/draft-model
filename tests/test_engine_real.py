"""The engine against the twin on the real Qwen3-0.6B.

Skipped unless both files exist::

    python scripts/fetch_model.py Qwen/Qwen3-0.6B
    python scripts/export_model.py models/Qwen3-0.6B --format q4

**What can and cannot be asserted here.** 8-bit activation quantization is a step function,
so two implementations that differ by even one ulp in the preceding float arithmetic land on
opposite sides of rounding boundaries. Measured on this model, a 7e-8 relative nudge to one
layer's input flips 229 of 1024 activation levels and moves that layer's output by 5e-3, and
the effect saturates rather than shrinking as the nudge gets smaller. Elementwise agreement
is therefore impossible by construction, and tightening these thresholds would only produce a
test that fails for the wrong reason.

What must hold is agreement where it is used: the same decoded tokens, the same top-1 choice
almost everywhere, and distributions close enough that acceptance measured on the twin
predicts the engine. Bit-exactness still holds strictly *within* the engine, which is what
greedy speculative decoding depends on, and that is asserted below.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from specdraft.engine import EngineModel  # noqa: E402
from specdraft.speculative import plain_generate  # noqa: E402
from specdraft.twin import QuantizedTwin  # noqa: E402

MODEL = Path(os.environ.get("SPECDRAFT_DRAFT_MODEL", "models/Qwen3-0.6B"))
ENGINE_FILE = Path(os.environ.get("SPECDRAFT_ENGINE_FILE", "models/Qwen3-0.6B-q4.sdm"))

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        not (MODEL / "config.json").is_file() or not ENGINE_FILE.is_file(),
        reason=f"need {MODEL} and {ENGINE_FILE}; see this module's docstring",
    ),
]

POSITIONS = 48


@pytest.fixture(scope="module")
def pair():
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    text = (
        "Speculative decoding runs a small draft model several steps ahead and then checks its "
        "guesses with one pass of the larger model, which keeps the output distribution exactly "
        "the same while doing less work per token."
    )
    ids = tokenizer(text, return_tensors="pt").input_ids[0][:POSITIONS]
    engine = EngineModel(ENGINE_FILE, max_positions=POSITIONS + 16)
    twin = QuantizedTwin.from_pretrained(MODEL, weight_format="q4")
    return tokenizer, ids, engine, twin


def logits_from_both(pair):
    tokenizer, ids, engine, twin = pair
    mine = engine.forward(ids, cache=engine.new_cache(len(ids) + 16))
    with torch.no_grad():
        theirs = twin.forward(ids, cache=twin.new_cache(len(ids) + 16))[:, : mine.shape[1]]
    return mine, theirs


def test_the_two_agree_on_the_top_token_nearly_everywhere(pair):
    mine, theirs = logits_from_both(pair)
    agreement = float((mine.argmax(-1) == theirs.argmax(-1)).float().mean())
    assert agreement > 0.95, f"top-1 agreement only {agreement:.3f}"


def test_distributions_are_close_enough_to_predict_acceptance(pair):
    """Acceptance measured on the twin is off by about this total variation distance."""
    mine, theirs = logits_from_both(pair)
    p = torch.softmax(mine.float(), dim=-1)
    q = torch.softmax(theirs.float(), dim=-1)
    tvd = float((0.5 * (p - q).abs().sum(-1)).mean())
    assert tvd < 0.05, f"mean TVD {tvd:.4f} is too large for the twin to stand in for the engine"


def test_greedy_continuations_are_identical(pair):
    tokenizer, ids, engine, twin = pair
    from_engine, _ = plain_generate(engine, ids, 8)
    with torch.no_grad():
        from_twin, _ = plain_generate(twin, ids, 8, vocab_limit=engine.vocab_limit)
    assert from_engine == from_twin, (
        f"engine {tokenizer.decode(from_engine)!r} vs twin {tokenizer.decode(from_twin)!r}"
    )


def test_the_engine_is_bit_exact_with_itself(pair):
    """k tokens at once must equal k single tokens: what greedy speculation relies on."""
    _, ids, engine, _ = pair
    short = ids[:6]

    together = engine.forward(short, cache=engine.new_cache(16)).numpy()
    one_at_a_time = engine.new_cache(16)
    rows = [engine.forward(short[i : i + 1], cache=one_at_a_time).numpy()[0] for i in range(len(short))]

    assert np.array_equal(together, np.stack(rows))
