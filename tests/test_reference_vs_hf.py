"""Check the from-scratch reference against Hugging Face's own Qwen3 (plan §7.1).

Skipped unless the weights are on disk::

    python scripts/fetch_model.py Qwen/Qwen3-0.6B

Set SPECDRAFT_DRAFT_MODEL to point somewhere else. Everything runs in fp32 on the CPU,
so any difference beyond float32 rounding is a real bug in the reference.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from specdraft.reference import Qwen3Reference, trim_vocab  # noqa: E402

MODEL_DIR = Path(os.environ.get("SPECDRAFT_DRAFT_MODEL", "models/Qwen3-0.6B"))

pytestmark = pytest.mark.skipif(
    not (MODEL_DIR / "config.json").is_file(),
    reason=f"no model at {MODEL_DIR}; run scripts/fetch_model.py first",
)

PROMPTS = [
    "The capital of France is",
    "def fibonacci(n):\n    if n < 2:",
    "Q: What is 12 * 7?\nA:",
]


@pytest.fixture(scope="module")
def models():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    ours = Qwen3Reference.from_pretrained(MODEL_DIR, dtype=torch.float32)
    # eager attention keeps Hugging Face's arithmetic closest to the reference's.
    theirs = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR, dtype=torch.float32, attn_implementation="eager"
    ).eval()
    return tokenizer, ours, theirs


@pytest.mark.parametrize("prompt", PROMPTS)
def test_logits_match(models, prompt):
    tokenizer, ours, theirs = models
    ids = tokenizer(prompt, return_tensors="pt").input_ids[0]

    mine = ours.forward(ids)
    with torch.no_grad():
        hf = theirs(ids.unsqueeze(0)).logits[0]

    limit = len(tokenizer)
    mine, hf = trim_vocab(mine, limit), trim_vocab(hf, limit)
    assert (mine.argmax(-1) == hf.argmax(-1)).all(), "top-1 token differs"
    assert (mine - hf).abs().max().item() < 1e-3, f"max logit gap {(mine - hf).abs().max().item():.2e}"


def test_hidden_states_match_layer_by_layer(models):
    """Where a mismatch appears tells you which layer is wrong."""
    tokenizer, ours, theirs = models
    ids = tokenizer(PROMPTS[0], return_tensors="pt").input_ids[0]

    captured: list[torch.Tensor] = []
    ours.forward(ids, capture=captured)
    with torch.no_grad():
        hf = theirs(ids.unsqueeze(0), output_hidden_states=True).hidden_states

    # hf[0] is the embedding output and hf[-1] is after the final norm, so hf[i + 1] is
    # the output of layer i for every layer but the last.
    for i, mine in enumerate(captured[:-1]):
        gap = (mine - hf[i + 1][0]).abs().max().item()
        assert gap < 1e-3, f"layer {i} differs by {gap:.2e}"


def test_greedy_generation_matches(models):
    tokenizer, ours, theirs = models
    ids = tokenizer(PROMPTS[0], return_tensors="pt").input_ids[0]

    mine = ours.greedy_generate(ids, max_new_tokens=32)
    with torch.no_grad():
        hf = theirs.generate(
            ids.unsqueeze(0), max_new_tokens=32, do_sample=False, use_cache=True
        )[0][len(ids) :]
    assert mine == hf.tolist()


def test_config_matches_the_checklist(models):
    """The gotchas from plan §5.4 that break from-scratch implementations."""
    tokenizer, ours, _ = models
    cfg = ours.config
    assert cfg.head_dim != cfg.hidden_size // cfg.num_attention_heads
    assert cfg.tie_word_embeddings and ours.lm_head is ours.embed_tokens
    assert cfg.vocab_size > len(tokenizer), "the embedding is padded past the tokenizer"
    assert cfg.hidden_size % 32 == 0 and cfg.intermediate_size % 32 == 0  # quantizable
