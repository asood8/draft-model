"""The training-data pipeline (plan §11.2).

Decontamination is the part worth testing hardest: if a training prompt that overlaps a
Spec-Bench prompt slips through, acceptance on that benchmark partly measures memorization and
the comparison the project exists to make becomes meaningless.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from specdraft.data import (  # noqa: E402
    DEFAULT_MIX,
    SOURCES,
    PromptRecord,
    build_sequence,
    decontaminate,
    generate_responses,
    ngrams,
    normalize,
    read_records,
    read_sequences,
    tokenize_records,
    write_records,
    write_sequences,
)
from specdraft.train import TrainedSequence  # noqa: E402

MODEL_DIR = Path(os.environ.get("SPECDRAFT_DRAFT_MODEL", "models/Qwen3-0.6B"))


@pytest.fixture(scope="module")
def tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(MODEL_DIR)


# ----------------------------------------------------------------------- decontamination


def test_normalization_ignores_case_and_spacing():
    assert normalize("  The   Capital\nof France ") == "the capital of france"


def test_ngrams_of_short_text_is_the_whole_thing():
    assert ngrams("two words", n=13) == {("two", "words")}
    assert ngrams("", n=5) == set()


def test_overlapping_prompts_are_removed():
    shared = "the quick brown fox jumps over the lazy dog while the cat watches from the window"
    records = [
        PromptRecord("a completely unrelated question about trains", source="keep"),
        PromptRecord(f"please continue: {shared}", source="drop"),
        PromptRecord(shared.upper(), source="drop"),  # case must not hide it
    ]
    kept, removed = decontaminate(records, [shared], n=13)
    assert removed == 2
    assert [record.source for record in kept] == ["keep"]


def test_short_overlaps_are_tolerated():
    """Common phrases must not empty the training set."""
    records = [PromptRecord("what is the capital of France?")]
    kept, removed = decontaminate(records, ["the capital of France is a city"], n=13)
    assert removed == 0 and len(kept) == 1


def test_a_shorter_ngram_is_stricter():
    records = [PromptRecord("one two three four five six")]
    _, lenient = decontaminate(records, ["three four five"], n=13)
    _, strict = decontaminate(records, ["three four five"], n=3)
    assert lenient == 0 and strict == 1


def test_every_named_source_exists_in_the_default_mix():
    assert set(DEFAULT_MIX) <= set(SOURCES)


# ------------------------------------------------------------------------------- storage


def test_records_round_trip(tmp_path):
    records = [
        PromptRecord("a question", "an answer", "ultrachat"),
        PromptRecord("unanswered", None, "code"),
    ]
    path = tmp_path / "records.jsonl"
    assert write_records(path, records) == 2
    back = read_records(path)
    assert back == records
    assert back[0].complete and not back[1].complete


def test_sequences_round_trip(tmp_path):
    sequences = [TrainedSequence([1, 2, 3, 4], 2, "math"), TrainedSequence([5, 6], 1)]
    path = tmp_path / "sequences.jsonl"
    assert write_sequences(path, sequences) == 2
    back = read_sequences(path)
    assert [s.tokens for s in back] == [[1, 2, 3, 4], [5, 6]]
    assert [s.response_start for s in back] == [2, 1]
    assert back[0].source == "math"


def test_unicode_survives_the_round_trip(tmp_path):
    records = [PromptRecord("¿cómo estás? 日本語 — emoji 🎯", "sí", "ultrachat")]
    path = tmp_path / "unicode.jsonl"
    write_records(path, records)
    assert read_records(path) == records


# -------------------------------------------------------------------- generating responses


def test_generation_batches_and_keeps_order():
    records = [PromptRecord(f"question {i}", source="src") for i in range(7)]
    seen_batches = []

    def generate(prompts):
        seen_batches.append(len(prompts))
        return [f"answer to {prompt}" for prompt in prompts]

    out = generate_responses(records, generate, batch_size=3, source_suffix="-target")
    assert seen_batches == [3, 3, 1]
    assert [record.response for record in out] == [f"answer to question {i}" for i in range(7)]
    assert all(record.source == "src-target" for record in out)


def test_a_generator_that_loses_prompts_is_caught():
    with pytest.raises(ValueError):
        generate_responses([PromptRecord("a"), PromptRecord("b")], lambda prompts: ["only one"])


# ---------------------------------------------------- templating, against the real tokenizer


@pytest.mark.skipif(
    not (MODEL_DIR / "tokenizer.json").is_file(), reason=f"no tokenizer at {MODEL_DIR}"
)
class TestWithRealTokenizer:
    def test_the_response_boundary_is_right(self, tokenizer):
        sequence = build_sequence(tokenizer, "What is 2 + 2?", "It is 4.", source="math")
        assert sequence is not None
        prompt_text = tokenizer.decode(sequence.tokens[: sequence.response_start])
        response_text = tokenizer.decode(sequence.tokens[sequence.response_start :])

        assert "What is 2 + 2?" in prompt_text
        assert "It is 4." in response_text
        assert sequence.response_tokens > 0
        assert sequence.source == "math"

    def test_non_thinking_mode_is_used(self, tokenizer):
        """Both models decode with thinking off, so training has to match."""
        sequence = build_sequence(tokenizer, "Hello", "Hi")
        prompt_text = tokenizer.decode(sequence.tokens[: sequence.response_start])
        assert "<think>" in prompt_text and "</think>" in prompt_text
        between = prompt_text.split("<think>")[1].split("</think>")[0]
        assert between.strip() == "", "the thinking block should be left empty"

    def test_the_response_ends_with_the_stop_token(self, tokenizer):
        sequence = build_sequence(tokenizer, "Hello", "Hi")
        assert sequence.tokens[-1] == tokenizer.eos_token_id

    def test_long_examples_are_truncated_not_dropped(self, tokenizer):
        sequence = build_sequence(tokenizer, "Count:", " one two three" * 500, max_length=128)
        assert sequence is not None
        assert len(sequence.tokens) <= 128
        assert sequence.response_start < len(sequence.tokens)

    def test_an_overlong_prompt_is_dropped(self, tokenizer):
        assert build_sequence(tokenizer, "word " * 400, "answer", max_length=64) is None

    def test_records_without_responses_are_skipped(self, tokenizer):
        records = [PromptRecord("a question", "an answer"), PromptRecord("no answer here", None)]
        assert len(tokenize_records(tokenizer, records)) == 1
