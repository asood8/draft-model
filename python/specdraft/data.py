"""Training data for distillation (plan §11.2 and §11.3).

Three things happen here, and the second is the one that protects the results:

* **Mixing.** Chat, code and math prompts in whatever proportions, from public datasets.
* **Decontamination.** Any training prompt sharing a long n-gram with an evaluation prompt is
  dropped. Without this, acceptance on Spec-Bench would partly measure memorization, and the
  whole comparison would be worthless.
* **Templating.** Prompts go through the model's own chat template in non-thinking mode, and
  each example records where the response begins, since the loss applies only to positions whose
  next token is part of the response.

Responses can come from the dataset, from the target, or from the draft; the plan compares all
three, and all three end up in the same format on disk.
"""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence

from .train import TrainedSequence

# How long an n-gram has to be before a shared one counts as contamination.
DECONTAMINATION_NGRAM = 13


@dataclass
class PromptRecord:
    prompt: str
    response: str | None = None
    source: str = ""

    @property
    def complete(self) -> bool:
        return bool(self.response)


# ------------------------------------------------------------------------ dataset adapters
#
# Each adapter returns prompts, and a response when the dataset ships one. Datasets are loaded
# lazily and streamed where possible, so a Kaggle session does not pull more than it needs.


def _take(dataset, count: int, seed: int):
    rows = dataset.shuffle(seed=seed) if hasattr(dataset, "shuffle") else dataset
    return list(rows.select(range(min(count, len(rows))))) if hasattr(rows, "select") else list(rows)[:count]


def load_ultrachat(count: int, seed: int = 0, split: str = "train_sft") -> list[PromptRecord]:
    """Chat prompts with ChatGPT's answers, which is the "fixed text" source of §11.3."""
    from datasets import load_dataset

    rows = _take(load_dataset("HuggingFaceH4/ultrachat_200k", split=split), count, seed)
    out = []
    for row in rows:
        messages = row.get("messages") or []
        user = next((m["content"] for m in messages if m["role"] == "user"), None)
        assistant = next((m["content"] for m in messages if m["role"] == "assistant"), None)
        if user:
            out.append(PromptRecord(user, assistant, "ultrachat"))
    return out


def load_magicoder(count: int, seed: int = 0) -> list[PromptRecord]:
    from datasets import load_dataset

    rows = _take(load_dataset("ise-uiuc/Magicoder-Evol-Instruct-110K", split="train"), count, seed)
    return [
        PromptRecord(row["instruction"], row.get("response"), "code")
        for row in rows
        if row.get("instruction")
    ]


def load_gsm8k(count: int, seed: int = 0) -> list[PromptRecord]:
    """The *train* split only: GSM8K's test set is one of Spec-Bench's categories."""
    from datasets import load_dataset

    rows = _take(load_dataset("openai/gsm8k", "main", split="train"), count, seed)
    return [PromptRecord(row["question"], row.get("answer"), "math") for row in rows]


def load_metamath(count: int, seed: int = 0) -> list[PromptRecord]:
    from datasets import load_dataset

    rows = _take(load_dataset("meta-math/MetaMathQA", split="train"), count, seed)
    return [
        PromptRecord(row["query"], row.get("response"), "math") for row in rows if row.get("query")
    ]


SOURCES: dict[str, Callable[..., list[PromptRecord]]] = {
    "ultrachat": load_ultrachat,
    "code": load_magicoder,
    "gsm8k": load_gsm8k,
    "metamath": load_metamath,
}

# The default mix: mostly chat, with code and math so the draft is not surprised by either.
DEFAULT_MIX = {"ultrachat": 20_000, "code": 5_000, "gsm8k": 2_500, "metamath": 2_500}


def load_prompt_mix(spec: dict[str, int] | None = None, seed: int = 0) -> list[PromptRecord]:
    spec = DEFAULT_MIX if spec is None else spec
    unknown = set(spec) - set(SOURCES)
    if unknown:
        raise ValueError(f"unknown sources: {sorted(unknown)}; expected {sorted(SOURCES)}")

    records: list[PromptRecord] = []
    for name, count in spec.items():
        if count > 0:
            records.extend(SOURCES[name](count, seed=seed))
    random.Random(seed).shuffle(records)
    return records


# -------------------------------------------------------------------------- decontamination


def normalize(text: str) -> str:
    """Lowercase and collapse whitespace, so trivial formatting differences do not hide a match."""
    return " ".join(text.lower().split())


def ngrams(text: str, n: int = DECONTAMINATION_NGRAM) -> set[tuple[str, ...]]:
    words = normalize(text).split()
    if len(words) < n:
        return {tuple(words)} if words else set()
    return {tuple(words[i : i + n]) for i in range(len(words) - n + 1)}


def build_contamination_index(
    evaluation_texts: Iterable[str], n: int = DECONTAMINATION_NGRAM
) -> set[tuple[str, ...]]:
    index: set[tuple[str, ...]] = set()
    for text in evaluation_texts:
        index |= ngrams(text, n)
    return index


def decontaminate(
    records: Sequence[PromptRecord],
    evaluation_texts: Iterable[str],
    n: int = DECONTAMINATION_NGRAM,
) -> tuple[list[PromptRecord], int]:
    """Drop training prompts that share an n-gram with any evaluation prompt.

    Returns the survivors and how many were removed, which belongs in the write-up: a number
    that is suspiciously large usually means the mix overlaps the benchmark by construction.
    """
    index = build_contamination_index(evaluation_texts, n)
    kept = [record for record in records if not (ngrams(record.prompt, n) & index)]
    return kept, len(records) - len(kept)


# ------------------------------------------------------------------ templating and tokenizing


def build_sequence(
    tokenizer,
    prompt: str,
    response: str,
    source: str = "",
    max_length: int = 2048,
    enable_thinking: bool = False,
) -> TrainedSequence | None:
    """Apply the chat template and mark where the response starts.

    Non-thinking mode, matching how the models are decoded. The response ends with the model's
    end-of-turn token so the draft learns to stop where the target stops. Returns None if the
    prompt alone already fills the budget.
    """
    prefix = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )
    prompt_ids = tokenizer(prefix, add_special_tokens=False).input_ids
    if len(prompt_ids) >= max_length - 1:
        return None

    end = tokenizer.eos_token or ""
    response_ids = tokenizer(response + end, add_special_tokens=False).input_ids
    response_ids = response_ids[: max_length - len(prompt_ids)]
    if not response_ids:
        return None
    return TrainedSequence(
        tokens=list(prompt_ids) + list(response_ids),
        response_start=len(prompt_ids),
        source=source,
    )


def tokenize_records(
    tokenizer,
    records: Iterable[PromptRecord],
    max_length: int = 2048,
    enable_thinking: bool = False,
) -> list[TrainedSequence]:
    out = []
    for record in records:
        if not record.complete:
            continue
        sequence = build_sequence(
            tokenizer, record.prompt, record.response, record.source, max_length, enable_thinking
        )
        if sequence is not None:
            out.append(sequence)
    return out


# ------------------------------------------------------------------------ generating responses


def generate_responses(
    records: Sequence[PromptRecord],
    generate: Callable[[list[str]], list[str]],
    batch_size: int = 16,
    source_suffix: str = "",
) -> list[PromptRecord]:
    """Fill in responses with a model's own text.

    ``generate`` takes a batch of prompts and returns their continuations, which keeps this
    independent of whether the text comes from the target on a GPU, the draft, or the engine. The
    plan's three data sources differ only in what gets passed here.
    """
    out: list[PromptRecord] = []
    batch: list[PromptRecord] = []
    for record in records:
        batch.append(record)
        if len(batch) == batch_size:
            out.extend(_apply_batch(batch, generate, source_suffix))
            batch = []
    if batch:
        out.extend(_apply_batch(batch, generate, source_suffix))
    return out


def _apply_batch(batch, generate, source_suffix):
    responses = generate([record.prompt for record in batch])
    if len(responses) != len(batch):
        raise ValueError("the generator returned a different number of responses than prompts")
    return [
        PromptRecord(record.prompt, response, record.source + source_suffix)
        for record, response in zip(batch, responses)
    ]


# ---------------------------------------------------------------------------------- storage


def write_records(path: str | Path, records: Iterable[PromptRecord]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")
            written += 1
    return written


def read_records(path: str | Path) -> list[PromptRecord]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return [PromptRecord(**json.loads(line)) for line in handle if line.strip()]


def write_sequences(path: str | Path, sequences: Iterable[TrainedSequence]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with path.open("w", encoding="utf-8") as handle:
        for sequence in sequences:
            handle.write(
                json.dumps(
                    {
                        "tokens": sequence.tokens,
                        "response_start": sequence.response_start,
                        "source": sequence.source,
                    }
                )
                + "\n"
            )
            written += 1
    return written


def read_sequences(path: str | Path) -> list[TrainedSequence]:
    out = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                out.append(
                    TrainedSequence(row["tokens"], row["response_start"], row.get("source", ""))
                )
    return out
