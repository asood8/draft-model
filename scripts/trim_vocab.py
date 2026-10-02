"""Trim a draft's output vocabulary and export it for the engine (plan section 8.2).

    python scripts/trim_vocab.py --model models/Qwen3-0.6B --keep 32768 \
        --wikitext-tokens 200000 --format q4 --out models/Qwen3-0.6B-trim32k-q4.sdm

A 0.6B draft's output layer spans 151,936 rows and is roughly a quarter of the bytes a decode step
reads; layer pruning cannot touch it. Keeping only the tokens that actually get generated removes
most of that, and stays exact: the draft simply never proposes a dropped token, and the acceptance
rule resamples from the residual, which still covers them.

Count the frequencies over text the draft will have to produce -- ideally the target's own
generations (``--data``) -- rather than over prompts, which are read and not written. The coverage
figure printed here is the ceiling on what trimming can cost in acceptance: whatever mass the
target puts on a dropped token is a guess the draft can no longer make.

The result is an engine file only: a trimmed vocabulary is not a valid Hugging Face model, and
llama.cpp would reject it for not matching the target's vocabulary.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from specdraft.data import read_records, read_sequences
from specdraft.export import write_model
from specdraft.prune import choose_vocabulary, token_frequencies
from specdraft.reference import Qwen3Config, load_safetensors
from specdraft.train import TrainedSequence


def sequences_from_data(path: Path, tokenizer, max_length: int) -> list[TrainedSequence]:
    import json

    first = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
    if "tokens" in first:
        return read_sequences(path)
    from specdraft.data import tokenize_records

    return tokenize_records(tokenizer, read_records(path), max_length=max_length)


def sequences_from_wikitext(tokenizer, token_budget: int) -> list[TrainedSequence]:
    from datasets import load_dataset

    data = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
    text = "\n\n".join(line for line in data["text"] if line.strip())
    ids = tokenizer(text, add_special_tokens=False).input_ids[:token_budget]
    window = 512
    return [
        # Every position counts as generated here: this is a stand-in for real generations.
        TrainedSequence(ids[start : start + window], response_start=1, source="wikitext")
        for start in range(0, len(ids) - window, window)
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True, help="the draft to trim")
    parser.add_argument("--out", type=Path, required=True, help="a .sdm engine file")
    parser.add_argument("--keep", type=int, default=32_768)
    parser.add_argument("--data", type=Path, default=None, help="jsonl of records or sequences")
    parser.add_argument("--wikitext-tokens", type=int, default=0,
                        help="count over WikiText instead, for a quick look")
    parser.add_argument("--format", default="q4", choices=["q4", "q8", "fp32"])
    parser.add_argument("--output-format", default=None, choices=[None, "q4", "q8", "fp32"])
    parser.add_argument("--max-sequence-tokens", type=int, default=2048)
    args = parser.parse_args()

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    config = Qwen3Config.from_pretrained(args.model)
    vocab_limit = len(tokenizer)

    if args.data is not None:
        sequences = sequences_from_data(args.data, tokenizer, args.max_sequence_tokens)
        where = str(args.data)
    elif args.wikitext_tokens > 0:
        sequences = sequences_from_wikitext(tokenizer, args.wikitext_tokens)
        where = f"{args.wikitext_tokens} tokens of WikiText"
    else:
        raise SystemExit("pass --data (preferred) or --wikitext-tokens")

    counts = token_frequencies(sequences, vocab_size=config.vocab_size)
    observed = int((counts > 0).sum())
    print(f"counted {int(counts.sum())} tokens from {where}: "
          f"{observed} distinct of {vocab_limit} in the tokenizer")

    always_keep = sorted(set(tokenizer.all_special_ids or []))
    ids, coverage = choose_vocabulary(counts, keep=min(args.keep, vocab_limit),
                                      always_keep=always_keep)
    print(f"keeping {len(ids)} tokens ({len(ids) / vocab_limit:.1%} of the vocabulary), "
          f"covering {coverage:.3%} of occurrences")
    print(f"so trimming costs at most about {1 - coverage:.3%} of acceptance on *this* text")
    if args.data is None:
        print("\nWARNING: counted on plain prose, which is not what the draft will have to "
              "predict. Measured on a 0.6B, a set chosen from WikiText covered 100% of WikiText "
              "but missed 23% of the model's own chat output -- markdown markers, capitalized "
              "names, newline runs -- so acceptance was capped near 0.77. Recount with --data "
              "pointing at the target's own generations before trusting the coverage figure.")

    state_dict = {k: v.to(torch.float32) for k, v in load_safetensors(args.model).items()}
    entries = write_model(
        args.out, state_dict, config, weight_format=args.format,
        output_format=args.output_format, vocab_limit=vocab_limit,
        output_map=ids.numpy(),
    )
    by_name = {entry.name: entry for entry in entries}
    output_bytes = by_name["output"].nbytes
    whole = by_name["token_embd"].nbytes
    per_step = sum(
        entry.nbytes for entry in entries
        if entry.name.startswith("blk.") and entry.format != "fp32"
    ) + output_bytes

    print(f"\nwrote {args.out}  ({args.out.stat().st_size / 1e6:.1f} MB on disk)")
    print(f"output layer: {whole / 1e6:.1f} MB untrimmed -> {output_bytes / 1e6:.1f} MB")
    print(f"bytes read per decode step: {per_step / 1e6:.1f} MB "
          f"(was {(per_step - output_bytes + whole) / 1e6:.1f} MB), "
          f"so c should fall to about {per_step / (per_step - output_bytes + whole):.2f} of before")
    print("this file is for the engine only: a trimmed vocabulary is not a valid "
          "Hugging Face model and llama.cpp would reject it")


if __name__ == "__main__":
    main()
