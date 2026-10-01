"""Score a draft's layers and remove the least useful ones (plan §8.1).

    python scripts/prune_draft.py --model models/Qwen3-0.6B --keep 14 \
        --out models/Qwen3-0.6B-keep14

Prints the influence profile, so it is visible which layers are doing little, then writes the
pruned model. Pruning on its own costs acceptance; the point is to pair it with distillation,
which wins the acceptance back at the lower cost:

    python scripts/train_draft.py --student models/Qwen3-0.6B-keep14 \
        --teacher models/Qwen3-4B --data data/target_generated.jsonl --loss tvd ...
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from specdraft.data import read_records
from specdraft.prune import (
    block_influence,
    choose_layers_to_keep,
    describe_pruning,
    prune_layers,
    save_pruned,
)
from specdraft.reference import Qwen3Reference

# Enough ordinary English, code and arithmetic to score layers on, for when no calibration file
# is given. Layer influence is a coarse statistic and does not need much text.
DEFAULT_CALIBRATION = [
    "Speculative decoding runs a small model ahead of a large one and checks its guesses in a "
    "single pass, which keeps the output distribution identical while doing less work per token.",
    "def fibonacci(n):\n    if n < 2:\n        return n\n    return fibonacci(n - 1) + fibonacci(n - 2)",
    "The capital of France is Paris, the capital of Italy is Rome, and the capital of Japan is "
    "Tokyo. Each of these cities is the seat of its national government.",
    "If a train leaves the station at nine in the morning travelling at sixty kilometres an hour, "
    "and a second train leaves two hours later at ninety, when does the second catch the first?",
    "Memory bandwidth, not arithmetic, is what limits generating one token at a time: the weights "
    "are read once per token and barely reused, so the clock matters less than the bus.",
]


def calibration_sequences(tokenizer, args) -> list[torch.Tensor]:
    if args.calibration is not None:
        records = read_records(args.calibration)[: args.sequences]
        texts = [
            record.prompt + ("\n" + record.response if record.response else "")
            for record in records
        ]
    else:
        texts = (DEFAULT_CALIBRATION * ((args.sequences // len(DEFAULT_CALIBRATION)) + 1))[
            : args.sequences
        ]
    out = []
    for text in texts:
        ids = tokenizer(text, add_special_tokens=False).input_ids[: args.length]
        if len(ids) >= 8:
            out.append(torch.tensor(ids, dtype=torch.long))
    if not out:
        raise SystemExit("no usable calibration text")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None, help="omit to only print the scores")
    parser.add_argument("--keep", type=int, default=None, help="how many layers to keep")
    parser.add_argument("--calibration", type=Path, default=None, help="jsonl of records")
    parser.add_argument("--sequences", type=int, default=16)
    parser.add_argument("--length", type=int, default=256)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--no-protect-last", action="store_true")
    args = parser.parse_args()

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = Qwen3Reference.from_pretrained(args.model, device=args.device)
    sequences = calibration_sequences(tokenizer, args)
    print(f"{model.config.num_hidden_layers} layers, "
          f"{len(sequences)} calibration sequences, "
          f"{sum(len(s) for s in sequences)} positions")

    scores = block_influence(model, sequences)
    order = sorted(range(len(scores)), key=lambda i: scores[i])
    print(f"\n{'layer':>5} {'influence':>10}  (lowest first means most removable)")
    for rank, index in enumerate(order):
        marker = "  <- least useful" if rank < 3 else ""
        print(f"{index:>5} {scores[index]:>10.4f}{marker}")

    if args.keep is None:
        return

    keep_indices = choose_layers_to_keep(scores, args.keep, not args.no_protect_last)
    dropped = [index for index in range(len(scores)) if index not in keep_indices]
    pruned = prune_layers(model, keep_indices)
    summary = describe_pruning(model.config, pruned.config)
    print(f"\nkeeping {keep_indices}")
    print(f"dropping {dropped}")
    print(f"bytes per token: {summary['bytes_per_token_before'] / 1e6:.1f} MB -> "
          f"{summary['bytes_per_token_after'] / 1e6:.1f} MB "
          f"({summary['bytes_ratio']:.2f} of the original, so c should fall by about as much)")

    if args.out is not None:
        save_pruned(
            args.out,
            pruned,
            source_model_dir=args.model,
            extra={
                "pruned_from": str(args.model),
                "kept_layers": keep_indices,
                "dropped_layers": dropped,
                "influence_scores": scores,
                **summary,
            },
        )
        print(f"\nwrote {args.out}")
        print("acceptance will have fallen; distil it back with scripts/train_draft.py")


if __name__ == "__main__":
    main()
