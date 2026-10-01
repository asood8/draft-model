"""Score a draft without decoding, and sweep gamma for free (plan section 11.6).

    python scripts/offline_acceptance.py \
        --target models/Qwen3-4B --draft runs/tvd-target/draft \
        --references data/references_greedy.jsonl --mode greedy \
        --c 0.18 --vk results/vk_Qwen3-4B-q4.json --tokenizer models/Qwen3-4B \
        --out results/offline_tvd.json

Decoding on a laptop CPU runs at tens of tokens a second, so measuring acceptance by actually
generating would take days across every draft, decoding mode, gamma and task. Instead each model runs
*once* over text the target already produced, and the rounds are simulated from the per-position
numbers. For greedy that is not an approximation but the same computation; for sampling it is exact
in distribution.

What comes out: acceptance, tokens per target pass at every gamma, the predicted speedup when c and
v(k) are supplied, and which kinds of token the draft gets wrong -- the last being what makes the
write-up more than a table.

Either side can be a Hugging Face directory (scored through the quantization twin, which is how
recipes get ranked on a GPU) or a ``.sdm`` engine file (which is what the final numbers should use,
since the twin tracks the engine only to within a couple of percent).

The references must be the target's own continuations at the settings being studied. ``--mode
greedy`` checks that and says so when they are not.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from specdraft.data import read_sequences
from specdraft.offline import (
    acceptance_by_class,
    best_gamma,
    concatenate,
    gamma_sweep,
    predicted_speedup,
    score_sequence,
    token_class_names,
)
from specdraft.reference import Qwen3Reference
from specdraft.sampling import GREEDY, SamplingConfig


def open_model(path: Path, weight_format: str | None, max_positions: int, device: str):
    """A Hugging Face directory becomes a twin; a .sdm file becomes the engine itself."""
    if path.suffix == ".sdm":
        from specdraft.engine import EngineModel

        return EngineModel(path, max_positions=max_positions), "engine"
    if weight_format is None:
        return Qwen3Reference.from_pretrained(path, device=device), "fp32"

    from specdraft.twin import QuantizedTwin

    twin = QuantizedTwin.from_pretrained(path, device=device, weight_format=weight_format)
    return twin, f"twin:{weight_format}"


def load_vk(path: Path | None, flat: float) -> dict[int, float] | float:
    """The measured v(k) curve from measure_vk.py, or a single number."""
    if path is None:
        return flat
    blob = json.loads(path.read_text(encoding="utf-8"))
    return {int(k): float(value) for k, value in blob["v"].items()}


def verification_cost(v: dict[int, float] | float, k: int) -> float:
    return float(v) if isinstance(v, (int, float)) else float(v.get(k, 1.0))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--draft", type=Path, required=True)
    parser.add_argument("--references", type=Path, required=True,
                        help="jsonl of sequences: the target's own continuations")
    parser.add_argument("--out", type=Path, default=Path("results/offline_acceptance.json"))
    parser.add_argument("--mode", default="greedy", choices=["greedy", "sampling"])
    parser.add_argument("--temperature", type=float, default=1.0, help="for sampling mode")
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--gammas", type=int, nargs="*", default=[1, 2, 3, 4, 5, 6, 8])
    parser.add_argument("--target-format", default=None, choices=[None, "q4", "q8"])
    parser.add_argument("--draft-format", default=None, choices=[None, "q4", "q8"])
    parser.add_argument("--tokenizer", type=Path, default=None,
                        help="enables the per-token-class breakdown")
    parser.add_argument("--c", type=float, default=None, help="draft step over target step")
    parser.add_argument("--vk", type=Path, default=None, help="a vk_*.json from measure_vk.py")
    parser.add_argument("--v", type=float, default=1.0, help="flat v, when no curve is given")
    parser.add_argument("--o", type=float, default=0.0, help="per-round overhead")
    parser.add_argument("--confidence-thresholds", type=float, nargs="*", default=[0.0])
    parser.add_argument("--limit", type=int, default=None, help="use only this many sequences")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--chunk", type=int, default=256)
    args = parser.parse_args()

    sequences = read_sequences(args.references)[: args.limit]
    if not sequences:
        raise SystemExit(f"no sequences in {args.references}")
    longest = max(len(sequence.tokens) for sequence in sequences)
    print(f"{len(sequences)} reference sequences, "
          f"{sum(s.response_tokens for s in sequences)} response positions")

    target, target_kind = open_model(args.target, args.target_format, longest + 8, args.device)
    draft, draft_kind = open_model(args.draft, args.draft_format, longest + 8, args.device)
    print(f"target {args.target.name} as {target_kind}; draft {args.draft.name} as {draft_kind}")

    config = GREEDY if args.mode == "greedy" else SamplingConfig(
        temperature=args.temperature, top_k=args.top_k, top_p=args.top_p
    )
    greedy = args.mode == "greedy"

    tokenizer = None
    vocab_limit = getattr(draft, "vocab_limit", None)
    if args.tokenizer is not None:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
        vocab_limit = len(tokenizer)

    scored = []
    by_source: dict[str, list] = {}
    not_greedy_references = 0

    for index, sequence in enumerate(sequences):
        tokens = torch.tensor(sequence.tokens, dtype=torch.long)
        metrics = score_sequence(
            target, draft, tokens, sequence.response_start, config=config,
            vocab_limit=vocab_limit, chunk=args.chunk,
        )
        if greedy and not metrics.is_greedy_reference:
            not_greedy_references += 1
        scored.append(metrics)
        by_source.setdefault(sequence.source or "all", []).append(metrics)
        if (index + 1) % 10 == 0:
            print(f"  scored {index + 1}/{len(sequences)}", flush=True)

    if not_greedy_references:
        print(f"\nWARNING: {not_greedy_references} of {len(sequences)} references are not this "
              "target's greedy output, so the greedy simulation is not exact for them. "
              "Regenerate them with the target before trusting these numbers.")

    pooled = concatenate(scored)
    accept = pooled.accept_prob(greedy=greedy)
    sweeps = {
        threshold: gamma_sweep(pooled, args.gammas, greedy=greedy, confidence_threshold=threshold)
        for threshold in args.confidence_thresholds
    }

    print(f"\nacceptance ({args.mode}): {float(np.mean(accept)):.4f}")
    per_source = {}
    for source, parts in sorted(by_source.items()):
        bucket = concatenate(parts)
        per_source[source] = bucket
        print(f"  {source:<24} greedy top-1 {bucket.greedy_alpha:.4f}   "
              f"1-TVD {bucket.sampling_alpha:.4f}   positions {len(bucket)}")

    v = load_vk(args.vk, args.v)
    primary = sweeps[args.confidence_thresholds[0]]
    header = f"\n{'gamma':>6} {'tokens/pass':>12}"
    if args.c is not None:
        header += f" {'v(gamma+1)':>11} {'speedup':>9}"
    print(header)
    for gamma in args.gammas:
        line = f"{gamma:>6} {primary[gamma]:>12.3f}"
        if args.c is not None:
            cost = verification_cost(v, gamma + 1)
            line += f" {cost:>11.2f} {predicted_speedup(primary[gamma], gamma, args.c, cost, args.o):>8.2f}x"
        print(line)

    if args.c is not None:
        gamma, speedup = best_gamma(primary, args.c, v, args.o)
        print(f"\nbest: {speedup:.2f}x at gamma={gamma}  (c={args.c}, o={args.o})")

    if len(args.confidence_thresholds) > 1:
        print("\nwith confidence-based early stopping:")
        for threshold, sweep in sweeps.items():
            gamma = max(sweep, key=sweep.__getitem__)
            line = f"  threshold {threshold:>4}: {sweep[gamma]:.3f} tokens/pass at gamma={gamma}"
            if args.c is not None:
                best = best_gamma(sweep, args.c, v, args.o)
                line += f", best {best[1]:.2f}x at gamma={best[0]}"
            print(line)

    profile = None
    if tokenizer is not None:
        classes = token_class_names(tokenizer, pooled.tokens)
        profile = acceptance_by_class(pooled, classes, greedy=greedy)
        print(f"\n{'token class':<22} {'share':>7} {'acceptance':>11}")
        for name, row in profile.items():
            print(f"{name:<22} {row['share']:>6.1%} {row['acceptance']:>11.3f}")

    record = {
        "target": str(args.target),
        "draft": str(args.draft),
        "target_kind": target_kind,
        "draft_kind": draft_kind,
        "mode": args.mode,
        "sampling": {"temperature": args.temperature, "top_k": args.top_k, "top_p": args.top_p},
        "sequences": len(sequences),
        "positions": int(len(accept)),
        "acceptance": float(np.mean(accept)),
        "greedy_top1": pooled.greedy_alpha,
        "sampling_acceptance": pooled.sampling_alpha,
        "per_source": {
            source: {
                "greedy_top1": bucket.greedy_alpha,
                "sampling_acceptance": bucket.sampling_alpha,
                "positions": len(bucket),
            }
            for source, bucket in per_source.items()
        },
        "tokens_per_pass": {
            str(threshold): {str(gamma): value for gamma, value in sweep.items()}
            for threshold, sweep in sweeps.items()
        },
        "c": args.c,
        "o": args.o,
        "v": v if isinstance(v, float) else {str(k): value for k, value in v.items()},
        "non_greedy_references": not_greedy_references,
        "acceptance_by_token_class": profile,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
