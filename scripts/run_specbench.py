"""Run Spec-Bench through the engine and write the headline table (plan §9).

    python scripts/run_specbench.py \
        --target models/Qwen3-4B-q4.sdm --draft models/Qwen3-0.6B-q4.sdm \
        --tokenizer models/Qwen3-4B --questions data/spec_bench/question.jsonl \
        --gamma 4 --per-category 10 --out results/specbench.json

Methods, all running inside the engine so the comparison is like for like:

* ``target`` — the model alone, which every speedup is measured against;
* ``speculative`` — the draft model, at the given γ;
* ``lookup`` — guesses copied from earlier text, which needs no draft at all and is hard to beat
  where text repeats;
* ``speculative+stop`` — the draft with confidence-based early stopping.

The question file is not bundled; see ``specdraft.specbench``. Methods are interleaved per
question, and speeds are medians, because this machine drifts while a suite runs (plan §13).
"""

from __future__ import annotations

import argparse
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

from specdraft import _engine as cpp
from specdraft.specbench import (
    aggregate,
    load_questions,
    run_benchmark,
    save_results,
    speedup_table,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=Path, required=True, help="a .sdm weights file")
    parser.add_argument("--draft", type=Path, default=None)
    parser.add_argument("--tokenizer", type=Path, required=True, help="a model directory")
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("results/specbench.json"))
    parser.add_argument("--gamma", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--per-category", type=int, default=None, help="subset size")
    parser.add_argument("--categories", nargs="*", default=None)
    parser.add_argument("--temperature", type=float, default=0.0, help="0 is greedy")
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--confidence-threshold", type=float, default=0.4)
    parser.add_argument("--context", type=int, default=4096, help="cache size per model")
    parser.add_argument("--cores", default="performance")
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--max-ngram", type=int, default=3)
    parser.add_argument("--methods", nargs="*", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--keep-text", action="store_true", help="store what was generated")
    args = parser.parse_args()

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    questions = load_questions(
        args.questions, args.categories, args.per_category, seed=args.seed
    )
    if not questions:
        raise SystemExit(f"no questions loaded from {args.questions}")
    categories = sorted({question.category for question in questions})
    print(f"{len(questions)} questions across {len(categories)} categories: {categories}")

    def open_model(path: Path):
        return cpp.Model(
            str(path),
            max_positions=args.context,
            cores=args.cores,
            threads=args.threads,
            max_batch=args.gamma + 1,
        )

    target = open_model(args.target)
    draft = open_model(args.draft) if args.draft is not None else None
    print(f"target {args.target.name} on {target.threads} threads"
          + (f", draft {args.draft.name}" if draft else ""))

    sampling = {
        "temperature": args.temperature,
        "top_k": args.top_k,
        "top_p": args.top_p,
        "max_new_tokens": args.max_new_tokens,
        "stop": [tokenizer.eos_token_id],
    }

    methods = {
        "target": lambda prompt: cpp.generate_plain(target, prompt, **sampling),
        "lookup": lambda prompt: cpp.generate_prompt_lookup(
            target, prompt, gamma=args.gamma, max_ngram=args.max_ngram, **sampling
        ),
    }
    if draft is not None:
        methods["speculative"] = lambda prompt: cpp.generate_speculative(
            target, draft, prompt, gamma=args.gamma, **sampling
        )
        methods["speculative+stop"] = lambda prompt: cpp.generate_speculative(
            target, draft, prompt, gamma=args.gamma,
            confidence_threshold=args.confidence_threshold, **sampling
        )
    if args.methods is not None:
        unknown = set(args.methods) - set(methods)
        if unknown:
            raise SystemExit(f"unknown methods {sorted(unknown)}; have {sorted(methods)}")
        methods = {name: methods[name] for name in args.methods}
    print(f"methods: {list(methods)}")

    done = {"count": 0}

    def report(result) -> None:
        done["count"] += 1
        print(f"  [{done['count']:>4}] {result.method:<18} {result.category:<14} "
              f"{result.tokens:>4} tok  {result.tokens_per_second:>6.1f} tok/s  "
              f"tau {result.tokens_per_target_forward:.2f}", flush=True)

    results = run_benchmark(
        questions, methods, tokenizer, max_new_tokens=args.max_new_tokens,
        keep_text=args.keep_text, on_result=report,
    )

    summary = aggregate(results)
    speedups = speedup_table(summary, baseline="target")

    print(f"\n{'method':<20} {'category':<14} {'tau':>6} {'tok/s':>8} {'speedup':>8} {'alpha':>6}")
    for method in summary:
        for category in sorted(summary[method]):
            row = summary[method][category]
            print(f"{method:<20} {category:<14} {row['tokens_per_target_forward']:>6.2f} "
                  f"{row['tokens_per_second']:>8.1f} {speedups[method][category]:>7.2f}x "
                  f"{row['alpha']:>6.3f}")

    path = save_results(
        args.out,
        results,
        extra={
            "target": str(args.target),
            "draft": None if args.draft is None else str(args.draft),
            "gamma": args.gamma,
            "sampling": {k: v for k, v in sampling.items() if k != "stop"},
            "confidence_threshold": args.confidence_threshold,
            "speedups": speedups,
            "threads": target.threads,
            "cores": args.cores,
            "cpu": platform.processor(),
            "questions": len(questions),
            "when": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        },
    )
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
