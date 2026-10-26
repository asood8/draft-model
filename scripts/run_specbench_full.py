"""Run the whole of Spec-Bench in resumable chunks, then merge them into one result.

    python scripts/run_specbench_full.py --target models/Qwen3-4B-q4.sdm \
        --draft models/Qwen3-0.6B-q4.sdm --tokenizer models/Qwen3-4B --gamma 1

The full benchmark is 480 questions, which at the rate this engine measures is about twelve hours
with four methods. A twelve-hour single invocation is a bad bet on this machine: the harness has
killed long jobs twice for memory pressure, and losing hour ten of twelve is painful enough to
discourage running it at all.

So the sweep is carved into chunks by `--offset-per-category`, each chunk written to its own file,
and a chunk whose file already exists is skipped. A killed run is restarted with the same command and
picks up where it stopped. The shuffle inside `load_questions` is seeded, so consecutive offsets cut
the same deterministic order into disjoint pieces -- there is a test for that.

Merging re-aggregates from the per-turn records rather than averaging the chunks' speedups. That
matters: a speedup is a ratio of totals, the aggregate uses medians over turns, and a
median-of-medians over chunks is not the median over the benchmark.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))

from specdraft.specbench import TurnResult, load_questions, save_results, speedup_table  # noqa: E402

# The six categories the benchmark reports, after two-turn questions are grouped as multi_turn.
CATEGORIES = ["math_reasoning", "multi_turn", "qa", "rag", "summarization", "translation"]
METHODS = ["target", "speculative", "lookup", "speculative+stop"]


def chunk_path(out_dir: Path, gamma: int, offset: int, size: int) -> Path:
    return out_dir / f"chunk_gamma{gamma}_off{offset:03d}_n{size}.json"


def largest_category(questions_file: Path) -> int:
    """How many questions the biggest category has, which sets how many chunks are needed."""
    counts: dict[str, int] = {}
    for question in load_questions(questions_file, CATEGORIES):
        counts[question.category] = counts.get(question.category, 0) + 1
    if not counts:
        raise SystemExit(f"no questions loaded from {questions_file}")
    for category in sorted(counts):
        print(f"  {category:<18}{counts[category]:>4}")
    return max(counts.values())


def run_chunk(args, offset: int, path: Path) -> int:
    cmd = [
        sys.executable, str(ROOT / "scripts" / "run_specbench.py"),
        "--target", str(args.target),
        "--draft", str(args.draft),
        "--tokenizer", str(args.tokenizer),
        "--questions", str(args.questions),
        "--gamma", str(args.gamma),
        "--per-category", str(args.chunk_size),
        "--offset-per-category", str(offset),
        "--categories", *CATEGORIES,
        "--methods", *args.methods,
        "--seed", str(args.seed),
        "--out", str(path),
    ]
    return subprocess.call(cmd, cwd=ROOT)


def merge(out_dir: Path, gamma: int, destination: Path) -> None:
    """Rebuild one benchmark-shaped result from the chunk files."""
    chunks = sorted(out_dir.glob(f"chunk_gamma{gamma}_off*.json"))
    if not chunks:
        raise SystemExit(f"nothing to merge in {out_dir}")

    turns: list[TurnResult] = []
    fields = set(TurnResult.__dataclass_fields__)
    meta: dict = {}
    for path in chunks:
        blob = json.loads(path.read_text(encoding="utf-8"))
        for row in blob["turns"]:
            turns.append(TurnResult(**{k: v for k, v in row.items() if k in fields}))
        meta = {k: v for k, v in blob.items()
                if k not in ("summary", "turns", "speedups", "decode_speedups",
                             "offset_per_category", "per_category", "questions")}

    seen = {(t.method, t.category, t.question_id, t.turn) for t in turns}
    if len(seen) != len(turns):
        raise SystemExit(f"{len(turns) - len(seen)} duplicate turn record(s) across chunks; "
                         "the chunk files overlap and the merge would double-count")

    questions = len({(t.category, t.question_id) for t in turns})
    extra = dict(meta)
    extra.update({
        "questions": questions,
        "chunks_merged": len(chunks),
        "turns_merged": len(turns),
    })
    path = save_results(destination, turns, extra)

    blob = json.loads(path.read_text(encoding="utf-8"))
    blob["speedups"] = speedup_table(blob["summary"], "target", "tokens_per_second")
    blob["decode_speedups"] = speedup_table(blob["summary"], "target",
                                            "decode_tokens_per_second")
    path.write_text(json.dumps(blob, indent=2), encoding="utf-8")

    print(f"\nmerged {len(chunks)} chunks, {len(turns)} turns, {questions} questions -> {path}")
    for method in [m for m in blob["speedups"] if m != "target"]:
        wall = blob["speedups"][method].get("all")
        decode = blob["decode_speedups"][method].get("all")
        print(f"  {method:<20} decode {decode:.4f}   wall {wall:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--draft", type=Path, default=None)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--questions", type=Path,
                        default=ROOT / "data" / "spec_bench" / "question.jsonl")
    parser.add_argument("--gamma", type=int, default=1)
    parser.add_argument("--chunk-size", type=int, default=5,
                        help="questions per category per chunk; smaller loses less to a kill")
    parser.add_argument("--methods", nargs="*", default=METHODS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results" / "specbench_full")
    parser.add_argument("--merge-only", action="store_true",
                        help="skip measuring and just rebuild the merged result from the chunks")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    destination = args.out_dir / f"specbench_full_gamma{args.gamma}.json"

    if args.merge_only:
        merge(args.out_dir, args.gamma, destination)
        return

    print("categories and sizes:")
    biggest = largest_category(args.questions)
    offsets = list(range(0, biggest, args.chunk_size))
    todo = [o for o in offsets if not chunk_path(args.out_dir, args.gamma, o,
                                                 args.chunk_size).exists()]
    done = len(offsets) - len(todo)
    # Spell the plan out. --chunk-size is questions *per category per chunk*, not a chunk count,
    # and reading it the other way turns a smoke test into the whole benchmark.
    per_chunk = args.chunk_size * len(CATEGORIES)
    print(f"\nplan: {len(offsets)} chunks x {args.chunk_size} question(s) per category "
          f"= {per_chunk} questions a chunk, {len(offsets) * per_chunk} in total")
    print(f"      gamma={args.gamma}, methods={args.methods}, out={args.out_dir}")
    if done:
        print(f"      {done} chunk(s) already on disk, so {len(todo)} to run")
    if not todo:
        print("\nnothing to measure; merging what is on disk")
        merge(args.out_dir, args.gamma, destination)
        return

    started = time.time()
    failures = []
    for index, offset in enumerate(todo, 1):
        path = chunk_path(args.out_dir, args.gamma, offset, args.chunk_size)
        elapsed = time.time() - started
        eta = f", eta {(elapsed / (index - 1)) * (len(todo) - index + 1) / 60:.0f} min" \
            if index > 1 else ""
        print(f"\n{'=' * 78}\nchunk {index}/{len(todo)}  offset {offset}  -> {path.name}"
              f"  [{elapsed / 60:.0f} min elapsed{eta}]\n{'=' * 78}", flush=True)
        code = run_chunk(args, offset, path)
        if code != 0:
            failures.append((offset, code))
            print(f"!! offset {offset} exited {code}", flush=True)
            if path.exists():  # a partial file would be silently merged later
                path.unlink()
                print(f"   removed the partial {path.name}", flush=True)

    print(f"\n{len(todo) - len(failures)} chunk(s) measured in "
          f"{(time.time() - started) / 60:.0f} min; "
          + (f"failures: {failures}" if failures else "no failures"))

    if failures:
        print("Not merging while chunks are missing. Re-run the same command to retry them.")
        sys.exit(1)
    merge(args.out_dir, args.gamma, destination)


if __name__ == "__main__":
    main()
