"""A balanced prompt set for the acceptance measurement, drawn from Spec-Bench.

    python scripts/build_accept_prompts.py --per-category 4 --out data/accept_prompts.jsonl

The offline scorer groups its results by each record's ``source`` field, so putting the Spec-Bench
category there means one scoring run reports acceptance per category. That breakdown is the point:
a draft that copies well from a supplied document and reasons badly has a very different best gamma
per task, and an overall average would hide it.

Two details. The MT-Bench half of Spec-Bench is spread over eight subcategories which Spec-Bench
treats as one ``multiturn`` task, so one prompt is taken from each of four of them rather than four
from one. And ``rag`` and ``summarization`` prompts carry documents of 400-1500 tokens; prompts over
the cap are skipped, since prefill dominates the cost of generating references.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

CATEGORIES = ("multiturn", "translation", "summarization", "qa", "math_reasoning", "rag")
# The MT-Bench subcategories to draw the `multiturn` prompts from, one each.
MULTITURN = ("writing", "coding", "reasoning", "extraction")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--questions", type=Path, default=Path("data/spec_bench/question.jsonl"),
                        help="from scripts/fetch_specbench.py")
    parser.add_argument("--tokenizer", default="models/Qwen3-4B")
    parser.add_argument("--per-category", type=int, default=4)
    parser.add_argument("--prompt-cap", type=int, default=800,
                        help="skip prompts longer than this many tokens, chat template included")
    parser.add_argument("--out", type=Path, default=Path("data/accept_prompts.jsonl"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.questions.exists():
        raise SystemExit(f"{args.questions} not found; run scripts/fetch_specbench.py first")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    questions = [json.loads(line) for line in
                 args.questions.read_text(encoding="utf-8").splitlines() if line.strip()]

    def length(prompt: str) -> int:
        text = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False,
                                             add_generation_prompt=True, enable_thinking=False)
        return len(tokenizer(text, add_special_tokens=False).input_ids)

    picked = []
    for category in CATEGORIES:
        if category == "multiturn":
            # Round-robin over the subcategories, so --per-category 4 gives one of each kind of
            # task rather than four of the first kind.
            buckets = [[question for question in questions if question["category"] == sub]
                       for sub in MULTITURN]
            pool = [question for group in zip(*buckets) for question in group]
        else:
            pool = [question for question in questions if question["category"] == category]

        taken = 0
        for question in pool:
            prompt = question["turns"][0]
            tokens = length(prompt)
            if tokens > args.prompt_cap:
                continue
            picked.append({"prompt": prompt, "source": category})
            print(f"  {category:>16} id={question['question_id']:<5} {tokens:>5} tokens")
            taken += 1
            if taken == args.per_category:
                break
        if taken < args.per_category:
            print(f"  only {taken} of {args.per_category} for {category} fit under "
                  f"{args.prompt_cap} tokens")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as handle:
        for record in picked:
            print(json.dumps(record), file=handle)
    total = sum(length(record["prompt"]) for record in picked)
    print(f"\nwrote {len(picked)} prompts to {args.out}: {total} prompt tokens in all")
    print("generating references costs mostly prefill, so that total is the figure to watch")


if __name__ == "__main__":
    main()
