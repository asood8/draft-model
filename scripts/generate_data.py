"""Build the training data for distillation (plan section 11.2 and section 11.3).

Two steps. First the prompt mix, decontaminated against the evaluation set::

    python scripts/generate_data.py prompts --out data/prompts.jsonl \
        --mix ultrachat=20000,code=5000,gsm8k=2500,metamath=2500 \
        --eval-prompts data/spec_bench/question.jsonl

Then the responses, which is what distinguishes the plan's three data sources:

* ``--source fixed`` keeps the responses the datasets already ship (cheapest, off-policy);
* ``--source target --model models/Qwen3-4B`` has the target write them (closest to what the
  draft must predict while decoding);
* ``--source draft --model models/Qwen3-0.6B`` has the draft write them, to be scored by the
  target during training (cheap, and what DistillSpec found works well).

::

    python scripts/generate_data.py responses --prompts data/prompts.jsonl \
        --source target --model models/Qwen3-4B --out data/target_generated.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from specdraft.data import (
    PromptRecord,
    decontaminate,
    generate_responses,
    load_prompt_mix,
    read_records,
    write_records,
)


def parse_mix(text: str) -> dict[str, int]:
    spec = {}
    for piece in text.split(","):
        if not piece.strip():
            continue
        name, _, count = piece.partition("=")
        spec[name.strip()] = int(count)
    return spec


def load_evaluation_prompts(path: Path) -> list[str]:
    """Read the prompts the training set must not overlap.

    Accepts Spec-Bench's question file (one JSON object per line with a "turns" list) as well as
    a plain list of strings or {"prompt": ...} records.
    """
    texts: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if isinstance(row, str):
            texts.append(row)
        elif "turns" in row:
            texts.extend(row["turns"])
        elif "prompt" in row:
            texts.append(row["prompt"])
        elif "question" in row:
            texts.append(row["question"])
    return texts


def build_prompts(args: argparse.Namespace) -> None:
    records = load_prompt_mix(parse_mix(args.mix), seed=args.seed)
    print(f"{len(records)} prompts from {parse_mix(args.mix)}")

    if args.eval_prompts is not None:
        evaluation = load_evaluation_prompts(args.eval_prompts)
        records, removed = decontaminate(records, evaluation, n=args.ngram)
        share = removed / max(1, removed + len(records))
        print(f"decontaminated against {len(evaluation)} evaluation prompts: "
              f"removed {removed} ({share:.2%})")
        if share > 0.05:
            print("  that is a large share; check whether the mix overlaps the benchmark by design")
    else:
        print("no --eval-prompts given, so nothing was decontaminated; "
              "do not report benchmark numbers from a draft trained on this")

    written = write_records(args.out, records)
    print(f"wrote {written} records to {args.out}")


def build_responses(args: argparse.Namespace) -> None:
    records = read_records(args.prompts)
    print(f"{len(records)} prompts")

    if args.source == "fixed":
        kept = [record for record in records if record.complete]
        print(f"{len(kept)} already carry a response; {len(records) - len(kept)} dropped")
        print(f"wrote {write_records(args.out, kept)} records to {args.out}")
        return

    if args.model is None:
        raise SystemExit("--model is required when generating responses")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.padding_side = "left"  # so every continuation starts at the same offset
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.float16 if args.device.startswith("cuda") else torch.float32
    ).to(args.device).eval()

    @torch.no_grad()
    def generate(prompts: list[str]) -> list[str]:
        texts = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,  # both models decode this way, so the data must match
            )
            for prompt in prompts
        ]
        batch = tokenizer(texts, return_tensors="pt", padding=True, truncation=True,
                          max_length=args.max_prompt_tokens).to(args.device)
        out = model.generate(
            **batch,
            max_new_tokens=args.max_new_tokens,
            do_sample=args.temperature > 0,
            temperature=args.temperature if args.temperature > 0 else None,
            top_p=args.top_p if args.temperature > 0 else None,
            pad_token_id=tokenizer.pad_token_id,
        )
        completions = out[:, batch["input_ids"].shape[1] :]
        return tokenizer.batch_decode(completions, skip_special_tokens=True)

    done = generate_responses(
        records, generate, batch_size=args.batch_size, source_suffix=f"-{args.source}"
    )
    print(f"wrote {write_records(args.out, done)} records to {args.out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    prompts = sub.add_parser("prompts", help="build and decontaminate the prompt mix")
    prompts.add_argument("--out", type=Path, required=True)
    prompts.add_argument("--mix", default="ultrachat=20000,code=5000,gsm8k=2500,metamath=2500")
    prompts.add_argument("--eval-prompts", type=Path, default=None)
    prompts.add_argument("--ngram", type=int, default=13)
    prompts.add_argument("--seed", type=int, default=0)
    prompts.set_defaults(run=build_prompts)

    responses = sub.add_parser("responses", help="fill in responses from a model")
    responses.add_argument("--prompts", type=Path, required=True)
    responses.add_argument("--out", type=Path, required=True)
    responses.add_argument("--source", default="target", choices=["fixed", "target", "draft"])
    responses.add_argument("--model", type=Path, default=None)
    responses.add_argument("--device", default="cuda:0")
    responses.add_argument("--batch-size", type=int, default=16)
    responses.add_argument("--max-new-tokens", type=int, default=512)
    responses.add_argument("--max-prompt-tokens", type=int, default=1024)
    responses.add_argument("--temperature", type=float, default=1.0,
                           help="0 for greedy; the plan samples at 1.0 for coverage")
    responses.add_argument("--top-p", type=float, default=1.0)
    responses.set_defaults(run=build_responses)

    args = parser.parse_args()
    args.run(args)


if __name__ == "__main__":
    main()
