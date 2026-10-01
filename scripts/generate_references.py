"""Generate the target's own continuations, which everything offline is scored against.

    python scripts/generate_references.py --model models/Qwen3-4B-q4.sdm \
        --tokenizer models/Qwen3-4B --prompts data/dev_prompts.jsonl \
        --mode greedy --max-new-tokens 256 --out data/references_greedy.jsonl

Why the text has to come from the target, and at the settings being studied:

* **Greedy.** The offline simulation is exact only because the speculative output *is* the
  target's greedy continuation, so the draft's context at every position is a prefix of it.
  Reference text from anywhere else makes the simulation an estimate of nothing in particular.
* **Sampling.** Exactness in distribution needs the reference drawn from the target at the same
  temperature and filters, so one file is needed per decoding mode the write-up reports.

Prompts are plain records (``prompt`` fields); the output is the sequence format the trainer and
the offline scorer read, with the boundary between prompt and continuation recorded.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from specdraft.data import read_records
from specdraft.train import TrainedSequence


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True,
                        help="a .sdm engine file, or a Hugging Face directory")
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--prompts", type=Path, required=True, help="jsonl of records")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--mode", default="greedy", choices=["greedy", "sampling"])
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--max-prompt-tokens", type=int, default=1024)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--weight-format", default="q4", choices=["q4", "q8", None],
                        help="only when --model is a Hugging Face directory")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--cores", default="performance")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    records = read_records(args.prompts)[: args.limit]
    if not records:
        raise SystemExit(f"no prompts in {args.prompts}")

    capacity = args.max_prompt_tokens + args.max_new_tokens + 8
    temperature = 0.0 if args.mode == "greedy" else args.temperature

    if args.model.suffix == ".sdm":
        from specdraft import _engine as cpp

        engine = cpp.Model(str(args.model), max_positions=capacity, cores=args.cores)
        print(f"{args.model.name} in the engine on {engine.threads} threads")

        def generate(prompt_ids: list[int], seed: int) -> list[int]:
            tokens, _ = cpp.generate_plain(
                engine, prompt_ids, max_new_tokens=args.max_new_tokens,
                temperature=temperature, top_k=args.top_k, top_p=args.top_p,
                stop=[tokenizer.eos_token_id], seed=seed,
            )
            return tokens
    else:
        import torch

        from specdraft.reference import Qwen3Reference
        from specdraft.sampling import GREEDY, SamplingConfig
        from specdraft.speculative import plain_generate
        from specdraft.twin import QuantizedTwin

        if args.weight_format is None:
            model = Qwen3Reference.from_pretrained(args.model, device=args.device)
            print(f"{args.model.name} at full precision")
        else:
            model = QuantizedTwin.from_pretrained(
                args.model, device=args.device, weight_format=args.weight_format
            )
            print(f"{args.model.name} as the engine runs it ({model.describe()})")
        config = GREEDY if args.mode == "greedy" else SamplingConfig(
            temperature=args.temperature, top_k=args.top_k, top_p=args.top_p
        )

        def generate(prompt_ids: list[int], seed: int) -> list[int]:
            generator = torch.Generator().manual_seed(seed)
            tokens, _ = plain_generate(
                model, prompt_ids, args.max_new_tokens, config=config,
                vocab_limit=len(tokenizer), stop={tokenizer.eos_token_id}, generator=generator,
            )
            return tokens

    started = time.perf_counter()
    written = 0
    produced = 0
    with args.out.open("w", encoding="utf-8") as handle:
        for index, record in enumerate(records):
            prompt_text = tokenizer.apply_chat_template(
                [{"role": "user", "content": record.prompt}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,  # both models decode this way
            )
            prompt_ids = tokenizer(prompt_text, add_special_tokens=False).input_ids
            if len(prompt_ids) > args.max_prompt_tokens:
                continue
            prompt_ids = [int(token) for token in prompt_ids]

            continuation = generate(prompt_ids, args.seed + index)
            if not continuation:
                continue
            sequence = TrainedSequence(
                tokens=prompt_ids + [int(token) for token in continuation],
                response_start=len(prompt_ids),
                source=record.source or "",
            )
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
            produced += len(continuation)

            elapsed = time.perf_counter() - started
            if written % 5 == 0:
                print(f"  {written}/{len(records)}  {produced} tokens  "
                      f"{produced / elapsed:.1f} tok/s", flush=True)

    elapsed = time.perf_counter() - started
    print(f"\nwrote {written} sequences ({produced} generated tokens) to {args.out} "
          f"in {elapsed / 60:.1f} min at {produced / max(elapsed, 1e-9):.1f} tok/s")
    print(f"mode {args.mode}"
          + ("" if args.mode == "greedy" else
             f" at temperature {args.temperature}, top-k {args.top_k}, top-p {args.top_p}")
          + ": score only this mode against these references")


if __name__ == "__main__":
    main()
