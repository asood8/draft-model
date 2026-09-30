"""WikiText-2 perplexity for the reference model and its quantized twins (plan §7.2).

    python scripts/perplexity.py models/Qwen3-0.6B --windows 8

Each configuration is one row of the Milestone 1 table: what the engine's number formats
cost in quality. Configurations are built one at a time, because two fp32 copies of a model
plus their quantized weights is already several gigabytes.

Perplexity is measured over non-overlapping windows with the padded embedding rows trimmed
away, so the numbers are comparable to each other rather than to any published figure.
"""

from __future__ import annotations

import argparse
import gc
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

from specdraft.reference import Qwen3Config, Qwen3Reference, load_safetensors
from specdraft.twin import QuantizedTwin

# name -> QuantizedTwin keyword arguments; None means the plain fp32 reference.
CONFIGURATIONS: dict[str, dict | None] = {
    "fp32": None,
    "q8": {"weight_format": "q8", "activation_format": None, "kv_dtype": None},
    "q4": {"weight_format": "q4", "activation_format": None, "kv_dtype": None},
    "q4 + q8 output": {
        "weight_format": "q4",
        "output_format": "q8",
        "activation_format": None,
        "kv_dtype": None,
    },
    "engine twin (q4 + a8 + fp16 kv)": {
        "weight_format": "q4",
        "activation_format": "a8",
        "kv_dtype": torch.float16,
    },
}


def load_wikitext_tokens(
    tokenizer,
    limit_tokens: int,
    repo: str = "Salesforce/wikitext",
    config: str = "wikitext-2-raw-v1",
) -> torch.Tensor:
    from datasets import load_dataset

    data = load_dataset(repo, config, split="test")
    text = "\n\n".join(line for line in data["text"] if line.strip())
    ids = tokenizer(text, return_tensors="pt").input_ids[0]
    return ids[:limit_tokens]


@torch.no_grad()
def perplexity(model, ids: torch.Tensor, window: int, vocab_limit: int, chunk: int = 128) -> tuple[float, float]:
    """Returns (perplexity, tokens per second)."""
    total_nll = 0.0
    total_tokens = 0
    started = time.perf_counter()

    for begin in range(0, len(ids) - 1, window):
        piece = ids[begin : begin + window]
        if len(piece) < 2:
            break
        hidden = model.forward(piece, hidden_only=True)
        for start in range(0, len(piece) - 1, chunk):
            stop = min(start + chunk, len(piece) - 1)
            logits = model.logits_from_hidden(hidden[start:stop])[:, :vocab_limit]
            log_probs = torch.log_softmax(logits.to(torch.float32), dim=-1)
            wanted = piece[start + 1 : stop + 1, None]
            total_nll -= float(log_probs.gather(-1, wanted).sum())
            total_tokens += stop - start
        del hidden

    elapsed = time.perf_counter() - started
    return float(torch.tensor(total_nll / total_tokens).exp()), total_tokens / elapsed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path, help="model directory, e.g. models/Qwen3-0.6B")
    parser.add_argument("--window", type=int, default=1024, help="tokens per window")
    parser.add_argument("--windows", type=int, default=8, help="how many windows to score")
    parser.add_argument("--configs", nargs="*", default=None, help="subset of configurations")
    parser.add_argument("--out", type=Path, default=Path("results"))
    parser.add_argument(
        "--dataset", default="Salesforce/wikitext", help="dataset repo id (namespace/name)"
    )
    parser.add_argument("--dataset-config", default="wikitext-2-raw-v1")
    args = parser.parse_args()

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    vocab_limit = len(tokenizer)
    ids = load_wikitext_tokens(
        tokenizer, args.window * args.windows, args.dataset, args.dataset_config
    )
    print(f"{len(ids)} tokens, vocab limit {vocab_limit}")

    config = Qwen3Config.from_pretrained(args.model)
    state_dict = {k: v.to(torch.float32) for k, v in load_safetensors(args.model).items()}

    wanted = args.configs or list(CONFIGURATIONS)
    rows = []
    for name in wanted:
        kwargs = CONFIGURATIONS[name]
        model = (
            Qwen3Reference(config, state_dict)
            if kwargs is None
            else QuantizedTwin(config, state_dict, **kwargs)
        )
        ppl, speed = perplexity(model, ids, args.window, vocab_limit)
        print(f"{name:34s} ppl {ppl:8.3f}   {speed:6.1f} tok/s")
        rows.append({"config": name, "perplexity": ppl, "tokens_per_second": speed})
        del model
        gc.collect()

    args.out.mkdir(parents=True, exist_ok=True)
    record = {
        "model": str(args.model),
        "window": args.window,
        "windows": args.windows,
        "tokens": int(len(ids)),
        "vocab_limit": vocab_limit,
        "rows": rows,
        "torch": torch.__version__,
        "cpu": platform.processor(),
        "when": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    path = args.out / f"perplexity_{args.model.name}.json"
    path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
