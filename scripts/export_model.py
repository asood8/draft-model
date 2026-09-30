"""Write the engine's weights file from a Hugging Face model directory.

    python scripts/export_model.py models/Qwen3-0.6B --format q4
    python scripts/export_model.py models/Qwen3-0.6B --format q4 --output-format q8

The vocabulary limit is taken from the tokenizer when one is present, so the engine never
computes or samples the padded embedding rows.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from specdraft.export import read_model, write_model
from specdraft.reference import Qwen3Config, load_safetensors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path, help="model directory, e.g. models/Qwen3-0.6B")
    parser.add_argument("--format", default="q4", choices=["q4", "q8", "fp32"])
    parser.add_argument("--output-format", default=None, choices=[None, "q4", "q8", "fp32"])
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    config = Qwen3Config.from_pretrained(args.model)
    state_dict = {k: v.to(torch.float32) for k, v in load_safetensors(args.model).items()}

    vocab_limit = config.vocab_size
    if (args.model / "tokenizer.json").is_file():
        from transformers import AutoTokenizer

        vocab_limit = len(AutoTokenizer.from_pretrained(args.model))

    suffix = args.format + (f"-{args.output_format}out" if args.output_format else "")
    out = args.out or args.model.parent / f"{args.model.name}-{suffix}.sdm"
    entries = write_model(
        out,
        state_dict,
        config,
        weight_format=args.format,
        output_format=args.output_format,
        vocab_limit=vocab_limit,
    )

    parameters = sum(
        int(torch.tensor(e.shape).prod()) for e in entries if e.format != "fp32"
    )
    quantized_bytes = sum(e.nbytes for e in entries if e.format != "fp32")
    size = out.stat().st_size
    print(f"{out}  {size / 1e6:.1f} MB  ({len(entries)} tensors, vocab limit {vocab_limit})")
    if parameters:
        print(f"quantized weights: {8 * quantized_bytes / parameters:.2f} bits each")

    # Read the directory back, so a broken file is caught here and not in C++.
    read_model(out)


if __name__ == "__main__":
    main()
