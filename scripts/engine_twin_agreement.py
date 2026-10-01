"""How well does the PyTorch twin predict the engine? (plan section 11.6)

    python scripts/engine_twin_agreement.py models/Qwen3-0.6B-q4.sdm models/Qwen3-0.6B

The whole offline evaluation strategy assumes the twin stands in for the engine: acceptance
measured on a GPU has to be what the engine will actually accept. The two cannot agree
elementwise, because 8-bit activation quantization is a step function and a 1-ulp difference
in the preceding float arithmetic flips a few hundred levels per layer. What has to agree is
the *distributions*: this script measures top-1 agreement and total variation distance, which
are exactly the quantities acceptance is computed from.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from specdraft import _engine as cpp
from specdraft.twin import QuantizedTwin


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("engine_file", type=Path, help="a .sdm weights file")
    parser.add_argument("model", type=Path, help="the Hugging Face directory it came from")
    parser.add_argument("--tokens", type=int, default=128)
    parser.add_argument("--weight-format", default="q4")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--out", type=Path, default=Path("results"))
    args = parser.parse_args()

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    text = (
        "Speculative decoding runs a small draft model several steps ahead and then checks "
        "its guesses with one pass of the larger model. The trick is that verifying several "
        "tokens at once costs little more than producing one, so the draft's work is almost "
        "free when its guesses are right. On a laptop CPU the picture changes, because the "
        "verification pass is no longer nearly free: weights are read once but the integer "
        "arithmetic grows with every extra token checked."
    )
    ids = tokenizer(text, return_tensors="pt").input_ids[0][: args.tokens]
    print(f"{len(ids)} tokens")

    engine = cpp.Model(str(args.engine_file), max_positions=len(ids) + 8)
    engine_logits = engine.forward(ids.numpy().astype(np.int32), all_logits=True)

    twin = QuantizedTwin.from_pretrained(args.model, weight_format=args.weight_format)
    with torch.no_grad():
        twin_logits = twin.forward(ids, cache=twin.new_cache(len(ids) + 8))[
            :, : engine_logits.shape[1]
        ].numpy()

    mine = torch.from_numpy(engine_logits)
    theirs = torch.from_numpy(twin_logits)
    top1 = float((mine.argmax(-1) == theirs.argmax(-1)).float().mean())
    top5_overlap = float(
        np.mean(
            [
                len(set(a.tolist()) & set(b.tolist())) / 5
                for a, b in zip(mine.topk(5, -1).indices, theirs.topk(5, -1).indices)
            ]
        )
    )
    p = torch.softmax(mine.float() / args.temperature, dim=-1)
    q = torch.softmax(theirs.float() / args.temperature, dim=-1)
    tvd = float((0.5 * (p - q).abs().sum(-1)).mean())
    relative = float(np.linalg.norm(engine_logits - twin_logits) / np.linalg.norm(twin_logits))

    record = {
        "engine_file": str(args.engine_file),
        "weight_format": args.weight_format,
        "tokens": int(len(ids)),
        "top1_agreement": top1,
        "top5_overlap": top5_overlap,
        "mean_tvd": tvd,
        "logit_relative_error": relative,
        "temperature": args.temperature,
        "when": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    print(f"top-1 agreement       {top1:.4f}")
    print(f"top-5 overlap         {top5_overlap:.4f}")
    print(f"mean TVD at T={args.temperature}      {tvd:.5f}  "
          f"(an acceptance rate measured on the twin is off by about this much)")
    print(f"logit relative error  {relative:.3e}  (cannot be small: quantization is a step function)")

    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / f"engine_twin_agreement_{args.engine_file.stem}.json"
    path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
