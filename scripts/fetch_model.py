"""Download a model's weights, config and tokenizer into models/<name>.

    python scripts/fetch_model.py Qwen/Qwen3-0.6B      # ~1.5 GB
    python scripts/fetch_model.py Qwen/Qwen3-4B        # ~8 GB

Only the files the project needs are fetched, so no duplicate .bin or .gguf copies.
"""

from __future__ import annotations

import argparse
from pathlib import Path

PATTERNS = [
    "config.json",
    "generation_config.json",
    "*.safetensors",
    "*.safetensors.index.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "chat_template.jinja",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repo", nargs="?", default="Qwen/Qwen3-0.6B", help="Hugging Face repo id")
    parser.add_argument("--out", default=None, help="target directory (default models/<name>)")
    parser.add_argument("--revision", default=None, help="pin a commit or tag")
    args = parser.parse_args()

    from huggingface_hub import snapshot_download

    out = Path(args.out) if args.out else Path("models") / args.repo.split("/")[-1]
    out.mkdir(parents=True, exist_ok=True)
    path = snapshot_download(
        repo_id=args.repo,
        revision=args.revision,
        local_dir=str(out),
        allow_patterns=PATTERNS,
    )
    total = sum(f.stat().st_size for f in Path(path).rglob("*") if f.is_file())
    print(f"{args.repo} -> {path} ({total / 1e9:.2f} GB)")


if __name__ == "__main__":
    main()
