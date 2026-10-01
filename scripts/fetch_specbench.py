"""Fetch Spec-Bench's question file.

    python scripts/fetch_specbench.py

Spec-Bench (Xia et al., ACL Findings 2024) is a set of prompts in six categories chosen because
speculative decoding behaves very differently across them. The questions are a small JSON-lines
file in the authors' repository; the models and code are not needed, only the prompts.

Nothing here is bundled because it is someone else's dataset. If the layout has moved, the error
message says where to look, and any JSON-lines file with `category` and `turns` fields works just as
well with the rest of this project.
"""

from __future__ import annotations

import argparse
import json
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path

CANDIDATES = (
    "https://raw.githubusercontent.com/hemingkx/Spec-Bench/main/data/spec_bench/question.jsonl",
    "https://raw.githubusercontent.com/hemingkx/Spec-Bench/master/data/spec_bench/question.jsonl",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("data/spec_bench/question.jsonl"))
    parser.add_argument("--url", default=None, help="override, if the repository has moved")
    args = parser.parse_args()

    urls = (args.url,) if args.url else CANDIDATES
    body = None
    for url in urls:
        try:
            print(f"trying {url}")
            with urllib.request.urlopen(url, timeout=60) as response:
                body = response.read()
            break
        except (urllib.error.URLError, urllib.error.HTTPError) as exc:
            print(f"  {type(exc).__name__}: {exc}")

    if body is None:
        raise SystemExit(
            "could not fetch the question file. Take data/spec_bench/question.jsonl from "
            "github.com/hemingkx/Spec-Bench by hand, or point --questions at any JSON-lines file "
            "with 'category' and 'turns' fields."
        )

    text = body.decode("utf-8")
    rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    if not rows or "turns" not in rows[0]:
        raise SystemExit("that file does not look like Spec-Bench questions (no 'turns' field)")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text, encoding="utf-8")

    from specdraft.specbench import normalize_category

    counts = Counter(normalize_category(str(row.get("category", "?"))) for row in rows)
    turns = Counter(len(row["turns"]) for row in rows)
    print(f"\nwrote {args.out}: {len(rows)} questions")
    for category, count in sorted(counts.items()):
        print(f"  {category:<16} {count}")
    print(f"turns per question: {dict(sorted(turns.items()))}")


if __name__ == "__main__":
    main()
