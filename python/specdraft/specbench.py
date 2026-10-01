"""Running Spec-Bench through the engine (plan §9.3).

Spec-Bench is a set of prompts in six categories, chosen because speculative decoding behaves
very differently across them: copying from the prompt is nearly unbeatable on summarization,
while math gives a draft little to work with. Every headline number in the write-up comes from
here.

The file itself is not bundled. Point ``--questions`` at Spec-Bench's ``question.jsonl`` (from
github.com/hemingkx/Spec-Bench), or at any JSON-lines file with ``category`` and ``turns`` fields.
Category names are normalized, since different releases spell them differently.

Multi-turn questions are run as conversations: the second turn is asked after the model's *own*
answer to the first, which is what makes the category a fair test rather than a pair of unrelated
prompts.
"""

from __future__ import annotations

import json
import random
import statistics
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Sequence

# What the method is handed (prompt token ids) and what it returns (tokens, statistics).
MethodFn = Callable[[list[int]], tuple[list[int], dict]]

# Spec-Bench's own file labels its MT-Bench portion with MT-Bench's eight sub-categories rather
# than one name, and its "math" label covers both MT-Bench's maths questions and GSM8K's. The
# distinguishing feature is the turn count: the 80 MT-Bench questions are the only two-turn ones,
# which is what `load_questions` groups on.
MT_BENCH_SUBCATEGORIES = (
    "writing", "roleplay", "reasoning", "coding", "extraction", "stem", "humanities",
)

CATEGORY_ALIASES = {
    "mt_bench": "multi_turn",
    "mt-bench": "multi_turn",
    "mtbench": "multi_turn",
    "multiturn": "multi_turn",
    "writing": "multi_turn",
    "translation": "translation",
    "wmt": "translation",
    "summarization": "summarization",
    "cnndm": "summarization",
    "cnn_dm": "summarization",
    "qa": "qa",
    "nq": "qa",
    "question_answering": "qa",
    "math": "math",
    "math_reasoning": "math",
    "gsm8k": "math",
    "rag": "rag",
    "retrieval": "rag",
}

CATEGORIES = ("multi_turn", "translation", "summarization", "qa", "math", "rag")


def normalize_category(name: str) -> str:
    key = name.strip().lower().replace(" ", "_").replace("-", "_")
    return CATEGORY_ALIASES.get(key, key)


@dataclass(frozen=True)
class Question:
    question_id: object
    category: str
    turns: list[str]


def load_questions(
    path: str | Path,
    categories: Sequence[str] | None = None,
    limit_per_category: int | None = None,
    seed: int = 0,
    group_multi_turn: bool = True,
) -> list[Question]:
    """Read the benchmark file, optionally taking a stratified subset.

    A subset is drawn per category with a fixed seed, so a short run is still balanced and is the
    same short run next time.
    """
    questions: list[Question] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        turns = row.get("turns")
        if turns is None:
            single = row.get("prompt") or row.get("question")
            turns = [single] if single else []
        if not turns:
            continue
        # A two-turn question is from MT-Bench, whatever its own label says, and belongs in the
        # multi-turn category the benchmark reports.
        category = normalize_category(str(row.get("category", "unknown")))
        if group_multi_turn and len(turns) > 1:
            category = "multi_turn"
        questions.append(
            Question(
                question_id=row.get("question_id", len(questions)),
                category=category,
                turns=list(turns),
            )
        )

    if categories is not None:
        wanted = {normalize_category(name) for name in categories}
        questions = [q for q in questions if q.category in wanted]

    if limit_per_category is not None:
        by_category: dict[str, list[Question]] = defaultdict(list)
        for question in questions:
            by_category[question.category].append(question)
        picked: list[Question] = []
        for category in sorted(by_category):
            rows = by_category[category]
            random.Random(seed).shuffle(rows)
            picked.extend(rows[:limit_per_category])
        questions = picked

    return questions


@dataclass
class TurnResult:
    method: str
    category: str
    question_id: object
    turn: int
    tokens: int
    seconds: float
    target_forwards: int
    proposed: int
    accepted: int
    rejections: int
    rounds: int
    tokens_per_target_forward: float
    tokens_per_second: float
    alpha: float
    accepted_lengths: list[int] = field(default_factory=list)
    text: str = ""


def run_question(
    question: Question,
    method: MethodFn,
    method_name: str,
    tokenizer,
    max_new_tokens: int = 256,
    enable_thinking: bool = False,
    keep_text: bool = True,
) -> list[TurnResult]:
    """Run every turn of one question, feeding the model's own answers back in."""
    conversation: list[dict[str, str]] = []
    results: list[TurnResult] = []

    for index, turn in enumerate(question.turns):
        conversation.append({"role": "user", "content": turn})
        prompt = tokenizer.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
        prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids
        tokens, stats = method(list(prompt_ids))
        text = tokenizer.decode(tokens, skip_special_tokens=True)
        conversation.append({"role": "assistant", "content": text})

        results.append(
            TurnResult(
                method=method_name,
                category=question.category,
                question_id=question.question_id,
                turn=index,
                tokens=len(tokens),
                seconds=float(stats.get("seconds", 0.0)),
                target_forwards=int(stats.get("target_forwards", 0)),
                proposed=int(stats.get("proposed", 0)),
                accepted=int(stats.get("accepted", 0)),
                rejections=int(stats.get("rejections", 0)),
                rounds=int(stats.get("rounds", 0)),
                tokens_per_target_forward=float(stats.get("tokens_per_target_forward", 0.0)),
                tokens_per_second=float(stats.get("tokens_per_second", 0.0)),
                alpha=float(stats.get("alpha", 0.0)),
                accepted_lengths=list(stats.get("accepted_lengths", [])),
                text=text if keep_text else "",
            )
        )
    return results


def run_benchmark(
    questions: Iterable[Question],
    methods: dict[str, MethodFn],
    tokenizer,
    max_new_tokens: int = 256,
    enable_thinking: bool = False,
    keep_text: bool = False,
    on_result: Callable[[TurnResult], None] | None = None,
) -> list[TurnResult]:
    """Every method on every question.

    Methods are interleaved per question rather than run in blocks, so that a machine which drifts
    while the suite runs does not favour whichever went first (plan §13).
    """
    results: list[TurnResult] = []
    for question in questions:
        for name, method in methods.items():
            for result in run_question(
                question, method, name, tokenizer, max_new_tokens, enable_thinking, keep_text
            ):
                results.append(result)
                if on_result is not None:
                    on_result(result)
    return results


def aggregate(results: Sequence[TurnResult]) -> dict[str, dict[str, dict[str, float]]]:
    """Per method, per category: the numbers the headline table is made of.

    Tokens per target forward is pooled over turns rather than averaged over them, since a long
    answer says more about the method than a short one. Speeds are medians, because a single slow
    turn on a laptop is the machine, not the method.
    """
    grouped: dict[tuple[str, str], list[TurnResult]] = defaultdict(list)
    for result in results:
        grouped[(result.method, result.category)].append(result)
        grouped[(result.method, "all")].append(result)

    out: dict[str, dict[str, dict[str, float]]] = defaultdict(dict)
    for (method, category), rows in grouped.items():
        tokens = sum(row.tokens for row in rows)
        forwards = sum(row.target_forwards for row in rows)
        # Taken from the engine's own counts rather than derived from the accepted-length
        # histogram: with early stopping or prompt lookup a round may offer fewer than gamma
        # guesses, so "accepted < gamma" is not the same thing as "rejected".
        accepted = sum(row.accepted for row in rows)
        rejections = sum(row.rejections for row in rows)
        rounds = sum(row.rounds for row in rows)
        proposed = sum(row.proposed for row in rows)
        out[method][category] = {
            "turns": len(rows),
            "tokens": tokens,
            "tokens_per_target_forward": tokens / forwards if forwards else 0.0,
            "tokens_per_second": statistics.median([row.tokens_per_second for row in rows]),
            "mean_accepted_per_round": accepted / rounds if rounds else 0.0,
            "mean_proposed_per_round": proposed / rounds if rounds else 0.0,
            "alpha": accepted / (accepted + rejections) if (accepted + rejections) else 0.0,
        }
    return dict(out)


def speedup_table(summary: dict[str, dict[str, dict[str, float]]], baseline: str) -> dict:
    """Each method's speed as a multiple of the baseline's, per category."""
    if baseline not in summary:
        raise ValueError(f"no results for the baseline method {baseline!r}")
    table: dict[str, dict[str, float]] = {}
    for method, categories in summary.items():
        table[method] = {}
        for category, numbers in categories.items():
            reference = summary[baseline].get(category, {}).get("tokens_per_second", 0.0)
            table[method][category] = (
                numbers["tokens_per_second"] / reference if reference else float("nan")
            )
    return table


def save_results(path: str | Path, results: Sequence[TurnResult], extra: dict | None = None) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    summary = aggregate(results)
    path.write_text(
        json.dumps(
            {
                "summary": summary,
                "turns": [asdict(result) for result in results],
                **(extra or {}),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return path
