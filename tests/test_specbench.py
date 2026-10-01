"""The Spec-Bench harness (plan §9.3).

No benchmark file is needed: the loader is checked against synthetic questions, and the runner
against the real engine on a tiny model, so the aggregation that produces the headline table is
exercised without an 8 GB download.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from specdraft.specbench import (  # noqa: E402
    CATEGORIES,
    Question,
    TurnResult,
    aggregate,
    load_questions,
    normalize_category,
    run_benchmark,
    save_results,
    speedup_table,
)

MODEL_DIR = Path(os.environ.get("SPECDRAFT_DRAFT_MODEL", "models/Qwen3-0.6B"))


# --------------------------------------------------------------------------- loading


def write_questions(path: Path, rows: list[dict]) -> Path:
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    return path


def test_category_names_are_normalized():
    assert normalize_category("MT-Bench") == "multi_turn"
    assert normalize_category("math_reasoning") == "math"
    assert normalize_category("CNN_DM") == "summarization"
    assert normalize_category("something_else") == "something_else"  # unknown names pass through
    assert set(CATEGORIES) <= {normalize_category(name) for name in CATEGORIES}


def test_loading_handles_the_shapes_a_question_file_takes(tmp_path):
    path = write_questions(
        tmp_path / "q.jsonl",
        [
            {"question_id": 1, "category": "mt_bench", "turns": ["first", "second"]},
            {"question_id": 2, "category": "translation", "prompt": "translate this"},
            {"question_id": 3, "category": "math", "question": "what is 2+2"},
            {},  # no usable text: skipped
        ],
    )
    questions = load_questions(path)
    assert [q.question_id for q in questions] == [1, 2, 3]
    assert questions[0].category == "multi_turn" and len(questions[0].turns) == 2
    assert questions[1].turns == ["translate this"]
    assert questions[2].turns == ["what is 2+2"]


def test_filtering_and_subsetting(tmp_path):
    rows = [
        {"question_id": i, "category": "math" if i % 2 else "qa", "turns": [f"q{i}"]}
        for i in range(10)
    ]
    path = write_questions(tmp_path / "q.jsonl", rows)

    assert {q.category for q in load_questions(path, categories=["math"])} == {"math"}

    subset = load_questions(path, limit_per_category=2)
    assert len(subset) == 4  # two from each category
    # A fixed seed means a short run is the same short run next time.
    assert [q.question_id for q in subset] == [
        q.question_id for q in load_questions(path, limit_per_category=2)
    ]


# ------------------------------------------------------------------------ aggregation


def make_result(method: str, category: str, **kwargs) -> TurnResult:
    defaults = dict(
        method=method,
        category=category,
        question_id=0,
        turn=0,
        tokens=100,
        seconds=1.0,
        target_forwards=50,
        proposed=200,
        accepted=150,
        rejections=50,
        rounds=50,
        tokens_per_target_forward=2.0,
        tokens_per_second=100.0,
        alpha=0.75,
    )
    defaults.update(kwargs)
    return TurnResult(**defaults)


def test_aggregation_pools_tokens_and_takes_median_speeds():
    results = [
        make_result("spec", "math", tokens=100, target_forwards=50, tokens_per_second=10.0),
        make_result("spec", "math", tokens=300, target_forwards=100, tokens_per_second=30.0),
        make_result("spec", "qa", tokens=50, target_forwards=50, tokens_per_second=20.0),
    ]
    summary = aggregate(results)

    math = summary["spec"]["math"]
    assert math["turns"] == 2
    assert math["tokens_per_target_forward"] == pytest.approx(400 / 150)  # pooled, not averaged
    assert math["tokens_per_second"] == pytest.approx(20.0)  # median of 10 and 30
    assert summary["spec"]["all"]["turns"] == 3
    assert summary["spec"]["all"]["tokens"] == 450


def test_alpha_comes_from_the_engines_counts():
    results = [make_result("spec", "math", accepted=30, rejections=10, rounds=10)]
    assert aggregate(results)["spec"]["math"]["alpha"] == pytest.approx(0.75)
    assert aggregate(results)["spec"]["math"]["mean_accepted_per_round"] == pytest.approx(3.0)


def test_speedups_are_relative_to_the_baseline():
    results = [
        make_result("target", "math", tokens_per_second=10.0),
        make_result("spec", "math", tokens_per_second=25.0),
    ]
    table = speedup_table(aggregate(results), baseline="target")
    assert table["target"]["math"] == pytest.approx(1.0)
    assert table["spec"]["math"] == pytest.approx(2.5)

    with pytest.raises(ValueError):
        speedup_table(aggregate(results), baseline="absent")


def test_results_are_saved_with_their_summary(tmp_path):
    path = save_results(
        tmp_path / "out.json", [make_result("spec", "math")], extra={"gamma": 4}
    )
    blob = json.loads(path.read_text(encoding="utf-8"))
    assert blob["gamma"] == 4
    assert blob["summary"]["spec"]["math"]["turns"] == 1
    assert len(blob["turns"]) == 1


# ----------------------------------------------------- the runner, against the engine


@pytest.mark.skipif(
    not (MODEL_DIR / "tokenizer.json").is_file(), reason=f"no tokenizer at {MODEL_DIR}"
)
def test_the_runner_drives_the_engine(tmp_path):
    """Multi-turn questions must be run as conversations, and every method must report τ."""
    import dataclasses

    from transformers import AutoTokenizer

    from specdraft import _engine as cpp
    from specdraft.export import write_model
    from specdraft.reference import TINY_CONFIG, perturbed_copy, random_reference

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    config = dataclasses.replace(TINY_CONFIG, vocab_size=151_680)
    target_reference = random_reference(config, seed=0)
    draft_reference = perturbed_copy(target_reference, sigma=0.02, seed=1)

    paths = {}
    for name, model in (("target", target_reference), ("draft", draft_reference)):
        path = tmp_path / f"{name}.sdm"
        write_model(path, model.state_dict(), config, weight_format="q8", vocab_limit=151_669)
        paths[name] = path

    target = cpp.Model(str(paths["target"]), max_positions=512, max_batch=5)
    draft = cpp.Model(str(paths["draft"]), max_positions=512, max_batch=5)
    common = {"max_new_tokens": 8, "stop": [tokenizer.eos_token_id]}
    methods = {
        "target": lambda prompt: cpp.generate_plain(target, prompt, **common),
        "speculative": lambda prompt: cpp.generate_speculative(
            target, draft, prompt, gamma=4, **common
        ),
        "lookup": lambda prompt: cpp.generate_prompt_lookup(target, prompt, gamma=4, **common),
    }

    questions = [
        Question(1, "multi_turn", ["Hello there.", "And what about afterwards?"]),
        Question(2, "math", ["What is two plus two?"]),
    ]
    results = run_benchmark(questions, methods, tokenizer, max_new_tokens=8, keep_text=True)

    assert len(results) == len(methods) * 3  # two turns plus one
    assert {result.turn for result in results if result.question_id == 1} == {0, 1}
    for result in results:
        assert 0 < result.tokens <= 8
        assert result.target_forwards > 0
        assert result.tokens_per_target_forward > 0

    summary = aggregate(results)
    assert set(summary) == set(methods)
    assert summary["lookup"]["all"]["mean_proposed_per_round"] >= 0.0
    # Speculative decoding must never need more target passes than plain decoding.
    assert (
        summary["speculative"]["all"]["tokens_per_target_forward"]
        >= summary["target"]["all"]["tokens_per_target_forward"]
    )
