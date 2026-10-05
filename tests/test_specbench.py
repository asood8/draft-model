"""The Spec-Bench harness (plan §9.3).

No benchmark file is needed: the loader is checked against synthetic questions, and the runner
against the real engine on a tiny model, so the aggregation that produces the headline table is
exercised without an 8 GB download.
"""

from __future__ import annotations

import json
import os
import types
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
    run_question,
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
        prefill_seconds=0.4,
        decode_seconds=0.6,
        decode_tokens_per_second=166.0,
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


def test_decode_speed_is_reported_separately_from_the_whole_call():
    """Speculative decoding changes decoding, so the speedup that matters excludes the prompt pass.

    On a long prompt the prompt pass is most of the wall clock -- 93% of it on a summarization
    question here -- and a method that prefills a second model pays for it twice, so a whole-call
    speedup reports mostly prefill and hides what the method did to generation.
    """
    results = [
        make_result("target", "rag", tokens=100, seconds=10.0, prefill_seconds=9.0,
                    decode_seconds=1.0, decode_tokens_per_second=100.0, tokens_per_second=10.0),
        make_result("spec", "rag", tokens=100, seconds=10.5, prefill_seconds=10.0,
                    decode_seconds=0.5, decode_tokens_per_second=200.0, tokens_per_second=9.5),
    ]
    summary = aggregate(results)
    assert summary["target"]["rag"]["prefill_share"] == pytest.approx(0.9)

    wall = speedup_table(summary, baseline="target")
    decode = speedup_table(summary, baseline="target", metric="decode_tokens_per_second")
    assert wall["spec"]["rag"] == pytest.approx(0.95)  # looks like a loss
    assert decode["spec"]["rag"] == pytest.approx(2.0)  # was twice as fast at the thing it changes


def test_a_turn_records_what_the_prompt_pass_cost(monkeypatch):
    """run_question has to carry the engine's prefill split through, not recompute it."""
    stats = {"seconds": 4.0, "prefill_seconds": 3.0, "tokens_per_second": 2.0,
             "tokens_per_target_forward": 1.0, "alpha": 0.5}

    class Tokenizer:
        def apply_chat_template(self, conversation, **kwargs):
            return "prompt"

        def __call__(self, text, **kwargs):
            return types.SimpleNamespace(input_ids=[1, 2, 3])

        def decode(self, tokens, **kwargs):
            return "answer"

    question = Question(question_id=0, category="qa", turns=["hello"])
    rows = run_question(question, lambda prompt: ([7, 8], stats), "spec", Tokenizer(),
                        keep_text=False)
    row = rows[0]
    assert row.prefill_seconds == pytest.approx(3.0)
    assert row.decode_seconds == pytest.approx(1.0)
    assert row.decode_tokens_per_second == pytest.approx(2.0)  # two tokens in one second


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

def test_mt_bench_questions_are_grouped_by_their_turn_count(tmp_path):
    """Spec-Bench labels its MT-Bench portion with MT-Bench's own eight sub-categories, and its
    "math" label covers both MT-Bench's maths questions and GSM8K's. Two turns means MT-Bench,
    which is the only reliable way to recover the six categories the benchmark reports."""
    rows = [
        {"question_id": 1, "category": "coding", "turns": ["write code", "now improve it"]},
        {"question_id": 2, "category": "math", "turns": ["2+2?", "and 3+3?"]},  # MT-Bench maths
        {"question_id": 3, "category": "math", "turns": ["a word problem"]},  # GSM8K
        {"question_id": 4, "category": "translation", "turns": ["translate"]},
    ]
    path = write_questions(tmp_path / "q.jsonl", rows)

    grouped = {q.question_id: q.category for q in load_questions(path)}
    assert grouped == {1: "multi_turn", 2: "multi_turn", 3: "math", 4: "translation"}

    ungrouped = {q.question_id: q.category for q in load_questions(path, group_multi_turn=False)}
    assert ungrouped[1] == "coding" and ungrouped[2] == "math"


@pytest.mark.skipif(
    not Path("data/spec_bench/question.jsonl").is_file(),
    reason="run scripts/fetch_specbench.py first",
)
def test_the_real_question_file_has_the_six_categories():
    """The benchmark is 480 questions in six categories of 80; if that is not what came out, either
    the file changed or the grouping is wrong."""
    from collections import Counter

    questions = load_questions("data/spec_bench/question.jsonl")
    counts = Counter(question.category for question in questions)
    assert len(questions) == 480
    assert set(counts) == set(CATEGORIES)
    assert set(counts.values()) == {80}


# --------------------------------------------------- the figures built from these results


def load_make_plots():
    """make_plots is a script, so it is imported by path, the way test_train_script does it."""
    import sys

    scripts = str(Path(__file__).resolve().parent.parent / "scripts")
    sys.path.insert(0, scripts)
    try:
        import make_plots

        return make_plots
    finally:
        sys.path.remove(scripts)


def write_run(path: Path, gamma: int, speedup: float, tau: float, decode: bool) -> None:
    blob = {
        "gamma": gamma,
        "target": "models/Qwen3-4B-q4.sdm",
        "speedups": {"speculative": {"all": 0.8}},  # the whole-call figure, deliberately different
        "summary": {"speculative": {"all": {"tokens_per_target_forward": tau, "alpha": 0.74}}},
    }
    if decode:
        blob["decode_speedups"] = {"speculative": {"all": speedup}}
    path.write_text(json.dumps(blob), encoding="utf-8")


def test_the_gamma_figure_reads_decode_speedups_and_falls_back(tmp_path):
    """A figure comparing against the cost model has to read the decode column.

    The model says nothing about a prompt pass, so comparing it against a whole-call speedup compares
    it against something it never claimed. Older result files have only the whole-call number, and are
    read rather than skipped, but flagged.
    """
    plots = load_make_plots()
    write_run(tmp_path / "specbench_gamma1.json", 1, 1.20, 1.72, decode=True)
    write_run(tmp_path / "specbench_gamma3.json", 3, 1.05, 2.68, decode=False)
    write_run(tmp_path / "specbench_short.json", 1, 9.9, 1.72, decode=True)  # another prompt set

    runs = plots.measured_runs(tmp_path)
    assert [run["gamma"] for run in runs] == [1, 3]  # sorted, and the short-prompt set left out
    assert runs[0]["decode"] == pytest.approx(1.20)
    assert runs[0]["decode_only"] is True
    assert runs[1]["decode"] == pytest.approx(0.8)  # fell back to the whole call
    assert runs[1]["decode_only"] is False


def test_the_prediction_uses_v_at_gamma_plus_one():
    """One pass verifies gamma guesses plus the token before them, so the cost is v(gamma+1)."""
    plots = load_make_plots()
    blob = {"v": {"2": 1.12, "3": 1.34}, "c": 0.186, "overhead": {"o_per_round": 0.007}}
    assert plots.predict(blob, 1, 1.72) == pytest.approx(1.72 / (0.186 + 1.12 + 0.007))
    assert plots.predict(blob, 2, 2.28) == pytest.approx(2.28 / (2 * 0.186 + 1.34 + 0.007))
    assert plots.predict({"c": 0.2}, 1, 1.7) is None  # no v(k) measured, no prediction
