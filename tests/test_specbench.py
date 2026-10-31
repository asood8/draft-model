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


def test_a_fixed_order_sweep_is_used_until_an_order_cancelled_one_exists(tmp_path):
    """The fixed-order sweep is superseded, but only once there is something to supersede it.

    Visiting gamma in one order makes drift indistinguishable from the effect: the sweep stored on
    2026-10-05 ran ascending and back to back and read 11-19% low in three of its four entries. That
    is a reason to prefer an order-cancelled sweep, not a reason to draw nothing -- a results tree
    nobody has re-measured still has the fixed-order files as its best available data.
    """
    plots = load_make_plots()
    write_run(tmp_path / "specbench_gamma1.json", 1, 1.20, 1.72, decode=True)
    write_run(tmp_path / "specbench_gamma2.json", 2, 1.01, 2.28, decode=True)

    assert {p.name for p in plots.quotable_sweep(tmp_path)} == {
        "specbench_gamma1.json", "specbench_gamma2.json"}

    # Now add an order-cancelled sweep beside it: two passes at each gamma.
    for label in ("A", "B"):
        write_run(tmp_path / f"specbench_2026-10-25_quiet_pass{label}_gamma1.json",
                  1, 1.19, 1.76, decode=True)
    assert {p.name for p in plots.quotable_sweep(tmp_path)} == {
        "specbench_2026-10-25_quiet_passA_gamma1.json",
        "specbench_2026-10-25_quiet_passB_gamma1.json"}


def test_a_single_pass_sweep_does_not_count_as_order_cancelled(tmp_path):
    """One pass is not a cancelled order, which is the only thing running two passes buys.

    The 2026-10-24 attempt was reaped three runs into eight, leaving pass A alone. Carrying a pass
    marker in the filename is not the same as having a reverse order to cancel against.
    """
    plots = load_make_plots()
    write_run(tmp_path / "specbench_gamma1.json", 1, 1.20, 1.72, decode=True)
    write_run(tmp_path / "specbench_2026-10-24_quiet_passA_gamma1.json", 1, 1.11, 1.70, decode=True)

    # Falls back to the fixed-order sweep rather than trusting a lone pass.
    assert {p.name for p in plots.quotable_sweep(tmp_path)} == {"specbench_gamma1.json"}


def test_passes_are_not_pooled_across_sessions(tmp_path):
    """A lone pass must not ride along on a different session's second pass.

    Grouping by gamma alone counts pass A of one night and both passes of another as three passes at
    that gamma, which both admits the lone run and averages across machine states -- the stitching
    error plan section 13 exists to prevent.
    """
    plots = load_make_plots()
    write_run(tmp_path / "specbench_2026-10-24_quiet_passA_gamma1.json", 1, 1.11, 1.70, decode=True)
    for label in ("A", "B"):
        write_run(tmp_path / f"specbench_2026-10-25_quiet_pass{label}_gamma1.json",
                  1, 1.19, 1.76, decode=True)

    chosen = {p.name for p in plots.quotable_sweep(tmp_path)}
    assert chosen == {"specbench_2026-10-25_quiet_passA_gamma1.json",
                      "specbench_2026-10-25_quiet_passB_gamma1.json"}
    assert all("2026-10-24" not in name for name in chosen)


def test_the_session_covering_more_gammas_wins(tmp_path):
    """Between two order-cancelled sessions, the one that swept further is the sweep."""
    plots = load_make_plots()
    for label in ("A", "B"):
        write_run(tmp_path / f"specbench_2026-10-20_quiet_pass{label}_gamma1.json",
                  1, 1.19, 1.76, decode=True)
        for gamma in (1, 2, 3):
            write_run(tmp_path / f"specbench_2026-10-25_quiet_pass{label}_gamma{gamma}.json",
                      gamma, 1.19, 1.76, decode=True)

    chosen = {p.name for p in plots.quotable_sweep(tmp_path)}
    assert len(chosen) == 6
    assert all("2026-10-25" in name for name in chosen)


def test_measured_runs_averages_the_passes_sharing_a_gamma(tmp_path):
    """Two passes at one gamma are one measurement, so the curve plots their average.

    Taking either pass alone reports whichever one happened to run on the cooler machine, which is
    how the 2026-10-24 reading that gamma=2 beat gamma=1 was produced.
    """
    plots = load_make_plots()
    write_run(tmp_path / "specbench_2026-10-25_quiet_passA_gamma1.json", 1, 1.10, 1.70, decode=True)
    write_run(tmp_path / "specbench_2026-10-25_quiet_passB_gamma1.json", 1, 1.30, 1.80, decode=True)

    runs = plots.measured_runs(tmp_path)
    assert len(runs) == 1
    assert runs[0]["gamma"] == 1
    assert runs[0]["decode"] == pytest.approx(1.20)  # not 1.10 and not 1.30
    assert runs[0]["tau"] == pytest.approx(1.75)
    assert runs[0]["passes_averaged"] == 2


def test_mean_blob_averages_v_c_and_o(tmp_path):
    """v(k) is averaged per k, and c and o with it, so a figure shows the measurement not a pass."""
    plots = load_make_plots()
    a = {"context": 512, "c": 0.20, "v": {"1": 1.0, "2": 1.10},
         "overhead": {"o_per_round": 0.006}}
    b = {"context": 512, "c": 0.22, "v": {"1": 1.0, "2": 1.30},
         "overhead": {"o_per_round": 0.008}}

    out = plots.mean_blob([a, b])
    assert out["c"] == pytest.approx(0.21)
    assert out["v"]["2"] == pytest.approx(1.20)
    assert out["overhead"]["o_per_round"] == pytest.approx(0.007)
    assert out["passes_averaged"] == 2
    assert out["context"] == 512  # not averaged; it identifies the measurement


def test_the_offset_carves_disjoint_chunks_that_cover_the_set(tmp_path):
    """A long run has to be resumable, and that means chunks that neither overlap nor skip.

    The shuffle is seeded, so consecutive offsets cut the same deterministic order into disjoint
    pieces. Without this, a 12-hour sweep killed at hour 10 -- which the harness on this machine has
    done twice -- has to start over.
    """
    path = tmp_path / "questions.jsonl"
    rows = [{"question_id": i, "category": "qa" if i < 12 else "math", "prompt": f"q{i}"}
            for i in range(24)]
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")

    chunks = [load_questions(path, limit_per_category=3, offset_per_category=k)
              for k in (0, 3, 6, 9)]
    ids = [[q.question_id for q in chunk] for chunk in chunks]

    for chunk in chunks:
        assert len(chunk) == 6  # 3 per category, two categories
    flat = [i for chunk in ids for i in chunk]
    assert len(flat) == len(set(flat)), "chunks overlap"
    assert set(flat) == {r["question_id"] for r in rows}, "chunks do not cover the set"


def test_the_offset_is_stable_across_calls(tmp_path):
    """The same offset must return the same questions, or a resumed run measures something else."""
    path = tmp_path / "questions.jsonl"
    rows = [{"question_id": i, "category": "qa", "prompt": f"q{i}"} for i in range(20)]
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")

    first = [q.question_id for q in
             load_questions(path, limit_per_category=4, offset_per_category=8)]
    again = [q.question_id for q in
             load_questions(path, limit_per_category=4, offset_per_category=8)]
    assert first == again
    assert len(first) == 4


def test_the_offset_does_nothing_without_a_limit(tmp_path):
    """Backwards compatibility: an offset alone must not silently drop questions."""
    path = tmp_path / "questions.jsonl"
    rows = [{"question_id": i, "category": "qa", "prompt": f"q{i}"} for i in range(10)]
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")

    assert len(load_questions(path, offset_per_category=5)) == 10


def test_an_offset_past_the_end_returns_nothing(tmp_path):
    """The driver uses this to know it has reached the end of a category."""
    path = tmp_path / "questions.jsonl"
    rows = [{"question_id": i, "category": "qa", "prompt": f"q{i}"} for i in range(6)]
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")

    assert load_questions(path, limit_per_category=3, offset_per_category=6) == []


# --------------------------------------------------- the full-benchmark figures


def write_merged(directory: Path, shares: dict[str, float] | None = None,
                 decode: bool = True) -> Path:
    """A merged whole-benchmark result, shaped like run_specbench_full.py writes one."""
    shares = {"qa": 0.05, "rag": 0.66, "math": 0.14} if shares is None else shares
    categories = list(shares)
    directory.mkdir(parents=True, exist_ok=True)
    blob = {
        "gamma": 1,
        "questions": 480,
        "speedups": {
            "speculative": {**{c: 1.05 for c in categories}, "all": 0.98},
            "lookup": {**{c: 1.01 for c in categories}, "all": 1.02},
        },
        "summary": {
            "target": {
                **{c: {"prefill_share": s, "prefill_seconds": 100.0} for c, s in shares.items()},
                "all": {"prefill_share": 0.44, "prefill_seconds": 100.0},
            },
            "speculative": {"all": {"prefill_seconds": 120.0}},
            "lookup": {"all": {"prefill_seconds": 100.5}},
        },
    }
    if decode:
        blob["decode_speedups"] = {
            "speculative": {**{c: 1.11 for c in categories}, "all": 1.11},
            "lookup": {**{c: 1.02 for c in categories}, "all": 1.01},
        }
    path = directory / "specbench_full_gamma1.json"
    path.write_text(json.dumps(blob), encoding="utf-8")
    return path


def test_the_full_benchmark_is_found_in_its_own_directory(tmp_path):
    """The merged runs are kept apart from the sweep files, so they need their own glob."""
    plots = load_make_plots()
    write_merged(tmp_path / "specbench_full")

    found = plots.full_benchmark(tmp_path)
    assert [path.name for path, _ in found] == ["specbench_full_gamma1.json"]
    assert found[0][1]["questions"] == 480


def test_a_run_without_decode_speedups_is_not_treated_as_a_full_benchmark(tmp_path):
    """Both figures read the decode column; a file lacking it would plot the wrong quantity."""
    plots = load_make_plots()
    write_merged(tmp_path / "specbench_full", decode=False)

    assert plots.full_benchmark(tmp_path) == []


def test_the_full_benchmark_figure_orders_categories_by_prefill_share(tmp_path, monkeypatch):
    """The ordering *is* the claim.

    The result the figure exists to show is that the wall-clock winner is monotonic in prefill
    share. Drawn alphabetically that is invisible, and a reader has no way to check it; drawn in
    share order it is the shape of the picture. So the order is worth a test rather than a habit.
    """
    plots = load_make_plots()
    write_merged(tmp_path / "specbench_full")
    captured = {}
    monkeypatch.setattr(plots, "finish",
                        lambda fig, axes, out, name, title: captured.update(
                            labels=[t.get_text() for t in axes.get_xticklabels()], name=name))

    plots.plot_full_benchmark(tmp_path, tmp_path)

    assert captured["name"] == "specbench_full_speedup.png"
    assert [label.split()[0] for label in captured["labels"]] == ["qa", "math", "rag"]


def test_the_crossover_figure_plots_against_an_ascending_prefill_share(tmp_path, monkeypatch):
    """A crossing is only readable if the x axis runs in the order the model says decides it."""
    plots = load_make_plots()
    write_merged(tmp_path / "specbench_full")
    captured = {}
    monkeypatch.setattr(plots, "finish",
                        lambda fig, axes, out, name, title: captured.update(
                            xs=[list(line.get_xdata()) for line in axes.get_lines()], name=name))

    plots.plot_prefill_crossover(tmp_path, tmp_path)

    assert captured["name"] == "prefill_crossover.png"
    drawn = [xs for xs in captured["xs"] if len(xs) == 3]
    assert drawn, "expected a series over the three categories"
    for xs in drawn:
        assert xs == sorted(xs) == [0.05, 0.14, 0.66]


def test_neither_figure_raises_when_the_prefill_share_is_missing(tmp_path):
    """Older merged results predate the per-category share; they are skipped, not fatal."""
    plots = load_make_plots()
    path = write_merged(tmp_path / "specbench_full")
    blob = json.loads(path.read_text(encoding="utf-8"))
    for category in ("qa", "rag", "math"):
        del blob["summary"]["target"][category]["prefill_share"]
    path.write_text(json.dumps(blob), encoding="utf-8")

    plots.plot_full_benchmark(tmp_path, tmp_path)
    plots.plot_prefill_crossover(tmp_path, tmp_path)

    assert not list(tmp_path.glob("*.png"))
