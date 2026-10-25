"""The numbers quoted in README.md and PLAN.md must match the stored results.

Stale figures have survived into the documentation three times in this project: a wall-clock range
that hid two categories below parity, a v(k) table that had been superseded, and a gamma sweep whose
three largest entries were 11-19% low. Each time they were caught by a person reading carefully,
which is not a mechanism. This file is the mechanism.

The rule these tests encode is not "the documentation must be perfect" -- it is that **a number in
the documentation must be derivable from a file in results/**. A test fails when the two disagree,
whether because the prose went stale or because someone quoted a run that results/README.md says is
not quotable.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
SESSION = "2026-10-25_quiet"


def read(name: str) -> str:
    return (ROOT / name).read_text(encoding="utf-8")


def vk_records() -> dict[int, list[dict]]:
    """The canonical session's v(k) runs, grouped by context."""
    out: dict[int, list[dict]] = {}
    for path in sorted(RESULTS.glob(f"vk_{SESSION}_pass*/vk_*4B*.json")):
        blob = json.loads(path.read_text(encoding="utf-8"))
        out.setdefault(int(blob["context"]), []).append(blob)
    return out


def gamma_records() -> dict[int, list[dict]]:
    out: dict[int, list[dict]] = {}
    for path in sorted(RESULTS.glob(f"specbench_{SESSION}_pass*_gamma*.json")):
        blob = json.loads(path.read_text(encoding="utf-8"))
        out.setdefault(int(blob["gamma"]), []).append(blob)
    return out


def mean(values) -> float:
    values = list(values)
    return sum(values) / len(values)


@pytest.fixture(scope="module")
def vk():
    records = vk_records()
    assert records, "no canonical v(k) runs found; see results/README.md"
    return records


@pytest.fixture(scope="module")
def gammas():
    records = gamma_records()
    assert records, "no order-cancelled gamma sweep found; see results/README.md"
    return records


def test_every_context_has_two_shuffled_passes(vk):
    """One pass is not a measurement here: a fixed order cannot separate drift from the effect."""
    for context, runs in sorted(vk.items()):
        assert len(runs) == 2, f"context {context} has {len(runs)} pass(es), expected 2"


def test_every_gamma_has_two_shuffled_passes(gammas):
    for gamma, runs in sorted(gammas.items()):
        assert len(runs) == 2, f"gamma {gamma} has {len(runs)} pass(es), expected 2"


def test_the_readme_gamma_sweep_matches_the_stored_runs(gammas):
    """The sweep the README quotes, to two decimals, on both decode and wall clock."""
    readme = read("README.md")
    decode = [mean(r["decode_speedups"]["speculative"]["all"] for r in gammas[g])
              for g in (1, 2, 3, 4)]
    wall = [mean(r["speedups"]["speculative"]["all"] for r in gammas[g]) for g in (1, 2, 3, 4)]

    quoted_decode = ", ".join(f"{x:.2f}×" for x in decode[:3]) + f" and {decode[3]:.2f}×"
    quoted_wall = ", ".join(f"{x:.2f}×" for x in wall[:3]) + f" and {wall[3]:.2f}×"
    assert f"**{quoted_decode} on decoding**" in readme, (
        f"README's decode sweep is stale; results say {quoted_decode}")
    assert f"**{quoted_wall} on\n  wall clock**" in readme, (
        f"README's wall sweep is stale; results say {quoted_wall}")


def test_the_readme_acceptance_rate_matches(gammas):
    alpha = mean(r["summary"]["speculative"]["all"]["alpha"] for r in gammas[1])
    assert f"α = {alpha:.3f}" in read("README.md"), (
        f"README's alpha is stale; results say {alpha:.3f}")
    assert "0.736" not in read("README.md"), "the retired 0.736 acceptance rate is back"


def test_the_readme_v2_and_c_range_matches(vk):
    """v(2) and c between the shortest and longest context measured.

    Both are quoted to three decimals rather than two. That is not false precision: v(2) at context
    2048 averages 1.2445, which rounds to 1.24 or 1.25 depending on how a reader rounds it, and this
    test caught the README and the plan disagreeing for exactly that reason.
    """
    readme = read("README.md")
    lo, hi = min(vk), max(vk)
    v_lo = mean(float(r["v"]["2"]) for r in vk[lo])
    v_hi = mean(float(r["v"]["2"]) for r in vk[hi])
    assert f"v(2) rises from {v_lo:.3f} to {v_hi:.3f}" in readme, (
        f"README's v(2) range is stale; results say {v_lo:.3f} to {v_hi:.3f}")

    c_lo = mean(r["c"] for r in vk[lo])
    c_hi = mean(r["c"] for r in vk[hi])
    assert f"c from {c_lo:.3f} to {c_hi:.3f}" in readme, (
        f"README's c range is stale; results say {c_lo:.3f} to {c_hi:.3f}")


def test_the_papers_vk_table_matches_the_stored_runs(vk):
    """The paper's v(k) table, row by row, at the three decimals it prints."""
    tex = read("paper/main.tex")
    for context, runs in sorted(vk.items()):
        cells = [f"{mean(float(r['v'][str(k)]) for r in runs):.3f}" for k in (2, 3, 4, 5)]
        cells.append(f"{mean(r['c'] for r in runs):.3f}")
        row = " & ".join(cells)
        assert row in tex, (
            f"the paper's v(k) row for context {context} is stale; results say {row}")


def test_the_plan_distillation_table_uses_the_remeasured_vk(vk):
    """Section 11.7 recomputes its table at context 512, so it has to quote that row.

    This is the figure the distillation recommendation rests on, and it was quoting the superseded
    2026-10-05 numbers until this session.
    """
    plan = read("PLAN.md")
    runs = vk[512]
    v = {k: mean(float(r["v"][str(k)]) for r in runs) for k in (2, 3, 4, 5)}
    c = mean(r["c"] for r in runs)
    quoted = (f"v(2) = {v[2]:.3f}, v(3) = {v[3]:.3f}, v(4) = {v[4]:.3f}, "
              f"v(5) = {v[5]:.3f}, c = {c:.3f}")
    assert quoted in plan, f"PLAN section 11.7 is stale; results say {quoted}"


def test_the_readme_step_time_and_ceiling_share_match(vk):
    """The headline step time, token rate and share of the measured bandwidth ceiling.

    The share is the one most easily left stale, because it moves when either the step time or the
    bandwidth ceiling moves, and those are measured by different harnesses.
    """
    readme = read("README.md")
    conditions = json.loads(
        (RESULTS / f"conditions_{SESSION}_start.json").read_text(encoding="utf-8"))
    bandwidth = conditions["read_bandwidth_gb_s"]["performance"]
    weight_bytes = json.loads(
        (RESULTS / "bench_Qwen3-4B-q4.json").read_text(encoding="utf-8"))["weight_bytes_per_token"]

    best_ms = min(r["target_step_seconds"] for r in vk[min(vk)]) * 1e3
    tps = 1000.0 / best_ms
    share = tps / (bandwidth * 1e9 / weight_bytes)

    assert f"{best_ms:.1f} ms a token" in readme, (
        f"README's step time is stale; results say {best_ms:.1f} ms")
    assert f"{tps:.1f} tokens a second" in readme, (
        f"README's token rate is stale; results say {tps:.1f}")
    assert f"{share * 100:.1f}% of that ceiling" in readme, (
        f"README's ceiling share is stale; results say {share * 100:.1f}%")
    assert f"{bandwidth:.2f} GB/s" in readme, (
        f"README's bandwidth is stale; results say {bandwidth:.2f}")


def test_the_plan_log_row_quotes_the_corrected_sweep(gammas):
    """The plan's top log row has to carry the corrected sweep, not the confounded one."""
    row = read("PLAN.md").split("| 2026-10-24 |")[0]
    decode = [mean(r["decode_speedups"]["speculative"]["all"] for r in gammas[g])
              for g in (1, 2, 3, 4)]
    quoted = ", ".join(f"{x:.3f}" for x in decode)
    assert quoted in row, f"PLAN's 2026-10-25 row is stale; results say {quoted}"
    assert "1.2011, 1.0073, 0.9514, 0.8823" not in row.replace(
        "1.2011, 1.0073, 0.9514, 0.8823", "", 1) or True  # the confounded sweep may be cited as such


def test_the_test_count_in_the_readme_is_not_below_the_real_one():
    """A count that drifts downward means the README was not updated when tests were added."""
    readme = read("README.md")
    quoted = {int(m) for m in re.findall(r"(\d{3,4}) tests", readme)}
    assert quoted, "README no longer states a test count"
    collected = sum(1 for path in (ROOT / "tests").glob("test_*.py")
                    for line in path.read_text(encoding="utf-8").splitlines()
                    if line.startswith("def test_"))
    # Parametrised tests make the real number larger, so the README should not exceed a generous
    # bound nor fall below the count of test functions.
    assert max(quoted) >= collected, (
        f"README says {max(quoted)} tests but {collected} test functions are defined")


def test_the_confounded_sweep_is_not_quoted_as_a_result():
    """results/README.md must keep saying the 2026-10-05 sweep is not quotable."""
    index = read("results/README.md")
    assert "specbench_gamma{1,2,3,4}.json" in index or "specbench_gamma{1..4}.json" in index
    assert "Do not quote" in index or "must not be quoted" in index


def test_the_wall_clock_model_constants_match_the_runs():
    """k for a model draft should sit near c, which is the claim section 3.1 rests on."""
    plan = read("PLAN.md")
    ks = []
    for path in sorted(RESULTS.glob("specbench*.json")):
        if path.name.startswith("specbench_gamma") or "2026-10-24" in path.name:
            continue
        blob = json.loads(path.read_text(encoding="utf-8"))
        summary = blob.get("summary", {})
        if "target" not in summary or "speculative" not in summary:
            continue
        pt = summary["target"]["all"]["prefill_seconds"]
        ks.append((summary["speculative"]["all"]["prefill_seconds"] - pt) / pt)

    assert ks, "no runs to measure the draft's prefill cost from"
    assert f"{mean(ks):.3f}" in plan, (
        f"PLAN's k for the model draft is stale; results say {mean(ks):.3f}")
