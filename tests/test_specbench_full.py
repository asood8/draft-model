"""The resumable full-benchmark driver, and especially its merge.

The merge runs once, after about twelve hours of measurement. That is the worst possible moment to
discover a bug in it, so it is tested here against synthetic chunk files rather than trusted.

The bug these tests mainly exist to prevent is subtle: a speedup is a ratio of pooled totals and the
aggregate takes medians over turns, so averaging each chunk's speedups gives a median-of-medians,
which is not the median over the benchmark. Chunking a sweep is only sound if the merge pools the
per-turn records.
"""
from __future__ import annotations

import importlib.util
import json
import statistics
from pathlib import Path

import pytest

from specdraft.specbench import TurnResult, save_results, speedup_table

ROOT = Path(__file__).resolve().parent.parent


def load_driver():
    """The driver is a script, so it is imported by path the way test_specbench does it."""
    spec = importlib.util.spec_from_file_location(
        "run_specbench_full", ROOT / "scripts" / "run_specbench_full.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def turn(method: str, category: str, qid: int, tps: float, decode_tps: float | None = None):
    decode_tps = tps if decode_tps is None else decode_tps
    return TurnResult(
        method=method, category=category, question_id=qid, turn=0,
        tokens=100, seconds=100.0 / tps, target_forwards=100, proposed=0, accepted=0,
        rejections=0, rounds=100, tokens_per_target_forward=1.0, tokens_per_second=tps,
        prefill_seconds=1.0, decode_seconds=100.0 / decode_tps,
        decode_tokens_per_second=decode_tps, alpha=0.5,
    )


def write_chunk(path: Path, turns, gamma: int = 1, offset: int = 0):
    blob_path = save_results(path, turns, {
        "gamma": gamma, "target": "models/t.sdm", "draft": "models/d.sdm",
        "offset_per_category": offset, "per_category": 5, "questions": len(turns) // 2,
    })
    blob = json.loads(blob_path.read_text(encoding="utf-8"))
    blob["speedups"] = speedup_table(blob["summary"], "target", "tokens_per_second")
    blob["decode_speedups"] = speedup_table(blob["summary"], "target",
                                            "decode_tokens_per_second")
    blob_path.write_text(json.dumps(blob, indent=2), encoding="utf-8")


def test_the_merge_pools_turns_rather_than_averaging_chunk_speedups(tmp_path):
    """The whole point of chunking: the merged number must be the benchmark's, not a mean of means.

    The chunks are deliberately unequal in size, which is what makes the two answers differ. A
    three-question chunk at 30 tok/s and a one-question chunk at 11 pool to a median of 30, giving
    3.00x; averaging the chunks' own speedups weights the single slow question as heavily as the
    three fast ones and gives 2.05x. Equal-sized chunks can accidentally agree, which an earlier
    version of this test did -- so the sizes here are 3 and 1 on purpose.
    """
    driver = load_driver()
    out = tmp_path / "chunks"
    out.mkdir()

    fast = [turn("target", "qa", i, 10.0) for i in (1, 2, 3)] + \
           [turn("speculative", "qa", i, 30.0) for i in (1, 2, 3)]
    slow = [turn("target", "qa", 4, 10.0), turn("speculative", "qa", 4, 11.0)]
    write_chunk(out / "chunk_gamma1_off000_n5.json", fast, offset=0)
    write_chunk(out / "chunk_gamma1_off005_n5.json", slow, offset=5)

    chunk_speedups = []
    for name in ("chunk_gamma1_off000_n5.json", "chunk_gamma1_off005_n5.json"):
        blob = json.loads((out / name).read_text(encoding="utf-8"))
        chunk_speedups.append(blob["speedups"]["speculative"]["all"])
    mean_of_chunks = statistics.fmean(chunk_speedups)
    assert mean_of_chunks == pytest.approx(2.05), "the test's own setup no longer discriminates"

    driver.merge(out, 1, out / "merged.json")
    merged = json.loads((out / "merged.json").read_text(encoding="utf-8"))

    pooled = statistics.median([30.0, 30.0, 30.0, 11.0]) / statistics.median([10.0] * 4)
    assert pooled == pytest.approx(3.0)
    assert merged["speedups"]["speculative"]["all"] == pytest.approx(pooled)
    assert merged["speedups"]["speculative"]["all"] != pytest.approx(mean_of_chunks), (
        "the merge averaged the chunks instead of pooling their turns")
    assert merged["turns_merged"] == 8
    assert merged["chunks_merged"] == 2


def test_the_merge_refuses_to_double_count_overlapping_chunks(tmp_path):
    """A retried chunk written under two names must not silently inflate the result."""
    driver = load_driver()
    out = tmp_path / "chunks"
    out.mkdir()

    turns = [turn("target", "qa", 1, 10.0), turn("speculative", "qa", 1, 12.0)]
    write_chunk(out / "chunk_gamma1_off000_n5.json", turns, offset=0)
    write_chunk(out / "chunk_gamma1_off005_n5.json", turns, offset=5)  # same questions again

    with pytest.raises(SystemExit, match="duplicate turn record"):
        driver.merge(out, 1, out / "merged.json")


def test_the_merge_counts_questions_not_turn_records(tmp_path):
    """A multi-turn question contributes several records but is one question."""
    driver = load_driver()
    out = tmp_path / "chunks"
    out.mkdir()

    turns = []
    for t in (0, 1):
        for method in ("target", "speculative"):
            row = turn(method, "multi_turn", 7, 10.0)
            row.turn = t
            turns.append(row)
    write_chunk(out / "chunk_gamma1_off000_n5.json", turns, offset=0)

    driver.merge(out, 1, out / "merged.json")
    merged = json.loads((out / "merged.json").read_text(encoding="utf-8"))
    assert merged["turns_merged"] == 4
    assert merged["questions"] == 1


def test_the_merge_separates_gammas(tmp_path):
    """Chunks for one gamma must not be merged into another's result."""
    driver = load_driver()
    out = tmp_path / "chunks"
    out.mkdir()

    write_chunk(out / "chunk_gamma1_off000_n5.json",
                [turn("target", "qa", 1, 10.0), turn("speculative", "qa", 1, 12.0)], gamma=1)
    write_chunk(out / "chunk_gamma2_off000_n5.json",
                [turn("target", "qa", 1, 10.0), turn("speculative", "qa", 1, 15.0)], gamma=2)

    driver.merge(out, 1, out / "g1.json")
    merged = json.loads((out / "g1.json").read_text(encoding="utf-8"))
    assert merged["turns_merged"] == 2
    assert merged["speedups"]["speculative"]["all"] == pytest.approx(1.2)


def test_the_merge_says_so_when_there_is_nothing_to_merge(tmp_path):
    driver = load_driver()
    out = tmp_path / "empty"
    out.mkdir()
    with pytest.raises(SystemExit, match="nothing to merge"):
        driver.merge(out, 1, out / "merged.json")


def test_the_chunk_filename_round_trips(tmp_path):
    """The driver decides what to skip from the filename, so it has to be unambiguous."""
    driver = load_driver()
    path = driver.chunk_path(tmp_path, gamma=2, offset=35, size=5)
    assert path.name == "chunk_gamma2_off035_n5.json"
    assert driver.chunk_path(tmp_path, 2, 35, 5) == path
    assert driver.chunk_path(tmp_path, 2, 5, 5) != path
