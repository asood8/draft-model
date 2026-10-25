"""Check the wall-clock extension of the cost model against the stored Spec-Bench runs.

    python scripts/fit_prefill_model.py --results results

The model in plan section 3 predicts *decoding*:

    S_decode(g) = tau(g) / (g*c + v(g+1) + o)

It is silent about the prompt pass, which is 27-88% of the clock in these runs and which speculative
decoding pays for twice, once into each model's KV cache. So the headline number this project
reports is a wall-clock one that its own model cannot predict, and the model cannot say which
drafting strategy to use: prompt lookup loses on decode in every run measured here and wins the
clock in two of three prompt sets, and nothing in the formula above can express that, because a
drafter needing no prompt pass of its own is identical in it to one that does.

Greedy speculative decoding emits exactly the tokens the target would, so both runs generate the
same N tokens and only the rates differ. Writing P for the prompt pass, D for the decode phase,
f = P_t / (P_t + D_t) for the target's prefill share and k = P_d / P_t for the draft's prompt pass
relative to the target's:

    target:       T_t = P_t + D_t
    speculative:  T_s = P_t + P_d + D_t / S_decode

    S_wall = 1 / [ (1 + k)*f + (1 - f) / S_decode ]                                       (1)

Two things make this worth writing down rather than only measuring. First, k is not a new free
parameter: it should be about c, the same draft-to-target cost ratio the decode model already uses,
because both are the same two models doing the same work on different numbers of tokens. Second, it
explains the drafting-strategy reversal for free, since prompt lookup runs no model and so has k = 0
exactly, and (1) then rewards it in proportion to f. That is a prediction, not a fit.

This script measures k directly -- the runs record prefill_seconds per method, so P_d = P_s - P_t --
compares it with c, scores (1) against every usable run, and reports the break-even prefill share at
which a zero-cost drafter overtakes a model draft.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

MODEL_DRAFT = ("speculative", "speculative+stop")


def usable(path: Path) -> bool:
    """Skip the runs results/README.md says must not be quoted.

    specbench_gamma{1..4}.json ran strictly ascending and back to back, so every larger gamma
    inherited the heat of the runs before it. The 2026-10-24 files are a single pass, with no
    reverse order to cancel against.
    """
    return not path.name.startswith("specbench_gamma") and "2026-10-24" not in path.name


def load_runs(results: Path) -> list[tuple[str, dict]]:
    out = []
    for path in sorted(results.glob("specbench*.json")):
        if not usable(path):
            continue
        blob = json.loads(path.read_text(encoding="utf-8"))
        if "target" in blob.get("summary", {}):
            out.append((path.stem.replace("specbench_", ""), blob))
    return out


def predict(f: float, s_decode: float, k: float) -> float:
    """Equation (1): the wall-clock speedup implied by a decode speedup and a prefill share."""
    return 1.0 / ((1.0 + k) * f + (1.0 - f) / s_decode)


def measure_k(runs: list[tuple[str, dict]]) -> dict[str, list[float]]:
    """k = P_draft / P_target, read straight off the recorded prefill seconds."""
    ks: dict[str, list[float]] = {}
    for _, blob in runs:
        s = blob["summary"]
        pt = s["target"]["all"]["prefill_seconds"]
        for method in (m for m in s if m != "target"):
            ks.setdefault(method, []).append(
                (s[method]["all"]["prefill_seconds"] - pt) / pt)
    return ks


def main() -> None:
    parser = argparse.ArgumentParser(description="Check the wall-clock cost model.")
    parser.add_argument("--results", type=Path, default=Path("results"))
    args = parser.parse_args()

    runs = load_runs(args.results)
    if not runs:
        print(f"no usable Spec-Bench runs in {args.results}")
        return

    print("=" * 96)
    print("STEP 1 -- k = P_draft / P_target, measured directly, against c")
    print("=" * 96)
    ks = measure_k(runs)
    print(f"{'method':<22}{'k, mean':>9}{'min':>9}{'max':>9}{'n':>5}")
    for method, vals in sorted(ks.items()):
        print(f"{method:<22}{sum(vals)/len(vals):>9.3f}{min(vals):>9.3f}{max(vals):>9.3f}"
              f"{len(vals):>5}")
    print("\nc (draft decode step / target decode step), 2026-10-25: 0.182 at context 128,"
          " 0.245 at 2048.")
    print("A model draft's prompt pass therefore costs about what its decode step does, so (1)"
          " needs no new measurement.")

    k = {m: sum(v) / len(v) for m, v in ks.items()}

    print("\n" + "=" * 96)
    print("STEP 2 -- score equation (1) against every usable run")
    print("=" * 96)
    print(f"{'run':<40}{'method':<22}{'f':>7}{'S_dec':>8}{'S_wall':>9}{'pred':>8}{'err':>8}")
    errs: dict[str, list[float]] = {}
    for name, blob in runs:
        f = blob["summary"]["target"]["all"]["prefill_share"]
        for method in (m for m in blob["summary"] if m != "target"):
            sd = blob["decode_speedups"][method]["all"]
            sw = blob["speedups"][method]["all"]
            pr = predict(f, sd, k[method])
            errs.setdefault(method, []).append(abs(pr / sw - 1))
            print(f"{name:<40}{method:<22}{f:>7.3f}{sd:>8.3f}{sw:>9.3f}{pr:>8.3f}"
                  f"{(pr / sw - 1) * 100:>+7.1f}%")

    print("\nmean absolute error:")
    every = [e for v in errs.values() for e in v]
    for method, v in sorted(errs.items()):
        print(f"  {method:<22}{sum(v)/len(v)*100:>6.1f}%   worst {max(v)*100:>5.1f}%   n={len(v)}")
    print(f"  {'OVERALL':<22}{sum(every)/len(every)*100:>6.1f}%   worst {max(every)*100:>5.1f}%")

    print("\n" + "=" * 96)
    print("STEP 3 -- the reversal a decode-only model cannot express")
    print("=" * 96)
    print(f"{'run':<40}{'f':>7}{'lookup':>9}{'model':>9}{'winner':>9}{'predicted':>11}")
    for name, blob in runs:
        s = blob["summary"]
        if "lookup" not in s or "speculative" not in s:
            continue
        f = s["target"]["all"]["prefill_share"]
        wl, wm = blob["speedups"]["lookup"]["all"], blob["speedups"]["speculative"]["all"]
        pl = predict(f, blob["decode_speedups"]["lookup"]["all"], k["lookup"])
        pm = predict(f, blob["decode_speedups"]["speculative"]["all"], k["speculative"])
        print(f"{name:<40}{f:>7.3f}{wl:>9.3f}{wm:>9.3f}"
              f"{('lookup' if wl > wm else 'model'):>9}"
              f"{('lookup' if pl > pm else 'model'):>11}")

    # Break-even: the prefill share at which a zero-cost drafter overtakes a model draft.
    #
    # It is a function of the two decode speedups, and those depend on the workload's acceptance
    # rate -- so it is reported per prompt set rather than over a pool of them. Averaging the short
    # and long sets together with the four-category subset mixes three different alphas and gives a
    # break-even belonging to none of them.
    print("\nbreak-even prefill share, per prompt set, at gamma=1")
    print(f"{'prompt set':<40}{'S_lookup':>10}{'S_model':>9}{'f*':>8}{'f':>8}{'side':>8}")
    for name, blob in runs:
        s = blob["summary"]
        if blob.get("gamma") != 1 or "lookup" not in s or "speculative" not in s:
            continue
        s_l = blob["decode_speedups"]["lookup"]["all"]
        s_m = blob["decode_speedups"]["speculative"]["all"]
        f = s["target"]["all"]["prefill_share"]
        gap = 1 / s_l - 1 / s_m
        if gap <= 0:  # lookup already wins on decode, so there is no prompt pass to repay
            print(f"{name:<40}{s_l:>10.3f}{s_m:>9.3f}{'n/a':>8}{f:>8.3f}{'lookup':>8}")
            continue
        star = gap / (gap + k["speculative"] - k["lookup"])
        print(f"{name:<40}{s_l:>10.3f}{s_m:>9.3f}{star:>8.3f}{f:>8.3f}"
              f"{('lookup' if f > star else 'model'):>8}")
    print("\nAbove f*, prefer a drafter that needs no prompt pass of its own, even though it"
          " accepts far less often.")
    print("f* moves with the workload, because it depends on both decode speedups and those depend"
          " on acceptance.")


if __name__ == "__main__":
    main()
