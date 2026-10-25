"""Extend the cost model from decode to wall clock, and check it.

The model this project is about predicts decoding:

    S_decode(g) = tau(g) / (g*c + v(g+1) + o)

That is what the engine's round loop does, and it is silent about the prompt pass -- which is 12-92%
of the clock here and which speculative decoding pays for *twice*, once into each model's KV cache.
So the headline number this project reports is a wall-clock one that its own model cannot predict,
and it cannot say which drafting strategy wins: prompt-lookup loses on decode and wins on the clock.

Derivation. Greedy speculative decoding emits exactly the tokens the target would, so both runs
generate the same N tokens and the only difference is rates. Writing P for the prompt pass and D for
the decode phase:

    target:      T_t = P_t + D_t
    speculative: T_s = P_t + P_d + D_t / S_decode

so, with f = P_t / (P_t + D_t) the target's prefill share and k = P_d / P_t the draft's prompt pass
relative to the target's,

    S_wall = 1 / [ (1 + k) * f + (1 - f) / S_decode ]                                       (1)

Two things make this worth writing down rather than just measuring. First, k is not a new free
parameter: it should be about c, the same draft-to-target cost ratio the decode model already uses,
because both are the same two models doing the same work on different numbers of tokens. Second, it
explains the drafting-strategy reversal for free: prompt lookup runs no model, so k = 0 exactly, and
(1) then rewards it in proportion to f. That is a prediction, not a fit.

This script measures k directly -- the runs record prefill_seconds per method, so P_d = P_s - P_t --
checks it against c, and then scores (1) against every usable Spec-Bench run.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(r"C:\Users\aadit\python\draft-model")
RES = ROOT / "results"
CATS = ["math", "multi_turn", "qa", "translation", "all"]
MODEL_DRAFT = ("speculative", "speculative+stop")


def usable(p: Path) -> bool:
    return not p.name.startswith("specbench_gamma") and "2026-10-24" not in p.name


def runs() -> list[tuple[str, dict]]:
    out = []
    for p in sorted(RES.glob("specbench*.json")):
        if not usable(p):
            continue
        blob = json.loads(p.read_text(encoding="utf-8"))
        if "summary" in blob and "target" in blob.get("summary", {}):
            out.append((p.stem.replace("specbench_", ""), blob))
    return out


print("=" * 104)
print("STEP 1 -- measure k = P_draft / P_target directly, and compare it with c")
print("=" * 104)
print(f"{'run':<42}{'gamma':>6}{'method':<20}{'k':>8}{'prefill share f':>17}")
ks: dict[str, list[float]] = {}
for name, blob in runs():
    s = blob["summary"]
    pt = s["target"]["all"]["prefill_seconds"]
    f = s["target"]["all"]["prefill_share"]
    for method in [m for m in s if m != "target"]:
        ps = s[method]["all"]["prefill_seconds"]
        k = (ps - pt) / pt
        ks.setdefault(method, []).append(k)
        print(f"{name:<42}{blob.get('gamma', '?'):>6}{method:<20}{k:>8.3f}{f:>17.3f}")

print("\nk by method, over every usable run:")
for method, vals in sorted(ks.items()):
    print(f"  {method:<20} mean {sum(vals)/len(vals):>7.3f}   "
          f"range {min(vals):>6.3f} to {max(vals):>6.3f}   n={len(vals)}")
print("\n  c (draft decode step / target decode step), 2026-10-25: 0.182 at context 128 "
      "rising to 0.245 at 2048")

K = {m: sum(v) / len(v) for m, v in ks.items()}


def predict(f: float, s_decode: float, k: float) -> float:
    return 1.0 / ((1.0 + k) * f + (1.0 - f) / s_decode)


print("\n" + "=" * 104)
print("STEP 2 -- score equation (1) against every usable run, using the measured k per method")
print("=" * 104)
print(f"{'run':<42}{'method':<20}{'f':>7}{'S_dec':>8}{'S_wall':>9}{'pred':>8}{'err':>8}")
errs: dict[str, list[float]] = {}
for name, blob in runs():
    s = blob["summary"]
    for method in [m for m in s if m != "target"]:
        f = s["target"]["all"]["prefill_share"]
        sd = blob["decode_speedups"][method]["all"]
        sw = blob["speedups"][method]["all"]
        pr = predict(f, sd, K[method])
        e = pr / sw - 1
        errs.setdefault(method, []).append(abs(e))
        print(f"{name:<42}{method:<20}{f:>7.3f}{sd:>8.3f}{sw:>9.3f}{pr:>8.3f}{e*100:>+7.1f}%")

print("\nmean absolute error by method:")
allerr = []
for method, v in sorted(errs.items()):
    allerr += v
    print(f"  {method:<20} {sum(v)/len(v)*100:>5.1f}%   worst {max(v)*100:>5.1f}%   n={len(v)}")
print(f"  {'OVERALL':<20} {sum(allerr)/len(allerr)*100:>5.1f}%   worst {max(allerr)*100:>5.1f}%")

print("\n" + "=" * 104)
print("STEP 3 -- the reversal the decode model cannot explain, predicted by equation (1)")
print("=" * 104)
print("For a zero-cost drafter (k=0) to beat a model draft on the clock it needs")
print("  f * (S_d_model - S_d_lookup) / (S_d_model * S_d_lookup) < k_model * f")
print("i.e. the model draft's decode advantage must not repay its second prompt pass.\n")
print(f"{'run':<42}{'f':>7}{'lookup wall':>13}{'spec wall':>11}{'winner':>9}{'predicted':>11}")
for name, blob in runs():
    s = blob["summary"]
    if "lookup" not in s or "speculative" not in s:
        continue
    f = s["target"]["all"]["prefill_share"]
    wl = blob["speedups"]["lookup"]["all"]
    wm = blob["speedups"]["speculative"]["all"]
    pl = predict(f, blob["decode_speedups"]["lookup"]["all"], K["lookup"])
    pm = predict(f, blob["decode_speedups"]["speculative"]["all"], K["speculative"])
    print(f"{name:<42}{f:>7.3f}{wl:>13.3f}{wm:>11.3f}"
          f"{('lookup' if wl > wm else 'model'):>9}"
          f"{('lookup' if pl > pm else 'model'):>11}")
