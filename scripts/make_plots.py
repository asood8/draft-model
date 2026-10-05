"""Turn the result JSONs into the figures the write-up is built around (plan section 12).

    python scripts/make_plots.py --results results --out results/figures

Reads whatever is present and skips what is not, so it can be run at any point and will draw more
as more measurements land:

* **v(k)** -- what each extra verified token costs. The figure the project exists to produce, since
  the usual speedup formula assumes this line is flat at 1.
* **Predicted against measured speedup** -- the formula is only worth having if it predicts; the
  distance from the diagonal is the part worth explaining.
* **Tokens per pass against gamma** -- where more guesses stop paying.
* **Thread configurations** -- performance cores against all cores, fixed split against dynamic
  chunks, with the measured bandwidth ceiling drawn in.
* **Acceptance by token class** -- which kinds of token a draft gets wrong, which is what makes the
  write-up more than a table.
* **Per-category speedup** -- the headline Spec-Bench comparison.

Plots are deliberately plain: one chart per figure, labelled axes, no styling that would make two
figures hard to compare.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # write files, never open a window
import matplotlib.pyplot as plt  # noqa: E402

FIGURE_SIZE = (7.0, 4.2)


def load(path: Path) -> dict | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        print(f"  {path.name}: not readable as JSON, skipping")
        return None


def finish(figure, axes, out: Path, name: str, title: str) -> None:
    axes.set_title(title)
    axes.grid(alpha=0.3)
    figure.tight_layout()
    path = out / name
    figure.savefig(path, dpi=150)
    plt.close(figure)
    print(f"  wrote {path}")


def measured_runs(results: Path, pattern: str = "specbench_gamma*.json") -> list[dict]:
    """Spec-Bench runs that form a gamma sweep, newest-sorted, with the numbers to compare.

    Only the sweep files by default: they are one prompt set measured at several gamma, which is what
    a curve needs. specbench_short and specbench_long are different prompt sets, so putting them on
    the same axes would compare acceptance rates and call it a cost model.

    `decode` is the speedup over the baseline's *decoding*. The whole-call figure is not comparable
    with a prediction: gamma*c + v(gamma+1) + o says nothing about a prompt pass, which is 45-92% of
    the wall clock here, and which speculative decoding pays for twice.
    """
    runs = []
    for path in sorted(results.glob(pattern)):
        blob = load(path)
        if blob is None or blob.get("gamma") is None:
            continue
        table = blob.get("decode_speedups") or blob.get("speedups")
        row = blob.get("summary", {}).get("speculative", {}).get("all")
        if not table or not row:
            continue
        speedup = table.get("speculative", {}).get("all")
        if not speedup:
            continue
        runs.append({
            "gamma": int(blob["gamma"]),
            "decode": float(speedup),
            "decode_only": "decode_speedups" in blob,
            "tau": float(row.get("tokens_per_target_forward", 0.0)),
            "alpha": float(row.get("alpha", 0.0)),
            "target": str(blob.get("target", "")),
        })
    return sorted(runs, key=lambda run: run["gamma"])


def predict(blob: dict, gamma: int, tau: float) -> float | None:
    """The speedup the cost model gives for a round of gamma guesses that emitted tau tokens."""
    cost = blob.get("v", {}).get(str(gamma + 1))
    c = blob.get("c")
    if cost is None or c is None:
        return None
    o = blob.get("overhead", {}).get("o_per_round", 0.0)
    return tau / (gamma * c + float(cost) + o)


def plot_vk(results: Path, out: Path) -> None:
    for path in sorted(results.glob("vk_*.json")):
        blob = load(path)
        if blob is None or "v" not in blob:
            continue
        ks = sorted(int(k) for k in blob["v"])
        values = [blob["v"][str(k)] for k in ks]

        figure, axes = plt.subplots(figsize=FIGURE_SIZE)
        axes.plot(ks, values, marker="o", label="measured v(k)")
        axes.plot(ks, [1.0] * len(ks), linestyle="--",
                  label="what the original formula assumes")
        axes.plot(ks, ks, linestyle=":", color="grey", label="k separate steps")
        axes.set_xlabel("tokens verified in one pass, k")
        axes.set_ylabel("cost, in ordinary target steps")
        axes.legend()
        name = path.stem.replace("vk_", "")
        finish(figure, axes, out, f"vk_{name}.png",
               f"What verifying k tokens costs ({name}, context {blob.get('context', '?')})")


def plot_vk_by_context(results: Path, out: Path) -> None:
    """v(k) at several context lengths, which is the figure that says why one number is not enough.

    The KV cache is read per token of the pass, not once per pass, so every extra token of context
    adds to the part of a verification pass that k tokens cannot share. Reading v(k) off a short
    context and applying it to long prompts over-predicts the speedup, which is exactly the mistake
    section 10.4 records.
    """
    by_target: dict[str, list[tuple[int, dict]]] = {}
    paths = list(results.glob("vk_*.json")) + list(results.glob("vk_context*/vk_*.json"))
    for path in sorted(paths):
        blob = load(path)
        if blob is None or "v" not in blob or blob.get("context") is None:
            continue
        by_target.setdefault(path.stem.replace("vk_", ""), []).append((int(blob["context"]), blob))

    for name, runs in by_target.items():
        if len(runs) < 2:
            continue
        figure, axes = plt.subplots(figsize=FIGURE_SIZE)
        for context, blob in sorted(runs):
            ks = sorted(int(k) for k in blob["v"])
            axes.plot(ks, [blob["v"][str(k)] for k in ks], marker="o",
                      label=f"context {context}, c = {blob.get('c', float('nan')):.3f}")
        axes.axhline(1.0, color="black", linewidth=1,
                     label="what the original formula assumes")
        axes.set_xlabel("tokens verified in one pass, k")
        axes.set_ylabel("cost, in ordinary target steps")
        axes.legend()
        finish(figure, axes, out, f"vk_by_context_{name}.png",
               f"Verification gets dearer as the context grows ({name})")


def plot_speedup_vs_gamma(results: Path, out: Path) -> None:
    for path in sorted(results.glob("vk_*.json")):
        blob = load(path)
        if blob is None or not blob.get("predictions"):
            continue
        rows = blob["predictions"]
        gammas = [row["gamma"] for row in rows]
        name = path.stem.replace("vk_", "")

        figure, axes = plt.subplots(figsize=FIGURE_SIZE)
        for alpha in sorted(rows[0]["speedup"], key=float):
            axes.plot(gammas, [row["speedup"][alpha] for row in rows], marker="o",
                      label=f"acceptance {float(alpha):.2f}")
        # The measured points, and what the model predicts for each of them from its own tau. Same
        # tau on both sides, so the only thing being tested is the denominator -- which is the term
        # this project adds. The alpha curves above are context for where a better draft would land.
        runs = [run for run in measured_runs(results)
                if not run["target"] or Path(run["target"]).stem == name]
        if runs:
            axes.plot([run["gamma"] for run in runs], [run["decode"] for run in runs],
                      marker="s", color="black", linewidth=2, label="measured (decoding)")
            predicted = [(run["gamma"], predict(blob, run["gamma"], run["tau"])) for run in runs]
            predicted = [(g, value) for g, value in predicted if value is not None]
            if predicted:
                axes.plot([g for g, _ in predicted], [value for _, value in predicted],
                          marker="x", linestyle="--", color="black",
                          label="predicted from the measured tau")

        axes.axhline(1.0, color="black", linewidth=1)
        axes.set_xlabel("guesses per round, gamma")
        axes.set_ylabel("speedup")
        axes.legend()
        finish(figure, axes, out, f"speedup_vs_gamma_{name}.png",
               f"Best gamma falls as verification gets dearer ({name}, c={blob.get('c', '?'):.3f})"
               if isinstance(blob.get("c"), float) else f"Predicted speedup ({name})")


def plot_tokens_per_pass(results: Path, out: Path) -> None:
    for path in sorted(results.glob("offline_*.json")):
        blob = load(path)
        if blob is None or "tokens_per_pass" not in blob:
            continue
        figure, axes = plt.subplots(figsize=FIGURE_SIZE)
        for threshold, sweep in sorted(blob["tokens_per_pass"].items(), key=lambda kv: float(kv[0])):
            gammas = sorted(int(g) for g in sweep)
            label = "no early stopping" if float(threshold) == 0 else f"stop below {threshold}"
            axes.plot(gammas, [sweep[str(g)] for g in gammas], marker="o", label=label)
        axes.plot([1, max(gammas)], [2, max(gammas) + 1], linestyle=":", color="grey",
                  label="perfect acceptance")
        axes.set_xlabel("guesses per round, gamma")
        axes.set_ylabel("tokens emitted per target pass")
        axes.legend()
        finish(figure, axes, out, f"{path.stem}_tokens_per_pass.png",
               f"Tokens per target pass ({blob.get('mode', '?')}, "
               f"acceptance {blob.get('acceptance', float('nan')):.3f})")


def plot_threads(results: Path, out: Path) -> None:
    for path in sorted(results.glob("bench_*.json")):
        blob = load(path)
        if blob is None or not blob.get("rows"):
            continue
        rows = blob["rows"]
        names = [row["configuration"] for row in rows]
        speeds = [row["decode_tokens_per_second"] for row in rows]
        ceiling = max(blob.get("read_bandwidth_gb_s", {"x": 0}).values()) * 1e9
        per_token = blob.get("weight_bytes_per_token", 0)

        figure, axes = plt.subplots(figsize=(7.6, 4.4))
        axes.barh(names, speeds, color="steelblue")
        if ceiling and per_token:
            limit = ceiling / per_token
            axes.axvline(limit, color="black", linestyle="--",
                         label=f"bandwidth ceiling, {limit:.0f} tok/s")
            axes.legend()
        axes.set_xlabel("decode tokens per second (median)")
        axes.invert_yaxis()
        finish(figure, axes, out, f"{path.stem}_threads.png",
               f"Thread configurations ({blob.get('physical_cores', '?')} cores, "
               f"context {blob.get('context', '?')})")


def plot_token_classes(results: Path, out: Path) -> None:
    for path in sorted(results.glob("offline_*.json")):
        blob = load(path)
        if blob is None or not blob.get("acceptance_by_token_class"):
            continue
        profile = blob["acceptance_by_token_class"]
        names = list(profile)
        acceptance = [profile[name]["acceptance"] for name in names]
        share = [profile[name]["share"] for name in names]

        figure, axes = plt.subplots(figsize=(7.6, 4.4))
        bars = axes.barh(names, acceptance, color="steelblue")
        for bar, fraction in zip(bars, share):
            axes.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                      f"{fraction:.0%} of tokens", va="center", fontsize=8)
        axes.axvline(blob.get("acceptance", 0.0), color="black", linestyle="--",
                     label="overall acceptance")
        axes.set_xlim(0, 1.25)
        axes.set_xlabel("acceptance")
        axes.invert_yaxis()
        axes.legend(loc="lower right")
        finish(figure, axes, out, f"{path.stem}_token_classes.png",
               "Where the draft is wrong, by kind of token")


def plot_specbench(results: Path, out: Path) -> None:
    """Per category, both speedups: decoding, and the whole call.

    They answer different questions and on long prompts they disagree by a factor of four, so the
    figure carries both rather than picking one. Decoding is what the cost model predicts; the whole
    call is what a user waits for, and on a 1450-token prompt most of that is the prompt pass.
    """
    for path in sorted(results.glob("specbench*.json")):
        blob = load(path)
        if blob is None or "speedups" not in blob:
            continue
        speedups = blob.get("decode_speedups") or blob["speedups"]
        decoding = "decode_speedups" in blob
        categories = [c for c in sorted(next(iter(speedups.values()))) if c != "all"]
        if not categories:
            continue
        methods = [m for m in speedups if m != "target"]

        figure, axes = plt.subplots(figsize=(8.4, 4.6))
        width = 0.8 / max(1, len(methods))
        for index, method in enumerate(methods):
            offsets = [i + index * width for i in range(len(categories))]
            axes.bar(offsets, [speedups[method].get(c, 0.0) for c in categories], width,
                     label=method)
            if decoding:  # the whole-call figure drawn as an outline behind each bar
                wall = blob["speedups"]
                axes.bar(offsets, [wall[method].get(c, 0.0) for c in categories], width,
                         fill=False, edgecolor="black", linewidth=0.8,
                         label="same, whole call" if index == 0 else None)
        axes.axhline(1.0, color="black", linewidth=1, label="target alone")
        axes.set_xticks([i + 0.4 - width / 2 for i in range(len(categories))])
        axes.set_xticklabels(categories, rotation=20, ha="right")
        axes.set_ylabel("speedup over the target alone"
                        + (", decoding" if decoding else ""))
        axes.legend()
        finish(figure, axes, out, f"{path.stem}_speedup.png",
               f"Spec-Bench, gamma={blob.get('gamma', '?')}")


def plot_predicted_vs_measured(results: Path, out: Path) -> None:
    """Needs both an offline prediction and a Spec-Bench measurement to compare.

    Measured means the speedup over the baseline's decoding, not over its whole call: the prediction
    describes rounds, and a prompt pass is neither a round nor affected by any of this.
    """
    offline = [load(path) for path in sorted(results.glob("offline_*.json"))]
    bench = [load(path) for path in sorted(results.glob("specbench*.json"))]
    pairs = []
    for measured in bench:
        if measured is None or "speedups" not in measured:
            continue
        for prediction in offline:
            if prediction is None or prediction.get("c") is None:
                continue
            gamma = str(measured.get("gamma", ""))
            sweep = prediction["tokens_per_pass"].get("0.0", {})
            if gamma not in sweep:
                continue
            tau = sweep[gamma]
            v = prediction["v"]
            cost = v if isinstance(v, float) else float(v.get(str(int(gamma) + 1), 1.0))
            predicted = tau / (int(gamma) * prediction["c"] + cost + prediction.get("o", 0.0))
            table = measured.get("decode_speedups") or measured["speedups"]
            actual = table.get("speculative", {}).get("all")
            if actual:
                pairs.append((predicted, actual))

    if not pairs:
        return
    figure, axes = plt.subplots(figsize=(5.4, 5.0))
    axes.scatter([p for p, _ in pairs], [m for _, m in pairs], color="steelblue")
    limit = max(max(p for p, _ in pairs), max(m for _, m in pairs)) * 1.15
    axes.plot([0, limit], [0, limit], linestyle="--", color="black", label="perfect prediction")
    axes.set_xlabel("predicted speedup")
    axes.set_ylabel("measured speedup (decoding)")
    axes.set_xlim(0, limit)
    axes.set_ylim(0, limit)
    axes.legend()
    finish(figure, axes, out, "predicted_vs_measured.png",
           "Does the cost model predict the engine?")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--out", type=Path, default=Path("results/figures"))
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    print(f"reading {args.results}, writing {args.out}")
    for draw in (plot_vk, plot_vk_by_context, plot_speedup_vs_gamma, plot_tokens_per_pass,
                 plot_threads,
                 plot_token_classes, plot_specbench, plot_predicted_vs_measured):
        try:
            draw(args.results, args.out)
        except Exception as exc:  # one broken result file should not stop the rest
            print(f"  {draw.__name__}: {type(exc).__name__}: {exc}")

    figures = sorted(args.out.glob("*.png"))
    print(f"\n{len(figures)} figures" + (":" if figures else ""))
    for figure in figures:
        print(f"  {figure.name}")


if __name__ == "__main__":
    main()
