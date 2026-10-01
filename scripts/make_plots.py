"""Turn the result JSONs into the figures the write-up is built around (plan §12).

    python scripts/make_plots.py --results results --out results/figures

Reads whatever is present and skips what is not, so it can be run at any point and will draw more
as more measurements land:

* **v(k)** — what each extra verified token costs. The figure the project exists to produce, since
  the usual speedup formula assumes this line is flat at 1.
* **Predicted against measured speedup** — the formula is only worth having if it predicts; the
  distance from the diagonal is the part worth explaining.
* **Tokens per pass against γ** — where more guesses stop paying.
* **Thread configurations** — performance cores against all cores, fixed split against dynamic
  chunks, with the measured bandwidth ceiling drawn in.
* **Acceptance by token class** — which kinds of token a draft gets wrong, which is what makes the
  write-up more than a table.
* **Per-category speedup** — the headline Spec-Bench comparison.

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


def plot_speedup_vs_gamma(results: Path, out: Path) -> None:
    for path in sorted(results.glob("vk_*.json")):
        blob = load(path)
        if blob is None or not blob.get("predictions"):
            continue
        rows = blob["predictions"]
        gammas = [row["gamma"] for row in rows]

        figure, axes = plt.subplots(figsize=FIGURE_SIZE)
        for alpha in sorted(rows[0]["speedup"], key=float):
            axes.plot(gammas, [row["speedup"][alpha] for row in rows], marker="o",
                      label=f"acceptance {float(alpha):.2f}")
        axes.axhline(1.0, color="black", linewidth=1)
        axes.set_xlabel("guesses per round, gamma")
        axes.set_ylabel("predicted speedup")
        axes.legend()
        name = path.stem.replace("vk_", "")
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
    for path in sorted(results.glob("specbench*.json")):
        blob = load(path)
        if blob is None or "speedups" not in blob:
            continue
        speedups = blob["speedups"]
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
        axes.axhline(1.0, color="black", linewidth=1, label="target alone")
        axes.set_xticks([i + 0.4 - width / 2 for i in range(len(categories))])
        axes.set_xticklabels(categories, rotation=20, ha="right")
        axes.set_ylabel("speedup over the target alone")
        axes.legend()
        finish(figure, axes, out, f"{path.stem}_speedup.png",
               f"Spec-Bench, gamma={blob.get('gamma', '?')}")


def plot_predicted_vs_measured(results: Path, out: Path) -> None:
    """Needs both an offline prediction and a Spec-Bench measurement to compare."""
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
            actual = measured["speedups"].get("speculative", {}).get("all")
            if actual:
                pairs.append((predicted, actual))

    if not pairs:
        return
    figure, axes = plt.subplots(figsize=(5.4, 5.0))
    axes.scatter([p for p, _ in pairs], [m for _, m in pairs], color="steelblue")
    limit = max(max(p for p, _ in pairs), max(m for _, m in pairs)) * 1.15
    axes.plot([0, limit], [0, limit], linestyle="--", color="black", label="perfect prediction")
    axes.set_xlabel("predicted speedup")
    axes.set_ylabel("measured speedup")
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
    for draw in (plot_vk, plot_speedup_vs_gamma, plot_tokens_per_pass, plot_threads,
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
