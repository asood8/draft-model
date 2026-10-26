# The paper

A draft of the write-up, in IEEEtran. `../docs/WRITEUP.md` is the prose version it was built from and
is the easier thing to read; this directory is what gets submitted.

**It has not been compiled.** There is no TeX distribution on the machine this was written on, so the
source has been checked structurally rather than built. `make check` runs those checks —
environment and brace balance, bare `%` after a digit (which silently eats the rest of its line),
`\ref` without `\label`, and `\cite` keys missing from `refs.bib`. It currently reports zero
problems, which is not the same as compiling.

## Building

```bash
make            # pdflatex, bibtex, pdflatex, pdflatex
make check      # structural checks, no toolchain needed
make pending    # list every number still to be measured
make clean
```

Needs IEEEtran, which ships with TeX Live and MiKTeX (`texlive-publishers` on Debian/Ubuntu). Figures
are read from `../results/figures/` via `\graphicspath`, so regenerate them with
`python scripts/make_plots.py` before building if the results have moved.

## What is still missing

Six `\pending{...}` markers, which render in red so they cannot be missed in a draft:

| marker | what it needs |
|---|---|
| full Spec-Bench evaluation | ~12 h of machine time. The single biggest gain available: the current figures use 8 prompts against the benchmark's 480, and the wall-clock model's strategy selection rests on three operating points drawn from two prompt sets |
| second-machine replication | ×2 — a second x86 machine, which also tests the paper's one falsifiable prediction: a machine with more bandwidth per core should show a *steeper* v(k) |
| distilled draft | 14–19 GPU-hours; the plan's §11.7 budgets it and recommends against it for this paper |
| repository URL | the public repository link |
| Zenodo DOI | a tagged release, for the artifact citation |

The llama.cpp comparison was a marker here until 2026-10-25 and is now **decided against** rather
than pending; §VII states the reasoning.

`make pending` lists them with line numbers.

## Choices worth knowing about

**IEEEtran, conference format.** Not required by any venue yet. It is the right default because the
natural submission target is an IEEE venue or TechRxiv, and because a two-column format forces the
tables to stay small, which this paper benefits from.

**Every number carries its context length and its machine state.** This is not stylistic. On the
hardware used here the same unchanged binary has produced token rates spanning 15% across four
consecutive runs, so a figure without its conditions is not a figure. All measured values in the
current draft come from the single session of 2026-10-25, whose read bandwidth drifted 0.12% between
its start and its end — see `../results/README.md` for which stored runs are comparable with which.

**Section IV (Methodology) is longer than a paper of this size would normally justify.** It is kept
that way on purpose: the fixed-order sweep failure it describes is a mistake we made and published
figures from, and it is the part of this work most likely to be useful to someone else measuring on
a machine they do not control.
