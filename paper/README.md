# The paper

`main.pdf` is the compiled paper, tracked in this repository and linked from the top-level README.
`main.tex` is its source, in IEEEtran. `../docs/WRITEUP.md` is the prose version it was built from
and is the easier thing to read.

It is published here rather than at a preprint server. That is a deliberate choice and not a
placeholder: nothing in the paper is waiting on a venue, and hosting it here keeps the text, the
figures, the stored measurements and the test suite that checks the text against the measurements in
one place, at one revision.

## Building

```bash
make            # pdflatex, bibtex, pdflatex, pdflatex
make check      # structural checks, no toolchain needed
make clean
```

Needs IEEEtran, which ships with TeX Live and MiKTeX (`texlive-publishers` on Debian/Ubuntu).
Figures are read from `../results/figures/` via `\graphicspath`, so regenerate them with
`python scripts/make_plots.py` before building if the results have moved.

`make check` runs checks that need no toolchain: environment and brace balance, bare `%` after a
digit (which silently eats the rest of its line), `\ref` without `\label`, and `\cite` keys missing
from `refs.bib`. Passing it is not the same as compiling, so it is a pre-filter rather than a gate.

## What the paper does not have

There were five `\pending{...}` markers here, rendered in red so a draft could be swept for them.
All five are resolved, and the macro that drew them is gone. Three were resolved by measurement or
by publication; two were resolved by being demoted to stated limitations, which is the honest
outcome rather than a workaround:

| was pending | outcome |
|---|---|
| full Spec-Bench evaluation | **Measured.** All 480 questions, 2240 turns, 2026-10-27. It changed the headline: speculation reads 1.11× on decoding but 0.98× on the clock, where the 8-prompt subset read 1.19× and 1.07× |
| repository URL | **Resolved**; the Artifacts section links this repository |
| Zenodo DOI | **Dropped.** No venue, so no artifact citation to anchor |
| second-machine replication | **A limitation.** Still the paper's one falsifiable prediction. Section VII now also notes the cheaper half of it: the engine picks its kernel at runtime, so forcing the slower path tests the same mechanism on one machine with memory traffic held fixed |
| distilled draft | **A limitation**, with the cost of not having it quantified: parity on the clock needs only +0.04× on decoding, against the +0.08× distillation is budgeted to buy |

## Choices worth knowing about

**IEEEtran, conference format.** Not required by anyone. It is kept because the two-column measure
forces the tables to stay small, which this paper benefits from, and because the format is
immediately legible as a paper rather than as documentation.

**Every number carries its context length and its machine state.** This is not stylistic. On the
hardware used here the same unchanged binary has produced token rates spanning 15% across four
consecutive runs, so a figure without its conditions is not a figure. See `../results/README.md` for
which stored runs are comparable with which — and note that the 480-question run spans two machine
states, so its *ratios* are quotable while its absolute rates are not.

**Section IV (Methodology) is longer than a paper of this size would normally justify.** It is kept
that way on purpose: the fixed-order sweep failure it describes is a mistake we made and published
figures from, and it is the part of this work most likely to be useful to someone else measuring on
a machine they do not control.

**The numbers in the text are tested.** `tests/test_documented_numbers.py` asserts that the figures
quoted in prose are derivable from the files in `results/`. Stale numbers have slipped through twice
in this project's history, which is why it is a test rather than a careful read.
