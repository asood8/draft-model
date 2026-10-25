# Build the paper. Needs a TeX distribution with IEEEtran; see README.md.
#
#   make          build main.pdf
#   make check    structural checks that do not need a toolchain
#   make clean    remove build products

LATEX ?= pdflatex
BIBTEX ?= bibtex
PYTHON ?= ../.venv/Scripts/python.exe

main.pdf: main.tex refs.bib
	$(LATEX) -interaction=nonstopmode -halt-on-error main.tex
	$(BIBTEX) main || true
	$(LATEX) -interaction=nonstopmode -halt-on-error main.tex
	$(LATEX) -interaction=nonstopmode -halt-on-error main.tex

.PHONY: check clean pending

check:
	$(PYTHON) check.py

# Every number still to be measured, so the draft can be swept for them.
pending:
	@grep -n '\\pending{' main.tex | grep -v newcommand

clean:
	rm -f main.pdf main.aux main.bbl main.blg main.log main.out main.spl
