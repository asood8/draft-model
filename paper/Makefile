# Build the paper. Needs a TeX distribution with IEEEtran; see README.md.
#
#   make          build main.pdf
#   make check    structural checks that do not need a toolchain
#   make clean    remove intermediates (NOT main.pdf, which is a tracked artifact)

LATEX ?= pdflatex
BIBTEX ?= bibtex
PYTHON ?= ../.venv/Scripts/python.exe

main.pdf: main.tex refs.bib
	$(LATEX) -interaction=nonstopmode -halt-on-error main.tex
	$(BIBTEX) main || true
	$(LATEX) -interaction=nonstopmode -halt-on-error main.tex
	$(LATEX) -interaction=nonstopmode -halt-on-error main.tex

.PHONY: check clean

check:
	$(PYTHON) check.py

clean:
	rm -f main.aux main.bbl main.blg main.log main.out main.spl
