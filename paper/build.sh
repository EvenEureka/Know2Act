#!/bin/bash
# Build the manuscript PDF (requires pdflatex and bibtex). Figures: python paper/scripts/make_figures.py (repo root).
set -e
cd "$(dirname "$0")"
pdflatex -interaction=nonstopmode main >/dev/null
bibtex main >/dev/null || true
pdflatex -interaction=nonstopmode main >/dev/null
pdflatex -interaction=nonstopmode main >/dev/null
cp main.pdf manuscript.pdf
echo "built paper/manuscript.pdf"
