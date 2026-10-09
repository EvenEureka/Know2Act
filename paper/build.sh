#!/bin/bash
# Build the anonymous review PDF (main.pdf). For the public PDF and the website figures, run
# paper/build_site_assets.sh instead.
set -e
cd "$(dirname "$0")"
pdflatex -interaction=nonstopmode main.tex >/dev/null
bibtex main >/dev/null
pdflatex -interaction=nonstopmode main.tex >/dev/null
pdflatex -interaction=nonstopmode main.tex | grep -E "Output written|Warning: (Reference|Citation)" || true
