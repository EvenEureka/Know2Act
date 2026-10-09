#!/bin/bash
# Rebuild the website's figures (PNG) and the public PDF from the paper sources.
# Usage: bash paper/build_site_assets.sh   (from the repository root)
set -e
ROOT=$(cd "$(dirname "$0")/.." && pwd)
PY=${PY:-python}   # needs matplotlib, numpy and Pillow
IMG=$ROOT/docs/static/img
TMP=$(mktemp -d)
cd "$ROOT/paper"

# Figures 2-4: the paper's own plotting scripts, with PNG export switched on
for s in gen_fig2_rl_gain gen_fig3_mechanism gen_fig4_decisions; do
  FIG_PREVIEW_DIR=$TMP FIG_PREVIEW_DPI=440 $PY figures/$s.py >/dev/null
done
cp $TMP/fig2_rl_gain.png $IMG/fig_rl_gain.png
cp $TMP/fig3_mechanism.png $IMG/fig_mechanism.png
cp $TMP/fig4_decisions.png $IMG/fig_decisions.png

# Figure 1: the TikZ source compiled on its own page
cat > $TMP/fig1.tex <<'TEX'
\documentclass{article}
\usepackage[paperwidth=40cm,paperheight=16cm,margin=1cm]{geometry}
\pagestyle{empty}
\usepackage{times}
\usepackage[T1]{fontenc}
\usepackage{amsmath}
\usepackage{xcolor}
\usepackage{tikz}
\usetikzlibrary{arrows.meta,positioning,calc,decorations.pathreplacing}
\definecolor{phaseA}{HTML}{6F9AD3}\definecolor{phaseB}{HTML}{B07A8F}
\definecolor{gptblue}{HTML}{5F86C2}\definecolor{qwenmint}{HTML}{4FA58C}
\definecolor{salmon}{HTML}{D9735A}\definecolor{sand}{HTML}{B8923A}
\definecolor{contractv}{HTML}{6B4C9A}
\begin{document}
\noindent\input{figures/fig1_overview_tikz}
\end{document}
TEX
timeout 120 pdflatex -interaction=nonstopmode -output-directory=$TMP $TMP/fig1.tex >/dev/null
timeout 60 gs -q -dNOPAUSE -dBATCH -sDEVICE=png16m -r400 -dTextAlphaBits=4 -dGraphicsAlphaBits=4 \
  -sOutputFile=$IMG/fig_overview.png $TMP/fig1.pdf
# trim the empty page around the drawing, keeping a small white margin
$PY -c "
from PIL import Image, ImageOps
im = Image.open('$IMG/fig_overview.png').convert('RGB')
x0, y0, x1, y1 = ImageOps.invert(im).getbbox()
m = 40
im.crop((max(x0 - m, 0), max(y0 - m, 0), min(x1 + m, im.width), min(y1 + m, im.height))).save('$IMG/fig_overview.png')
"

# Public PDF: camera-ready layout with the author block instead of the anonymous review header
sed -e 's/\\usepackage\[review\]{acl}/\\usepackage[final]{acl}\n\\newcommand{\\skyrlissue}{ (SkyRL issue \\#2451)}/' \
    -e 's/\\author{Anonymous ARR Submission}/\\author{Yiwen Lu \\\\ University of Notre Dame \\\\ \\texttt{ylu37@nd.edu}}/' \
    main.tex > main_public.tex
for step in 1 2; do timeout 120 pdflatex -interaction=nonstopmode main_public.tex >/dev/null || true; done
timeout 60 bibtex main_public >/dev/null || true
for step in 1 2; do timeout 120 pdflatex -interaction=nonstopmode main_public.tex >/dev/null || true; done
cp main_public.pdf "$ROOT/docs/static/paper.pdf"; cp main_public.pdf "$ROOT/paper/manuscript.pdf"
rm -rf $TMP
echo "assets rebuilt in $ROOT/docs/static and paper/manuscript.pdf"
