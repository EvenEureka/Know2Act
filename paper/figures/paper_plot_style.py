"""Shared figure style, matched to the MemoHarness / SkillRC figures (pastel fills, darker label
colours, Droid Sans, no left/top/right spines, light grid). ACL column width is ~3.1 in."""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
COL_W, PAGE_W = 3.1, 6.3  # inches
INK, MUTED, GRID, GREY_BAND = "#3A3A3A", "#7A7A7A", "#E6E6E6", "#F1F1F1"
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Droid Sans", "DejaVu Sans"], "font.size": 7.5,
    "axes.edgecolor": "#BDBDBD", "axes.linewidth": 0.7, "xtick.color": INK, "ytick.color": MUTED,
    "xtick.labelsize": 7, "ytick.labelsize": 6.8, "axes.labelcolor": INK, "axes.labelsize": 7.3,
    "pdf.fonttype": 42, "savefig.bbox": "tight", "savefig.pad_inches": 0.03, "legend.frameon": False,
})
# (fill, dark) per arm; fixed across all figures
ARM = {"base": ("#E3E1F2", "#8A85B8"), "ActionSFT": ("#BDCFEA", "#5F86C2"),
       "ActionSFT, seed 1": ("#BDCFEA", "#5F86C2"), "ActionSFT, seed 2": ("#D9E4F4", "#5F86C2"),
       "ActObs": ("#F4C4B5", "#D9735A"),
       "ActObs, seed 1": ("#F4C4B5", "#D9735A"), "ActObs, seed 2": ("#F9E0D7", "#D9735A"), "ActionSFT-LR": ("#BFE3D9", "#4FA58C"),
       "ActionSFT+Wiki": ("#EFE0B6", "#B8923A"), "noise": ("#E4E4E4", "#8C8C8C"),
       "Contract": ("#D5CDEB", "#6B4C9A")}


def style(ax, ygrid=True, xgrid=False):
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(axis="both", length=0)
    if ygrid:
        ax.yaxis.grid(True, color=GRID, linewidth=0.7)
    if xgrid:
        ax.xaxis.grid(True, color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)


def title(fig, text, x=0.01, y=1.0, size=8.5):
    fig.text(x, y, text, ha="left", va="bottom", fontsize=size, color=INK)


def legend(fig, items, y=1.0, x=0.99, loc="lower right", ncol=None):
    fig.legend(handles=[Patch(facecolor=ARM[k][0], edgecolor="none", label=lab) for lab, k in items],
               loc=loc, bbox_to_anchor=(x, y), ncol=ncol or len(items), fontsize=6.6, handlelength=0.9,
               handleheight=0.9, columnspacing=0.9, handletextpad=0.35)


def load(name="results.json"):
    return json.load(open(os.path.join(HERE, "..", "data", name)))


PREVIEW = os.environ.get("FIG_PREVIEW_DIR")  # optional PNG previews (not part of the paper)


def save(fig, name):
    path = os.path.join(HERE, name)
    fig.savefig(path)
    if PREVIEW:
        os.makedirs(PREVIEW, exist_ok=True)
        fig.savefig(os.path.join(PREVIEW, name.replace(".pdf", ".png")), dpi=int(os.environ.get("FIG_PREVIEW_DPI", 220)))
    plt.close(fig)
    print("saved", path)
