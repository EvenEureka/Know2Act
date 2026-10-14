"""Figures for the Know2Act (ALF-CF) manuscript and project page.

Reads the frozen study-level outputs:
  results/main_analysis_test.json       (scripts/alfcf_main_analysis.py)
  results/imitator_reference.json       (scripts/alfcf_imitator_reference.py data)
and writes paper/figures/*.pdf plus docs/static/img/core.png. If an input is missing, a labelled placeholder is
written instead, so the manuscript always builds.
usage (from the repository root): python paper/scripts/make_figures.py
"""
import json
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RES = os.path.join(ROOT, "results")
FIG = os.path.join(ROOT, "paper", "figures")
IMG = os.path.join(ROOT, "docs", "static", "img")
ARMS = ["N100", "N100M", "NQ25op", "NQ50op", "NQ50", "SA50v2"]
LABEL = {"N100": "N100", "N100M": "N100M", "NQ25op": "NQ25op", "NQ50op": "NQ50op", "NQ50": "NQ50", "SA50v2": "SA50v2"}
# reference categorical palette (light), validated: blue, orange, aqua, yellow; ink and surface tokens
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
INK, INK2, GRID, SURF, REF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb", "#8a8984"
plt.rcParams.update({"font.size": 8, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF})


def load(name):
    p = os.path.join(RES, name)
    return json.load(open(p)) if os.path.exists(p) else None


def placeholder(path, text):
    fig, ax = plt.subplots(figsize=(3.0, 2.0))
    ax.axis("off")
    ax.text(0.5, 0.5, text, ha="center", va="center", color=INK2, wrap=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def t975(df):
    table = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228}
    return table.get(df, 1.96)


def mean_ci(v):
    v = [x for x in v if x is not None]
    if not v:
        return None, None
    m = sum(v) / len(v)
    if len(v) < 2:
        return m, 0.0
    sd = math.sqrt(sum((x - m) ** 2 for x in v) / (len(v) - 1))
    return m, t975(len(v) - 1) * sd / math.sqrt(len(v))


def reference_bounds(ref, arm):
    vals = [v for k, v in ref.items() if k.split("|")[0] == arm]
    if not vals:
        return None
    if vals[0]["kind"] == "point":
        return (sum(v["S_star"] for v in vals) / len(vals),) * 2
    return (sum(v["S_star_low"] for v in vals) / len(vals), sum(v["S_star_high"] for v in vals) / len(vals))


def fig_flip_by_arm(main, ref, out_paths):
    """Core figure: S_flip@first per arm (seeds, mean +- t CI) against the imitator reference."""
    per = main["per_arm"]["primary"]
    arms = [a for a in ARMS if a in per]
    fig, ax = plt.subplots(figsize=(3.0, 2.2))
    for i, arm in enumerate(arms):
        lo_hi = reference_bounds(ref, arm) if ref else None
        if lo_hi:
            lo, hi = lo_hi
            if hi - lo > 1e-9:
                ax.add_patch(plt.Rectangle((i - 0.32, lo), 0.64, hi - lo, color=GRID, zorder=0, lw=0))
            else:
                ax.plot([i - 0.32, i + 0.32], [lo, lo], color=REF, lw=2, zorder=1)
        seeds = [x for x in per[arm]["S_flip_first_L"]["per_seed"] if x is not None]
        n = len(seeds)
        for j, s in enumerate(seeds):
            ax.scatter(i + (j - (n - 1) / 2) * 0.05, s, s=9, color=C[0], alpha=0.55, lw=0, zorder=3)
        m, h = mean_ci(seeds)
        if m is not None:
            ax.errorbar(i + 0.22, m, yerr=h, fmt="o", ms=4, color=INK, elinewidth=1.2, capsize=0, zorder=4)
    ax.set_xticks(range(len(arms)))
    ax.set_xticklabels([LABEL[a] for a in arms], rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("flip effect at first decision")
    ax.set_ylim(-0.05, 1.02)
    ax.axhline(0, color=GRID, lw=0.8, zorder=0)
    ax.grid(axis="y", color=GRID, lw=0.5)
    for p in out_paths:
        fig.savefig(p, bbox_inches="tight", dpi=220)
    plt.close(fig)


def fig_dissociation(main, out_path):
    """Per arm: flip effect, removal effect, random-order-note effect, presence effect (means over seeds)."""
    per = main["per_arm"]["primary"]
    arms = [a for a in ARMS if a in per]
    series = [("flip", "S_flip_first_L"), ("removal", "removal_null"), ("random order", "S_random"),
              ("presence", "presence_effect")]
    fig, ax = plt.subplots(figsize=(3.0, 2.3))
    w = 0.19
    for k, (name, key) in enumerate(series):
        xs, ys, es = [], [], []
        for i, arm in enumerate(arms):
            m, h = mean_ci(per[arm][key]["per_seed"])
            if m is None:
                continue
            xs.append(i + (k - 1.5) * (w + 0.01)); ys.append(m); es.append(h)
        ax.bar(xs, ys, width=w, color=C[k], label=name, zorder=2)
        ax.errorbar(xs, ys, yerr=es, fmt="none", ecolor=INK2, elinewidth=0.8, zorder=3)
    ax.axhline(0, color=INK2, lw=0.6)
    ax.set_xticks(range(len(arms)))
    ax.set_xticklabels([LABEL[a] for a in arms], rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("effect on next action")
    ax.grid(axis="y", color=GRID, lw=0.5, zorder=0)
    ax.legend(frameon=False, ncol=2, fontsize=7.5, loc="lower center", bbox_to_anchor=(0.5, 1.0), handlelength=1.2, columnspacing=0.8)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main():
    os.makedirs(FIG, exist_ok=True)
    os.makedirs(IMG, exist_ok=True)
    main_json, ref = load("main_analysis_test.json"), load("imitator_reference.json")
    if main_json:
        fig_flip_by_arm(main_json, ref, [os.path.join(FIG, "fig_flip_by_arm.pdf"), os.path.join(IMG, "core.png")])
        fig_dissociation(main_json, os.path.join(FIG, "fig_dissociation.pdf"))
    else:
        for p in [os.path.join(FIG, "fig_flip_by_arm.pdf"), os.path.join(FIG, "fig_dissociation.pdf"), os.path.join(IMG, "core.png")]:
            placeholder(p, "Results pending:\nheld-out test analysis not yet run")
    print("figures written to", FIG)


if __name__ == "__main__":
    main()
