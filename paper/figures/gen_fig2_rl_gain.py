"""Fig. 2: RL gain (post-GRPO minus pre-GRPO pass@k of the same initialisation) at k = 1 and 16,
with 95% bootstrap CIs over primary questions. Data: paper/data/rl_gain_ci.json."""
import numpy as np

from paper_plot_style import ARM, COL_W, INK, MUTED, legend, load, plt, save, style, title

G = load("rl_gain_ci.json")
ARMS = ["ActionSFT, seed 1", "ActionSFT, seed 2", "ActObs", "ActionSFT-LR"]
fig, axes = plt.subplots(1, 2, figsize=(COL_W, 1.75), sharey=True)
w = 0.19
for ax, (ds, name) in zip(axes, (("spider", "Spider-dev"), ("bird_dev", "BIRD-dev"))):
    for gi, k in enumerate((1, 16)):
        for i, arm in enumerate(ARMS):
            r = G[arm][f"{ds}/p{k}"]
            x = gi + (i - 1.5) * w
            fill, dark = ARM[arm]
            ax.bar(x, r["gain"], w * 0.92, color=fill, zorder=2, linewidth=0)
            ax.plot([x, x], r["ci95"], color=dark, lw=0.9, zorder=3)
            ax.text(x, r["ci95"][1] + 0.3, f"{r['gain']:.1f}", ha="center", va="bottom", fontsize=5.6, color=dark,
                    rotation=90)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["pass@1", "pass@16"])
    ax.set_xlim(-0.55, 1.55); ax.set_ylim(0, 13.5)
    ax.text(0.5, 1.02, name, transform=ax.transAxes, ha="center", va="bottom", fontsize=7.5, color=INK,
            fontweight="bold")
    style(ax)
axes[0].set_ylabel("gain during RL (points)", color=MUTED)
legend(fig, [("ActionSFT s1", "ActionSFT, seed 1"), ("ActionSFT s2", "ActionSFT, seed 2"), ("ActObs", "ActObs"),
             ("ActionSFT-LR", "ActionSFT-LR")], y=-0.12, x=0.5, loc="upper center")
fig.subplots_adjust(wspace=0.12)
save(fig, "fig2_rl_gain.pdf")
