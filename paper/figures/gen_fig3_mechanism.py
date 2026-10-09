"""Fig. 3: (a) observation CE by part for Qwen3-4B / ActionSFT / ActObs on unseen vs training databases;
(b) held-out action KL to ActionSFT vs the data-order noise floor (grey band).
Panel order follows the paper: what is learned first, then whether it reaches the actions."""
import numpy as np

from paper_plot_style import ARM, COL_W, GREY_BAND, INK, MUTED, legend, load, plt, save, style

D = load()
kl, ob = D["diag"]["kl_vs_A"], D["diag"]["obs_breakdown"]
fig = plt.figure(figsize=(COL_W, 2.9))
gs = fig.add_gridspec(2, 2, height_ratios=[1.25, 0.78], hspace=0.62, wspace=0.08)

# (a) observation CE by part
parts = [("rows/wrapper", "templ."), ("rows/first3", "header\nstart"), ("rows/line1", "header\nrest"),
         ("rows/rest", "row\nvalues")]
models = [("base", "Qwen3-4B"), ("A", "ActionSFT"), ("B", "ActObs")]
w, x = 0.26, np.arange(len(parts))
for j, (split, name) in enumerate((("heldout", "unseen databases"), ("train", "training databases"))):
    ax = fig.add_subplot(gs[0, j])
    for i, (m, lab) in enumerate(models):
        fill, dark = ARM[{"base": "base", "A": "ActionSFT", "B": "ActObs"}[m]]
        vals = [ob[m][split][p]["ce"] for p, _ in parts]
        ax.bar(x + (i - 1) * w, vals, w * 0.92, color=fill, zorder=2)
        if m == "B":
            for xi, v in zip(x + (i - 1) * w, vals):
                ax.text(xi, v + 0.15, f"{v:.2f}", ha="center", va="bottom", fontsize=5.4, color=dark, rotation=90)
    ax.set_xticks(x); ax.set_xticklabels([p[1] for p in parts], fontsize=6.2)
    ax.set_ylim(0, 8.4)
    ax.text(0.5, 1.02, name, transform=ax.transAxes, ha="center", va="bottom", fontsize=7, color=INK)
    style(ax)
    if j == 0:
        ax.set_ylabel("observation CE (nats)", color=MUTED)
        ax.set_title("(a) What is learned from observations", fontsize=7.6, color=INK, fontweight="bold",
                     loc="left", pad=13)
    else:
        ax.set_yticklabels([])
legend(fig, [("Qwen3-4B", "base"), ("ActionSFT", "ActionSFT"), ("ActObs", "ActObs")], y=1.0, x=0.99,
       loc="lower right")

# (b) action KL to ActionSFT
ax = fig.add_subplot(gs[1, :])
rows = [("ActionSFT-LR (0.85× lr)", kl["C085"], "ActionSFT-LR"), ("ActionSFT+Wiki", kl["D"], "ActionSFT+Wiki"),
        ("ActObs", kl["B"], "ActObs"), ("ActionSFT, new data order", kl["Aseed1"], "noise")]
ax.axvspan(0, kl["Aseed1"], color=GREY_BAND, zorder=0, lw=0)
for yi, (lab, v, arm) in zip(range(len(rows))[::-1], rows):
    fill, dark = ARM[arm]
    ax.barh(yi, v, 0.62, color="#CFCFCF" if arm == "noise" else fill, zorder=2)
    ax.text(v + 0.00012, yi, f"{v:.4f}", va="center", fontsize=6.4, color=dark)
ax.set_yticks(range(len(rows))[::-1]); ax.set_yticklabels([r[0] for r in rows], color=INK)
ax.set_xlim(0, 0.0092); ax.set_xticks([0, 0.002, 0.004, 0.006, 0.008])
ax.set_xticklabels(["0", ".002", ".004", ".006", ".008"])
style(ax, ygrid=False, xgrid=True)
ax.set_title("(b) Does it reach the actions? Held-out action KL to ActionSFT", fontsize=7.6, color=INK,
             fontweight="bold", loc="left", pad=3)

save(fig, "fig3_mechanism.pdf")
