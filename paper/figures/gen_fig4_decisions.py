"""Fig. 4: decision sensitivity relative to ActionSFT (forest plot, 95% question-cluster bootstrap CIs).
S = P(submit | observed shape == gold shape) - P(submit | shape != gold); bars show S(arm) - S(ActionSFT).
Grey rows are noise references (different data order before RL; different RL seed after RL).
Data: results/decision_evidence.json."""
import json
import os

from paper_plot_style import ARM, COL_W, GREY_BAND, INK, MUTED, legend, plt, save, style

E = json.load(open(os.path.join(os.path.dirname(__file__), "..", "..", "results", "decision_evidence.json")))
ROWS = [("SFT", "ActionSFT (new data order)", "noise"), ("SFT", "ActObs", None), ("SFT", "ActionSFT-LR", None),
        ("SFT", "Outcome-Aux", None), ("SFT", "Outcome-in-Action", None), ("SFT", "Contract", None),
        ("SFT", "Contract-Placebo", None),
        ("GRPO", "ActionSFT (seed 2)", "noise"), ("GRPO", "ActObs", None), ("GRPO", "ActionSFT-LR", None),
        ("GRPO", "Contract", None), ("GRPO", "ActObs (seed 2)", None), ("GRPO", "Outcome-in-Action", None)]
ROWS = [r for r in ROWS if f"{r[0]}/bird_dev/{r[1]}" in E]
DS = [("spider_dev", "Spider-dev", ARM["ActionSFT"]), ("bird_dev", "BIRD-dev", ARM["ActionSFT-LR"])]

fig, ax = plt.subplots(figsize=(COL_W, 0.215 * len(ROWS) + 0.75))
ys, labels, y = [], [], 0.0
for k, (stage, name, kind) in enumerate(ROWS):
    if k > 0 and ROWS[k - 1][0] != stage:
        y -= 0.55
    # after GRPO, ActObs has two RL seeds; name the first explicitly
    ys.append(y); labels.append("ActObs (seed 1)" if (stage, name) == ("GRPO", "ActObs") and
                                f"GRPO/bird_dev/ActObs (seed 2)" in E else name)
    if kind == "noise":
        ax.axhspan(y - 0.42, y + 0.42, color=GREY_BAND, zorder=0, lw=0)
    for j, (ds, _, (fill, dark)) in enumerate(DS):
        r = E[f"{stage}/{ds}/{name}"]
        yy = y + (0.17 if j == 0 else -0.17)
        ax.barh(yy, r["dS_vs_ActionSFT"], 0.3, color="#CFCFCF" if kind == "noise" else fill, zorder=2)
        ax.plot(r["dS_ci95"], [yy, yy], color=MUTED if kind == "noise" else dark, lw=1.0, zorder=3)
    y -= 1.0
ax.axvline(0, color="#9E9E9E", lw=0.8, zorder=1)
ax.set_yticks(ys); ax.set_yticklabels(labels, fontsize=6.5, color=INK)
for t, (stage, name, kind) in zip(ax.get_yticklabels(), ROWS):
    if kind == "noise":
        t.set_color(MUTED); t.set_fontstyle("italic")
sft_mid = sum(ys[i] for i, r in enumerate(ROWS) if r[0] == "SFT") / sum(r[0] == "SFT" for r in ROWS)
rl_mid = sum(ys[i] for i, r in enumerate(ROWS) if r[0] == "GRPO") / sum(r[0] == "GRPO" for r in ROWS)
for mid, txt in ((sft_mid, "after SFT"), (rl_mid, "after GRPO")):
    ax.text(1.02, mid, txt, transform=ax.get_yaxis_transform(), rotation=270, va="center", ha="left",
            fontsize=6.5, color=MUTED)
ax.set_xlim(-0.11, 0.11)
ax.set_xlabel("change in decision sensitivity vs. ActionSFT", color=MUTED)
style(ax, ygrid=False, xgrid=True)
ax.set_title("Does the knowledge change when the agent submits?", fontsize=7.6, color=INK,
             fontweight="bold", loc="right", pad=5)
legend(fig, [("Spider-dev", "ActionSFT"), ("BIRD-dev", "ActionSFT-LR")], y=-0.02, x=0.55, loc="upper center")
save(fig, "fig4_decisions.pdf")
