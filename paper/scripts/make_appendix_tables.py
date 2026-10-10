"""Write every appendix table as LaTeX from the result files (no hand-typed numbers).
Output: paper/tables/*.tex. Re-run after new results land."""
import glob
import json
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
R, OUT = os.path.join(ROOT, "results"), os.path.join(ROOT, "paper", "tables")
os.makedirs(OUT, exist_ok=True)
J = lambda p: json.load(open(p))  # noqa: E731
RES = J(os.path.join(ROOT, "paper", "data", "results.json"))
PEND = r"\pending"


def f(v, d=1):
    return PEND if v is None else f"{v:.{d}f}"


def write(name, body):
    open(os.path.join(OUT, name), "w").write(body)
    print("wrote", name)


def table(cols, header, rows, caption, label, star=False, size="\\small", fit=True):
    env = "table*" if star else "table"
    width = "\\textwidth" if star else "\\linewidth"
    # fit=False keeps the natural font size instead of scaling the tabular to the column width
    box = f"\\resizebox{{{width}}}{{!}}{{%" if fit else "{\\setlength{\\tabcolsep}{3.5pt}%"
    lines = [f"\\begin{{{env}}}[t]", "\\centering" + size, box,
             f"\\begin{{tabular}}{{{cols}}}", "\\toprule", header, "\\midrule"]
    lines += rows + ["\\bottomrule", "\\end{tabular}}", f"\\caption{{{caption}}}", f"\\label{{{label}}}",
                     f"\\end{{{env}}}", ""]
    return "\n".join(lines)


# --- full pass@k ---------------------------------------------------------------------------------
E = RES["eval"]
MODELS = [("Qwen3-4B (base)", "base"), ("\\armA{}", "sft_A"), ("\\armA{}, data order 2", "sft_Aseed1"),
          ("\\armB{}", "sft_B"), ("\\armC{}", "sft_C"), ("\\armD{}", "sft_D"), ("\\armE{}", "sft_E1"),
          ("\\armF{}", "sft_E2"), ("\\armG{}", "sft_E3"), ("\\armGp{}", "sft_E3p"),
          ("GRPO \\armA{}, seed 1", "grpo_A_s1"), ("GRPO \\armA{}, seed 2", "grpo_A_s2"),
          ("GRPO \\armB{}, seed 1", "grpo_B_s1"), ("GRPO \\armB{}, seed 2", "grpo_B_s2"),
          ("GRPO \\armC{}, seed 1", "grpo_C_s1"), ("GRPO \\armF{}, seed 1", "grpo_E2_s1"),
          ("GRPO \\armG{}, seed 1", "grpo_E3_s1")]
rows = []
for lab, key in MODELS:
    e = E.get(key)
    vals = []
    for ds in ("spider", "bird"):
        c = e[f"{ds}_primary_curve"] if e else None
        vals += [f(c[k - 1]) if c else PEND for k in (1, 2, 4, 8, 16)]
    if lab.startswith("GRPO \\armA{}, seed 1"):
        rows.append("\\midrule")
    rows.append(lab + " & " + " & ".join(vals) + " \\\\")
write("full_passk.tex", table("@{}l" + "c" * 10 + "@{}",
      "& \\multicolumn{5}{c}{Spider-dev (983 questions)} & \\multicolumn{5}{c}{BIRD-dev (1,532 questions)} \\\\\n"
      "\\cmidrule(lr){2-6}\\cmidrule(lr){7-11}\nModel & p@1 & p@2 & p@4 & p@8 & p@16 & p@1 & p@2 & p@4 & p@8 & p@16 \\\\",
      rows, "pass@$k$ (\\%) on the primary sets for every model, 16 attempts per question (unbiased estimator).",
      "tab:fullpassk", star=True))

# --- paired contrasts ------------------------------------------------------------------------------
P = {}
for fn in glob.glob(f"{R}/eval/primary_pairs_*.json"):
    P.update(J(fn)["pairs"])
PA = J(f"{R}/paper_analyses.json")["pooled_primary"]["pairs"]
_e2 = os.path.join(R, "eval", "pooled_E2_vs_meanA.json")  # Outcome-in-Action GRPO vs mean of the ActionSFT seeds (2026-10-10)
if os.path.exists(_e2):
    PA.update({"grpoE2-grpoAmean/" + k.split("/")[1]: v for k, v in J(_e2).items() if k.startswith("E2-meanA/")})
CON = [("\\armA{} s2 $-$ \\armA{} s1", "A1->A2", "grpoA2-grpoA1"),
       ("\\armB{} s1 $-$ \\armA{} s1", "A1->B1", "grpoB1-grpoA1"),
       ("\\armB{} s1 $-$ \\armA{} s2", "A2->B1", "grpoB1-grpoA2"),
       ("\\armB{} s2 $-$ \\armA{} s1", "A1->B2", None), ("\\armB{} s2 $-$ \\armA{} s2", "A2->B2", None),
       ("\\armB{} s2 $-$ \\armB{} s1", "B1->B2", "grpoB2-grpoB1"),
       ("\\armC{} s1 $-$ \\armB{} s1", "B1->C1", "grpoC1-grpoB1"),
       ("\\armG{} s1 $-$ \\armA{} s1", "A1->E3", None), ("\\armG{} s1 $-$ \\armA{} s2", "A2->E3", None),
       ("\\armF{} s1 $-$ \\armA{} s1", "A1->E2", None), ("\\armF{} s1 $-$ \\armA{} s2", "A2->E2", None)]


def ci(d):
    return f"{d['B_minus_A_pp']:+.2f} [{d['ci95'][0]:+.2f}, {d['ci95'][1]:+.2f}]"


rows = []
for lab, key, pooled in CON:
    d = P.get(key, {})
    cells = [ci(d[k]) if k in d else PEND for k in ("spider/pass@1", "spider/pass@16", "bird_dev/pass@1", "bird_dev/pass@16")]
    pl = PA.get(f"{pooled}/p16") if pooled else None
    cells.append(f"{pl['diff_pp']:+.2f} [{pl['ci95'][0]:+.2f}, {pl['ci95'][1]:+.2f}]" if pl else "--")
    rows.append(lab + " & " + " & ".join(cells) + " \\\\")
for lab, key in (("mean(\\armB{}) $-$ mean(\\armA{})", "grpoBmean-grpoAmean"),
                 ("\\armB{} s1 $-$ mean(\\armA{})", "grpoB1-grpoAmean"), ("\\armB{} s2 $-$ mean(\\armA{})", "grpoB2-grpoAmean"), ("\\armG{} s1 $-$ mean(\\armA{})", "grpoE3-grpoAmean"), ("\\armF{} s1 $-$ mean(\\armA{})", "grpoE2-grpoAmean"),
                 ("\\armC{} s1 $-$ mean(\\armA{})", "grpoC1-grpoAmean")):
    p1, p16 = PA.get(f"{key}/p1"), PA.get(f"{key}/p16")
    if p1:
        rows.append(f"{lab} (pooled) & \\multicolumn{{4}}{{c}}{{pass@1: {p1['diff_pp']:+.2f} [{p1['ci95'][0]:+.2f}, {p1['ci95'][1]:+.2f}]}} & "
                    f"{p16['diff_pp']:+.2f} [{p16['ci95'][0]:+.2f}, {p16['ci95'][1]:+.2f}] \\\\")
write("pairs.tex", table("@{}lccccc@{}",
      "Contrast (after GRPO) & Spider p@1 & Spider p@16 & BIRD p@1 & BIRD p@16 & Pooled p@16 \\\\",
      rows, "Paired differences in points with 95\\% bootstrap CIs over primary questions (10,000 resamples; 2,000 for "
      "\\armC{}$-$\\armB{}). Pooled rows combine Spider and BIRD (2,515 questions); the first is the pre-registered primary outcome.",
      "tab:pairs", star=True))

# --- observation CE breakdown ------------------------------------------------------------------------
OB = RES["diag"]["obs_breakdown"]
PARTS = [("rows/wrapper", "template"), ("rows/first3", "header (first 3 tokens)"), ("rows/line1", "header (rest of line)"),
         ("rows/rest", "row values"), ("error/first3", "error (first 3 tokens)"), ("error/line1", "error message"),
         ("empty/first3", "empty result (first 3 tokens)")]
rows = []
for key, lab in PARTS:
    cells = []
    for split in ("heldout", "train"):
        for m in ("base", "A", "B", "D"):
            v = OB[m][split].get(key, {}).get("ce")
            cells.append(f(v, 2))
        cells.append(f"{OB['B'][split].get(key, {}).get('n_tokens', 0):,}")
    rows.append(lab + " & " + " & ".join(cells) + " \\\\")
write("obs_ce.tex", table("@{}l" + "c" * 10 + "@{}",
      "& \\multicolumn{5}{c}{Unseen (held-out) databases} & \\multicolumn{5}{c}{Training databases} \\\\\n"
      "\\cmidrule(lr){2-6}\\cmidrule(lr){7-11}\nPart & base & \\armA{} & \\armB{} & \\armD{} & tokens & base & \\armA{} & \\armB{} & \\armD{} & tokens \\\\",
      rows, "Observation cross-entropy (nats per token) by part on 300 held-out and 300 training trajectories. "
      "The ``rows'' parts also include invalid-action notices, which the breakdown does not separate.",
      "tab:obsce", star=True))

# --- RL dynamics -------------------------------------------------------------------------------------
LOGS = {"\\armA{}, seed 1": "grpo_A_s1.1508971", "\\armA{}, seed 2": "grpo_A_s2.1510961", "\\armB{}, seed 1": "grpo_B_s1.1510918",
        "\\armB{}, seed 2": "grpo_B_s2.1510964", "\\armC{}, seed 1": "grpo_C_s1.1510955", "\\armG{}, seed 1": "grpo_E3_s1.1516402"}
MOVE = {"\\armA{}, seed 1": "grpo_A_s1_1508971", "\\armA{}, seed 2": "grpo_A_s2_1510961", "\\armB{}, seed 1": "grpo_B_s1_1510918",
        "\\armB{}, seed 2": "grpo_B_s2_1510964", "\\armC{}, seed 1": "grpo_C_s1_1510955", "\\armG{}, seed 1": "grpo_E3_s1_1516402"}
rows = []
for lab, lg in LOGS.items():
    t = open(os.path.join(ROOT, "logs", lg + ".log"), errors="ignore").read()
    v = [float(x) for x in re.findall(r"avg_final_rewards': '([0-9.]+)'", t)]
    w = lambda a, b: sum(v[a:b]) / len(v[a:b])  # noqa: E731
    mv = RES["rl_move"].get(MOVE[lab], {})
    rows.append(f"{lab} & {w(0, 10):.3f} & {w(30, 40):.3f} & {w(60, 70):.3f} & {w(115, 135):.3f} & "
                f"{f(mv.get('kl_act'), 3)} & {f(mv.get('ent_act'), 3)} \\\\")
write("rl_dynamics.tex", table("@{}lcccccc@{}",
      "& \\multicolumn{4}{c}{Reward, steps} & KL to & \\\\ \\cmidrule(lr){2-5}\n"
      "GRPO run & 1--10 & 31--40 & 61--70 & 116--135 & SFT & Entropy \\\\", rows,
      "Mean training reward over step windows, and action KL to the run's own SFT checkpoint and action entropy on 300 held-out trajectories. "
      "\\armA{} seed 1 is the only run on 4 GPUs.", "tab:rldyn", star=True, fit=False))

# --- behaviour ------------------------------------------------------------------------------------------
B = J(f"{R}/behaviour_obs.json")
BR = [("\\armA{} SFT", "sftA"), ("\\armB{} SFT", "sftB"), ("\\armE{} SFT", "sftE1"), ("\\armF{} SFT", "sftE2"),
      ("\\armG{} SFT", "sftE3"), ("\\armA{} GRPO s1", "grpoA1"), ("\\armA{} GRPO s2", "grpoA2"), ("\\armB{} GRPO", "grpoB1"),
      ("\\armC{} GRPO", "grpoC1")]
rows = []
for lab, k in BR:
    cells = []
    for ds in ("spider_dev", "bird_dev"):
        r = B.get(f"{k}/{ds}")
        cells += [f(r["turns"], 2), f(100 * r["traj_with_error"]), f(100 * r["solved_after_error"]),
                  f(100 * r["error_retry_same_sql"]), f(100 * r["empty_then_answer"])] if r else [PEND] * 5
    rows.append(lab + " & " + " & ".join(cells) + " \\\\")
write("behaviour.tex", table("@{}l" + "c" * 10 + "@{}",
      "& \\multicolumn{5}{c}{Spider-dev} & \\multicolumn{5}{c}{BIRD-dev} \\\\\n\\cmidrule(lr){2-6}\\cmidrule(lr){7-11}\n"
      "Model & turns & err.\\ traj. & solved & retry & empty$\\to$ans. & turns & err.\\ traj. & solved & retry & empty$\\to$ans. \\\\",
      rows, "Behaviour after observations (all questions, 16 samples). Columns give mean turns; \\% of trajectories with an SQL error; "
      "\\% of those solved; \\% of failed queries re-issued unchanged; \\% of empty results followed directly by an answer.",
      "tab:behaviour", star=True))

# --- decision sensitivity ---------------------------------------------------------------------------
D = J(f"{R}/decision_evidence.json")
rows = []
for stage in ("SFT", "GRPO"):
    names = [k.split("/", 2)[2] for k in D if k.startswith(f"{stage}/bird_dev/")]
    rows.append(f"\\multicolumn{{9}}{{@{{}}l}}{{\\emph{{after {stage}}}}} \\\\")
    for n in names:
        cells = []
        for ds in ("spider_dev", "bird_dev"):
            r = D.get(f"{stage}/{ds}/{n}")
            ds_ = (f"{r['dS_vs_ActionSFT']:+.3f} [{r['dS_ci95'][0]:+.3f}, {r['dS_ci95'][1]:+.3f}]"
                   if "dS_vs_ActionSFT" in r else "ref.")
            cells += [f"{r['p_submit_match']:.3f}", f"{r['p_submit_mismatch']:.3f}", f"{r['S']:.3f}", ds_]
        lab = n.replace("ActionSFT-LR", "\\armC{}").replace("ActionSFT", "\\armA{}").replace("ActObs", "\\armB{}") \
               .replace("Outcome-Aux", "\\armE{}").replace("Outcome-in-Action", "\\armF{}") \
               .replace("Contract-Placebo", "\\armGp{}").replace("Contract", "\\armG{}")
        rows.append(lab + " & " + " & ".join(cells) + " \\\\")
write("decisions.tex", table("@{}lcccccccc@{}",
      "& \\multicolumn{4}{c}{Spider-dev} & \\multicolumn{4}{c}{BIRD-dev} \\\\\n\\cmidrule(lr){2-5}\\cmidrule(lr){6-9}\n"
      "Model & P(sub$\\mid$m) & P(sub$\\mid$mm) & $S$ & $\\Delta S$ vs.\\ \\armA{} & P(sub$\\mid$m) & P(sub$\\mid$mm) & $S$ & $\\Delta S$ vs.\\ \\armA{} \\\\",
      rows, "Decision sensitivity. P(sub$\\mid$m) and P(sub$\\mid$mm) are the probabilities of submitting right after an observation whose "
      "coarse shape matches or does not match the gold result. $S$ is their difference, and $\\Delta S$ the difference from \\armA{} at the same "
      "stage, with 95\\% question-cluster bootstrap CIs (2,000 resamples).", "tab:decisions", star=True))

# --- selection (all runs) ---------------------------------------------------------------------------
S = J(f"{R}/paper_analyses.json")["selection"]
SR = [("\\armA{} SFT", "sftA"), ("\\armB{} SFT", "sftB"), ("\\armC{} SFT", "sftC"), ("\\armF{} SFT", "sftE2"),
      ("\\armG{} SFT", "sftE3"), ("\\armGp{} SFT", "sftE3p"), ("\\armA{} GRPO s1", "grpoA1"), ("\\armA{} GRPO s2", "grpoA2"),
      ("\\armB{} GRPO", "grpoB1"), ("\\armC{} GRPO", "grpoC1"), ("\\armG{} GRPO", "grpoE3")]
KEYS = ["random", "free_rule", "vote", "e3_contract", "e3_contract_vote", "oracle_shape"]
rows = []
for lab, k in SR:
    cells = []
    for ds in ("spider_dev", "bird_dev"):
        r = S.get(f"{k}/{ds}")
        cells += [f(r[x]) for x in KEYS] if r else [PEND] * len(KEYS)
    rows.append(lab + " & " + " & ".join(cells) + " \\\\")
hdr = "rand. & rule & vote & contr. & c.+v. & oracle"
write("selection_all.tex", table("@{}l" + "c" * 12 + "@{}",
      f"& \\multicolumn{{6}}{{c}}{{Spider-dev}} & \\multicolumn{{6}}{{c}}{{BIRD-dev}} \\\\\n\\cmidrule(lr){{2-7}}\\cmidrule(lr){{8-13}}\n"
      f"Samples from & {hdr} & {hdr} \\\\", rows,
      "Selecting one of 16 samples per question by: a random pick (= pass@1); the validity rule (no error, non-empty); "
      "execution majority vote; \\armG's modal contract; contract then vote; and the gold coarse shape (oracle). "
      "BIRD uses all 1,532 primary questions.", "tab:selectall", star=True))

# --- kill-test categories -----------------------------------------------------------------------------
K = J(f"{R}/killtest_shape_v3.json")
KR = [("\\armA{} SFT", "sftA"), ("\\armF{} SFT", "sftE2"), ("\\armA{} GRPO s1", "grpoA1"), ("\\armB{} GRPO s1", "grpoB1")]
CAT = [("wrong/error", "error"), ("wrong/noanswer", "no answer"), ("wrong/empty", "empty"), ("wrong/cols_differ", "\\#cols differ"),
       ("wrong/rowclass_differ", "1 vs.\\ many rows"), ("wrong/rowcount_differ", "row count differs"),
       ("wrong/same_exact_shape", "same shape")]
rows = []
for lab, k in KR:
    cells = []
    for ds in ("spider_dev", "bird_dev"):
        r = K.get(f"{k}/{ds}", {})
        cells += [f(100 * r.get(c, 0.0)) for c, _ in CAT]
    rows.append(lab + " & " + " & ".join(cells) + " \\\\")
hdr = " & ".join(x for _, x in CAT)
write("wrong_answers.tex", table("@{}l" + "c" * 14 + "@{}",
      f"& \\multicolumn{{7}}{{c}}{{Spider-dev}} & \\multicolumn{{7}}{{c}}{{BIRD-dev}} \\\\\n\\cmidrule(lr){{2-8}}\\cmidrule(lr){{9-15}}\n"
      f"Model & {hdr} & {hdr} \\\\", rows,
      "Wrong final answers by how their result differs from the gold result (\\% of wrong answers). Only ``same shape'' errors are invisible to any shape check.",
      "tab:wrong", star=True, size="\\scriptsize"))

# --- construction statistics + E2 confusion -------------------------------------------------------------
E3R = J(os.path.join(ROOT, "data", "sft", "e3_report.json"))
VR = J(os.path.join(ROOT, "data", "sft", "variants_report.json"))["train"]
g = E3R["gold_shape"]
rows = [f"gold shape of the 6,294 SFT questions & one row {g['one']:,}; several rows {g['many']:,}; no rows {g['empty']:,}; failed {g['fail']} \\\\",
        f"\\armG{{}} lines inserted & contracts {E3R['e3']['contract']:,}; checks: match {E3R['e3']['check_match']:,}, mismatch {E3R['e3']['check_mismatch']:,} \\\\",
        f"\\armGp{{}} lines inserted & contracts {E3R['e3p']['contract']:,}; checks: match {E3R['e3p']['check_match']:,}, mismatch {E3R['e3p']['check_mismatch']:,} \\\\",
        f"placebo contract equals gold shape & {100 * E3R['placebo_same_as_gold']:.1f}\\% of questions \\\\",
        f"outcome lines (\\armE{{}}/\\armF{{}}) & rows {VR['e1_kind/rows']:,}; no rows {VR['e1_kind/empty']:,}; error {VR['e1_kind/error']:,}; "
        f"timeout {VR['e1_kind/timeout']}; skipped invalid-action turns {VR['e2_noinsert/invalid']:,} \\\\"]
write("construction.tex", table("@{}p{0.34\\linewidth}p{0.62\\linewidth}@{}", "Quantity & Value \\\\", rows,
      "Statistics of the inserted lines in the 24,043 SFT trajectories.", "tab:construction", fit=False))
C = J(f"{R}/paper_analyses.json")["e2_confusion"]
tot = {}
for v in C.values():
    for kk, n in v.items():
        tot[kk] = tot.get(kk, 0) + n
kinds = ["rows", "empty", "error", "timeout"]
rows = [f"predicts {p} & " + " & ".join(f"{tot.get(f'{p}->{a}', 0):,}" for a in kinds) + " \\\\" for p in kinds]
write("e2_confusion.tex", table("@{}lcccc@{}", "& \\multicolumn{4}{c}{actual outcome} \\\\\n\\cmidrule(lr){2-5}\n& rows & no rows & error & timeout \\\\",
      rows, "\\armF{}'s predicted vs.\\ actual outcome of each SQL call in its own pre-RL rollouts (Spider-dev and BIRD-dev).", "tab:e2conf"))
