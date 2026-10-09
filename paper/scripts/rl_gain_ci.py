"""RL gain per initialisation with paired bootstrap CIs (Fig. 2).

gain_k(q) = pass@k after GRPO - pass@k of the same SFT initialisation, per primary question;
the figure reports the mean over questions and a 95% bootstrap CI (2,000 resamples, seed 0).
Output: paper/data/rl_gain_ci.json
"""
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import passk  # noqa: E402

R = os.path.join(ROOT, "results")
DROP = {("spider", int(q)) for q in json.load(open(f"{R}/gold_flags/spider_dev.json"))} | \
       {("bird_dev", int(q)) for q in json.load(open(f"{R}/gold_flags/bird_dev.json"))}
PAIRS = {"ActionSFT, seed 1": ("grpo_A_s1_1508971", "pre_A"),
         "ActionSFT, seed 2": ("grpo_A_s2_1510961", "pre_A"),
         "ActObs, seed 1": ("grpo_B_s1_1510918", "pre_B"),
         "ActObs, seed 2": ("grpo_B_s2_1510964", "pre_B"),
         "ActionSFT-LR": ("grpo_C_s1_1510955", "pre_C")}
KS = (1, 4, 16)


def per_q(run):
    d, _ = passk.load(f"{R}/eval/{run}/*.jsonl")
    return {q: v for q, v in d.items() if q not in DROP}


out = {}
rng = np.random.default_rng(0)
for name, (post, pre) in PAIRS.items():
    a, b = per_q(post), per_q(pre)
    out[name] = {}
    for ds in ("spider", "bird_dev"):
        qs = sorted(q for q in a if q[0] == ds and q in b)
        idx = rng.integers(0, len(qs), size=(2000, len(qs)))
        for k in KS:
            g = np.array([passk.pass_at_k(len(a[q]), sum(x == 1.0 for x in a[q]), k) -
                          passk.pass_at_k(len(b[q]), sum(x == 1.0 for x in b[q]), k) for q in qs]) * 100
            boots = g[idx].mean(1)
            out[name][f"{ds}/p{k}"] = {"gain": round(float(g.mean()), 3),
                                       "ci95": [round(float(np.percentile(boots, 2.5)), 3),
                                                round(float(np.percentile(boots, 97.5)), 3)], "n": len(qs)}
json.dump(out, open(os.path.join(ROOT, "paper", "data", "rl_gain_ci.json"), "w"), indent=1)
print(json.dumps({k: {kk: vv["gain"] for kk, vv in v.items()} for k, v in out.items()}, indent=1))
