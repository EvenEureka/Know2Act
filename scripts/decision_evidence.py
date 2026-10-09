"""Evidence for "learned but not used": do any of the SFT variants change the agent's decisions?

Decision probe (no model calls, from eval rollouts): after every SQL observation that is followed by
another action, did the agent submit (<solution>) right away? Each observation is labelled by whether
its coarse result shape equals the GOLD result's coarse shape (empty / 1 row / several rows + #cols).
  S = P(submit | shape matches gold) - P(submit | shape does not match gold)   ("decision sensitivity")
A policy that uses answer knowledge should submit when the result looks right and keep querying when
it does not, i.e. have a larger S. We compare every arm with ActionSFT (A) using a question-cluster
bootstrap (2,000 resamples) and report the data-order noise reference (A, seed-1 data order).
Also reported: P(submit | mismatch), the decision that matters most, with the same CIs.
Output: results/decision_evidence.json
"""
import collections
import glob
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from e3_readouts import coarse, obs_coarse  # noqa: E402

R = os.path.join(ROOT, "results")
RUNS = {"SFT": {"ActionSFT": "pre_A", "ActionSFT (new data order)": "pre_Aseed1", "ActObs": "pre_B",
                "ActionSFT-LR": "pre_C", "Outcome-Aux": "pre_E1", "Outcome-in-Action": "pre_E2",
                "Contract": "pre_E3", "Contract-Placebo": "pre_E3p"},
        "GRPO": {"ActionSFT": "grpo_A_s1_1508971", "ActionSFT (seed 2)": "grpo_A_s2_1510961",
                 "ActObs": "grpo_B_s1_1510918", "ActionSFT-LR": "grpo_C_s1_1510955",
                 "Contract": "grpo_E3_s1_1516402"}}
for name, pat in (("ActObs (seed 2)", "grpo_B_s2_15109*"), ("Outcome-in-Action", "grpo_E2_s1_*")):
    c = [d for d in sorted(glob.glob(f"{R}/eval/{pat}")) if glob.glob(f"{d}/*.jsonl")]
    if c:
        RUNS["GRPO"][name] = os.path.basename(c[-1])
gold = json.load(open(f"{R}/gold_shapes_dev.json"))


def gold_coarse(ds):
    return {int(q): coarse(tuple(g["shape"])) for q, g in gold[ds].items()
            if g["shape"][0] == "ok" and g["shape"][1] > 0}


def counts(run, ds, gc):
    """per question: [n_match, sub_match, n_mis, sub_mis]"""
    c = collections.defaultdict(lambda: np.zeros(4))
    for f in sorted(glob.glob(f"{R}/eval/{run}/{ds}.shard*.jsonl")):
        for line in open(f):
            r = json.loads(line)
            q = r["qid"]
            if q not in gc:
                continue
            s = r["segments"]
            for i in range(1, len(s) - 1):
                if s[i]["type"] != "obs" or s[i + 1]["type"] != "act":
                    continue
                got = obs_coarse(s[i]["text"])
                if got is None:
                    continue
                sub = float("<solution>" in s[i + 1]["text"])
                if got == gc[q]:
                    c[q] += (1, sub, 0, 0)
                else:
                    c[q] += (0, 0, 1, sub)
    return c


def stats(m):
    """m: [Q, 4] summed counts -> (S, p_sub_match, p_sub_mismatch)"""
    t = m.sum(0)
    pm, pmm = t[1] / max(t[0], 1), t[3] / max(t[2], 1)
    return pm - pmm, pm, pmm


rng = np.random.default_rng(0)
out = {}
for stage, runs in RUNS.items():
    for ds in ("spider_dev", "bird_dev"):
        gc = gold_coarse(ds)
        qs = sorted(gc)
        C = {name: counts(run, ds, gc) for name, run in runs.items()}
        M = {name: np.array([C[name][q] for q in qs]) for name in runs}
        idx = rng.integers(0, len(qs), size=(2000, len(qs)))
        ref = M["ActionSFT"]
        for name, m in M.items():
            S, pm, pmm = stats(m)
            row = {"S": round(S, 4), "p_submit_match": round(pm, 4), "p_submit_mismatch": round(pmm, 4),
                   "n_match": int(m[:, 0].sum()), "n_mismatch": int(m[:, 2].sum())}
            if name != "ActionSFT":
                bS, bmm = [], []
                for ix in idx:
                    s1, _, mm1 = stats(m[ix]); s0, _, mm0 = stats(ref[ix])
                    bS.append(s1 - s0); bmm.append(mm1 - mm0)
                row["dS_vs_ActionSFT"] = round(S - stats(ref)[0], 4)
                row["dS_ci95"] = [round(float(np.percentile(bS, q)), 4) for q in (2.5, 97.5)]
                row["dPmis_vs_ActionSFT"] = round(pmm - stats(ref)[2], 4)
                row["dPmis_ci95"] = [round(float(np.percentile(bmm, q)), 4) for q in (2.5, 97.5)]
            out[f"{stage}/{ds}/{name}"] = row
            print(stage, ds, f"{name:28s}", row, flush=True)
json.dump(out, open(f"{R}/decision_evidence.json", "w"), indent=1)
