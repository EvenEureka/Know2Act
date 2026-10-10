"""Paired question-bootstrap intervals for the selection gains of Table 2 (contract vs. the free rule,
contract + vote vs. vote alone), from the execution caches written by scripts/paper_analyses.py.

Per question the selection rules give an expected accuracy (mean reward of the picked set; ties in the
vote averaged), so a gain is a paired per-question difference and its interval comes from resampling
questions (10,000 draws). Output: results/selection_ci.json
"""
import collections
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from e3_readouts import coarse  # noqa: E402
from paper_analyses import R, RUNS, e3_contracts, gold_coarse, pick, vote_acc  # noqa: E402

gold = json.load(open(f"{R}/gold_shapes_dev.json"))
valid = lambda x: x["shape"][0] == "ok" and x["shape"][1] > 0  # noqa: E731


def per_question(run, ds, gc, contracts):
    by_q = collections.defaultdict(list)
    for line in open(f"{R}/exec_cache/{run}.{ds}.jsonl"):
        x = json.loads(line)
        if x["qid"] in gc:
            by_q[x["qid"]].append(x)
    rows = {}
    for q, xs in by_q.items():
        mean = lambda ys: sum(y["reward"] for y in ys) / len(ys)  # noqa: E731
        v = pick(xs, valid) or xs
        c = contracts.get(q)
        cm = pick(v, lambda x: c is not None and coarse(tuple(x["shape"])) == c)
        rows[q] = {"random": mean(xs), "free_rule": mean(v), "vote": vote_acc(v),
                   "contract": mean(cm or v), "contract_vote": vote_acc(cm or v)}
    return rows


def ci(rows, a, b, n=10000, seed=0):
    qs = sorted(rows)
    d = np.array([rows[q][a] - rows[q][b] for q in qs])
    rng = np.random.default_rng(seed)
    draws = d[rng.integers(0, len(d), (n, len(d)))].mean(1)
    return {"gain_pts": round(100 * d.mean(), 2), "ci95_pts": [round(100 * float(np.percentile(draws, 2.5)), 2),
                                                            round(100 * float(np.percentile(draws, 97.5)), 2)],
            "questions": len(qs), "share_questions_changed": round(float((d != 0).mean()), 3)}


out = {}
for ds in ("spider_dev", "bird_dev"):
    gc = gold_coarse(gold, ds)
    contracts = e3_contracts(ds, "sftE3")
    for run in ("sftA", "sftE3", "grpoA1", "grpoA2", "grpoE3"):
        if run not in RUNS:
            continue
        rows = per_question(run, ds, gc, contracts if run != "grpoE3" else e3_contracts(ds, "grpoE3"))
        out[f"{run}/{ds}"] = {"contract_vs_free_rule": ci(rows, "contract", "free_rule"),
                              "contract_vote_vs_vote": ci(rows, "contract_vote", "vote"),
                              "vote_vs_free_rule": ci(rows, "vote", "free_rule"),
                              "means": {k: round(100 * float(np.mean([r[k] for r in rows.values()])), 2) for k in rows[next(iter(rows))]}}
        print(run, ds, json.dumps(out[f"{run}/{ds}"]), flush=True)
json.dump(out, open(f"{R}/selection_ci.json", "w"), indent=1)
