"""Read-outs of the decision-point intervention (scripts/contract_intervention.py).

For each run (one Contract model x one dataset; shards merged) and each decision point we have K
continuations per condition. Everything is paired on the decision point and bootstrapped over
questions (cluster bootstrap, 2,000 resamples).

  flip      S_int = P(submit | contract matches the observed result) - P(submit | it does not), with the
            observation held fixed: for every decision point one of {orig, flip} matches and the other
            does not. This is the causal counterpart of the decision sensitivity S of
            scripts/decision_evidence.py (there the result shape varies; here only the contract does).
            Also split by the rollout's own state (own contract matched / mismatched the result).
  natural   the shape-driven asymmetry on the same continuations: P(submit | own_match) - P(submit |
            not own_match) among the orig continuations (not paired; for scale).
  fidelity  under flip, does the regenerated check line follow the new contract?
  remove    P(submit) and accuracy without any contract/check lines, vs orig; share still writing a check.
  gold      P(submit) and accuracy with the gold contract where the own contract was wrong, vs orig.
  accuracy  final reward per condition, paired differences vs orig.
"""
import argparse
import collections
import glob
import json
import os

import numpy as np


def load(run_dir):
    groups = collections.defaultdict(lambda: collections.defaultdict(list))
    for f in sorted(glob.glob(os.path.join(run_dir, "*.jsonl"))):
        for line in open(f):
            x = json.loads(line)
            groups[(x["qid"], x["sample"], x["dp"])][x["cond"]].append(x)
    return groups


def mean(xs, key):
    return float(np.mean([float(x[key]) for x in xs]))


def boot(per_q, fn, n=2000, seed=0):
    """per_q: dict qid -> list of per-decision-point values (tuples); fn maps a list of values to a
    statistic. Returns (point, lo, hi) with a question-cluster bootstrap."""
    qs = sorted(per_q)
    rng = np.random.default_rng(seed)
    point = fn([v for q in qs for v in per_q[q]])
    draws = []
    for _ in range(n):
        ix = rng.integers(0, len(qs), len(qs))
        draws.append(fn([v for j in ix for v in per_q[qs[j]]]))
    return round(point, 4), round(float(np.percentile(draws, 2.5)), 4), round(float(np.percentile(draws, 97.5)), 4)


def diff_stat(vals):
    return float(np.mean([a - b for a, b in vals]))


def analyse(groups, dp_rank=1):
    out = {}
    flip, flip_by_state = collections.defaultdict(list), {True: collections.defaultdict(list), False: collections.defaultdict(list)}
    natural = {True: [], False: []}
    fid = collections.Counter()
    rem, gold, acc = collections.defaultdict(list), collections.defaultdict(list), {c: collections.defaultdict(list) for c in ("flip", "remove", "gold")}
    sub_cond = collections.defaultdict(list)
    n_dp = 0
    for (q, s, dp), conds in groups.items():
        if "orig" not in conds or conds["orig"][0]["dp_rank"] != dp_rank:
            continue
        n_dp += 1
        o = conds["orig"]
        po, ao = mean(o, "submit"), mean(o, "reward") if all(x["reward"] is not None for x in o) else float("nan")
        natural[o[0]["own_match"]].append(po)
        for c in ("flip", "remove", "gold"):
            if c in conds:
                sub_cond[c].append((mean(conds[c], "submit"), po))
                acc[c][q].append((np.mean([x["reward"] == 1.0 for x in conds[c]]), np.mean([x["reward"] == 1.0 for x in o])))
        if "flip" in conds:
            f = conds["flip"]
            match_cond, mis_cond = (o, f) if o[0]["own_match"] else (f, o)
            v = (mean(match_cond, "submit"), mean(mis_cond, "submit"))
            flip[q].append(v)
            flip_by_state[o[0]["own_match"]][q].append(v)
            for x in f:
                want = "match" if x["new_match"] else "mismatch"
                fid["n"] += 1
                fid["no_check"] += x["check_said"] is None
                fid["follows_new"] += x["check_said"] == want
                fid["follows_old"] += x["check_said"] == ("match" if x["own_match"] else "mismatch")
        if "remove" in conds:
            r = conds["remove"]
            rem[q].append((mean(r, "submit"), po))
            fid["remove_n"] += len(r)
            fid["remove_writes_check"] += sum(x["check_said"] is not None for x in r)
        if "gold" in conds:
            gold[q].append((mean(conds["gold"], "submit"), po))
    out["decision_points"] = n_dp
    out["p_submit_orig"] = round(float(np.mean([v for vs in natural.values() for v in vs])), 4)
    out["natural_asymmetry"] = {"p_submit_own_match": round(float(np.mean(natural[True])), 4) if natural[True] else None,
                                "p_submit_own_mismatch": round(float(np.mean(natural[False])), 4) if natural[False] else None,
                                "n_match": len(natural[True]), "n_mismatch": len(natural[False])}
    if flip:
        S, lo, hi = boot(flip, diff_stat)
        out["flip"] = {"S_int": S, "ci95": [lo, hi], "n": sum(len(v) for v in flip.values()),
                       "p_submit_matching_contract": round(float(np.mean([a for vs in flip.values() for a, _ in vs])), 4),
                       "p_submit_mismatching_contract": round(float(np.mean([b for vs in flip.values() for _, b in vs])), 4)}
        for state, name in ((True, "own_matched_then_flipped_to_mismatch"), (False, "own_mismatched_then_flipped_to_match")):
            if flip_by_state[state]:
                S, lo, hi = boot(flip_by_state[state], diff_stat)
                out["flip"][name] = {"S_int": S, "ci95": [lo, hi], "n": sum(len(v) for v in flip_by_state[state].values())}
        out["flip"]["check_fidelity"] = {"follows_new_contract": round(fid["follows_new"] / max(1, fid["n"]), 4),
                                         "follows_old_contract": round(fid["follows_old"] / max(1, fid["n"]), 4),
                                         "no_check_line": round(fid["no_check"] / max(1, fid["n"]), 4), "n": fid["n"]}
    for name, d in (("remove", rem), ("gold", gold)):
        if d:
            D, lo, hi = boot(d, diff_stat)
            out[name] = {"dP_submit_vs_orig": D, "ci95": [lo, hi], "n": sum(len(v) for v in d.values()),
                         "p_submit": round(float(np.mean([a for vs in d.values() for a, _ in vs])), 4)}
    # teacher-forced check sentence: P(submit | check says match) - P(submit | check says mismatch), paired
    fc = collections.defaultdict(list)
    for (q, s, dp), conds in groups.items():
        if "fcheck_orig" in conds and "fcheck_flip" in conds and conds["fcheck_orig"][0]["dp_rank"] == dp_rank:
            a, b = conds["fcheck_orig"], conds["fcheck_flip"]
            m, mm = (a, b) if a[0]["forced_check"] == "match" else (b, a)
            fc[q].append((mean(m, "submit"), mean(mm, "submit")))
    if fc:
        S, lo, hi = boot(fc, diff_stat)
        out["forced_check"] = {"S_check": S, "ci95": [lo, hi], "n": sum(len(v) for v in fc.values()),
                               "p_submit_check_match": round(float(np.mean([a for vs in fc.values() for a, _ in vs])), 4),
                               "p_submit_check_mismatch": round(float(np.mean([b for vs in fc.values() for _, b in vs])), 4)}
    if fid["remove_n"]:
        out["remove"]["writes_check_anyway"] = round(fid["remove_writes_check"] / fid["remove_n"], 4)
    out["accuracy"] = {}
    for c, d in acc.items():
        if d:
            D, lo, hi = boot(d, diff_stat)
            out["accuracy"][c] = {"acc_cond": round(100 * float(np.mean([a for vs in d.values() for a, _ in vs])), 2),
                                  "acc_orig": round(100 * float(np.mean([b for vs in d.values() for _, b in vs])), 2),
                                  "diff_pts": round(100 * D, 2), "ci95_pts": [round(100 * lo, 2), round(100 * hi, 2)]}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help="NAME=results/intervention/<dir>")
    ap.add_argument("--out", required=True)
    ap.add_argument("--smoke", action="store_true", help="assert the smoke run produced usable records")
    a = ap.parse_args()
    rep = {}
    for spec in a.runs:
        name, d = spec.split("=", 1)
        groups = load(d)
        rep[name] = analyse(groups)
        if a.smoke:
            assert rep[name]["decision_points"] >= 5, rep[name]
            assert "flip" in rep[name] and rep[name]["flip"]["n"] >= 3, rep[name]
            assert rep[name]["flip"]["check_fidelity"]["no_check_line"] < 0.5, rep[name]
            assert all(x["reward"] is not None for g in groups.values() for xs in g.values() for x in xs)
        print(name, json.dumps(rep[name], indent=1), flush=True)
    json.dump(rep, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
