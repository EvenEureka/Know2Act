"""Readouts for the E3 answer-contract arms (PREREG Addendum 7, predictions 1, 2, 4 and 6).

For each eval run (rollouts on Spider-dev / BIRD-dev, primary set: gold non-empty and executable):
  contract_acc     the contract a rollout writes ("Expected answer: ...") equals the gold coarse shape;
                   compared with always writing the most frequent dev-set gold shape (majority_acc)
  check_correct    a written "Check: got X; this (does not) match(es)" line agrees with the real
                   observation and the rollout's own contract
  submit_after_*   P(next action contains <solution>) after a check saying "matches" / "does not match"
  selection        among the 16 samples of a question, pick uniformly among those whose final result
                   matches the question's majority contract (all samples if none do); compared with a
                   random pick (pass@1) and with the free rule (non-empty, no error)
"""
import argparse
import collections
import glob
import json
import os
import re
import sys
from multiprocessing import Pool

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from shape_killtest import DB_DIR, SOL, shape  # noqa: E402

CONTRACT = re.compile(r"Expected answer: (no rows|1 row, (\d+) columns?|several rows, (\d+) columns?)\.")
CHECK = re.compile(r"Check: got ([^;]+); this (matches|does not match) the expected answer\.")
OBS_SHAPE = re.compile(r"(no rows|error|timeout|1 row, (\d+) columns?|several rows, (\d+) columns?)")


def parse_contract(m):
    if m is None:
        return None
    if m.group(1) == "no rows":
        return ("empty",)
    return ("one", int(m.group(2))) if m.group(2) else ("many", int(m.group(3)))


def coarse(s):
    """Exact shape from shape_killtest -> contract-style coarse shape."""
    if s[0] != "ok":
        return (s[0],)
    if s[1] == 0:
        return ("empty",)
    return ("one" if s[1] == 1 else "many", s[2])


def obs_coarse(text):
    p = text.split("<observation>", 1)[-1]
    p = p[1:] if p.startswith(">") else p
    if p.startswith("Your previous action is invalid"):
        return None
    if p.startswith("Error executing SQL"):
        return ("error",)
    if p.startswith("SQL Timeout"):
        return ("timeout",)
    if p.startswith("Empty DataFrame"):
        return ("empty",)
    body = p.split("\n<reminder>")[0]
    trunc = body.startswith("Truncated to 50 lines")
    if trunc:
        body = body.split(": ", 1)[1]
    lines = body.rstrip("\n").split("\n")
    return ("many" if trunc or len(lines) > 2 else "one", len(lines[0].split()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help="NAME=results/eval/<dir>")
    ap.add_argument("--procs", type=int, default=8)
    ap.add_argument("--no_selection", action="store_true")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rep = {}
    with Pool(a.procs) as pool:
        for ds in ("spider_dev", "bird_dev"):
            df = pd.read_parquet(f"{ROOT}/data/v1/{ds}.parquet")
            dbs = {i: f"{ROOT}/data/omnisql/data/{DB_DIR[r['data']]}/{r['db_id']}/{r['db_id']}.sqlite" for i, r in df.iterrows()}
            gsql = {i: r["reward_spec"]["ground_truth"] for i, r in df.iterrows()}
            gold = dict(zip(gsql, pool.map(shape, [(dbs[i], gsql[i], 30) for i in gsql], chunksize=8)))
            prim = {q for q, g in gold.items() if g[0] == "ok" and g[1] > 0}
            gc = {q: coarse(gold[q]) for q in prim}
            maj_shape, maj_n = collections.Counter(gc.values()).most_common(1)[0]
            for spec in a.runs:
                name, d = spec.split("=", 1)
                st = collections.Counter()
                samples = collections.defaultdict(list)
                for f in sorted(glob.glob(f"{ROOT}/{d}/{ds}.shard*.jsonl")):
                    for line in open(f):
                        r = json.loads(line)
                        q = r["qid"]
                        if q not in prim:
                            continue
                        segs = r["segments"]
                        acts = [s for s in segs if s["type"] == "act"]
                        c = parse_contract(CONTRACT.search(acts[0]["text"])) if acts else None
                        st["rollouts"] += 1
                        st["has_contract"] += c is not None
                        st["contract_correct"] += c == gc[q]
                        for i, s in enumerate(segs):
                            if s["type"] != "act" or i == 0 or segs[i - 1]["type"] != "obs":
                                continue
                            m = CHECK.search(s["text"])
                            if not m:
                                continue
                            said_match = m.group(2) == "matches"
                            real = obs_coarse(segs[i - 1]["text"])
                            st["checks"] += 1
                            if c is not None and real is not None:
                                st["check_correct"] += said_match == (real == c)
                            key = "match" if said_match else "mismatch"
                            st[f"n_after_{key}"] += 1
                            st[f"submit_after_{key}"] += "<solution>" in s["text"]
                        m = SOL.findall(acts[-1]["text"]) if acts else []
                        samples[q].append((float(r["reward"] == 1.0), c, m[-1].strip() if m else None))
                out = {"rollouts": st["rollouts"], "has_contract": round(st["has_contract"] / st["rollouts"], 4),
                       "contract_acc": round(st["contract_correct"] / st["rollouts"], 4),
                       "majority_acc": round(maj_n / len(prim), 4), "majority_shape": list(maj_shape),
                       "check_correct": round(st["check_correct"] / max(1, st["checks"]), 4)}
                for key in ("match", "mismatch"):
                    out[f"submit_after_{key}"] = round(st[f"submit_after_{key}"] / max(1, st[f"n_after_{key}"]), 4)
                    out[f"n_after_{key}"] = st[f"n_after_{key}"]
                if not a.no_selection:
                    uniq = list({(dbs[q], s) for q, xs in samples.items() for _, _, s in xs})
                    res = dict(zip(uniq, pool.map(shape, uniq, chunksize=8)))
                    acc = rule = own = 0.0
                    for q, xs in samples.items():
                        acc += sum(x[0] for x in xs) / len(xs)
                        cs = collections.Counter(x[1] for x in xs if x[1] is not None)
                        pred = cs.most_common(1)[0][0] if cs else None
                        for flt, tgt in ((lambda s: s[0] == "ok" and s[1] > 0, "rule"),
                                         (lambda s: pred is not None and coarse(s) == pred, "own")):
                            m = [x[0] for x in xs if flt(res[(dbs[q], x[2])])] or [x[0] for x in xs]
                            if tgt == "rule":
                                rule += sum(m) / len(m)
                            else:
                                own += sum(m) / len(m)
                    n = len(samples)
                    out |= {"pass@1": round(100 * acc / n, 2), "rule_select": round(100 * rule / n, 2),
                            "own_contract_select": round(100 * own / n, 2)}
                rep[f"{name}/{ds}"] = out
                print(name, ds, json.dumps(out), flush=True)
                json.dump(rep, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
