"""$0 kill test (2026-10-07): would knowing the SHAPE of the correct answer help a SQL agent?

For every eval sample, execute its final <solution> SQL and the gold SQL, and compare result shapes:
  coarse shape = (empty | 1 row | >1 rows, number of columns); exact shape = (distinct rows, columns)
(SkyRL's reward compares frozensets of rows, so rows are counted distinct.)
Reports, per run and dataset:
  * among wrong samples: the fraction whose shape differs from gold (catchable by a shape check),
    split into error / empty / row-count / column-count mismatches;
  * oracle-shape selection: pick uniformly among the 16 samples whose shape matches gold (all samples
    if none does) vs a uniformly random sample (= pass@1). This is an upper bound for "predict the
    answer shape, then select or retry", with a perfect shape predictor.
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
sys.path.insert(0, ROOT)
from skyrl_gym.envs.sql.utils import execute_sql_wrapper_single  # noqa: E402

DB_DIR = {"spider": "spider/database", "bird": "bird/train/train_databases"}
SOL = re.compile(r"<solution>(.*?)</solution>", re.S)


def shape(job):
    db, sql, timeout = job if len(job) == 3 else (*job, 10)
    if sql is None:
        return ("noanswer",)
    res = execute_sql_wrapper_single(db, sql, timeout, "")
    rows, ok = res[2], res[3]
    if not ok:
        return ("error",)
    rows = set(rows)
    n_cols = len(next(iter(rows))) if rows else 0
    return ("ok", len(rows), n_cols)


def coarse(s):
    if s[0] != "ok":
        return s[0]
    return ("empty" if s[1] == 0 else "1row" if s[1] == 1 else "many", s[2] if s[1] else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help="NAME=results/eval/<dir>")
    ap.add_argument("--procs", type=int, default=16)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    report = {}
    with Pool(a.procs) as pool:
        for ds, parquet in (("spider_dev", "spider_dev"), ("bird_dev", "bird_dev")):
            df = pd.read_parquet(f"{ROOT}/data/v1/{parquet}.parquet")
            dbs = {i: f"{ROOT}/data/omnisql/data/{DB_DIR[r['data']]}/{r['db_id']}/{r['db_id']}.sqlite"
                   for i, r in df.iterrows()}
            gold_sql = {i: eval(r["reward_spec"])["ground_truth"] if isinstance(r["reward_spec"], str)
                        else r["reward_spec"]["ground_truth"] for i, r in df.iterrows()}
            gold = dict(zip(gold_sql, pool.map(shape, [(dbs[i], gold_sql[i], 30) for i in gold_sql], chunksize=8)))
            for spec in a.runs:
                name, d = spec.split("=", 1)
                samples = []
                for f in sorted(glob.glob(f"{ROOT}/{d}/{ds}.shard*.jsonl")):
                    for line in open(f):
                        r = json.loads(line)
                        acts = [s["text"] for s in r["segments"] if s["type"] == "act"]
                        m = SOL.findall(acts[-1]) if acts else []
                        samples.append((r["qid"], r["reward"], m[-1].strip() if m else None))
                uniq = list({(dbs[q], s) for q, _, s in samples})
                cache = dict(zip(uniq, pool.map(shape, uniq, chunksize=8)))
                st = collections.Counter()
                by_q = collections.defaultdict(list)
                for q, rew, s in samples:
                    g, p = gold[q], cache[(dbs[q], s)]
                    if g[0] != "ok" or g[1] == 0:
                        continue  # primary set: drop failing / empty golds
                    match_c, match_e = coarse(p) == coarse(g), p == g
                    rule = p[0] == "ok" and p[1] > 0          # free rule: no error, no empty result
                    cols = p[0] == "ok" and p[1] > 0 and p[2] == g[2]  # Self-Debugging-style column spec
                    by_q[q].append((float(rew == 1.0), match_c, match_e, rule, cols))  # format failures score -1
                    st["n"] += 1
                    if rew != 1.0:
                        st["wrong"] += 1
                        if p[0] != "ok":
                            st["wrong/" + p[0]] += 1
                        elif p[1] == 0:
                            st["wrong/empty"] += 1
                        elif p[2] != g[2]:
                            st["wrong/cols_differ"] += 1
                        elif not match_c:
                            st["wrong/rowclass_differ"] += 1
                        elif not match_e:
                            st["wrong/rowcount_differ"] += 1
                        else:
                            st["wrong/same_exact_shape"] += 1
                    else:
                        st["right_but_shape_differs_exact"] += not match_e
                acc = 0.0
                sel = collections.Counter()
                for q, xs in by_q.items():
                    acc += sum(x[0] for x in xs) / len(xs)
                    for k, key in ((3, "rule_select_nonempty_noerror"), (4, "oracle_cols_select"),
                                   (1, "oracle_shape_select_coarse"), (2, "oracle_shape_select_exact")):
                        m = [x[0] for x in xs if x[k]] or [x[0] for x in xs]
                        sel[key] += sum(m) / len(m)
                nq = len(by_q)
                out = {"questions": nq, "pass@1": round(100 * acc / nq, 2)}
                out |= {k: round(100 * v / nq, 2) for k, v in sel.items()}
                w = st["wrong"]
                out |= {k: round(v / w, 4) for k, v in st.items() if k.startswith("wrong/")}
                out["wrong_frac_catchable_exact"] = round(1 - st["wrong/same_exact_shape"] / w, 4)
                out["right_but_shape_differs_exact"] = round(st["right_but_shape_differs_exact"] / max(1, st["n"] - w), 4)
                report[f"{name}/{ds}"] = out
                print(name, ds, json.dumps(out), flush=True)
                json.dump(report, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
