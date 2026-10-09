"""CPU analyses requested by the 2026-10-08 internal review of PAPER_PLAN.md (all on existing rollouts).

  --part exec      execute every sample's final <solution> SQL once per run and cache
                   (shape, result hash) in results/exec_cache/<run>.<ds>.jsonl; gold coarse shapes go
                   to results/gold_shapes_dev.json
  --part analyse   from the caches and rollouts:
    behaviour      P(submit right after an observation | observed coarse shape == / != gold shape),
                   overall and for the first vs later observations (is "revise on mismatch" already
                   in A, i.e. teacher imitation?)
    selection      per run, accuracy of picking one of the 16 samples by: random (= pass@1), free rule
                   (non-empty, no error), execution majority vote, E3's modal contract (+ vote), and the
                   gold-shape oracle (upper bound)
    e2_confusion   predicted vs actual outcome kind in E2's own rollouts
    pooled         PREREG primary outcome: pass@16 (and pass@1) pooled over Spider-dev + BIRD-dev primary
                   questions, paired bootstrap (10k) between runs; A seeds also averaged per question
    question_only  naive-Bayes shape classifier from the question text alone (trained on the SFT pool's
                   gold shapes), accuracy on the dev sets, vs E3's contracts
"""
import argparse
import collections
import glob
import hashlib
import json
import math
import os
import re
import sys
from multiprocessing import Pool

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from e3_readouts import CONTRACT, coarse, obs_coarse, parse_contract  # noqa: E402
from shape_killtest import DB_DIR, SOL  # noqa: E402
from skyrl_gym.envs.sql.utils import execute_sql_wrapper_single  # noqa: E402

R = os.path.join(ROOT, "results")
DS = {"spider_dev": "spider", "bird_dev": "bird_dev"}  # parquet name -> data_source in rollouts
RUNS = {"sftA": "pre_A", "sftB": "pre_B", "sftC": "pre_C", "sftE2": "pre_E2", "sftE3": "pre_E3",
        "sftE3p": "pre_E3p", "grpoA1": "grpo_A_s1_1508971", "grpoA2": "grpo_A_s2_1510961",
        "grpoB1": "grpo_B_s1_1510918", "grpoC1": "grpo_C_s1_1510955"}
for name, pat in (("grpoB2", "grpo_B_s2_15109*"), ("grpoE2", "grpo_E2_s1_*"), ("grpoE3", "grpo_E3_s1_*")):
    c = [d for d in sorted(glob.glob(f"{R}/eval/{pat}")) if glob.glob(f"{d}/*.jsonl")]
    if c:
        RUNS[name] = os.path.basename(c[-1])
RUNS = {k: v for k, v in RUNS.items() if glob.glob(f"{R}/eval/{v}/*.jsonl")}


def run_sql(job):
    db, sql, timeout = job
    if sql is None:
        return ["noanswer"], None
    res = execute_sql_wrapper_single(db, sql, timeout, "")
    rows, ok = res[2], res[3]
    if not ok:
        return ["error"], None
    rows = set(rows)
    h = hashlib.md5(repr(sorted(map(repr, rows))).encode()).hexdigest()
    return ["ok", len(rows), len(next(iter(rows))) if rows else 0], h


def dev_tables(ds):
    df = pd.read_parquet(f"{ROOT}/data/v1/{ds}.parquet")
    dbs = {i: f"{ROOT}/data/omnisql/data/{DB_DIR[r['data']]}/{r['db_id']}/{r['db_id']}.sqlite" for i, r in df.iterrows()}
    return df, dbs


def rollouts(run, ds):
    for f in sorted(glob.glob(f"{R}/eval/{RUNS[run]}/{ds}.shard*.jsonl")):
        for line in open(f):
            yield json.loads(line)


def final_sql(r):
    acts = [s["text"] for s in r["segments"] if s["type"] == "act"]
    m = SOL.findall(acts[-1]) if acts else []
    return m[-1].strip() if m else None


def part_exec(procs, runs):
    os.makedirs(f"{R}/exec_cache", exist_ok=True)
    # Gold results are executed once and then frozen: re-running them under a 30 s timeout on a loaded node
    # silently dropped 17 BIRD questions from the primary set (2026-10-09), so later runs reuse the cache.
    gold_path = f"{R}/gold_shapes_dev.json"
    gold = json.load(open(gold_path)) if os.path.exists(gold_path) else {}
    with Pool(procs) as pool:
        for ds in DS:
            df, dbs = dev_tables(ds)
            if ds not in gold:
                g = pool.map(run_sql, [(dbs[i], r["reward_spec"]["ground_truth"], 30) for i, r in df.iterrows()], chunksize=8)
                gold[ds] = {str(i): {"shape": s, "hash": h} for i, (s, h) in zip(df.index, g)}
            for run in runs:
                out = f"{R}/exec_cache/{run}.{ds}.jsonl"
                if os.path.exists(out):
                    continue
                recs = [(r["qid"], r["sample"], float(r["reward"] == 1.0), final_sql(r)) for r in rollouts(run, ds)]
                uniq = list({(dbs[q], s) for q, _, _, s in recs})
                res = dict(zip(uniq, pool.map(run_sql, [(d, s, 10) for d, s in uniq], chunksize=8)))
                with open(out + ".tmp", "w") as f:
                    for q, smp, rew, s in recs:
                        shp, h = res[(dbs[q], s)]
                        f.write(json.dumps({"qid": q, "sample": smp, "reward": rew, "shape": shp, "hash": h}) + "\n")
                os.replace(out + ".tmp", out)
                print("cached", run, ds, len(recs), flush=True)
    json.dump(gold, open(f"{R}/gold_shapes_dev.json", "w"))


def gold_coarse(gold, ds):
    out = {}
    for q, g in gold[ds].items():
        s = g["shape"]
        if s[0] == "ok" and s[1] > 0:  # primary set
            out[int(q)] = coarse(tuple(s))
    return out


def e3_contracts(ds, run="sftE3"):
    """Modal contract per question in a run's rollouts (sftE3 by default; a placebo run gives its own contracts)."""
    votes = collections.defaultdict(collections.Counter)
    if run not in RUNS:
        return {}
    for r in rollouts(run, ds):
        acts = [s for s in r["segments"] if s["type"] == "act"]
        c = parse_contract(CONTRACT.search(acts[0]["text"])) if acts else None
        if c is not None:
            votes[r["qid"]][c] += 1
    return {q: v.most_common(1)[0][0] for q, v in votes.items()}


def behaviour(run, ds, gc):
    st = collections.Counter()
    for r in rollouts(run, ds):
        q = r["qid"]
        if q not in gc:
            continue
        segs = r["segments"]
        k_obs = 0
        for i in range(1, len(segs) - 1):
            if segs[i]["type"] != "obs" or segs[i + 1]["type"] != "act":
                continue
            got = obs_coarse(segs[i]["text"])
            if got is None:
                continue
            k_obs += 1
            pos = "first" if k_obs == 1 else "later"
            key = "match" if got == gc[q] else "mismatch"
            sub = "<solution>" in segs[i + 1]["text"]
            for p in ("all", pos):
                st[f"{p}/{key}/n"] += 1
                st[f"{p}/{key}/submit"] += sub
    return {f"{p}/{k}": {"n": st[f"{p}/{k}/n"], "p_submit": round(st[f"{p}/{k}/submit"] / max(1, st[f"{p}/{k}/n"]), 4)}
            for p in ("all", "first", "later") for k in ("match", "mismatch")}


def pick(xs, flt):
    m = [x for x in xs if flt(x)]
    return m if m else None


def vote_acc(xs):
    """Execution-majority vote among xs (largest result cluster; ties averaged)."""
    groups = collections.defaultdict(list)
    for x in xs:
        groups[x["hash"] or ("none", x["shape"][0])].append(x["reward"])
    top = max(len(v) for v in groups.values())
    tied = [v for v in groups.values() if len(v) == top]
    return sum(sum(v) / len(v) for v in tied) / len(tied)


def selection(run, ds, gc, contracts):
    by_q = collections.defaultdict(list)
    for line in open(f"{R}/exec_cache/{run}.{ds}.jsonl"):
        x = json.loads(line)
        if x["qid"] in gc:
            by_q[x["qid"]].append(x)
    valid = lambda x: x["shape"][0] == "ok" and x["shape"][1] > 0  # noqa: E731
    acc = collections.Counter()
    for q, xs in by_q.items():
        mean = lambda ys: sum(y["reward"] for y in ys) / len(ys)  # noqa: E731
        acc["random"] += mean(xs)
        v = pick(xs, valid) or xs
        acc["free_rule"] += mean(v)
        acc["vote"] += vote_acc(v)
        c = contracts.get(q)
        cm = pick(v, lambda x: c is not None and coarse(tuple(x["shape"])) == c)
        acc["e3_contract"] += mean(cm or v)
        acc["e3_contract_vote"] += vote_acc(cm or v)
        om = pick(xs, lambda x: x["shape"][0] == "ok" and coarse(tuple(x["shape"])) == gc[q])
        acc["oracle_shape"] += mean(om or xs)
    n = len(by_q)
    return {k: round(100 * v / n, 2) for k, v in acc.items()} | {"questions": n}


def e2_confusion(run, ds):
    pred_re = re.compile(r"Expected result: (.*?)\.\s*\n?<sql>", re.S)
    conf = collections.Counter()
    for r in rollouts(run, ds):
        s = r["segments"]
        for i, seg in enumerate(s[:-1]):
            if seg["type"] != "act" or s[i + 1]["type"] != "obs":
                continue
            m = pred_re.search(seg["text"])
            got = obs_coarse(s[i + 1]["text"])
            if m is None or got is None:
                continue
            p = m.group(1).strip()
            pk = "error" if p.startswith("error") else "empty" if p.startswith("no rows") else \
                 "timeout" if p.startswith("timeout") else "rows"
            ak = got[0] if got[0] in ("error", "empty", "timeout") else "rows"
            conf[f"{pk}->{ak}"] += 1
    return dict(conf)


def pooled(gc_all):
    """Per-question pass@k over both datasets' primary questions, paired bootstrap between runs."""
    import passk
    per = {}
    for run in [r for r in RUNS if r.startswith("grpo")]:
        per_q, _ = passk.load(f"{R}/eval/{RUNS[run]}/*.jsonl")
        per[run] = {k: {key: passk.pass_at_k(len(v), sum(x == 1.0 for x in v), k) for key, v in per_q.items()
                        if key in gc_all and len(v) >= k} for k in (1, 16)}
    keys = sorted(gc_all)
    if "grpoA1" in per and "grpoA2" in per:
        per["grpoAmean"] = {k: {q: (per["grpoA1"][k][q] + per["grpoA2"][k][q]) / 2 for q in keys} for k in (1, 16)}
    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(keys), size=(10000, len(keys)))
    out = {"runs": {r: {f"p{k}": round(100 * np.mean([per[r][k][q] for q in keys]), 2) for k in (1, 16)} for r in per}}
    pairs = [("grpoA1", "grpoA2"), ("grpoAmean", "grpoB1"), ("grpoA1", "grpoB1"), ("grpoA2", "grpoB1"),
             ("grpoAmean", "grpoC1"), ("grpoB1", "grpoC1"), ("grpoAmean", "grpoB2"), ("grpoAmean", "grpoE3"),
             ("grpoAmean", "grpoE2")]
    out["pairs"] = {}
    for a, b in pairs:
        if a in per and b in per:
            for k in (1, 16):
                d = np.array([per[b][k][q] - per[a][k][q] for q in keys])
                boots = d[idx].mean(1)
                out["pairs"][f"{b}-{a}/p{k}"] = {"diff_pp": round(100 * d.mean(), 2),
                                                 "ci95": [round(100 * np.percentile(boots, 2.5), 2),
                                                          round(100 * np.percentile(boots, 97.5), 2)]}
    out["n_questions"] = len(keys)
    return out


def tokens(text):
    w = re.findall(r"[a-z]+|\d+", text.lower())
    return w + [a + "_" + b for a, b in zip(w, w[1:])]


def question_only(gc_by_ds):
    """Multinomial naive Bayes (add-one) on question words + bigrams -> gold coarse shape."""
    pool = pd.read_parquet(f"{ROOT}/data/v1/sft_pool.parquet")
    shapes = json.load(open(f"{ROOT}/data/sft/sft_pool_gold_shape.json"))
    lab = lambda s: "empty" if s[0] == "empty" else f"{s[0]}-{s[1]}"  # noqa: E731
    cls_count, word_count, tot = collections.Counter(), collections.defaultdict(collections.Counter), collections.Counter()
    vocab = set()
    for i, r in pool.iterrows():
        s = shapes.get(str(i))
        if not s:
            continue
        y = lab(s)
        cls_count[y] += 1
        for t in tokens(r["question"]):
            word_count[y][t] += 1
            tot[y] += 1
            vocab.add(t)
    n = sum(cls_count.values())
    V = len(vocab)

    def predict(text):
        best, arg = -1e18, None
        for y in cls_count:
            lp = math.log(cls_count[y] / n) + sum(math.log((word_count[y][t] + 1) / (tot[y] + V)) for t in tokens(text))
            if lp > best:
                best, arg = lp, y
        return arg

    out = {}
    for ds, gc in gc_by_ds.items():
        df = pd.read_parquet(f"{ROOT}/data/v1/{ds}.parquet")
        ok = sum(predict(df.loc[q, "question"]) == lab(list(c)) for q, c in gc.items())
        out[ds] = round(ok / len(gc), 4)
    return out


def part_analyse(out_path):
    gold = json.load(open(f"{R}/gold_shapes_dev.json"))
    rep = {"runs": RUNS, "behaviour": {}, "selection": {}, "selection_own_contract": {}, "e2_confusion": {}}
    gc_by_ds, gc_all = {}, set()
    for ds, src in DS.items():
        gc = gold_coarse(gold, ds)
        gc_by_ds[ds] = gc
        gc_all |= {(src, q) for q in gc}
        contracts = e3_contracts(ds)
        rep[f"e3_contract_acc_modal/{ds}"] = round(sum(contracts.get(q) == c for q, c in gc.items()) / len(gc), 4)
        for run in RUNS:
            rep["behaviour"][f"{run}/{ds}"] = behaviour(run, ds, gc)
            if os.path.exists(f"{R}/exec_cache/{run}.{ds}.jsonl"):
                rep["selection"][f"{run}/{ds}"] = selection(run, ds, gc, contracts)
        # each contract-writing run selected by its own modal contract (for the placebo: its deranged contracts)
        for run in ("sftE3", "sftE3p", "grpoE3"):
            if run in RUNS and os.path.exists(f"{R}/exec_cache/{run}.{ds}.jsonl"):
                rep["selection_own_contract"][f"{run}/{ds}"] = selection(run, ds, gc, e3_contracts(ds, run))
        for run in ("sftE2", "grpoE2"):
            if run in RUNS:
                rep["e2_confusion"][f"{run}/{ds}"] = e2_confusion(run, ds)
        print("done", ds, flush=True)
    rep["pooled_primary"] = pooled(gc_all)
    rep["question_only_shape_acc"] = question_only(gc_by_ds)
    json.dump(rep, open(out_path, "w"), indent=1)
    print(json.dumps({k: v for k, v in rep.items() if k in ("pooled_primary", "question_only_shape_acc")}, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=["exec", "analyse"], required=True)
    ap.add_argument("--procs", type=int, default=8)
    ap.add_argument("--runs", nargs="*", default=None)
    ap.add_argument("--out", default=f"{R}/paper_analyses.json")
    a = ap.parse_args()
    if a.part == "exec":
        part_exec(a.procs, a.runs or list(RUNS))
    else:
        part_analyse(a.out)
