"""E3 "answer contract" SFT data (PREREG Addendum 7, 2026-10-08).

Every training trajectory gets two kinds of action-text insertions (class 1, action-only SFT like A):
  contract  at the start of the first action: the coarse shape of the CORRECT answer, from executing
            the gold SQL, e.g. "Expected answer: 1 row, 1 column." / "Expected answer: several rows,
            2 columns." / "Expected answer: no rows."
  check     at the start of the action after each SQL observation: the observed coarse shape and
            whether it matches the contract, e.g. "Check: got several rows, 2 columns; this does not
            match the expected answer." Errors and timeouts never match; invalid-action notices get
            no check. The check is a function of text the model can see (its contract and the
            observation), so it is a skill to compute, not a hindsight label.
E3p (placebo) is identical except that each question's contract is taken from another question
(seeded derangement over questions); its checks are computed against that placebo contract.
Reward and RL are unchanged; the contract's accuracy is never rewarded.
"""
import argparse
import collections
import json
import os
import random
import sys
from multiprocessing import Pool

import pandas as pd
from transformers import AutoTokenizer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from build_variants import ERROR, INVALID, TRUNC, runs  # noqa: E402
from skyrl_gym.envs.sql.utils import execute_sql_wrapper_single  # noqa: E402

DB_DIR = {"spider": "spider/database", "bird": "bird/train/train_databases"}


def plural(n, word):
    return f"{n} {word}{'' if n == 1 else 's'}"


def gold_shape(job):
    """Coarse shape of the gold result: ('empty',), ('one', cols), ('many', cols) or None on failure."""
    db, sql = job
    res = execute_sql_wrapper_single(db, sql, 30, "")
    rows, ok = res[2], res[3]
    if not ok:
        return None
    rows = set(rows)
    if not rows:
        return ("empty",)
    return ("one" if len(rows) == 1 else "many", len(next(iter(rows))))


def describe(shape):
    if shape[0] == "empty":
        return "no rows"
    if shape[0] in ("error", "timeout"):
        return shape[0]
    return f"{'1 row' if shape[0] == 'one' else 'several rows'}, {plural(shape[1], 'column')}"


def obs_shape(payload):
    """Coarse shape of an observation payload, or None for invalid-action notices."""
    p = payload[1:] if payload.startswith(">") else payload
    if p.startswith(INVALID):
        return None
    if p.startswith(ERROR):
        return ("error",)
    if p.startswith("SQL Timeout"):
        return ("timeout",)
    if p.startswith("Empty DataFrame"):
        return ("empty",)
    trunc = p.startswith(TRUNC)
    lines = (p[len(TRUNC):] if trunc else p).rstrip("\n").split("\n")
    n_rows, n_cols = len(lines) - 1, len(lines[0].split())
    return ("many" if trunc or n_rows > 1 else "one", n_cols)


def insert(tok, seg_ids, line, after_think):
    """Insert `line` at the start of an action (after a leading '</think>\\n\\n' if asked/present)."""
    text = tok.decode(seg_ids)
    cut = len("</think>\n\n") if after_think and text.startswith("</think>\n\n") else 0
    a = tok.encode(text[:cut], add_special_tokens=False)
    b = tok.encode(line, add_special_tokens=False)
    c = tok.encode(text[cut:], add_special_tokens=False)
    if tok.decode(a + b + c) != text[:cut] + line + text[cut:]:
        return None, None
    return a + b + c, (len(a), len(a) + len(b))


def make(rec, tok, contract, stats):
    ids, cls = rec["input_ids"], rec["token_class"]
    acts, obs = runs(cls, (1,)), runs(cls, (2, 3))
    obs_start_by_end = {e: s for s, e in obs}  # observation that ends exactly where an action starts
    new_ids, new_cls, spans, pos = [], [], [], 0
    for k, (s, e) in enumerate(acts):
        new_ids += ids[pos:s]
        new_cls += cls[pos:s]
        seg = ids[s:e]
        line, kind = None, None
        if k == 0 and contract is not None:
            line, kind = f"Expected answer: {describe(contract)}.\n", "contract"
        elif k > 0 and contract is not None:
            o_start = obs_start_by_end.get(s)
            if o_start is not None:
                pay = [ids[t] for t in range(o_start, s) if cls[t] == 2]
                got = obs_shape(tok.decode(pay))
                if got is not None:
                    ok = got == contract
                    line = f"Check: got {describe(got)}; this {'matches' if ok else 'does not match'} the expected answer.\n"
                    kind = "check_match" if ok else "check_mismatch"
        if line is not None:
            new_seg, span = insert(tok, seg, line, after_think=k > 0)
            if new_seg is None:
                stats["skip_roundtrip"] += 1
            else:
                spans.append([kind, len(new_ids) + span[0], len(new_ids) + span[1]])
                seg = new_seg
                stats[kind] += 1
        new_ids += seg
        new_cls += [1] * len(seg)
        pos = e
    new_ids += ids[pos:]
    new_cls += cls[pos:]
    return rec | {"input_ids": new_ids, "token_class": new_cls, "e3_spans": spans}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--procs", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    pool_df = pd.read_parquet(f"{ROOT}/data/v1/sft_pool.parquet")
    jobs = [(f"{ROOT}/data/omnisql/data/{DB_DIR[r['data']]}/{r['db_id']}/{r['db_id']}.sqlite",
             r["reward_spec"]["ground_truth"]) for _, r in pool_df.iterrows()]
    with Pool(a.procs) as p:
        shapes = p.map(gold_shape, jobs, chunksize=16)
    gold = dict(enumerate(shapes))
    qids = [q for q, s in gold.items() if s is not None]
    perm = qids[:]
    random.Random(a.seed).shuffle(perm)
    placebo = {perm[i]: gold[perm[(i + 1) % len(perm)]] for i in range(len(perm))}  # derangement
    report = {"gold_shape": collections.Counter(s[0] if s else "fail" for s in shapes),
              "placebo_same_as_gold": sum(placebo[q] == gold[q] for q in qids) / len(qids)}
    recs = [json.loads(l) for l in open(f"{ROOT}/data/sft/train.jsonl") if l.strip()]
    for name, table in (("e3", gold), ("e3p", placebo)):
        st = collections.Counter()
        with open(f"{ROOT}/data/sft/train_{name}.jsonl", "w") as f:
            for r in recs:
                f.write(json.dumps(make(r, tok, table.get(r["qid"]), st)) + "\n")
        report[name] = dict(st)
        print(name, dict(st), flush=True)
    report["gold_shape"] = dict(report["gold_shape"])
    json.dump(report, open(f"{ROOT}/data/sft/e3_report.json", "w"), indent=1)
    json.dump({str(q): gold[q] for q in gold}, open(f"{ROOT}/data/sft/sft_pool_gold_shape.json", "w"))
    print(json.dumps(report))


if __name__ == "__main__":
    main()
