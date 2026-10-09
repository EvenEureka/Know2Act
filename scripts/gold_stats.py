"""Descriptives for PREREG §5: how many tasks have a gold SQL that returns an empty result (any query
returning nothing then scores 1), or that fails / times out."""
import json
import os
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from skyrl_gym.envs.sql.utils import execute_sql_wrapper_single  # noqa: E402

DB_DIR = {"synsql": "SynSQL-2.5M/databases", "spider": "spider/database", "bird": "bird/train/train_databases"}
out = {}
for name in ["v1/rl_train", "v1/spider_dev", "v1/bird_dev"]:
    df = pd.read_parquet(os.path.join(ROOT, "data", name + ".parquet"))
    empty = fail = 0
    flags = {}
    for qid, r in df.iterrows():
        db = os.path.join(ROOT, "data/omnisql/data", DB_DIR[r["data"]], r["db_id"], r["db_id"] + ".sqlite")
        res = execute_sql_wrapper_single(db, r["reward_spec"]["ground_truth"], 30, "")
        rows, ok = res[2], res[3]
        if not ok:
            fail += 1
            flags[int(qid)] = "fail"
        elif len(rows) == 0:
            empty += 1
            flags[int(qid)] = "empty"
    out[name] = {"n": len(df), "gold_empty": empty, "gold_fail_or_timeout": fail,
                 "gold_empty_frac": round(empty / len(df), 4)}
    print(name, out[name], flush=True)
    os.makedirs(os.path.join(ROOT, "results/gold_flags"), exist_ok=True)
    json.dump(flags, open(os.path.join(ROOT, "results/gold_flags", name.split("/")[-1] + ".json"), "w"))
json.dump(out, open(os.path.join(ROOT, "results/gold_stats.json"), "w"), indent=1)
