"""Build the RL / eval / SFT-pool parquets in the exact SkyRL-SQL row format.

Outputs (under data/):
  rl/train.parquet          SkyRL-SQL-653 unchanged (RL tasks)
  eval/spider_dev.parquet   SkyRL-SQL validation unchanged (1034 Spider-dev)
  eval/bird_dev.parquet     BIRD-dev 1534, built from OmniSQL dev_bird.json + bird dev.json
  sft/pool.parquet          Spider-train + train_others questions for teacher rollouts (SFT data)
  sft/heldout.parquet       questions on held-out DBs, for mechanism measurements only
  manifest.json             counts, overlap checks, sha256 of every output

The system message is copied byte-for-byte from the SkyRL parquet; the user message follows the
pattern verified on all 1687 SkyRL rows:
  "{db_details}:" + schema + ";\n {external_knowledge}: " + ek + ";\n {question}: " + question
BIRD rows use data="bird"; the env reads DBs from {db_path}/bird/train/train_databases, which we
point at the dev databases with a symlink (no upstream code change).
"""
import hashlib
import json
import os
import random
import re

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OMNI = os.path.join(ROOT, "data", "omnisql", "data")
SEED = 20261003
N_HELDOUT_DBS = 12


def schema_from_input_seq(s):
    a = s.index("Database Schema:\n") + len("Database Schema:\n")
    b = s.index("\nThis schema describes")
    return s[a:b]


def user_msg(schema, ek, question):
    return "{db_details}:" + schema + ";\n {external_knowledge}: " + ek + ";\n {question}: " + question


def norm_sql(s):
    return re.sub(r"\s+", " ", s.strip().rstrip(";")).lower()


def norm_q(s):
    return re.sub(r"\s+", " ", s.strip()).lower()


def row(system, schema, ek, question, sql, db_id, data, data_source, split, index):
    return {
        "db_id": db_id, "sql": sql, "question": question, "schema": schema,
        "external_knowledge": ek, "data": data, "data_source": data_source,
        "prompt": [{"role": "system", "content": system},
                   {"role": "user", "content": user_msg(schema, ek, question)}],
        "env_class": "text2sql",
        "reward_spec": {"ground_truth": sql, "method": "rule"},
        "extra_info": {"index": index, "split": split},
    }


def sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def main():
    train = pd.read_parquet(os.path.join(ROOT, "data/skyrl_sql/train.parquet"))
    val = pd.read_parquet(os.path.join(ROOT, "data/skyrl_sql/validation.parquet"))
    system = train.iloc[0]["prompt"][0]["content"]
    assert all(r[0]["content"] == system for r in list(train["prompt"]) + list(val["prompt"]))
    # check the user-message pattern on every SkyRL row before relying on it
    for df in (train, val):
        for _, r in df.iterrows():
            assert r["prompt"][1]["content"] == user_msg(r["schema"], r["external_knowledge"], r["question"])
    man = {"system_sha256": hashlib.sha256(system.encode()).hexdigest()}

    os.makedirs(os.path.join(ROOT, "data/rl"), exist_ok=True)
    os.makedirs(os.path.join(ROOT, "data/eval"), exist_ok=True)
    os.makedirs(os.path.join(ROOT, "data/sft"), exist_ok=True)
    train.to_parquet(os.path.join(ROOT, "data/rl/train.parquet"))
    val.to_parquet(os.path.join(ROOT, "data/eval/spider_dev.parquet"))

    # ---- BIRD dev ----
    omni_bird = json.load(open(os.path.join(OMNI, "dev_bird.json")))
    bird = json.load(open(os.path.join(OMNI, "bird/dev_20240627/dev.json")))
    assert len(omni_bird) == len(bird) == 1534
    mism = sum(norm_sql(o["output_seq"]) != norm_sql(b["SQL"]) for o, b in zip(omni_bird, bird))
    assert mism == 0, f"BIRD alignment mismatches: {mism}"
    rows = [row(system, schema_from_input_seq(o["input_seq"]), b["evidence"] or "None", b["question"],
                b["SQL"], b["db_id"], "bird", "bird_dev", "bird_dev", i)
            for i, (o, b) in enumerate(zip(omni_bird, bird))]
    pd.DataFrame(rows).to_parquet(os.path.join(ROOT, "data/eval/bird_dev.parquet"))
    link = os.path.join(OMNI, "bird/train/train_databases")
    os.makedirs(os.path.dirname(link), exist_ok=True)
    if not os.path.islink(link):
        os.symlink("../dev_20240627/dev_databases", link)
    missing = [r["db_id"] for r in rows
               if not os.path.exists(os.path.join(link, r["db_id"], r["db_id"] + ".sqlite"))]
    assert not missing, f"BIRD dbs missing: {sorted(set(missing))}"
    man["bird_dev"] = {"rows": len(rows), "dbs": len({r['db_id'] for r in rows}), "evidence_empty": sum(1 for b in bird if not b["evidence"])}

    # ---- SFT pool: Spider train + train_others ----
    omni_tr = json.load(open(os.path.join(OMNI, "train_spider.json")))
    sp_tr = json.load(open(os.path.join(OMNI, "spider/train_spider.json")))
    sp_ot = json.load(open(os.path.join(OMNI, "spider/train_others.json")))
    assert len(omni_tr) == len(sp_tr) == 7000
    # OmniSQL train output_seq is a CoT answer, not bare SQL, so align on the question text instead
    def q_of(inp):
        m = re.search(r"Question:\n(.*?)\n\nInstructions:", inp, re.S)
        return norm_q(m.group(1)) if m else None
    mism = sum(q_of(o["input_seq"]) != norm_q(s["question"]) for o, s in zip(omni_tr, sp_tr))
    assert mism == 0, f"Spider-train alignment mismatches: {mism}"
    man["spider_train_alignment_mismatch"] = mism
    schema_by_db = {}
    for o, s in zip(omni_tr, sp_tr):
        schema_by_db.setdefault(s["db_id"], schema_from_input_seq(o["input_seq"]))
    rl_keys = {(r["db_id"], norm_q(r["question"])) for _, r in train.iterrows()}
    rl_dbs = set(train["db_id"])
    cand, dropped_overlap, no_schema = [], 0, 0
    for src, items in (("train_spider", sp_tr), ("train_others", sp_ot)):
        for i, s in enumerate(items):
            if (s["db_id"], norm_q(s["question"])) in rl_keys:
                dropped_overlap += 1
                continue
            if s["db_id"] not in schema_by_db:
                no_schema += 1
                continue
            if not os.path.exists(os.path.join(OMNI, "spider/database", s["db_id"], s["db_id"] + ".sqlite")):
                no_schema += 1
                continue
            cand.append((src, i, s))
    dbs = sorted({s["db_id"] for _, _, s in cand} - rl_dbs)
    random.Random(SEED).shuffle(dbs)
    held_dbs = set(dbs[:N_HELDOUT_DBS])
    pool, held = [], []
    for src, i, s in cand:
        r = row(system, schema_by_db[s["db_id"]], "None", s["question"], s["query"], s["db_id"],
                "spider", "spider_train", src, i)
        (held if s["db_id"] in held_dbs else pool).append(r)
    pd.DataFrame(pool).to_parquet(os.path.join(ROOT, "data/sft/pool.parquet"))
    pd.DataFrame(held).to_parquet(os.path.join(ROOT, "data/sft/heldout.parquet"))
    man["sft"] = {"pool_rows": len(pool), "heldout_rows": len(held), "heldout_dbs": sorted(held_dbs),
                  "dropped_rl_question_overlap": dropped_overlap, "dropped_no_schema_or_db": no_schema,
                  "pool_dbs_shared_with_rl": len({r['db_id'] for r in pool} & rl_dbs),
                  "train_others_without_omnisql_schema": sum(1 for s in sp_ot if s["db_id"] not in schema_by_db)}
    for rel in ["rl/train.parquet", "eval/spider_dev.parquet", "eval/bird_dev.parquet",
                "sft/pool.parquet", "sft/heldout.parquet"]:
        man.setdefault("sha256", {})[rel] = sha(os.path.join(ROOT, "data", rel))
    json.dump(man, open(os.path.join(ROOT, "data/manifest.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in man.items() if k != "sha256"}, indent=1))


if __name__ == "__main__":
    main()
