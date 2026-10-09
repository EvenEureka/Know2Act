"""Minimal reproduction: a constant empty-result answer earns reward 1.0 on 44% of SkyRL-SQL-653.

Uses the unmodified SkyRL text2sql env (skyrl_gym, commit 99824f0 / main as of 2026-10-08), the
unmodified HF dataset NovaSky-AI/SkyRL-SQL-653-data-newfmt (train.parquet) and the OmniSQL
databases (seeklhy/OmniSQL-datasets data.zip), exactly as in the SkyRL-SQL recipe docs.

usage: python repro_empty_gold.py --train train.parquet --db_path <OmniSQL data/ folder> [--out empty_gold_ids.txt]
"""
import argparse
import collections

import pandas as pd
import skyrl_gym
from omegaconf import DictConfig

ANSWER = "<think>I will return the final query.</think>\n<solution>SELECT 1 WHERE 0;</solution>"

p = argparse.ArgumentParser()
p.add_argument("--train", required=True)
p.add_argument("--db_path", required=True)
p.add_argument("--out", default=None, help="write the indices of affected tasks")
a = p.parse_args()
df = pd.read_parquet(a.train)
hits, by_src = [], collections.Counter()
for i, r in df.iterrows():
    extras = {k: r[k] for k in r.index if k not in ("prompt", "env_class")} | {"max_turns": 6}
    env = skyrl_gym.make("text2sql", env_config=DictConfig({"db_path": a.db_path}), extras=extras)
    env.init(list(r["prompt"]))
    if env.step(ANSWER)["reward"] == 1.0:
        hits.append(int(i))
        by_src[r["data"]] += 1
print(f'constant answer "SELECT 1 WHERE 0;" -> reward 1.0 on {len(hits)}/{len(df)} tasks '
      f"({len(hits) / len(df):.1%}); by source: {dict(by_src)}")
if a.out:
    open(a.out, "w").write("\n".join(map(str, hits)) + "\n")
