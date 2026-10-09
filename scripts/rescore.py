"""Offline re-scoring of rollout records under SkyRL's strict and our lenient format rule (CPU).

The env's reward input is the concatenation of every assistant action and observation content in
order; rollout records keep both texts, so the reward can be recomputed without the GPU.
"""
import argparse
import collections
import json
import os
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from skyrl_gym.envs.sql.utils import compute_score_single  # noqa: E402

from agenticrl.sql_lenient import compute_score_lenient  # noqa: E402

DB_DIR = {"synsql": "SynSQL-2.5M/databases", "spider": "spider/database", "bird": "bird/train/train_databases"}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rollouts", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--db_path", default=os.path.join(ROOT, "data/omnisql/data"))
    a = p.parse_args()
    df = pd.read_parquet(a.data)
    recs = [json.loads(line) for line in open(a.rollouts)]
    strict, lenient, turns, act, obs = [], [], [], 0, 0
    for r in recs:
        row = df.loc[r["qid"]]
        db = os.path.join(a.db_path, DB_DIR[row["data"]], row["db_id"], row["db_id"] + ".sqlite")
        segs = [s for s in r["segments"] if not s.get("dropped")]
        text = "".join(s["text"] for s in segs)
        finished = r["stop_reason"] != "length" and "<solution>" in text
        strict.append(compute_score_single(text, row["reward_spec"]["ground_truth"], db) if finished else -1.0)
        lenient.append(compute_score_lenient(text, row["reward_spec"]["ground_truth"], db) if finished else -1.0)
        turns.append(r["turns"])
        act += sum(s["n_tokens"] for s in segs if s["type"] == "act")
        obs += sum(s["n_tokens"] for s in segs if s["type"] == "obs")
    n = len(recs)
    kept = [i for i, x in enumerate(lenient) if x >= 0]
    k_act = sum(sum(s["n_tokens"] for s in recs[i]["segments"] if s["type"] == "act" and not s.get("dropped")) for i in kept)
    k_obs = sum(sum(s["n_tokens"] for s in recs[i]["segments"] if s["type"] == "obs" and not s.get("dropped")) for i in kept)
    out = {"n": n, "strict_acc": sum(x == 1 for x in strict) / n, "strict_fmt_fail": sum(x < 0 for x in strict) / n,
           "lenient_acc": sum(x == 1 for x in lenient) / n, "lenient_fmt_fail": sum(x < 0 for x in lenient) / n,
           "mean_turns": sum(turns) / n, "turn_hist": dict(sorted(collections.Counter(turns).items())),
           "obs_share_all": round(obs / max(1, act + obs), 4), "obs_share_lenient_kept": round(k_obs / max(1, k_act + k_obs), 4),
           "kept_multi_turn": sum(1 for i in kept if recs[i]["turns"] > 1), "kept": len(kept),
           "mean_act_tokens_kept": round(k_act / max(1, len(kept)), 1), "mean_obs_tokens_kept": round(k_obs / max(1, len(kept)), 1)}
    print(json.dumps(out))


if __name__ == "__main__":
    main()
