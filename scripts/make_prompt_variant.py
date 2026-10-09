"""Write a copy of a SkyRL-format parquet whose system message carries a protocol variant.

The same transformation is applied to the SFT pool, the RL tasks and both eval sets, so every stage
sees an identical prompt. Variants:
  orig    unchanged SkyRL-SQL system message
  verify  one added line in the Format section: run at least one SQL query before the final solution
"""
import argparse
import hashlib

import pandas as pd

ANCHOR = ("- You can use SQL tool written within a single <sql>your sql</sql> block to explore or verify. "
          "SQL tool output will be shown as dataframe inside <observation>...</observation>. "
          "Based on this observation, you can think again and refine.\n")
VERIFY = ("- Before giving the final solution, you MUST execute at least one SQL query with the SQL tool "
          "(for example, your candidate query) and check its result in the observation.\n")


def apply(system, variant):
    if variant == "orig":
        return system
    assert system.count(ANCHOR) == 1, "anchor line not found exactly once"
    return system.replace(ANCHOR, ANCHOR + VERIFY)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True)
    p.add_argument("--dst", required=True)
    p.add_argument("--variant", choices=["orig", "verify"], required=True)
    a = p.parse_args()
    df = pd.read_parquet(a.src)
    new = []
    for msgs in df["prompt"]:
        msgs = [dict(m) for m in msgs]
        assert msgs[0]["role"] == "system"
        msgs[0]["content"] = apply(msgs[0]["content"], a.variant)
        new.append(msgs)
    df["prompt"] = new
    df.to_parquet(a.dst)
    sysmsg = new[0][0]["content"]
    print(a.dst, len(df), "system sha256", hashlib.sha256(sysmsg.encode()).hexdigest()[:16])


if __name__ == "__main__":
    main()
