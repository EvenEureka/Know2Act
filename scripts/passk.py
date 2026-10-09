"""Unbiased pass@k (Chen et al. 2021) from rollout.py eval outputs, with paired bootstrap comparisons.

usage:
  passk.py --runs A=results/eval/A/*.jsonl B=results/eval/B/*.jsonl --ks 1,2,4,8,16 --out summary.json
Per run and data_source: pass@k averaged over questions, format-failure rate, mean turns.
For every pair of runs: difference in pass@k on the shared questions with a 95% paired bootstrap
CI over questions (questions resampled jointly for both runs).
"""
import argparse
import collections
import glob
import json
import math
import random


def pass_at_k(n, c, k):
    if n - c < k:
        return 1.0
    return 1.0 - math.prod((n - c - i) / (n - i) for i in range(k))


def load(pattern):
    per_q = collections.defaultdict(list)
    meta = collections.defaultdict(lambda: {"fmt_fail": 0, "turns": 0, "n": 0})
    for fn in sorted(glob.glob(pattern)):
        for line in open(fn):
            r = json.loads(line)
            key = (r["data_source"], r["qid"])
            per_q[key].append(r["reward"])
            m = meta[r["data_source"]]
            m["n"] += 1
            m["fmt_fail"] += r["reward"] < 0
            m["turns"] += r["turns"]
    return per_q, meta


def scores(per_q, k):
    return {key: pass_at_k(len(rs), sum(x == 1.0 for x in rs), k) for key, rs in per_q.items() if len(rs) >= k}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="+", required=True, help="NAME=glob")
    p.add_argument("--ks", default="1,2,4,8,16")
    p.add_argument("--boot", type=int, default=10000)
    p.add_argument("--out", default=None)
    p.add_argument("--exclude", nargs="*", default=[],
                   help="DATA_SOURCE=results/gold_flags/<set>.json: drop questions whose gold is empty/failing")
    a = p.parse_args()
    ks = [int(x) for x in a.ks.split(",")]
    drop = set()
    for spec in a.exclude:
        ds, path = spec.split("=", 1)
        drop |= {(ds, int(q)) for q in json.load(open(path))}
    runs = {}
    for spec in a.runs:
        name, pat = spec.split("=", 1)
        per_q, meta = load(pat)
        runs[name] = ({k: v for k, v in per_q.items() if k not in drop}, meta)
    summary = {"runs": {}, "pairs": {}}
    for name, (per_q, meta) in runs.items():
        out = {}
        for ds in sorted({d for d, _ in per_q}):
            sub = {kq: v for kq, v in per_q.items() if kq[0] == ds}
            n_samples = collections.Counter(len(v) for v in sub.values())
            row = {"questions": len(sub), "samples_per_q": dict(n_samples),
                   "format_fail": round(meta[ds]["fmt_fail"] / meta[ds]["n"], 4),
                   "mean_turns": round(meta[ds]["turns"] / meta[ds]["n"], 3)}
            for k in ks:
                s = scores(sub, k)
                if s:
                    row[f"pass@{k}"] = round(100 * sum(s.values()) / len(s), 2)
            out[ds] = row
        summary["runs"][name] = out
    names = list(runs)
    rng = random.Random(0)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            A, B = names[i], names[j]
            res = {}
            for ds in summary["runs"][A]:
                keys = sorted(k for k in runs[A][0] if k[0] == ds and k in runs[B][0])
                if not keys:
                    continue
                for k in ks:
                    sa, sb = scores(runs[A][0], k), scores(runs[B][0], k)
                    common = [q for q in keys if q in sa and q in sb]
                    if not common:
                        continue
                    d = [sb[q] - sa[q] for q in common]
                    mean = sum(d) / len(d)
                    boots = sorted(sum(d[rng.randrange(len(d))] for _ in d) / len(d) for _ in range(a.boot))
                    res[f"{ds}/pass@{k}"] = {"B_minus_A_pp": round(100 * mean, 2),
                                             "ci95": [round(100 * boots[int(0.025 * a.boot)], 2),
                                                      round(100 * boots[int(0.975 * a.boot)], 2)],
                                             "questions": len(common)}
            summary["pairs"][f"{A}->{B}"] = res
    txt = json.dumps(summary, indent=1)
    print(txt)
    if a.out:
        open(a.out, "w").write(txt)


if __name__ == "__main__":
    main()
