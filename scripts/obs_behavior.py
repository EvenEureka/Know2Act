"""Behaviour after observations (paper App. E; all questions) in eval rollouts: errors, empty results, retries, recovery."""
import json, glob, re, sys, collections
SQL = re.compile(r"<sql>(.*?)</sql>", re.S)
def kind(t):
    t = t[:300]
    if "Error executing SQL" in t: return "error"
    if "Empty DataFrame" in t: return "empty"
    return "rows"
def stats(pattern):
    n = collections.Counter()
    for f in glob.glob(pattern):
        for l in open(f):
            r = json.loads(l); segs = r["segments"]; n["traj"] += 1; n["turns"] += r["turns"]
            n["reward1"] += r["reward"] == 1.0
            had_err = False
            for i, s in enumerate(segs):
                if s["type"] != "obs": continue
                k = kind(s["text"]); n["obs_" + k] += 1
                prev = SQL.findall(segs[i - 1]["text"]) if i > 0 else []
                nxt = segs[i + 1]["text"] if i + 1 < len(segs) else ""
                nxt_sql = SQL.findall(nxt)
                if k in ("error", "empty"):
                    n[k + "_followed"] += 1
                    if prev and nxt_sql and prev[-1].strip() == nxt_sql[-1].strip(): n[k + "_retry_same"] += 1
                    if "<solution>" in nxt: n[k + "_then_answer"] += 1
                if k == "error": had_err = True
            if had_err:
                n["traj_err"] += 1; n["traj_err_solved"] += r["reward"] == 1.0
    return n
report = {}
args = [a for a in sys.argv[1:] if not a.startswith("--out=")]
out = next((a[6:] for a in sys.argv[1:] if a.startswith("--out=")), None)
for name, pat in [(a.split("=")[0], a.split("=")[1]) for a in args]:
    for ds in ("spider_dev", "bird_dev"):
        n = stats(pat + f"/{ds}.shard*.jsonl"); T = n["traj"]
        report[f"{name}/{ds}"] = {"turns": round(n["turns"]/T, 3), "traj_with_error": round(n["traj_err"]/T, 4),
            "solved_after_error": round(n["traj_err_solved"]/max(1, n["traj_err"]), 4),
            "error_retry_same_sql": round(n["error_retry_same"]/max(1, n["error_followed"]), 4),
            "empty_obs_per_traj": round(n["obs_empty"]/T, 4),
            "empty_then_answer": round(n["empty_then_answer"]/max(1, n["empty_followed"]), 4),
            "empty_retry_same_sql": round(n["empty_retry_same"]/max(1, n["empty_followed"]), 4), "trajectories": T}
        print(f"{name:8s} {ds:10s} turns {n['turns']/T:.2f} | traj w/ error {n['traj_err']/T:.3f} solved-after-error {n['traj_err_solved']/max(1,n['traj_err']):.3f}"
              f" | error->same SQL {n['error_retry_same']/max(1,n['error_followed']):.3f} | empty obs/traj {n['obs_empty']/T:.3f}"
              f" empty->answer {n['empty_then_answer']/max(1,n['empty_followed']):.3f} empty->same {n['empty_retry_same']/max(1,n['empty_followed']):.3f}")

if out:
    json.dump(report, open(out, "w"), indent=1)
