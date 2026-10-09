"""Collect every number the manuscript uses into paper/data/results.json (single source of truth).

Re-run after pending runs finish; anything missing is stored as null and typeset as \\pending{}.
Primary set = questions whose gold result is non-empty and executable (results/gold_flags).
"""
import glob
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import passk  # noqa: E402

R = os.path.join(ROOT, "results")
DROP = {("spider", int(q)) for q in json.load(open(f"{R}/gold_flags/spider_dev.json"))} | \
       {("bird_dev", int(q)) for q in json.load(open(f"{R}/gold_flags/bird_dev.json"))}


def latest(pattern):
    """Newest run dir matching a pattern that actually contains rollouts (failed attempts are empty)."""
    c = [d for d in sorted(glob.glob(pattern)) if glob.glob(f"{d}/*.jsonl")]
    return c[-1] if c else None


EVAL = {"base": "pre_base", "sft_A": "pre_A", "sft_B": "pre_B", "sft_C": "pre_C", "sft_D": "pre_D",
        "sft_E1": "pre_E1", "sft_E2": "pre_E2", "sft_E3": "pre_E3", "sft_E3p": "pre_E3p", "sft_Aseed1": "pre_Aseed1",
        "grpo_A_s1": "grpo_A_s1_1508971", "grpo_A_s2": "grpo_A_s2_1510961", "grpo_B_s1": "grpo_B_s1_1510918",
        "grpo_C_s1": "grpo_C_s1_1510955"}
for name, pat in (("grpo_B_s2", "grpo_B_s2_15109*"), ("grpo_E2_s1", "grpo_E2_s1_*"), ("grpo_E3_s1", "grpo_E3_s1_*")):
    d = latest(f"{R}/eval/{pat}")
    EVAL[name] = os.path.basename(d) if d else None


def point(run_dir):
    per_q, meta = passk.load(f"{R}/eval/{run_dir}/*.jsonl")
    out = {}
    for ds, key in (("spider", "spider"), ("bird_dev", "bird")):
        for subset, keep in (("all", lambda q: True), ("primary", lambda q: q not in DROP)):
            sub = {q: v for q, v in per_q.items() if q[0] == ds and keep(q)}
            for k in (1, 16):
                s = passk.scores(sub, k)
                out[f"{key}_{subset}_p{k}"] = round(100 * sum(s.values()) / len(s), 2) if s else None
            if subset == "primary":  # full pass@k curve for the RL-gain figure
                out[f"{key}_primary_curve"] = [round(100 * sum(v.values()) / len(v), 3) if v else None
                                               for v in (passk.scores(sub, k) for k in range(1, 17))]
            out[f"{key}_{subset}_n"] = len(sub)
        out[f"{key}_format_fail"] = round(meta[ds]["fmt_fail"] / meta[ds]["n"], 4)
    return out


def jload(path):
    try:
        return json.load(open(path))
    except (OSError, ValueError):
        return None


def summ(path, key):
    d = jload(path)
    return round(d["summary"][key], 4) if d and key in d.get("summary", {}) else None


def train_curve(run, key, buckets=((0, 5), (5, 20), (20, 50), (50, 100), (100, 200), (200, 300), (300, 375))):
    f = glob.glob(f"{R}/sft/{run}/train_log.rank0.jsonl")
    if not f:
        return None
    rows = [json.loads(l) for l in open(f[0])]
    out = []
    for a, b in buckets:
        v = [r[key] for r in rows[a:b] if r.get(key) is not None]
        out.append({"steps": [a, b], "mean": round(sum(v) / len(v), 4) if v else None})
    return out


res = {"eval": {}, "pairs": {}, "sft": {}, "rl_move": {}, "diag": {}, "e3": {}, "killtest": None, "audit": {}}
for name, d in EVAL.items():
    res["eval"][name] = point(d) if d and glob.glob(f"{R}/eval/{d}/*.jsonl") else None
    res["eval"][name] = res["eval"][name] and res["eval"][name] | {"dir": d}
for f in glob.glob(f"{R}/eval/primary_pairs_*.json"):
    res["pairs"][os.path.basename(f)] = jload(f)["pairs"]

SFT = {"A": "sft_A", "B": "sft_B", "C085": "sft_C085", "C070": "sft_C070", "C055": "sft_C055",
       "C040": "sft_C040", "D": "sft_D", "Dobs": "sft_Dobs", "E1": "sft_E1"}
for arm, m in SFT.items():
    res["sft"][arm] = {k: summ(f"{R}/measure/{m}.json", k) for k in
                       ("kl_act", "ce_act", "ce_payload", "ce_wrapper", "ent_act", "grad_cos", "grad_ratio")}
res["sft"]["lr_mult"] = {"A": 1.0, "C085": 0.85, "C070": 0.70, "C055": 0.55, "C040": 0.40}
res["sft"]["train_curves"] = {"B_obs_ce": train_curve("sft_B_1507462", "obs_ce"),
                              "D_wiki_ce": train_curve("sft_D_1507466", "dilute_ce"),
                              "Dobs_ce": train_curve("sft_Dobs_1513674", "dilute_ce"),
                              "A_act_ce": train_curve("sft_A_1507464", "act_ce"),
                              "B_act_ce": train_curve("sft_B_1507462", "act_ce")}
for f in glob.glob(f"{R}/measure/move_grpo_*.json"):
    res["rl_move"][re.sub(r"^move_|\.json$", "", os.path.basename(f))] = {
        k: summ(f, k) for k in ("kl_act", "ent_act")}
res["diag"]["kl_vs_A"] = {n: summ(f"{R}/measure/diag_kl_{n}_vs_A.json", "kl_act") for n in ("B", "C085", "D", "Aseed1")}
res["diag"]["kl_vs_A_e2ctx"] = {n: summ(f"{R}/measure/diag_e2ctx_{n}_vs_A.json", "kl_act") for n in ("B", "E1", "E2")}
res["diag"]["label_ce_e1slot"] = {n: summ(f"{R}/measure/diag_label_e1slot_{n}.json", "ce_payload") for n in ("A", "B", "E1")}
res["diag"]["label_ce_e2slot"] = {n: summ(f"{R}/measure/diag_label_e2slot_{n}.json", "ce_act") for n in ("A", "B", "E2")}
res["diag"]["train_traj"] = {n: {k: summ(f"{R}/measure/diag_train_{n}.json", k) for k in
                                 ("ce_act", "ce_payload", "ce_wrapper", "grad_cos", "grad_ratio")} for n in ("A", "B")}
res["diag"]["obs_breakdown"] = jload(f"{R}/measure/diag_obs_breakdown.json")
for f in glob.glob(f"{R}/e3_readouts_*.json"):
    res["e3"][re.sub(r"^e3_readouts_|\.json$", "", os.path.basename(f))] = jload(f)
res["killtest"] = jload(f"{R}/killtest_shape_v3.json")
res["audit"] = {"sql653_empty_reward": [290, 653], "by_source": {"synsql": 283, "spider": 7},
                "now_dependent_total": 22, "now_dependent_empty": 18, "spider_dev_empty": [49, 1034],
                "rl_tasks_used": 363}
os.makedirs(f"{ROOT}/paper/data", exist_ok=True)
json.dump(res, open(f"{ROOT}/paper/data/results.json", "w"), indent=1)
print("pending:", [k for k, v in res["eval"].items() if v is None])
