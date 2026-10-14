"""FT1 rerun on the generated ALF-CF v1 games (Know2Act PREREG Addendum 18, step 2). CPU only, no model calls.

Same definitions as ft1_prior.py, applied to the ALF-CF world (protocol: protocol/addendum18_alfcf_world_and_checks.md):
  priors are fitted on the ALF-CF TRAIN split with ft1_prior.fit_prior (per-instance p(o, t) and the object-agnostic
  marginal q(t)); the target's location is the receptacle that holds it after the move (meta.json "hits").
  Policies: U uniform; V ALFRED class constraints; M_cf object-agnostic marginal; C_cf object-conditional prior;
  C_std the standard ALFWorld prior (same train games before the move, i.e. T- / ALFRED data prior);
  Q4 / Q8 Qwen3-4B / 8B zero-shot type rankings from FT4 (only for the 274 valid games, both variants).
  Metrics: E[cost], top-1 hit; Delta_M = E[M_cf] - E[C_cf] (bar 1.0); Delta_std = E[C_std] - E[C_cf];
  decisions per game along the C_cf path (visits to the first hit) and decision points with >= 2 remaining types;
  CDF_A (C_cf top type differs from M_cf top type), strong version (prior gap >= 0.10);
  CDF_flip (20 foils from the scene's LEARNABLE classes) and the count of flippable decision points (>= 1 foil
  whose top type differs), which is the unit of the main study's power analysis.
Primary population: test split, learnable classes. Reported also: unlearnable, val, by subset and scene.
usage: python ft1cf_analysis.py <alfcf_dir> <rules.json> <out.json>
"""
import json
import math
import os
import random
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ft1_prior as F  # noqa: E402

ALF, RULES, OUT = sys.argv[1:4]
CLEAN = "--as_registered" not in sys.argv  # default: goal-leak fix applied
rules = json.load(open(RULES))
UN = set(rules["unlearnable"])
K2A = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REQ = {"single_contrast_0.10": 327, "three_contrasts_0.10": 437, "single_contrast_0.05": 1309}


def load(split):
    gs = []
    for l in open(os.path.join(ALF, "index.jsonl")):
        m = json.loads(l)
        if m["split"] != split:
            continue
        meta = json.load(open(os.path.join(ALF, split, m["id"], "meta.json")))
        if CLEAN:  # goal-leak fix (2026-10-13 13:30): drop the goal's '?r' pseudo-receptacle and pseudo-hit
            meta["receps"] = {k: v for k, v in meta["receps"].items() if not k.startswith("?")}
            meta["hits"] = [h for h in meta["hits"] if not h.startswith("?")]
            meta["orig_hits"] = [h for h in meta["orig_hits"] if not h.startswith("?")]
        g = dict(path=os.path.join(ALF, split, m["id"]), split=split, src_split=meta["src_split"], src=meta["src"],
                 task_type=meta["task_type"], target=meta["target"], floor=meta["floor"], scene=meta["scene"],
                 receps=meta["receps"], hits=sorted(meta["hits"]), learnable=meta["learnable"], moved=meta["moved"],
                 n_expert_search=0, n_walk=0)
        gs.append(g)
    return gs


def main():
    rng = random.Random(0)
    S = {s: load(s) for s in ["train", "val", "test"]}
    p_cf, q_cf = F.fit_prior(S["train"])
    std_train = [dict(g, hits=sorted(h for h in json.load(open(os.path.join(g["path"], "meta.json")))["orig_hits"]
                                     if not (CLEAN and h.startswith("?")))) for g in S["train"]]
    p_std, q_std = F.fit_prior(std_train)
    learn_by_scene = defaultdict(set)
    for g in S["train"]:
        if g["learnable"]:
            learn_by_scene[g["scene"]].add(g["target"])
    foil_pool = {k: sorted(v) for k, v in learn_by_scene.items()}

    # FT4 zero-shot rankings, keyed by source game path (valid games only)
    probes = {}
    for l in open(os.path.join(K2A, "results", "ft4_probes.jsonl")):
        p = json.loads(l)
        probes[p["idx"]] = p
    sys.argv_backup = sys.argv
    import importlib.util
    spec = importlib.util.spec_from_file_location("alfcf_rules", os.path.join(K2A, "scripts", "alfcf_rules.py"))
    R = importlib.util.module_from_spec(spec)
    sys.argv = ["x", "/dev/null"]
    spec.loader.exec_module(R)
    sys.argv = sys.argv_backup
    zs = {}
    for tag in ["qwen3-4b", "qwen3-8b"]:
        for l in open(os.path.join(K2A, "results", f"ft4_resp_{tag}.jsonl")):
            r = json.loads(l)
            p = probes[r["idx"]]
            if p["split"] == "train":
                continue
            rel = p["path"] if not os.path.isabs(p["path"]) else os.path.relpath(p["path"], F.DATA)
            ranked = R.parse_ranking(r["response"], list(p["n"]))
            zs[(tag, rel)] = ranked

    def evaluate(games):
        per = []
        for g in games:
            n, h = F.type_counts(g)
            types = list(n)
            if not g["hits"]:
                continue
            po = p_cf.get(g["target"], {})
            pol = {"U": [types],
                   "V": [x for x in ([t for t in types if t in F.VALID.get(g["target"], set())],
                                     [t for t in types if t not in F.VALID.get(g["target"], set())]) if x],
                   "M_cf": F.groups_from_scores(types, q_cf), "C_cf": F.groups_from_scores(types, po),
                   "C_std": F.groups_from_scores(types, p_std.get(g["target"], {}))}
            for tag in ["qwen3-4b", "qwen3-8b"]:
                rk = zs.get((tag, g["src"]))
                if rk is not None:
                    miss = [t for t in types if t not in rk]
                    pol["Q" + tag[-2:]] = [[t] for t in rk if t in n] + ([miss] if miss else [])
            row = dict(scene=g["scene"], task_type=g["task_type"], learnable=g["learnable"], moved=g["moved"],
                       src_split=g["src_split"], target=g["target"])
            for k, grp in pol.items():
                c, _ = F.exp_cost(grp, n, h)
                row[f"cost_{k}"] = c
                row[f"top1_{k}"] = F.top1(grp, n, h)
            # decision analysis along the C_cf path (as ft1_prior.decision_analysis, foils = learnable classes)
            foils = [f for f in foil_pool.get(g["scene"], []) if f != g["target"]]
            if len(foils) > 20:
                foils = rng.sample(foils, 20)
            dp = critA = critS = 0
            flip = 0.0
            flippable = 0
            dp_all = 0
            for _ in range(20):
                path = F.sample_path(pol["C_cf"], n, set(g["hits"]), g["receps"], rng)
                visited = set()
                for r in path:
                    rem = {g["receps"][x] for x in g["receps"] if x not in visited}
                    dp_all += 1
                    if len(rem) >= 2:
                        dp += 1
                        aC = F.argmax_set(rem, po)
                        aM = F.argmax_set(rem, q_cf)
                        if not (aC & aM):
                            critA += 1
                            if max(po.get(t, 0) for t in aC) - max(po.get(t, 0) for t in aM) >= 0.10:
                                critS += 1
                        if foils:
                            fl = [not (F.argmax_set(rem, p_cf.get(f, {})) & aC) for f in foils]
                            flip += sum(fl) / len(fl)
                            flippable += any(fl)
                    visited.add(r)
            row.update(dp=dp / 20, dp_all=dp_all / 20, critA=critA / 20, critS=critS / 20, flip=flip / 20,
                       flippable=flippable / 20)
            per.append(row)
        return per

    def summ(rows):
        if not rows:
            return {"n": 0}
        s = {"n": len(rows)}
        for k in ["U", "V", "M_cf", "C_cf", "C_std", "Q4b", "Q8b"]:
            v = [r[f"cost_{k}"] for r in rows if f"cost_{k}" in r and not math.isnan(r[f"cost_{k}"])]
            if v:
                s[f"E_cost_{k}"] = sum(v) / len(v)
                s[f"top1_{k}"] = sum(r[f"top1_{k}"] for r in rows if f"top1_{k}" in r) / len(v)
                s[f"n_{k}"] = len(v)
        s["Delta_M"] = s["E_cost_M_cf"] - s["E_cost_C_cf"]
        s["Delta_V"] = s["E_cost_V"] - s["E_cost_C_cf"]
        s["Delta_std"] = s["E_cost_C_std"] - s["E_cost_C_cf"]
        dp = sum(r["dp"] for r in rows)
        s["decisions_per_game"] = sum(r["dp_all"] for r in rows) / len(rows)
        s["dp_ge2_per_game"] = dp / len(rows)
        s["CDF_A"] = sum(r["critA"] for r in rows) / dp if dp else float("nan")
        s["CDF_A_strong"] = sum(r["critS"] for r in rows) / dp if dp else float("nan")
        s["CDF_flip"] = sum(r["flip"] for r in rows) / dp if dp else float("nan")
        s["flippable_dp_total"] = sum(r["flippable"] for r in rows)
        s["flippable_share"] = s["flippable_dp_total"] / dp if dp else float("nan")
        # game-cluster bootstrap CI of Delta_M
        d = [r["cost_M_cf"] - r["cost_C_cf"] for r in rows]
        b = random.Random(1)
        bs = sorted(sum(b.choice(d) for _ in d) / len(d) for _ in range(2000))
        s["Delta_M_CI95"] = [bs[50], bs[1949]]
        return s

    res = {"priors_fitted_on": "ALF-CF train split", "n_games": {k: len(v) for k, v in S.items()},
           "power_requirements_per_arm": REQ}
    for split in ["test", "val"]:
        per = evaluate(S[split])
        out = {"learnable": summ([r for r in per if r["learnable"]]),
               "unlearnable": summ([r for r in per if not r["learnable"]]),
               "all": summ(per),
               "learnable_moved": summ([r for r in per if r["learnable"] and r["moved"]]),
               "learnable_unmoved": summ([r for r in per if r["learnable"] and not r["moved"]])}
        for key in ["src_split", "scene"]:
            for val in sorted({r[key] for r in per}):
                out[f"learnable|{key}={val}"] = summ([r for r in per if r["learnable"] and r[key] == val])
        res[split] = out
    t = res["test"]
    L, U = t["learnable"], t["unlearnable"]
    verdict = "GO"
    if L["Delta_M"] < 0.3 or L["CDF_A"] < 0.20:
        verdict = "NO-GO"
    elif (L["Delta_M"] < 1.0 or L["CDF_A"] < 0.40 or L["flippable_dp_total"] < REQ["three_contrasts_0.10"]
          or L["decisions_per_game"] < 1.5 or U["Delta_M"] > L["Delta_M"] - 1.0):
        verdict = "REDESIGN"
    res["verdict_prereg_rule"] = verdict
    json.dump(res, open(OUT, "w"), indent=1)
    for split in ["test", "val"]:
        for k in ["learnable", "unlearnable", "all", "learnable_moved"]:
            print(split, k, {a: (round(b, 3) if isinstance(b, float) else b) for a, b in res[split][k].items()})
    print("VERDICT", verdict)


if __name__ == "__main__":
    main()
