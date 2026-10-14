"""Per-model read-outs of ALF-CF evaluation v3 (PREREG Addendum 20). Pure python.
Populations: test -> primary = held-out-floor-plan games (src_split train) + valid_unseen; valid_seen separately;
val -> all games (all from train floor plans). For each population:
  primary-type estimand S_flip@first (learnable, own flips, one dp per game, game bootstrap), unlearnable, Res@first;
  covariates: note-writing rate in free greedy rollouts, note top-1 hit, P(intact top|intact), P(conv|remove), success;
  Fix 2b presence measures: PE = P(intact top|remove) - P(intact top|presence),
     S_flip_vs_presence = P(foil top|flip) - P(foil top|presence), follow-normalised S_flip = S_flip / P(intact top|intact);
  Fix 3 copying characterisation: S_random@first, plausibility slope (most - least plausible tercile), removal null
     P(intact top|intact) - P(intact top|remove), cf rollouts (P first visit = foil top; visits cf_foil - cf_true);
  later positions (by p and convention status) and the 2x2 (earlier notes kept/removed x intact/flip) at p >= 1.
usage: python alfcf_eval_v3_analysis.py <run_dir> <alfcf_dir> <split> <out.json>
"""
import glob
import json
import os
import random
import sys
from collections import defaultdict

RUN, ALF, SPLIT, OUT = sys.argv[1:5]


def rate(d, cond, typ):
    o = d["outcomes"].get(cond)
    return None if not o or typ is None else sum(t == typ for t in o) / len(o)


def mean(v):
    v = [x for x in v if x is not None]
    return sum(v) / len(v) if v else float("nan")


def boot(rows, fn, B=2000, seed=0):
    by = defaultdict(list)
    for r in rows:
        by[r["gid"]].append(r)
    g = sorted(by)
    if not g:
        return [float("nan")] * 3 + [0]
    est = fn(rows)
    rng = random.Random(seed)
    vals = sorted(fn([r for x in (rng.choice(g) for _ in g) for r in by[x]]) for _ in range(B))
    return [est, vals[int(0.025 * B)], vals[int(0.975 * B) - 1], len(rows)]


def eff(cond, base, key, base_key=None):
    base_key = base_key or key

    def f(R):
        R = [d for d in R if d["outcomes"].get(cond) and d["outcomes"].get(base) and d.get(key)]
        return mean([rate(d, cond, d[key]) for d in R]) - mean([rate(d, base, d[base_key]) for d in R]) if R else float("nan")
    return f


def block(dps, rolls, cfr, meta):
    res = {}
    turns = [t for r in rolls for t in r["turns"]]
    res["n_games"] = len(rolls)
    res["success"] = mean([r["won"] for r in rolls])
    res["note_writing_rate"] = mean([t["note"] is not None for t in turns if t["is_dp"]])
    res["format_compliance"] = mean([t["fmt_ok"] for t in turns])
    res["fallback_dps_per_game"] = sum(t["is_dp"] for t in turns) / max(1, len(rolls))
    P0 = [d for d in dps if d["p"] == 0]
    for cl, sel in [("L", True), ("U", False)]:
        rows = [d for d in P0 if d["learnable"] == sel]
        fl = [d for d in rows if d.get("flip")]
        b = {"n_games_first_dp": len(rows), "n_with_valid_own_flip": len(fl), "excluded_no_valid_flip": len(rows) - len(fl),
             "S_flip": boot(fl, eff("flip", "intact", "flip_top")),
             "P_fliptop_flip": mean([rate(d, "flip", d["flip_top"]) for d in fl]),
             "P_fliptop_intact": mean([rate(d, "intact", d["flip_top"]) for d in fl]),
             "P_fliptop_presence": mean([rate(d, "presence", d["flip_top"]) for d in fl]),
             "S_flip_vs_presence": boot(fl, eff("flip", "presence", "flip_top")),
             "follow_intact": mean([rate(d, "intact", d["intact_top"]) for d in rows]),
             "P_intacttop_remove": mean([rate(d, "remove", d["intact_top"]) for d in rows]),
             "P_intacttop_presence": mean([rate(d, "presence", d["intact_top"]) for d in rows]),
             "presence_effect": boot(rows, lambda R: mean([rate(d, "remove", d["intact_top"]) for d in R])
                                     - mean([rate(d, "presence", d["intact_top"]) for d in R])),
             "removal_null": boot(rows, lambda R: mean([rate(d, "intact", d["intact_top"]) for d in R])
                                  - mean([rate(d, "remove", d["intact_top"]) for d in R])),
             "note_top1_hit": mean([d["note_hit"] for d in rows]),
             "S_random": boot([d for d in rows if d.get("random")], eff("random", "intact", "random_top")),
             "placebo_effect": boot(rows, eff("placebo", "intact", "placebo_top")),
             "S_flip_tmpl": boot([d for d in fl if d.get("tmpl_top")], eff("flip_tmpl", "intact", "tmpl_top")),
             "own_minus_tmpl_paired": boot([d for d in fl if d.get("tmpl_top")],
                                           lambda R: eff("flip", "intact", "flip_top")(R) - eff("flip_tmpl", "intact", "tmpl_top")(R))}
        fi = b["follow_intact"]
        b["S_flip_follow_normalised"] = b["S_flip"][0] / fi if fi and fi == fi else float("nan")
        if sel:
            rc = [d for d in rows if d["conv"]]
            b["P_conv"] = {c: mean([rate(d, c, d["conv"]) for d in rc]) for c in
                           ["intact", "flip", "remove", "placebo", "random", "presence"]}
            b["P_conv_n"] = len(rc)
            terc = sorted(d["foil_plausibility"] for d in fl)
            if len(terc) >= 6:
                c1, c2 = terc[len(terc) // 3], terc[2 * len(terc) // 3]
                most = boot([d for d in fl if d["foil_plausibility"] < c1], eff("flip", "intact", "flip_top"))
                least = boot([d for d in fl if d["foil_plausibility"] >= c2], eff("flip", "intact", "flip_top"))
                b["plausibility"] = {"most": most, "least": least, "slope_most_minus_least": most[0] - least[0], "cuts": [c1, c2]}
        res[f"first_{cl}"] = b
    fL = [d for d in P0 if d["learnable"] and d.get("flip")]
    fU = [d for d in P0 if not d["learnable"] and d.get("flip")]
    f = eff("flip", "intact", "flip_top")
    rng = random.Random(1)
    bs = sorted(f([rng.choice(fU) for _ in fU]) - f([rng.choice(fL) for _ in fL]) for _ in range(2000)) if fL and fU else []
    res["Res_first_U_minus_L"] = [f(fU) - f(fL), bs[50], bs[1949]] if bs else [float("nan")] * 3
    later = {}
    for cl, sel in [("L", True), ("U", False)]:
        for pk, ps in [("p1", {1}), ("p2", {2}), ("p3+", None)]:
            rows = [d for d in dps if d["learnable"] == sel and d["p"] >= 1 and d.get("flip") and (d["p"] in ps if ps else d["p"] >= 3)]
            later[f"{cl}|{pk}"] = boot(rows, f)
        if sel:
            for st in ["unvisited", "partial", "refuted"]:
                later[f"L|status={st}"] = boot([d for d in dps if d["learnable"] and d["p"] >= 1 and d.get("flip")
                                                and d.get("conv_status") == st], f)
    res["later_S_flip"] = later
    two = {}
    for cl, sel in [("L", True), ("U", False)]:
        rows = [d for d in dps if d["learnable"] == sel and d["p"] >= 1 and d.get("flip")
                and d["outcomes"].get("nh_flip") and d["outcomes"].get("nh_intact")]
        two[cl] = {"n": len(rows), "S_flip_kept": boot(rows, f),
                   "S_flip_notes_removed": boot(rows, eff("nh_flip", "nh_intact", "flip_top")),
                   "removed_minus_kept": boot(rows, lambda R: eff("nh_flip", "nh_intact", "flip_top")(R) - f(R))}
    res["later_2x2"] = two
    cf = {}
    for cl, sel in [("L", True), ("U", False)]:
        for mode in ["cf_foil", "cf_true"]:
            R = [r for r in cfr if r["mode"] == mode and r["learnable"] == sel]
            cf[f"{mode}|{cl}"] = {"n": len(R), "P_first_visit_eq_foil_top": mean([r["first_visit_type"] == r["foil_top0"] for r in R]),
                                  "fallback_visits": mean([r["n_fallback_visits"] for r in R]), "success": mean([r["won"] for r in R])}
        a, b2 = cf[f"cf_foil|{cl}"]["fallback_visits"], cf[f"cf_true|{cl}"]["fallback_visits"]
        cf[f"visits_foil_minus_true|{cl}"] = a - b2
    res["counterfactual_rollouts"] = cf
    return res


def main():
    rolls = [json.loads(l) for f in sorted(glob.glob(os.path.join(RUN, "rollouts.shard*.jsonl"))) for l in open(f)]
    dps = [json.loads(l) for f in sorted(glob.glob(os.path.join(RUN, "interventions.shard*.jsonl"))) for l in open(f)]
    cfr = [json.loads(l) for f in sorted(glob.glob(os.path.join(RUN, "cf_rollouts.shard*.jsonl"))) for l in open(f)]
    meta = {r["gid"]: json.load(open(os.path.join(ALF, SPLIT, r["gid"], "meta.json"))) for r in rolls}
    for d in dps:
        d["flip_top"] = d["flip_types"][0] if d.get("flip_types") else None
        d["intact_top"] = d["intact_types"][0]
        d["random_top"] = d["random_types"][0] if d.get("random_types") else None
        d["placebo_top"] = d["placebo_types"][0]
        d["tmpl_top"] = d["flip_tmpl_types"][0] if (d.get("flip_tmpl") and d.get("flip_tmpl_types")) else None
        m = meta[d["gid"]]
        hit = {m["receps"][r] for r in m["hits"] if not r.startswith("?")}
        d["note_hit"] = d["intact_top"] in hit
        d["conv"] = d["conv_type"] if (m["moved"] and d.get("conv_type") in d["remaining"]) else None
    pops = {"all": lambda g: True}
    if SPLIT == "test":
        pops = {"primary": lambda g: meta[g]["src_split"] in ("train", "valid_unseen"),
                "valid_seen": lambda g: meta[g]["src_split"] == "valid_seen", "all": lambda g: True}
    res = {"split": SPLIT}
    for name, sel in pops.items():
        res[name] = block([d for d in dps if sel(d["gid"])], [r for r in rolls if sel(r["gid"])],
                          [r for r in cfr if sel(r["gid"])], meta)
    json.dump(res, open(OUT, "w"), indent=1)
    key = "primary" if SPLIT == "test" else "all"
    b = res[key]
    print(json.dumps({k: v for k, v in b.items() if not isinstance(v, dict)}, indent=1))
    print("first_L", json.dumps({k: v for k, v in b["first_L"].items()}))


if __name__ == "__main__":
    main()
