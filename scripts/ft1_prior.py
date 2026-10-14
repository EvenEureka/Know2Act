"""FT1: location-prior entropy, search cost under several priors, critical-decision fraction.
Decision-value statistics of standard ALFWorld (written before computing; see protocol/).
Read-only over the ALFWorld json_2.1.1 data (env ALFWORLD_JSON).
"""
import os, re, json, math, random, sys
from collections import defaultdict, Counter

DATA = os.environ.get("ALFWORLD_JSON", "data/alfworld/json_2.1.1")  # ALFWorld json_2.1.1 games
OUT = sys.argv[1] if len(sys.argv) > 1 else "ft1_results.json"
TASK_TYPES = ["pick_and_place_simple", "look_at_obj_in_light", "pick_clean_then_place_in_recep",
              "pick_heat_then_place_in_recep", "pick_cool_then_place_in_recep", "pick_two_obj_and_place"]

import alfworld.gen.constants as K
VALID = defaultdict(set)  # object class (lower) -> set of receptacle classes (lower)
for r, objs in K.VAL_RECEPTACLE_OBJECTS.items():
    for o in objs:
        VALID[o.lower()].add(r.lower())

RE_RT = re.compile(r"\(receptacleType\s+(\S+)\s+(\S+)Type\)")
RE_OT = re.compile(r"\(objectType\s+(\S+)\s+(\S+)Type\)")
RE_IN = re.compile(r"\(inReceptacle\s+(\S+)\s+(\S+)\)")


def scene_type(fp):
    n = int(re.sub(r"\D", "", str(fp)))
    return {0: "kitchen", 2: "living", 3: "bedroom", 4: "bathroom"}.get(n // 100, "other")


def load_split(split):
    games = []
    root = os.path.join(DATA, split)
    for dp, dns, fns in os.walk(root):
        if "traj_data.json" not in fns:
            continue
        if "movable" in dp or "Sliced" in dp:
            continue
        tj = json.load(open(os.path.join(dp, "traj_data.json")))
        if tj["task_type"] not in TASK_TYPES:
            continue
        gp = os.path.join(dp, "game.tw-pddl")
        if not os.path.exists(gp):
            continue
        gd = json.load(open(gp))
        if not gd.get("solvable"):
            continue
        pddl = open(os.path.join(dp, "initial_state.pddl")).read()
        rtype = {i: t.lower() for i, t in RE_RT.findall(pddl)}
        otype = {i: t.lower() for i, t in RE_OT.findall(pddl)}
        parent = {o: r for o, r in RE_IN.findall(pddl)}
        tgt = tj["pddl_params"]["object_target"].lower()
        # top-level receptacle of each target instance
        hits = set()
        for oid, ot in otype.items():
            if ot != tgt:
                continue
            cur, seen = oid, 0
            while cur in parent and cur not in rtype and seen < 5:
                cur = parent[cur]; seen += 1
            if cur in rtype:
                hits.add(cur)
        # expert walkthrough: go to / open before first take of target
        wt = gd.get("walkthrough") or []
        n_search = 0
        for a in wt:
            if a.startswith("take ") and a.split()[1] == tgt:
                break
            if a.startswith("go to") or a.startswith("open "):
                n_search += 1
        games.append(dict(path=dp, split=split, task_type=tj["task_type"], target=tgt,
                          floor=tj["scene"]["floor_plan"], scene=scene_type(tj["scene"]["floor_plan"]),
                          receps=rtype, hits=sorted(hits), n_expert_search=n_search,
                          n_walk=len(wt)))
    return games


def type_counts(g):
    n = Counter(g["receps"].values())
    h = Counter(g["receps"][r] for r in g["hits"])
    return n, h


def fit_prior(games, exclude_floor=None):
    """p[o][t] = P(instance of type t contains o | o target, t present); q[t] agnostic."""
    num = defaultdict(Counter); den = defaultdict(Counter)
    qn = Counter(); qd = Counter()
    for g in games:
        if exclude_floor is not None and g["floor"] == exclude_floor:
            continue
        n, h = type_counts(g)
        for t, c in n.items():
            num[g["target"]][t] += h.get(t, 0); den[g["target"]][t] += c
            qn[t] += h.get(t, 0); qd[t] += c
    p = {o: {t: num[o][t] / den[o][t] for t in den[o]} for o in den}
    q = {t: qn[t] / qd[t] for t in qd}
    return p, q


def groups_from_scores(types, score):
    """types present -> list of groups (lists of types) in descending score; ties merged."""
    by = defaultdict(list)
    for t in types:
        by[round(score.get(t, 0.0), 12)].append(t)
    return [by[s] for s in sorted(by, reverse=True)]


def exp_cost(groups, n, h):
    """expected #visits to first hit; uniform random order within each group of types."""
    cost = 0.0
    for grp in groups:
        sz = sum(n[t] for t in grp); hh = sum(h.get(t, 0) for t in grp)
        if hh > 0:
            return cost + (sz + 1) / (hh + 1), hh / sz
        cost += sz
    return float("nan"), 0.0


def top1(groups, n, h):
    grp = groups[0]
    sz = sum(n[t] for t in grp); hh = sum(h.get(t, 0) for t in grp)
    return hh / sz


def policies(g, p, q):
    n, h = type_counts(g)
    types = list(n)
    o = g["target"]
    po = p.get(o, {})
    pol = {}
    pol["U"] = [types]
    val = [t for t in types if t in VALID.get(o, set())]
    rest = [t for t in types if t not in VALID.get(o, set())]
    pol["V"] = [x for x in (val, rest) if x]
    pol["M"] = groups_from_scores(types, q)
    # C: conditional prior; unseen (o,t) pairs fall back to 0
    pol["C"] = groups_from_scores(types, po)
    # C|V: conditional prior restricted to ALFRED valid classes first (sanity variant)
    return pol, n, h


def sample_path(groups, n, hits_set, recep_types, rng):
    """sample an instance visiting order consistent with groups; return list of visited types until hit."""
    order = []
    inst_by_type = defaultdict(list)
    for r, t in recep_types.items():
        inst_by_type[t].append(r)
    for grp in groups:
        insts = [r for t in grp for r in inst_by_type[t]]
        rng.shuffle(insts)
        order += insts
    path = []
    for r in order:
        path.append(r)
        if r in hits_set:
            break
    return path


def argmax_set(types, score):
    if not types:
        return set()
    m = max(score.get(t, 0.0) for t in types)
    return {t for t in types if abs(score.get(t, 0.0) - m) < 1e-12}


def decision_analysis(g, p, q, foil_pool, rng, n_paths=20, n_foils=20):
    pol, n, h = policies(g, p, q)
    po = p.get(g["target"], {})
    hits_set = set(g["hits"])
    out = dict(dp=0, critA=0, critA_strong=0, flip=0.0, dp_all=0)
    foils = [f for f in foil_pool if f != g["target"]]
    if len(foils) > n_foils:
        foils = rng.sample(foils, n_foils)
    for _ in range(n_paths):
        path = sample_path(pol["C"], n, hits_set, g["receps"], rng)
        visited = set()
        for r in path:
            remaining_types = {g["receps"][x] for x in g["receps"] if x not in visited}
            out["dp_all"] += 1
            if len(remaining_types) >= 2:
                out["dp"] += 1
                aC = argmax_set(remaining_types, po)
                aM = argmax_set(remaining_types, q)
                if not (aC & aM):
                    out["critA"] += 1
                    best_c = max(po.get(t, 0) for t in aC)
                    best_m = max(po.get(t, 0) for t in aM)
                    if best_c - best_m >= 0.10:
                        out["critA_strong"] += 1
                if foils:
                    fl = 0
                    for f in foils:
                        aF = argmax_set(remaining_types, p.get(f, {}))
                        if not (aF & aC):
                            fl += 1
                    out["flip"] += fl / len(foils)
            visited.add(r)
    for k in out:
        out[k] /= n_paths
    return out


def entropy(c):
    s = sum(c.values())
    return -sum(v / s * math.log2(v / s) for v in c.values() if v > 0)


def main():
    rng = random.Random(0)
    S = {s: load_split(s) for s in ["train", "valid_seen", "valid_unseen"]}
    res = {"n_games": {s: len(v) for s, v in S.items()}}
    train = S["train"]
    # ---- entropy over train: type containing target | object class
    cond = defaultdict(Counter); marg = Counter(); objc = Counter()
    for g in train:
        n, h = type_counts(g)
        tot = sum(h.values())
        if tot == 0:
            continue
        ts = [t for t in h if h[t] > 0]
        for t in ts:
            cond[g["target"]][t] += 1 / len(ts); marg[t] += 1 / len(ts)
        objc[g["target"]] += 1
    N = sum(objc.values())
    HT = entropy(marg)
    HTO = sum(objc[o] / N * entropy(cond[o]) for o in cond)
    norm = []
    for o in cond:
        k = len(VALID.get(o, [])) or len(cond[o])
        if k > 1:
            norm.append((objc[o], entropy(cond[o]) / math.log2(k)))
    res["entropy_train"] = dict(H_T=HT, H_T_given_O=HTO, I_O_T=HT - HTO,
                                mean_norm_H_by_valid=sum(w * x for w, x in norm) / sum(w for w, _ in norm),
                                n_object_classes=len(cond))
    # per-object top types for the record
    res["top_types_by_object"] = {o: [(t, round(v / objc[o], 3)) for t, v in cond[o].most_common(4)]
                                  for o, _ in objc.most_common(40)}
    # foil pools by scene type (target classes seen in train for that scene type)
    foil_pool = defaultdict(set)
    for g in train:
        foil_pool[g["scene"]].add(g["target"])
    foil_pool = {k: sorted(v) for k, v in foil_pool.items()}

    def evaluate(games, prior_fn, tag):
        agg = defaultdict(list); per = []
        for g in games:
            p, q = prior_fn(g)
            pol, n, h = policies(g, p, q)
            if not g["hits"]:
                agg["no_hit"].append(1); continue
            row = dict(task_type=g["task_type"], scene=g["scene"], target=g["target"],
                       N=sum(n.values()), n_types=len(n), H=len(g["hits"]),
                       n_expert_search=g["n_expert_search"], n_walk=g["n_walk"])
            for k, grp in pol.items():
                c, _ = exp_cost(grp, n, h)
                row[f"cost_{k}"] = c; row[f"top1_{k}"] = top1(grp, n, h)
            da = decision_analysis(g, p, q, foil_pool.get(g["scene"], []), rng)
            row.update({f"da_{k}": v for k, v in da.items()})
            per.append(row)
        def m(key, rows=per):
            v = [r[key] for r in rows if not (isinstance(r[key], float) and math.isnan(r[key]))]
            return sum(v) / len(v) if v else float("nan")
        def summarize(rows):
            s = {"n": len(rows)}
            for k in ["U", "V", "M", "C"]:
                s[f"E_cost_{k}"] = m(f"cost_{k}", rows); s[f"top1_{k}"] = m(f"top1_{k}", rows)
            s["Delta_M"] = s["E_cost_M"] - s["E_cost_C"]; s["Delta_V"] = s["E_cost_V"] - s["E_cost_C"]
            dp = sum(r["da_dp"] for r in rows)
            s["decision_points_per_game_C"] = sum(r["da_dp_all"] for r in rows) / len(rows)
            s["CDF_A"] = sum(r["da_critA"] for r in rows) / dp
            s["CDF_A_strong"] = sum(r["da_critA_strong"] for r in rows) / dp
            s["CDF_flip"] = sum(r["da_flip"] for r in rows) / dp
            s["mean_N_receps"] = m("N", rows); s["mean_types"] = m("n_types", rows)
            s["mean_target_instances_receps"] = m("H", rows)
            s["expert_search_actions"] = m("n_expert_search", rows)
            s["expert_walk_len"] = m("n_walk", rows)
            return s
        out = {"all": summarize(per)}
        for key in ["task_type", "scene"]:
            for val in sorted({r[key] for r in per}):
                out[f"{key}={val}"] = summarize([r for r in per if r[key] == val])
        out["no_hit_games"] = len(agg["no_hit"])
        return out, per

    full_prior = fit_prior(train)
    valid = S["valid_seen"] + S["valid_unseen"]
    res["valid_all"], per_valid = evaluate(valid, lambda g: full_prior, "valid")
    res["valid_seen"], _ = evaluate(S["valid_seen"], lambda g: full_prior, "vs")
    res["valid_unseen"], _ = evaluate(S["valid_unseen"], lambda g: full_prior, "vu")
    lofo_cache = {}
    def lofo(g):
        f = g["floor"]
        if f not in lofo_cache:
            lofo_cache[f] = fit_prior(train, exclude_floor=f)
        return lofo_cache[f]
    res["train_lofo"], _ = evaluate(train, lofo, "train")
    json.dump(res, open(OUT, "w"), indent=1)
    json.dump(per_valid, open(OUT.replace(".json", "_per_valid.json"), "w"))
    for k in ["valid_all", "valid_seen", "valid_unseen", "train_lofo"]:
        a = res[k]["all"]
        print(k, {x: (round(y, 3) if isinstance(y, float) else y) for x, y in a.items()})
    print("entropy", res["entropy_train"], "n_games", res["n_games"])


if __name__ == "__main__":
    main()
