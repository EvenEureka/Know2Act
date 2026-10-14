"""ALF-CF v1 generation rules (Know2Act, PREREG Addendum 18, step 1). CPU only, deterministic (seed 20261013).

Output (alfcf/rules_v1.json):
  * floor-plan split of the ALFWorld train set per scene type: 21 train / 3 val / 3 test floor plans (seeded);
    valid_seen and valid_unseen go to test (valid_seen rooms are train floor plans; reported separately).
    Variants per base game (independent per-game draws): train 3, val 1, test 2.
  * learnable / unlearnable target classes: 12 of 49 classes (~25%) unlearnable, seeded draw accepted only if
    their share of train games is in [0.20, 0.30] and every scene type has one with >= 20 train games.
  * for each learnable (scene type, class): a primary convention class r1 and a secondary r2, chosen greedily
    (classes in descending train-game count) by the first level of LADDER that has a candidate, subject to
      - never one of the class's top-3 ALFRED types (type share) and destination share < 0.10;
      - LADDER level: floor-plan presence, normalised ALFRED data rank, min over Qwen3-4B/8B of the mean
        normalised zero-shot rank (FT4 responses; per-scene object-agnostic rank when < 3 probes);
      - USAGE_CAP: no r1 class may hold more than 30% of the scene type's learnable train games, so that an
        object-agnostic heuristic cannot recover the convention;
    and a cost that prefers low data/model rank, low usage and few instances. r2 is the cheapest other class
    meeting the same level with presence >= 0.9 (fallbacks logged).
  * per-game placement (alfcf_generate.py), PI = 0.8: with prob PI every instance of the target class moves into
    ONE uniformly drawn instance of r1 (if r1 is in the room and is not the task destination class), else of r2
    (same conditions), else the original placement is kept ("no_slot"); with prob 1 - PI the original placement
    is kept. Unlearnable classes: with prob PI every instance moves into one instance of a receptacle class drawn
    uniformly per game from the classes present in the room (destination class excluded).
usage: python alfcf_rules.py <out.json>
"""
import json
import os
import random
import re
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import alfcf_common as C  # noqa: E402

PI = 0.8
UNLEARNABLE_SHARE = 0.25
SEED = 20261013
P_R2 = 0.90                      # floor-plan presence of the secondary class r2
MAX_DEST_SHARE = 0.10            # destination share of the class's tasks must be < this (both r1 and r2)
# relaxation ladder for r1 (first level with a candidate is used; the level is logged per class):
#   pres = floor-plan presence,
#   model = min over Qwen3-4B/8B of the mean normalised zero-shot rank (0 = ranked first)
#   data = min normalised rank of the type in the class's ALFRED type-share ranking (0 = most common type),
#          ranked among the scene type's classes present in >= 40% of its floor plans; never a top-3 type
LADDER = [dict(pres=0.60, data=0.50, model=0.40), dict(pres=0.60, data=0.40, model=0.30),
          dict(pres=0.40, data=0.40, model=0.30), dict(pres=0.40, data=0.30, model=0.20),
          dict(pres=0.40, data=0.20, model=0.10)]
USAGE_CAP = 0.30                 # no r1 class may hold more than 30% of a scene type's learnable train games
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(C.K2A, "alfcf", "rules_v1.json")


def parse_ranking(resp, types):  # identical to ft4_score.parse
    txt = resp.split("</think>")[-1].strip().lower()
    line = [l for l in txt.splitlines() if l.strip()]
    line = line[0] if line else ""
    seen = []
    for tok in re.split(r"[,;\n]|\band\b|>", line):
        tok = re.sub(r"[^a-z ]", "", tok).strip()
        tok = re.sub(r"\s+\d+$", "", tok).replace(" ", "")
        cand = None
        if tok in types:
            cand = tok
        elif tok.endswith("s") and tok[:-1] in types:
            cand = tok[:-1]
        elif tok.endswith("es") and tok[:-2] in types:
            cand = tok[:-2]
        if cand and cand not in seen:
            seen.append(cand)
    return seen


def model_ranks():
    """mean normalised rank of type t for (scene, o), per model; plus per-scene object-agnostic fallback."""
    probes = {}
    for l in open(os.path.join(C.K2A, "results", "ft4_probes.jsonl")):
        p = json.loads(l)
        probes[p["idx"]] = p
    out = {}
    for tag in ["qwen3-4b", "qwen3-8b"]:
        acc = defaultdict(list)
        acc_sc = defaultdict(list)
        for l in open(os.path.join(C.K2A, "results", f"ft4_resp_{tag}.jsonl")):
            r = json.loads(l)
            p = probes[r["idx"]]
            types = list(p["n"])
            if len(types) < 2:
                continue
            ranked = parse_ranking(r["response"], types)
            missing = [t for t in types if t not in ranked]
            pos = {t: i for i, t in enumerate(ranked)}
            # unlisted types share the mean of the remaining positions
            mid = (len(ranked) + len(types) - 1) / 2
            for t in missing:
                pos[t] = mid
            for t in types:
                nr = pos[t] / (len(types) - 1)
                acc[(p["scene"], p["target"], t)].append(nr)
                acc_sc[(p["scene"], t)].append(nr)
        out[tag] = ({k: (sum(v) / len(v), len(v)) for k, v in acc.items()},
                    {k: sum(v) / len(v) for k, v in acc_sc.items()})
    return out


def main():
    rng = random.Random(SEED)
    S = {s: C.load_games(s) for s in ["train", "valid_seen", "valid_unseen"]}
    train = S["train"]

    # ---------------- floor-plan split
    split_of_floor = {}
    for sc in C.SCENES:
        fls = sorted({g["floor"] for g in train if g["scene"] == sc}, key=lambda x: int(re.sub(r"\D", "", str(x))))
        r = random.Random(C.stable_seed(SEED, "floors", sc))
        r.shuffle(fls)
        for i, f in enumerate(fls):
            split_of_floor[f] = "train" if i < 21 else ("val" if i < 24 else "test")

    # ---------------- statistics used by the rule (standard ALFWorld train set = environment prior)
    share_std = C.type_share(train)
    pres = {}
    inst = defaultdict(list)
    for sc in C.SCENES:
        floors = defaultdict(set)
        nfl = len({g["floor"] for g in train if g["scene"] == sc})
        for g in train:
            if g["scene"] != sc:
                continue
            for t, k in Counter(g["receps"].values()).items():
                floors[t].add(g["floor"])
                inst[(sc, t)].append(k)
        pres[sc] = {t: len(v) / nfl for t, v in floors.items()}
    mean_inst = {k: sum(v) / len(v) for k, v in inst.items()}
    dest_share = defaultdict(Counter)
    n_games = Counter()
    for g in train:
        n_games[(g["scene"], g["target"])] += 1
        if g["dest"]:
            dest_share[(g["scene"], g["target"])][g["dest"]] += 1
    dest_share = {k: {t: c / n_games[k] for t, c in v.items()} for k, v in dest_share.items()}
    MR = model_ranks()

    def mrank(tag, sc, o, t):
        per, scn = MR[tag]
        v = per.get((sc, o, t))
        if v and v[1] >= 3:
            return v[0], "class"
        return scn.get((sc, t), 0.5), "scene_fallback"

    # ---------------- learnable / unlearnable classes
    classes = sorted({g["target"] for g in train})
    games_per_class = Counter(g["target"] for g in train)
    n_un = round(UNLEARNABLE_SHARE * len(classes))
    attempt = 0
    while True:
        r = random.Random(C.stable_seed(SEED, "unlearnable", attempt))
        un = set(r.sample(classes, n_un))
        share = sum(games_per_class[c] for c in un) / len(train)
        ok_scene = all(any(n_games[(sc, c)] >= 20 for c in un) for sc in C.SCENES)
        if 0.20 <= share <= 0.30 and ok_scene:
            break
        attempt += 1

    # ---------------- convention assignment, learnable (scene, class) pairs
    conv = {}
    log = {}
    for sc in C.SCENES:
        pairs = sorted([(n, o) for (s, o), n in n_games.items() if s == sc and o not in un], reverse=True)
        tot = sum(n for n, _ in pairs)
        usage = Counter()
        for n, o in pairs:
            sh = share_std.get((sc, o), {})
            top3 = [t for t, _ in sorted(sh.items(), key=lambda kv: -kv[1])[:3]]
            ds = dest_share.get((sc, o), {})
            common = sorted([t for t, pr in pres[sc].items() if pr >= 0.4], key=lambda t: (-sh.get(t, 0.0), t))
            # normalised data rank with ties averaged; types outside `common` count as least common (rank 1)
            drank = {}
            for t in common:
                tied = [i for i, u in enumerate(common) if abs(sh.get(u, 0.0) - sh.get(t, 0.0)) < 1e-9]
                drank[t] = (sum(tied) / len(tied)) / max(1, len(common) - 1)
            rows = []
            for t, pr in pres[sc].items():
                m4, src4 = mrank("qwen3-4b", sc, o, t)
                m8, src8 = mrank("qwen3-8b", sc, o, t)
                rows.append(dict(t=t, presence=round(pr, 3), data_share=round(sh.get(t, 0.0), 3),
                                 data_rank=round(drank.get(t, 1.0), 3), data_top3=t in top3, rank4=round(m4, 3), rank8=round(m8, 3),
                                 rank_src=src4 + "/" + src8, dest_share=round(ds.get(t, 0.0), 3),
                                 mean_inst=round(mean_inst[(sc, t)], 2)))

            def cost(row):
                return (4.0 * usage[row["t"]] / max(tot, 1) + 2.0 * (1 - row["data_rank"])
                        + 2.0 * (1 - min(row["rank4"], row["rank8"])) + 0.1 * row["mean_inst"])

            def ok(x, L, pres_min):
                return (x["presence"] >= pres_min and not x["data_top3"] and x["data_rank"] >= L["data"]
                        and min(x["rank4"], x["rank8"]) >= L["model"] and x["dest_share"] < MAX_DEST_SHARE)

            level, capped = None, True
            for li, L in enumerate(LADDER):
                c1 = [x for x in rows if ok(x, L, L["pres"])
                      and (usage[x["t"]] == 0 or usage[x["t"]] + n <= USAGE_CAP * tot)]
                if c1:
                    level = li
                    break
            if level is None:  # no candidate under the usage cap at any level: last level, cap ignored (logged)
                level, capped = len(LADDER) - 1, False
                c1 = [x for x in rows if ok(x, LADDER[-1], LADDER[-1]["pres"])]
            if not c1:
                print(json.dumps(rows, indent=0))
                raise RuntimeError(f"no candidates for {sc}/{o}")
            L = LADDER[level]
            r1 = min(c1, key=lambda x: (cost(x), x["t"]))
            c2 = [x for x in rows if ok(x, L, P_R2) and x["t"] != r1["t"]]
            r2_rule = "same_level_presence_0.9"
            if not c2:
                c2 = [x for x in rows if ok(x, L, L["pres"]) and x["t"] != r1["t"]]
                r2_rule = "same_level_any_presence"
            if not c2:
                c2 = [x for x in rows if x["presence"] >= P_R2 and x["t"] != r1["t"] and not x["data_top3"]
                      and x["dest_share"] < MAX_DEST_SHARE]
                r2_rule = "fallback_presence_0.9_not_top3"
            if not c2:
                c2 = [x for x in rows if x["presence"] >= 0.6 and x["t"] != r1["t"] and x["dest_share"] < MAX_DEST_SHARE]
                r2_rule = "fallback_presence_0.6_any"
            r2 = min(c2, key=lambda x: (cost(x), x["t"]))
            usage[r1["t"]] += n
            conv[f"{sc}|{o}"] = {"r1": r1["t"], "r2": r2["t"], "train_games": n, "ladder_level": level,
                                 "usage_cap_respected": capped,
                                 "r2_rule": r2_rule,
                                 "r1_stats": {k: r1[k] for k in ["presence", "data_share", "data_rank", "rank4", "rank8",
                                                                  "dest_share", "mean_inst"]},
                                 "r2_stats": {k: r2[k] for k in ["presence", "data_share", "data_rank", "rank4", "rank8",
                                                                  "dest_share", "mean_inst"]}}
            log[f"{sc}|{o}"] = rows

    # ---------------- summaries
    by_scene = {sc: Counter(v["r1"] for k, v in conv.items() if k.startswith(sc + "|")) for sc in C.SCENES}
    split_counts = Counter()
    for s, gs in S.items():
        for g in gs:
            sp = "test" if s != "train" else split_of_floor[g["floor"]]
            split_counts[(sp, s)] += 1
    res = dict(version="alfcf_v1", date="2026-10-13", PI=PI, seed=SEED,
               thresholds=dict(P_R2=P_R2, MAX_DEST_SHARE=MAX_DEST_SHARE, ladder=LADDER, usage_cap=USAGE_CAP),
               variants=dict(train=3, val=1, test=2),
               split_of_floor={str(k): v for k, v in split_of_floor.items()},
               split_counts={f"{a}|{b}": c for (a, b), c in sorted(split_counts.items())},
               unlearnable=sorted(un), unlearnable_draw_attempt=attempt,
               unlearnable_train_game_share=round(sum(games_per_class[c] for c in un) / len(train), 4),
               n_classes=len(classes), learnable=sorted(set(classes) - un),
               convention=conv, r1_usage_by_scene={k: dict(v) for k, v in by_scene.items()},
               candidate_log=log)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(res, open(OUT, "w"), indent=1)
    print("unlearnable", sorted(un), "share", res["unlearnable_train_game_share"], "attempt", attempt)
    print("split counts", res["split_counts"])
    for sc in C.SCENES:
        print(sc, dict(by_scene[sc]))
        for k, v in conv.items():
            if k.startswith(sc + "|"):
                print("   ", k, v)


if __name__ == "__main__":
    main()
