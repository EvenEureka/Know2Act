"""Generate ALF-CF v1 games from the frozen rules (alfcf/rules_v1.json). CPU, deterministic.

For each base game and each variant (train 3, val 1, test 2) it draws, with a per-(game, variant) seed:
  moved ~ Bernoulli(PI); the convention class (learnable: r1, else r2, else no_slot; unlearnable: uniform over the
  classes present, destination excluded); and one uniformly drawn instance of that class. Every instance of the
  target class is moved there by editing only the :init block (FT2 fixes: the original file's embedded domain and
  grammar are kept; the destination class is never used). Writes <out>/<split>/<id>/game.tw-pddl and meta.json,
  plus <out>/index.jsonl. Never writes to the source data.
usage: python alfcf_generate.py <rules.json> <out_dir>
"""
import json
import os
import random
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import alfcf_common as C  # noqa: E402

rules = json.load(open(sys.argv[1]))
OUT = sys.argv[2]
PI = rules["PI"]
UN = set(rules["unlearnable"])
CONV = rules["convention"]
NV = rules["variants"]


def main():
    S = {s: C.load_games(s) for s in ["train", "valid_seen", "valid_unseen"]}
    idx = open(os.path.join(OUT, "index.jsonl"), "w") if os.path.isdir(OUT) else None
    if idx is None:
        os.makedirs(OUT)
        idx = open(os.path.join(OUT, "index.jsonl"), "w")
    stats = Counter()
    for src_split, games in S.items():
        for g in games:
            split = "test" if src_split != "train" else rules["split_of_floor"][str(g["floor"])]
            gd0 = json.load(open(os.path.join(g["path"], "game.tw-pddl")))
            pddl0 = open(os.path.join(g["path"], "initial_state.pddl")).read()
            types_present = sorted(set(g["receps"].values()))
            learnable = g["target"] not in UN
            for v in range(NV[split]):
                rng = random.Random(C.stable_seed("alfcf_v1", g["rel"], v))
                moved = rng.random() < PI
                cls, slot_rule = None, None
                if learnable:
                    cv = CONV[f'{g["scene"]}|{g["target"]}']
                    for key in ("r1", "r2"):
                        if cv[key] in types_present and cv[key] != g["dest"]:
                            cls, slot_rule = cv[key], key
                            break
                    if cls is None:
                        slot_rule = "no_slot"
                else:
                    cands = [t for t in types_present if t != g["dest"]]
                    cls, slot_rule = rng.choice(cands), "uniform"
                draw_inst = rng.random()  # always consumed, so draws do not depend on `moved`
                new_r, n_inst = None, 0
                pddl = pddl0
                if moved and cls is not None:
                    insts = sorted(r for r, t in g["receps"].items() if t == cls)
                    new_r = insts[int(draw_inst * len(insts))]
                    pddl, n_inst = C.move_all_instances(pddl0, g["target"], new_r)
                hits = [new_r] if new_r else list(g["hits"])
                gid = g["rel"].replace("/", "__") + f"__v{v}"
                d = os.path.join(OUT, split, gid)
                os.makedirs(d, exist_ok=True)
                json.dump({"pddl_domain": gd0["pddl_domain"], "grammar": gd0["grammar"], "pddl_problem": pddl,
                           "solvable": None, "walkthrough": None}, open(os.path.join(d, "game.tw-pddl"), "w"))
                meta = dict(id=gid, split=split, src_split=src_split, src=g["rel"], variant=v, floor=g["floor"],
                            scene=g["scene"], task_type=g["task_type"], target=g["target"], dest=g["dest"],
                            toggle=g["toggle"], learnable=learnable, moved=bool(moved and new_r),
                            pi_draw_moved=moved, cf_class=cls, slot_rule=slot_rule, new_r=new_r,
                            n_moved_instances=n_inst, receps=g["receps"], hits=hits, orig_hits=list(g["hits"]))
                json.dump(meta, open(os.path.join(d, "meta.json"), "w"))
                idx.write(json.dumps({k: meta[k] for k in meta if k not in ("receps",)}) + "\n")
                stats[(split, learnable, meta["moved"], slot_rule if moved else "unmoved_draw")] += 1
    idx.close()
    summ = {"|".join(map(str, k)): c for k, c in sorted(stats.items(), key=lambda kv: str(kv[0]))}
    json.dump(summ, open(os.path.join(OUT, "generation_counts.json"), "w"), indent=1)
    for k, c in summ.items():
        print(k, c)


if __name__ == "__main__":
    main()
