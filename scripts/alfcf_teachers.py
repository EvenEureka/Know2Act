"""Scripted, non-privileged ALF-CF teachers T+ and T- (Know2Act PREREG Addendum 18, step 3). CPU only.

Both teachers search for the task's target object using only observations, visiting receptacle TYPES in the order
of a location prior over the types that still have unvisited instances, and one uniformly drawn unvisited instance
of the chosen type (open it if closed). They remember every object they have seen. Once the target is in view they
finish the task by script (take, clean/heat/cool, move to the destination; second instance; lamp).
  T+ : prior = ALF-CF type-level prior p_cf(o, t) fitted on the ALF-CF train split (knows the convention).
  T- : prior = standard ALFWorld type-level prior p_std(o, t) fitted on the same games before the move.
Ties: scene marginal of the same world, then name.

Every `go to` taken while the target has never been seen is a fallback decision point (dp). At each dp we record the
remaining types, the T+ note ranking (the "written location note": top-3 of p_cf over the remaining types), the
placebo ranking (alphabetical = order of the room description), a flip ranking (p_cf of another learnable class o'
of the same scene type whose top remaining type differs; one o' drawn per dp), and a note draw shared by both
teachers at the same dp index: real 0.63, placebo 0.07, none 0.30 (notes present 70%; placebo 10% of notes).
Solvability = the teacher wins within MAX_STEPS.
usage: python alfcf_teachers.py <alfcf_dir> <rules.json> <split> <out.jsonl> [--procs N] [--limit N] [--priors_only]
"""
import argparse
import json
import multiprocessing as mp
import os
import random
import re
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import alfcf_common as C  # noqa: E402

MAX_STEPS = 150
NOTE_P = {"real": 0.63, "placebo": 0.07, "none": 0.30}
OBJ_RE = re.compile(r"\b(?:a|an) ([a-z]+ \d+)")


def load_meta(alfcf_dir, split):
    out = []
    for l in open(os.path.join(alfcf_dir, "index.jsonl")):
        m = json.loads(l)
        if m["split"] == split:
            out.append(m)
    return out


def fit_priors(alfcf_dir, rules):
    """Type-level priors from the ALF-CF train split (p_cf) and from the same base games before the move (p_std)."""
    num_cf, den, num_std = defaultdict(Counter), defaultdict(Counter), defaultdict(Counter)
    mnum_cf, mnum_std, mden = defaultdict(Counter), defaultdict(Counter), defaultdict(Counter)
    for l in open(os.path.join(alfcf_dir, "index.jsonl")):
        m = json.loads(l)
        if m["split"] != "train":
            continue
        meta = json.load(open(os.path.join(alfcf_dir, "train", m["id"], "meta.json")))
        rt, meta["hits"] = C.strip_goal_ids(meta["receps"], meta["hits"])
        _, meta["orig_hits"] = C.strip_goal_ids(meta["receps"], meta["orig_hits"])
        key = f'{meta["scene"]}|{meta["target"]}'
        hit_cf = {rt[r] for r in meta["hits"]}
        hit_std = {rt[r] for r in meta["orig_hits"]}
        for t in set(rt.values()):
            den[key][t] += 1
            num_cf[key][t] += t in hit_cf
            num_std[key][t] += t in hit_std
            mden[meta["scene"]][t] += 1
            mnum_cf[meta["scene"]][t] += t in hit_cf
            mnum_std[meta["scene"]][t] += t in hit_std
    pr = lambda num, d: {k: {t: num[k][t] / d[k][t] for t in d[k]} for k in d}
    return dict(p_cf=pr(num_cf, den), p_std=pr(num_std, den), m_cf=pr(mnum_cf, mden), m_std=pr(mnum_std, mden),
                learnable_by_scene={sc: sorted({k.split("|")[1] for k in den if k.startswith(sc + "|")}
                                               - set(rules["unlearnable"])) for sc in C.SCENES})


def ranking(P, world, scene, obj, types):
    p = P[f"p_{world}"].get(f"{scene}|{obj}", {})
    m = P[f"m_{world}"].get(scene, {})
    return C.rank_types(types, p, m)


class Teacher:
    def __init__(self, env, state, meta, P, which, rng):
        self.draw, self.mix_s = 0, 0.5
        self.e, self.s, self.meta, self.P, self.which, self.rng = env, state, meta, P, which, rng
        self.intro = state["feedback"]
        part = re.split("you see", self.intro, flags=re.I)[-1].split("Your task is to")[0]
        self.receps = [re.sub(r"^(and )?an? ", "", r.strip().strip(" .\n")) for r in part.split(",") if r.strip()]
        self.visited = set()
        self.seen = {}          # object name -> receptacle name
        self.turns = []
        self.loc = None
        self.won = False

    def act(self, cmd, dp=None):
        if len(self.turns) >= MAX_STEPS:
            raise RuntimeError("step limit")
        self.s, _, done = self.e.step(cmd)
        ob = self.s["feedback"]
        self.turns.append({"action": cmd, "obs": ob, "dp": dp})
        if cmd.startswith("go to "):
            self.loc = cmd[6:]
        if "you see" in ob:
            for o in OBJ_RE.findall(ob.split("you see", 1)[1]):
                self.seen.setdefault(o, self.loc)
        if self.s.get("won"):
            self.won = True
        return ob

    def goto(self, r, dp=None):
        """go to r; co-located receptacles answer "Nothing happens." -> examine r; open r if closed."""
        ob = self.act(f"go to {r}", dp)
        if ob.strip().startswith("Nothing happens") and f"examine {r}" in self.s["admissible_commands"]:
            self.loc = r
            ob = self.act(f"examine {r}")
        if "is closed" in ob and f"open {r}" in self.s["admissible_commands"]:
            self.act(f"open {r}")
        self.visited.add(r)

    def visit(self, r, dp=None):
        self.goto(r, dp)

    def seen_of(self, cls, avoid=()):
        for o, r in self.seen.items():
            if o.split()[0] == cls and o not in avoid and r is not None:
                return o, r
        return None, None

    def search(self, cls, prior_world, avoid=(), notes=False):
        """visit types by prior until an instance of cls is seen; dps recorded only when notes=True."""
        while True:
            o, r = self.seen_of(cls, avoid)
            if o:
                return o, r
            rem = [r for r in self.receps if r not in self.visited]
            if not rem:
                return None, None
            types = sorted({r.split()[0] for r in rem})
            rank = ranking(self.P, prior_world, self.meta["scene"], cls, types)
            chosen = rank[0]
            dp = None
            if notes:
                dp = self.make_dp(types, chosen)
                if self.which in ("mix", "mixall"):  # Add. 19 R2 (real-note dps only) / Add. 20 Fix 2 (every dp)
                    plus_c, minus_c = dp["note_rank"][0], ranking(self.P, "std", self.meta["scene"], cls, types)[0]
                    coin = random.Random(C.stable_seed("mix", self.meta["id"], dp["k"], self.draw)).random()
                    if self.which == "mixall":
                        by = "plus" if coin < self.mix_s else "minus"
                    else:
                        by = "plus" if (dp["draw"] != "real" or coin < self.mix_s) else "minus"
                    chosen = plus_c if by == "plus" else minus_c
                    dp.update(chosen=chosen, chosen_by=by, plus_choice=plus_c, minus_choice=minus_c)
            inst = sorted(r for r in rem if r.split()[0] == chosen)
            r = self.rng.choice(inst)
            self.visit(r, dp)

    def make_dp(self, types, chosen):
        sc, obj = self.meta["scene"], self.meta["target"]
        k = sum(1 for t in self.turns if t["dp"] is not None)
        drng = random.Random(C.stable_seed("dp", self.meta["id"], k) if self.draw == 0
                             else C.stable_seed("dp", self.meta["id"], k, self.draw))
        u = drng.random()
        draw = "real" if u < NOTE_P["real"] else ("placebo" if u < NOTE_P["real"] + NOTE_P["placebo"] else "none")
        note_rank = ranking(self.P, "cf", sc, obj, types)[:3]
        std_rank = ranking(self.P, "std", sc, obj, types)[:3]
        flips = []
        for o2 in self.P["learnable_by_scene"][sc]:
            if o2 == obj:
                continue
            r2 = ranking(self.P, "cf", sc, o2, types)[:3]
            if r2[0] != note_rank[0]:
                flips.append((o2, r2))
        flip = drng.choice(flips) if flips else None
        return {"k": k, "types": types, "n_types": len(types), "note_rank": note_rank, "std_rank": std_rank,
                "placebo_rank": sorted(types)[:3], "chosen": chosen, "draw": draw,
                "flip_obj": flip[0] if flip else None, "flip_rank": flip[1] if flip else None,
                "n_flip_cands": len(flips)}

    def goto_cls(self, cls):
        for r in self.receps:
            if r.split()[0] == cls:
                if self.loc != r:
                    self.goto(r)
                return r

    def take(self, o, r):
        if self.loc != r:
            self.goto(r)
        self.act(f"take {o} from {r}")
        self.seen.pop(o, None)

    def run(self, tj_params):
        m = self.meta
        prior = "cf" if self.which in ("plus", "mix", "mixall") else "std"
        o, r = self.search(m["target"], prior, notes=True)
        self.found_type = r.split()[0] if r else None
        self.n_dp = sum(1 for t in self.turns if t["dp"] is not None)
        if o is None:
            return
        self.take(o, r)
        tt = m["task_type"]
        if tt == "look_at_obj_in_light":
            lamp, lr = self.search(m["toggle"], prior)
            if lamp:
                if self.loc != lr:
                    self.goto(lr)
                self.act(f"use {lamp}")
            return
        for kind, app in [("clean", "sinkbasin"), ("heat", "microwave"), ("cool", "fridge")]:
            if f"pick_{kind}_then" in tt:
                a = self.goto_cls(app)
                self.act(f"{kind} {o} with {a}")
        d = self.goto_cls(m["dest"])
        self.act(f"move {o} to {d}")
        if tt == "pick_two_obj_and_place" and not self.won:
            o2, r2 = self.search(m["target"], prior, avoid=(o,))
            if o2:
                self.take(o2, r2)
                d = self.goto_cls(m["dest"])
                self.act(f"move {o2} to {d}")


def make_env(gp):
    import textworld
    from textworld.envs import PddlEnv
    from alfworld.agents.environment.alfred_tw_env import AlfredInfos, AlfredDemangler
    req = textworld.EnvInfos(admissible_commands=True, won=True)
    e = AlfredInfos(AlfredDemangler(PddlEnv(req), shuffle=False))
    e.load(gp)
    return e, e.reset()


_G = {}


def init_worker(alfcf_dir, P, teachers=("plus", "minus"), draw=0, mix=0.5):
    _G["dir"], _G["P"], _G["teachers"], _G["draw"], _G["mix"] = alfcf_dir, P, teachers, draw, mix


def run_one(m):
    d = os.path.join(_G["dir"], m["split"], m["id"])
    meta = json.load(open(os.path.join(d, "meta.json")))
    meta["receps"], meta["hits"] = C.strip_goal_ids(meta["receps"], meta["hits"])  # goal-leak fix (13:30)
    out = []
    for which in _G["teachers"]:
        rec = dict(id=m["id"], split=m["split"], teacher=which, draw=_G["draw"], mix_s=_G["mix"], scene=meta["scene"], target=meta["target"],
                   task_type=meta["task_type"], learnable=meta["learnable"], moved=meta["moved"],
                   cf_class=meta["cf_class"], slot_rule=meta["slot_rule"])
        try:
            env, st = make_env(os.path.join(d, "game.tw-pddl"))
            d0 = _G["draw"]
            T = Teacher(env, st, meta, _G["P"], which,
                        random.Random(C.stable_seed("teacher", m["id"], which) if d0 == 0
                                      else C.stable_seed("teacher", m["id"], which, d0)))
            T.draw, T.mix_s = d0, _G["mix"]
            err = None
            try:
                T.run(None)
            except Exception as ex:  # step limit or env error
                err = str(ex)[:200]
            hit_types = sorted({meta["receps"][r] for r in meta["hits"]})
            rec.update(intro=T.intro, won=bool(T.won), err=err, n_steps=len(T.turns), turns=T.turns,
                       found_type=getattr(T, "found_type", None), n_dp=getattr(T, "n_dp", None),
                       hit_types=hit_types,
                       found_consistent=(getattr(T, "found_type", None) in hit_types))
        except Exception as ex:
            rec.update(won=False, err="load: " + str(ex)[:200], turns=[])
        out.append(rec)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("alfcf_dir")
    ap.add_argument("rules")
    ap.add_argument("split")
    ap.add_argument("out")
    ap.add_argument("--procs", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--priors_only", action="store_true")
    ap.add_argument("--teachers", default="plus,minus", help="comma list of plus, minus, mix (Addendum 19), mixall (Addendum 20)")
    ap.add_argument("--draw", type=int, default=0, help="data-construction draw (0 = Addendum 18 seeds)")
    ap.add_argument("--mix", type=float, default=0.5, help="T+ share at real-note dps for teacher 'mix'")
    a = ap.parse_args()
    rules = json.load(open(a.rules))
    pp = os.path.join(a.alfcf_dir, "priors_v1.json")
    if not os.path.exists(pp):
        json.dump(fit_priors(a.alfcf_dir, rules), open(pp, "w"))
    P = json.load(open(pp))
    if a.priors_only:
        return
    metas = load_meta(a.alfcf_dir, a.split)
    if a.limit:
        metas = metas[: a.limit]
    n = 0
    with open(a.out, "w") as f, mp.Pool(a.procs, initializer=init_worker,
                                             initargs=(a.alfcf_dir, P, tuple(a.teachers.split(",")), a.draw, a.mix)) as pool:
        for recs in pool.imap(run_one, metas, chunksize=4):
            for r in recs:
                f.write(json.dumps(r) + "\n")
            n += 1
            if n % 500 == 0:
                print(n, "games", flush=True)
    print("done", a.split, n)


if __name__ == "__main__":
    main()
