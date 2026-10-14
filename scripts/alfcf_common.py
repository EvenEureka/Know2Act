"""Shared helpers for ALF-CF (counterfactual-placement ALFWorld), Know2Act PREREG Addendum 18.

Read-only over the ALFWorld data. Runs under a python with alfworld 0.4.2 and textworld 1.7.0;
the tokenisation step runs under the training env (transformers).
"""
import json
import math
import os
import re
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ft1_prior as F  # noqa: E402  (FT1 loaders and cost functions, unchanged)

SRC = os.environ.get("ALFWORLD_ROOT", "data/alfworld")  # ALFWorld data root (contains json_2.1.1, logic)
DATA = F.DATA
K2A = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repository root
SCENES = ["kitchen", "living", "bedroom", "bathroom"]

RE_RT = re.compile(r"\(receptacleType\s+(\S+)\s+(\S+)Type\)")
RE_OT = re.compile(r"\(objectType\s+(\S+)\s+(\S+)Type\)")
RE_RL = re.compile(r"\(receptacleAtLocation\s+(\S+)\s+(\S+)\)")
RE_IN = re.compile(r"\(inReceptacle\s+(\S+)\s+(\S+)\)")


def strip_goal_ids(receps, hits):
    """BUG FIX (2026-10-13 13:30, logged in PREREG Addendum 18 results): ft1_prior.load_split parses the whole
    initial_state.pddl, so the goal's existential variables leak in: (receptacleType ?r <Dest>Type) adds a fake
    receptacle '?r' of the destination class, and (objectType ?o <Target>Type) + (inReceptacle ?o ?r) make it a
    hit of the target in every non-look game. Ids starting with '?' are not objects of the room; drop them."""
    return ({k: v for k, v in receps.items() if not k.startswith("?")}, [h for h in hits if not h.startswith("?")])


def load_games(split, clean=True):
    """FT1 loader plus task fields (destination, toggle target); clean=True drops the goal-variable leak."""
    gs = F.load_split(split)
    for g in gs:
        if clean:
            g["receps"], g["hits"] = strip_goal_ids(g["receps"], g["hits"])
        tj = json.load(open(os.path.join(g["path"], "traj_data.json")))
        p = tj["pddl_params"]
        g["dest"] = p["parent_target"].lower()
        g["toggle"] = p["toggle_target"].lower()
        g["rel"] = os.path.relpath(g["path"], DATA)
    return gs


def type_share(games):
    """share[(scene, o)][t] = fraction of games (scene, target o) in which some instance of t holds o,
    among games where t is present. Type-level location prior."""
    num = defaultdict(Counter)
    den = defaultdict(Counter)
    for g in games:
        types = set(g["receps"].values())
        hit_types = {g["receps"][r] for r in g["hits"]}
        for t in types:
            den[(g["scene"], g["target"])][t] += 1
            num[(g["scene"], g["target"])][t] += t in hit_types
    return {k: {t: num[k][t] / den[k][t] for t in den[k]} for k in den}


def marginal_share(games):
    num, den = defaultdict(Counter), defaultdict(Counter)
    for g in games:
        hit_types = {g["receps"][r] for r in g["hits"]}
        for t in set(g["receps"].values()):
            den[g["scene"]][t] += 1
            num[g["scene"]][t] += t in hit_types
    return {s: {t: num[s][t] / den[s][t] for t in den[s]} for s in den}


def rank_types(types, score, tiebreak):
    """types sorted by descending score; ties by descending tiebreak score, then name."""
    return sorted(types, key=lambda t: (-round(score.get(t, 0.0), 9), -round(tiebreak.get(t, 0.0), 9), t))


# ---------------------------------------------------------------- note strings
def note_text(obj, ranked_types):
    ts = list(ranked_types)[:3]
    if not ts:
        return None
    s = f"The {obj} is most likely in a {ts[0]}"
    for t in ts[1:]:
        s += f", then a {t}"
    return s + "."


def parse_note(text):
    """-> (object, [types]) or None. Inverse of note_text."""
    m = re.match(r"\s*The (\S+) is most likely in a (\S+?)(?:,|\.)(.*)$", text.strip())
    if not m:
        return None
    ts = [m.group(2)] + re.findall(r"then a (\S+?)(?:,|\.|$)", m.group(3))
    return m.group(1), ts


# ---------------------------------------------------------------- PDDL editing (FT2 fixes kept)
def parse_problem(pddl_full):
    gi = pddl_full.find("(:goal")
    pddl = pddl_full[:gi]
    rtype = {i: t.lower() for i, t in RE_RT.findall(pddl)}
    otype = {i: t.lower() for i, t in RE_OT.findall(pddl)}
    rloc = {r: l for r, l in RE_RL.findall(pddl)}
    return gi, rtype, otype, rloc


def move_all_instances(pddl_full, target, new_r):
    """Move every instance of class `target` into receptacle id new_r; edit only the :init part."""
    gi, rtype, otype, rloc = parse_problem(pddl_full)
    pddl, goal = pddl_full[:gi], pddl_full[gi:]
    ids = [i for i, t in otype.items() if t == target]
    for i in ids:
        pat_in = rf"\(inReceptacle {re.escape(i)} \S+\)"
        if re.search(pat_in, pddl):
            pddl = re.sub(pat_in, f"(inReceptacle {i} {new_r})", pddl)
        else:
            pddl = pddl.replace("(:init", f"(:init\n        (inReceptacle {i} {new_r})", 1)
        pddl = re.sub(rf"\(objectAtLocation {re.escape(i)} \S+\)", f"(objectAtLocation {i} {rloc[new_r]})", pddl)
    return pddl + goal, len(ids)


def stable_seed(*parts):
    import hashlib
    return int(hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:12], 16)
