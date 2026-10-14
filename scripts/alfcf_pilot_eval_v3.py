"""Know2Act ALF-CF evaluation v3 (PREREG Addendum 20) = evaluation v2 (Addendum 19) + the content-free `presence`
condition (Fix 2b: template with every type slot = "receptacle") + `--split` (val | test). Nothing else changed.
Original v2 docstring follows.
Know2Act ALF-CF evaluation v2 (PREREG Addendum 19, R1/R4/R5). Project env (vLLM); TextWorld envs run in
subprocesses running ALFWORLD_PYTHON (alfcf_envworker.py).

1. Free greedy rollouts on the val games (<= 50 steps). A fallback decision point (dp) is a `go to` taken while the
   target has never been seen; p = its index among the game's fallback dps (p = 0 is the first). Recorded per dp:
   remaining types, the convention class of the room (learnable: r1 if present and not the destination, else r2) and
   how many of its instances were visited before the dp (unvisited / partial / refuted).
2. At every dp with >= 2 remaining types, the model's history is kept, a condition's note line (or none) is placed
   before a prefilled '<action>', and the action is sampled K times (temperature 0.7, common random numbers):
     intact     the model's own note (as written; else its greedy note after '<prediction>The', flagged forced)
     flip       the model's own greedy note for o', o' drawn uniformly (seeded) among VALID candidates: all learnable
                same-scene classes != target whose own note (target renamed to o' in the task line and in earlier
                notes) puts first a type that is unvisited and differs from the intact top; renamed back.
                Recorded: number of valid candidates and the foil plausibility (normalised rank of the foil top in the
                true class's ALF-CF prior over the remaining types). No valid candidate -> no flip at that dp.
     remove     no note line
     placebo    template with the remaining types in alphabetical order (trained form; manipulation check)
     random     template with a seeded random order of the remaining types (never trained), top != intact top and
                != alphabetical first
     flip_tmpl  (p = 0 only) template ranking of the same o' from the ALF-CF train prior (equivalence check)
     nh_intact / nh_flip  (p >= 1 only) as intact / flip but with every earlier '<prediction>' line removed
3. Counterfactual rollouts (episode level) for games whose first dp has a valid own flip: greedy rollouts with a
   forced template note at every note slot (target unseen, last observation neither 'is closed' nor 'Nothing
   happens'): cf_foil uses o' (the first-dp foil class), cf_true uses the true class.
usage: python alfcf_pilot_eval_v3.py --model M --alfcf data/alfcf_v1 --games val_games_ok.json --rules rules_v1.json
         --priors priors_v1.json --text_ref text_N100.jsonl --out DIR --shard i/n
"""
import argparse
import json
import os
import random
import re
import subprocess
import sys
import time
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import alfcf_text as T  # noqa: E402

ENV_PYTHON = os.environ.get("ALFWORLD_PYTHON", sys.executable)  # a python with alfworld/textworld for the env workers
WORKER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "alfcf_envworker.py")
FMT = re.compile(r"^(?:<prediction>(.*?)</prediction>\n)?<action>(.*?)</action>$", re.S)
GOTO = re.compile(r"^\s*go to ([a-z]+) (\d+)\s*</action>")
ACT = re.compile(r"<action>(.*?)</action>", re.S)
NOTE_LINE = re.compile(r"<prediction>.*?</prediction>\n", re.S)


class EnvPool:
    def __init__(self, n):
        self.p = [subprocess.Popen([ENV_PYTHON, "-I", "-u", WORKER], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                   text=True, bufsize=1) for _ in range(n)]
        self.w = {}

    def call(self, reqs):
        by = defaultdict(list)
        for q in reqs:
            if q["key"] not in self.w:
                self.w[q["key"]] = len(self.w) % len(self.p)
            by[self.w[q["key"]]].append(q)
        for w, qs in by.items():
            for q in qs:
                self.p[w].stdin.write(json.dumps(q) + "\n")
            self.p[w].stdin.flush()
        out = {}
        for w, qs in by.items():
            for _ in qs:
                r = json.loads(self.p[w].stdout.readline())
                out[r["key"]] = r
        return out

    def close(self):
        for p in self.p:
            p.stdin.close()
            p.wait()


def main():
    ap = argparse.ArgumentParser()
    for k in ["--model", "--alfcf", "--games", "--rules", "--priors", "--text_ref", "--out"]:
        ap.add_argument(k, required=True)
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max_steps", type=int, default=50)
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--temp", type=float, default=0.7)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--active", type=int, default=120)
    ap.add_argument("--gpu_mem", type=float, default=0.85)
    ap.add_argument("--max_model_len", type=int, default=16384)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    ref = json.loads(open(a.text_ref).readline())
    assert ref["system"] == T.SYSTEM, "system prompt differs from the SFT data"
    rules = json.load(open(a.rules))
    P = json.load(open(a.priors))
    learn_by_scene = defaultdict(list)
    for k in rules["convention"]:
        sc, o = k.split("|")
        learn_by_scene[sc].append(o)
    si, sn = (int(x) for x in a.shard.split("/"))
    games = json.load(open(a.games))[si::sn]
    if a.limit:
        games = games[: a.limit]
    metas = {g: json.load(open(os.path.join(a.alfcf, a.split, g, "meta.json"))) for g in games}

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt
    tok = AutoTokenizer.from_pretrained(a.model)
    llm = LLM(model=a.model, gpu_memory_utilization=a.gpu_mem, max_model_len=a.max_model_len,
              enable_prefix_caching=True, seed=0)
    enc_cache = {}

    def enc(s):
        if s not in enc_cache:
            enc_cache[s] = list(tok(s, add_special_tokens=False)["input_ids"])
        return enc_cache[s]

    def pids(user):
        txt = tok.apply_chat_template([{"role": "system", "content": T.SYSTEM}, {"role": "user", "content": user}],
                                      tokenize=False, add_generation_prompt=True, enable_thinking=False)
        return list(tok(txt, add_special_tokens=False)["input_ids"])

    def render(user, segs, strip_notes=False, rename=None, target=None):
        if rename:
            user = re.sub(rf"\b{re.escape(target)}\b", rename, user)
        ids = pids(user)
        for s in segs:
            txt = s["text"]
            if s["type"] == "act":
                if strip_notes:
                    txt = NOTE_LINE.sub("", txt)
                elif rename:
                    txt = txt.replace(f"<prediction>The {target} ", f"<prediction>The {rename} ")
            ids = ids + enc(txt)
        return ids

    def gen(prompts, sps):
        if not prompts:
            return []
        outs = llm.generate([TokensPrompt(prompt_token_ids=p) for p in prompts], sps, use_tqdm=False)
        return [(o.outputs[0].text, o.outputs[0].finish_reason) for o in outs]

    greedy = lambda stop, n: SamplingParams(temperature=0.0, max_tokens=n, stop=[stop], include_stop_str_in_output=True)

    def prior_rank(cls, scene, types):
        p = P["p_cf"].get(f"{scene}|{cls}", {})
        m = P["m_cf"].get(scene, {})
        return sorted(types, key=lambda t: (-round(p.get(t, 0), 9), -round(m.get(t, 0), 9), t))

    def conv_type(meta, room_types):
        if not meta["learnable"]:
            return None
        cv = rules["convention"].get(f'{meta["scene"]}|{meta["target"]}')
        for key in ("r1", "r2"):
            if cv and cv[key] in room_types and cv[key] != meta["dest"]:
                return cv[key]
        return None

    # ------------------------------------------------------------ rollouts (free or forced-note)
    def rollouts(jobs, tag):
        """jobs: list of (key, gid, forced_class or None). Returns finished episode dicts."""
        pool = EnvPool(a.workers)
        eps, todo, done = [], list(jobs), []
        while todo or eps:
            while todo and len(eps) < a.active:
                key, gid, forced = todo.pop(0)
                meta = metas[gid]
                r = pool.call([{"op": "reset", "key": key, "game": os.path.join(a.alfcf, a.split, gid, "game.tw-pddl")}])[key]
                intro = r["obs"].strip()
                receps = T.receps_from_intro(intro)
                eps.append(dict(key=key, gid=gid, meta=meta, forced=forced, user=intro, ids=pids(intro), segs=[],
                                turns=[], seen=False, visited=set(), receps=receps, last_obs="", won=False,
                                room_types=sorted({x.split()[0] for x in receps}), err=r.get("error")))
            prompts, sps = [], []
            for e in eps:
                e["_pre"] = ""
                if e["forced"] and not e["seen"] and "is closed" not in e["last_obs"] \
                        and not e["last_obs"].strip().startswith("Nothing happens"):
                    rem = sorted({x.split()[0] for x in e["receps"] if x not in e["visited"]})
                    if rem:
                        e["_pre"] = f"<prediction>{T.note_text(e['meta']['target'], prior_rank(e['forced'], e['meta']['scene'], rem))}</prediction>\n"
                prompts.append(e["ids"] + (enc(e["_pre"]) if e["_pre"] else []))
                sps.append(greedy("</action>", 160))
            outs = gen(prompts, sps)
            reqs = []
            for e, (txt, fin) in zip(eps, outs):
                full = e["_pre"] + txt
                m = FMT.match(full.strip()) if fin == "stop" else None
                note = m.group(1) if m else None
                mm = ACT.search(full)
                act = (m.group(2) if m else (mm.group(1) if mm else "look")).strip()
                dp = None
                if not e["seen"] and act.startswith("go to "):
                    rem = sorted({x.split()[0] for x in e["receps"] if x not in e["visited"]})
                    ct = conv_type(e["meta"], e["room_types"])
                    n_ct = sum(1 for x in e["receps"] if x.split()[0] == ct) if ct else 0
                    v_ct = sum(1 for x in e["visited"] if x.split()[0] == ct) if ct else 0
                    p = sum(1 for t in e["turns"] if t["dp"] is not None)
                    dp = dict(gid=e["gid"], turn=len(e["turns"]), p=p, user=e["user"], wrote_note=note is not None,
                              note=note, action=act, act_type=act[6:].split()[0] if len(act) > 6 else None,
                              remaining=rem, conv_type=ct, conv_status=(None if not ct else "unvisited" if v_ct == 0
                                                                        else "refuted" if v_ct >= n_ct else "partial"))
                    if tag == "free":
                        dp["prefix_ids"], dp["prefix_segs"] = list(e["ids"]), list(e["segs"])
                e["turns"].append(dict(text=full, fmt_ok=m is not None, finish=fin, action=act, note=note, dp=dp,
                                       forced_note=bool(e["_pre"])))
                e["_act"] = act
                reqs.append({"op": "step", "key": e["key"], "action": act})
            res = pool.call(reqs)
            keep = []
            for e in eps:
                r = res[e["key"]]
                obs = r["obs"]
                act = e["_act"]
                e["turns"][-1]["obs"] = obs
                e["last_obs"] = obs
                if act.startswith("go to "):
                    e["visited"].add(act[6:].strip())
                if "you see" in obs:
                    for o in T.OBJ_RE.findall(obs.split("you see", 1)[1]):
                        if o.split()[0] == e["meta"]["target"] and not e["seen"]:
                            e["seen"] = True
                            e["seen_after_visits"] = len(e["visited"])
                gtxt = e["turns"][-1]["text"]
                e["ids"] = e["ids"] + enc(gtxt) + enc(T.OBS_OPEN + obs + T.OBS_CLOSE)
                e["segs"] = e["segs"] + [{"type": "act", "text": gtxt}, {"type": "obs", "text": T.OBS_OPEN + obs + T.OBS_CLOSE}]
                e["won"] = e["won"] or bool(r.get("won"))
                if e["won"] or len(e["turns"]) >= a.max_steps or len(e["ids"]) > a.max_model_len - 400:
                    pool.call([{"op": "close", "key": e["key"]}])
                    done.append(e)
                else:
                    keep.append(e)
            eps = keep
        pool.close()
        return done

    t0 = time.time()
    free = rollouts([(g, g, None) for g in games], "free")
    print(f"free rollouts: {len(free)} games, {time.time() - t0:.0f}s", flush=True)
    dps = []
    for e in free:
        for t in e["turns"]:
            if t["dp"] is not None:
                d = t["dp"]
                d.update(dp_id=f'{e["gid"]}#{d["turn"]}', scene=e["meta"]["scene"], target=e["meta"]["target"],
                         learnable=e["meta"]["learnable"], moved=e["meta"]["moved"], src_split=e["meta"]["src_split"])
                dps.append(d)
    with open(os.path.join(a.out, f"rollouts.shard{si}.jsonl"), "w") as f:
        for e in free:
            f.write(json.dumps(dict(gid=e["gid"], won=e["won"], n_steps=len(e["turns"]), err=e["err"],
                                    seen_after_visits=e.get("seen_after_visits"),
                                    turns=[{k: v for k, v in t.items() if k != "dp"} | {"is_dp": t["dp"] is not None}
                                           for t in e["turns"]])) + "\n")

    # ------------------------------------------------------------ interventions
    dps = [d for d in dps if len(d["remaining"]) >= 2]
    need = [d for d in dps if not (d["wrote_note"] and T.parse_note(d["note"] or "")
                                   and T.parse_note(d["note"])[0] == d["target"])]
    outs = gen([d["prefix_ids"] + enc("<prediction>The") for d in need], [greedy("</prediction>", 60) for _ in need])
    for d, (txt, fin) in zip(need, outs):
        d["note_forced"] = True
        d["note"] = "The" + txt.replace("</prediction>", "")
    for d in dps:
        d.setdefault("note_forced", False)
        pn = T.parse_note(d["note"] or "")
        d["intact_types"] = pn[1] if pn and pn[0] == d["target"] else None
    n_before = len(dps)
    dps = [d for d in dps if d["intact_types"]]
    # own notes for every candidate o'
    jobs = []
    for d in dps:
        d["cands"] = [o for o in learn_by_scene[d["scene"]] if o != d["target"]]
        for o2 in d["cands"]:
            ids = render(d["user"], d["prefix_segs"], rename=o2, target=d["target"])
            jobs.append((d, o2, ids + enc(f"<prediction>The {o2} is most likely in a")))
    outs = gen([j[2] for j in jobs], [greedy("</prediction>", 50) for _ in jobs])
    for (d, o2, _), (txt, fin) in zip(jobs, outs):
        note2 = f"The {o2} is most likely in a" + txt.replace("</prediction>", "")
        pn = T.parse_note(note2)
        d.setdefault("own", {})[o2] = pn[1] if pn else None
        d.setdefault("own_text", {})[o2] = note2.strip()
    for d in dps:
        valid = [o for o in d["cands"] if d["own"].get(o) and d["own"][o][0] in d["remaining"]
                 and d["own"][o][0] != d["intact_types"][0]]
        d["n_valid_own"] = len(valid)
        d["flip"] = None
        if valid:
            o2 = random.Random(T.stable_seed("foil_v2", d["dp_id"])).choice(valid)
            d["flip_obj"], d["flip_types"] = o2, d["own"][o2]
            d["flip"] = re.sub(rf"^The {re.escape(o2)} ", f"The {d['target']} ", d["own_text"][o2])
            true_rank = prior_rank(d["target"], d["scene"], d["remaining"])
            d["foil_plausibility"] = true_rank.index(d["flip_types"][0]) / max(1, len(true_rank) - 1)
            if d["p"] == 0:
                rk = prior_rank(o2, d["scene"], d["remaining"])[:3]
                d["flip_tmpl"] = T.note_text(d["target"], rk) if rk[0] != d["intact_types"][0] else None
                d["flip_tmpl_types"] = rk
        d["placebo_types"] = d["remaining"][:3]
        d["placebo"] = T.note_text(d["target"], d["placebo_types"])
        rr = random.Random(T.stable_seed("random_note", d["dp_id"]))
        d["random"] = None
        for _ in range(50):
            perm = list(d["remaining"])
            rr.shuffle(perm)
            if perm[0] != d["intact_types"][0] and perm[0] != d["remaining"][0]:
                d["random_types"] = perm[:3]
                d["random"] = T.note_text(d["target"], perm[:3])
                break
        del d["own_text"]
    # conditions
    reqs, keys = [], []
    for d in dps:
        conds = {"intact": d["note"].strip(), "remove": None, "placebo": d["placebo"],
                 "presence": f"The {d['target']} is most likely in a receptacle, then a receptacle, then a receptacle."}
        if d["flip"]:
            conds["flip"] = d["flip"]
        if d["random"]:
            conds["random"] = d["random"]
        if d["p"] == 0 and d.get("flip_tmpl"):
            conds["flip_tmpl"] = d["flip_tmpl"]
        base = d["prefix_ids"]
        nh = render(d["user"], d["prefix_segs"], strip_notes=True) if d["p"] >= 1 else None
        if nh is not None:
            conds["nh_intact"] = d["note"].strip()
            if d["flip"]:
                conds["nh_flip"] = d["flip"]
        for cond, note in conds.items():
            pre = (nh if cond.startswith("nh_") else base) + (enc(f"<prediction>{note}</prediction>\n") if note else []) + enc("<action>")
            for k in range(a.K):
                reqs.append((pre, SamplingParams(temperature=a.temp, top_p=1.0, max_tokens=40, stop=["</action>"],
                                                 include_stop_str_in_output=True,
                                                 seed=T.stable_seed("crn", d["dp_id"], k) % (2 ** 31))))
                keys.append((d["dp_id"], cond))
    outs = gen([r[0] for r in reqs], [r[1] for r in reqs])
    res = defaultdict(lambda: defaultdict(list))
    for (did, cond), (txt, fin) in zip(keys, outs):
        m = GOTO.match(txt)
        res[did][cond].append(m.group(1) if m else ("NONSEARCH" if txt.strip().endswith("</action>") else "INVALID"))
    with open(os.path.join(a.out, f"interventions.shard{si}.jsonl"), "w") as f:
        for d in dps:
            for k in ["prefix_ids", "prefix_segs", "user"]:
                d.pop(k, None)
            d["outcomes"] = res[d["dp_id"]]
            f.write(json.dumps(d) + "\n")
    print(f"interventions: {len(dps)} dps ({n_before - len(dps)} without a parsable intact note), "
          f"{len(reqs)} samples, {time.time() - t0:.0f}s", flush=True)

    # ------------------------------------------------------------ counterfactual rollouts
    first = {}
    for d in dps:
        if d["p"] == 0 and d["flip"]:
            first[d["gid"]] = d
    jobs = [(f"{g}|cf_foil", g, first[g]["flip_obj"]) for g in sorted(first)] + \
           [(f"{g}|cf_true", g, metas[g]["target"]) for g in sorted(first)]
    cf = rollouts(jobs, "cf")
    with open(os.path.join(a.out, f"cf_rollouts.shard{si}.jsonl"), "w") as f:
        for e in cf:
            gdp = [t for t in e["turns"] if t["dp"] is not None]
            f.write(json.dumps(dict(key=e["key"], gid=e["gid"], mode=e["key"].split("|")[1], forced=e["forced"],
                                    foil_top0=first[e["gid"]]["flip_types"][0],
                                    first_visit_type=gdp[0]["dp"]["act_type"] if gdp else None,
                                    first_forced_note=next((t["text"] for t in e["turns"] if t["forced_note"]), None),
                                    n_fallback_visits=len(gdp), seen=e["seen"], won=e["won"], n_steps=len(e["turns"]),
                                    learnable=e["meta"]["learnable"])) + "\n")
    print(f"counterfactual rollouts: {len(cf)} episodes, total {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
