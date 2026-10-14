"""FT4 step 1 (CPU): build zero-shot location probes from real ALFWorld intros.
Probe set (fixed before computing): all valid_seen + valid_unseen games, plus 226 random train games (seed 0).
Stores intro text (from TextWorld, exactly as the agent sees it), target class, type counts n, hit counts h,
and the train-estimated prior p_C(o, .) and agnostic q(.) for scoring."""
import json, random, sys, os, re
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ft1_prior as F
import textworld
from textworld.envs import PddlEnv
from alfworld.agents.environment.alfred_tw_env import AlfredInfos, AlfredDemangler

OUT = sys.argv[1]
S = {s: F.load_split(s) for s in ["train", "valid_seen", "valid_unseen"]}
rng = random.Random(0)
train_sample = rng.sample(S["train"], 226)
probes = S["valid_seen"] + S["valid_unseen"] + train_sample
p_full, q_full = F.fit_prior(S["train"])
req = textworld.EnvInfos(admissible_commands=False)
env = AlfredInfos(AlfredDemangler(PddlEnv(req), shuffle=False))
lofo = {}
with open(OUT, "w") as f:
    for i, g in enumerate(probes):
        gp = os.path.join(g["path"], "game.tw-pddl")
        env.load(gp)
        s = env.reset()
        intro = s["feedback"]
        n, h = F.type_counts(g)
        if g["split"] == "train":
            fl = g["floor"]
            if fl not in lofo:
                lofo[fl] = F.fit_prior(S["train"], exclude_floor=fl)
            p, q = lofo[fl]
        else:
            p, q = p_full, q_full
        rec = dict(idx=i, split=g["split"], path=g["path"], task_type=g["task_type"], scene=g["scene"],
                   target=g["target"], intro=intro, n=dict(n), h=dict(h),
                   pC={t: p.get(g["target"], {}).get(t, 0.0) for t in n}, q={t: q.get(t, 0.0) for t in n},
                   valid=sorted(F.VALID.get(g["target"], [])))
        f.write(json.dumps(rec) + "\n")
        if i % 50 == 0:
            print(i, g["split"], g["target"], intro[-160:].replace("\n", " "), flush=True)
print("done", len(probes))
