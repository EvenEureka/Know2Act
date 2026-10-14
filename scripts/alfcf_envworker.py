"""TextWorld/ALFWorld env server over stdin/stdout JSON lines (runs under a python with alfworld/textworld).
ops: {"op": "reset", "key", "game"} -> {"key", "obs", "admissible"}
     {"op": "step", "key", "action"} -> {"key", "obs", "won", "admissible"}
     {"op": "close", "key"} -> {"key"}
"""
import json
import sys

import textworld
from textworld.envs import PddlEnv
from alfworld.agents.environment.alfred_tw_env import AlfredInfos, AlfredDemangler

envs = {}
for line in sys.stdin:
    q = json.loads(line)
    k = q["key"]
    try:
        if q["op"] == "reset":
            e = AlfredInfos(AlfredDemangler(PddlEnv(textworld.EnvInfos(admissible_commands=True, won=True)),
                                            shuffle=False))
            e.load(q["game"])
            s = e.reset()
            envs[k] = e
            r = {"key": k, "obs": s["feedback"], "won": False}
        elif q["op"] == "step":
            s, _, done = envs[k].step(q["action"])
            r = {"key": k, "obs": s["feedback"], "won": bool(s.get("won"))}
        else:
            envs.pop(k, None)
            r = {"key": k}
    except Exception as ex:  # report, never die
        r = {"key": k, "error": str(ex)[:300], "obs": "Nothing happens.", "won": False}
    sys.stdout.write(json.dumps(r) + "\n")
    sys.stdout.flush()
