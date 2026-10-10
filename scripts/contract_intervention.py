"""Decision-point intervention for the Contract arm (PREREG Addendum 10, 2026-10-09).

The direct test of "learned but not used": hold the model and the decision context fixed and change
only the contract at the point where the agent decides whether to submit or keep querying.

For every evaluation rollout of a Contract model (SFT or GRPO) we take the decision point after the
first SQL observation (``--dp first``; ``--dp all`` takes every one). The prefix is the task prompt, the
actions and the observations exactly as the model saw them, re-tokenised from their text. The model
then continues from that prefix under conditions that differ only in the contract line of the first
action (and, for later decision points, in the earlier check lines, which are recomputed from the new
contract because the check is a deterministic function of contract and observation):
  orig    the rollout's own contract (re-sampled continuation; same-sampling baseline)
  flip    a contract chosen so that the observed result's match status flips: a result that matched
          the own contract now mismatches (1 row <-> several rows, no rows -> 1 row, 1 column), and a
          result that mismatched now matches (contract := observed shape). Error and timeout results
          can never match and are skipped for this condition.
  remove  no contract line and no check lines at all (an action-only-style prefix)
  gold    the gold result's coarse shape (only where it differs from the own contract)
  fcheck_orig / fcheck_flip
          the own contract is kept and the next action's check line is teacher-forced into the prefix:
          consistent with the contract (fcheck_orig) or asserting the opposite match status
          (fcheck_flip), so the model continues right after a check sentence. This separates the
          contract line from the check sentence as the thing a decision could follow.
Each condition is continued K times to the end of the episode with the evaluation sampling settings
(T=0.6, top-p 0.95, top-k 20) and common random numbers across conditions. We record the next action
(submit or query, and what its check line says) and the final reward. Everything except the contract
text is identical across conditions: same prefix tokens, same replayed environment state, same seeds.

Output: one JSON line per continuation (see ``record_out``). ``--dry_run`` builds the prefixes and
replays the environments on CPU without loading vLLM, and prints the counts.
"""
import argparse
import collections
import concurrent.futures as cf
import copy
import json
import os
import sys
import time

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from build_e3 import describe  # noqa: E402
from e3_readouts import CHECK, CONTRACT, coarse, obs_coarse, parse_contract  # noqa: E402
from rollout import Traj, make_env, run_chunk, template_kwargs, vllm_generate_fn  # noqa: E402

CONDS = ("orig", "flip", "remove", "gold", "fcheck_orig", "fcheck_flip")


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--rollouts", required=True, help="results/eval/<run> directory of the same model")
    p.add_argument("--data", required=True, help="SkyRL-format parquet (same as the rollouts)")
    p.add_argument("--ds", required=True, choices=["spider_dev", "bird_dev"])
    p.add_argument("--db_path", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--samples", type=int, default=4, help="rollout samples per question to intervene on")
    p.add_argument("--k", type=int, default=4, help="continuations per condition")
    p.add_argument("--conds", default="orig,flip,remove,gold,fcheck_orig,fcheck_flip")
    p.add_argument("--dp", choices=["first", "all"], default="first")
    p.add_argument("--shard", default="0/1")
    p.add_argument("--limit", type=int, default=-1, help="questions (after sharding)")
    p.add_argument("--chunk", type=int, default=768, help="continuations in flight per vLLM chunk")
    p.add_argument("--env_workers", type=int, default=32)
    p.add_argument("--dry_run", action="store_true")
    # sampling / engine, same defaults as scripts/jobs/eval.sge
    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--top_p", type=float, default=0.95)
    p.add_argument("--top_k", type=int, default=20)
    p.add_argument("--max_turns", type=int, default=6)
    p.add_argument("--max_generate_length", type=int, default=1024)
    p.add_argument("--max_input_length", type=int, default=8192)
    p.add_argument("--max_model_len", type=int, default=32768)
    p.add_argument("--tp", type=int, default=1)
    p.add_argument("--gpu_mem", type=float, default=0.88)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--enable_thinking", choices=["default", "true", "false"], default="false")
    return p.parse_args()


def flip_shape(s):
    """A plausible contract that the observed shape s does NOT match (None for error/timeout)."""
    if s[0] == "empty":
        return ("one", 1)
    if s[0] == "one":
        return ("many", s[1])
    if s[0] == "many":
        return ("one", s[1])
    return None


def contract_line(shape):
    return f"Expected answer: {describe(shape)}."


def check_line(got, shape):
    ok = got == shape
    return f"Check: got {describe(got)}; this {'matches' if ok else 'does not match'} the expected answer."


def rewrite_prefix(segs, upto, shape, remove):
    """Modified texts for segments[0..upto]: the contract line of the first action and the check lines
    of later actions follow `shape`; with remove=True both are deleted. Observations are unchanged."""
    texts = []
    for k in range(upto + 1):
        s = segs[k]
        t = s["text"]
        if s["type"] == "act":
            if k == 0:
                m = CONTRACT.search(t)
                assert m is not None
                new = "" if remove else contract_line(shape)
                t = t[: m.start()] + new + t[m.end():]
                if remove and t.startswith("\n"):
                    t = t[1:]
            else:
                m = CHECK.search(t)
                if m is not None:
                    got = obs_coarse(segs[k - 1]["text"]) if segs[k - 1]["type"] == "obs" else None
                    if remove or got is None:
                        new = ""
                    else:
                        new = check_line(got, shape)
                    t = t[: m.start()] + new + t[m.end():]
                    if not new and t[m.start():].startswith("\n"):
                        t = t[: m.start()] + t[m.start() + 1:]
        texts.append(t)
    return texts


def decision_points(segs, first_only):
    out = []
    for i in range(1, len(segs) - 1):
        if segs[i]["type"] == "obs" and segs[i + 1]["type"] == "act" and not segs[i].get("dropped"):
            if obs_coarse(segs[i]["text"]) is not None:
                out.append(i)
                if first_only:
                    break
    return out


def build_env(row, db_path, max_turns, act_texts, obs_texts):
    """Replay the prefix actions so that the environment is in the state after the decision point's
    observation. Returns (env, n_mismatching_observations)."""
    env = make_env(row, db_path, max_turns)
    bad = 0
    for a, o in zip(act_texts, obs_texts):
        out = env.step(a)
        got = "".join(x["content"] for x in out["observations"])
        # SkyRL renders result sets through a Python set, so row order depends on PYTHONHASHSEED; the
        # prefix always uses the recorded text, and only an order-insensitive difference is a real mismatch
        if sorted(got.split("\n")) != sorted(o.split("\n")):
            bad += 1
    return env, bad


class Item:
    """One (rollout, decision point, condition) with its prefix ids and replayed env."""

    def __init__(self, qid, sample, dp, dp_rank, cond, prefix_ids, env, meta):
        self.qid, self.sample, self.dp, self.dp_rank, self.cond = qid, sample, dp, dp_rank, cond
        self.prefix_ids, self.env, self.meta = prefix_ids, env, meta


def make_items(r, row, tok, args, gold_c, kwargs):
    """All condition items for one rollout record."""
    segs = r["segments"]
    acts = [s for s in segs if s["type"] == "act"]
    own = parse_contract(CONTRACT.search(acts[0]["text"])) if acts else None
    if own is None:
        return [], "no_contract"
    prompt_ids = tok.apply_chat_template(list(row["prompt"]), add_generation_prompt=True, tokenize=True,
                                         return_dict=False, **kwargs)
    items = []
    for rank, i in enumerate(decision_points(segs, args.dp == "first"), start=1):
        got = obs_coarse(segs[i]["text"])
        own_match = got == own
        conds = {}  # cond -> (contract shape or None, remove, forced check text or None)
        can_match = got[0] in ("empty", "one", "many")
        for c in args.conds.split(","):
            if c == "orig":
                conds[c] = (own, False, None)
            elif c == "flip":
                new = flip_shape(got) if own_match else (got if can_match else None)
                if new is not None and new != own:
                    conds[c] = (new, False, None)
            elif c == "remove":
                conds[c] = (None, True, None)
            elif c == "gold":
                if gold_c is not None and gold_c != own:
                    conds[c] = (gold_c, False, None)
            elif c == "fcheck_orig":
                conds[c] = (own, False, check_line(got, own) + "\n")
            elif c == "fcheck_flip":
                if can_match:
                    verdict = "does not match" if own_match else "matches"
                    conds[c] = (own, False, f"Check: got {describe(got)}; this {verdict} the expected answer.\n")
        obs_texts = [segs[k]["text"] for k in range(1, i + 1, 2)]
        base_env = None
        for c, (shape, remove, forced) in conds.items():
            texts = rewrite_prefix(segs, i, shape, remove)
            act_texts = [texts[k] for k in range(0, i, 2)]
            if base_env is None:  # the environment state does not depend on the contract text
                base_env, bad = build_env(row, args.db_path, args.max_turns, act_texts, obs_texts)
            ids = list(prompt_ids)
            for t in texts:
                ids.extend(tok.encode(t, add_special_tokens=False))
            if forced is not None:
                ids.extend(tok.encode(forced, add_special_tokens=False))
            meta = {"contract_own": list(own), "contract_new": None if shape is None else list(shape),
                    "obs_shape": list(got), "own_match": own_match,
                    "new_match": None if shape is None else (got == shape),
                    "forced_check": None if forced is None else ("match" if "; this matches" in forced else "mismatch"),
                    "gold_shape": None if gold_c is None else list(gold_c), "obs_replay_bad": bad,
                    "prefix_tokens": len(ids), "prior_acts": len(act_texts)}
            items.append(Item(r["qid"], r["sample"], i, rank, c, ids, base_env, meta))
    return items, "ok"


def record_out(item, k, t, prefix_tokens):
    first = next((s for s in t.segments if s["type"] == "act"), None)
    text = first["text"] if first else ""
    m = CHECK.search(text)
    return {"qid": item.qid, "sample": item.sample, "dp": item.dp, "dp_rank": item.dp_rank, "cond": item.cond,
            "k": k, **item.meta, "submit": "<solution>" in text, "has_sql": "<sql>" in text,
            "check_said": None if m is None else ("match" if m.group(2) == "matches" else "mismatch"),
            "check_got": None if m is None else m.group(1), "reward": t.reward, "turns_total": t.turns,
            "stop_reason": t.stop_reason, "first_act": text[:600]}


def main():
    args = parse()
    from agenticrl.sql_lenient import register_as_text2sql
    register_as_text2sql()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    kwargs = template_kwargs(args)
    df = pd.read_parquet(args.data)
    i, k = (int(x) for x in args.shard.split("/"))
    qids = list(df.index)[i::k]
    if args.limit > 0:
        qids = qids[: args.limit]
    qset = set(int(q) for q in qids)
    gold = json.load(open(f"{ROOT}/results/gold_shapes_dev.json"))[args.ds]
    gold_c = {int(q): coarse(tuple(g["shape"])) for q, g in gold.items() if g["shape"][0] == "ok" and g["shape"][1] > 0}
    # rollouts of this model on this dataset, restricted to the primary questions of this shard
    recs = []
    import glob
    for f in sorted(glob.glob(f"{args.rollouts}/{args.ds}.shard*.jsonl")):
        for line in open(f):
            r = json.loads(line)
            if r["qid"] in qset and r["qid"] in gold_c and r["sample"] < args.samples:
                recs.append(r)
    done = set()
    if os.path.exists(args.out) and not args.dry_run:
        for line in open(args.out):
            x = json.loads(line)
            done.add((x["qid"], x["sample"], x["dp"], x["cond"]))
    print(f"{len(recs)} rollouts on {len(qset)} questions ({len(done)} (rollout, dp, cond) groups already done)", flush=True)

    stats = collections.Counter()
    t0 = time.time()

    def build(r):
        row = df.loc[r["qid"]]
        items, why = make_items(r, row, tok, args, gold_c.get(r["qid"]), kwargs)
        return [it for it in items if (it.qid, it.sample, it.dp, it.cond) not in done], why

    with cf.ThreadPoolExecutor(args.env_workers) as pool:
        built = list(pool.map(build, recs))
    items = [it for its, _ in built for it in its]
    for _, why in built:
        stats[why] += 1
    for it in items:
        stats[f"cond/{it.cond}"] += 1
        stats["obs_replay_bad"] += it.meta["obs_replay_bad"]
    stats["decision_points"] = len({(it.qid, it.sample, it.dp) for it in items})
    print(json.dumps({"built_s": round(time.time() - t0), **stats}), flush=True)
    if args.dry_run:
        for it in items[:3]:
            print(json.dumps({"cond": it.cond, **it.meta, "prefix_tail": tok.decode(it.prefix_ids[-160:])}), flush=True)
        return

    generate = vllm_generate_fn(args)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "a") as fout, cf.ThreadPoolExecutor(args.env_workers) as pool:
        per_chunk = max(1, args.chunk // args.k)
        for c in range(0, len(items), per_chunk):
            part = items[c: c + per_chunk]
            trajs, owners = [], []
            for it in part:
                for kk in range(args.k):
                    env = copy.deepcopy(it.env)
                    # sample id encodes k so that the K continuations get different seeds, while the same
                    # (question, sample, k, turn) seed is shared across conditions (common random numbers)
                    t = Traj(it.qid, it.sample * 16 + kk, df.loc[it.qid], it.prefix_ids, env)
                    t.turns = it.meta["prior_acts"]
                    trajs.append(t)
                    owners.append((it, kk))
            run_chunk(trajs, generate, tok, args, pool)
            for (it, kk), t in zip(owners, trajs):
                fout.write(json.dumps(record_out(it, kk, t, it.meta["prefix_tokens"])) + "\n")
            fout.flush()
            sub = sum("<solution>" in next((s["text"] for s in t.segments if s["type"] == "act"), "") for t in trajs)
            print(json.dumps({"done_items": c + len(part), "of": len(items), "elapsed_s": round(time.time() - t0),
                              "p_submit_chunk": round(sub / len(trajs), 3)}), flush=True)


if __name__ == "__main__":
    main()
