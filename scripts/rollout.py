"""Multi-turn text2sql rollouts that reproduce SkyRL's recipe-mode token stream exactly.

Used for teacher trajectories (SFT data) and for pass@k evaluation, so both see what GRPO sees
(SkyRL commit 99824f0, skyrl_gym_generator.agent_loop with use_conversation_multi_turn=false):
  * prompt_ids = apply_chat_template([system, user], add_generation_prompt=True, **chat_template_kwargs)
  * each turn: vLLM continues from the current ids; the engine's token ids are kept verbatim
    (trailing EOS dropped), the decoded text goes to env.step
  * observations are appended as tokenizer.encode(obs, add_special_tokens=False), no role tokens
  * the loop stops when the env says done, or before generating if len(ids) > max_input_length
  * the response ends at the last generated token (a trailing observation is dropped) and gets
    <|im_end|> appended unless the last stop reason was "length"
vLLM sampling mirrors SkyRL's get_vllm_sampling_params (min_tokens=1, skip_special_tokens,
include_stop_str_in_output, stop strings).

Output: one JSON line per trajectory with segments [{"type": "act"|"obs", "ids", "text", ...}].
"""
import argparse
import concurrent.futures as cf
import json
import os
import time

import pandas as pd
from omegaconf import DictConfig

STOP = ["</sql>", "</solution>"]


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, help="SkyRL-format parquet")
    p.add_argument("--model", required=True)
    p.add_argument("--out", required=True, help="output jsonl")
    p.add_argument("--db_path", required=True)
    p.add_argument("--n", type=int, default=1, help="samples per question")
    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--top_p", type=float, default=0.95)
    p.add_argument("--top_k", type=int, default=-1)
    p.add_argument("--max_turns", type=int, default=6)
    p.add_argument("--max_generate_length", type=int, default=1024)
    p.add_argument("--max_input_length", type=int, default=8192)
    p.add_argument("--max_model_len", type=int, default=32768)
    p.add_argument("--tp", type=int, default=1)
    p.add_argument("--gpu_mem", type=float, default=0.9)
    p.add_argument("--shard", default="0/1", help="i/k: this process handles rows i::k")
    p.add_argument("--limit", type=int, default=-1)
    p.add_argument("--chunk", type=int, default=1024, help="trajectories in flight per chunk")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save_tokens", action="store_true", help="store token ids (needed for SFT data)")
    p.add_argument("--env_workers", type=int, default=32)
    p.add_argument("--lenient", action="store_true",
                   help="bind text2sql to agenticrl.sql_lenient.LenientSQLEnv (no <think> format rules)")
    p.add_argument("--enable_thinking", choices=["default", "true", "false"], default="default",
                   help="Qwen3 chat_template_kwargs; must match generator.chat_template_kwargs in GRPO")
    return p.parse_args()


def template_kwargs(args):
    return {} if args.enable_thinking == "default" else {"enable_thinking": args.enable_thinking == "true"}


class Traj:
    def __init__(self, qid, sample, row, prompt_ids, env):
        self.qid, self.sample, self.row = qid, sample, row
        self.prompt_ids = list(prompt_ids)
        self.ids = list(prompt_ids)
        self.env = env
        self.segments = []
        self.done = False
        self.reward = None
        self.stop_reason = None
        self.turns = 0
        self.response_end = len(prompt_ids) - 1  # index of the last generated token


def make_env(row, db_path, max_turns):
    import skyrl_gym
    extras = {k: row[k] for k in row.index if k not in ("prompt", "env_class")}
    extras["max_turns"] = max_turns
    env = skyrl_gym.make("text2sql", env_config=DictConfig({"db_path": db_path}), extras=extras)
    env.init(list(row["prompt"]))
    return env


def apply_turn(t, out_ids, text, finish_reason, step_out, tokenizer):
    """Update one trajectory exactly like _update_agent_loop_state_with_singleturn_chat_template."""
    eos = tokenizer.eos_token_id
    t.stop_reason = "length" if finish_reason == "length" else "stop"
    new_obs = step_out["observations"]
    obs_ids = []
    for o in new_obs:
        obs_ids.extend(tokenizer.encode(o["content"], add_special_tokens=False))
    resp = list(out_ids)
    if resp and resp[-1] == eos:
        resp = resp[:-1]
    t.segments.append({"type": "act", "ids": resp, "text": text, "finish": t.stop_reason})
    if obs_ids:
        t.segments.append({"type": "obs", "ids": obs_ids, "text": "".join(o["content"] for o in new_obs)})
    t.response_end = len(t.ids) + len(resp) - 1
    t.ids += resp + obs_ids
    t.turns += 1
    t.done = bool(step_out["done"])
    if t.done:
        t.reward = float(step_out["reward"])


def finalize(t, tokenizer):
    """Drop a trailing observation (response ends at the last generated token); append EOS."""
    if t.segments and t.segments[-1]["type"] == "obs":
        t.segments[-1]["dropped"] = True
    eos = tokenizer.eos_token_id
    last_act = next((s for s in reversed(t.segments) if s["type"] == "act"), None)
    t.eos_appended = bool(t.stop_reason != "length" and last_act is not None
                          and (not last_act["ids"] or last_act["ids"][-1] != eos))
    if t.reward is None:  # stopped on max_input_length before the env finished
        t.reward = -1.0


def record(t, save_tokens):
    segs = []
    for s in t.segments:
        d = {k: v for k, v in s.items() if k != "ids" or save_tokens}
        d["n_tokens"] = len(s["ids"])
        segs.append(d)
    rec = {"qid": t.qid, "sample": t.sample, "db_id": t.row["db_id"], "data": t.row["data"],
           "data_source": t.row["data_source"], "reward": t.reward, "turns": t.turns,
           "stop_reason": t.stop_reason, "eos_appended": t.eos_appended, "segments": segs}
    if save_tokens:
        rec["prompt_ids"] = t.prompt_ids
    return rec


def run_chunk(trajs, generate, tokenizer, args, pool):
    """generate(list_of_(ids, seed)) -> list of (token_ids, text, finish_reason)."""
    while True:
        for t in trajs:
            if not t.done and len(t.ids) > args.max_input_length:
                t.done, t.stop_reason = True, "length"
        active = [t for t in trajs if not t.done]
        if not active:
            break
        print(json.dumps({"round_active": len(active), "of": len(trajs),
                          "max_turns_so_far": max(t.turns for t in trajs)}), flush=True)
        seeds = [(args.seed * 1_000_003 + t.qid * 1009 + t.sample * 101 + t.turns) % (2**31) for t in active]
        outs = generate([(t.ids, s) for t, s in zip(active, seeds)])
        steps = list(pool.map(lambda tp: tp[0].env.step(tp[1][1]), zip(active, outs)))
        for t, (ids, text, fin), so in zip(active, outs, steps):
            apply_turn(t, ids, text, fin, so, tokenizer)
    for t in trajs:
        finalize(t, tokenizer)


def vllm_generate_fn(args):
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt
    llm = LLM(model=args.model, tensor_parallel_size=args.tp, gpu_memory_utilization=args.gpu_mem,
              max_model_len=args.max_model_len, enable_prefix_caching=True, seed=args.seed)

    def generate(batch):
        prompts = [TokensPrompt(prompt_token_ids=ids) for ids, _ in batch]
        sps = [SamplingParams(min_tokens=1, skip_special_tokens=True, include_stop_str_in_output=True,
                              max_tokens=min(args.max_generate_length, args.max_model_len - len(ids) - 1),
                              temperature=args.temperature, top_p=args.top_p, top_k=args.top_k,
                              stop=STOP, seed=seed) for ids, seed in batch]
        outs = llm.generate(prompts, sps, use_tqdm=False)
        return [(list(o.outputs[0].token_ids), o.outputs[0].text, o.outputs[0].finish_reason) for o in outs]
    return generate


def main():
    args = parse()
    if args.lenient:
        from agenticrl.sql_lenient import register_as_text2sql
        register_as_text2sql()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    df = pd.read_parquet(args.data)
    i, k = (int(x) for x in args.shard.split("/"))
    df = df.iloc[i::k]
    if args.limit > 0:
        df = df.iloc[: args.limit]
    done_keys = set()
    if os.path.exists(args.out):  # resume
        with open(args.out) as f:
            for line in f:
                r = json.loads(line)
                done_keys.add((r["qid"], r["sample"]))
    todo = [(int(idx), s) for idx in df.index for s in range(args.n) if (int(idx), s) not in done_keys]
    print(f"{len(todo)} trajectories to run ({len(done_keys)} already done)", flush=True)
    generate = vllm_generate_fn(args)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    t0 = time.time()
    with open(args.out, "a") as fout, cf.ThreadPoolExecutor(args.env_workers) as pool:
        for c in range(0, len(todo), args.chunk):
            part = todo[c: c + args.chunk]
            trajs = []
            for qid, s in part:
                row = df.loc[qid]
                prompt_ids = tok.apply_chat_template(list(row["prompt"]), add_generation_prompt=True,
                                                     tokenize=True, return_dict=False, **template_kwargs(args))
                trajs.append(Traj(qid, s, row, prompt_ids, make_env(row, args.db_path, args.max_turns)))
            run_chunk(trajs, generate, tok, args, pool)
            for t in trajs:
                fout.write(json.dumps(record(t, args.save_tokens)) + "\n")
            fout.flush()
            rew = [t.reward for t in trajs]
            print(json.dumps({"done": c + len(part), "of": len(todo), "elapsed_s": round(time.time() - t0),
                              "acc": round(sum(r == 1.0 for r in rew) / len(rew), 4),
                              "format_fail": round(sum(r < 0 for r in rew) / len(rew), 4),
                              "mean_turns": round(sum(t.turns for t in trajs) / len(trajs), 2)}), flush=True)


if __name__ == "__main__":
    main()
