"""SFT with per-token class masks (used for every ALF-CF arm with --lam 0).

Every example carries `input_ids` and a parallel `token_class` list:
    0 = never supervised (system/user prompt, chat-template tokens, loss-masked spans such as foil type names)
    1 = assistant tokens (note lines and actions)
    2 = observation payload   (environment output content)
    3 = observation wrapper   (fixed text around the payload, e.g. <observation> tags)

Per-example loss (`--norm pooled`):
    L = -[sum_A log p + lam * sum_O log p] / (|A| + lam * |O|)
With --lam 0 (all ALF-CF arms) only class-1 tokens are supervised. `--norm separate` and `--dilute FILE`
(an unrelated-text auxiliary loss) are options of the trainer that this study does not use.
The batch loss is the mean of per-example losses (micro-batch 1, gradient accumulation).

DDP with fp32 master weights and bf16 autocast (pure-bf16 updates would round away lr~1e-5
steps on many weights), ZeRO-1 sharded AdamW states and a chunked, recomputed LM head.
Launch: torchrun --nproc_per_node 2 scripts/sft_train.py --data ... --model Qwen/Qwen3-4B --out ... --lam 0
"""
import argparse
import hashlib
import json
import math
import os
import random
import time

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.distributed.optim import ZeroRedundancyOptimizer
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.checkpoint import checkpoint
from transformers import AutoModelForCausalLM, AutoTokenizer

ACTION, OBS_PAYLOAD, OBS_WRAPPER = 1, 2, 3
CE_CHUNK = 2048  # target positions per LM-head chunk; logits are recomputed in backward


def _chunk_logps(lm_head, h, tgt):
    return -F.cross_entropy(lm_head(h).float(), tgt, reduction="none")


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, help="jsonl with input_ids, token_class, id")
    p.add_argument("--model", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--lam", type=float, default=0.0, help="weight of the observation tokens in the loss; 0 (all ALF-CF arms) supervises only class-1 tokens")
    p.add_argument("--obs_classes", default="2,3", help="token classes counted as observation (O)")
    p.add_argument("--norm", choices=["pooled", "separate"], default="pooled")
    p.add_argument("--dilute", default=None, help="jsonl of unrelated-text token chunks for an auxiliary loss (not used in this study)")
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--global_batch", type=int, default=64)
    p.add_argument("--warmup_ratio", type=float, default=0.03)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--max_len", type=int, default=12288)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max_steps", type=int, default=-1, help="debug: stop early")
    p.add_argument("--log_every", type=int, default=1)
    p.add_argument("--no_grad_ckpt", action="store_true")
    return p.parse_args()


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class PositionLogProbs(torch.nn.Module):
    """Wraps a causal LM so DDP sees one forward per example.

    Returns log p(ids[t] | ids[<t]) at the requested target positions (t >= 1), running the LM head
    only there (the full [T, vocab] logits of a 12k-token trajectory would cost ~7 GB in fp32).
    An optional second sequence (the unrelated text of --dilute) goes through the same forward call.
    """

    def __init__(self, lm):
        super().__init__()
        self.lm = lm

    def _logps(self, ids, positions):
        hidden = self.lm.model(input_ids=ids[None], use_cache=False).last_hidden_state[0]
        h, tgt = hidden[positions - 1], ids[positions]
        # Chunked + checkpointed LM head: an 8k-token trajectory scores ~7k targets, whose fp32
        # [n, 151936] logits plus their softmax grads (~10 GB) caused the round-1 OOM (Addendum 1).
        out = []
        for s in range(0, h.shape[0], CE_CHUNK):
            hs, ts = h[s:s + CE_CHUNK], tgt[s:s + CE_CHUNK]
            if torch.is_grad_enabled():
                out.append(checkpoint(_chunk_logps, self.lm.lm_head, hs, ts, use_reentrant=False))
            else:
                out.append(_chunk_logps(self.lm.lm_head, hs, ts))
        return torch.cat(out) if out else h.new_zeros(0)

    def forward(self, ids, positions, d_ids=None, d_positions=None):
        lp = self._logps(ids, positions)
        d_lp = self._logps(d_ids, d_positions) if d_ids is not None else None
        return lp, d_lp


def example_loss(model, ex, args, obs_classes, dilute_pool, rng, device):
    ids = torch.tensor(ex["input_ids"][: args.max_len], device=device)
    cls = torch.tensor(ex["token_class"][: args.max_len], device=device)
    pos = torch.arange(len(ids), device=device)
    act_pos = pos[(cls == ACTION) & (pos > 0)]
    obs_mask = torch.zeros_like(cls, dtype=torch.bool)
    for c in obs_classes:
        obs_mask |= cls == c
    obs_pos = pos[obs_mask & (pos > 0)]
    stats = {"n_act": int(act_pos.numel()), "n_obs": int(obs_pos.numel())}

    supervise_obs = args.lam > 0 and args.dilute is None
    tgt = torch.cat([act_pos, obs_pos]) if supervise_obs else act_pos
    d_ids = d_pos = None
    if args.dilute is not None and args.lam > 0 and obs_pos.numel() > 0:
        # --dilute: token-count-matched unrelated text (|D| = |O| of this example), pooled as in the loss above
        n = int(obs_pos.numel())
        chunk = list(dilute_pool[rng.randrange(len(dilute_pool))])
        while len(chunk) < n + 1:  # a few examples have |O| > one 4096-token chunk
            chunk += dilute_pool[rng.randrange(len(dilute_pool))]
        start = rng.randrange(max(1, len(chunk) - n - 1))
        d_ids = torch.tensor(chunk[start: start + n + 1], device=device)
        d_pos = torch.arange(1, len(d_ids), device=device)
    lp, d_lp = model(ids, tgt, d_ids, d_pos)
    lp_act = lp[: act_pos.numel()]
    lp_obs = lp[act_pos.numel():]
    sum_a = -lp_act.sum()

    if d_lp is not None:
        sum_o = -d_lp.sum()
        n_o = d_lp.numel()
        stats["dilute_ce"] = float(-d_lp.mean().detach())
    else:
        sum_o = -lp_obs.sum()
        n_o = lp_obs.numel()

    n_a = lp_act.numel()
    if args.norm == "pooled":
        denom = n_a + args.lam * n_o
        loss = (sum_a + args.lam * sum_o) / max(denom, 1e-6)
    else:
        loss = sum_a / max(n_a, 1) + (args.lam * sum_o / max(n_o, 1) if n_o else 0.0)
    if n_a:
        stats["act_ce"] = float(-lp_act.mean().detach())
    if lp_obs.numel():
        stats["obs_ce"] = float(-lp_obs.mean().detach())
    return loss, stats


def main():
    args = parse()
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    local = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local)
    device = torch.device("cuda", local)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    obs_classes = [int(c) for c in args.obs_classes.split(",") if c]

    data = load_jsonl(args.data)
    dilute_pool = [d["input_ids"] for d in load_jsonl(args.dilute)] if args.dilute else None
    accum = args.global_batch // world
    assert accum * world == args.global_batch, "global_batch must be divisible by world size"
    steps_per_epoch = len(data) // args.global_batch
    total_steps = max(1, int(steps_per_epoch * args.epochs))
    if args.max_steps > 0:
        total_steps = min(total_steps, args.max_steps)
    warmup = max(1, int(round(args.warmup_ratio * total_steps)))

    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32,
                                                 attn_implementation="sdpa").to(device)
    if not args.no_grad_ckpt:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.use_cache = False
    model.train()
    ddp = DDP(PositionLogProbs(model), device_ids=[local], gradient_as_bucket_view=True)
    # ZeRO-1: AdamW states sharded over ranks (same update as plain AdamW; Addendum 1)
    opt = ZeroRedundancyOptimizer(model.parameters(), optimizer_class=torch.optim.AdamW, lr=args.lr,
                                  betas=(0.9, 0.95), eps=1e-8, weight_decay=args.weight_decay, fused=True)

    def lr_at(step):
        if step < warmup:
            return args.lr * (step + 1) / warmup
        prog = (step - warmup) / max(1, total_steps - warmup)
        return args.lr * 0.5 * (1 + math.cos(math.pi * prog))

    # identical example order across arms: permutation depends only on the seed
    order = list(range(len(data)))
    random.Random(args.seed).shuffle(order)
    rng = random.Random(args.seed * 1000 + rank)
    os.makedirs(args.out, exist_ok=True)
    log = open(os.path.join(args.out, f"train_log.rank{rank}.jsonl"), "w")
    if rank == 0:
        meta = vars(args) | {"world": world, "accum": accum, "total_steps": total_steps, "warmup": warmup,
                             "n_examples": len(data), "data_sha256": sha256_file(args.data)}
        json.dump(meta, open(os.path.join(args.out, "run_args.json"), "w"), indent=1)

    t0 = time.time()
    for step in range(total_steps):
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        agg = {"loss": 0.0, "n_act": 0, "n_obs": 0, "act_ce": [], "obs_ce": [], "dilute_ce": []}
        for k in range(accum):
            idx = order[(step * args.global_batch + k * world + rank) % len(order)]
            ctx = ddp.no_sync() if k < accum - 1 else torch.enable_grad()
            with ctx, torch.autocast("cuda", dtype=torch.bfloat16):
                loss, st = example_loss(ddp, data[idx], args, obs_classes, dilute_pool, rng, device)
                (loss / accum).backward()
            agg["loss"] += float(loss.detach()) / accum
            agg["n_act"] += st["n_act"]
            agg["n_obs"] += st["n_obs"]
            for key in ("act_ce", "obs_ce", "dilute_ce"):
                if key in st:
                    agg[key].append(st[key])
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        if step % args.log_every == 0 or step == total_steps - 1:
            rec = {"step": step, "lr": lr_at(step), "loss": agg["loss"], "grad_norm": float(gnorm),
                   "n_act": agg["n_act"], "n_obs": agg["n_obs"], "elapsed_s": round(time.time() - t0, 1),
                   "max_mem_gb": round(torch.cuda.max_memory_allocated() / 1e9, 1)}
            for key in ("act_ce", "obs_ce", "dilute_ce"):
                if agg[key]:
                    rec[key] = sum(agg[key]) / len(agg[key])
            log.write(json.dumps(rec) + "\n")
            log.flush()
            if rank == 0:
                print(json.dumps(rec), flush=True)
    dist.barrier()
    if rank == 0:
        model.to(torch.bfloat16).save_pretrained(os.path.join(args.out, "model"), safe_serialization=True)
        AutoTokenizer.from_pretrained(args.model).save_pretrained(os.path.join(args.out, "model"))
        json.dump({"status": "done", "steps": total_steps, "elapsed_s": round(time.time() - t0, 1)},
                  open(os.path.join(args.out, "completed.json"), "w"))
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
