"""Mechanism measurements on held-out SFT-format trajectories (teacher-forced, one GPU).

For a checkpoint M and a reference R (base model for SFT checkpoints, the SFT init for GRPO ones):
  * mean CE per token class (1 action, 2 observation payload, 3 observation wrapper)
  * action tokens: KL(M || R) over the full vocabulary, entropy of M, top-1 - top-2 logit margin
  * --grad: cos(g_obs, g_act) and ||g_obs|| / ||g_act||, gradients summed over tokens (as in
    ActObs §5.1), computed per partition of the trajectories (4 partitions -> mean and std)
Writes one JSON with the per-class means and per-trajectory values.
"""
import argparse
import json
import math
import random

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM


def load_jsonl(path, limit, seed):
    rows = [json.loads(line) for line in open(path) if line.strip()]
    random.Random(seed).shuffle(rows)
    return rows[:limit] if limit > 0 else rows


@torch.no_grad()
def forward_stats(model, ref, ex, device, max_len):
    ids = torch.tensor(ex["input_ids"][:max_len], device=device)
    cls = torch.tensor(ex["token_class"][:max_len], device=device)
    tgt = torch.arange(1, len(ids), device=device)
    sup = cls[1:] > 0
    pos = tgt[sup]
    h = model.model(input_ids=ids[None]).last_hidden_state[0][pos - 1]
    logits = model.lm_head(h).float()
    lp = torch.log_softmax(logits, -1)
    tok_lp = lp.gather(1, ids[pos][:, None])[:, 0]
    c = cls[pos]
    out = {}
    for k, name in ((1, "act"), (2, "payload"), (3, "wrapper")):
        m = c == k
        if m.any():
            out[f"ce_{name}"] = float(-tok_lp[m].mean())
            out[f"n_{name}"] = int(m.sum())
    act = c == 1
    if act.any():
        p = lp[act].exp()
        out["ent_act"] = float(-(p * lp[act]).sum(-1).mean())
        top2 = logits[act].topk(2, -1).values
        out["margin_act"] = float((top2[:, 0] - top2[:, 1]).mean())
        if ref is not None:
            hr = ref.model(input_ids=ids[None]).last_hidden_state[0][pos - 1][act]
            lpr = torch.log_softmax(ref.lm_head(hr).float(), -1)
            out["kl_act"] = float((p * (lp[act] - lpr)).sum(-1).mean())
    return out


def _summed_ce(model, ids, pos, chunk=2048):
    """Sum of token CE at target positions; LM head chunked and recomputed (as in sft_train.py)."""
    from torch.utils.checkpoint import checkpoint
    h = model.model(input_ids=ids[None]).last_hidden_state[0][pos - 1]
    tgt = ids[pos]
    total = 0.0
    for s in range(0, h.shape[0], chunk):
        total = total + checkpoint(
            lambda hs, ts: F.cross_entropy(model.lm_head(hs).float(), ts, reduction="sum"),
            h[s:s + chunk], tgt[s:s + chunk], use_reentrant=False)
    return total


def grad_stats(model, batch, device, max_len):
    """cos(g_act, g_obs) and ||g_obs|| / ||g_act|| for summed-over-tokens gradients of one batch.

    The action gradient is parked on the CPU parameter by parameter and the products are
    accumulated per parameter; concatenating two 4B-element fp32 vectors on the GPU (16 GB each)
    caused the round-1 measurement OOM (Addendum 2).
    """
    parked = None
    dot = n_act = n_obs = 0.0
    for key, classes in (("act", (1,)), ("obs", (2, 3))):
        model.zero_grad(set_to_none=True)
        for ex in batch:
            ids = torch.tensor(ex["input_ids"][:max_len], device=device)
            cls = torch.tensor(ex["token_class"][:max_len], device=device)
            tgt = torch.arange(1, len(ids), device=device)
            m = torch.zeros_like(cls[1:], dtype=torch.bool)
            for k in classes:
                m |= cls[1:] == k
            pos = tgt[m]
            if pos.numel() == 0:
                continue
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                loss = _summed_ce(model, ids, pos)
            loss.backward()
        if key == "act":
            parked = [None if p.grad is None else p.grad.detach().float().cpu() for p in model.parameters()]
            n_act = sum(float((g * g).sum()) for g in parked if g is not None)
        else:
            for p, ga in zip(model.parameters(), parked):
                if p.grad is None:
                    continue
                go = p.grad.detach().float().cpu()
                n_obs += float((go * go).sum())
                if ga is not None:
                    dot += float((ga * go).sum())
    model.zero_grad(set_to_none=True)
    return dot / math.sqrt(n_act * n_obs), math.sqrt(n_obs / n_act)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--ref", default=None)
    ap.add_argument("--data", required=True, help="held-out SFT-format jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=300)
    ap.add_argument("--max_len", type=int, default=12288)
    ap.add_argument("--grad", action="store_true")
    ap.add_argument("--grad_n", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    dev = torch.device("cuda")
    rows = load_jsonl(a.data, a.limit, a.seed)
    model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.bfloat16).to(dev).eval()
    ref = AutoModelForCausalLM.from_pretrained(a.ref, torch_dtype=torch.bfloat16).to(dev).eval() if a.ref else None
    per = [forward_stats(model, ref, ex, dev, a.max_len) | {"id": ex.get("id")} for ex in rows]
    summary = {}
    for key in ("ce_act", "ce_payload", "ce_wrapper", "ent_act", "margin_act", "kl_act"):
        vals = [p[key] for p in per if key in p]
        if vals:
            w = [p.get("n_act" if key in ("ent_act", "margin_act", "kl_act", "ce_act") else
                       "n_" + key.split("_")[1], 1) for p in per if key in p]
            summary[key] = sum(v * x for v, x in zip(vals, w)) / sum(w)  # token-weighted mean
    def dump():
        json.dump({"model": a.model, "ref": a.ref, "n": len(per), "summary": summary, "per_traj": per},
                  open(a.out, "w"))

    dump()  # forward stats survive even if the gradient pass fails
    if a.grad:
        del ref
        torch.cuda.empty_cache()
        gm = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.float32).to(dev)
        gm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        gm.train()
        grows = load_jsonl(a.data, a.grad_n, a.seed + 1)
        parts = [grows[i::4] for i in range(4)]
        cos, ratio = [], []
        for part in parts:
            c, r = grad_stats(gm, part, dev, a.max_len)
            cos.append(c)
            ratio.append(r)
        mean = lambda v: sum(v) / len(v)  # noqa: E731
        std = lambda v: math.sqrt(sum((x - mean(v)) ** 2 for x in v) / max(1, len(v) - 1))  # noqa: E731
        summary |= {"grad_cos": mean(cos), "grad_cos_std": std(cos), "grad_ratio": mean(ratio),
                    "grad_ratio_std": std(ratio)}
    dump()
    print(json.dumps({"model": a.model, "summary": summary}))


if __name__ == "__main__":
    main()
