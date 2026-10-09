"""D2 diagnostic: observation-token CE split by outcome type and by part (teacher-forced, one GPU).

Each observation segment (a contiguous run of token classes 2/3) is typed from its payload text:
  error  "Error executing SQL: ..."    empty  "Empty DataFrame ..."    rows  any other result table
and its tokens are split into parts:
  wrapper  class 3 (the <observation>/<reminder> template)
  first3   the first 3 payload tokens (which outcome happened)
  line1    the rest of the first payload line (column header for `rows`, message for `error`)
  rest     the remaining payload (row values for `rows`)
Run on held-out trajectories (12 unseen DBs) and on training trajectories (seen once in SFT) to
separate what generalises from what was memorised.
"""
import argparse
import collections
import json

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from measure import load_jsonl


def obs_segments(cls):
    runs, s = [], None
    for i, c in enumerate(cls + [0]):
        if c in (2, 3) and s is None:
            s = i
        elif c not in (2, 3) and s is not None:
            runs.append((s, i))
            s = None
    return runs


def outcome(text):
    if "Error executing SQL" in text:
        return "error"
    if "Empty DataFrame" in text:
        return "empty"
    return "rows"


@torch.no_grad()
def token_logps(model, ids, dev, chunk=2048):
    """lp[t] = log p(ids[t] | ids[:t]) for t >= 1 (lp[0] unused)."""
    x = torch.tensor(ids, device=dev)
    h = model.model(input_ids=x[None]).last_hidden_state[0][:-1]
    lp = torch.zeros(len(ids))
    for s in range(0, h.shape[0], chunk):
        out = torch.log_softmax(model.lm_head(h[s:s + chunk]).float(), -1)
        lp[1 + s:1 + s + out.shape[0]] = out.gather(1, x[1 + s:1 + s + chunk][:, None])[:, 0].cpu()
    return lp


def breakdown(model, tok, rows, dev, max_len):
    nll, cnt = collections.Counter(), collections.Counter()
    for ex in rows:
        ids, cls = ex["input_ids"][:max_len], ex["token_class"][:max_len]
        lp = token_logps(model, ids, dev)
        act = [t for t in range(1, len(ids)) if cls[t] == 1]
        nll["act"] += float(-lp[act].sum())
        cnt["act"] += len(act)
        for s, e in obs_segments(cls):
            pay = [t for t in range(s, e) if cls[t] == 2]
            kind = outcome(tok.decode([ids[t] for t in pay]))
            nl = next((i for i, t in enumerate(pay) if "\n" in tok.decode([ids[t]])), len(pay))
            parts = {"wrapper": [t for t in range(s, e) if cls[t] == 3], "first3": pay[:3],
                     "line1": pay[3:max(3, nl + 1)], "rest": pay[max(3, nl + 1):]}
            cnt[f"seg/{kind}"] += 1
            for part, ts in parts.items():
                ts = [t for t in ts if t >= 1]
                nll[f"{kind}/{part}"] += float(-lp[ts].sum())
                cnt[f"{kind}/{part}"] += len(ts)
    return {k: {"ce": nll[k] / cnt[k] if cnt[k] else None, "n_tokens": cnt[k]} for k in nll} | \
           {k: cnt[k] for k in cnt if k.startswith("seg/")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True, help="NAME=path")
    ap.add_argument("--data", nargs="+", required=True, help="NAME=jsonl")
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=300)
    ap.add_argument("--max_len", type=int, default=12288)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    dev = torch.device("cuda")
    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    sets = {d.split("=", 1)[0]: load_jsonl(d.split("=", 1)[1], a.limit, a.seed) for d in a.data}
    res = {}
    for spec in a.models:
        name, path = spec.split("=", 1)
        model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16).to(dev).eval()
        res[name] = {ds: breakdown(model, tok, rows, dev, a.max_len) for ds, rows in sets.items()}
        json.dump(res, open(a.out, "w"), indent=1)  # partial results survive a later failure
        print(name, json.dumps({ds: {k: v for k, v in r.items() if k.endswith(("act", "first3", "rest"))}
                                for ds, r in res[name].items()}), flush=True)
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
