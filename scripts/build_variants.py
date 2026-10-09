"""SFT data variants for the follow-up on why observation supervision leaves the action policy
unchanged (PREREG Addendum 6, 2026-10-07). All variants start from data/sft/{train,heldout}.jsonl.

  e1         Auxiliary outcome prediction: after the </sql> call of every action that is followed
             by an observation, insert the same outcome line as e2 ("\nExpected result: ...") as an
             auxiliary target (class 2), i.e. in the observation slot, never generated at test time.
             The real observation tokens become class 4 (unsupervised). Train with lam 1, which
             weights the line exactly as e2 does (both are pooled with the action tokens).
  e2         Predict-then-act: before the <sql> call of every action that is followed by an
             observation, insert one line stating the outcome, e.g. "Expected result: 3 rows, 2
             columns." / "Expected result: no rows." / "Expected result: error (no such column: GPA).",
             derived from the real observation. The line is part of the action (class 1) and is
             trained with action-only SFT (lam 0). `expect_spans` records the inserted token ranges.
  obsdilute  Dilution pool for control D_obs: observation tokens (classes 2/3) of training
             trajectories, concatenated in shuffled order and cut into 4096-token chunks. Used with
             --dilute and lam 1 like control D, but the extra text is SQL observations seen out of
             context (hard to predict) instead of Wikipedia.
"""
import argparse
import collections
import json
import random

from transformers import AutoTokenizer

INVALID = "Your previous action is invalid"
ERROR = "Error executing SQL: "
TRUNC = "Truncated to 50 lines since returned response too long: "


def runs(cls, keep):
    """Contiguous [start, end) runs of positions whose class is in `keep`."""
    out, s = [], None
    for i, c in enumerate(list(cls) + [-1]):
        if c in keep and s is None:
            s = i
        elif c not in keep and s is not None:
            out.append((s, i))
            s = None
    return out


def parse_obs(payload):
    """Outcome of one observation payload: (kind, label for e2, character cut for the e1 line)."""
    p = payload[1:] if payload.startswith(">") else payload
    off = len(payload) - len(p)
    if p.startswith(INVALID):
        return "invalid", None, off + len(p.split("\n")[0])
    if p.startswith(ERROR):
        msg = p[len(ERROR):].split(", db file:")[0].strip()
        return "error", f"error ({msg})", off + len(ERROR) + len(p[len(ERROR):].split(", db file:")[0])
    if p.startswith("SQL Timeout"):
        return "timeout", "timeout", off + len("SQL Timeout")
    if p.startswith("Empty DataFrame"):
        return "empty", "no rows", off + len("Empty DataFrame")
    trunc = p.startswith(TRUNC)
    table = p[len(TRUNC):] if trunc else p
    lines = table.rstrip("\n").split("\n")
    n_cols, n_rows = len(lines[0].split()), len(lines) - 1
    rows = "more than 50 rows" if trunc else f"{n_rows} row{'s' if n_rows != 1 else ''}"
    cut = off + (len(TRUNC) if trunc else 0) + len(lines[0])
    return "rows", f"{rows}, {n_cols} column{'s' if n_cols != 1 else ''}", cut


def make_e1(rec, tok, stats):
    """The e2 outcome line, but placed after </sql> as an auxiliary target (class 2, not generated
    at test time); the real observation tokens become class 4 (unsupervised)."""
    ids, cls = rec["input_ids"], rec["token_class"]
    acts, obs = runs(cls, (1,)), runs(cls, (2, 3))
    obs_at = {s: e for s, e in obs}
    new_ids, new_cls, spans, pos = [], [], [], 0
    for s, e in acts:
        new_ids += ids[pos:e]
        new_cls += cls[pos:e]
        pos = e
        if e in obs_at and "<sql>" in tok.decode(ids[s:e]):
            pay = [ids[t] for t in range(e, obs_at[e]) if cls[t] == 2]
            kind, label, _ = parse_obs(tok.decode(pay))
            if label is not None:
                line = tok.encode(f"\nExpected result: {label}.", add_special_tokens=False)
                spans.append([len(new_ids), len(new_ids) + len(line)])
                new_ids += line
                new_cls += [2] * len(line)
                stats[f"e1_kind/{kind}"] += 1
                stats["e1_label_tokens"] += len(line)
    new_ids += ids[pos:]
    new_cls += cls[pos:]
    new_cls = [4 if c in (2, 3) else c for c in new_cls]
    for a, b in spans:
        new_cls[a:b] = [2] * (b - a)
    return rec | {"input_ids": new_ids, "token_class": new_cls, "expect_spans": spans}


def make_e2(rec, tok, stats):
    ids, cls = rec["input_ids"], rec["token_class"]
    acts, obs = runs(cls, (1,)), runs(cls, (2, 3))
    obs_after = {s: (s2, e2) for (s, e) in acts for (s2, e2) in obs if s2 == e}
    new_ids, new_cls, spans, pos = [], [], [], 0
    for s, e in acts:
        new_ids += ids[pos:s]
        new_cls += cls[pos:s]
        seg = ids[s:e]
        if s in obs_after:
            o_s, o_e = obs_after[s]
            pay = [ids[t] for t in range(o_s, o_e) if cls[t] == 2]
            kind, label, _ = parse_obs(tok.decode(pay))
            text = tok.decode(seg)
            i = text.rfind("<sql>")
            if label is not None and i >= 0:
                pre = text[:i]
                line = ("" if pre.endswith("\n") or not pre else "\n") + f"Expected result: {label}.\n"
                a = tok.encode(pre, add_special_tokens=False)
                b = tok.encode(line, add_special_tokens=False)
                c = tok.encode(text[i:], add_special_tokens=False)
                if tok.decode(a + b + c) == pre + line + text[i:]:
                    spans.append([len(new_ids) + len(a), len(new_ids) + len(a) + len(b)])
                    seg = a + b + c
                    stats[f"e2_kind/{kind}"] += 1
                else:
                    stats["e2_skip_roundtrip"] += 1
            else:
                stats[f"e2_noinsert/{kind}"] += 1
        new_ids += seg
        new_cls += [1] * len(seg)
        pos = e
    new_ids += ids[pos:]
    new_cls += cls[pos:]
    return rec | {"input_ids": new_ids, "token_class": new_cls, "expect_spans": spans}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--out_dir", default="data/sft")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    report = {}
    for split in ("train", "heldout"):
        recs = [json.loads(l) for l in open(f"{a.out_dir}/{split}.jsonl") if l.strip()]
        st = collections.Counter()
        with open(f"{a.out_dir}/{split}_e1.jsonl", "w") as f1, open(f"{a.out_dir}/{split}_e2.jsonl", "w") as f2:
            for r in recs:
                f1.write(json.dumps(make_e1(r, tok, st)) + "\n")
                f2.write(json.dumps(make_e2(r, tok, st)) + "\n")
        report[split] = dict(st)
        print(split, json.dumps(report[split]), flush=True)
        if split == "train":
            pool = [r["input_ids"][s:e] for r in recs for s, e in runs(r["token_class"], (2, 3))]
            random.Random(a.seed).shuffle(pool)
            flat = [t for seg in pool for t in seg]
            with open(f"{a.out_dir}/dilution_obs.jsonl", "w") as f:
                for i in range(0, len(flat) - 4096 + 1, 4096):
                    f.write(json.dumps({"input_ids": flat[i:i + 4096]}) + "\n")
            report["obsdilute"] = {"tokens": len(flat), "chunks": len(flat) // 4096}
    json.dump(report, open(f"{a.out_dir}/variants_report.json", "w"), indent=1)


if __name__ == "__main__":
    main()
