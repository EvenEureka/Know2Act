"""Turn rollout records (rollout.py --save_tokens) into SFT examples with per-token classes.

input_ids = prompt_ids + segment ids (a dropped trailing observation excluded) + [EOS if appended],
i.e. exactly the sequence SkyRL's GRPO trains on. token_class:
  0 prompt / chat template, 1 action (incl. the appended EOS), 2 observation payload, 3 observation
  wrapper. An observation is "\n\n<observation>{payload}\n<reminder>...</reminder></observation>\n\n";
tokens overlapping the payload characters are class 2, the rest class 3.

Pre-registered filter: keep trajectories that finished (stop_reason != "length") with a well-formed
answer (reward >= 0); wrong-but-well-formed answers are kept (Pi et al. 2602.21193 found no success
filtering best on synthetic terminal data).
"""
import argparse
import glob
import hashlib
import json
import os
import random

from transformers import AutoTokenizer

OPEN, REMIND = "\n\n<observation>", "\n<reminder>"


def obs_classes(tok, text, ids):
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    if enc["input_ids"] != list(ids):
        raise ValueError("observation ids differ from re-tokenized text")
    a = len(OPEN) if text.startswith(OPEN) else 0
    b = text.rfind(REMIND)
    b = b if b >= 0 else len(text)
    return [2 if (s < b and e > a) else 3 for s, e in enc["offset_mapping"]]


def build_example(rec, tok):
    ids = list(rec["prompt_ids"])
    cls = [0] * len(ids)
    for seg in rec["segments"]:
        if seg.get("dropped"):
            continue
        ids += seg["ids"]
        if seg["type"] == "act":
            cls += [1] * len(seg["ids"])
        else:
            cls += obs_classes(tok, seg["text"], seg["ids"])
    if rec["eos_appended"]:
        ids.append(tok.eos_token_id)
        cls.append(1)
    return {"id": f'{rec["qid"]}_{rec["sample"]}', "qid": rec["qid"], "reward": rec["reward"],
            "input_ids": ids, "token_class": cls}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rollouts", required=True, help="glob of rollout jsonl files")
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--max_len", type=int, default=12288)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    stats = {"read": 0, "kept": 0, "drop_length": 0, "drop_format": 0, "drop_too_long": 0,
             "correct": 0, "tok_act": 0, "tok_payload": 0, "tok_wrapper": 0}
    out = []
    for fn in sorted(glob.glob(args.rollouts)):
        for line in open(fn):
            rec = json.loads(line)
            stats["read"] += 1
            if rec["stop_reason"] == "length":
                stats["drop_length"] += 1
                continue
            if rec["reward"] < 0:
                stats["drop_format"] += 1
                continue
            ex = build_example(rec, tok)
            if len(ex["input_ids"]) > args.max_len:
                stats["drop_too_long"] += 1
                continue
            stats["kept"] += 1
            stats["correct"] += rec["reward"] == 1.0
            for c, key in ((1, "tok_act"), (2, "tok_payload"), (3, "tok_wrapper")):
                stats[key] += sum(1 for x in ex["token_class"] if x == c)
            out.append(ex)
    random.Random(args.seed).shuffle(out)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        for ex in out:
            f.write(json.dumps(ex) + "\n")
    obs = stats["tok_payload"] + stats["tok_wrapper"]
    stats["obs_token_share"] = round(obs / max(1, obs + stats["tok_act"]), 4)
    stats["sha256"] = hashlib.sha256(open(args.out, "rb").read()).hexdigest()
    json.dump(stats, open(args.out + ".stats.json", "w"), indent=1)
    print(json.dumps(stats))


if __name__ == "__main__":
    main()
