"""Unrelated-text token chunks for control D: Wikipedia paragraphs from cached HotpotQA (distractor).

Paragraphs are de-duplicated by title, joined with blank lines, and cut into fixed-size chunks of
Qwen3 tokens. sft_train.py samples a |O|-token window from a random chunk for each example.
"""
import argparse
import glob
import hashlib
import json
import os
import random

import pandas as pd
from transformers import AutoTokenizer


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--target_tokens", type=int, default=30_000_000)
    p.add_argument("--chunk", type=int, default=4096)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    hf_home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    files = sorted(glob.glob(f"{hf_home}/hub/datasets--hotpotqa--hotpot_qa/snapshots/*/distractor/train-*.parquet"))
    seen, paras = set(), []
    for fn in files:
        for ctx in pd.read_parquet(fn, columns=["context"])["context"]:
            for title, sents in zip(ctx["title"], ctx["sentences"]):
                if title in seen:
                    continue
                seen.add(title)
                paras.append(title + "\n" + "".join(sents))
    random.Random(a.seed).shuffle(paras)
    buf, n_tok, chunks = [], 0, []
    for para in paras:
        buf.extend(tok.encode(para + "\n\n", add_special_tokens=False))
        while len(buf) >= a.chunk:
            chunks.append(buf[: a.chunk])
            buf = buf[a.chunk:]
            n_tok += a.chunk
        if n_tok >= a.target_tokens:
            break
    with open(a.out, "w") as f:
        for c in chunks:
            f.write(json.dumps({"input_ids": c}) + "\n")
    stats = {"paragraphs_available": len(paras), "chunks": len(chunks), "tokens": n_tok,
             "sha256": hashlib.sha256(open(a.out, "rb").read()).hexdigest()}
    json.dump(stats, open(a.out + ".stats.json", "w"), indent=1)
    print(stats)


if __name__ == "__main__":
    main()
