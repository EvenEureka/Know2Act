"""Tokenise ALF-CF text arms (alfcf_build_v3.py output) into sft_train.py examples (training env).

input_ids = chat-template prompt (system + user, generation prompt, thinking off) + segment ids, each segment
tokenised on its own (segment boundaries fall on newlines or on '<action>', which tokenise stably for Qwen3).
token_class: 0 prompt; 1 assistant text (note line + action); 2 observation payload; 3 observation wrapper.
usage: python alfcf_tokenize.py <text_arm.jsonl> <out.jsonl> <tokenizer_dir> [--max_len 12288]
"""
import argparse
import hashlib
import json

from transformers import AutoTokenizer

OPEN, CLOSE = "\n<observation>", "</observation>\n"


def prompt_ids(tok, system, user):
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    txt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    return list(tok(txt, add_special_tokens=False)["input_ids"])


def seg_ids(tok, seg):
    if seg["type"] == "act":
        if seg.get("mask_spans"):  # Addendum 20 Fix 1: mask only tokens overlapping the given character spans
            enc = tok(seg["text"], add_special_tokens=False, return_offsets_mapping=True)
            spans = seg["mask_spans"]
            cls = [0 if any(s < b and e > a for a, b in spans) else 1 for s, e in enc["offset_mapping"]]
            return enc["input_ids"], cls
        ids = tok(seg["text"], add_special_tokens=False)["input_ids"]
        return ids, [1 if seg.get("loss", True) else 0] * len(ids)  # Addendum 19: loss-masked note/action segments
    enc = tok(seg["text"], add_special_tokens=False, return_offsets_mapping=True)
    a, b = len(OPEN), len(seg["text"]) - len(CLOSE)
    cls = [2 if (s < b and e > a) else 3 for s, e in enc["offset_mapping"]]
    return enc["input_ids"], cls


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("text")
    ap.add_argument("out")
    ap.add_argument("tokenizer")
    ap.add_argument("--max_len", type=int, default=12288)
    a = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    st = {"read": 0, "kept": 0, "drop_too_long": 0, "tok_act": 0, "tok_payload": 0, "tok_wrapper": 0, "tok_prompt": 0,
          "max_len_seen": 0}
    with open(a.out, "w") as fo:
        for l in open(a.text):
            ex = json.loads(l)
            st["read"] += 1
            ids = prompt_ids(tok, ex["system"], ex["user"])
            cls = [0] * len(ids)
            for s in ex["segments"]:
                i, c = seg_ids(tok, s)
                ids += i
                cls += c
            st["max_len_seen"] = max(st["max_len_seen"], len(ids))
            if len(ids) > a.max_len:
                st["drop_too_long"] += 1
                continue
            st["kept"] += 1
            st["tok_prompt"] += cls.count(0)
            st["tok_act"] += cls.count(1)
            st["tok_payload"] += cls.count(2)
            st["tok_wrapper"] += cls.count(3)
            fo.write(json.dumps({"id": ex["id"], "input_ids": ids, "token_class": cls}) + "\n")
    h = hashlib.sha256(open(a.out, "rb").read()).hexdigest()
    st["sha256"] = h
    json.dump(st, open(a.out + ".stats.json", "w"), indent=1)
    print(json.dumps(st))


if __name__ == "__main__":
    main()
