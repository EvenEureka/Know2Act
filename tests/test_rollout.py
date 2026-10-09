"""CPU checks that scripts/rollout.py builds SkyRL's recipe-mode token stream (fake engine, real env)."""
import concurrent.futures as cf
import glob
import os
import sys
import types
import unittest
from pathlib import Path

import pandas as pd
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "skyrl-upstream" / "skyrl-gym"))
import rollout as ro  # noqa: E402

# These tests need the built data (scripts/build_datasets.py, scripts/extract_omnisql.py) and a local
# Qwen3-4B tokenizer; without them the module is skipped rather than failed.
HF_HOME = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
SNAPSHOTS = glob.glob(f"{HF_HOME}/hub/models--Qwen--Qwen3-4B/snapshots/*/")
DB = str(ROOT / "data" / "omnisql" / "data")
if not SNAPSHOTS or not (ROOT / "data" / "eval" / "spider_dev.parquet").exists() or not os.path.isdir(DB):
    raise unittest.SkipTest("needs data/eval/spider_dev.parquet, data/omnisql/ and a cached Qwen3-4B tokenizer")
TOK = AutoTokenizer.from_pretrained(SNAPSHOTS[0])
ROW = pd.read_parquet(ROOT / "data" / "eval" / "spider_dev.parquet").iloc[[0]]  # concert_singer, count singers


def args(**kw):
    base = dict(max_input_length=29000, seed=0, max_turns=6)
    base.update(kw)
    return types.SimpleNamespace(**base)


def scripted(turn_texts, eos_on_last=True, finish=None):
    """Fake generate: turn k of every trajectory returns turn_texts[k] tokenized (EOS appended on last)."""
    turn = {"k": 0}

    def gen(batch):
        k = turn["k"]
        turn["k"] += 1
        text = turn_texts[k]
        ids = TOK.encode(text, add_special_tokens=False)
        if eos_on_last and k == len(turn_texts) - 1:
            ids = ids + [TOK.eos_token_id]
        fin = finish[k] if finish else "stop"
        return [(list(ids), text, fin) for _ in batch]
    return gen


class TestRollout(unittest.TestCase):
    def run_one(self, gen, a):
        row = ROW.iloc[0]
        prompt_ids = TOK.apply_chat_template(list(row["prompt"]), add_generation_prompt=True,
                                             tokenize=True, return_dict=False)
        t = ro.Traj(int(ROW.index[0]), 0, row, prompt_ids, ro.make_env(row, DB, a.max_turns))
        with cf.ThreadPoolExecutor(2) as pool:
            ro.run_chunk([t], gen, TOK, a, pool)
        return t, prompt_ids

    def test_two_turn_success(self):
        t1 = "<think>count singers</think>\n<sql>SELECT count(*) FROM singer</sql>"
        t2 = "<think>6 rows</think>\n<solution>SELECT count(*) FROM singer</solution>"
        t, p = self.run_one(scripted([t1, t2]), args())
        self.assertEqual(t.reward, 1.0)
        self.assertEqual([s["type"] for s in t.segments], ["act", "obs", "act"])
        a1, o1, a2 = t.segments
        self.assertEqual(a1["ids"], TOK.encode(t1, add_special_tokens=False))
        self.assertTrue(o1["text"].startswith("\n\n<observation>") and "5 turns left" in o1["text"])
        self.assertEqual(o1["ids"], TOK.encode(o1["text"], add_special_tokens=False))
        self.assertEqual(a2["ids"], TOK.encode(t2, add_special_tokens=False))  # EOS stripped
        self.assertEqual(t.ids, p + a1["ids"] + o1["ids"] + a2["ids"])
        self.assertTrue(t.eos_appended)
        self.assertEqual(t.response_end, len(t.ids) - 1)

    def test_invalid_then_max_turns(self):
        bad = "<think>hmm</think> no tool call"
        t, _ = self.run_one(scripted([bad] * 2, eos_on_last=False), args(max_turns=2))
        self.assertEqual(t.turns, 2)
        self.assertEqual(t.reward, -1.0)  # no <solution> before max turns -> format fail
        self.assertIn("invalid", t.segments[1]["text"])

    def test_length_finish_no_eos(self):
        t1 = "<think>long"
        t, _ = self.run_one(scripted([t1, t1], eos_on_last=False, finish=["length", "length"]),
                            args(max_turns=2))
        self.assertEqual(t.stop_reason, "length")
        self.assertFalse(t.eos_appended)

    def test_max_input_length_stop(self):
        t1 = "<think>x</think>\n<sql>SELECT count(*) FROM singer</sql>"
        t, p = self.run_one(scripted([t1] * 6), args(max_input_length=len(TOK.apply_chat_template(
            list(ROW.iloc[0]["prompt"]), add_generation_prompt=True, tokenize=True, return_dict=False)) + 5))
        self.assertEqual(t.turns, 1)  # second turn is skipped: context already over the cap
        self.assertEqual(t.stop_reason, "length")
        self.assertTrue(t.segments[-1].get("dropped"))  # trailing observation is not part of the response
        self.assertEqual(t.reward, -1.0)


if __name__ == "__main__":
    unittest.main()
