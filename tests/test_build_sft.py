"""CPU check: rollout record -> SFT example reproduces the GRPO token stream and labels observations."""
import concurrent.futures as cf
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "tests"))
import build_sft_data as bs  # noqa: E402
import rollout as ro  # noqa: E402
from test_rollout import DB, ROW, TOK, args, scripted  # noqa: E402


class TestBuildSFT(unittest.TestCase):
    def test_example_matches_stream_and_classes(self):
        t1 = "<think>count singers</think>\n<sql>SELECT count(*) FROM singer</sql>"
        t2 = "<think>six</think>\n<solution>SELECT count(*) FROM singer</solution>"
        row = ROW.iloc[0]
        p = TOK.apply_chat_template(list(row["prompt"]), add_generation_prompt=True, tokenize=True,
                                    return_dict=False)
        t = ro.Traj(int(ROW.index[0]), 0, row, p, ro.make_env(row, DB, 6))
        with cf.ThreadPoolExecutor(2) as pool:
            ro.run_chunk([t], scripted([t1, t2]), TOK, args(), pool)
        ex = bs.build_example(ro.record(t, save_tokens=True), TOK)
        self.assertEqual(ex["input_ids"], t.ids + [TOK.eos_token_id])
        cls = ex["token_class"]
        self.assertEqual(len(cls), len(ex["input_ids"]))
        self.assertEqual(cls[: len(p)], [0] * len(p))
        a1, o1, a2 = (s for s in t.segments)
        n_act = len(a1["ids"]) + len(a2["ids"]) + 1
        self.assertEqual(sum(c == 1 for c in cls), n_act)
        payload = TOK.decode([i for i, c in zip(ex["input_ids"], cls) if c == 2])
        wrapper = TOK.decode([i for i, c in zip(ex["input_ids"], cls) if c == 3])
        self.assertIn("6", payload)
        self.assertNotIn("reminder", payload)
        self.assertIn("<observation>", wrapper)
        self.assertIn("turns left", wrapper)
        self.assertEqual(sum(c in (2, 3) for c in cls), len(o1["ids"]))


if __name__ == "__main__":
    unittest.main()
