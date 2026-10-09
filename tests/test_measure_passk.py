"""CPU checks for scripts/measure.py (per-class CE, KL) and scripts/passk.py (unbiased estimator)."""
import math
import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import measure  # noqa: E402
import passk  # noqa: E402
from test_sft_loss import EX, manual_logps, tiny_lm  # noqa: E402


class TestPassK(unittest.TestCase):
    def test_known_values(self):
        self.assertAlmostEqual(passk.pass_at_k(16, 4, 1), 0.25)
        self.assertEqual(passk.pass_at_k(16, 1, 16), 1.0)
        self.assertEqual(passk.pass_at_k(16, 0, 8), 0.0)
        # 1 - C(12,2)/C(16,2)
        self.assertAlmostEqual(passk.pass_at_k(16, 4, 2), 1 - math.comb(12, 2) / math.comb(16, 2))


class TestMeasure(unittest.TestCase):
    def test_ce_and_self_kl(self):
        m = tiny_lm(0)
        st = measure.forward_stats(m, m, EX, torch.device("cpu"), 10_000)
        lp = manual_logps(m, EX["input_ids"])
        cls = EX["token_class"][1:]
        act = [-l for l, c in zip(lp, cls) if c == 1]
        pay = [-l for l, c in zip(lp, cls) if c == 2]
        wra = [-l for l, c in zip(lp, cls) if c == 3]
        self.assertAlmostEqual(st["ce_act"], float(sum(act) / len(act)), places=4)
        self.assertAlmostEqual(st["ce_payload"], float(sum(pay) / len(pay)), places=4)
        self.assertAlmostEqual(st["ce_wrapper"], float(sum(wra) / len(wra)), places=4)
        self.assertAlmostEqual(st["kl_act"], 0.0, places=5)  # KL to itself
        self.assertEqual((st["n_act"], st["n_payload"], st["n_wrapper"]), (8, 6, 2))

    def test_grad_stats_match_full_vectors(self):
        """Streaming per-parameter cos/ratio must equal the naive concatenated-vector computation."""
        import torch.nn.functional as F
        m = tiny_lm(0)
        m.train()
        cos, ratio = measure.grad_stats(m, [EX], torch.device("cpu"), 10_000)
        vecs = {}
        for key, classes in (("act", (1,)), ("obs", (2, 3))):
            m.zero_grad(set_to_none=True)
            ids = torch.tensor(EX["input_ids"])
            logits = m(input_ids=ids[None]).logits[0].float()
            pos = [t for t in range(1, len(ids)) if EX["token_class"][t] in classes]
            F.cross_entropy(logits[[t - 1 for t in pos]], ids[pos], reduction="sum").backward()
            vecs[key] = torch.cat([p.grad.flatten() for p in m.parameters() if p.grad is not None])
        self.assertAlmostEqual(cos, float(F.cosine_similarity(vecs["act"], vecs["obs"], dim=0)), places=4)
        self.assertAlmostEqual(ratio, float(vecs["obs"].norm() / vecs["act"].norm()), places=4)

    def test_kl_positive_for_different_model(self):
        st = measure.forward_stats(tiny_lm(0), tiny_lm(1), EX, torch.device("cpu"), 10_000)
        self.assertGreater(st["kl_act"], 0.0)


if __name__ == "__main__":
    unittest.main()
