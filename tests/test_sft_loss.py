"""CPU checks of the SFT loss variants in scripts/sft_train.py against hand-computed values."""
import random
import sys
import types
import unittest
from pathlib import Path

import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import sft_train as st  # noqa: E402


def tiny_lm(seed=0):
    torch.manual_seed(seed)
    cfg = Qwen2Config(vocab_size=97, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=256)
    return Qwen2ForCausalLM(cfg).eval()


def args_for(lam, norm="pooled", dilute=None):
    return types.SimpleNamespace(lam=lam, norm=norm, dilute=dilute, max_len=10_000)


# prompt(0) x4, action(1) x5, wrapper(3) x2, payload(2) x6, action(1) x3
EX = {
    "input_ids": [5, 9, 11, 2, 40, 41, 42, 43, 44, 70, 71, 20, 21, 22, 23, 24, 25, 50, 51, 52],
    "token_class": [0, 0, 0, 0, 1, 1, 1, 1, 1, 3, 3, 2, 2, 2, 2, 2, 2, 1, 1, 1],
}


def manual_logps(lm, ids):
    with torch.no_grad():
        logits = lm(input_ids=torch.tensor(ids)[None]).logits[0].float()
    lp = torch.log_softmax(logits, -1)
    return torch.tensor([lp[t - 1, ids[t]] for t in range(1, len(ids))])  # index t-1 -> token t


class TestLoss(unittest.TestCase):
    def setUp(self):
        self.lm = tiny_lm()
        self.wrapped = st.PositionLogProbs(self.lm)
        lp = manual_logps(self.lm, EX["input_ids"])
        cls = EX["token_class"][1:]
        self.a = torch.tensor([l for l, c in zip(lp, cls) if c == 1])
        self.o = torch.tensor([l for l, c in zip(lp, cls) if c in (2, 3)])

    def run_loss(self, args, dilute_pool=None):
        with torch.no_grad():
            loss, stats = st.example_loss(self.wrapped, EX, args, [2, 3], dilute_pool,
                                          random.Random(0), torch.device("cpu"))
        return float(loss), stats

    def test_action_only(self):
        loss, stats = self.run_loss(args_for(0.0))
        self.assertAlmostEqual(loss, float(-self.a.mean()), places=4)
        self.assertEqual((stats["n_act"], stats["n_obs"]), (8, 8))

    def test_actobs_pooled_eq1(self):
        loss, _ = self.run_loss(args_for(1.0))
        expect = -(self.a.sum() + self.o.sum()) / (len(self.a) + len(self.o))
        self.assertAlmostEqual(loss, float(expect), places=4)

    def test_pooled_lambda(self):
        lam = 0.25
        loss, _ = self.run_loss(args_for(lam))
        expect = -(self.a.sum() + lam * self.o.sum()) / (len(self.a) + lam * len(self.o))
        self.assertAlmostEqual(loss, float(expect), places=4)

    def test_separate(self):
        loss, _ = self.run_loss(args_for(1.0, norm="separate"))
        expect = -(self.a.mean() + self.o.mean())
        self.assertAlmostEqual(loss, float(expect), places=4)

    def test_dilution_count_matched(self):
        pool = [list(range(3, 90))]
        loss, stats = self.run_loss(args_for(1.0, dilute="x"), dilute_pool=pool)
        self.assertIn("dilute_ce", stats)
        # loss pools 8 action + 8 unrelated tokens; recompute from the same chunk the rng picked
        rng = random.Random(0)
        chunk = pool[rng.randrange(1)]
        start = rng.randrange(max(1, len(chunk) - 8 - 1))
        d = manual_logps(self.lm, chunk[start:start + 9])
        expect = -(self.a.sum() + d.sum()) / (len(self.a) + len(d))
        self.assertAlmostEqual(loss, float(expect), places=4)

    def test_chunked_lm_head_matches_unchunked(self):
        """CE_CHUNK only changes memory: loss and gradients must equal the one-chunk computation."""
        def loss_and_grads(chunk):
            old = st.CE_CHUNK
            st.CE_CHUNK = chunk
            try:
                lm = tiny_lm(0)
                lm.train()
                loss, _ = st.example_loss(st.PositionLogProbs(lm), EX, args_for(1.0), [2, 3], None,
                                          random.Random(0), torch.device("cpu"))
                loss.backward()
                return float(loss), [p.grad.clone() for p in lm.parameters() if p.grad is not None]
            finally:
                st.CE_CHUNK = old
        l1, g1 = loss_and_grads(3)
        l2, g2 = loss_and_grads(10_000)
        self.assertAlmostEqual(l1, l2, places=5)
        self.assertEqual(len(g1), len(g2))
        for a, b in zip(g1, g2):
            self.assertTrue(torch.allclose(a, b, atol=1e-6, rtol=1e-4))

    def test_dilution_long_obs_uses_concatenated_chunks(self):
        seen = []

        class Recorder(st.PositionLogProbs):
            def forward(self, ids, positions, d_ids=None, d_positions=None):
                seen.append(None if d_positions is None else len(d_positions))
                return super().forward(ids, positions, d_ids, d_positions)

        short_pool = [list(range(3, 8))]  # 5-token chunks, shorter than |O| = 8
        with torch.no_grad():
            st.example_loss(Recorder(self.lm), EX, args_for(1.0, dilute="x"), [2, 3], short_pool,
                            random.Random(0), torch.device("cpu"))
        self.assertEqual(seen, [8])

    def test_scored_positions_per_arm(self):
        """Which target positions each arm scores: A only (lam=0), A+O (ActObs), A + unrelated (D)."""
        seen = []

        class Recorder(st.PositionLogProbs):
            def forward(self, ids, positions, d_ids=None, d_positions=None):
                seen.append((positions.tolist(), None if d_positions is None else len(d_positions)))
                return super().forward(ids, positions, d_ids, d_positions)

        rec = Recorder(self.lm)
        act = [i for i, c in enumerate(EX["token_class"]) if c == 1]
        obs = [i for i, c in enumerate(EX["token_class"]) if c in (2, 3)]
        for args, pool, expect_pos, expect_d in [
            (args_for(0.0), None, act, None),
            (args_for(1.0), None, act + obs, None),
            (args_for(1.0, norm="separate"), None, act + obs, None),
            (args_for(1.0, dilute="x"), [list(range(3, 90))], act, len(obs)),
        ]:
            seen.clear()
            with torch.no_grad():
                st.example_loss(rec, EX, args, [2, 3], pool, random.Random(0), torch.device("cpu"))
            self.assertEqual(seen, [(expect_pos, expect_d)])


if __name__ == "__main__":
    unittest.main()
