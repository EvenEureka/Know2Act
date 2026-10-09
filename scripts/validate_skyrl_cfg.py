"""CPU dry run: parse the GRPO overrides with SkyRL's own config code and run its validator."""
import sys

from skyrl.train.config import SkyRLTrainConfig
from skyrl.train.utils.utils import validate_cfg

cfg = SkyRLTrainConfig.from_cli_overrides(sys.argv[1:])
try:
    validate_cfg(cfg)
    print("validate_cfg: OK")
except Exception as e:  # some checks need GPUs / model files; report and continue
    print("validate_cfg raised:", type(e).__name__, e)
g, t = cfg.generator, cfg.trainer
print("model", t.policy.model.path)
print("batch", t.train_batch_size, "mini", t.policy_mini_batch_size, "n", g.n_samples_per_prompt,
      "steps", t.max_training_steps, "lr", t.policy.optimizer_config.lr, "kl", t.algorithm.use_kl_loss)
print("loss_reduction", t.algorithm.loss_reduction, "policy_loss", t.algorithm.policy_loss_type,
      "clip", t.algorithm.eps_clip_low, t.algorithm.eps_clip_high)
print("multi_turn", g.use_conversation_multi_turn, "max_turns", g.max_turns, "stop", list(g.sampling_params.stop),
      "T", g.sampling_params.temperature, "top_p", g.sampling_params.top_p, "top_k", g.sampling_params.top_k)
print("eval T", g.eval_sampling_params.temperature, "eval top_p", g.eval_sampling_params.top_p)
print("chat_template_kwargs", {k: (v, type(v).__name__) for k, v in dict(g.chat_template_kwargs).items()} if g.chat_template_kwargs else None)
