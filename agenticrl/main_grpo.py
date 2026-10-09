"""GRPO entrypoint: SkyRL's BasePPOExp with "text2sql" bound to LenientSQLEnv (no algorithm changes).

The environment is registered inside the Ray task, so every worker sees the rebinding.
"""
import sys

import ray
from skyrl.train.config import SkyRLTrainConfig
from skyrl.train.entrypoints.main_base import BasePPOExp, validate_cfg
from skyrl.train.utils import initialize_ray


@ray.remote(num_cpus=1)
def entrypoint(cfg):
    from agenticrl.sql_lenient import register_as_text2sql
    register_as_text2sql()
    print("text2sql -> agenticrl.sql_lenient:LenientSQLEnv (no <think> format rules)", flush=True)
    BasePPOExp(cfg).run()


if __name__ == "__main__":
    cfg = SkyRLTrainConfig.from_cli_overrides(sys.argv[1:])
    validate_cfg(cfg)
    initialize_ray(cfg)
    ray.get(entrypoint.remote(cfg))
