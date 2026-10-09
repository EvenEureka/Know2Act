"""SkyRL text2sql with the two <think> format rules removed; everything else is SkyRL's SQLEnv.

SkyRL's verify_format_and_extract (skyrl_gym/envs/sql/utils.py @99824f0) requires (1) exactly one
<solution>...</solution>, (2) no think/sql/observation tags inside it, (3) at least one
<think>...</think> block and (4) every </observation> followed by <think>. Qwen3 in non-thinking mode
never writes <think> tags (pilot P1b: 0/100), so rules (3)-(4) fail every trajectory although the
agent uses the tool normally. This env keeps (1)-(2), the execution-match outcome reward, the
observation text, turn limits and DB handling unchanged.

register_as_text2sql() rebinds the "text2sql" id to this class, so configs (environment.skyrl_gym.
text2sql.db_path) and data (env_class="text2sql") stay untouched.
"""
import re

from skyrl_gym.envs.sql import utils as U
from skyrl_gym.envs.sql.env import SQLEnv


def verify_format_lenient(output: str):
    if output.count(U.SOLUTION_START) != 1:
        return False, None
    _, tail = output.split(U.SOLUTION_START, 1)
    if tail.count(U.SOLUTION_END) != 1:
        return False, None
    solution_text, _ = tail.split(U.SOLUTION_END, 1)
    if re.search(r"</?(think|sql|observation)\b", solution_text, re.I):
        return False, None
    return True, solution_text.strip()


def compute_score_lenient(completion: str, reference: str, db_file: str, timeout: int = 30) -> float:
    try:
        ok, pred_sql = verify_format_lenient(completion)
        if not ok:
            return -1.0
        pred = U.execute_sql_wrapper_single(db_file, pred_sql, timeout, completion)
        ref = U.execute_sql_wrapper_single(db_file, reference, timeout, completion)
        p, g = pred[2], ref[2]
        return 1.0 if (p is not None and g is not None and p == g) else 0.0
    except Exception as e:  # same fallback as SkyRL's compute_score_single
        print(f"Unexpected error: {e}; Setting reward as 0")
        return 0


class LenientSQLEnv(SQLEnv):
    def _get_reward(self, action: str, done: bool) -> float:
        if done:
            chat_history_str = "".join([item["content"] for item in self.chat_history])
            return compute_score_lenient(chat_history_str, self.gold_sql, self.db_file)
        return 0


def register_as_text2sql():
    from skyrl_gym.envs.registration import deregister, register
    deregister("text2sql")
    register(id="text2sql", entry_point="agenticrl.sql_lenient:LenientSQLEnv")
