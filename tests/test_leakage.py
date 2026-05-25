from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bora.runtime import StepwiseEnvironment


def make_config(*, solver_gold: bool = False, verifier_gold: bool = False) -> dict:
    return {
        "mode": "eval",
        "random_seed": 21,
        "episode_seed": 21,
        "total_budget": 128,
        "seed_rollout_tokens": 64,
        "max_active_branches": 3,
        "verify_topk": 2,
        "actions": ["STOP", "THINK_64", "THINK_192", "VERIFY", "BRANCH"],
        "cost": {
            "lambda_tok": 0.15,
            "lambda_ver": 0.05,
            "lambda_lat": 0.02,
            "lat_norm_ms": 180.0,
        },
        "solver": {
            "backend": "mock",
            "max_new_tokens_short": 64,
            "max_new_tokens_long": 192,
            "allow_gold_fallback": solver_gold,
        },
        "verifier": {
            "backend": "mock",
            "tokens_per_call": 64,
            "allow_gold_fallback": verifier_gold,
        },
    }


class LeakageTests(unittest.TestCase):
    def test_eval_mode_rejects_solver_gold_fallback(self) -> None:
        with self.assertRaises(ValueError):
            StepwiseEnvironment(make_config(solver_gold=True))

    def test_eval_mode_rejects_verifier_gold_fallback(self) -> None:
        with self.assertRaises(ValueError):
            StepwiseEnvironment(make_config(verifier_gold=True))


if __name__ == "__main__":
    unittest.main()
