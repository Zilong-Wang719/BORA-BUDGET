from __future__ import annotations

import sys
import unittest
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bora.runtime import StepwiseEnvironment


def make_config(total_budget: int) -> dict:
    return {
        "mode": "train",
        "random_seed": 7,
        "episode_seed": 7,
        "total_budget": total_budget,
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
            "allow_gold_fallback": False,
        },
        "verifier": {
            "backend": "mock",
            "tokens_per_call": 64,
            "allow_gold_fallback": False,
        },
    }


PROBLEM = {
    "qid": "budget_case",
    "question": "What is 8 + 4?",
    "answer": "12",
    "difficulty": "easy",
}


class BudgetTests(unittest.TestCase):
    def test_think64_does_not_exceed_budget(self) -> None:
        env = StepwiseEnvironment(make_config(total_budget=50))
        state = env.reset(PROBLEM)
        state, delta = env.step(state, "THINK_64", target_branch_id=0)
        self.assertLessEqual(state.total_tokens, state.total_budget)
        self.assertLessEqual(delta.solver_tokens, 50)
        self.assertGreaterEqual(delta.remaining_budget_after, 0)

    def test_think192_truncates_to_remaining_budget(self) -> None:
        env = StepwiseEnvironment(make_config(total_budget=70))
        state = env.reset(PROBLEM)
        state, delta = env.step(state, "THINK_192", target_branch_id=0)
        self.assertLessEqual(state.total_tokens, state.total_budget)
        self.assertLessEqual(delta.solver_tokens, 70)
        self.assertGreaterEqual(delta.remaining_budget_after, 0)

    def test_verify_uses_only_affordable_calls(self) -> None:
        env = StepwiseEnvironment(make_config(total_budget=130))
        state = env.reset(PROBLEM)
        state = env.seed_rollout(state)
        extra_branch = deepcopy(state.branches[0])
        extra_branch.branch_id = 1
        state.branches.append(extra_branch)
        state, delta = env.step(state, "VERIFY", target_branch_id=0)
        self.assertEqual(delta.verifier_calls, 1)
        self.assertEqual(delta.verified_branch_ids, [0])
        self.assertLessEqual(state.total_tokens, state.total_budget)


if __name__ == "__main__":
    unittest.main()
