from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bora.runtime import StepwiseEnvironment
from bora.types import Branch


def make_config() -> dict:
    return {
        "mode": "train",
        "random_seed": 13,
        "episode_seed": 13,
        "total_budget": 256,
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
    "qid": "branch_case",
    "question": "What is 10 + 5?",
    "answer": "15",
    "difficulty": "easy",
}


class BranchTargetingTests(unittest.TestCase):
    def test_decision_branch_matches_executed_branch_for_think(self) -> None:
        env = StepwiseEnvironment(make_config())
        state = env.reset(PROBLEM)
        state = env.seed_rollout(state)
        state.branches.append(
            Branch(
                branch_id=1,
                parent_id=None,
                current_answer="15",
                confidence=0.95,
                done=True,
                answer_history=["15"],
                conf_history=[0.95],
            )
        )
        env.next_branch_id = 2
        state, delta = env.step(state, "THINK_64", target_branch_id=0)
        self.assertEqual(delta.decision_branch_id, 0)
        self.assertEqual(delta.executed_branch_id, 0)
        self.assertGreaterEqual(state.branches[0].depth, 2)

    def test_branch_action_forks_requested_source_branch(self) -> None:
        env = StepwiseEnvironment(make_config())
        state = env.reset(PROBLEM)
        state = env.seed_rollout(state)
        state.branches.append(
            Branch(
                branch_id=1,
                parent_id=None,
                current_answer="15",
                confidence=0.95,
                done=True,
                answer_history=["15"],
                conf_history=[0.95],
            )
        )
        env.next_branch_id = 2
        state, delta = env.step(state, "BRANCH", target_branch_id=0)
        self.assertEqual(delta.decision_branch_id, 0)
        self.assertEqual(delta.source_branch_id, 0)
        self.assertEqual(delta.executed_branch_id, delta.new_branch_id)
        self.assertIsNotNone(delta.new_branch_id)
        new_branch = next(branch for branch in state.branches if branch.branch_id == delta.new_branch_id)
        self.assertEqual(new_branch.parent_id, 0)


if __name__ == "__main__":
    unittest.main()
