from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts.collect_rollouts import explorer_policy


class DummyEnv:
    def __init__(self, feasible: list[str]) -> None:
        self._feasible = feasible
        self.rng = np.random.default_rng(7)

    def feasible_actions(self, state) -> list[str]:
        return list(self._feasible)


class DummyState:
    def __init__(self, step_idx: int) -> None:
        self.step_idx = step_idx


class CollectRolloutPolicyTests(unittest.TestCase):
    def test_stop_only_remains_valid_choice(self) -> None:
        action = explorer_policy(DummyState(step_idx=0), DummyEnv(["STOP"]))
        self.assertEqual(action, "STOP")

    def test_early_stop_is_removed_only_when_other_choices_exist(self) -> None:
        action = explorer_policy(DummyState(step_idx=0), DummyEnv(["STOP", "THINK_64"]))
        self.assertEqual(action, "THINK_64")

    def test_policy_handles_action_sets_without_branch(self) -> None:
        action = explorer_policy(DummyState(step_idx=1), DummyEnv(["THINK_64", "THINK_192", "VERIFY"]))
        self.assertIn(action, {"THINK_64", "THINK_192", "VERIFY"})


if __name__ == "__main__":
    unittest.main()
