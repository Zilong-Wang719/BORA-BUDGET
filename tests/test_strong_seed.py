from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bora.bandit import LinTS, make_default_prior
from bora.proxy import ConstantProxy
from bora.runtime import StepwiseEnvironment, run_bora_episode


def make_config() -> dict:
    return {
        "mode": "train",
        "random_seed": 23,
        "episode_seed": 23,
        "total_budget": 256,
        "seed_rollout_mode": "standard_cot",
        "seed_rollout_tokens": 128,
        "max_active_branches": 1,
        "verify_topk": 1,
        "actions": ["STOP", "THINK_64", "THINK_192", "VERIFY"],
        "cost": {
            "lambda_tok": 0.15,
            "lambda_ver": 0.05,
            "lambda_lat": 0.02,
            "lat_norm_ms": 180.0,
        },
        "bandit": {
            "lam": 1.0,
            "ts_scale": 0.20,
            "min_decisions_before_stop": 0,
            "early_stop_if_done": True,
            "early_stop_confidence": 0.90,
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
    "qid": "strong_seed_case",
    "question": "What is 8 + 4?",
    "answer": "12",
    "difficulty": "easy",
}


class StrongSeedTests(unittest.TestCase):
    def test_standard_cot_seed_sets_final_answer(self) -> None:
        env = StepwiseEnvironment(make_config())
        state = env.seed_rollout(env.reset(PROBLEM))
        branch = state.branches[0]
        self.assertEqual(branch.current_answer, "12")
        self.assertTrue(branch.done)
        self.assertGreaterEqual(branch.confidence, 0.90)
        self.assertLessEqual(state.total_tokens, state.total_budget)

    def test_early_stop_does_not_spend_after_strong_seed(self) -> None:
        env = StepwiseEnvironment(make_config())
        state = env.seed_rollout(env.reset(PROBLEM))
        feature_dim = len(env.extract_features(state))
        bandit = LinTS(
            arms=list(env.config["actions"]),
            d=feature_dim,
            lam=float(env.config["bandit"]["lam"]),
            ts_scale=float(env.config["bandit"]["ts_scale"]),
        )
        bandit.load_prior(make_default_prior(list(env.config["actions"]), feature_dim, 1.0))
        record, _ = run_bora_episode(env, bandit, ConstantProxy(0.5), state)
        self.assertEqual(record.prediction, "12")
        self.assertEqual(record.actions, ["SEED", "STOP"])
        self.assertEqual(record.solver_tokens, state.spent_solver_tokens)

    def test_selective_rescue_skips_when_verifier_is_strong(self) -> None:
        config = make_config()
        config["rescue"] = {
            "enabled": True,
            "verify_seed": True,
            "verifier_accept_threshold": 0.72,
            "low_confidence_threshold": 0.90,
            "action": "THINK_192",
            "max_steps": 1,
        }
        env = StepwiseEnvironment(config)
        state = env.seed_rollout(env.reset(PROBLEM))
        feature_dim = len(env.extract_features(state))
        bandit = LinTS(
            arms=list(env.config["actions"]),
            d=feature_dim,
            lam=float(env.config["bandit"]["lam"]),
            ts_scale=float(env.config["bandit"]["ts_scale"]),
        )
        bandit.load_prior(make_default_prior(list(env.config["actions"]), feature_dim, 1.0))
        record, _ = run_bora_episode(env, bandit, ConstantProxy(0.5), state)
        self.assertEqual(record.prediction, "12")
        self.assertEqual(record.actions, ["SEED", "VERIFY", "STOP"])
        self.assertFalse(record.metadata["rescue"]["triggered"])

    def test_selective_rescue_runs_when_verifier_is_weak(self) -> None:
        config = make_config()
        config["rescue"] = {
            "enabled": True,
            "verify_seed": True,
            "verifier_accept_threshold": 0.72,
            "low_confidence_threshold": 0.90,
            "action": "THINK_192",
            "max_steps": 1,
        }
        env = StepwiseEnvironment(config)
        state = env.seed_rollout(env.reset(PROBLEM))
        branch = state.branches[0]
        branch.current_answer = "999"
        branch.answer_history[-1] = "999"
        feature_dim = len(env.extract_features(state))
        bandit = LinTS(
            arms=list(env.config["actions"]),
            d=feature_dim,
            lam=float(env.config["bandit"]["lam"]),
            ts_scale=float(env.config["bandit"]["ts_scale"]),
        )
        bandit.load_prior(make_default_prior(list(env.config["actions"]), feature_dim, 1.0))
        record, _ = run_bora_episode(env, bandit, ConstantProxy(0.5), state)
        self.assertIn("VERIFY", record.actions)
        self.assertIn("THINK_192", record.actions)
        self.assertTrue(record.metadata["rescue"]["triggered"])
        self.assertIn("weak_verifier", record.metadata["rescue"]["reasons"])


if __name__ == "__main__":
    unittest.main()
