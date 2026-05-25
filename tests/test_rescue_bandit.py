from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bora.rescue_bandit import (
    BRANCH_2,
    SEED_FEATURE_NAMES,
    THINK_RESEED,
    THINK_VERIFY,
    THINK_VERIFY_2,
    VERIFY_ADOPT,
    build_seed_state,
    constrain_rescue_actions_for_seed,
    execute_rescue_action,
    feasible_rescue_actions,
    is_clean_high_confidence,
    rescue_reward,
)
from bora.runtime import StepwiseEnvironment, select_best_branch
from bora.types import SolverOutput, VerifierResult


def config() -> dict:
    return {
        "mode": "eval",
        "task": "math",
        "random_seed": 3,
        "total_budget": 256,
        "seed_rollout_mode": "standard_cot",
        "seed_rollout_tokens": 96,
        "max_active_branches": 4,
        "verify_topk": 1,
        "actions": ["STOP", "THINK_64", "THINK_192", "VERIFY", "BRANCH"],
        "cost": {
            "lambda_tok": 0.1,
            "lambda_ver": 0.0,
            "lambda_lat": 0.0,
            "lat_norm_ms": 8000,
        },
        "solver": {
            "backend": "mock",
            "allow_gold_fallback": False,
            "max_new_tokens_short": 64,
            "max_new_tokens_long": 128,
            "max_new_tokens_rescue": 96,
        },
        "verifier": {
            "backend": "mock",
            "tokens_per_call": 32,
            "allow_gold_fallback": False,
        },
        "rescue_bandit": {
            "clean_confidence": 0.90,
            "reseed_tokens": 64,
            "lambda_flip": 1.0,
            "min_seed_reasoning_words": 5,
        },
    }


class RescueBanditTests(unittest.TestCase):
    def test_seed_state_features_are_stable_and_clean(self) -> None:
        env = StepwiseEnvironment(config())
        problem = {"qid": "p1", "question": "What is 2 + 2?", "answer": "4"}
        state = env.seed_rollout(env.reset(problem))
        features = build_seed_state(env, state)
        self.assertEqual(len(features.vector), len(SEED_FEATURE_NAMES))
        self.assertTrue(features.metadata["answer_parse_success"])
        self.assertTrue(is_clean_high_confidence(features, min_confidence=0.90))

    def test_verify_adopt_can_replace_wrong_seed_answer(self) -> None:
        env = StepwiseEnvironment(config())
        problem = {"qid": "p1", "question": "What is 2 + 2?", "answer": "4"}
        state = env.seed_rollout(env.reset(problem))
        branch = select_best_branch(state.branches, prefer_undone=False)
        branch.current_answer = "5"
        branch.answer_history.append("5")
        execution = execute_rescue_action(env, state, problem, VERIFY_ADOPT)
        self.assertEqual(execution.final_answer, "4")
        self.assertGreater(execution.extra_verifier_tokens, 0)

    def test_verify_adopt_can_use_cached_verifier_candidate(self) -> None:
        env = StepwiseEnvironment(config())
        problem = {"qid": "p1", "question": "What is 2 + 2?", "answer": "4"}
        state = env.seed_rollout(env.reset(problem))
        branch = select_best_branch(state.branches, prefer_undone=False)
        branch.current_answer = "5"
        branch.answer_history.append("5")
        cached = VerifierResult(
            score=0.1,
            token_cost=32,
            latency_ms=100,
            explanation="The correct answer is 4.",
            candidate_answer="4",
        )

        execution = execute_rescue_action(
            env,
            state,
            problem,
            VERIFY_ADOPT,
            cached_verifier_result=cached,
        )

        self.assertEqual(execution.final_answer, "4")
        self.assertEqual(execution.extra_verifier_tokens, 0)
        self.assertTrue(execution.metadata["used_cached_verifier"])

    def test_branch_rescue_can_be_prevented_from_overriding_seed(self) -> None:
        cfg = config()
        cfg["rescue_bandit"] = {
            **cfg["rescue_bandit"],
            "allow_branch_override": False,
        }
        env = StepwiseEnvironment(cfg)
        problem = {"qid": "p1", "question": "What is 2 + 2?", "answer": "4"}
        state = env.seed_rollout(env.reset(problem))
        branch = select_best_branch(state.branches, prefer_undone=False)
        branch.current_answer = "5"
        branch.answer_history.append("5")

        execution = execute_rescue_action(env, state, problem, BRANCH_2)

        self.assertEqual(execution.final_answer, "5")
        self.assertTrue(execution.metadata["branch_override_disabled"])
        self.assertEqual(execution.metadata["branch_seed_answer"], "5")

    def test_think_reseed_runs_as_thinking_rescue_action(self) -> None:
        cfg = config()
        cfg["rescue_bandit"] = {
            **cfg["rescue_bandit"],
            "think_reseed_tokens": 64,
        }
        env = StepwiseEnvironment(cfg)
        problem = {"qid": "p1", "question": "What is 2 + 2?", "answer": "4"}
        state = env.seed_rollout(env.reset(problem))
        actions = feasible_rescue_actions(env, state, actions=[THINK_RESEED])

        self.assertEqual(actions, [THINK_RESEED])

        execution = execute_rescue_action(env, state, problem, THINK_RESEED)

        self.assertEqual(execution.action, THINK_RESEED)
        self.assertEqual(execution.metadata["thinking_mode"], "think")
        self.assertIn(THINK_RESEED, state.action_history)

    def test_think_verify_adopts_only_with_verifier_agreement(self) -> None:
        cfg = config()
        cfg["rescue_bandit"] = {
            **cfg["rescue_bandit"],
            "think_verify_tokens": 64,
        }
        env = StepwiseEnvironment(cfg)
        problem = {"qid": "p1", "question": "What is 2 + 2?", "answer": "4"}
        state = env.seed_rollout(env.reset(problem))
        branch = select_best_branch(state.branches, prefer_undone=False)
        branch.current_answer = "5"
        branch.answer_history.append("5")

        class FakeThinkingSolver:
            def generate(self, *args, **kwargs) -> SolverOutput:
                self.enable_thinking = kwargs.get("enable_thinking")
                return SolverOutput(
                    step_text="Thinking audit says the corrected answer is 4.",
                    current_answer="4",
                    confidence=0.91,
                    done=True,
                    token_cost=48,
                    latency_ms=120,
                )

        fake_solver = FakeThinkingSolver()
        env.solver = fake_solver
        cached = VerifierResult(
            score=0.1,
            token_cost=32,
            latency_ms=100,
            explanation="The correct answer is 4.",
            candidate_answer="4",
        )

        execution = execute_rescue_action(
            env,
            state,
            problem,
            THINK_VERIFY,
            cached_verifier_result=cached,
        )

        self.assertEqual(execution.final_answer, "4")
        self.assertTrue(execution.metadata["think_verify_result"]["adopted"])
        self.assertEqual(
            execution.metadata["think_verify_result"]["gate_reason"],
            "thinking_agrees_with_verifier_candidate",
        )
        self.assertTrue(fake_solver.enable_thinking)

    def test_think_verify_reverts_to_seed_without_agreement(self) -> None:
        cfg = config()
        cfg["rescue_bandit"] = {
            **cfg["rescue_bandit"],
            "think_verify_tokens": 64,
        }
        env = StepwiseEnvironment(cfg)
        problem = {"qid": "p1", "question": "What is 2 + 2?", "answer": "4"}
        state = env.seed_rollout(env.reset(problem))
        branch = select_best_branch(state.branches, prefer_undone=False)
        branch.current_answer = "5"
        branch.answer_history.append("5")

        class FakeThinkingSolver:
            def generate(self, *args, **kwargs) -> SolverOutput:
                return SolverOutput(
                    step_text="Thinking audit proposes a different answer 4.",
                    current_answer="4",
                    confidence=0.91,
                    done=True,
                    token_cost=48,
                    latency_ms=120,
                )

        env.solver = FakeThinkingSolver()
        cached = VerifierResult(
            score=0.1,
            token_cost=32,
            latency_ms=100,
            explanation="The verifier proposes 6.",
            candidate_answer="6",
        )

        execution = execute_rescue_action(
            env,
            state,
            problem,
            THINK_VERIFY,
            cached_verifier_result=cached,
        )

        self.assertEqual(execution.final_answer, "5")
        self.assertFalse(execution.metadata["think_verify_result"]["adopted"])
        self.assertEqual(
            execution.metadata["think_verify_result"]["gate_reason"],
            "no_independent_agreement",
        )
        self.assertEqual(select_best_branch(state.branches, prefer_undone=False).current_answer, "5")

    def test_think_verify_2_adopts_only_when_two_thinking_samples_agree(self) -> None:
        cfg = config()
        cfg["total_budget"] = 512
        cfg["rescue_bandit"] = {
            **cfg["rescue_bandit"],
            "think_verify_2_tokens": 64,
        }
        env = StepwiseEnvironment(cfg)
        problem = {"qid": "p1", "question": "What is 2 + 2?", "answer": "4"}
        state = env.seed_rollout(env.reset(problem))
        branch = select_best_branch(state.branches, prefer_undone=False)
        branch.current_answer = "5"
        branch.answer_history.append("5")

        class FakeThinkingSolver:
            def __init__(self) -> None:
                self.calls = 0
                self.enable_thinking_flags: list[bool | None] = []

            def generate(self, *args, **kwargs) -> SolverOutput:
                self.calls += 1
                self.enable_thinking_flags.append(kwargs.get("enable_thinking"))
                return SolverOutput(
                    step_text=f"Thinking sample {self.calls} says answer 4.",
                    current_answer="4",
                    confidence=0.91,
                    done=True,
                    token_cost=48,
                    latency_ms=120,
                )

        fake_solver = FakeThinkingSolver()
        env.solver = fake_solver

        execution = execute_rescue_action(env, state, problem, THINK_VERIFY_2)

        self.assertEqual(execution.final_answer, "4")
        self.assertEqual(fake_solver.calls, 2)
        self.assertEqual(fake_solver.enable_thinking_flags, [True, True])
        self.assertTrue(execution.metadata["think_verify_2_result"]["adopted"])
        self.assertEqual(
            execution.metadata["think_verify_2_result"]["gate_reason"],
            "thinking_pair_agreement",
        )
        self.assertIn("THINK_VERIFY_2_1", state.action_history)
        self.assertIn("THINK_VERIFY_2_2", state.action_history)
        self.assertIn(THINK_VERIFY_2, state.action_history)

    def test_think_verify_2_reverts_to_seed_when_samples_disagree(self) -> None:
        cfg = config()
        cfg["total_budget"] = 512
        cfg["rescue_bandit"] = {
            **cfg["rescue_bandit"],
            "think_verify_2_tokens": 64,
        }
        env = StepwiseEnvironment(cfg)
        problem = {"qid": "p1", "question": "What is 2 + 2?", "answer": "4"}
        state = env.seed_rollout(env.reset(problem))
        branch = select_best_branch(state.branches, prefer_undone=False)
        branch.current_answer = "5"
        branch.answer_history.append("5")

        class FakeThinkingSolver:
            def __init__(self) -> None:
                self.answers = ["4", "6"]

            def generate(self, *args, **kwargs) -> SolverOutput:
                answer = self.answers.pop(0)
                return SolverOutput(
                    step_text=f"Thinking sample says answer {answer}.",
                    current_answer=answer,
                    confidence=0.91,
                    done=True,
                    token_cost=48,
                    latency_ms=120,
                )

        env.solver = FakeThinkingSolver()

        execution = execute_rescue_action(env, state, problem, THINK_VERIFY_2)

        self.assertEqual(execution.final_answer, "5")
        self.assertFalse(execution.metadata["think_verify_2_result"]["adopted"])
        self.assertEqual(
            execution.metadata["think_verify_2_result"]["gate_reason"],
            "thinking_samples_disagree",
        )
        self.assertEqual(select_best_branch(state.branches, prefer_undone=False).current_answer, "5")

    def test_think_verify_2_cost_guard_blocks_override(self) -> None:
        cfg = config()
        cfg["total_budget"] = 512
        cfg["rescue_bandit"] = {
            **cfg["rescue_bandit"],
            "think_verify_2_tokens": 64,
            "think_verify_2_max_single_tokens": 40,
        }
        env = StepwiseEnvironment(cfg)
        problem = {"qid": "p1", "question": "What is 2 + 2?", "answer": "4"}
        state = env.seed_rollout(env.reset(problem))
        branch = select_best_branch(state.branches, prefer_undone=False)
        branch.current_answer = "5"
        branch.answer_history.append("5")

        class FakeThinkingSolver:
            def generate(self, *args, **kwargs) -> SolverOutput:
                return SolverOutput(
                    step_text="Thinking sample says answer 4.",
                    current_answer="4",
                    confidence=0.91,
                    done=True,
                    token_cost=48,
                    latency_ms=120,
                )

        env.solver = FakeThinkingSolver()

        execution = execute_rescue_action(env, state, problem, THINK_VERIFY_2)

        self.assertEqual(execution.final_answer, "5")
        self.assertFalse(execution.metadata["think_verify_2_result"]["adopted"])
        self.assertEqual(
            execution.metadata["think_verify_2_result"]["gate_reason"],
            "cost_guard_single_sample_too_expensive",
        )

    def test_think_verify_2_requires_verifier_confirmation_when_enabled(self) -> None:
        cfg = config()
        cfg["total_budget"] = 512
        cfg["rescue_bandit"] = {
            **cfg["rescue_bandit"],
            "think_verify_2_tokens": 64,
            "think_verify_2_require_confirmation": True,
            "think_verify_2_confirm_score": 0.85,
        }
        env = StepwiseEnvironment(cfg)
        problem = {"qid": "p1", "question": "What is 2 + 2?", "answer": "4"}
        state = env.seed_rollout(env.reset(problem))
        branch = select_best_branch(state.branches, prefer_undone=False)
        branch.current_answer = "5"
        branch.answer_history.append("5")

        class FakeThinkingSolver:
            def generate(self, *args, **kwargs) -> SolverOutput:
                return SolverOutput(
                    step_text="Thinking sample says answer 4.",
                    current_answer="4",
                    confidence=0.91,
                    done=True,
                    token_cost=32,
                    latency_ms=120,
                )

        class FakeVerifier:
            def score(self, problem, branch, rng) -> VerifierResult:
                return VerifierResult(
                    score=0.93,
                    token_cost=12,
                    latency_ms=30,
                    explanation="The candidate answer is confirmed.",
                    candidate_answer="4",
                    score_parse_success=True,
                )

        env.solver = FakeThinkingSolver()
        env.verifier = FakeVerifier()

        execution = execute_rescue_action(env, state, problem, THINK_VERIFY_2)

        self.assertEqual(execution.final_answer, "4")
        result = execution.metadata["think_verify_2_result"]
        self.assertTrue(result["adopted"])
        self.assertTrue(result["confirmation_passed"])
        self.assertEqual(result["confirmation_reason"], "confirmation_verifier_candidate_agrees")
        self.assertGreater(execution.extra_verifier_tokens, 0)

    def test_think_verify_2_rejects_when_verifier_returns_seed(self) -> None:
        cfg = config()
        cfg["total_budget"] = 512
        cfg["rescue_bandit"] = {
            **cfg["rescue_bandit"],
            "think_verify_2_tokens": 64,
            "think_verify_2_require_confirmation": True,
        }
        env = StepwiseEnvironment(cfg)
        problem = {"qid": "p1", "question": "What is 2 + 2?", "answer": "4"}
        state = env.seed_rollout(env.reset(problem))
        branch = select_best_branch(state.branches, prefer_undone=False)
        branch.current_answer = "5"
        branch.answer_history.append("5")

        class FakeThinkingSolver:
            def generate(self, *args, **kwargs) -> SolverOutput:
                return SolverOutput(
                    step_text="Thinking sample says answer 4.",
                    current_answer="4",
                    confidence=0.91,
                    done=True,
                    token_cost=32,
                    latency_ms=120,
                )

        class FakeVerifier:
            def score(self, problem, branch, rng) -> VerifierResult:
                return VerifierResult(
                    score=0.2,
                    token_cost=12,
                    latency_ms=30,
                    explanation="The seed answer is still preferred.",
                    candidate_answer="5",
                    score_parse_success=True,
                )

        env.solver = FakeThinkingSolver()
        env.verifier = FakeVerifier()

        execution = execute_rescue_action(env, state, problem, THINK_VERIFY_2)

        self.assertEqual(execution.final_answer, "5")
        result = execution.metadata["think_verify_2_result"]
        self.assertFalse(result["adopted"])
        self.assertEqual(result["gate_reason"], "confirmation_verifier_returns_seed")

    def test_verifier_disagreement_constrains_to_verify_actions(self) -> None:
        env = StepwiseEnvironment(config())
        problem = {"qid": "p1", "question": "What is 2 + 2?", "answer": "4"}
        state = env.seed_rollout(env.reset(problem))
        branch = select_best_branch(state.branches, prefer_undone=False)
        branch.current_answer = "5"
        branch.answer_history.append("5")
        features = build_seed_state(
            env,
            state,
            verifier_result=VerifierResult(
                score=0.1,
                token_cost=32,
                latency_ms=100,
                explanation="The correct answer is 4.",
                candidate_answer="4",
            ),
        )

        constrained, reason = constrain_rescue_actions_for_seed(
            features,
            ["RESEED_2_VOTE", "VERIFY_ADOPT", "VERIFY_ONLY"],
            {
                "prefer_verify_adopt_on_disagreement": True,
                "verify_adopt_disagreement_require_candidate": True,
                "verify_adopt_disagreement_require_malformed_or_low_confidence": False,
            },
            clean_confidence=0.90,
        )

        self.assertEqual(constrained, ["VERIFY_ADOPT", "VERIFY_ONLY"])
        self.assertEqual(reason, "verifier_disagreement_verify_adopt")

    def test_high_risk_clean_seed_constrains_to_configured_actions(self) -> None:
        cfg = config()
        cfg["rescue_bandit"] = {
            **cfg["rescue_bandit"],
            "trigger_high_risk_clean_seed": True,
            "high_risk_clean_terms": ["years old"],
            "high_risk_clean_actions": ["VERIFY_ADOPT"],
        }
        env = StepwiseEnvironment(cfg)
        problem = {
            "qid": "p1",
            "question": "Lena is 7 years old. How old is she in 5 years?",
            "answer": "12",
        }
        state = env.seed_rollout(env.reset(problem))
        features = build_seed_state(env, state)

        self.assertIn("high_risk_clean_seed", features.metadata["triggers"])
        constrained, reason = constrain_rescue_actions_for_seed(
            features,
            ["VERIFY_ADOPT", "RESEED_1", "RESEED_2_VOTE"],
            cfg["rescue_bandit"],
            clean_confidence=0.90,
        )
        self.assertEqual(constrained, ["VERIFY_ADOPT"])
        self.assertEqual(reason, "high_risk_clean")

    def test_decimal_high_risk_clean_seed_uses_decimal_action_set(self) -> None:
        cfg = config()
        cfg["rescue_bandit"] = {
            **cfg["rescue_bandit"],
            "trigger_high_risk_clean_seed": True,
            "high_risk_clean_actions": ["VERIFY_ADOPT"],
            "high_risk_clean_decimal_actions": ["RESEED_1"],
        }
        env = StepwiseEnvironment(cfg)
        problem = {
            "qid": "p1",
            "question": "How many jumps per second does Bobby do?",
            "answer": "0.5",
        }
        state = env.seed_rollout(env.reset(problem))
        branch = select_best_branch(state.branches, prefer_undone=False)
        branch.current_answer = "0.5"
        branch.answer_history.append("0.5")
        features = build_seed_state(env, state)

        constrained, reason = constrain_rescue_actions_for_seed(
            features,
            ["VERIFY_ADOPT", "RESEED_1"],
            cfg["rescue_bandit"],
            clean_confidence=0.90,
        )

        self.assertIn("decimal_answer", features.metadata["high_risk_clean_reasons"])
        self.assertEqual(constrained, ["RESEED_1"])
        self.assertEqual(reason, "high_risk_clean")

    def test_feasible_rescue_actions_respect_budget(self) -> None:
        cfg = config()
        cfg["total_budget"] = 128
        env = StepwiseEnvironment(cfg)
        problem = {"qid": "p1", "question": "What is 2 + 2?", "answer": "4"}
        state = env.seed_rollout(env.reset(problem))
        actions = feasible_rescue_actions(env, state)
        self.assertIn("VERIFY_ONLY", actions)
        self.assertNotIn("RESEED_2_VOTE", actions)

    def test_reward_penalizes_negative_flip(self) -> None:
        reward = rescue_reward(
            seed_answer="4",
            final_answer="5",
            gold_answer="4",
            extra_total_tokens=0,
            extra_latency_ms=0,
            verifier_calls=0,
            total_budget=256,
            cost_cfg=config()["cost"],
            negative_flip_penalty=1.0,
        )
        self.assertEqual(reward, -2.0)


if __name__ == "__main__":
    unittest.main()
