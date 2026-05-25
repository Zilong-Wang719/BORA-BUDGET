from __future__ import annotations

from time import perf_counter
from typing import Any, Callable

from bora.answer_extraction import extract_explicit_answer, extract_numeric_answer
from bora.common import is_correct, normalize_answer
from bora.llm import get_llm_backend
from bora.runtime import (
    StepwiseEnvironment,
    aggregate_final_answer,
    detect_negative_flip,
    select_best_branch,
)
from bora.types import Branch, Checkpoint, EpisodeRecord, State


PolicyFn = Callable[[State, StepwiseEnvironment], str]


STANDARD_COT_PROMPT = (
    "Please reason step by step, and put your final numeric answer within "
    "\\boxed{{}}.\n\n"
    "Problem:\n{question}"
)


def _record(env: StepwiseEnvironment, state: State, stop_reason: str) -> EpisodeRecord:
    prediction, metadata = aggregate_final_answer(state.branches)
    return EpisodeRecord(
        qid=state.qid,
        prediction=prediction,
        gold_answer=state.gold_answer,
        correct=is_correct(prediction, state.gold_answer),
        total_tokens=state.total_tokens,
        solver_tokens=state.spent_solver_tokens,
        verifier_tokens=state.spent_verifier_tokens,
        latency_ms=state.spent_latency_ms,
        branches_used=len(state.branches),
        stop_reason=stop_reason,
        actions=list(state.action_history),
        metadata={
            **metadata,
            "negative_flip": detect_negative_flip(state),
            "difficulty": state.metadata.get("difficulty", "unknown"),
        },
    )


def run_policy_episode(
    env: StepwiseEnvironment,
    problem: dict[str, Any],
    policy: PolicyFn,
    min_steps_before_stop: int = 1,
) -> EpisodeRecord:
    state = env.reset(problem)
    state = env.seed_rollout(state)
    while True:
        feasible = env.feasible_actions(state)
        action = policy(state, env)
        if action not in feasible:
            action = "STOP" if "STOP" in feasible else feasible[0]
        if action == "STOP" and state.step_idx < min_steps_before_stop and "THINK_64" in feasible:
            action = "THINK_64"
        if action == "STOP":
            state.action_history.append("STOP")
            break
        chosen_branch = select_best_branch(state.branches, prefer_undone=False)
        state, _ = env.step(state, action, target_branch_id=chosen_branch.branch_id)
        if env.budget_exhausted(state):
            break
    return _record(env, state, stop_reason="STOP_OR_BUDGET")


def full_cot_policy(state: State, env: StepwiseEnvironment) -> str:
    best = select_best_branch(state.branches, prefer_undone=False)
    return "STOP" if best.done and best.confidence >= 0.70 else "THINK_64"


def stop_confidence_heuristic_policy(state: State, env: StepwiseEnvironment) -> str:
    best = select_best_branch(state.branches, prefer_undone=False)
    if best.done and best.confidence >= 0.72:
        return "STOP"
    if len(best.conf_history) >= 2 and best.conf_history[-1] < best.conf_history[-2]:
        return "VERIFY" if "VERIFY" in env.feasible_actions(state) else "THINK_64"
    return "THINK_64"


def verify_then_stop_heuristic_policy(state: State, env: StepwiseEnvironment) -> str:
    best = select_best_branch(state.branches, prefer_undone=False)
    remaining_ratio = max(state.total_budget - state.total_tokens, 0) / max(state.total_budget, 1)
    threshold = 0.82 - 0.25 * (1.0 - remaining_ratio)
    if best.confidence >= threshold and best.current_answer is not None:
        return "STOP"
    if not best.prm_scores and "VERIFY" in env.feasible_actions(state):
        return "VERIFY"
    return "THINK_64"


def adaptive_verify_branch_heuristic_policy(state: State, env: StepwiseEnvironment) -> str:
    best = select_best_branch(state.branches, prefer_undone=False)
    feasible = env.feasible_actions(state)
    if best.done and best.prm_mean >= 0.75:
        return "STOP"
    if not best.prm_scores and "VERIFY" in feasible:
        return "VERIFY"
    if best.prm_scores and best.prm_mean < 0.45 and "BRANCH" in feasible:
        return "BRANCH"
    if state.total_budget - state.total_tokens > 192 and "THINK_192" in feasible and best.confidence < 0.60:
        return "THINK_192"
    return "THINK_64"


def _run_standard_direct_cot_candidate(
    config: dict[str, Any],
    problem: dict[str, Any],
    *,
    action: str = "STANDARD_DIRECT_COT",
) -> dict[str, Any]:
    backend = get_llm_backend(config, "solver")
    solver_cfg = {**config.get("llm", {}), **config.get("solver", {}), **config.get("baseline", {})}
    enable_thinking = solver_cfg.get("standard_cot_enable_thinking", solver_cfg.get("enable_thinking"))
    prompt_suffix = "/think" if enable_thinking is True else "/no_think"
    user_prompt = f"{STANDARD_COT_PROMPT.format(question=problem['question']).rstrip()}\n\n{prompt_suffix}"
    prompt = backend.render_prompt(
        system_prompt="You are a careful math reasoner.",
        user_prompt=user_prompt,
        enable_thinking=enable_thinking,
    )
    max_new_tokens = int(solver_cfg.get("standard_cot_max_new_tokens", 1024))
    temperature = float(solver_cfg.get("standard_cot_temperature", 0.7))
    top_p = float(solver_cfg.get("standard_cot_top_p", 0.8))
    repetition_penalty = float(solver_cfg.get("repetition_penalty", 1.0))
    started = perf_counter()
    generation = backend.generate_text(
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
    )
    latency_ms = generation.latency_ms or int((perf_counter() - started) * 1000)
    prediction = extract_explicit_answer(generation.text, prefer_numeric=True)
    if prediction is None:
        prediction = extract_numeric_answer(generation.text)
    return {
        "prediction": prediction,
        "prompt_tokens": generation.prompt_tokens,
        "completion_tokens": generation.completion_tokens,
        "latency_ms": latency_ms,
        "text": generation.text,
        "enable_thinking": enable_thinking,
        "prompt_suffix": prompt_suffix,
        "action": action,
    }


def _candidate_branch(candidate: dict[str, Any], *, branch_id: int = 0) -> Branch:
    answer = candidate.get("prediction")
    trace = str(candidate.get("text") or candidate.get("completion_head") or "")
    token_cost = int(candidate.get("completion_tokens") or 0)
    checkpoint = Checkpoint(
        step_index=0,
        step_text=trace,
        current_answer=answer,
        confidence=0.92 if normalize_answer(answer) is not None else 0.25,
        done=normalize_answer(answer) is not None,
        action=str(candidate.get("action", "STANDARD_DIRECT_COT")),
        token_cost=token_cost,
    )
    return Branch(
        branch_id=branch_id,
        parent_id=None,
        trace=trace,
        current_answer=answer,
        confidence=checkpoint.confidence,
        done=checkpoint.done,
        solver_tokens=token_cost,
        verifier_tokens=0,
        answer_history=[answer] if answer is not None else [],
        conf_history=[checkpoint.confidence],
        checkpoints=[checkpoint],
        action_history=[checkpoint.action],
    )


def run_standard_direct_cot(config: dict[str, Any], problem: dict[str, Any]) -> EpisodeRecord:
    candidate = _run_standard_direct_cot_candidate(config, problem)
    prediction = candidate["prediction"]
    return EpisodeRecord(
        qid=problem["qid"],
        prediction=prediction,
        gold_answer=problem.get("answer"),
        correct=is_correct(prediction, problem.get("answer")),
        total_tokens=candidate["completion_tokens"],
        solver_tokens=candidate["completion_tokens"],
        verifier_tokens=0,
        latency_ms=candidate["latency_ms"],
        branches_used=1,
        stop_reason="DIRECT_COT",
        actions=["STANDARD_DIRECT_COT"],
        metadata={
            "prompt_tokens": candidate["prompt_tokens"],
            "completion_tokens": candidate["completion_tokens"],
            "completion_head": candidate["text"][:500],
            "enable_thinking": candidate["enable_thinking"],
            "prompt_suffix": candidate["prompt_suffix"],
            "contains_think_tag": "<think>" in candidate["text"],
            "contains_end_think_tag": "</think>" in candidate["text"],
        },
    )


def run_self_consistency(
    config: dict[str, Any],
    problem: dict[str, Any],
    repeats: int = 3,
) -> EpisodeRecord:
    votes: dict[str, int] = {}
    candidates: list[dict[str, Any]] = []
    total_tokens = 0
    total_latency = 0
    for idx in range(repeats):
        candidate = _run_standard_direct_cot_candidate(
            {**config, "episode_seed": int(config.get("episode_seed", 0)) + idx},
            problem,
            action=f"STANDARD_DIRECT_COT_SC_{idx + 1}",
        )
        candidates.append(candidate)
        normalized = normalize_answer(candidate["prediction"])
        if normalized is not None:
            votes[normalized] = votes.get(normalized, 0) + 1
        total_tokens += int(candidate["completion_tokens"])
        total_latency += int(candidate["latency_ms"])
    prediction = max(votes.items(), key=lambda item: item[1])[0] if votes else None
    return EpisodeRecord(
        qid=problem["qid"],
        prediction=prediction,
        gold_answer=problem.get("answer"),
        correct=is_correct(prediction, problem.get("answer")),
        total_tokens=total_tokens,
        solver_tokens=total_tokens,
        verifier_tokens=0,
        latency_ms=total_latency,
        branches_used=repeats,
        stop_reason="STANDARD_COT_MAJORITY_VOTE",
        actions=[str(candidate["action"]) for candidate in candidates],
        metadata={
            "votes": votes,
            "candidates": [
                {
                    "prediction": candidate["prediction"],
                    "completion_tokens": candidate["completion_tokens"],
                    "completion_head": candidate["text"][:300],
                }
                for candidate in candidates
            ],
        },
    )


def run_verifier_rerank(
    config: dict[str, Any],
    problem: dict[str, Any],
    repeats: int = 3,
) -> EpisodeRecord:
    env = StepwiseEnvironment(config)
    candidates: list[dict[str, Any]] = []
    total_solver = 0
    total_verifier = 0
    total_latency = 0
    for idx in range(repeats):
        candidate = _run_standard_direct_cot_candidate(
            {**config, "episode_seed": int(config.get("episode_seed", 0)) + idx},
            problem,
            action=f"STANDARD_DIRECT_COT_RERANK_{idx + 1}",
        )
        branch = _candidate_branch(candidate, branch_id=idx)
        verifier_result = env.verifier.score(problem, branch, env.rng)
        candidate["verifier_score"] = verifier_result.score
        candidate["verifier_candidate_answer"] = verifier_result.candidate_answer
        candidate["verifier_score_parse_success"] = verifier_result.score_parse_success
        candidate["verifier_explanation"] = verifier_result.explanation
        candidate["verifier_tokens"] = verifier_result.token_cost
        candidate["verifier_latency_ms"] = verifier_result.latency_ms
        candidates.append(candidate)
        total_solver += int(candidate["completion_tokens"])
        total_verifier += int(verifier_result.token_cost)
        total_latency += int(candidate["latency_ms"]) + int(verifier_result.latency_ms)
    best = max(
        candidates,
        key=lambda item: (
            float(item.get("verifier_score", 0.0)),
            normalize_answer(item.get("prediction")) is not None,
        ),
    )
    prediction = best.get("prediction")
    return EpisodeRecord(
        qid=problem["qid"],
        prediction=prediction,
        gold_answer=problem.get("answer"),
        correct=is_correct(prediction, problem.get("answer")),
        total_tokens=total_solver + total_verifier,
        solver_tokens=total_solver,
        verifier_tokens=total_verifier,
        latency_ms=total_latency,
        branches_used=repeats,
        stop_reason="VERIFIER_RERANK",
        actions=[str(candidate["action"]) for candidate in candidates] + ["VERIFIER_RERANK"],
        metadata={
            "selected_prediction": prediction,
            "selected_verifier_score": best.get("verifier_score"),
            "candidates": [
                {
                    "prediction": candidate["prediction"],
                    "completion_tokens": candidate["completion_tokens"],
                    "verifier_score": candidate.get("verifier_score"),
                    "verifier_candidate_answer": candidate.get("verifier_candidate_answer"),
                    "completion_head": candidate["text"][:300],
                }
                for candidate in candidates
            ],
        },
    )


BASELINE_REGISTRY: dict[str, Callable[..., EpisodeRecord]] = {
    "standard_direct_cot": run_standard_direct_cot,
    "full_cot": lambda config, problem: run_policy_episode(
        StepwiseEnvironment(config),
        problem,
        full_cot_policy,
        min_steps_before_stop=2,
    ),
    "self_consistency": lambda config, problem: run_self_consistency(config, problem, repeats=3),
    "verifier_rerank": lambda config, problem: run_verifier_rerank(config, problem, repeats=3),
    "stop_confidence_heuristic": lambda config, problem: run_policy_episode(
        StepwiseEnvironment(config),
        problem,
        stop_confidence_heuristic_policy,
        min_steps_before_stop=1,
    ),
    "verify_then_stop_heuristic": lambda config, problem: run_policy_episode(
        StepwiseEnvironment(config),
        problem,
        verify_then_stop_heuristic_policy,
        min_steps_before_stop=1,
    ),
    "adaptive_verify_branch_heuristic": lambda config, problem: run_policy_episode(
        StepwiseEnvironment(config),
        problem,
        adaptive_verify_branch_heuristic_policy,
        min_steps_before_stop=1,
    ),
}
