from __future__ import annotations

import re
from collections import Counter
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import numpy as np

from bora.answer_extraction import extract_explicit_answer, extract_numeric_answer
from bora.common import is_correct, normalize_answer
from bora.runtime import (
    StepwiseEnvironment,
    aggregate_final_answer,
    detect_negative_flip,
    select_best_branch,
)
from bora.types import Branch, State, TransitionDelta, VerifierResult
from bora.types import Checkpoint


ACCEPT_SEED = "ACCEPT_SEED"
VERIFY_ONLY = "VERIFY_ONLY"
RESEED_1 = "RESEED_1"
RESEED_2_VOTE = "RESEED_2_VOTE"
THINK_RESEED = "THINK_RESEED"
THINK_VERIFY = "THINK_VERIFY"
THINK_VERIFY_2 = "THINK_VERIFY_2"
BRANCH_2 = "BRANCH_2"
THINK_REPAIR = "THINK_REPAIR"
VERIFY_ADOPT = "VERIFY_ADOPT"

GATE_ACCEPT = "ACCEPT"
GATE_RESCUE = "RESCUE"

RESCUE_ACTIONS = [
    ACCEPT_SEED,
    VERIFY_ONLY,
    RESEED_1,
    RESEED_2_VOTE,
    THINK_RESEED,
    THINK_VERIFY,
    THINK_VERIFY_2,
    BRANCH_2,
    THINK_REPAIR,
    VERIFY_ADOPT,
]

RESCUE_ONLY_ACTIONS = [
    VERIFY_ONLY,
    RESEED_1,
    RESEED_2_VOTE,
    THINK_RESEED,
    THINK_VERIFY,
    THINK_VERIFY_2,
    BRANCH_2,
    THINK_REPAIR,
    VERIFY_ADOPT,
]

SEED_FEATURE_NAMES = [
    "answer_parse_success",
    "answer_format_quality",
    "explicit_answer_present",
    "multiple_explicit_answers",
    "multiple_numeric_answers",
    "trace_chars_norm",
    "trace_words_norm",
    "final_answer_position",
    "seed_confidence",
    "seed_done",
    "seed_solver_tokens_norm",
    "remaining_budget_ratio",
    "answer_is_integer",
    "answer_is_decimal",
    "answer_is_negative",
    "answer_history_len_norm",
    "same_answer_streak_norm",
    "answer_flip_count_norm",
    "verifier_used",
    "verifier_score",
    "verifier_score_parse_success",
    "verifier_candidate_exists",
    "verifier_answer_agrees",
    "verifier_answer_disagrees",
    "verifier_rationale_length_norm",
    "short_or_degenerate_reasoning",
    "arithmetic_pattern_risk",
    "malformed_final_answer",
    "bias",
]

RISK_TERMS = [
    "percent",
    "percentage",
    "ratio",
    "fraction",
    "half",
    "third",
    "quarter",
    "rate",
    "per ",
    "speed",
    "time",
    "hour",
    "minute",
    "cents",
    "dollar",
    "profit",
    "left over",
]

HIGH_RISK_CLEAN_TERMS = [
    "older",
    "years old",
    "in 5 years",
    "grade",
    "assignment",
    "minimum",
    "whole number",
    "average",
    "tire",
    "tires",
    "tricycle",
    "bicycle",
    "vehicle",
    "transportation",
    "per minute",
    "per second",
    "jump",
]


@dataclass
class SeedStateFeatures:
    vector: np.ndarray
    metadata: dict[str, Any]


@dataclass
class RescueExecution:
    action: str
    final_answer: str | None
    extra_solver_tokens: int
    extra_verifier_tokens: int
    extra_latency_ms: int
    verifier_calls: int
    candidates: list[str | None]
    metadata: dict[str, Any]

    @property
    def extra_total_tokens(self) -> int:
        return self.extra_solver_tokens + self.extra_verifier_tokens


def _norm01(value: float, scale: float) -> float:
    if scale <= 0:
        return 0.0
    return min(max(value / scale, 0.0), 1.0)


def _explicit_candidates(trace: str) -> list[str]:
    candidates: list[str] = []
    for match in re.finditer(r"\\boxed\{([^{}]+)\}", trace or ""):
        value = extract_numeric_answer(match.group(1)) or match.group(1).strip()
        if value:
            candidates.append(value)
    for line in (trace or "").splitlines():
        value = extract_explicit_answer(line, prefer_numeric=True)
        if value:
            candidates.append(value)
    return candidates


def _same_answer_streak(branch: Branch) -> int:
    if not branch.answer_history:
        return 0
    target = normalize_answer(branch.answer_history[-1])
    streak = 0
    for answer in reversed(branch.answer_history):
        if normalize_answer(answer) == target:
            streak += 1
        else:
            break
    return streak


def _answer_flip_count(branch: Branch) -> int:
    flips = 0
    previous: str | None = None
    for answer in branch.answer_history:
        normalized = normalize_answer(answer)
        if normalized is None:
            continue
        if previous is not None and normalized != previous:
            flips += 1
        previous = normalized
    return flips


def _answer_type_features(answer: str | None) -> tuple[float, float, float]:
    normalized = normalize_answer(answer)
    if normalized is None:
        return 0.0, 0.0, 0.0
    return (
        float(re.fullmatch(r"-?\d+", normalized) is not None),
        float(re.fullmatch(r"-?\d+\.\d+", normalized) is not None),
        float(normalized.startswith("-")),
    )


def _arithmetic_pattern_risk(question: str) -> bool:
    lowered = question.lower()
    return any(term in lowered for term in RISK_TERMS)


def _high_risk_clean_reasons(
    env: StepwiseEnvironment,
    *,
    question: str,
    trace_words: int,
    answer: str | None,
) -> list[str]:
    cfg = env.config.get("rescue_bandit", {})
    if not bool(cfg.get("trigger_high_risk_clean_seed", False)):
        return []
    reasons: list[str] = []
    if trace_words >= int(cfg.get("high_risk_clean_trace_words", 300)):
        reasons.append("long_reasoning")
    normalized = normalize_answer(answer)
    if normalized is not None and re.fullmatch(r"-?\d+\.\d+", normalized):
        reasons.append("decimal_answer")
    lowered = question.lower()
    terms = cfg.get("high_risk_clean_terms", HIGH_RISK_CLEAN_TERMS)
    matched_terms = [str(term) for term in terms if str(term).lower() in lowered]
    if matched_terms:
        reasons.append("risk_terms:" + ",".join(matched_terms[:3]))
    return reasons


def build_seed_state(
    env: StepwiseEnvironment,
    state: State,
    *,
    verifier_result: VerifierResult | None = None,
) -> SeedStateFeatures:
    branch = select_best_branch(state.branches, prefer_undone=False)
    answer_norm = normalize_answer(branch.current_answer)
    parse_success = answer_norm is not None
    explicit_candidates = _explicit_candidates(branch.trace)
    normalized_explicit = [normalize_answer(item) for item in explicit_candidates]
    normalized_explicit = [item for item in normalized_explicit if item is not None]
    explicit_present = bool(normalized_explicit)
    multiple_explicit = len(set(normalized_explicit)) > 1

    numeric_candidates = re.findall(r"-?\d+(?:\.\d+)?", branch.trace or "")
    multiple_numeric = len(set(numeric_candidates[-5:])) > 1 if numeric_candidates else False

    answer_format_quality = 0.0
    if explicit_present:
        answer_format_quality = 1.0
    elif parse_success:
        answer_format_quality = 0.5

    trace = branch.trace or ""
    trace_chars = len(trace)
    trace_words = len(trace.split())
    final_position = 0.0
    if answer_norm is not None and trace:
        idx = trace.rfind(str(branch.current_answer or answer_norm))
        final_position = (idx + 1) / len(trace) if idx >= 0 else 0.0

    verifier_score = 0.5
    verifier_used = verifier_result is not None
    verifier_parse_success = False
    verifier_candidate_exists = False
    verifier_answer_agrees = False
    verifier_answer_disagrees = False
    verifier_rationale_len = 0
    verifier_candidate_norm: str | None = None
    if verifier_result is not None:
        verifier_score = float(verifier_result.score)
        verifier_parse_success = bool(verifier_result.score_parse_success)
        verifier_candidate_norm = normalize_answer(verifier_result.candidate_answer)
        verifier_candidate_exists = verifier_candidate_norm is not None
        verifier_answer_agrees = (
            verifier_candidate_norm is not None
            and answer_norm is not None
            and verifier_candidate_norm == answer_norm
        )
        verifier_answer_disagrees = (
            verifier_candidate_norm is not None
            and answer_norm is not None
            and verifier_candidate_norm != answer_norm
        )
        verifier_rationale_len = len((verifier_result.explanation or "").split())

    short_or_degenerate = trace_words < int(
        env.config.get("rescue_bandit", {}).get("min_seed_reasoning_words", 35)
    )
    malformed_final_answer = parse_success and not explicit_present
    answer_is_integer, answer_is_decimal, answer_is_negative = _answer_type_features(branch.current_answer)
    remaining_budget_ratio = env.remaining_budget(state) / max(state.total_budget, 1)
    seed_solver_tokens_norm = branch.solver_tokens / max(state.total_budget, 1)
    clean_confidence = float(env.config.get("rescue_bandit", {}).get("clean_confidence", 0.92))
    high_risk_clean_reasons = _high_risk_clean_reasons(
        env,
        question=state.question,
        trace_words=trace_words,
        answer=branch.current_answer,
    )
    high_risk_clean_seed = (
        bool(high_risk_clean_reasons)
        and parse_success
        and explicit_present
        and not multiple_explicit
        and bool(branch.done)
        and float(branch.confidence) >= clean_confidence
    )
    triggers = []
    if not parse_success:
        triggers.append("answer_unparseable")
    if malformed_final_answer:
        triggers.append("malformed_final_answer")
    if multiple_explicit:
        triggers.append("multiple_explicit_answers")
    if verifier_used and not verifier_parse_success:
        triggers.append("verifier_output_unparseable")
    if verifier_answer_disagrees:
        triggers.append("verifier_answer_disagrees_with_seed")
    if short_or_degenerate:
        triggers.append("short_or_degenerate_reasoning")
    if bool(env.config.get("rescue_bandit", {}).get("trigger_arithmetic_pattern_risk", False)):
        if _arithmetic_pattern_risk(state.question):
            triggers.append("arithmetic_pattern_risk")
    if high_risk_clean_seed:
        triggers.append("high_risk_clean_seed")

    metadata = {
        "feature_names": list(SEED_FEATURE_NAMES),
        "seed_answer": branch.current_answer,
        "seed_answer_normalized": answer_norm,
        "answer_parse_success": parse_success,
        "answer_format_quality": answer_format_quality,
        "explicit_answer_present": explicit_present,
        "multiple_explicit_answers": multiple_explicit,
        "multiple_numeric_answers": multiple_numeric,
        "trace_chars": trace_chars,
        "trace_words": trace_words,
        "final_answer_position": final_position,
        "seed_confidence": branch.confidence,
        "seed_done": branch.done,
        "verifier_used": verifier_used,
        "verifier_score": verifier_score if verifier_used else None,
        "verifier_score_parse_success": verifier_parse_success,
        "verifier_candidate_answer": verifier_result.candidate_answer if verifier_result else None,
        "verifier_candidate_answer_normalized": verifier_candidate_norm,
        "verifier_answer_agrees": verifier_answer_agrees,
        "verifier_answer_disagrees": verifier_answer_disagrees,
        "short_or_degenerate_reasoning": short_or_degenerate,
        "arithmetic_pattern_risk": _arithmetic_pattern_risk(state.question),
        "high_risk_clean_seed": high_risk_clean_seed,
        "high_risk_clean_reasons": high_risk_clean_reasons,
        "triggers": triggers,
    }
    values = [
        float(parse_success),
        answer_format_quality,
        float(explicit_present),
        float(multiple_explicit),
        float(multiple_numeric),
        _norm01(trace_chars, 6000.0),
        _norm01(trace_words, 800.0),
        final_position,
        branch.confidence,
        float(branch.done),
        seed_solver_tokens_norm,
        remaining_budget_ratio,
        answer_is_integer,
        answer_is_decimal,
        answer_is_negative,
        _norm01(len(branch.answer_history), 5.0),
        _norm01(_same_answer_streak(branch), 5.0),
        _norm01(_answer_flip_count(branch), 5.0),
        float(verifier_used),
        verifier_score,
        float(verifier_parse_success),
        float(verifier_candidate_exists),
        float(verifier_answer_agrees),
        float(verifier_answer_disagrees),
        _norm01(verifier_rationale_len, 120.0),
        float(short_or_degenerate),
        float(_arithmetic_pattern_risk(state.question)),
        float(malformed_final_answer),
        1.0,
    ]
    return SeedStateFeatures(vector=np.asarray(values, dtype=float), metadata=metadata)


def is_clean_high_confidence(
    seed_features: SeedStateFeatures,
    *,
    min_confidence: float = 0.92,
) -> bool:
    meta = seed_features.metadata
    return (
        not meta["triggers"]
        and bool(meta["answer_parse_success"])
        and bool(meta["seed_done"])
        and float(meta["seed_confidence"]) >= min_confidence
    )


def should_prioritize_verify_adopt(
    seed_features: SeedStateFeatures,
    rescue_cfg: dict[str, Any],
    *,
    clean_confidence: float,
) -> bool:
    if not bool(rescue_cfg.get("prefer_verify_adopt_on_disagreement", True)):
        return False
    meta = seed_features.metadata
    triggers = set(meta.get("triggers", []))
    has_candidate = normalize_answer(meta.get("verifier_candidate_answer")) is not None
    if bool(rescue_cfg.get("verify_adopt_disagreement_require_candidate", True)) and not has_candidate:
        return False
    has_disagreement = bool(meta.get("verifier_answer_disagrees")) or (
        "verifier_answer_disagrees_with_seed" in triggers
    )
    if not has_disagreement:
        return False
    if not bool(rescue_cfg.get("verify_adopt_disagreement_require_malformed_or_low_confidence", True)):
        return True
    is_malformed_or_unparseable = bool(
        {"malformed_final_answer", "answer_unparseable"} & triggers
    )
    is_low_confidence = float(meta.get("seed_confidence", 1.0)) < clean_confidence
    return is_malformed_or_unparseable or is_low_confidence


def constrain_rescue_actions_for_seed(
    seed_features: SeedStateFeatures,
    feasible_actions: list[str],
    rescue_cfg: dict[str, Any],
    *,
    clean_confidence: float,
) -> tuple[list[str], str | None]:
    if should_prioritize_verify_adopt(
        seed_features,
        rescue_cfg,
        clean_confidence=clean_confidence,
    ):
        constrained = [
            action
            for action in (VERIFY_ADOPT, VERIFY_ONLY)
            if action in feasible_actions
        ]
        if constrained:
            return constrained, "verifier_disagreement_verify_adopt"
    triggers = set(seed_features.metadata.get("triggers", []))
    if "high_risk_clean_seed" in triggers:
        reasons = set(seed_features.metadata.get("high_risk_clean_reasons", []))
        if "decimal_answer" in reasons:
            requested = rescue_cfg.get("high_risk_clean_decimal_actions", [RESEED_1])
        else:
            requested = rescue_cfg.get("high_risk_clean_actions", [VERIFY_ADOPT])
        constrained = [action for action in requested if action in feasible_actions]
        if constrained:
            return constrained, "high_risk_clean"
    return feasible_actions, None


def score_seed_without_adoption(
    env: StepwiseEnvironment,
    state: State,
    problem: dict[str, Any],
    *,
    action_label: str = "SEED_VERIFY_FEATURE",
) -> VerifierResult:
    branch = select_best_branch(state.branches, prefer_undone=False)
    result = env.verifier.score(problem, branch, env.rng)
    branch.prm_scores.append(result.score)
    branch.prm_mean = float(np.mean(branch.prm_scores))
    branch.prm_min = float(np.min(branch.prm_scores))
    branch.verifier_tokens += result.token_cost
    state.spent_verifier_tokens += result.token_cost
    state.spent_latency_ms += result.latency_ms
    state.action_history.append(action_label)
    return result


def score_candidate_without_adoption(
    env: StepwiseEnvironment,
    state: State,
    problem: dict[str, Any],
    *,
    candidate_answer: str | None,
    action_label: str,
) -> tuple[VerifierResult, TransitionDelta]:
    seed_branch = select_best_branch(state.branches, prefer_undone=False)
    candidate_branch = deepcopy(seed_branch)
    candidate_branch.current_answer = candidate_answer
    candidate_branch.done = normalize_answer(candidate_answer) is not None
    if candidate_answer is not None:
        candidate_branch.answer_history = list(candidate_branch.answer_history) + [candidate_answer]
    candidate_branch.trace = (
        f"{candidate_branch.trace.strip()}\n\n"
        f"Independent candidate answer to verify: {candidate_answer}"
    ).strip()
    delta = TransitionDelta(
        action=action_label,
        executed_branch_id=seed_branch.branch_id,
        remaining_budget_before=env.remaining_budget(state),
    )
    result = env.verifier.score(problem, candidate_branch, env.rng)
    state.spent_verifier_tokens += result.token_cost
    state.spent_latency_ms += result.latency_ms
    state.action_history.append(action_label)
    delta.verifier_tokens = result.token_cost
    delta.latency_ms = result.latency_ms
    delta.verifier_calls = 1
    delta.remaining_budget_after = env.remaining_budget(state)
    return result, delta


def prepare_seed_features(
    env: StepwiseEnvironment,
    state: State,
    problem: dict[str, Any],
    *,
    verify_uncertain: bool,
    clean_confidence: float,
) -> tuple[SeedStateFeatures, VerifierResult | None]:
    seed_features = build_seed_state(env, state)
    verifier_result = None
    if (
        verify_uncertain
        and not is_clean_high_confidence(seed_features, min_confidence=clean_confidence)
        and env.remaining_budget(state) >= int(env.config.get("verifier", {}).get("tokens_per_call", 64))
    ):
        verifier_result = score_seed_without_adoption(env, state, problem)
        seed_features = build_seed_state(env, state, verifier_result=verifier_result)
    return seed_features, verifier_result


def feasible_rescue_actions(
    env: StepwiseEnvironment,
    state: State,
    actions: list[str] | None = None,
) -> list[str]:
    rescue_cfg = env.config.get("rescue_bandit", {})
    remaining = env.remaining_budget(state)
    verify_cost = int(env.config.get("verifier", {}).get("tokens_per_call", 64))
    reseed_tokens = int(rescue_cfg.get("reseed_tokens", 384))
    think_reseed_tokens = int(rescue_cfg.get("think_reseed_tokens", reseed_tokens))
    think_verify_tokens = int(rescue_cfg.get("think_verify_tokens", think_reseed_tokens))
    think_verify_2_tokens = int(rescue_cfg.get("think_verify_2_tokens", think_verify_tokens))
    repair_tokens = int(env.config.get("solver", {}).get("max_new_tokens_rescue", 192))
    branch_tokens = int(env.config.get("solver", {}).get("max_new_tokens_short", 96))
    requested = actions or list(rescue_cfg.get("actions", RESCUE_ONLY_ACTIONS))
    feasible: list[str] = []
    for action in requested:
        if action == ACCEPT_SEED:
            feasible.append(action)
        elif action in {VERIFY_ONLY, VERIFY_ADOPT} and remaining >= verify_cost:
            feasible.append(action)
        elif action == RESEED_1 and remaining >= min(reseed_tokens, 64):
            feasible.append(action)
        elif action == RESEED_2_VOTE and remaining >= 2 * min(reseed_tokens, 64):
            feasible.append(action)
        elif action == THINK_RESEED and remaining >= min(think_reseed_tokens, 64):
            feasible.append(action)
        elif action == THINK_VERIFY and remaining >= min(think_verify_tokens, 64):
            feasible.append(action)
        elif action == THINK_VERIFY_2 and remaining >= 2 * min(think_verify_2_tokens, 64):
            feasible.append(action)
        elif action == THINK_REPAIR and remaining >= min(repair_tokens, 64):
            feasible.append(action)
        elif (
            action == BRANCH_2
            and remaining >= branch_tokens
            and int(env.config.get("max_active_branches", 1)) >= 3
        ):
            feasible.append(action)
    return feasible


def _record_delta(delta: TransitionDelta, deltas: list[dict[str, Any]]) -> None:
    deltas.append(
        {
            "action": delta.action,
            "solver_tokens": delta.solver_tokens,
            "verifier_tokens": delta.verifier_tokens,
            "latency_ms": delta.latency_ms,
            "verifier_calls": delta.verifier_calls,
            "remaining_budget_before": delta.remaining_budget_before,
            "remaining_budget_after": delta.remaining_budget_after,
        }
    )


def _vote(candidates: list[str | None], *, default: str | None) -> str | None:
    normalized = [normalize_answer(candidate) for candidate in candidates]
    normalized = [candidate for candidate in normalized if candidate is not None]
    if not normalized:
        return normalize_answer(default)
    counts = Counter(normalized)
    return counts.most_common(1)[0][0]


def _set_root_answer(state: State, answer: str | None, *, confidence: float = 0.88) -> None:
    branch = select_best_branch(state.branches, prefer_undone=False)
    if answer is None:
        return
    branch.current_answer = answer
    branch.answer_history.append(answer)
    branch.confidence = max(branch.confidence, confidence)
    branch.done = normalize_answer(answer) is not None


def _restore_root_answer(
    state: State,
    answer: str | None,
    *,
    confidence: float,
    done: bool,
) -> None:
    branch = select_best_branch(state.branches, prefer_undone=False)
    branch.current_answer = answer
    branch.confidence = confidence
    branch.done = done
    if answer is not None:
        branch.answer_history.append(answer)


def _apply_thinking_rescue_candidate(
    env: StepwiseEnvironment,
    state: State,
    *,
    max_tokens: int,
    action: str,
) -> tuple[TransitionDelta, str | None]:
    branch = select_best_branch(state.branches, prefer_undone=False)
    delta = TransitionDelta(
        action=action,
        executed_branch_id=branch.branch_id,
        remaining_budget_before=env.remaining_budget(state),
    )
    actual_max_tokens = min(max_tokens, env.remaining_budget(state))
    if actual_max_tokens <= 0:
        delta.remaining_budget_after = env.remaining_budget(state)
        return delta, branch.current_answer
    output = env.solver.generate(
        env.problem or {},
        branch,
        "rescue",
        actual_max_tokens,
        env.rng,
        enable_thinking=True,
    )
    checkpoint = Checkpoint(
        step_index=branch.depth,
        step_text=output.step_text,
        current_answer=output.current_answer,
        confidence=output.confidence,
        done=output.done,
        action=action,
        token_cost=output.token_cost,
    )
    branch.checkpoints.append(checkpoint)
    branch.trace = (branch.trace + "\n\n" + output.step_text).strip()
    branch.current_answer = output.current_answer
    branch.confidence = output.confidence
    branch.done = output.done
    branch.solver_tokens += output.token_cost
    if output.current_answer is not None:
        branch.answer_history.append(output.current_answer)
    branch.conf_history.append(output.confidence)
    branch.action_history.append(action)

    state.spent_solver_tokens += output.token_cost
    state.spent_latency_ms += output.latency_ms
    delta.solver_tokens += output.token_cost
    delta.latency_ms += output.latency_ms
    delta.steps_added += 1
    delta.remaining_budget_after = env.remaining_budget(state)
    return delta, output.current_answer


def _sample_thinking_rescue_candidate(
    env: StepwiseEnvironment,
    state: State,
    *,
    max_tokens: int,
    action: str,
) -> tuple[TransitionDelta, str | None, dict[str, Any]]:
    """Sample a thinking candidate without mutating the live seed branch."""
    seed_branch = select_best_branch(state.branches, prefer_undone=False)
    sample_branch = deepcopy(seed_branch)
    delta = TransitionDelta(
        action=action,
        executed_branch_id=seed_branch.branch_id,
        remaining_budget_before=env.remaining_budget(state),
    )
    actual_max_tokens = min(max_tokens, env.remaining_budget(state))
    if actual_max_tokens <= 0:
        delta.remaining_budget_after = env.remaining_budget(state)
        return delta, seed_branch.current_answer, {"skipped": True}
    output = env.solver.generate(
        env.problem or {},
        sample_branch,
        "rescue",
        actual_max_tokens,
        env.rng,
        enable_thinking=True,
    )
    state.spent_solver_tokens += output.token_cost
    state.spent_latency_ms += output.latency_ms
    delta.solver_tokens += output.token_cost
    delta.latency_ms += output.latency_ms
    delta.steps_added += 1
    delta.remaining_budget_after = env.remaining_budget(state)
    return (
        delta,
        output.current_answer,
        {
            "thinking_answer": output.current_answer,
            "thinking_answer_normalized": normalize_answer(output.current_answer),
            "confidence": output.confidence,
            "done": output.done,
            "token_cost": output.token_cost,
            "latency_ms": output.latency_ms,
            "step_text_head": output.step_text[:500],
        },
    )


def _thinking_agreement_gate(
    *,
    seed_answer: str | None,
    thinking_answer: str | None,
    verifier_result: VerifierResult | None,
    rescue_cfg: dict[str, Any],
) -> tuple[bool, str]:
    seed_norm = normalize_answer(seed_answer)
    thinking_norm = normalize_answer(thinking_answer)
    verifier_norm = normalize_answer(verifier_result.candidate_answer if verifier_result else None)
    if thinking_norm is None:
        return False, "no_thinking_candidate"
    if seed_norm is not None and thinking_norm == seed_norm:
        return False, "thinking_agrees_with_seed"
    if verifier_norm is not None and thinking_norm == verifier_norm:
        return True, "thinking_agrees_with_verifier_candidate"
    if (
        seed_norm is None
        and bool(rescue_cfg.get("think_verify_adopt_unparseable_seed_without_verifier", False))
    ):
        return True, "unparseable_seed_thinking_candidate"
    return False, "no_independent_agreement"


def _thinking_pair_agreement_gate(
    *,
    seed_answer: str | None,
    thinking_answers: list[str | None],
) -> tuple[bool, str]:
    seed_norm = normalize_answer(seed_answer)
    normalized = [normalize_answer(answer) for answer in thinking_answers]
    if len(normalized) < 2:
        return False, "missing_thinking_sample"
    if any(answer is None for answer in normalized):
        return False, "missing_thinking_candidate"
    if len(set(normalized)) != 1:
        return False, "thinking_samples_disagree"
    agreed = normalized[0]
    if seed_norm is not None and agreed == seed_norm:
        return False, "thinking_pair_agrees_with_seed"
    return True, "thinking_pair_agreement"


def _thinking_pair_cost_guard(
    *,
    sample_results: list[dict[str, Any]],
    max_tokens: int,
    rescue_cfg: dict[str, Any],
) -> tuple[bool, str]:
    token_costs = [
        int(sample.get("token_cost", 0))
        for sample in sample_results
        if isinstance(sample.get("token_cost"), (int, float))
    ]
    if len(token_costs) < 2:
        return False, "cost_guard_missing_sample_tokens"
    max_single = int(rescue_cfg.get("think_verify_2_max_single_tokens", max_tokens))
    max_total = int(rescue_cfg.get("think_verify_2_max_total_tokens", 2 * max_tokens))
    cap_ratio = float(rescue_cfg.get("think_verify_2_cap_ratio", 0.95))
    cap_margin_tokens = int(rescue_cfg.get("think_verify_2_cap_margin_tokens", 0))
    for token_cost in token_costs:
        if token_cost > max_single:
            return False, "cost_guard_single_sample_too_expensive"
        if max_tokens > 0 and token_cost >= int(max_tokens * cap_ratio):
            return False, "cost_guard_sample_near_cap"
        if cap_margin_tokens > 0 and token_cost >= max_tokens - cap_margin_tokens:
            return False, "cost_guard_sample_near_cap_margin"
    if sum(token_costs) > max_total:
        return False, "cost_guard_total_too_expensive"
    return True, "cost_guard_passed"


def _thinking_pair_confirmation_gate(
    *,
    seed_answer: str | None,
    agreed_answer: str | None,
    verifier_result: VerifierResult | None,
    rescue_cfg: dict[str, Any],
) -> tuple[bool, str]:
    if not bool(rescue_cfg.get("think_verify_2_require_confirmation", False)):
        return True, "confirmation_not_required"
    agreed_norm = normalize_answer(agreed_answer)
    seed_norm = normalize_answer(seed_answer)
    verifier_norm = normalize_answer(verifier_result.candidate_answer if verifier_result else None)
    if agreed_norm is None:
        return False, "confirmation_missing_agreed_answer"
    if verifier_result is None:
        return False, "confirmation_missing_verifier"
    if verifier_norm is not None and verifier_norm == agreed_norm:
        return True, "confirmation_verifier_candidate_agrees"
    if verifier_norm is not None and seed_norm is not None and verifier_norm == seed_norm:
        return False, "confirmation_verifier_returns_seed"
    if (
        bool(rescue_cfg.get("think_verify_2_allow_high_score_confirmation", False))
        and bool(verifier_result.score_parse_success)
        and float(verifier_result.score)
        >= float(rescue_cfg.get("think_verify_2_confirm_score", 0.85))
    ):
        return True, "confirmation_verifier_high_score"
    return False, "confirmation_failed"


def execute_rescue_action(
    env: StepwiseEnvironment,
    state: State,
    problem: dict[str, Any],
    action: str,
    *,
    cached_verifier_result: VerifierResult | None = None,
) -> RescueExecution:
    start_solver = state.spent_solver_tokens
    start_verifier = state.spent_verifier_tokens
    start_latency = state.spent_latency_ms
    deltas: list[dict[str, Any]] = []
    candidates = [select_best_branch(state.branches, prefer_undone=False).current_answer]
    seed_answer = candidates[0]
    verifier_calls = 0
    metadata: dict[str, Any] = {"deltas": deltas}
    rescue_cfg = env.config.get("rescue_bandit", {})

    if action == ACCEPT_SEED:
        state.action_history.append(ACCEPT_SEED)
    elif action == VERIFY_ONLY:
        if cached_verifier_result is not None:
            result = cached_verifier_result
            state.action_history.append(f"{VERIFY_ONLY}_CACHED")
            metadata["used_cached_verifier"] = True
        else:
            result = score_seed_without_adoption(env, state, problem, action_label=VERIFY_ONLY)
            verifier_calls += 1
        metadata["verifier_result"] = {
            "score": result.score,
            "candidate_answer": result.candidate_answer,
            "score_parse_success": result.score_parse_success,
            "explanation": result.explanation,
        }
    elif action == VERIFY_ADOPT:
        if cached_verifier_result is not None:
            result = cached_verifier_result
            state.action_history.append(f"{VERIFY_ADOPT}_CACHED")
            metadata["used_cached_verifier"] = True
        else:
            result = score_seed_without_adoption(env, state, problem, action_label=VERIFY_ADOPT)
            verifier_calls += 1
        seed_norm = normalize_answer(candidates[0])
        candidate_norm = normalize_answer(result.candidate_answer)
        should_adopt = candidate_norm is not None and (
            candidate_norm != seed_norm
            or result.score < float(rescue_cfg.get("adopt_below_score", 0.35))
        )
        if should_adopt:
            _set_root_answer(
                state,
                result.candidate_answer,
                confidence=float(rescue_cfg.get("adopt_confidence", 0.88)),
            )
            candidates.append(result.candidate_answer)
        metadata["verifier_result"] = {
            "score": result.score,
            "candidate_answer": result.candidate_answer,
            "score_parse_success": result.score_parse_success,
            "adopted": should_adopt,
            "explanation": result.explanation,
        }
    elif action in {RESEED_1, RESEED_2_VOTE}:
        repeats = 1 if action == RESEED_1 else 2
        max_tokens = int(rescue_cfg.get("reseed_tokens", 384))
        for idx in range(repeats):
            delta = env._apply_standard_cot_seed(
                state,
                max_tokens,
                action=f"{action}_{idx + 1}" if repeats > 1 else action,
            )
            state.step_idx += 1
            state.action_history.append(delta.action)
            _record_delta(delta, deltas)
            candidates.append(select_best_branch(state.branches, prefer_undone=False).current_answer)
            if env.remaining_budget(state) <= 0:
                break
        voted = _vote(candidates, default=candidates[0])
        _set_root_answer(state, voted, confidence=float(rescue_cfg.get("vote_confidence", 0.90)))
        metadata["vote_candidates"] = candidates
        metadata["voted_answer"] = voted
    elif action == THINK_RESEED:
        max_tokens = int(rescue_cfg.get("think_reseed_tokens", rescue_cfg.get("reseed_tokens", 384)))
        delta = env._apply_standard_cot_seed(
            state,
            max_tokens,
            action=THINK_RESEED,
            enable_thinking=True,
        )
        state.step_idx += 1
        state.action_history.append(THINK_RESEED)
        _record_delta(delta, deltas)
        candidates.append(select_best_branch(state.branches, prefer_undone=False).current_answer)
        metadata["thinking_mode"] = "think"
        metadata["think_reseed_tokens"] = max_tokens
    elif action == THINK_VERIFY:
        branch = select_best_branch(state.branches, prefer_undone=False)
        seed_confidence = float(branch.confidence)
        seed_done = bool(branch.done)
        result = cached_verifier_result
        if result is not None:
            state.action_history.append(f"{THINK_VERIFY}_VERIFIER_CACHED")
            metadata["used_cached_verifier"] = True
        elif bool(rescue_cfg.get("think_verify_use_verifier", True)):
            result = score_seed_without_adoption(env, state, problem, action_label=f"{THINK_VERIFY}_SEED_VERIFY")
            verifier_calls += 1
        max_tokens = int(
            rescue_cfg.get(
                "think_verify_tokens",
                rescue_cfg.get("think_reseed_tokens", rescue_cfg.get("reseed_tokens", 384)),
            )
        )
        delta, thinking_answer = _apply_thinking_rescue_candidate(
            env,
            state,
            max_tokens=max_tokens,
            action=THINK_VERIFY,
        )
        state.step_idx += 1
        state.action_history.append(THINK_VERIFY)
        _record_delta(delta, deltas)
        candidates.append(thinking_answer)
        should_adopt, gate_reason = _thinking_agreement_gate(
            seed_answer=seed_answer,
            thinking_answer=thinking_answer,
            verifier_result=result,
            rescue_cfg=rescue_cfg,
        )
        if should_adopt:
            _set_root_answer(
                state,
                thinking_answer,
                confidence=float(rescue_cfg.get("think_verify_adopt_confidence", 0.90)),
            )
        else:
            _restore_root_answer(
                state,
                seed_answer,
                confidence=seed_confidence,
                done=seed_done,
            )
        metadata["thinking_mode"] = "think"
        metadata["think_verify_tokens"] = max_tokens
        metadata["think_verify_result"] = {
            "thinking_answer": thinking_answer,
            "verifier_candidate_answer": result.candidate_answer if result else None,
            "verifier_score": result.score if result else None,
            "adopted": should_adopt,
            "gate_reason": gate_reason,
        }
    elif action == THINK_VERIFY_2:
        branch = select_best_branch(state.branches, prefer_undone=False)
        seed_confidence = float(branch.confidence)
        seed_done = bool(branch.done)
        max_tokens = int(
            rescue_cfg.get(
                "think_verify_2_tokens",
                rescue_cfg.get(
                    "think_verify_tokens",
                    rescue_cfg.get("think_reseed_tokens", rescue_cfg.get("reseed_tokens", 384)),
                ),
            )
        )
        thinking_answers: list[str | None] = []
        sample_results: list[dict[str, Any]] = []
        for idx in range(2):
            delta, thinking_answer, sample_meta = _sample_thinking_rescue_candidate(
                env,
                state,
                max_tokens=max_tokens,
                action=f"{THINK_VERIFY_2}_{idx + 1}",
            )
            state.step_idx += 1
            state.action_history.append(delta.action)
            _record_delta(delta, deltas)
            thinking_answers.append(thinking_answer)
            candidates.append(thinking_answer)
            sample_results.append(sample_meta)
            if env.remaining_budget(state) <= 0:
                break
        state.action_history.append(THINK_VERIFY_2)
        should_adopt, gate_reason = _thinking_pair_agreement_gate(
            seed_answer=seed_answer,
            thinking_answers=thinking_answers,
        )
        cost_guard_passed = False
        cost_guard_reason = "cost_guard_not_checked"
        confirmation_result = None
        confirmation_passed = False
        confirmation_reason = "confirmation_not_checked"
        agreed_answer = thinking_answers[0] if should_adopt else None
        if should_adopt:
            cost_guard_passed, cost_guard_reason = _thinking_pair_cost_guard(
                sample_results=sample_results,
                max_tokens=max_tokens,
                rescue_cfg=rescue_cfg,
            )
            if not cost_guard_passed:
                should_adopt = False
                gate_reason = cost_guard_reason
        if should_adopt and bool(rescue_cfg.get("think_verify_2_require_confirmation", False)):
            confirmation_result, confirm_delta = score_candidate_without_adoption(
                env,
                state,
                problem,
                candidate_answer=agreed_answer,
                action_label=f"{THINK_VERIFY_2}_CONFIRM",
            )
            verifier_calls += 1
            _record_delta(confirm_delta, deltas)
            confirmation_passed, confirmation_reason = _thinking_pair_confirmation_gate(
                seed_answer=seed_answer,
                agreed_answer=agreed_answer,
                verifier_result=confirmation_result,
                rescue_cfg=rescue_cfg,
            )
            if not confirmation_passed:
                should_adopt = False
                gate_reason = confirmation_reason
        elif should_adopt:
            confirmation_passed, confirmation_reason = _thinking_pair_confirmation_gate(
                seed_answer=seed_answer,
                agreed_answer=agreed_answer,
                verifier_result=None,
                rescue_cfg=rescue_cfg,
            )
        if should_adopt:
            _set_root_answer(
                state,
                agreed_answer,
                confidence=float(rescue_cfg.get("think_verify_2_adopt_confidence", 0.91)),
            )
        else:
            _restore_root_answer(
                state,
                seed_answer,
                confidence=seed_confidence,
                done=seed_done,
            )
        metadata["thinking_mode"] = "think"
        metadata["think_verify_2_tokens"] = max_tokens
        metadata["think_verify_2_result"] = {
            "thinking_answers": thinking_answers,
            "thinking_answers_normalized": [normalize_answer(answer) for answer in thinking_answers],
            "agreed_answer": agreed_answer,
            "adopted": should_adopt,
            "gate_reason": gate_reason,
            "cost_guard_passed": cost_guard_passed,
            "cost_guard_reason": cost_guard_reason,
            "confirmation_passed": confirmation_passed,
            "confirmation_reason": confirmation_reason,
            "confirmation_verifier_result": {
                "score": confirmation_result.score,
                "candidate_answer": confirmation_result.candidate_answer,
                "score_parse_success": confirmation_result.score_parse_success,
                "explanation": confirmation_result.explanation,
            } if confirmation_result else None,
            "sample_results": sample_results,
        }
    elif action == THINK_REPAIR:
        branch = select_best_branch(state.branches, prefer_undone=False)
        state, delta = env.step(
            state,
            "THINK_192",
            target_branch_id=branch.branch_id,
            think_mode="rescue",
        )
        _record_delta(delta, deltas)
        candidates.append(select_best_branch(state.branches, prefer_undone=False).current_answer)
    elif action == BRANCH_2:
        for _ in range(2):
            if "BRANCH" not in env.feasible_actions(state):
                break
            branch = select_best_branch(state.branches, prefer_undone=False)
            state, delta = env.step(state, "BRANCH", target_branch_id=branch.branch_id)
            _record_delta(delta, deltas)
            candidates.append(select_best_branch(state.branches, prefer_undone=False).current_answer)
    else:
        raise ValueError(f"Unsupported rescue action: {action}")

    final_answer, aggregate_meta = aggregate_final_answer(state.branches)
    if action == BRANCH_2 and not bool(rescue_cfg.get("allow_branch_override", True)):
        metadata["branch_override_disabled"] = True
        metadata["branch_aggregate_answer"] = final_answer
        metadata["branch_seed_answer"] = seed_answer
        final_answer = normalize_answer(seed_answer)
    metadata["aggregate"] = aggregate_meta
    return RescueExecution(
        action=action,
        final_answer=final_answer,
        extra_solver_tokens=state.spent_solver_tokens - start_solver,
        extra_verifier_tokens=state.spent_verifier_tokens - start_verifier,
        extra_latency_ms=state.spent_latency_ms - start_latency,
        verifier_calls=verifier_calls,
        candidates=candidates,
        metadata=metadata,
    )


def clone_env_for_rescue(
    base_env: StepwiseEnvironment,
    problem: dict[str, Any],
    *,
    seed: int,
) -> StepwiseEnvironment:
    env = StepwiseEnvironment({**base_env.config, "episode_seed": seed})
    env.problem = problem
    env.next_branch_id = base_env.next_branch_id
    return env


def clone_state(state: State) -> State:
    return deepcopy(state)


def rescue_reward(
    *,
    seed_answer: str | None,
    final_answer: str | None,
    gold_answer: str | None,
    extra_total_tokens: int,
    extra_latency_ms: int,
    verifier_calls: int,
    total_budget: int,
    cost_cfg: dict[str, Any],
    negative_flip_penalty: float,
) -> float:
    seed_correct = is_correct(seed_answer, gold_answer)
    final_correct = is_correct(final_answer, gold_answer)
    negative_flip = seed_correct and not final_correct and normalize_answer(final_answer) is not None
    return float(final_correct) - float(seed_correct) - (
        float(cost_cfg.get("lambda_tok", 0.0)) * extra_total_tokens / max(total_budget, 1)
    ) - (
        float(cost_cfg.get("lambda_lat", 0.0))
        * extra_latency_ms
        / float(cost_cfg.get("lat_norm_ms", 8000.0))
    ) - (
        float(cost_cfg.get("lambda_ver", 0.0)) * verifier_calls
    ) - (
        negative_flip_penalty * float(negative_flip)
    )


def rescue_row_from_execution(
    *,
    problem: dict[str, Any],
    seed_state: State,
    seed_features: SeedStateFeatures,
    action: str,
    execution: RescueExecution,
    reward: float,
    state_id: str | None = None,
    source_tag: str | None = None,
    seed_rollout_tokens: int | None = None,
) -> dict[str, Any]:
    seed_branch = select_best_branch(seed_state.branches, prefer_undone=False)
    seed_answer = seed_branch.current_answer
    final_correct = is_correct(execution.final_answer, problem.get("answer"))
    seed_correct = is_correct(seed_answer, problem.get("answer"))
    row_state_id = state_id or str(problem["qid"])
    return {
        "qid": problem["qid"],
        "state_id": row_state_id,
        "source_tag": source_tag,
        "seed_rollout_tokens": seed_rollout_tokens,
        "features": seed_features.vector.tolist(),
        "feature_names": list(SEED_FEATURE_NAMES),
        "seed_metadata": seed_features.metadata,
        "action": action,
        "marginal_reward": reward,
        "seed_answer": seed_answer,
        "final_answer": execution.final_answer,
        "gold_answer": problem.get("answer"),
        "seed_correct": int(seed_correct),
        "final_correct": int(final_correct),
        "negative_flip": int(seed_correct and not final_correct),
        "extra_solver_tokens": execution.extra_solver_tokens,
        "extra_verifier_tokens": execution.extra_verifier_tokens,
        "extra_total_tokens": execution.extra_total_tokens,
        "extra_latency_ms": execution.extra_latency_ms,
        "verifier_calls": execution.verifier_calls,
        "candidates": execution.candidates,
        "execution_metadata": execution.metadata,
    }


def record_from_rescue_state(
    state: State,
    *,
    prediction: str | None,
    stop_reason: str,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    return {
        "qid": state.qid,
        "prediction": prediction,
        "gold_answer": state.gold_answer,
        "correct": is_correct(prediction, state.gold_answer),
        "total_tokens": state.total_tokens,
        "solver_tokens": state.spent_solver_tokens,
        "verifier_tokens": state.spent_verifier_tokens,
        "latency_ms": state.spent_latency_ms,
        "branches_used": len(state.branches),
        "stop_reason": stop_reason,
        "actions": list(state.action_history),
        "metadata": {
            **metadata,
            "negative_flip": detect_negative_flip(state),
            "difficulty": state.metadata.get("difficulty", "unknown"),
        },
    }
