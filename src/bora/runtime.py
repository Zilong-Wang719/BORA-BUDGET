from __future__ import annotations

from collections import Counter
from copy import deepcopy
from statistics import mean
from typing import Any

import numpy as np

from bora.bandit import LinTS
from bora.common import clamp, entropy_from_counts, is_correct, normalize_answer
from bora.proxy import predict_correctness
from bora.solver import build_solver
from bora.types import Branch, Checkpoint, EpisodeRecord, State, TransitionDelta
from bora.verifier import build_verifier


def same_answer_streak(branch: Branch) -> int:
    if not branch.answer_history:
        return 0
    streak = 0
    last = normalize_answer(branch.answer_history[-1])
    for answer in reversed(branch.answer_history):
        if normalize_answer(answer) == last:
            streak += 1
        else:
            break
    return streak


def answer_flip_count(branch: Branch) -> int:
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


def recent_stable_checkpoint_index(branch: Branch) -> int:
    if branch.depth <= 1:
        return max(branch.depth - 1, 0)
    checkpoints = branch.checkpoints
    for idx in range(len(checkpoints) - 1, 0, -1):
        current = checkpoints[idx]
        previous = checkpoints[idx - 1]
        if (
            normalize_answer(current.current_answer)
            and normalize_answer(current.current_answer)
            == normalize_answer(previous.current_answer)
            and current.confidence >= 0.50
            and previous.confidence >= 0.50
        ):
            return idx
    return max(branch.depth - 1, 0)


def branch_vote_shares(branches: list[Branch]) -> dict[int, float]:
    answers = [normalize_answer(branch.current_answer) for branch in branches]
    counter = Counter(answer for answer in answers if answer is not None)
    total = sum(counter.values())
    if total <= 0:
        return {branch.branch_id: 0.0 for branch in branches}
    return {
        branch.branch_id: counter.get(normalize_answer(branch.current_answer), 0) / total
        for branch in branches
    }


def branch_score(branch: Branch, vote_share: float = 0.0) -> float:
    length_norm = min(branch.depth / 6.0, 1.0)
    return (
        0.30 * vote_share
        + 0.25 * branch.prm_mean
        + 0.25 * branch.confidence
        + 0.10 * float(branch.done)
        - 0.10 * length_norm
    )


def sort_branches(branches: list[Branch]) -> list[Branch]:
    shares = branch_vote_shares(branches)
    for branch in branches:
        branch.score = branch_score(branch, shares.get(branch.branch_id, 0.0))
    return sorted(branches, key=lambda item: item.score, reverse=True)


def select_best_branch(branches: list[Branch], prefer_undone: bool = True) -> Branch:
    ranked = sort_branches(branches)
    if prefer_undone:
        for branch in ranked:
            if not branch.done:
                return branch
    return ranked[0]


def aggregate_final_answer(branches: list[Branch]) -> tuple[str | None, dict[str, Any]]:
    ranked = sort_branches(branches)
    votes: dict[str, float] = {}
    for branch in ranked:
        answer = normalize_answer(branch.current_answer)
        if answer is None:
            continue
        votes[answer] = votes.get(answer, 0.0) + max(branch.score, 0.0) + 1.0
    if votes:
        prediction = max(votes.items(), key=lambda item: item[1])[0]
    else:
        prediction = normalize_answer(ranked[0].current_answer) if ranked else None
    metadata = {
        "branch_scores": {branch.branch_id: branch.score for branch in ranked},
        "votes": votes,
    }
    return prediction, metadata


def clone_branch_prefix(branch: Branch, checkpoint_idx: int, new_branch_id: int) -> Branch:
    checkpoints = deepcopy(branch.checkpoints[: checkpoint_idx + 1])
    trace_parts = [checkpoint.step_text for checkpoint in checkpoints]
    answer_history = [cp.current_answer for cp in checkpoints if cp.current_answer is not None]
    conf_history = [cp.confidence for cp in checkpoints]
    current_answer = checkpoints[-1].current_answer if checkpoints else None
    confidence = checkpoints[-1].confidence if checkpoints else 0.0
    done = checkpoints[-1].done if checkpoints else False
    return Branch(
        branch_id=new_branch_id,
        parent_id=branch.branch_id,
        trace="\n\n".join(trace_parts),
        current_answer=current_answer,
        confidence=confidence,
        done=done,
        solver_tokens=sum(cp.token_cost for cp in checkpoints),
        verifier_tokens=0,
        prm_scores=[],
        prm_mean=0.0,
        prm_min=0.0,
        answer_history=answer_history,
        conf_history=conf_history,
        score=0.0,
        checkpoints=checkpoints,
        action_history=list(branch.action_history),
    )


def best_branch_correct(state: State) -> bool:
    best = select_best_branch(state.branches, prefer_undone=False)
    return is_correct(best.current_answer, state.gold_answer)


def detect_negative_flip(state: State) -> bool:
    gold = state.gold_answer
    for branch in state.branches:
        seen_correct = False
        for answer in branch.answer_history:
            if is_correct(answer, gold):
                seen_correct = True
            elif seen_correct and normalize_answer(answer) is not None:
                return True
    return False


class StepwiseEnvironment:
    def __init__(
        self,
        config: dict[str, Any],
        seed: int | None = None,
    ) -> None:
        self.config = config
        self.solver = build_solver(config)
        self.verifier = build_verifier(config)
        base_seed = int(
            config.get("episode_seed", config.get("random_seed", 0))
            if seed is None
            else seed
        )
        self.rng = np.random.default_rng(base_seed)
        self.problem: dict[str, Any] | None = None
        self.next_branch_id = 1

    def reset(self, problem: dict[str, Any]) -> State:
        self.problem = problem
        self.next_branch_id = 1
        root = Branch(branch_id=0, parent_id=None)
        return State(
            qid=str(problem["qid"]),
            question=str(problem["question"]),
            gold_answer=problem.get("answer"),
            branches=[root],
            spent_solver_tokens=0,
            spent_verifier_tokens=0,
            spent_latency_ms=0,
            step_idx=0,
            total_budget=int(self.config["total_budget"]),
            metadata={"difficulty": problem.get("difficulty", "unknown")},
            action_history=[],
        )

    def seed_rollout(self, state: State, max_tokens: int | None = None) -> State:
        tokens = max_tokens or int(self.config.get("seed_rollout_tokens", 64))
        seed_mode = str(
            self.config.get(
                "seed_rollout_mode",
                self.config.get("solver", {}).get("seed_rollout_mode", "stepwise"),
            )
        )
        if seed_mode == "standard_cot":
            self._apply_standard_cot_seed(state, tokens)
        else:
            self._apply_think(state, "THINK_64", tokens, mode="continue")
        state.action_history.append("SEED")
        return state

    def remaining_budget(self, state: State) -> int:
        return max(state.total_budget - state.total_tokens, 0)

    def get_branch(self, state: State, branch_id: int | None) -> Branch:
        if branch_id is None:
            return select_best_branch(state.branches, prefer_undone=False)
        for branch in state.branches:
            if branch.branch_id == branch_id:
                return branch
        raise KeyError(f"Unknown branch_id={branch_id}")

    def extract_features(
        self,
        state: State,
        decision_branch_id: int | None = None,
    ) -> np.ndarray:
        best = self.get_branch(state, decision_branch_id)
        spent_ratio = state.total_tokens / max(state.total_budget, 1)
        remaining_ratio = self.remaining_budget(state) / max(state.total_budget, 1)
        max_decisions = max(state.total_budget // 64, 1)
        step_idx_norm = state.step_idx / max_decisions

        answers = [normalize_answer(branch.current_answer) for branch in state.branches]
        counter = Counter(answer for answer in answers if answer is not None)
        total_answered = sum(counter.values())
        ordered_counts = sorted(counter.values(), reverse=True)
        top1 = ordered_counts[0] / total_answered if total_answered else 0.0
        top2 = ordered_counts[1] / total_answered if len(ordered_counts) > 1 else 0.0
        answer_entropy = entropy_from_counts(counter)

        conf_latest = best.confidence
        conf_slope = 0.0
        if len(best.conf_history) >= 2:
            conf_slope = best.conf_history[-1] - best.conf_history[-2]
        conf_window = best.conf_history[-3:]
        conf_var = float(np.var(conf_window)) if conf_window else 0.0

        prm_scores = [branch.prm_mean for branch in state.branches if branch.prm_scores]
        prm_mean_best = best.prm_mean
        prm_min_best = best.prm_min
        prm_var_pool = float(np.var(prm_scores)) if prm_scores else 0.0

        pool_flips = [answer_flip_count(branch) for branch in state.branches]
        avg_depth = mean(branch.depth for branch in state.branches)
        features = np.array(
            [
                spent_ratio,
                remaining_ratio,
                step_idx_norm,
                len(counter),
                top1,
                top1 - top2,
                answer_entropy,
                conf_latest,
                conf_slope,
                conf_var,
                float(best.done),
                prm_mean_best,
                prm_min_best,
                prm_var_pool,
                float(bool(prm_scores)),
                answer_flip_count(best),
                float(mean(pool_flips)) if pool_flips else 0.0,
                same_answer_streak(best),
                len(state.branches),
                best.depth,
                avg_depth,
                1.0,
            ],
            dtype=float,
        )
        return features

    def feasible_actions(self, state: State) -> list[str]:
        remaining = self.remaining_budget(state)
        verify_cost = int(self.config.get("verifier", {}).get("tokens_per_call", 64))
        feasible: list[str] = []
        if any(branch.current_answer for branch in state.branches):
            feasible.append("STOP")
        if remaining >= 16:
            feasible.append("THINK_64")
        if remaining >= 48:
            feasible.append("THINK_192")
        if remaining >= verify_cost and any(branch.current_answer for branch in state.branches):
            feasible.append("VERIFY")
        if remaining >= 16 and len(state.branches) < int(self.config.get("max_active_branches", 3)):
            feasible.append("BRANCH")
        return [action for action in self.config["actions"] if action in feasible]

    def budget_exhausted(self, state: State) -> bool:
        return self.remaining_budget(state) <= 0

    def final_aggregate(self, state: State) -> tuple[str | None, dict[str, Any]]:
        return aggregate_final_answer(state.branches)

    def step(
        self,
        state: State,
        action: str,
        target_branch_id: int | None = None,
        think_mode: str | None = None,
    ) -> tuple[State, TransitionDelta]:
        delta = TransitionDelta(
            action=action,
            decision_branch_id=target_branch_id,
            remaining_budget_before=self.remaining_budget(state),
        )
        if action == "THINK_64":
            delta = self._apply_think(
                state,
                action,
                int(self.config["solver"]["max_new_tokens_short"]),
                mode=think_mode or "continue",
                target_branch=self.get_branch(state, target_branch_id),
                delta=delta,
            )
        elif action == "THINK_192":
            max_tokens = int(self.config["solver"]["max_new_tokens_long"])
            if think_mode == "rescue":
                max_tokens = int(self.config["solver"].get("max_new_tokens_rescue", max_tokens))
            delta = self._apply_think(
                state,
                action,
                max_tokens,
                mode=think_mode or "continue",
                target_branch=self.get_branch(state, target_branch_id),
                delta=delta,
            )
        elif action == "VERIFY":
            delta = self._apply_verify(
                state,
                target_branch=self.get_branch(state, target_branch_id),
                delta=delta,
            )
        elif action == "BRANCH":
            delta = self._apply_branch(
                state,
                source_branch=self.get_branch(state, target_branch_id),
                delta=delta,
            )
        elif action == "STOP":
            delta.executed_branch_id = target_branch_id
        else:
            raise ValueError(f"Unsupported action: {action}")
        delta.remaining_budget_after = self.remaining_budget(state)
        state.step_idx += 1
        state.action_history.append(action)
        return state, delta

    def _apply_think(
        self,
        state: State,
        action: str,
        max_tokens: int,
        mode: str,
        target_branch: Branch | None = None,
        delta: TransitionDelta | None = None,
    ) -> TransitionDelta:
        branch = target_branch or select_best_branch(state.branches, prefer_undone=False)
        delta = delta or TransitionDelta(action=action)
        delta.executed_branch_id = branch.branch_id
        actual_max_tokens = min(max_tokens, self.remaining_budget(state))
        if actual_max_tokens <= 0:
            return delta
        blocks = 1 if mode == "rescue" or actual_max_tokens <= 64 else 3
        block_budget = max(actual_max_tokens // blocks, 16)
        for _ in range(blocks):
            remaining = self.remaining_budget(state)
            if remaining <= 0:
                break
            this_block_budget = min(block_budget, remaining)
            if this_block_budget <= 0:
                break
            output = self.solver.generate(
                self.problem or {},
                branch,
                mode,
                this_block_budget,
                self.rng,
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
            if output.done:
                break
        return delta

    def _apply_standard_cot_seed(
        self,
        state: State,
        max_tokens: int,
        action: str = "SEED",
        enable_thinking: bool | None = None,
    ) -> TransitionDelta:
        branch = self.get_branch(state, 0)
        delta = TransitionDelta(
            action=action,
            executed_branch_id=branch.branch_id,
            remaining_budget_before=self.remaining_budget(state),
        )
        actual_max_tokens = min(max_tokens, self.remaining_budget(state))
        if actual_max_tokens <= 0:
            delta.remaining_budget_after = self.remaining_budget(state)
            return delta
        output = self.solver.generate_standard_cot_seed(
            self.problem or {},
            actual_max_tokens,
            self.rng,
            enable_thinking=enable_thinking,
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
        delta.remaining_budget_after = self.remaining_budget(state)
        return delta

    def _apply_verify(
        self,
        state: State,
        target_branch: Branch | None = None,
        delta: TransitionDelta | None = None,
    ) -> TransitionDelta:
        delta = delta or TransitionDelta(action="VERIFY")
        if target_branch is not None:
            delta.executed_branch_id = target_branch.branch_id
        top_k = int(self.config.get("verify_topk", 2))
        tokens_per_call = int(self.config["verifier"]["tokens_per_call"])
        max_calls = self.remaining_budget(state) // tokens_per_call
        if max_calls <= 0:
            return delta
        for branch in sort_branches(state.branches)[: min(top_k, max_calls)]:
            result = self.verifier.score(self.problem or {}, branch, self.rng)
            branch.prm_scores.append(result.score)
            branch.prm_mean = float(np.mean(branch.prm_scores))
            branch.prm_min = float(np.min(branch.prm_scores))
            adopt_threshold = self.config.get("verifier", {}).get("adopt_reference_answer_below")
            if (
                adopt_threshold is not None
                and result.candidate_answer is not None
                and result.score < float(adopt_threshold)
                and normalize_answer(result.candidate_answer) is not None
            ):
                branch.current_answer = result.candidate_answer
                branch.answer_history.append(result.candidate_answer)
                branch.confidence = max(branch.confidence, float(self.config.get("verifier", {}).get("adopt_confidence", 0.88)))
                branch.done = True
            branch.verifier_tokens += result.token_cost

            state.spent_verifier_tokens += result.token_cost
            state.spent_latency_ms += result.latency_ms

            delta.verifier_tokens += result.token_cost
            delta.latency_ms += result.latency_ms
            delta.verifier_calls += 1
            delta.verified_branch_ids.append(branch.branch_id)
        return delta

    def _apply_branch(
        self,
        state: State,
        source_branch: Branch | None = None,
        delta: TransitionDelta | None = None,
    ) -> TransitionDelta:
        source = source_branch or select_best_branch(state.branches, prefer_undone=False)
        delta = delta or TransitionDelta(action="BRANCH")
        delta.source_branch_id = source.branch_id
        stable_idx = recent_stable_checkpoint_index(source)
        new_branch = clone_branch_prefix(source, stable_idx, self.next_branch_id)
        self.next_branch_id += 1
        state.branches.append(new_branch)
        delta.new_branch_id = new_branch.branch_id
        delta = self._apply_think(
            state,
            "BRANCH",
            int(self.config["solver"]["max_new_tokens_short"]),
            mode="branch",
            target_branch=new_branch,
            delta=delta,
        )
        if len(state.branches) > int(self.config.get("max_active_branches", 3)):
            ranked = sort_branches(state.branches)
            state.branches = ranked[: int(self.config.get("max_active_branches", 3))]
        return delta


def _record_from_state(
    state: State,
    prediction: str | None,
    metadata: dict[str, Any],
    stop_reason: str,
) -> EpisodeRecord:
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
            "rescue": state.metadata.get("rescue", {}),
        },
    )


def run_bora_episode(
    env: StepwiseEnvironment,
    bandit: LinTS,
    proxy_model: Any,
    initial_state: State,
) -> tuple[EpisodeRecord, State]:
    state = initial_state
    transition_log: list[dict[str, Any]] = []
    min_decisions_before_stop = int(env.config.get("bandit", {}).get("min_decisions_before_stop", 1))
    early_stop_if_done = bool(env.config.get("bandit", {}).get("early_stop_if_done", False))
    early_stop_confidence = float(env.config.get("bandit", {}).get("early_stop_confidence", 0.90))
    rescue_cfg = dict(env.config.get("rescue", {}))
    rescue_enabled = bool(rescue_cfg.get("enabled", False))
    rescue_checked = False

    def append_transition(delta: TransitionDelta, reason: str | None = None) -> None:
        transition_log.append(
            {
                "step_idx": state.step_idx,
                "action": delta.action,
                "reason": reason,
                "decision_branch_id": delta.decision_branch_id,
                "executed_branch_id": delta.executed_branch_id,
                "source_branch_id": delta.source_branch_id,
                "new_branch_id": delta.new_branch_id,
                "verified_branch_ids": list(delta.verified_branch_ids),
                "remaining_budget_before": delta.remaining_budget_before,
                "remaining_budget_after": delta.remaining_budget_after,
            }
        )

    while True:
        decision_branch = select_best_branch(state.branches, prefer_undone=False)
        if rescue_enabled and not rescue_checked:
            rescue_checked = True
            rescue_meta: dict[str, Any] = {
                "checked": True,
                "triggered": False,
                "reasons": [],
                "verifier_score": None,
            }
            state.metadata["rescue"] = rescue_meta
            reasons: list[str] = []
            if normalize_answer(decision_branch.current_answer) is None:
                reasons.append("malformed_answer")
            if not decision_branch.done:
                reasons.append("not_done")
            if decision_branch.confidence < float(rescue_cfg.get("low_confidence_threshold", 0.90)):
                reasons.append("low_confidence")

            if bool(rescue_cfg.get("verify_seed", False)) and "VERIFY" in env.feasible_actions(state):
                state, delta = env.step(
                    state,
                    "VERIFY",
                    target_branch_id=decision_branch.branch_id,
                )
                append_transition(delta, reason="rescue_gate_verify")
                decision_branch = select_best_branch(state.branches, prefer_undone=False)
                rescue_meta["verifier_score"] = decision_branch.prm_mean
                if decision_branch.prm_mean < float(rescue_cfg.get("verifier_accept_threshold", 0.70)):
                    reasons.append("weak_verifier")

            if reasons:
                rescue_meta["triggered"] = True
                rescue_meta["reasons"] = reasons
                rescue_action = str(rescue_cfg.get("action", "THINK_192"))
                max_steps = int(rescue_cfg.get("max_steps", 1))
                for _ in range(max_steps):
                    feasible = env.feasible_actions(state)
                    if rescue_action not in feasible:
                        break
                    decision_branch = select_best_branch(state.branches, prefer_undone=False)
                    if str(rescue_cfg.get("strategy", "rescue_think")) == "standard_cot_reseed":
                        max_tokens = int(
                            rescue_cfg.get(
                                "max_new_tokens",
                                env.config.get("solver", {}).get("max_new_tokens_rescue", 384),
                            )
                        )
                        delta = env._apply_standard_cot_seed(
                            state,
                            max_tokens,
                            action=rescue_action,
                        )
                        state.step_idx += 1
                        state.action_history.append(rescue_action)
                    else:
                        state, delta = env.step(
                            state,
                            rescue_action,
                            target_branch_id=decision_branch.branch_id,
                            think_mode="rescue",
                        )
                    append_transition(delta, reason="selective_rescue")
                    if bool(rescue_cfg.get("verify_after_rescue", False)) and "VERIFY" in env.feasible_actions(state):
                        decision_branch = select_best_branch(state.branches, prefer_undone=False)
                        state, delta = env.step(
                            state,
                            "VERIFY",
                            target_branch_id=decision_branch.branch_id,
                        )
                        append_transition(delta, reason="post_rescue_verify")
                    decision_branch = select_best_branch(state.branches, prefer_undone=False)
                    if decision_branch.done and normalize_answer(decision_branch.current_answer) is not None:
                        break
            else:
                rescue_meta["reasons"] = []
            decision_branch = select_best_branch(state.branches, prefer_undone=False)

        if (
            early_stop_if_done
            and decision_branch.done
            and decision_branch.confidence >= early_stop_confidence
            and normalize_answer(decision_branch.current_answer) is not None
        ):
            state.action_history.append("STOP")
            break
        features = env.extract_features(state, decision_branch_id=decision_branch.branch_id)
        p0 = predict_correctness(proxy_model, features)

        feasible = env.feasible_actions(state)
        if state.step_idx < min_decisions_before_stop and "STOP" in feasible:
            feasible = [action for action in feasible if action != "STOP"]
        action = bandit.select(features, feasible, env.rng)
        if action == "STOP":
            state.action_history.append("STOP")
            break

        next_state, delta = env.step(state, action, target_branch_id=decision_branch.branch_id)
        next_decision_branch = select_best_branch(next_state.branches, prefer_undone=False)
        next_features = env.extract_features(
            next_state,
            decision_branch_id=next_decision_branch.branch_id,
        )
        p1 = predict_correctness(proxy_model, next_features)
        append_transition(delta)

        reward = (
            (p1 - p0)
            - float(env.config["cost"]["lambda_tok"]) * (delta.total_tokens / max(state.total_budget, 1))
            - float(env.config["cost"]["lambda_ver"]) * delta.verifier_calls
            - float(env.config["cost"]["lambda_lat"])
            * (delta.latency_ms / float(env.config["cost"]["lat_norm_ms"]))
        )
        bandit.update(features, action, reward)
        state = next_state
        if env.budget_exhausted(state):
            break

    prediction, metadata = env.final_aggregate(state)
    record = _record_from_state(state, prediction, metadata, stop_reason="STOP_OR_BUDGET")
    record.metadata["transition_log"] = transition_log
    return record, state
