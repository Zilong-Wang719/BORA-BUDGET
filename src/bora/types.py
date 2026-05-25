from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Checkpoint:
    step_index: int
    step_text: str
    current_answer: str | None
    confidence: float
    done: bool
    action: str
    token_cost: int


@dataclass
class SolverOutput:
    step_text: str
    current_answer: str | None
    confidence: float
    done: bool
    token_cost: int
    latency_ms: int


@dataclass
class VerifierResult:
    score: float
    token_cost: int
    latency_ms: int
    explanation: str
    candidate_answer: str | None = None
    score_parse_success: bool = True


@dataclass
class Branch:
    branch_id: int
    parent_id: int | None
    trace: str = ""
    current_answer: str | None = None
    confidence: float = 0.0
    done: bool = False
    solver_tokens: int = 0
    verifier_tokens: int = 0
    prm_scores: list[float] = field(default_factory=list)
    prm_mean: float = 0.0
    prm_min: float = 0.0
    answer_history: list[str] = field(default_factory=list)
    conf_history: list[float] = field(default_factory=list)
    score: float = 0.0
    checkpoints: list[Checkpoint] = field(default_factory=list)
    action_history: list[str] = field(default_factory=list)

    @property
    def depth(self) -> int:
        return len(self.checkpoints)


@dataclass
class State:
    qid: str
    question: str
    gold_answer: str | None
    branches: list[Branch]
    spent_solver_tokens: int
    spent_verifier_tokens: int
    spent_latency_ms: int
    step_idx: int
    total_budget: int
    metadata: dict[str, Any] = field(default_factory=dict)
    action_history: list[str] = field(default_factory=list)

    @property
    def total_tokens(self) -> int:
        return self.spent_solver_tokens + self.spent_verifier_tokens


@dataclass
class TransitionDelta:
    action: str
    solver_tokens: int = 0
    verifier_tokens: int = 0
    latency_ms: int = 0
    verifier_calls: int = 0
    steps_added: int = 0
    decision_branch_id: int | None = None
    executed_branch_id: int | None = None
    source_branch_id: int | None = None
    new_branch_id: int | None = None
    remaining_budget_before: int = 0
    remaining_budget_after: int = 0
    verified_branch_ids: list[int] = field(default_factory=list)

    @property
    def total_tokens(self) -> int:
        return self.solver_tokens + self.verifier_tokens


@dataclass
class EpisodeRecord:
    qid: str
    prediction: str | None
    gold_answer: str | None
    correct: bool
    total_tokens: int
    solver_tokens: int
    verifier_tokens: int
    latency_ms: int
    branches_used: int
    stop_reason: str
    actions: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)
