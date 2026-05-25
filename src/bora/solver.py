from __future__ import annotations

import ast
import operator
from typing import Any

import numpy as np

from bora.answer_extraction import (
    clean_candidate,
    extract_explicit_answer,
    extract_last_number,
    extract_numeric_answer,
    extract_tagged_sections,
    parse_bool,
    parse_score,
)
from bora.common import clamp, normalize_answer
from bora.llm import get_llm_backend, resolve_backend_config
from bora.types import Branch, SolverOutput


CONTINUE_PROMPT = """You are solving a math problem step by step.
Continue from the existing trace by at most one concise reasoning block.
Do not restart from scratch.
Output exactly in this format:

[STEP]
...

[CURRENT_ANSWER]
...

[CONFIDENCE]
...

[DONE]
...
"""


BRANCH_PROMPT = """You are exploring an alternative path.
Given the problem and the partial reasoning prefix, continue with a different next step
than the obvious continuation if possible.
Output exactly in this format:

[STEP]
...

[CURRENT_ANSWER]
...

[CONFIDENCE]
...

[DONE]
...
"""

RESCUE_PROMPT = """You are auditing a completed math solution that may contain a mistake.
Do not assume the current answer is correct.
Check the reasoning, recompute the key arithmetic independently, and correct the answer if needed.
Output exactly in this format:

[STEP]
...

[CURRENT_ANSWER]
...

[CONFIDENCE]
...

[DONE]
true
"""


SYSTEM_PROMPT = (
    "You are a careful math reasoner. Follow the requested format exactly and do not "
    "add extra sections, examples, or commentary."
)

STANDARD_COT_SEED_PROMPT = (
    "Please reason step by step, and put your final numeric answer within "
    "\\boxed{{}}.\n\n"
    "Problem:\n{question}\n\n"
)


SAFE_OPERATORS: dict[type[ast.AST], Any] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _eval_ast(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _eval_ast(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.Num):
        return float(node.n)
    if isinstance(node, ast.BinOp) and type(node.op) in SAFE_OPERATORS:
        left = _eval_ast(node.left)
        right = _eval_ast(node.right)
        return float(SAFE_OPERATORS[type(node.op)](left, right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in SAFE_OPERATORS:
        operand = _eval_ast(node.operand)
        return float(SAFE_OPERATORS[type(node.op)](operand))
    raise ValueError("Unsupported arithmetic expression.")


def extract_expression(question: str) -> str | None:
    normalized = question.strip()
    if not normalized:
        return None
    if normalized.endswith("?"):
        normalized = normalized[:-1]
    prefixes = ["What is ", "Compute ", "Evaluate "]
    for prefix in prefixes:
        if normalized.startswith(prefix):
            normalized = normalized[len(prefix) :]
            break
    filtered = "".join(ch for ch in normalized if ch in "0123456789+-*/(). ")
    filtered = " ".join(filtered.split())
    return filtered or None


def infer_reference_answer(
    question: str,
    gold_answer: str | None = None,
    allow_gold_fallback: bool = False,
) -> str | None:
    expression = extract_expression(question)
    if expression:
        try:
            parsed = ast.parse(expression, mode="eval")
            value = _eval_ast(parsed)
            if abs(value - round(value)) < 1e-9:
                return str(int(round(value)))
            return normalize_answer(str(value))
        except Exception:
            pass
    if allow_gold_fallback:
        return normalize_answer(gold_answer)
    return None


def _format_number(value: float) -> str:
    if abs(value - round(value)) < 1e-9:
        return str(int(round(value)))
    return f"{value:.6f}".rstrip("0").rstrip(".")


def _perturb_answer(answer: str | None, rng: np.random.Generator) -> str | None:
    if answer is None:
        return None
    normalized = normalize_answer(answer)
    if normalized is None:
        return answer
    try:
        value = float(normalized)
    except (TypeError, ValueError):
        suffix = int(rng.integers(1, 4))
        return f"{normalized}_{suffix}"
    candidates = [value + 1.0, value - 1.0]
    if value != 0:
        candidates.extend([value * 2.0, value / 2.0])
    proposal = float(rng.choice(candidates))
    return _format_number(proposal)


class MockMathSolver:
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.backend = str(config.get("solver", {}).get("backend", "mock"))
        self.mode = str(config.get("mode", "eval"))
        self.allow_gold_fallback = bool(
            config.get("solver", {}).get("allow_gold_fallback", False)
        )
        solver_cfg = config.get("solver", {})
        self.temperature_think = float(solver_cfg.get("temperature_think", 0.7))
        self.temperature_branch = float(solver_cfg.get("temperature_branch", 1.0))
        if self.mode == "eval" and self.allow_gold_fallback:
            raise ValueError("allow_gold_fallback must be false in eval mode")

    def generate(
        self,
        problem: dict[str, Any],
        branch: Branch,
        mode: str,
        max_new_tokens: int,
        rng: np.random.Generator,
        enable_thinking: bool | None = None,
    ) -> SolverOutput:
        del enable_thinking
        target = infer_reference_answer(
            problem["question"],
            problem.get("answer"),
            allow_gold_fallback=self.allow_gold_fallback,
        )
        temperature = self.temperature_branch if mode == "branch" else self.temperature_think
        depth = branch.depth
        base_success = 0.38 + 0.18 * depth + 0.08 * len(branch.prm_scores)
        if mode == "branch":
            base_success += 0.12
        if mode == "rescue":
            base_success += 0.20
        base_success -= 0.10 * max(temperature - 0.7, 0.0)
        if branch.current_answer and target and normalize_answer(branch.current_answer) == target:
            base_success += 0.15
        success_prob = clamp(base_success, 0.15, 0.93)
        proposal = target if rng.random() < success_prob else _perturb_answer(target, rng)

        done_bias = 0.20 if max_new_tokens <= 64 else 0.45
        done = depth >= 1 and (proposal == target or rng.random() < done_bias)

        confidence = 0.25 + 0.18 * depth + 0.18 * float(proposal == target)
        if mode == "branch":
            confidence += 0.08
        confidence -= 0.06 * max(temperature - 0.7, 0.0)
        confidence += float(rng.uniform(-0.06, 0.06))
        confidence = clamp(confidence, 0.05, 0.99)

        expression = extract_expression(problem["question"]) or problem["question"]
        if depth == 0:
            step_text = f"Parse the expression {expression} and isolate the main computation."
        elif mode == "rescue" and proposal == target:
            step_text = f"Audit the prior reasoning, recompute the result, and correct the answer to {proposal}."
        elif mode == "rescue":
            step_text = f"Audit the prior reasoning and keep the best rescued candidate answer {proposal}."
        elif proposal == target:
            step_text = f"Compute the arithmetic carefully and keep the candidate answer at {proposal}."
        elif mode == "branch":
            step_text = f"Try an alternative computation path and test the competing answer {proposal}."
        else:
            step_text = f"Continue the current branch and compare the running answer against {proposal}."

        token_cost = min(max_new_tokens, max(18, len(step_text.split()) * 4 + 8))
        latency_ms = 20 + token_cost * 3 + int(rng.integers(0, 18))
        return SolverOutput(
            step_text=step_text,
            current_answer=proposal,
            confidence=confidence,
            done=done,
            token_cost=token_cost,
            latency_ms=latency_ms,
        )

    def generate_standard_cot_seed(
        self,
        problem: dict[str, Any],
        max_new_tokens: int,
        rng: np.random.Generator,
        enable_thinking: bool | None = None,
    ) -> SolverOutput:
        del enable_thinking
        target = infer_reference_answer(
            problem["question"],
            problem.get("answer"),
            allow_gold_fallback=self.allow_gold_fallback,
        )
        if target is None:
            target = _perturb_answer(problem.get("answer"), rng)
        step_text = (
            "Solve the problem end to end with a complete chain of thought and report "
            f"the final candidate as \\boxed{{{target}}}."
        )
        token_cost = min(max_new_tokens, max(32, len(step_text.split()) * 4))
        return SolverOutput(
            step_text=step_text,
            current_answer=target,
            confidence=0.92 if target is not None else 0.20,
            done=target is not None,
            token_cost=token_cost,
            latency_ms=20 + token_cost * 3,
        )


def _budget_token_scope(config: dict[str, Any], section: str) -> str:
    shared = dict(config.get("llm", {}))
    scoped = dict(config.get(section, {}))
    return str({**shared, **scoped}.get("budget_token_scope", "completion"))


def _trace_window_chars(config: dict[str, Any], section: str) -> int:
    backend_cfg = resolve_backend_config(config, section)
    return int(backend_cfg.get("trace_window_chars", 4000))


def _trim_trace(trace: str, limit_chars: int) -> str:
    text = trace.strip()
    if not text:
        return "(empty)"
    if len(text) <= limit_chars:
        return text
    return "[... truncated earlier reasoning ...]\n" + text[-limit_chars:]


def _standard_cot_prompt(question: str, enable_thinking: bool | None) -> str:
    suffix = "/think" if enable_thinking is True else "/no_think"
    return f"{STANDARD_COT_SEED_PROMPT.format(question=question).rstrip()}\n\n{suffix}"


class TransformersMathSolver:
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        solver_cfg = {**config.get("llm", {}), **config.get("solver", {})}
        self.backend = str(solver_cfg.get("backend", "hf_transformers"))
        self.backend_client = get_llm_backend(config, "solver")
        self.temperature_think = float(solver_cfg.get("temperature_think", 0.3))
        self.temperature_branch = float(solver_cfg.get("temperature_branch", 0.7))
        self.top_p = float(solver_cfg.get("top_p", 0.9))
        self.repetition_penalty = float(solver_cfg.get("repetition_penalty", 1.0))
        self.standard_cot_temperature = float(solver_cfg.get("standard_cot_temperature", 0.7))
        self.standard_cot_top_p = float(solver_cfg.get("standard_cot_top_p", 0.8))
        self.standard_cot_confidence = float(solver_cfg.get("standard_cot_confidence", 0.92))
        self.enable_thinking = solver_cfg.get("enable_thinking")
        self.standard_cot_enable_thinking = solver_cfg.get(
            "standard_cot_enable_thinking",
            self.enable_thinking,
        )
        self.trace_window_chars = _trace_window_chars(config, "solver")
        self.budget_token_scope = _budget_token_scope(config, "solver")

    def _build_user_prompt(self, problem: dict[str, Any], branch: Branch, mode: str) -> str:
        trace = _trim_trace(branch.trace, self.trace_window_chars)
        if mode == "branch":
            prompt_template = BRANCH_PROMPT
            direction = "Explore a materially different next step from the current path if possible."
        elif mode == "rescue":
            prompt_template = RESCUE_PROMPT
            direction = "Audit and correct the existing solution; provide the final numeric answer."
        else:
            prompt_template = CONTINUE_PROMPT
            direction = "Continue the existing reasoning without restarting the solution."
        return (
            f"{prompt_template.strip()}\n\n"
            f"Question:\n{problem['question']}\n\n"
            f"Existing trace:\n{trace}\n\n"
            f"Additional instruction:\n{direction}\n"
        )

    def _parse_output(self, raw_text: str, branch: Branch) -> tuple[str, str | None, float, bool]:
        sections = extract_tagged_sections(raw_text)
        step_text = clean_candidate(sections.get("STEP", "")) or raw_text.strip()
        answer_block = sections.get("CURRENT_ANSWER")
        current_answer = clean_candidate(answer_block) if answer_block else None
        if not current_answer:
            current_answer = extract_explicit_answer(raw_text, prefer_numeric=False)
        if not current_answer:
            current_answer = extract_numeric_answer(raw_text)
        if not current_answer:
            current_answer = branch.current_answer

        confidence = parse_score(sections.get("CONFIDENCE"))
        if confidence is None:
            confidence = branch.confidence if branch.conf_history else 0.5
        confidence = clamp(float(confidence), 0.01, 0.99)

        done = parse_bool(sections.get("DONE"))
        if done is None:
            done = bool(current_answer) and confidence >= 0.78
        if not step_text:
            step_text = "(no new reasoning produced)"
        return step_text, current_answer, confidence, bool(done)

    def generate(
        self,
        problem: dict[str, Any],
        branch: Branch,
        mode: str,
        max_new_tokens: int,
        rng: np.random.Generator,
        enable_thinking: bool | None = None,
    ) -> SolverOutput:
        del rng
        prompt = self.backend_client.render_prompt(
            system_prompt=SYSTEM_PROMPT,
            user_prompt=self._build_user_prompt(problem, branch, mode),
            enable_thinking=self.enable_thinking if enable_thinking is None else enable_thinking,
        )
        temperature = self.temperature_branch if mode == "branch" else self.temperature_think
        generation = self.backend_client.generate_text(
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=self.top_p,
            repetition_penalty=self.repetition_penalty,
        )
        step_text, current_answer, confidence, done = self._parse_output(
            generation.text,
            branch,
        )
        token_cost = (
            generation.total_tokens
            if self.budget_token_scope == "total"
            else generation.completion_tokens
        )
        token_cost = max(int(token_cost), 1)
        return SolverOutput(
            step_text=step_text,
            current_answer=current_answer,
            confidence=confidence,
            done=done,
            token_cost=token_cost,
            latency_ms=generation.latency_ms,
        )

    def generate_standard_cot_seed(
        self,
        problem: dict[str, Any],
        max_new_tokens: int,
        rng: np.random.Generator,
        enable_thinking: bool | None = None,
    ) -> SolverOutput:
        del rng
        if enable_thinking is None:
            enable_thinking = self.standard_cot_enable_thinking
        prompt = self.backend_client.render_prompt(
            system_prompt="You are a careful math reasoner.",
            user_prompt=_standard_cot_prompt(problem["question"], enable_thinking),
            enable_thinking=enable_thinking,
        )
        generation = self.backend_client.generate_text(
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=self.standard_cot_temperature,
            top_p=self.standard_cot_top_p,
            repetition_penalty=self.repetition_penalty,
        )
        explicit_answer = extract_explicit_answer(generation.text, prefer_numeric=True)
        current_answer = explicit_answer or extract_numeric_answer(generation.text)
        confidence = self.standard_cot_confidence if explicit_answer else 0.72
        if current_answer is None:
            confidence = 0.20
        token_cost = (
            generation.total_tokens
            if self.budget_token_scope == "total"
            else generation.completion_tokens
        )
        return SolverOutput(
            step_text=generation.text.strip() or "(no reasoning produced)",
            current_answer=current_answer,
            confidence=clamp(confidence, 0.01, 0.99),
            done=current_answer is not None,
            token_cost=max(int(token_cost), 1),
            latency_ms=generation.latency_ms,
        )


def build_solver(config: dict[str, Any]) -> MockMathSolver | TransformersMathSolver:
    backend = str(config.get("solver", {}).get("backend", "mock"))
    if backend in {"mock", "mock_train_simulator"}:
        return MockMathSolver(config)
    if backend in {"hf_transformers", "transformers", "vllm"}:
        return TransformersMathSolver(config)
    raise ValueError(f"Unsupported solver backend: {backend}")
