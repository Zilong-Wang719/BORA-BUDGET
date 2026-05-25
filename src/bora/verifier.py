from __future__ import annotations

import re
from typing import Any

import numpy as np

from bora.answer_extraction import (
    clean_candidate,
    extract_explicit_answer,
    extract_numeric_answer,
    extract_tagged_sections,
    parse_score,
)
from bora.common import clamp, normalize_answer
from bora.llm import get_llm_backend, resolve_backend_config
from bora.solver import infer_reference_answer
from bora.types import Branch, VerifierResult


class SimpleMathVerifier:
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.backend = str(config.get("verifier", {}).get("backend", "mock"))
        self.mode = str(config.get("mode", "eval"))
        self.tokens_per_call = int(config.get("verifier", {}).get("tokens_per_call", 64))
        self.allow_gold_fallback = bool(
            config.get("verifier", {}).get("allow_gold_fallback", False)
        )
        if self.mode == "eval" and self.allow_gold_fallback:
            raise ValueError("allow_gold_fallback must be false in eval mode")

    def score(
        self,
        problem: dict[str, Any],
        branch: Branch,
        rng: np.random.Generator,
    ) -> VerifierResult:
        target = infer_reference_answer(
            problem["question"],
            problem.get("answer"),
            allow_gold_fallback=self.allow_gold_fallback,
        )
        answer = normalize_answer(branch.current_answer)
        stability_bonus = 0.05 * min(len(branch.prm_scores), 2)
        depth_bonus = 0.04 * min(branch.depth, 3)

        if answer is None:
            score = 0.12 + 0.20 * branch.confidence
            explanation = "No answer was available; verifier stays cautious."
        elif target is not None and answer == target:
            score = 0.72 + 0.18 * branch.confidence + stability_bonus + depth_bonus
            explanation = "Current answer matches the arithmetic reference."
        else:
            score = 0.18 + 0.28 * branch.confidence + depth_bonus - 0.05 * branch.depth
            explanation = "Current answer disagrees with the arithmetic reference."

        score += float(rng.uniform(-0.03, 0.03))
        score = clamp(score, 0.01, 0.99)
        latency_ms = 18 + self.tokens_per_call * 2 + int(rng.integers(0, 12))
        return VerifierResult(
            score=score,
            token_cost=self.tokens_per_call,
            latency_ms=latency_ms,
            explanation=explanation,
            candidate_answer=target,
            score_parse_success=True,
        )


VERIFIER_SYSTEM_PROMPT = (
    "You are a strict math verifier. Independently solve the problem from the question, "
    "then compare your independently computed answer with the current answer. "
    "Do not use hidden gold answers. If the current answer disagrees with your independent "
    "calculation, give a low score even when the trace sounds plausible."
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


def _parse_verifier_score(sections: dict[str, str], raw_text: str) -> float | None:
    score = parse_score(sections.get("SCORE"))
    if score is not None:
        return score

    # Be deliberately stricter than parse_score() on free-form verifier text.
    # Otherwise problem numbers in an unformatted chain-of-thought can be
    # mistaken for a confidence score and then clamped to 0.99.
    patterns = [
        r"\b(?:score|rating|confidence)\s*[:=]\s*([01](?:\.\d+)?)\b",
        r"\b([01](?:\.\d+)?)\s*/\s*1(?:\.0+)?\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, raw_text or "", flags=re.IGNORECASE)
        if match:
            return parse_score(match.group(1))
    return None


def _extract_verifier_candidate_answer(sections: dict[str, str], raw_text: str) -> str | None:
    candidate_answer = extract_explicit_answer(
        sections.get("REFERENCE_ANSWER", ""),
        prefer_numeric=True,
    )
    if candidate_answer is None:
        candidate_answer = extract_numeric_answer(sections.get("REFERENCE_ANSWER", ""))
    if candidate_answer is not None:
        return candidate_answer

    # Some Qwen verifier responses ignore the requested tags but still state an
    # independent answer in prose, e.g. "the answer should be 10:00 am".
    fallback_patterns = [
        r"\b(?:the\s+)?(?:correct\s+)?(?:final\s+)?answer\s+(?:should\s+be|is)\s+([^.\n;]+)",
        r"\bcorrect\s+(?:latest\s+start\s+time|answer|result)\s+is\s+([^.\n;]+)",
        r"\bshould\s+be\s+([^.\n;]+)",
    ]
    for pattern in fallback_patterns:
        match = re.search(pattern, raw_text or "", flags=re.IGNORECASE)
        if not match:
            continue
        candidate = extract_explicit_answer(match.group(1), prefer_numeric=True)
        if candidate is None:
            candidate = extract_numeric_answer(match.group(1))
        if candidate is not None:
            return candidate
    return None


class TransformersVerifier:
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        verifier_cfg = {**config.get("llm", {}), **config.get("verifier", {})}
        self.backend = str(verifier_cfg.get("backend", "hf_transformers"))
        self.backend_client = get_llm_backend(config, "verifier")
        self.tokens_per_call = int(verifier_cfg.get("tokens_per_call", 96))
        self.temperature = float(verifier_cfg.get("temperature", 0.0))
        self.top_p = float(verifier_cfg.get("top_p", 1.0))
        self.repetition_penalty = float(verifier_cfg.get("repetition_penalty", 1.0))
        self.trace_window_chars = _trace_window_chars(config, "verifier")
        self.budget_token_scope = _budget_token_scope(config, "verifier")

    def _build_user_prompt(self, problem: dict[str, Any], branch: Branch) -> str:
        trace = _trim_trace(branch.trace, self.trace_window_chars)
        answer = branch.current_answer or "unknown"
        return (
            "Output exactly in this format:\n\n"
            "[REFERENCE_ANSWER]\n"
            "your independently computed final numeric answer\n\n"
            "[SCORE]\n"
            "0.0 to 1.0\n\n"
            "[EXPLANATION]\n"
            "one concise sentence\n\n"
            f"Question:\n{problem['question']}\n\n"
            f"Reasoning trace:\n{trace}\n\n"
            f"Current answer:\n{answer}\n"
        )

    def score(
        self,
        problem: dict[str, Any],
        branch: Branch,
        rng: np.random.Generator,
    ) -> VerifierResult:
        del rng
        prompt = self.backend_client.render_prompt(
            system_prompt=VERIFIER_SYSTEM_PROMPT,
            user_prompt=self._build_user_prompt(problem, branch),
        )
        generation = self.backend_client.generate_text(
            prompt=prompt,
            max_new_tokens=self.tokens_per_call,
            temperature=self.temperature,
            top_p=self.top_p,
            repetition_penalty=self.repetition_penalty,
        )
        sections = extract_tagged_sections(generation.text)
        score = _parse_verifier_score(sections, generation.text)
        score_parse_success = score is not None
        if score is None:
            score = 0.50 if branch.current_answer else 0.25
        candidate_answer = _extract_verifier_candidate_answer(sections, generation.text)
        explanation = clean_candidate(sections.get("EXPLANATION", "")) or generation.text.strip()
        token_cost = (
            generation.total_tokens
            if self.budget_token_scope == "total"
            else generation.completion_tokens
        )
        token_cost = max(int(token_cost), 1)
        return VerifierResult(
            score=clamp(float(score), 0.01, 0.99),
            token_cost=token_cost,
            latency_ms=generation.latency_ms,
            explanation=explanation,
            candidate_answer=candidate_answer,
            score_parse_success=score_parse_success,
        )


def build_verifier(config: dict[str, Any]) -> SimpleMathVerifier | TransformersVerifier:
    backend = str(config.get("verifier", {}).get("backend", "mock"))
    if backend == "mock":
        return SimpleMathVerifier(config)
    if backend in {"hf_transformers", "transformers", "vllm"}:
        return TransformersVerifier(config)
    raise ValueError(f"Unsupported verifier backend: {backend}")
