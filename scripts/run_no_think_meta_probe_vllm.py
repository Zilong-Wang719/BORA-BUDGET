from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from time import perf_counter
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bora.common import dump_json, load_config, load_problem_split
from bora.llm import _ensure_tokenizer_padding, _render_prompt_with_tokenizer, detect_prompt_template


SCALAR_META_KEYS = {
    "confidence": "meta_confidence",
    "difficulty": "meta_difficulty",
    "needs_thinking": "meta_needs_thinking",
    "calculation_risk": "meta_calculation_risk",
    "ambiguity_risk": "meta_ambiguity_risk",
    "answer_stability": "meta_answer_stability",
}

DIAGNOSTIC_SCORE_KEYS = {
    "correctness_probability": "meta_diag_correctness_probability",
    "seed_error_probability": "meta_diag_seed_error_probability",
    "thinking_value": "meta_diag_thinking_value",
    "answer_support": "meta_diag_answer_support",
    "missing_step_risk": "meta_diag_missing_step_risk",
    "arithmetic_risk": "meta_diag_arithmetic_risk",
    "algebra_risk": "meta_diag_algebra_risk",
    "casework_risk": "meta_diag_casework_risk",
    "geometry_diagram_risk": "meta_diag_geometry_diagram_risk",
    "conceptual_risk": "meta_diag_conceptual_risk",
    "expression_parse_risk": "meta_diag_expression_parse_risk",
    "answer_format_risk": "meta_diag_answer_format_risk",
    "incomplete_reasoning_risk": "meta_diag_incomplete_reasoning_risk",
    "verification_need": "meta_diag_verification_need",
    "counterexample_risk": "meta_diag_counterexample_risk",
}

DISCRIMINATIVE_SCORE_KEYS = {
    "seed_support": "meta_disc_seed_support",
    "observable_error_evidence": "meta_disc_observable_error_evidence",
    "latent_error_risk": "meta_disc_latent_error_risk",
    "thinking_fix_probability": "meta_disc_thinking_fix_probability",
    "thinking_harm_probability": "meta_disc_thinking_harm_probability",
    "net_thinking_utility": "meta_disc_net_thinking_utility",
    "budget_2k_value": "meta_disc_budget_2k_value",
    "budget_4k_value": "meta_disc_budget_4k_value",
    "budget_6k_value": "meta_disc_budget_6k_value",
    "budget_8k_value": "meta_disc_budget_8k_value",
    "budget_10k_value": "meta_disc_budget_10k_value",
    "budget_12k_value": "meta_disc_budget_12k_value",
    "adoption_safety": "meta_disc_adoption_safety",
    "answer_format_sensitivity": "meta_disc_answer_format_sensitivity",
}

VERDICTS = ["accept", "think_light", "think_full"]
FAILURE_MODES = [
    "none",
    "arithmetic",
    "algebra",
    "casework",
    "diagram",
    "conceptual",
    "parsing",
    "format",
    "incomplete",
    "guessing",
]
BUDGET_LABELS = ["none", "2k", "4k", "6k", "8k", "10k", "12k"]
HARD_SIGNALS = [
    "geometry_diagram",
    "multi_case",
    "algebraic_transform",
    "long_computation",
    "exact_simplification",
    "proof_like",
    "answer_format_sensitive",
]
SEED_VALIDITY = [
    "well_supported",
    "plausible_gap",
    "unsupported",
    "contradictory",
    "unparseable",
]
OBSERVABLE_FLAWS = [
    "none",
    "arithmetic_inconsistency",
    "unsupported_leap",
    "missing_cases",
    "wrong_interpretation",
    "final_answer_mismatch",
    "diagram_dependency",
    "format_or_parse_issue",
    "incomplete_trace",
]
PROBE_DECISIONS = ["accept", "verify_only", "think_8k", "think_12k"]


def _chunks(items: list[Any], size: int) -> list[list[Any]]:
    return [items[idx : idx + size] for idx in range(0, len(items), size)]


def _load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def _records(payload: dict[str, Any], method: str | None = None) -> list[dict[str, Any]]:
    if isinstance(payload.get("records"), list):
        return payload["records"]
    if isinstance(payload.get("rows"), list):
        return payload["rows"]
    if method is not None:
        block = payload.get(method)
        if isinstance(block, dict) and isinstance(block.get("records"), list):
            return block["records"]
        raise KeyError(f"Method {method!r} not found in payload.")
    candidates = [
        value.get("records")
        for value in payload.values()
        if isinstance(value, dict) and isinstance(value.get("records"), list)
    ]
    if len(candidates) != 1:
        raise ValueError("Could not infer records block; pass --seed-method.")
    return candidates[0]


def _completion_head(record: dict[str, Any], max_chars: int) -> str:
    metadata = record.get("metadata") or {}
    text = str(metadata.get("completion_head") or record.get("completion_head") or "")
    return text[:max_chars]


def _build_scalar_meta_prompt(problem: dict[str, Any], seed: dict[str, Any], *, include_trace_chars: int) -> str:
    seed_answer = seed.get("prediction")
    seed_trace = _completion_head(seed, include_trace_chars)
    trace_block = f"\nPrevious no-thinking reasoning excerpt:\n{seed_trace}\n" if seed_trace else ""
    return f"""You are evaluating a previous no-thinking math solution. Do not solve the problem from scratch.

Problem:
{problem.get("question")}

Previous no-thinking final answer:
{seed_answer}
{trace_block}
Return ONLY compact JSON with numeric scores from 0 to 100:
{{
  "confidence": <probability the previous answer is correct>,
  "difficulty": <problem difficulty>,
  "needs_thinking": <how much this problem would benefit from thinking mode>,
  "calculation_risk": <risk of arithmetic or algebra error>,
  "ambiguity_risk": <risk from ambiguous parsing or missing diagram/context>,
  "answer_stability": <how stable the previous answer seems under re-checking>
}}
"""


def _build_diagnostic_meta_prompt(problem: dict[str, Any], seed: dict[str, Any], *, include_trace_chars: int) -> str:
    seed_answer = seed.get("prediction")
    seed_trace = _completion_head(seed, include_trace_chars)
    trace_block = f"\nPrevious no-thinking reasoning excerpt:\n{seed_trace}\n" if seed_trace else ""
    return f"""You are a strict diagnostic controller for math reasoning. Your job is NOT to solve the problem fully.
Inspect the previous /no_think answer and decide what kind of extra computation would be useful.
Be critical: a short fluent solution can still be wrong. Prefer concrete failure modes over vague confidence.

Problem:
{problem.get("question")}

Previous /no_think final answer:
{seed_answer}
{trace_block}
Return ONLY compact JSON. Scores must be integers from 0 to 100.
Allowed categorical values:
- verdict: "accept", "think_light", or "think_full"
- primary_failure_mode: one of {FAILURE_MODES}
- expected_budget: one of {BUDGET_LABELS}
- hard_signals: list drawn from {HARD_SIGNALS}

{{
  "verdict": <category>,
  "primary_failure_mode": <category>,
  "expected_budget": <category>,
  "hard_signals": [<categories>],
  "correctness_probability": <0-100>,
  "seed_error_probability": <0-100>,
  "thinking_value": <0-100>,
  "answer_support": <0-100, how well the visible reasoning supports the final answer>,
  "missing_step_risk": <0-100>,
  "arithmetic_risk": <0-100>,
  "algebra_risk": <0-100>,
  "casework_risk": <0-100>,
  "geometry_diagram_risk": <0-100>,
  "conceptual_risk": <0-100>,
  "expression_parse_risk": <0-100>,
  "answer_format_risk": <0-100>,
  "incomplete_reasoning_risk": <0-100>,
  "verification_need": <0-100>,
  "counterexample_risk": <0-100>
}}
"""


def _build_discriminative_meta_prompt(problem: dict[str, Any], seed: dict[str, Any], *, include_trace_chars: int) -> str:
    seed_answer = seed.get("prediction")
    seed_trace = _completion_head(seed, include_trace_chars)
    trace_block = f"\nPrevious no-thinking reasoning excerpt:\n{seed_trace}\n" if seed_trace else ""
    return f"""You are a discriminative meta-probe for budgeted math reasoning. Do NOT solve the problem from scratch.
Your task is to decide whether a later controller should preserve the /no_think seed, verify it, or pay for thinking.

Important calibration rules:
- Do not mark an answer as high-risk only because the problem looks hard.
- Prefer ACCEPT when the visible reasoning directly supports the final answer and no concrete flaw is visible.
- Assign high observable_error_evidence only when you can point to a concrete issue in the seed trace.
- Assign high thinking_fix_probability only when extra thinking is likely to change an incorrect seed into a correct answer.
- Assign high thinking_harm_probability when overriding a plausible seed may introduce a wrong answer.
- Scores should be discriminative: avoid setting every risk high.

Problem:
{problem.get("question")}

Previous /no_think final answer:
{seed_answer}
{trace_block}
Return ONLY compact JSON. Scores must be integers from 0 to 100.
Allowed categorical values:
- seed_validity: one of {SEED_VALIDITY}
- observable_flaw: one of {OBSERVABLE_FLAWS}
- decision: one of {PROBE_DECISIONS}
- expected_budget: one of {BUDGET_LABELS}
- hard_signals: list drawn from {HARD_SIGNALS}

{{
  "seed_validity": <category>,
  "observable_flaw": <category>,
  "decision": <category>,
  "expected_budget": <category>,
  "hard_signals": [<categories>],
  "seed_support": <0-100, direct visible support for the seed answer>,
  "observable_error_evidence": <0-100, concrete evidence the seed is wrong>,
  "latent_error_risk": <0-100, risk not directly visible but plausible>,
  "thinking_fix_probability": <0-100, chance thinking would fix a wrong seed>,
  "thinking_harm_probability": <0-100, chance thinking would flip a correct seed wrong>,
  "net_thinking_utility": <0-100, expected value of invoking thinking after cost and harm>,
  "budget_2k_value": <0-100>,
  "budget_4k_value": <0-100>,
  "budget_6k_value": <0-100>,
  "budget_8k_value": <0-100>,
  "budget_10k_value": <0-100>,
  "budget_12k_value": <0-100>,
  "adoption_safety": <0-100, safety of adopting a thinking answer over the seed>,
  "answer_format_sensitivity": <0-100>
}}
"""


def _build_meta_prompt(
    problem: dict[str, Any],
    seed: dict[str, Any],
    *,
    include_trace_chars: int,
    probe_style: str,
) -> str:
    if probe_style == "discriminative":
        return _build_discriminative_meta_prompt(problem, seed, include_trace_chars=include_trace_chars)
    if probe_style == "diagnostic":
        return _build_diagnostic_meta_prompt(problem, seed, include_trace_chars=include_trace_chars)
    return _build_scalar_meta_prompt(problem, seed, include_trace_chars=include_trace_chars)


def _extract_json(text: str) -> dict[str, Any]:
    stripped = text.strip()
    candidates = [stripped]
    match = re.search(r"\{.*\}", stripped, flags=re.DOTALL)
    if match:
        candidates.insert(0, match.group(0))
    for candidate in candidates:
        try:
            payload = json.loads(candidate)
        except Exception:
            continue
        if isinstance(payload, dict):
            return payload
    return {}


def _coerce_score(value: Any) -> float | None:
    if isinstance(value, str):
        match = re.search(r"-?\d+(?:\.\d+)?", value)
        value = match.group(0) if match else value
    try:
        score = float(value)
    except (TypeError, ValueError):
        return None
    if score <= 1.0:
        score *= 100.0
    return max(0.0, min(100.0, score))


def _canonical_category(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text).strip("_")
    if text in {"full", "think", "think_12k", "12k"}:
        return "think_full"
    if text in {"light", "think_8k", "8k"}:
        return "think_light"
    if text in {"no", "none", "seed", "keep_seed"}:
        return "accept"
    return text


def _canonical_budget(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text).strip("_")
    if text in {"no", "none", "accept", "seed", "keep_seed"}:
        return "none"
    if text in {"2", "2k", "think_2k", "light_2k"}:
        return "2k"
    if text in {"4", "4k", "think_4k", "light_4k"}:
        return "4k"
    if text in {"6", "6k", "think_6k"}:
        return "6k"
    if text in {"8", "8k", "think_8k", "think_light", "light"}:
        return "8k"
    if text in {"10", "10k", "think_10k"}:
        return "10k"
    if text in {"12", "12k", "think_12k", "think_full", "full"}:
        return "12k"
    return text


def _list_values(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [_canonical_category(item) for item in value]
    if isinstance(value, str):
        chunks = re.split(r"[,;/|]+", value)
        return [_canonical_category(item) for item in chunks if item.strip()]
    return [_canonical_category(value)]


def _parse_scalar_meta(text: str) -> dict[str, float]:
    payload = _extract_json(text)
    features: dict[str, float] = {}
    for source_key, feature_key in SCALAR_META_KEYS.items():
        value = payload.get(source_key)
        if value is None:
            pattern = source_key.replace("_", r"[_\\s-]*")
            match = re.search(pattern + r"[^0-9-]*(-?\d+(?:\.\d+)?)", text, flags=re.IGNORECASE)
            value = match.group(1) if match else None
        score = _coerce_score(value)
        if score is not None:
            features[feature_key] = score / 100.0
    if "meta_confidence" in features and "meta_needs_thinking" in features:
        features["meta_confidence_minus_needs_thinking"] = (
            features["meta_confidence"] - features["meta_needs_thinking"]
        )
    if "meta_difficulty" in features and "meta_answer_stability" in features:
        features["meta_difficulty_minus_stability"] = (
            features["meta_difficulty"] - features["meta_answer_stability"]
        )
    return features


def _parse_diagnostic_meta(text: str) -> dict[str, float]:
    payload = _extract_json(text)
    features: dict[str, float] = {}
    for source_key, feature_key in DIAGNOSTIC_SCORE_KEYS.items():
        value = payload.get(source_key)
        if value is None:
            pattern = source_key.replace("_", r"[_\\s-]*")
            match = re.search(pattern + r"[^0-9-]*(-?\d+(?:\.\d+)?)", text, flags=re.IGNORECASE)
            value = match.group(1) if match else None
        score = _coerce_score(value)
        if score is not None:
            features[feature_key] = score / 100.0

    verdict = _canonical_category(payload.get("verdict"))
    for item in VERDICTS:
        features[f"meta_diag_verdict_{item}"] = 1.0 if verdict == item else 0.0

    failure_mode = _canonical_category(payload.get("primary_failure_mode"))
    for item in FAILURE_MODES:
        features[f"meta_diag_failure_{item}"] = 1.0 if failure_mode == item else 0.0

    budget = _canonical_budget(payload.get("expected_budget"))
    for item in BUDGET_LABELS:
        features[f"meta_diag_budget_{item}"] = 1.0 if budget == item else 0.0

    signals = set(_list_values(payload.get("hard_signals")))
    for item in HARD_SIGNALS:
        features[f"meta_diag_signal_{item}"] = 1.0 if item in signals else 0.0
    features["meta_diag_signal_count"] = float(sum(1 for item in HARD_SIGNALS if item in signals))

    if "meta_diag_seed_error_probability" in features and "meta_diag_thinking_value" in features:
        features["meta_diag_error_x_thinking_value"] = (
            features["meta_diag_seed_error_probability"] * features["meta_diag_thinking_value"]
        )
    if "meta_diag_correctness_probability" in features and "meta_diag_verification_need" in features:
        features["meta_diag_low_conf_x_verify_need"] = (
            (1.0 - features["meta_diag_correctness_probability"]) * features["meta_diag_verification_need"]
        )
    risk_keys = [
        "meta_diag_arithmetic_risk",
        "meta_diag_algebra_risk",
        "meta_diag_casework_risk",
        "meta_diag_geometry_diagram_risk",
        "meta_diag_conceptual_risk",
        "meta_diag_expression_parse_risk",
        "meta_diag_answer_format_risk",
        "meta_diag_incomplete_reasoning_risk",
    ]
    values = [features[key] for key in risk_keys if key in features]
    if values:
        features["meta_diag_max_structured_risk"] = max(values)
        features["meta_diag_mean_structured_risk"] = sum(values) / len(values)
    return features


def _parse_discriminative_meta(text: str) -> dict[str, float]:
    payload = _extract_json(text)
    features: dict[str, float] = {}
    for source_key, feature_key in DISCRIMINATIVE_SCORE_KEYS.items():
        value = payload.get(source_key)
        if value is None:
            pattern = source_key.replace("_", r"[_\\s-]*")
            match = re.search(pattern + r"[^0-9-]*(-?\d+(?:\.\d+)?)", text, flags=re.IGNORECASE)
            value = match.group(1) if match else None
        score = _coerce_score(value)
        if score is not None:
            features[feature_key] = score / 100.0

    seed_validity = _canonical_category(payload.get("seed_validity"))
    for item in SEED_VALIDITY:
        features[f"meta_disc_seed_validity_{item}"] = 1.0 if seed_validity == item else 0.0

    observable_flaw = _canonical_category(payload.get("observable_flaw"))
    for item in OBSERVABLE_FLAWS:
        features[f"meta_disc_flaw_{item}"] = 1.0 if observable_flaw == item else 0.0

    decision = _canonical_category(payload.get("decision"))
    # Preserve budget-aware decision categories that the generic canonicalizer
    # would otherwise collapse into think_light / think_full.
    decision = {
        "8k": "think_8k",
        "12k": "think_12k",
        "think_light": "think_8k",
        "think_full": "think_12k",
    }.get(decision, decision)
    for item in PROBE_DECISIONS:
        features[f"meta_disc_decision_{item}"] = 1.0 if decision == item else 0.0

    budget = _canonical_budget(payload.get("expected_budget"))
    budget_to_k = {"none": 0.0, "2k": 2.0, "4k": 4.0, "6k": 6.0, "8k": 8.0, "10k": 10.0, "12k": 12.0}
    features["meta_disc_expected_budget_k"] = budget_to_k.get(budget, 0.0)
    for item in BUDGET_LABELS:
        features[f"meta_disc_budget_{item}"] = 1.0 if budget == item else 0.0

    signals = set(_list_values(payload.get("hard_signals")))
    for item in HARD_SIGNALS:
        features[f"meta_disc_signal_{item}"] = 1.0 if item in signals else 0.0
    features["meta_disc_signal_count"] = float(sum(1 for item in HARD_SIGNALS if item in signals))

    if "meta_disc_thinking_fix_probability" in features and "meta_disc_thinking_harm_probability" in features:
        features["meta_disc_fix_minus_harm"] = (
            features["meta_disc_thinking_fix_probability"] - features["meta_disc_thinking_harm_probability"]
        )
    if "meta_disc_observable_error_evidence" in features and "meta_disc_thinking_fix_probability" in features:
        features["meta_disc_error_x_fix"] = (
            features["meta_disc_observable_error_evidence"] * features["meta_disc_thinking_fix_probability"]
        )
    if "meta_disc_seed_support" in features and "meta_disc_observable_error_evidence" in features:
        features["meta_disc_low_support_x_error"] = (
            (1.0 - features["meta_disc_seed_support"]) * features["meta_disc_observable_error_evidence"]
        )
    if "meta_disc_net_thinking_utility" in features and "meta_disc_thinking_harm_probability" in features:
        features["meta_disc_utility_minus_harm"] = (
            features["meta_disc_net_thinking_utility"] - features["meta_disc_thinking_harm_probability"]
        )

    budget_values = {
        2: features.get("meta_disc_budget_2k_value"),
        4: features.get("meta_disc_budget_4k_value"),
        6: features.get("meta_disc_budget_6k_value"),
        8: features.get("meta_disc_budget_8k_value"),
        10: features.get("meta_disc_budget_10k_value"),
        12: features.get("meta_disc_budget_12k_value"),
    }
    present_budget_values = {budget: value for budget, value in budget_values.items() if value is not None}
    if present_budget_values:
        best_budget, best_value = max(present_budget_values.items(), key=lambda item: item[1])
        features["meta_disc_best_budget_k"] = float(best_budget)
        features["meta_disc_best_budget_value"] = float(best_value)
        features["meta_disc_12k_minus_8k_value"] = (
            float(features.get("meta_disc_budget_12k_value", 0.0))
            - float(features.get("meta_disc_budget_8k_value", 0.0))
        )
        for budget_item in [2, 4, 6, 8, 10, 12]:
            features[f"meta_disc_best_budget_{budget_item}k"] = 1.0 if best_budget == budget_item else 0.0
    return features


def _parse_meta(text: str, *, probe_style: str) -> dict[str, float]:
    if probe_style == "discriminative":
        return _parse_discriminative_meta(text)
    if probe_style == "diagnostic":
        return _parse_diagnostic_meta(text)
    return _parse_scalar_meta(text)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", choices=["train", "dev", "test"], default="dev")
    parser.add_argument("--seed-result", type=Path, required=True)
    parser.add_argument("--seed-method", default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--partial-output", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--qid-file", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--max-num-seqs", type=int, default=8)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.82)
    parser.add_argument("--random-seed", type=int, default=None)
    parser.add_argument("--include-trace-chars", type=int, default=900)
    parser.add_argument("--probe-style", choices=["scalar", "diagnostic", "discriminative"], default="scalar")
    parser.add_argument("--progress-every", type=int, default=50)
    args = parser.parse_args()

    loaded_config = load_config(args.config)
    random_seed = int(args.random_seed if args.random_seed is not None else loaded_config.get("random_seed", 0))
    config = {**loaded_config, "mode": "eval", "random_seed": random_seed}
    llm_cfg = dict(config.get("llm", {}))
    solver_cfg = {**llm_cfg, **config.get("solver", {})}
    model_name = str(solver_cfg["model_name"])
    prompt_template = str(solver_cfg.get("prompt_template", "auto"))
    if prompt_template == "auto":
        prompt_template = detect_prompt_template(model_name)

    problems = load_problem_split(config, args.split)
    problem_by_qid = {str(problem["qid"]): problem for problem in problems}
    seed_records = {str(row["qid"]): row for row in _records(_load_json(args.seed_result), args.seed_method)}
    if args.qid_file is not None:
        qids = [line.strip() for line in args.qid_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    else:
        qids = [str(problem["qid"]) for problem in problems if str(problem["qid"]) in seed_records]
    if args.limit is not None:
        qids = qids[: args.limit]
    items = [
        {"qid": qid, "problem": problem_by_qid[qid], "seed": seed_records[qid]}
        for qid in qids
        if qid in problem_by_qid and qid in seed_records
    ]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    partial_handle = None
    if args.partial_output is not None:
        args.partial_output.parent.mkdir(parents=True, exist_ok=True)
        partial_handle = args.partial_output.open("w", encoding="utf-8")

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tokenizer = _ensure_tokenizer_padding(
        AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=bool(solver_cfg.get("trust_remote_code", True)),
            local_files_only=bool(solver_cfg.get("local_files_only", True)),
        )
    )
    engine = LLM(
        model=model_name,
        tokenizer=model_name,
        trust_remote_code=bool(solver_cfg.get("trust_remote_code", True)),
        dtype=str(solver_cfg.get("torch_dtype", "float16")),
        tensor_parallel_size=int(args.tensor_parallel_size),
        gpu_memory_utilization=float(args.gpu_memory_utilization),
        max_model_len=int(args.max_model_len),
        max_num_seqs=int(args.max_num_seqs),
        skip_tokenizer_init=False,
        seed=random_seed,
    )
    sampling_params = SamplingParams(
        n=1,
        max_tokens=int(args.max_new_tokens),
        temperature=0.0,
        top_p=1.0,
        repetition_penalty=float(solver_cfg.get("repetition_penalty", 1.0)),
        skip_special_tokens=True,
    )

    records: list[dict[str, Any]] = []
    completed = 0
    for batch in _chunks(items, max(1, int(args.batch_size))):
        prompts = []
        for item in batch:
            user_prompt = _build_meta_prompt(
                item["problem"],
                item["seed"],
                include_trace_chars=int(args.include_trace_chars),
                probe_style=str(args.probe_style),
            )
            prompts.append(
                _render_prompt_with_tokenizer(
                    tokenizer,
                    prompt_template=prompt_template,
                    system_prompt="You are a concise math self-evaluator.",
                    user_prompt=user_prompt,
                    enable_thinking=False,
                )
            )
        started = perf_counter()
        outputs = engine.generate(prompts, sampling_params, use_tqdm=False)
        batch_latency_ms = int((perf_counter() - started) * 1000)
        per_item_latency_ms = int(batch_latency_ms / max(1, len(outputs)))
        for item, output in zip(batch, outputs):
            best = output.outputs[0]
            text = best.text.strip()
            features = _parse_meta(text, probe_style=str(args.probe_style))
            prompt_tokens = len(getattr(output, "prompt_token_ids", []) or [])
            completion_tokens = len(getattr(best, "token_ids", []) or [])
            record = {
                "qid": item["qid"],
                "seed_prediction": item["seed"].get("prediction"),
                "seed_correct": item["seed"].get("correct"),
                "meta_features": features,
                "raw_output": text,
                "total_tokens": completion_tokens,
                "latency_ms": per_item_latency_ms,
                "metadata": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "max_new_tokens": int(args.max_new_tokens),
                    "enable_thinking": False,
                    "prompt_suffix": "/no_think",
                    "backend": "vllm_batched_meta_probe",
                    "probe_style": str(args.probe_style),
                    "random_seed": random_seed,
                },
            }
            records.append(record)
            completed += 1
            if partial_handle is not None:
                partial_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                partial_handle.flush()
            if args.progress_every > 0 and completed % int(args.progress_every) == 0:
                print(f"[{completed}/{len(items)}] qid={record['qid']} meta_keys={sorted(features)}", flush=True)
    if partial_handle is not None:
        partial_handle.close()

    avg_tokens = sum(int(row["total_tokens"]) for row in records) / max(1, len(records))
    parsed = sum(1 for row in records if row["meta_features"])
    payload = {
        "no_think_meta_probe": {
            "summary": {
                "count": len(records),
                "parsed_count": parsed,
                "parsed_rate": parsed / max(1, len(records)),
                "avg_total_tokens": avg_tokens,
                "avg_latency_ms": sum(float(row["latency_ms"]) for row in records) / max(1, len(records)),
            },
            "records": records,
        }
    }
    dump_json(args.output, payload)
    print(f"wrote no-thinking meta probe to {args.output} parsed={parsed}/{len(records)} avg_tokens={avg_tokens:.1f}")


if __name__ == "__main__":
    main()
