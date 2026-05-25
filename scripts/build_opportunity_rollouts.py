from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bora.common import dump_jsonl, is_correct, load_config, load_problem_split, normalize_answer


KEYWORD_GROUPS = {
    "geometry": ["angle", "triangle", "circle", "radius", "area", "perimeter", "polar"],
    "algebra": ["solve", "equation", "polynomial", "function", "roots", "variable"],
    "number_theory": ["integer", "divisible", "prime", "modulo", "remainder"],
    "fraction_decimal": ["fraction", "decimal", "percent", "%", "ratio"],
    "comparison": ["minimum", "maximum", "least", "greatest", "more than", "less than"],
    "proof_like": ["prove", "show that", "find all", "such that"],
}


def _load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def _records(payload: dict[str, Any], method: str | None = None) -> list[dict[str, Any]]:
    if isinstance(payload.get("rows"), list):
        return payload["rows"]
    if isinstance(payload.get("records"), list):
        return payload["records"]
    if method is not None:
        block = payload.get(method)
        if isinstance(block, dict) and isinstance(block.get("records"), list):
            return block["records"]
        raise KeyError(f"Method {method!r} not found. Available top-level keys: {sorted(payload)}")
    candidates = [
        value.get("records")
        for value in payload.values()
        if isinstance(value, dict) and isinstance(value.get("records"), list)
    ]
    if len(candidates) != 1:
        raise ValueError("Could not infer records block; pass --*-method explicitly.")
    return candidates[0]


def _qid_map(path: str | None, method: str | None = None) -> dict[str, dict[str, Any]]:
    if not path:
        return {}
    return {str(row["qid"]): row for row in _records(_load_json(path), method)}


def _float_answer(value: Any) -> float | None:
    norm = normalize_answer(value)
    if norm is None:
        return None
    try:
        return float(norm.replace(",", ""))
    except Exception:
        return None


def _sign(value: float | None) -> int:
    if value is None or value == 0:
        return 0
    return 1 if value > 0 else -1


def _is_integer(value: float | None) -> bool:
    return value is not None and math.isfinite(value) and abs(value - round(value)) < 1e-9


def _numeric_count(text: str) -> int:
    return len(re.findall(r"-?\d+(?:\.\d+)?", text or ""))


def _completion_head(record: dict[str, Any]) -> str:
    metadata = record.get("metadata") or {}
    return str(metadata.get("completion_head") or record.get("thinking_sample_head") or "")


def _completion_text(record: dict[str, Any]) -> str:
    metadata = record.get("metadata") or {}
    return str(
        metadata.get("completion_text")
        or metadata.get("completion_tail")
        or metadata.get("completion_head")
        or record.get("thinking_sample_head")
        or ""
    )


def _token_value(record: dict[str, Any], *keys: str) -> int:
    metadata = record.get("metadata") or {}
    for key in keys:
        value = record.get(key)
        if value is None:
            value = metadata.get(key)
        try:
            return int(value)
        except (TypeError, ValueError):
            continue
    return 0


def _add_feature(features: dict[str, float], name: str, value: Any) -> None:
    if isinstance(value, bool):
        features[name] = 1.0 if value else 0.0
        return
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if math.isfinite(float(value)):
            features[name] = float(value)


def _seed_feature_metadata(bora_record: dict[str, Any] | None) -> dict[str, Any]:
    if not bora_record:
        return {}
    return ((bora_record.get("metadata") or {}).get("seed_features") or {})


def _old_trigger_nonbranch_or_long300(bora_record: dict[str, Any] | None) -> bool:
    if not bora_record:
        return False
    metadata = bora_record.get("metadata") or {}
    seed_features = _seed_feature_metadata(bora_record)
    hard_accepted = bool(metadata.get("hard_accepted", True))
    rescue_action = metadata.get("rescue_action")
    trace_words = seed_features.get("trace_words")
    return (not hard_accepted and rescue_action != "BRANCH_2") or (
        isinstance(trace_words, (int, float)) and trace_words >= 300
    )


def _compact_candidate(value: Any) -> str:
    text = normalize_answer(value)
    if text is None:
        text = str(value or "")
    text = re.sub(r"\s+", "", text)
    text = text.strip().strip("$").strip()
    text = text.replace("\\left", "").replace("\\right", "")
    return text.lower()


def _clean_trace_candidate(raw: str | None) -> str | None:
    if raw is None:
        return None
    candidate = str(raw).strip()
    if not candidate:
        return None
    candidate = re.sub(r"^[\s>*`_~#:]+", "", candidate)
    candidate = candidate.strip().strip("$")
    candidate = re.split(
        r"\b(?:wait|however|but|alternatively|let me|because|therefore|thus|hence)\b",
        candidate,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0]
    candidate = re.split(r"(?:---|</think>)", candidate, maxsplit=1)[0]
    candidate = candidate.strip().rstrip(" .。!！;；:")
    if len(candidate) > 240:
        return None
    return candidate or None


def _balanced_brace_content(text: str, start: int) -> tuple[str, int] | None:
    if start < 0 or start >= len(text) or text[start] != "{":
        return None
    depth = 0
    out: list[str] = []
    for idx in range(start, len(text)):
        ch = text[idx]
        if ch == "{":
            depth += 1
            if depth > 1:
                out.append(ch)
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return "".join(out), idx + 1
            out.append(ch)
        else:
            out.append(ch)
    return None


def _all_boxed_candidates(text: str) -> list[str]:
    out: list[str] = []
    cursor = 0
    marker = r"\boxed"
    while True:
        idx = text.find(marker, cursor)
        if idx < 0:
            break
        brace = text.find("{", idx + len(marker))
        if brace < 0:
            cursor = idx + len(marker)
            continue
        result = _balanced_brace_content(text, brace)
        if result is None:
            cursor = idx + len(marker)
            continue
        candidate = _clean_trace_candidate(result[0])
        if candidate:
            out.append(candidate)
        cursor = result[1]
    return out


def _final_region(text: str) -> str:
    lower = text.lower()
    markers = [
        "final answer",
        "final result",
        "final conclusion",
        "answer:",
        "answer is",
        "therefore",
        "thus",
        "hence",
        "we conclude",
    ]
    idx = max(lower.rfind(marker) for marker in markers)
    return text[idx:] if idx >= 0 else text[-1800:]


def _explicit_trace_candidates(text: str) -> list[str]:
    region = _final_region(text)
    compact_region = re.sub(r"\s+", " ", region).strip()
    candidates: list[str] = []
    patterns = [
        r"(?:final\s+(?:answer|result|conclusion)\s*(?:is|:)?\s*)(.+?)(?:$|[.!。]\s)",
        r"(?:the\s+)?(?:answer|result|solution|value|minimum|maximum|number|angle)\s+(?:is|are|=|:)\s*(.+?)(?:$|[.!。]\s)",
        r"(?:we\s+)?(?:conclude|obtain|get|have|find)\s+(?:that\s+)?(?:the\s+)?(?:answer|result|solution|value)?\s*(?:is|are|=|:)?\s*(.+?)(?:$|[.!。]\s)",
    ]
    for pattern in patterns:
        for match in re.finditer(pattern, compact_region, flags=re.IGNORECASE):
            candidate = _clean_trace_candidate(match.group(1))
            if candidate:
                candidates.append(candidate)
    for line in reversed([line.strip() for line in region.splitlines() if line.strip()]):
        candidate = re.search(r"(?:answer|result)\s*(?:is|=|:)\s*(.+)$", line, flags=re.IGNORECASE)
        if candidate:
            cleaned = _clean_trace_candidate(candidate.group(1))
            if cleaned:
                candidates.append(cleaned)
                break
    return candidates


def _dedupe_compact(candidates: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        compact = _compact_candidate(candidate)
        if not compact or compact in seen:
            continue
        seen.add(compact)
        out.append(candidate)
    return out


def _numeric_values(candidates: list[str]) -> list[float]:
    values: list[float] = []
    for candidate in candidates:
        value = _float_answer(candidate)
        if value is not None and math.isfinite(value):
            values.append(value)
    return values


def _count_markers(text: str, terms: list[str]) -> int:
    lowered = text.lower()
    return sum(len(re.findall(rf"\b{re.escape(term)}\b", lowered)) for term in terms)


def _add_seed_self_consistency_features(
    features: dict[str, float],
    *,
    seed_answer: Any,
    completion_text: str,
) -> None:
    """Add non-generative trace self-consistency features.

    These features inspect only the seed trace produced before the controller
    runs. They are designed to capture whether the seed answer is stable inside
    its own reasoning: multiple final candidates, conflicting boxed answers,
    revision language, and disagreement between the stored prediction and the
    trace's final candidate.
    """

    text = completion_text or ""
    boxed = _all_boxed_candidates(text)
    explicit = _explicit_trace_candidates(text)
    candidates = _dedupe_compact(boxed + explicit)
    final_region = _final_region(text)
    final_nums = re.findall(r"-?\d+(?:\.\d+)?", final_region)
    numeric_values = _numeric_values(candidates)
    unique_numeric = sorted({round(value, 10) for value in numeric_values})
    seed_compact = _compact_candidate(seed_answer)
    candidate_compacts = [_compact_candidate(candidate) for candidate in candidates]
    boxed_compacts = [_compact_candidate(candidate) for candidate in boxed]
    last_candidate = candidates[-1] if candidates else None
    first_candidate = candidates[0] if candidates else None
    last_compact = _compact_candidate(last_candidate) if last_candidate is not None else ""
    first_compact = _compact_candidate(first_candidate) if first_candidate is not None else ""
    distinct_candidate_count = len(set(candidate_compacts))
    distinct_boxed_count = len(set(boxed_compacts))
    correction_markers = [
        "wait",
        "mistake",
        "actually",
        "however",
        "but",
        "recheck",
        "recompute",
        "instead",
        "not correct",
        "wrong",
    ]
    confidence_markers = [
        "verify",
        "check",
        "confirm",
        "indeed",
        "consistent",
    ]
    correction_count = _count_markers(text, correction_markers)
    confidence_count = _count_markers(text, confidence_markers)

    _add_feature(features, "seed_sc_completion_char_len", len(text))
    _add_feature(features, "seed_sc_boxed_count", len(boxed))
    _add_feature(features, "seed_sc_unique_boxed_count", distinct_boxed_count)
    _add_feature(features, "seed_sc_explicit_candidate_count", len(explicit))
    _add_feature(features, "seed_sc_candidate_count", len(candidates))
    _add_feature(features, "seed_sc_unique_candidate_count", distinct_candidate_count)
    _add_feature(features, "seed_sc_has_candidate", bool(candidates))
    _add_feature(features, "seed_sc_multiple_distinct_candidates", distinct_candidate_count > 1)
    _add_feature(features, "seed_sc_multiple_distinct_boxed", distinct_boxed_count > 1)
    _add_feature(features, "seed_sc_prediction_in_candidates", bool(seed_compact and seed_compact in candidate_compacts))
    _add_feature(features, "seed_sc_prediction_matches_last_candidate", bool(seed_compact and seed_compact == last_compact))
    _add_feature(features, "seed_sc_prediction_matches_first_candidate", bool(seed_compact and seed_compact == first_compact))
    _add_feature(features, "seed_sc_prediction_conflicts_candidates", bool(candidates and seed_compact and seed_compact not in candidate_compacts))
    _add_feature(features, "seed_sc_final_region_numeric_count", len(final_nums))
    _add_feature(features, "seed_sc_candidate_numeric_count", len(numeric_values))
    _add_feature(features, "seed_sc_unique_numeric_candidate_count", len(unique_numeric))
    _add_feature(features, "seed_sc_multiple_numeric_candidates", len(unique_numeric) > 1)
    numeric_spread = max(unique_numeric) - min(unique_numeric) if len(unique_numeric) > 1 else 0.0
    _add_feature(features, "seed_sc_numeric_spread_log1p", math.log1p(abs(numeric_spread)))
    _add_feature(features, "seed_sc_correction_marker_count", correction_count)
    _add_feature(features, "seed_sc_has_correction_marker", correction_count > 0)
    _add_feature(features, "seed_sc_confidence_marker_count", confidence_count)
    _add_feature(features, "seed_sc_has_confidence_marker", confidence_count > 0)


def _extract_features(problem: dict[str, Any], seed: dict[str, Any], bora: dict[str, Any] | None) -> dict[str, float]:
    features: dict[str, float] = {}
    seed_answer = seed.get("prediction")
    seed_value = _float_answer(seed_answer)
    seed_norm = normalize_answer(seed_answer)
    seed_total_tokens = _token_value(seed, "total_tokens", "solver_tokens")
    metadata = seed.get("metadata") or {}
    completion_tokens = _token_value(seed, "solver_tokens", "completion_tokens")
    prompt_tokens = _token_value(seed, "prompt_tokens")
    if not prompt_tokens:
        prompt_tokens = int(metadata.get("prompt_tokens") or 0)
    completion_head = _completion_head(seed)
    completion_text = _completion_text(seed)
    question = str(problem.get("question") or "")
    lower_question = question.lower()

    _add_feature(features, "seed_answer_parse_success", seed_norm is not None)
    _add_feature(features, "seed_answer_numeric_parse_success", seed_value is not None)
    _add_feature(features, "seed_answer_is_integer", _is_integer(seed_value))
    _add_feature(features, "seed_answer_is_decimal", seed_value is not None and not _is_integer(seed_value))
    _add_feature(features, "seed_answer_is_negative", seed_value is not None and seed_value < 0)
    _add_feature(features, "seed_answer_is_zero", seed_value == 0 if seed_value is not None else False)
    _add_feature(features, "seed_answer_sign", _sign(seed_value))
    _add_feature(features, "seed_answer_abs_log1p", math.log1p(abs(seed_value)) if seed_value is not None else 0.0)
    _add_feature(features, "seed_answer_digit_count", len(re.findall(r"\d", seed_norm or "")))
    _add_feature(features, "seed_total_tokens", seed_total_tokens)
    _add_feature(features, "seed_completion_tokens", completion_tokens)
    _add_feature(features, "seed_prompt_tokens", prompt_tokens)
    _add_feature(features, "seed_completion_head_numeric_count", _numeric_count(completion_head))
    _add_feature(features, "question_numeric_count", _numeric_count(question))
    _add_feature(features, "question_char_len", len(question))
    _add_feature(features, "question_word_len", len(question.split()))
    _add_seed_self_consistency_features(
        features,
        seed_answer=seed_answer,
        completion_text=completion_text,
    )

    for group, terms in KEYWORD_GROUPS.items():
        _add_feature(features, f"question_has_{group}", any(term in lower_question for term in terms))

    if bora:
        bora_meta = bora.get("metadata") or {}
        seed_features = _seed_feature_metadata(bora)
        _add_feature(features, "old_bora_hard_accepted", bool(bora_meta.get("hard_accepted", True)))
        _add_feature(features, "old_bora_forced_rescue", bool(bora_meta.get("forced_rescue", False)))
        _add_feature(features, "old_bora_nonbranch_rescued", (not bool(bora_meta.get("hard_accepted", True))) and bora_meta.get("rescue_action") != "BRANCH_2")
        _add_feature(features, "old_trigger_nonbranch_or_long300", _old_trigger_nonbranch_or_long300(bora))
        rescue_action = str(bora_meta.get("rescue_action") or "NONE")
        for action in ["NONE", "VERIFY_ADOPT", "VERIFY_ONLY", "RESEED_1", "RESEED_2_VOTE", "BRANCH_2"]:
            _add_feature(features, f"old_rescue_action_{action}", rescue_action == action)
        for key, value in seed_features.items():
            if key in {"feature_names", "seed_answer", "seed_answer_normalized", "verifier_candidate_answer", "triggers"}:
                continue
            if isinstance(value, (bool, int, float)):
                _add_feature(features, f"sf_{key}", value)
        for trigger in seed_features.get("triggers") or []:
            if isinstance(trigger, str):
                _add_feature(features, f"sf_trigger_{trigger}", True)
    else:
        _add_feature(features, "old_trigger_nonbranch_or_long300", False)

    return features


def _near_cap(record: dict[str, Any], *, cap_ratio: float, cap_margin_tokens: int) -> bool:
    metadata = record.get("metadata") or record.get("execution_metadata") or {}
    max_new_tokens = int(metadata.get("max_new_tokens") or 0)
    completion_tokens = _token_value(record, "solver_tokens", "total_tokens", "extra_total_tokens", "completion_tokens")
    if max_new_tokens <= 0:
        return False
    if completion_tokens >= int(max_new_tokens * cap_ratio):
        return True
    return cap_margin_tokens > 0 and completion_tokens >= max_new_tokens - cap_margin_tokens


def _sign_guard_pass(seed_answer: Any, think_answer: Any) -> bool:
    seed_value = _float_answer(seed_answer)
    think_value = _float_answer(think_answer)
    if seed_value is None or think_value is None:
        # Structured answers such as tuples, intervals, and sets do not have a
        # scalar sign. Treat the sign guard as non-applicable rather than
        # rejecting an otherwise well-formed extracted answer.
        return True
    seed_sign = _sign(seed_value)
    think_sign = _sign(think_value)
    return seed_sign == 0 or think_sign == 0 or seed_sign == think_sign


def build_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    config = {**load_config(args.config), "mode": "eval", "random_seed": int(args.random_seed)}
    problems = {str(row["qid"]): row for row in load_problem_split(config, args.split)}
    seed_by_qid = _qid_map(args.seed_result, args.seed_method)
    think_by_qid = _qid_map(args.think_result, args.think_method)
    bora_by_qid = _qid_map(args.bora_result, args.bora_method) if args.bora_result else {}

    qids = sorted(set(seed_by_qid) & set(think_by_qid) & set(problems))
    if args.limit is not None:
        qids = qids[: int(args.limit)]

    rows: list[dict[str, Any]] = []
    for qid in qids:
        problem = problems[qid]
        seed = seed_by_qid[qid]
        think = think_by_qid[qid]
        bora = bora_by_qid.get(qid)
        gold = problem.get("answer") or seed.get("gold_answer") or think.get("gold_answer")
        seed_answer = seed.get("prediction")
        think_answer = think.get("prediction") or think.get("thinking_answer")
        seed_correct = bool(seed.get("correct")) if "correct" in seed else is_correct(seed_answer, gold)
        think_correct = bool(think.get("correct")) if "correct" in think else is_correct(think_answer, gold)
        seed_tokens = _token_value(seed, "total_tokens", "solver_tokens")
        think_tokens = _token_value(think, "total_tokens", "solver_tokens", "extra_total_tokens")
        think_near_cap = _near_cap(
            think,
            cap_ratio=float(args.cap_ratio),
            cap_margin_tokens=int(args.cap_margin_tokens),
        )
        think_parse_success = normalize_answer(think_answer) is not None
        main_filter = bool(think_parse_success and not think_near_cap)
        strict_filter = bool(main_filter and _sign_guard_pass(seed_answer, think_answer))
        features = _extract_features(problem, seed, bora)
        harmful = bool(seed_correct and not think_correct)
        helpful = bool((not seed_correct) and think_correct)
        wrong_to_wrong = bool((not seed_correct) and (not think_correct))
        utility = (
            int(think_correct)
            - int(seed_correct)
            - float(args.token_penalty) * think_tokens
            - float(args.harm_penalty) * int(harmful)
        )
        rows.append(
            {
                "qid": qid,
                "question": problem.get("question"),
                "gold_answer": gold,
                "seed_answer": seed_answer,
                "seed_answer_normalized": normalize_answer(seed_answer),
                "think_answer": think_answer,
                "think_answer_normalized": normalize_answer(think_answer),
                "seed_correct": seed_correct,
                "think_correct": think_correct,
                "helpful": helpful,
                "harmful": harmful,
                "wrong_to_wrong": wrong_to_wrong,
                "utility": utility,
                "seed_total_tokens": seed_tokens,
                "think_total_tokens": think_tokens,
                "think_near_cap": think_near_cap,
                "main_filter_pass": main_filter,
                "strict_filter_pass": strict_filter,
                "old_trigger_nonbranch_or_long300": bool(features.get("old_trigger_nonbranch_or_long300", 0.0)),
                "features": features,
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build offline opportunity rollouts from /no_think seed and /think candidate results."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", choices=["train", "dev", "test"], default="test")
    parser.add_argument("--seed-result", required=True)
    parser.add_argument("--think-result", required=True)
    parser.add_argument("--bora-result", default=None)
    parser.add_argument("--seed-method", default="standard_direct_cot")
    parser.add_argument("--think-method", default="standard_direct_cot")
    parser.add_argument("--bora-method", default=None)
    parser.add_argument("--random-seed", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--cap-ratio", type=float, default=0.95)
    parser.add_argument("--cap-margin-tokens", type=int, default=128)
    parser.add_argument("--token-penalty", type=float, default=0.0)
    parser.add_argument("--harm-penalty", type=float, default=1.0)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    rows = build_rows(args)
    dump_jsonl(args.output, rows)
    helpful = sum(bool(row["helpful"]) for row in rows)
    harmful = sum(bool(row["harmful"]) for row in rows)
    seed_correct = sum(bool(row["seed_correct"]) for row in rows)
    think_correct = sum(bool(row["think_correct"]) for row in rows)
    print(
        f"wrote {len(rows)} opportunity rows to {args.output}; "
        f"seed={seed_correct}/{len(rows)} think={think_correct}/{len(rows)} "
        f"helpful={helpful} harmful={harmful}"
    )


if __name__ == "__main__":
    main()
