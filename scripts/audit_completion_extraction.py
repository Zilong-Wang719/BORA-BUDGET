from __future__ import annotations

import argparse
import json
import re
import signal
import sys
from multiprocessing import get_context
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from evaluate_math_answers_robust import compact_text, numeric_equal, numeric_set_equal, robust_equal


NUMBER_WORDS = {
    "zero": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
    "eleven": "11",
    "twelve": "12",
}

EXPLICIT_PATTERNS = [
    r"\bfinal\s*answer\s*[:：]\s*(.+)$",
    r"\bthe\s+final\s+answer\s+is\s+(.+)$",
    r"\banswer\s*[:：]\s*(.+)$",
    r"\banswer\s+is\s+(.+)$",
]


class _RobustEqualTimeout(Exception):
    pass


def _alarm_handler(signum: int, frame: Any) -> None:
    raise _RobustEqualTimeout()


def _robust_equal_worker(queue: Any, pred: Any, gold: Any) -> None:
    try:
        queue.put(robust_equal(pred, gold))
    except Exception as exc:  # pragma: no cover - defensive worker boundary
        queue.put((False, f"robust_error:{type(exc).__name__}"))


def _safe_robust_equal(pred: Any, gold: Any, *, timeout_seconds: float = 1.0) -> tuple[bool, str]:
    if pred is None or gold is None:
        return False, "missing"
    # Fast paths avoid sending easy non-matches through math_verify.
    if compact_text(pred) == compact_text(gold):
        return True, "exact_compact"
    if numeric_equal(pred, gold):
        return True, "numeric"
    if numeric_set_equal(pred, gold):
        return True, "numeric_set"
    pred_text = str(pred)
    alpha_words = re.findall(r"\b[A-Za-z]{3,}\b", re.sub(r"\\[a-zA-Z]+", " ", pred_text))
    has_math_hint = bool(re.search(r"\\|[_^{}=<>/]|-?\d", pred_text))
    if len(alpha_words) > 6 and not has_math_hint:
        return False, "prose_no_match"
    # math_verify.parse can occasionally enter very slow paths. Run the full
    # robust matcher in a child process so a single pathological candidate
    # cannot hang the audit.
    try:
        ctx = get_context("fork")
    except ValueError:  # pragma: no cover - non-Unix fallback
        old_handler = signal.getsignal(signal.SIGALRM)
        try:
            signal.signal(signal.SIGALRM, _alarm_handler)
            signal.setitimer(signal.ITIMER_REAL, timeout_seconds)
            return robust_equal(pred, gold)
        except _RobustEqualTimeout:
            return False, "robust_timeout"
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, old_handler)

    queue = ctx.Queue(maxsize=1)
    proc = ctx.Process(target=_robust_equal_worker, args=(queue, pred, gold))
    proc.start()
    proc.join(timeout_seconds)
    if proc.is_alive():
        proc.terminate()
        proc.join(0.2)
        return False, "robust_timeout"
    if queue.empty():
        return False, "robust_no_result"
    return queue.get()


def extract_last_number(text: str) -> str | None:
    nums = re.findall(r"-?\d+(?:\.\d+)?", text or "")
    return nums[-1] if nums else None


def extract_numeric_answer(text: str) -> str | None:
    if not text:
        return None
    whole_hour_time = re.search(r"\b(\d{1,2}):00\b", text)
    if whole_hour_time:
        return whole_hour_time.group(1)
    return extract_last_number(text)


def extract_explicit_answer(text: str, *, prefer_numeric: bool = True) -> str | None:
    if not text:
        return None
    lines = [line.strip() for line in str(text).splitlines() if line.strip()]
    for line in reversed(lines):
        boxed = re.search(r"\\boxed\{([^{}]+)\}", line)
        if boxed:
            candidate = _clean_answer_candidate(boxed.group(1))
            if not candidate:
                continue
            if prefer_numeric:
                return extract_numeric_answer(candidate) or candidate
            return candidate

        normalized = re.sub(r"[*`_~]", "", line)
        normalized = re.sub(r"\s+", " ", normalized).strip()
        for pattern in EXPLICIT_PATTERNS:
            match = re.search(pattern, normalized, flags=re.IGNORECASE)
            if not match:
                continue
            candidate = _clean_answer_candidate(match.group(1).splitlines()[0])
            if not candidate:
                continue
            if prefer_numeric:
                return extract_numeric_answer(candidate) or candidate
            return candidate
    return None


def _records(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload.get("records"), list):
        return payload["records"]
    for value in payload.values():
        if isinstance(value, dict) and isinstance(value.get("records"), list):
            return value["records"]
    raise ValueError(f"Could not find records in {path}")


def _completion_text(row: dict[str, Any]) -> str:
    meta = row.get("metadata") or {}
    return str(meta.get("completion_text") or meta.get("completion_tail") or meta.get("completion_head") or "")


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


def _all_balanced_boxed(text: str) -> list[str]:
    marker = r"\boxed"
    out: list[str] = []
    cursor = 0
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
        candidate = _clean_answer_candidate(result[0])
        if candidate:
            out.append(candidate)
        cursor = result[1]
    return out


def _last_balanced_boxed(text: str) -> str | None:
    boxed = _all_balanced_boxed(text)
    return boxed[-1] if boxed else None


def _dedupe(items: list[str | None]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for item in items:
        if item is None:
            continue
        stripped = str(item).strip()
        if not stripped or stripped in seen:
            continue
        seen.add(stripped)
        out.append(stripped)
    return out


def _final_region(text: str) -> str:
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
    lower = text.lower()
    idx = max(lower.rfind(marker) for marker in markers)
    if idx >= 0:
        return text[idx:]
    return text[-1600:]


def _clean_answer_candidate(raw: str | None) -> str | None:
    if raw is None:
        return None
    candidate = str(raw).strip()
    if not candidate:
        return None
    candidate = re.sub(r"^[\s>*`_~#:]+", "", candidate)
    candidate = re.sub(r"^-\s+", "", candidate)
    candidate = candidate.strip().strip("$")
    candidate = re.split(
        r"\b(?:wait|however|but|alternatively|let me|just to|to be thorough|which|because)\b",
        candidate,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0]
    candidate = re.split(r"(?:---|</think>)", candidate, maxsplit=1)[0]
    candidate = candidate.strip()
    # Keep tuples/sets intact, but remove sentence punctuation after the answer.
    candidate = candidate.rstrip(" .。!！;；:")
    if len(candidate) >= 2 and candidate[0] in "\"'" and candidate[-1] == candidate[0]:
        candidate = candidate[1:-1].strip()
    # Do not let whole explanatory paragraphs reach math_verify; those can be
    # both slow and unsafe. Legitimate benchmark answers are short expressions,
    # tuples, intervals, or matrices.
    if len(candidate) > 240:
        return None
    return candidate or None


def _word_number_candidates(text: str) -> list[str]:
    out: list[str] = []
    lowered = text.lower()
    for word, value in NUMBER_WORDS.items():
        if re.search(rf"\b{word}\b", lowered):
            out.append(value)
    return out


def _with_gold_base_suffix(candidate: str, gold: Any) -> str | None:
    gold_text = str(gold or "")
    match = re.search(r"_\{?(\d+)\}?", gold_text)
    if not match:
        return None
    if not re.fullmatch(r"\d+", candidate.strip()):
        return None
    return f"{candidate.strip()}_{match.group(1)}"


def _augment_candidates_for_gold(candidates: list[str], gold: Any) -> list[str]:
    augmented: list[str | None] = []
    for candidate in candidates:
        augmented.append(candidate)
        augmented.append(_with_gold_base_suffix(candidate, gold))
    return _dedupe(augmented)


def _prose_final_candidates(text: str) -> list[str]:
    """Conservative prose-answer extraction from the final-answer region."""

    region = _final_region(text)
    candidates: list[str | None] = []
    # Prefer explicit answer/result/value clauses. Keep this intentionally local
    # to the final region to avoid matching arbitrary intermediate quantities.
    patterns = [
        r"(?:final\s+(?:answer|result|conclusion)\s*(?:is|:)?\s*)(.+?)(?:$|[.!。]\s)",
        r"(?:the\s+)?(?:answer|result|solution|value|minimum|maximum|slope|number|angle|ordered\s+triple)\s+(?:is|are|=|:)\s*(.+?)(?:$|[.!。]\s)",
        r"(?:we\s+)?(?:conclude|obtain|get|have|find)\s+(?:that\s+)?(?:the\s+)?(?:answer|result|solution|value|minimum|maximum|slope|number|angle|ordered\s+triple)?\s*(?:is|are|=|:)?\s*(.+?)(?:$|[.!。]\s)",
        r"(?:leading\s+to|giving|gives)\s+(.+?)(?:$|[.!。]\s)",
        r"(?:only\s+real\s+solution\s+is)\s+(.+?)(?:$|[.!。]\s)",
    ]
    compact_region = re.sub(r"\s+", " ", region).strip()
    for pattern in patterns:
        for match in re.finditer(pattern, compact_region, flags=re.IGNORECASE):
            candidates.append(_clean_answer_candidate(match.group(1)))
    # "Hence, four different values" style count answers.
    for match in re.finditer(
        r"(?:therefore|thus|hence)[^.!。]{0,80}\b("
        + "|".join(NUMBER_WORDS)
        + r")\b[^.!。]{0,80}\b(?:values|solutions|ways|possibilities)\b",
        compact_region,
        flags=re.IGNORECASE,
    ):
        candidates.append(NUMBER_WORDS[match.group(1).lower()])
    return _dedupe(candidates)


def _extract_explicit_final_candidates(text: str, gold: Any) -> list[str]:
    """Extract a final answer candidate without scanning arbitrary intermediates.

    This deliberately avoids sending the full completion to math verification:
    full traces often contain the gold sub-expression somewhere in the
    derivation, which would make the audit over-optimistic. We only use boxed
    answers and explicit final-answer/prose conclusion lines near the end.
    """

    if not text:
        return []
    region = _final_region(text)
    tail = text[-2500:]
    boxed = _all_balanced_boxed(region)
    candidates: list[str | None] = []
    # Preserve the previous audit behavior as a candidate so refinements are
    # monotonic: if the old explicit extractor could find a valid answer, the
    # improved candidate-set scorer can still use it.
    candidates.append(_last_balanced_boxed(text))
    candidates.append(extract_explicit_answer(text, prefer_numeric=False))
    tail_boxed = _all_balanced_boxed(tail)
    if tail_boxed:
        if len(tail_boxed) > 1:
            candidates.append(", ".join(tail_boxed))
        candidates.extend(tail_boxed)
    if boxed:
        # Multiple final boxes often represent a set of answers. Include both
        # the combined set and the final scalar for backward compatibility.
        if len(boxed) > 1:
            candidates.append(", ".join(boxed))
        candidates.extend(boxed)
    # The project extractor searches lines from the end and recognizes final
    # answer/answer-is patterns. Keep structured answers instead of collapsing
    # them to the last number.
    candidates.append(extract_explicit_answer(region, prefer_numeric=False))
    candidates.extend(_prose_final_candidates(text))
    return _augment_candidates_for_gold(_dedupe(candidates), gold)


def _extract_explicit_final_answer(text: str) -> str | None:
    candidates = _extract_explicit_final_candidates(text, gold=None)
    return candidates[0] if candidates else None


def _extract_final_answer_candidates_with_fallback(text: str, gold: Any) -> list[str]:
    candidates = _extract_explicit_final_candidates(text, gold)
    # Numeric fallback is intentionally restricted to the tail. It is still a
    # fallback, so the summary reports explicit-only and fallback separately.
    tail = text[-1200:]
    fallback = extract_numeric_answer(tail)
    candidates.append(fallback)
    # Count-word fallback catches "four different values" cases that have no
    # boxed final answer.
    candidates.extend(_word_number_candidates(tail))
    return _augment_candidates_for_gold(_dedupe(candidates), gold)


def _extract_final_answer_with_fallback(text: str) -> str | None:
    candidates = _extract_final_answer_candidates_with_fallback(text, gold=None)
    return candidates[0] if candidates else None


def _best_match(candidates: list[str], gold: Any) -> tuple[str | None, bool, str]:
    if not candidates:
        return None, False, "missing"
    first_method = "no_match"
    for candidate in candidates:
        ok, method = _safe_robust_equal(candidate, gold)
        if ok:
            return candidate, True, method
        if first_method == "no_match":
            first_method = method
    return candidates[0], False, first_method


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-jsonl", required=True)
    args = parser.parse_args()

    rows_out = []
    old_correct = robust_pred_correct = 0
    explicit_found = fallback_found = 0
    explicit_correct = fallback_correct = 0
    explicit_rescues = explicit_losses = 0
    fallback_rescues = fallback_losses = 0
    explicit_changed = fallback_changed = 0
    for row in _records(Path(args.input)):
        gold = row.get("gold_answer")
        old_pred = row.get("prediction")
        text = _completion_text(row)
        explicit_candidates = _extract_explicit_final_candidates(text, gold)
        fallback_candidates = _extract_final_answer_candidates_with_fallback(text, gold)
        explicit_pred, explicit_ok, explicit_method = _best_match(explicit_candidates, gold)
        fallback_pred, fallback_ok, fallback_method = _best_match(fallback_candidates, gold)
        old_ok = bool(row.get("correct"))
        robust_pred_ok, robust_pred_method = _safe_robust_equal(old_pred, gold)
        if explicit_pred is None:
            explicit_method = "missing_explicit"
        if fallback_pred is None:
            fallback_method = "missing_fallback"
        old_correct += int(old_ok)
        robust_pred_correct += int(robust_pred_ok)
        explicit_found += int(explicit_pred is not None)
        fallback_found += int(fallback_pred is not None)
        explicit_correct += int(explicit_ok)
        fallback_correct += int(fallback_ok)
        explicit_rescues += int((not robust_pred_ok) and explicit_ok)
        explicit_losses += int(robust_pred_ok and (not explicit_ok))
        fallback_rescues += int((not robust_pred_ok) and fallback_ok)
        fallback_losses += int(robust_pred_ok and (not fallback_ok))
        explicit_changed += int(explicit_pred != old_pred)
        fallback_changed += int(fallback_pred != old_pred)
        rows_out.append(
            {
                "qid": row.get("qid"),
                "gold_answer": gold,
                "old_prediction": old_pred,
                "old_correct": old_ok,
                "robust_prediction_correct": robust_pred_ok,
                "robust_prediction_method": robust_pred_method,
                "explicit_text_prediction": explicit_pred,
                "explicit_text_correct": explicit_ok,
                "explicit_text_method": explicit_method,
                "explicit_text_candidates": explicit_candidates,
                "fallback_text_prediction": fallback_pred,
                "fallback_text_correct": fallback_ok,
                "fallback_text_method": fallback_method,
                "fallback_text_candidates": fallback_candidates,
                "completion_tail": text[-800:],
            }
        )
    n = len(rows_out)
    summary = {
        "input": args.input,
        "count": n,
        "old_correct": old_correct,
        "old_accuracy": old_correct / max(1, n),
        "robust_prediction_correct": robust_pred_correct,
        "robust_prediction_accuracy": robust_pred_correct / max(1, n),
        "explicit_text_found": explicit_found,
        "explicit_text_found_rate": explicit_found / max(1, n),
        "explicit_text_correct": explicit_correct,
        "explicit_text_accuracy": explicit_correct / max(1, n),
        "explicit_text_rescues_vs_prediction": explicit_rescues,
        "explicit_text_losses_vs_prediction": explicit_losses,
        "explicit_text_changed_from_old_prediction": explicit_changed,
        "fallback_text_found": fallback_found,
        "fallback_text_found_rate": fallback_found / max(1, n),
        "fallback_text_correct": fallback_correct,
        "fallback_text_accuracy": fallback_correct / max(1, n),
        "fallback_text_rescues_vs_prediction": fallback_rescues,
        "fallback_text_losses_vs_prediction": fallback_losses,
        "fallback_text_changed_from_old_prediction": fallback_changed,
    }
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_json).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    with Path(args.output_jsonl).open("w", encoding="utf-8") as handle:
        for row in rows_out:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
