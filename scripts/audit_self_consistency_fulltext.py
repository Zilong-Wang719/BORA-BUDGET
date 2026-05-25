from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from audit_completion_extraction import (
    _extract_final_answer_candidates_with_fallback,
    _safe_robust_equal,
)
from evaluate_math_answers_robust import compact_text, numeric_equal, numeric_set_equal

try:
    from bora.common import normalize_answer
except Exception:  # pragma: no cover - local lightweight fallback

    def normalize_answer(value: Any) -> str | None:
        if value is None:
            return None
        text = str(value).strip()
        return text.lower().replace(" ", "") if text else None


def _records(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload.get("records"), list):
        return payload["records"]
    block = payload.get("self_consistency")
    if isinstance(block, dict) and isinstance(block.get("records"), list):
        return block["records"]
    for value in payload.values():
        if isinstance(value, dict) and isinstance(value.get("records"), list):
            return value["records"]
    raise ValueError(f"Could not find records in {path}")


def _majority_vote(predictions: list[str | None]) -> str | None:
    votes: Counter[str] = Counter()
    first_seen: dict[str, int] = {}
    representative: dict[str, str] = {}
    for idx, prediction in enumerate(predictions):
        normalized = normalize_answer(prediction)
        if normalized is None:
            continue
        votes[normalized] += 1
        first_seen.setdefault(normalized, idx)
        representative.setdefault(normalized, str(prediction))
    if not votes:
        return None
    winner = max(votes, key=lambda item: (votes[item], -first_seen[item]))
    return representative[winner]


def _quick_equal(pred: Any, gold: Any) -> bool:
    """Cheap diagnostic equality for per-sample stats.

    The paper-facing majority accuracy below still uses the timeout-protected
    robust matcher. This helper avoids spawning one math_verify subprocess for
    every SC candidate while still giving useful any-candidate diagnostics.
    """

    if pred is None or gold is None:
        return False
    return compact_text(pred) == compact_text(gold) or numeric_equal(pred, gold) or numeric_set_equal(pred, gold)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-jsonl", required=True)
    args = parser.parse_args()

    rows_out: list[dict[str, Any]] = []
    old_correct = 0
    fixed_majority_correct = 0
    any_candidate_correct = 0
    fixed_rescues = 0
    fixed_losses = 0
    total_candidate_samples = 0
    corrected_candidate_samples = 0

    for row in _records(Path(args.input)):
        gold = row.get("gold_answer")
        old_ok = bool(row.get("correct"))
        old_correct += int(old_ok)
        candidate_rows = []
        fixed_predictions: list[str | None] = []
        sample_correct_flags: list[bool] = []
        for candidate in (row.get("metadata") or {}).get("candidates", []):
            text = str(
                candidate.get("completion_text")
                or candidate.get("completion_tail")
                or candidate.get("completion_head")
                or ""
            )
            candidates = _extract_final_answer_candidates_with_fallback(text, gold=None)
            pred = candidates[0] if candidates else None
            ok = _quick_equal(pred, gold)
            method = "first_candidate_gold_free"
            fixed_predictions.append(pred)
            sample_correct_flags.append(ok)
            total_candidate_samples += 1
            corrected_candidate_samples += int(ok)
            candidate_rows.append(
                {
                    "sample_index": candidate.get("sample_index"),
                    "old_prediction": candidate.get("prediction"),
                    "fixed_prediction": pred,
                    "fixed_correct": ok,
                    "fixed_method": method,
                    "fixed_candidates": candidates,
                    "completion_tokens": candidate.get("completion_tokens"),
                }
            )
        majority_prediction = _majority_vote(fixed_predictions)
        majority_ok, majority_method = _safe_robust_equal(majority_prediction, gold)
        any_ok = any(sample_correct_flags)
        fixed_majority_correct += int(majority_ok)
        any_candidate_correct += int(any_ok)
        fixed_rescues += int((not old_ok) and majority_ok)
        fixed_losses += int(old_ok and (not majority_ok))
        rows_out.append(
            {
                "qid": row.get("qid"),
                "gold_answer": gold,
                "old_prediction": row.get("prediction"),
                "old_correct": old_ok,
                "fixed_majority_prediction": majority_prediction,
                "fixed_majority_correct": majority_ok,
                "fixed_majority_method": majority_method,
                "any_candidate_correct": any_ok,
                "candidate_rows": candidate_rows,
                "total_tokens": row.get("total_tokens") or row.get("solver_tokens"),
            }
        )

    n = len(rows_out)
    summary = {
        "input": args.input,
        "count": n,
        "old_correct": old_correct,
        "old_accuracy": old_correct / max(1, n),
        "fixed_majority_correct": fixed_majority_correct,
        "fixed_majority_accuracy": fixed_majority_correct / max(1, n),
        "fixed_rescues_vs_old": fixed_rescues,
        "fixed_losses_vs_old": fixed_losses,
        "any_candidate_correct": any_candidate_correct,
        "any_candidate_accuracy": any_candidate_correct / max(1, n),
        "candidate_sample_correct": corrected_candidate_samples,
        "candidate_sample_count": total_candidate_samples,
        "candidate_sample_accuracy": corrected_candidate_samples / max(1, total_candidate_samples),
    }
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_json).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    with Path(args.output_jsonl).open("w", encoding="utf-8") as handle:
        for row in rows_out:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
