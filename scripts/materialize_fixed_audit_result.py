from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load_records(payload: dict[str, Any], method: str | None) -> list[dict[str, Any]]:
    if isinstance(payload.get("records"), list):
        return payload["records"]
    if method is not None and isinstance(payload.get(method), dict):
        records = payload[method].get("records")
        if isinstance(records, list):
            return records
    for value in payload.values():
        if isinstance(value, dict) and isinstance(value.get("records"), list):
            return value["records"]
    raise ValueError("Could not find records block")


def _summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(records)
    correct = sum(bool(row.get("correct")) for row in records)
    tokens = [float(row.get("total_tokens") or row.get("solver_tokens") or 0.0) for row in records]
    return {
        "accuracy": correct / max(1, n),
        "n": n,
        "correct": correct,
        "avg_tokens": sum(tokens) / max(1, len(tokens)),
        "avg_total_tokens": sum(tokens) / max(1, len(tokens)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rewrite a result JSON with fixed predictions/correctness from completion audit JSONL."
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--audit-jsonl", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--method", default="standard_direct_cot")
    parser.add_argument(
        "--prediction-key",
        default="fallback_text_prediction",
        help="Audit JSONL key to use as fixed prediction.",
    )
    parser.add_argument(
        "--correct-key",
        default="fallback_text_correct",
        help="Audit JSONL key to use as fixed correctness.",
    )
    args = parser.parse_args()

    payload = json.loads(Path(args.input).read_text(encoding="utf-8"))
    audit_by_qid: dict[str, dict[str, Any]] = {}
    with Path(args.audit_jsonl).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            audit_by_qid[str(row["qid"])] = row

    records = _load_records(payload, args.method)
    updated = 0
    for record in records:
        qid = str(record.get("qid"))
        audit = audit_by_qid.get(qid)
        if not audit:
            continue
        fixed_prediction = audit.get(args.prediction_key)
        fixed_correct = bool(audit.get(args.correct_key))
        metadata = record.setdefault("metadata", {})
        metadata["original_prediction_before_fixed_audit"] = record.get("prediction")
        metadata["original_correct_before_fixed_audit"] = record.get("correct")
        metadata["fixed_audit_method"] = args.prediction_key
        metadata["fixed_audit_candidates"] = audit.get("fallback_text_candidates")
        record["prediction"] = fixed_prediction
        record["correct"] = fixed_correct
        record["fixed_audit_correct"] = fixed_correct
        updated += 1

    # Preserve original structure but refresh common summary blocks.
    summary = _summary(records)
    if isinstance(payload.get(args.method), dict):
        payload[args.method]["summary"] = summary
    payload.setdefault("metadata", {})
    payload["metadata"]["fixed_audit_materialized"] = True
    payload["metadata"]["fixed_audit_updated_records"] = updated

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": args.output, "updated": updated, **summary}, indent=2))


if __name__ == "__main__":
    main()
