from __future__ import annotations

import json
from pathlib import Path

from evaluate_math_answers_robust import load_records, robust_equal


PATTERNS = [
    "artifacts/remote_stage_main/math500_lowercap_tokenmatched_20260519/think*k_seed*/standard_direct_cot_think*k_math500_seed*.json",
    "artifacts/remote_stage_main/math500_highcap_think_20260517_8gpu/think*k_seed*/standard_direct_cot_think*k_math500_seed*.json",
]


def evaluate(path: Path) -> dict:
    records = load_records(path, method=None)
    old_correct = 0
    new_correct = 0
    old_to_new = 0
    new_to_old = 0
    for row in records:
        old = bool(row.get("correct"))
        ok, _method = robust_equal(row.get("prediction"), row.get("gold_answer"))
        old_correct += int(old)
        new_correct += int(ok)
        old_to_new += int((not old) and ok)
        new_to_old += int(old and not ok)
    return {
        "path": str(path),
        "count": len(records),
        "old_correct": old_correct,
        "old_accuracy": old_correct / max(1, len(records)),
        "robust_correct": new_correct,
        "robust_accuracy": new_correct / max(1, len(records)),
        "old_to_new_correct": old_to_new,
        "old_correct_to_new_wrong": new_to_old,
    }


def main() -> None:
    paths: list[Path] = []
    for pattern in PATTERNS:
        paths.extend(Path(".").glob(pattern))
    rows = [evaluate(path) for path in sorted(paths)]
    for row in rows:
        print(
            f"{row['path']}: old={row['old_correct']}/{row['count']} "
            f"({100*row['old_accuracy']:.2f}%) robust={row['robust_correct']}/{row['count']} "
            f"({100*row['robust_accuracy']:.2f}%) "
            f"+{row['old_to_new_correct']} -{row['old_correct_to_new_wrong']}"
        )
    out = Path("artifacts/eval_audit/math500_available_robust_recheck_20260520.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
