from __future__ import annotations

import argparse
import json
from pathlib import Path

from evaluate_math_answers_robust import load_records, robust_equal


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--limit", type=int, default=12)
    args = parser.parse_args()

    records = load_records(Path(args.input), method=None)
    flips_pos = []
    flips_neg = []
    for row in records:
        ok, method = robust_equal(row.get("prediction"), row.get("gold_answer"))
        old = bool(row.get("correct"))
        item = {
            "qid": row.get("qid"),
            "method": method,
            "prediction": row.get("prediction"),
            "gold_answer": row.get("gold_answer"),
            "old_correct": old,
            "robust_correct": ok,
        }
        if (not old) and ok:
            flips_pos.append(item)
        elif old and not ok:
            flips_neg.append(item)
    print(f"old_wrong_to_robust_correct={len(flips_pos)}")
    for item in flips_pos[: args.limit]:
        print(json.dumps(item, ensure_ascii=False))
    print(f"old_correct_to_robust_wrong={len(flips_neg)}")
    for item in flips_neg[: args.limit]:
        print(json.dumps(item, ensure_ascii=False))


if __name__ == "__main__":
    main()
