from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


def _numeric_answer(value: Any) -> str | None:
    text = str(value).strip()
    if not text:
        return None
    text = text.replace(",", "")
    if re.fullmatch(r"-?\d+(?:\.\d+)?", text):
        return text
    matches = re.findall(r"-?\d+(?:\.\d+)?", text)
    if len(matches) == 1 and len(text) <= 32:
        return matches[0]
    return None


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _load_svamp(limit: int | None) -> list[dict[str, Any]]:
    from datasets import load_dataset

    dataset = load_dataset("ChilleD/SVAMP", split="test")
    rows: list[dict[str, Any]] = []
    for idx, row in enumerate(dataset):
        answer = _numeric_answer(row.get("Answer"))
        if answer is None:
            continue
        question = row.get("question_concat") or f"{row.get('Body', '').strip()} {row.get('Question', '').strip()}".strip()
        rows.append(
            {
                "qid": f"svamp_{row.get('ID') or idx}",
                "problem": question,
                "answer": answer,
                "dataset": "svamp",
                "metadata": {
                    "source_id": row.get("ID"),
                    "type": row.get("Type"),
                    "equation": row.get("Equation"),
                    "split": "test",
                },
            }
        )
        if limit is not None and len(rows) >= limit:
            break
    return rows


def _load_theoremqa(limit: int | None) -> list[dict[str, Any]]:
    from datasets import load_dataset

    dataset = load_dataset("TIGER-Lab/TheoremQA", split="test")
    rows: list[dict[str, Any]] = []
    allowed_types = {"integer", "float", "number", "numeric"}
    for idx, row in enumerate(dataset):
        answer_type = str(row.get("Answer_type") or "").strip().lower()
        answer = _numeric_answer(row.get("Answer"))
        if answer is None:
            continue
        if answer_type and answer_type not in allowed_types:
            continue
        rows.append(
            {
                "qid": f"theoremqa_{idx}",
                "problem": str(row.get("Question") or "").strip(),
                "answer": answer,
                "dataset": "theoremqa_numeric",
                "metadata": {
                    "answer_type": row.get("Answer_type"),
                    "has_picture": row.get("Picture") is not None,
                    "split": "test",
                },
            }
        )
        if limit is not None and len(rows) >= limit:
            break
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["svamp", "theoremqa"], required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    rows = _load_svamp(args.limit) if args.dataset == "svamp" else _load_theoremqa(args.limit)
    if not rows:
        raise SystemExit(f"No rows prepared for dataset={args.dataset}")
    _write_jsonl(Path(args.output), rows)
    print(f"wrote {len(rows)} rows to {args.output}")
    print(json.dumps(rows[0], ensure_ascii=False, indent=2)[:1200])


if __name__ == "__main__":
    main()
