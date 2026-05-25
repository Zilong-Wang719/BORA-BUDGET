from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any


def _load_mbpp(requested_split: str) -> tuple[list[dict[str, Any]], str, str]:
    try:
        from datasets import DatasetDict, load_dataset
    except Exception as exc:  # pragma: no cover - remote dependency check.
        raise RuntimeError("The `datasets` package is required to prepare MBPP.") from exc

    candidates: list[tuple[str, str | None]] = [
        ("google-research-datasets/mbpp", "sanitized"),
        ("mbpp", "sanitized"),
        ("Muennighoff/mbpp", None),
        ("google-research-datasets/mbpp", None),
    ]
    split_preferences = [requested_split, "test", "validation", "prompt", "train"]
    errors: list[str] = []
    for dataset_name, config_name in candidates:
        try:
            loaded = load_dataset(dataset_name, config_name) if config_name else load_dataset(dataset_name)
            if isinstance(loaded, DatasetDict):
                split_name = next((name for name in split_preferences if name in loaded), None)
                if split_name is None:
                    split_name = next(iter(loaded.keys()))
                rows = [dict(row) for row in loaded[split_name]]
                return rows, dataset_name, split_name
            rows = [dict(row) for row in loaded]
            return rows, dataset_name, requested_split
        except Exception as exc:
            errors.append(f"{dataset_name}/{config_name or '-'}: {exc}")
    raise RuntimeError("Could not load an MBPP dataset candidate:\n" + "\n".join(errors))


def _as_tests(row: dict[str, Any]) -> list[str]:
    tests: list[str] = []
    for key in ("test_list", "tests", "test"):
        value = row.get(key)
        if value is None:
            continue
        if isinstance(value, str):
            tests.extend(line.strip() for line in value.splitlines() if line.strip())
        elif isinstance(value, (list, tuple)):
            tests.extend(str(item).strip() for item in value if str(item).strip())
    # Preserve order while removing duplicates.
    seen: set[str] = set()
    out: list[str] = []
    for test in tests:
        if test not in seen:
            seen.add(test)
            out.append(test)
    return out


def _as_imports(row: dict[str, Any]) -> list[str]:
    value = row.get("test_imports") or row.get("imports") or []
    if isinstance(value, str):
        return [line.strip() for line in value.splitlines() if line.strip()]
    if isinstance(value, (list, tuple)):
        return [str(item).strip() for item in value if str(item).strip()]
    return []


def _qid(row: dict[str, Any], index: int) -> str:
    raw = row.get("task_id") or row.get("id") or row.get("qid") or index
    return f"mbpp_{raw}"


def _prompt(row: dict[str, Any]) -> str:
    for key in ("prompt", "text", "question", "description"):
        value = row.get(key)
        if value:
            return str(value).strip()
    raise ValueError(f"MBPP row is missing a prompt-like field: keys={sorted(row.keys())}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare a small MBPP-style coding benchmark JSONL.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--include-tests-in-prompt", action="store_true")
    args = parser.parse_args()

    rows, dataset_name, split_name = _load_mbpp(args.split)
    prepared: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        tests = _as_tests(row)
        if not tests:
            continue
        prompt = _prompt(row)
        question = prompt
        if args.include_tests_in_prompt:
            question = (
                f"{prompt}\n\n"
                "Your solution should pass these tests, which also define the expected function name:\n"
                + "\n".join(tests)
            )
        prepared.append(
            {
                "qid": _qid(row, index),
                "question": question,
                "answer": row.get("code") or "",
                "difficulty": "unknown",
                "metadata": {
                    "dataset": dataset_name,
                    "split": split_name,
                    "task_id": row.get("task_id") or row.get("id") or index,
                    "prompt": prompt,
                    "tests": tests,
                    "test_imports": _as_imports(row),
                    "reference_code": row.get("code") or "",
                    "source": "mbpp",
                },
            }
        )

    rng = random.Random(args.seed)
    rng.shuffle(prepared)
    if args.limit is not None and args.limit > 0:
        prepared = prepared[: args.limit]

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for item in prepared:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(
        f"wrote {len(prepared)} MBPP examples to {output_path} "
        f"from {dataset_name}/{split_name}"
    )


if __name__ == "__main__":
    main()
