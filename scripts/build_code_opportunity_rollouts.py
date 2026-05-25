from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bora.common import dump_jsonl


def _load_records(path: str | Path) -> dict[str, dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    records = payload.get("standard_direct_cot", {}).get("records") or payload.get("records") or []
    return {str(record["qid"]): record for record in records}


def _completion_text(record: dict[str, Any]) -> str:
    metadata = dict(record.get("metadata") or {})
    return str(metadata.get("completion_text") or metadata.get("completion_tail") or "")


def _syntax_ok(code: str) -> bool:
    try:
        ast.parse(code or "")
        return True
    except SyntaxError:
        return False


def _ast_counts(code: str) -> dict[str, int]:
    try:
        tree = ast.parse(code or "")
    except SyntaxError:
        return {
            "num_defs": 0,
            "num_imports": 0,
            "num_loops": 0,
            "num_ifs": 0,
            "num_returns": 0,
            "num_asserts": 0,
            "num_calls": 0,
        }
    counts = {
        "num_defs": 0,
        "num_imports": 0,
        "num_loops": 0,
        "num_ifs": 0,
        "num_returns": 0,
        "num_asserts": 0,
        "num_calls": 0,
    }
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            counts["num_defs"] += 1
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            counts["num_imports"] += 1
        elif isinstance(node, (ast.For, ast.While, ast.AsyncFor)):
            counts["num_loops"] += 1
        elif isinstance(node, ast.If):
            counts["num_ifs"] += 1
        elif isinstance(node, ast.Return):
            counts["num_returns"] += 1
        elif isinstance(node, ast.Assert):
            counts["num_asserts"] += 1
        elif isinstance(node, ast.Call):
            counts["num_calls"] += 1
    return counts


def _feature_prefix(prefix: str, record: dict[str, Any]) -> dict[str, float]:
    code = str(record.get("prediction") or "")
    text = _completion_text(record)
    metadata = dict(record.get("metadata") or {})
    code_eval = dict(metadata.get("code_eval") or {})
    counts = _ast_counts(code)
    lines = [line for line in code.splitlines() if line.strip()]
    return {
        f"{prefix}_total_tokens": float(record.get("total_tokens") or record.get("solver_tokens") or 0),
        f"{prefix}_completion_words": float(len(text.split())),
        f"{prefix}_completion_chars": float(len(text)),
        f"{prefix}_code_chars": float(len(code)),
        f"{prefix}_code_lines": float(len(lines)),
        f"{prefix}_has_code_block": float("```" in text),
        f"{prefix}_syntax_ok": float(_syntax_ok(code)),
        f"{prefix}_blocked": float((code_eval.get("error_type") == "blocked_snippet")),
        f"{prefix}_empty": float(not bool(code.strip())),
        **{f"{prefix}_{key}": float(value) for key, value in counts.items()},
    }


def _old_trigger_like(seed: dict[str, Any]) -> bool:
    features = _feature_prefix("seed", seed)
    return (
        features.get("seed_syntax_ok", 0.0) <= 0.0
        or features.get("seed_code_chars", 0.0) < 80
        or features.get("seed_total_tokens", 0.0) > 700
    )


def _filter_pass(record: dict[str, Any]) -> bool:
    metadata = dict(record.get("metadata") or {})
    code_eval = dict(metadata.get("code_eval") or {})
    return bool(code_eval.get("syntax_ok")) and code_eval.get("error_type") != "blocked_snippet"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build code-benchmark opportunity rollout rows.")
    parser.add_argument("--seed-results", required=True)
    parser.add_argument("--think-results", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    seed_records = _load_records(args.seed_results)
    think_records = _load_records(args.think_results)
    rows: list[dict[str, Any]] = []
    for qid in sorted(seed_records):
        if qid not in think_records:
            continue
        seed = seed_records[qid]
        think = think_records[qid]
        seed_correct = bool(seed.get("correct"))
        think_correct = bool(think.get("correct"))
        seed_tokens = float(seed.get("total_tokens") or seed.get("solver_tokens") or 0)
        think_tokens = float(think.get("total_tokens") or think.get("solver_tokens") or 0)
        # Trigger-time features must be seed-only. Thinking-output features would
        # leak counterfactual information that is unavailable before triggering.
        features = _feature_prefix("seed", seed)
        # Analyzer's trace baseline looks for sf_trace_words.
        features["sf_trace_words"] = features.get("seed_completion_words", 0.0)
        features["seed_total_tokens"] = seed_tokens
        features["think_total_tokens"] = think_tokens
        rows.append(
            {
                "qid": qid,
                "seed_answer": seed.get("prediction"),
                "think_answer": think.get("prediction"),
                "seed_answer_normalized": None,
                "think_answer_normalized": None,
                "seed_correct": seed_correct,
                "think_correct": think_correct,
                "helpful": (not seed_correct) and think_correct,
                "harmful": seed_correct and (not think_correct),
                "wrong_to_wrong": (not seed_correct) and (not think_correct),
                "seed_total_tokens": seed_tokens,
                "think_total_tokens": think_tokens,
                "main_filter_pass": _filter_pass(think),
                "strict_filter_pass": _filter_pass(think),
                "old_trigger_nonbranch_or_long300": _old_trigger_like(seed),
                "features": features,
            }
        )

    dump_jsonl(args.output, rows)
    print(
        f"wrote {len(rows)} code opportunity rows to {args.output}; "
        f"seed_acc={sum(r['seed_correct'] for r in rows)/max(1,len(rows)):.3f} "
        f"think_acc={sum(r['think_correct'] for r in rows)/max(1,len(rows)):.3f} "
        f"helpful={sum(r['helpful'] for r in rows)} harmful={sum(r['harmful'] for r in rows)}"
    )


if __name__ == "__main__":
    main()
