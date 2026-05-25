from __future__ import annotations

import argparse
import ast
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bora.common import dump_json
from bora.eval import summarize_records


SAFE_IMPORTS = """
from __future__ import annotations
from typing import *
import collections
import functools
import itertools
import math
import operator
import random
import re
import statistics
import string
import heapq
import bisect
"""

DANGEROUS_SNIPPETS = [
    "import os",
    "from os",
    "subprocess",
    "socket",
    "shutil",
    "pathlib",
    "__import__",
    "eval(",
    "exec(",
    "open(",
    "input(",
    "compile(",
]


def _load_payload(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _records(payload: dict[str, Any]) -> list[dict[str, Any]]:
    if "standard_direct_cot" in payload:
        return list(payload["standard_direct_cot"].get("records") or [])
    if "records" in payload:
        return list(payload.get("records") or [])
    raise KeyError("Expected payload with standard_direct_cot.records or records.")


def _tests(record: dict[str, Any]) -> list[str]:
    value = (record.get("metadata") or {}).get("tests") or []
    if isinstance(value, str):
        return [line.strip() for line in value.splitlines() if line.strip()]
    return [str(item).strip() for item in value if str(item).strip()]


def _imports(record: dict[str, Any]) -> list[str]:
    value = (record.get("metadata") or {}).get("test_imports") or []
    if isinstance(value, str):
        return [line.strip() for line in value.splitlines() if line.strip()]
    return [str(item).strip() for item in value if str(item).strip()]


def _blocked(code: str) -> str | None:
    lowered = code.lower()
    for snippet in DANGEROUS_SNIPPETS:
        if snippet in lowered:
            return snippet
    return None


def _syntax_ok(code: str) -> tuple[bool, str | None]:
    try:
        ast.parse(code or "")
        return True, None
    except SyntaxError as exc:
        return False, f"{exc.__class__.__name__}: {exc}"


def _run_one(record: dict[str, Any], timeout: float) -> tuple[bool, dict[str, Any]]:
    code = str(record.get("prediction") or "")
    tests = _tests(record)
    imports = _imports(record)
    syntax_ok, syntax_error = _syntax_ok(code)
    if not code.strip():
        return False, {"error_type": "empty_code", "syntax_ok": False, "error": "empty code"}
    if not syntax_ok:
        return False, {"error_type": "syntax_error", "syntax_ok": False, "error": syntax_error}
    blocked = _blocked(code)
    if blocked is not None:
        return False, {"error_type": "blocked_snippet", "syntax_ok": True, "error": blocked}
    if not tests:
        return False, {"error_type": "missing_tests", "syntax_ok": True, "error": "no tests"}

    script = "\n".join(
        [
            SAFE_IMPORTS,
            "\n".join(imports),
            code,
            "",
            "def __bora_run_tests__():",
            *[f"    {test}" for test in tests],
            "",
            "__bora_run_tests__()",
            "",
        ]
    )
    with tempfile.TemporaryDirectory(prefix="bora_code_eval_") as tmp:
        path = Path(tmp) / "candidate.py"
        path.write_text(script, encoding="utf-8")
        try:
            completed = subprocess.run(
                [sys.executable, str(path)],
                cwd=tmp,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=timeout,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            return False, {
                "error_type": "timeout",
                "syntax_ok": True,
                "error": str(exc),
            }
    passed = completed.returncode == 0
    return passed, {
        "error_type": None if passed else "runtime_or_assertion",
        "syntax_ok": True,
        "returncode": completed.returncode,
        "stdout_tail": completed.stdout[-1000:],
        "stderr_tail": completed.stderr[-1000:],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate generated Python code against MBPP tests.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--timeout", type=float, default=5.0)
    args = parser.parse_args()

    payload = _load_payload(args.input)
    records = _records(payload)
    evaluated: list[dict[str, Any]] = []
    for idx, record in enumerate(records, start=1):
        passed, info = _run_one(record, timeout=float(args.timeout))
        out = dict(record)
        metadata = dict(out.get("metadata") or {})
        metadata["code_eval"] = info
        metadata["code_eval_passed"] = passed
        out["metadata"] = metadata
        out["correct"] = bool(passed)
        evaluated.append(out)
        if idx % 25 == 0:
            print(f"[{idx}/{len(records)}] pass_rate={sum(r['correct'] for r in evaluated)/idx:.3f}", flush=True)

    result = {"standard_direct_cot": {"summary": summarize_records(evaluated), "records": evaluated}}
    dump_json(args.output, result)
    print(f"wrote evaluated code results to {args.output}")
    print(result["standard_direct_cot"]["summary"])


if __name__ == "__main__":
    main()
