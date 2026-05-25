from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
import warnings
from collections import Counter
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any


warnings.filterwarnings("ignore", category=SyntaxWarning)

try:
    from math_verify import ExprExtractionConfig, LatexExtractionConfig, StringExtractionConfig, parse, verify
except Exception:  # pragma: no cover - optional dependency
    ExprExtractionConfig = LatexExtractionConfig = StringExtractionConfig = None
    parse = verify = None

try:
    import sympy as sp
    from sympy.parsing.sympy_parser import (
        convert_xor,
        implicit_multiplication_application,
        parse_expr,
        standard_transformations,
    )
except Exception:  # pragma: no cover - optional dependency
    sp = None
    parse_expr = None
    standard_transformations = ()
    implicit_multiplication_application = None
    convert_xor = None


LATEX_HINT_RE = re.compile(r"\\[a-zA-Z]+|[_^{}]")
NUM_RE = re.compile(r"-?(?:\d+(?:\.\d*)?|\.\d+)")
FULL_NUM_RE = re.compile(r"^-?(?:\d+(?:\.\d*)?|\.\d+)$")
SIMPLE_FRACTION_RE = re.compile(r"^\s*(-?\d+)\s*/\s*(-?\d+)\s*$")


def load_records(path: Path, method: str | None) -> list[dict[str, Any]]:
    if path.suffix == ".jsonl":
        rows = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return payload
    if isinstance(payload.get("records"), list):
        return payload["records"]
    if method:
        block = payload.get(method)
        if isinstance(block, dict) and isinstance(block.get("records"), list):
            return block["records"]
    candidates = [
        value.get("records")
        for value in payload.values()
        if isinstance(value, dict) and isinstance(value.get("records"), list)
    ]
    if len(candidates) == 1:
        return candidates[0]
    raise ValueError(f"Could not find records in {path}; pass --method if needed.")


def _strip_answer_prefix(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^(final\s+answer|answer|答案)\s*[:：]\s*", "", text, flags=re.I)
    text = text.strip().rstrip(".。")
    return text.strip()


def _unbox(text: str) -> str:
    # Handles one-level boxed answers, which is enough for benchmark outputs.
    match = re.search(r"\\boxed\{([^{}]+)\}", text)
    if match:
        return match.group(1).strip()
    return text


def clean_text(text: Any) -> str:
    text = "" if text is None else str(text)
    text = text.strip()
    text = _unbox(text)
    text = _strip_answer_prefix(text)
    replacements = {
        "\\left": "",
        "\\right": "",
        "\\displaystyle": "",
        "\\!": "",
        "\\,": " ",
        "\\;": " ",
        "\\:": " ",
        "\\leqslant": "<=",
        "\\leq": "<=",
        "\\geqslant": ">=",
        "\\geq": ">=",
        "≤": "<=",
        "≥": ">=",
        "−": "-",
        "–": "-",
        "—": "-",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def canonical_text(text: Any) -> str:
    text = clean_text(text).lower()
    text = text.strip("$ ")
    text = re.sub(r"\\text\{([^{}]+)\}", r"\1", text)
    text = text.replace("\\text", "")
    text = re.sub(r"\s+", " ", text)
    text = text.strip(" .。{}")
    return text


def compact_text(text: Any) -> str:
    return re.sub(r"\s+", "", canonical_text(text))


def decimal_value(text: Any) -> Decimal | None:
    s = clean_text(text).replace(",", "").strip()
    if not s:
        return None
    match = SIMPLE_FRACTION_RE.fullmatch(s)
    if match:
        den = Decimal(match.group(2))
        if den == 0:
            return None
        return Decimal(match.group(1)) / den
    if "=" in s and len(s.split("=")) == 2:
        s = s.split("=")[-1].strip()
    if not FULL_NUM_RE.fullmatch(s):
        nums = NUM_RE.findall(s)
        if len(nums) == 1 and len(s) <= 32 and re.sub(NUM_RE, "", s).strip() == "":
            s = nums[0]
        else:
            return None
    try:
        return Decimal(s)
    except (InvalidOperation, ValueError):
        return None


def numeric_equal(pred: Any, gold: Any, *, abs_tol: Decimal = Decimal("1e-6")) -> bool:
    p = decimal_value(pred)
    g = decimal_value(gold)
    if p is None or g is None:
        return False
    return abs(p - g) <= abs_tol


def split_top_level(text: str) -> list[str]:
    items = []
    buf = []
    depth = 0
    for ch in text:
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth = max(0, depth - 1)
        if ch == "," and depth == 0:
            item = "".join(buf).strip()
            if item:
                items.append(item)
            buf = []
            continue
        buf.append(ch)
    item = "".join(buf).strip()
    if item:
        items.append(item)
    return items


def numeric_set_equal(pred: Any, gold: Any) -> bool:
    p_items = split_top_level(clean_text(pred).strip("()[]{}"))
    g_items = split_top_level(clean_text(gold).strip("()[]{}"))
    if len(p_items) <= 1 or len(p_items) != len(g_items):
        return False
    p_vals = [decimal_value(item) for item in p_items]
    g_vals = [decimal_value(item) for item in g_items]
    if any(value is None for value in p_vals + g_vals):
        return False
    p_vals = sorted(p_vals)
    g_vals = sorted(g_vals)
    return all(abs(p - g) <= Decimal("1e-6") for p, g in zip(p_vals, g_vals))


def _math_verify_variants(text: Any) -> list[str]:
    cleaned = clean_text(text)
    variants = [cleaned]
    if LATEX_HINT_RE.search(cleaned) and not re.search(r"\$|\\\(|\\\[|\\boxed", cleaned):
        variants.append(f"${cleaned}$")
    if "\\frac" in cleaned and "$" not in cleaned:
        variants.append(f"$\\displaystyle {cleaned}$")
    return [variant for idx, variant in enumerate(variants) if variant and variant not in variants[:idx]]


def math_verify_equal(pred: Any, gold: Any) -> tuple[bool, str]:
    if parse is None or verify is None:
        return False, "math_verify_unavailable"
    configs = [LatexExtractionConfig(), ExprExtractionConfig()]
    string_gold = canonical_text(gold)
    if string_gold in {"all real numbers", "no solution", "none", "dne"} and StringExtractionConfig:
        configs = [LatexExtractionConfig(), ExprExtractionConfig(), StringExtractionConfig(strings=(string_gold,))]
    for gold_variant in _math_verify_variants(gold):
        gold_parsed = parse(gold_variant, extraction_config=configs, fallback_mode="first_match")
        if not gold_parsed:
            continue
        for pred_variant in _math_verify_variants(pred):
            pred_parsed = parse(pred_variant, extraction_config=configs, fallback_mode="first_match")
            if not pred_parsed:
                continue
            try:
                if verify(
                    gold_parsed,
                    pred_parsed,
                    strict=False,
                    allow_set_relation_comp=True,
                    timeout_seconds=5,
                ):
                    return True, "math_verify"
            except Exception:
                continue
    return False, "math_verify_no_match"


def structured_non_numeric_answer(text: Any) -> bool:
    """Return True when a gold answer is not a single numeric value.

    Math-Verify is intentionally aggressive about extracting sub-expressions
    from text. That is useful for full model generations, but unsafe when we
    only have an already-extracted scalar prediction. For example, matching
    prediction ``5`` against gold ``6-5i`` should be rejected instead of letting
    a sub-expression extractor find the ``5`` inside the gold expression.
    """

    if decimal_value(text) is not None:
        return False
    s = clean_text(text)
    return bool(
        LATEX_HINT_RE.search(s)
        or re.search(r"[a-zA-Z]", s)
        or re.search(r"[(),;]|<=|>=|<|>|=", s)
    )


def scalar_prediction_allowed_for_gold(gold: Any) -> bool:
    """Allow scalar predictions for harmless formatting wrappers/units only."""

    s = clean_text(gold).lower()
    if decimal_value(gold) is not None:
        return True
    # Units or notation where the expected mathematical value is still scalar.
    if re.fullmatch(r"-?(?:\d+(?:\.\d*)?|\.\d+)\s*(?:\^?\\?circ|degrees?|degree|cents?|dollars?|\$|%)", s):
        return True
    if re.fullmatch(r"-?(?:\d+(?:\.\d*)?|\.\d+)\s*\\text\{[^{}]+\}", str(gold).strip().lower()):
        return True
    # Disallow scalar extraction from genuinely structured answers.
    if any(token in s for token in ["sqrt", "pi", "infty", "pmatrix", "matrix", "begin", "_"]):
        return False
    if re.search(r"[a-zA-Z]", s) and not re.search(r"\\text\{[^{}]+\}", str(gold)):
        return False
    if re.search(r"[,;]|<=|>=|<|>|=", s):
        return False
    return False


def _latex_to_sympyish(text: str) -> str:
    text = clean_text(text)
    text = text.strip("$ ")
    text = re.sub(r"\\frac\{([^{}]+)\}\{([^{}]+)\}", r"(\1)/(\2)", text)
    text = text.replace("^", "**")
    text = text.replace("{", "(").replace("}", ")")
    text = re.sub(r"\\sqrt\(([^()]+)\)", r"sqrt(\1)", text)
    text = re.sub(r"\\sqrt\{([^{}]+)\}", r"sqrt(\1)", text)
    text = re.sub(r"\\[a-zA-Z]+", "", text)
    return text.strip()


def sympy_equal(pred: Any, gold: Any) -> bool:
    if parse_expr is None or sp is None:
        return False
    p_text = _latex_to_sympyish(str(pred))
    g_text = _latex_to_sympyish(str(gold))
    if not p_text or not g_text:
        return False
    transformations = standard_transformations
    if implicit_multiplication_application is not None:
        transformations += (implicit_multiplication_application,)
    if convert_xor is not None:
        transformations += (convert_xor,)
    try:
        p_expr = parse_expr(p_text, transformations=transformations, evaluate=True)
        g_expr = parse_expr(g_text, transformations=transformations, evaluate=True)
        if sp.simplify(p_expr - g_expr) == 0:
            return True
    except Exception:
        return False
    return False


def robust_equal(pred: Any, gold: Any) -> tuple[bool, str]:
    if pred is None or gold is None:
        return False, "missing"
    if canonical_text(pred) == canonical_text(gold):
        return True, "exact_text"
    if compact_text(pred) == compact_text(gold):
        return True, "exact_compact"
    if numeric_equal(pred, gold):
        return True, "numeric"
    if numeric_set_equal(pred, gold):
        return True, "numeric_set"
    pred_is_numeric = decimal_value(pred) is not None
    if (
        pred_is_numeric
        and structured_non_numeric_answer(gold)
        and not scalar_prediction_allowed_for_gold(gold)
    ):
        return False, "safe_reject_scalar_vs_structured_gold"
    ok, method = math_verify_equal(pred, gold)
    if ok:
        return True, method
    if sympy_equal(pred, gold):
        return True, "sympy"
    return False, method


def get_prediction(row: dict[str, Any], *, text_field: str | None, pred_field: str) -> Any:
    if text_field:
        current: Any = row
        for part in text_field.split("."):
            if not isinstance(current, dict):
                current = None
                break
            current = current.get(part)
        if current:
            # Let math-verify extract from the full text when available. The older
            # vLLM runner only stores a pre-extracted prediction, so this path is
            # mostly for new runs/audits that keep completion_text.
            return current
    return row.get(pred_field)


def main() -> None:
    parser = argparse.ArgumentParser(description="Robust math answer evaluator with math_verify fallback.")
    parser.add_argument("--input", required=True, help="Result JSON/JSONL file.")
    parser.add_argument("--method", default=None, help="Top-level method key for JSON results.")
    parser.add_argument("--prediction-field", default="prediction")
    parser.add_argument("--gold-field", default="gold_answer")
    parser.add_argument("--text-field", default=None, help="Optional nested field containing full model output text.")
    parser.add_argument("--output-jsonl", default=None)
    parser.add_argument("--output-summary", default=None)
    parser.add_argument("--max-examples", type=int, default=None)
    args = parser.parse_args()

    records = load_records(Path(args.input), args.method)
    if args.max_examples is not None:
        records = records[: args.max_examples]

    rows = []
    methods: Counter[str] = Counter()
    old_correct = 0
    new_correct = 0
    flips_pos = 0
    flips_neg = 0
    for row in records:
        pred = get_prediction(row, text_field=args.text_field, pred_field=args.prediction_field)
        gold = row.get(args.gold_field)
        is_ok, method = robust_equal(pred, gold)
        old = bool(row.get("correct"))
        old_correct += int(old)
        new_correct += int(is_ok)
        flips_pos += int((not old) and is_ok)
        flips_neg += int(old and not is_ok)
        methods[method] += 1
        rows.append(
            {
                "qid": row.get("qid"),
                "prediction": pred,
                "gold_answer": gold,
                "old_correct": old,
                "robust_correct": is_ok,
                "match_method": method,
                "old_prediction": row.get(args.prediction_field),
                "total_tokens": row.get("total_tokens") or row.get("solver_tokens"),
            }
        )

    summary = {
        "input": args.input,
        "count": len(records),
        "old_correct": old_correct,
        "old_accuracy": old_correct / max(1, len(records)),
        "robust_correct": new_correct,
        "robust_accuracy": new_correct / max(1, len(records)),
        "old_to_new_correct": flips_pos,
        "old_correct_to_new_wrong": flips_neg,
        "match_methods": dict(methods),
        "math_verify_available": parse is not None,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    if args.output_jsonl:
        out = Path(args.output_jsonl)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    if args.output_summary:
        out = Path(args.output_summary)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
