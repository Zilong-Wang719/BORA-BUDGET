from __future__ import annotations

import argparse
import csv
import json
import math
import re
import warnings
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from fractions import Fraction
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def normalize_answer(answer: str | None) -> str | None:
    """Mirror src/bora/common.py without importing project dependencies."""
    if answer is None:
        return None
    cleaned = answer.strip()
    if not cleaned:
        return None
    cleaned = cleaned.replace(",", "")
    if "=" in cleaned:
        cleaned = cleaned.split("=")[-1].strip()
    lowered = cleaned.lower()
    if lowered in {"unknown", "n/a", "none"}:
        return None
    whole_hour_time = re.search(r"\b(-?\d{1,2}):00\b", cleaned)
    if whole_hour_time:
        return whole_hour_time.group(1)
    try:
        value = Decimal(cleaned)
        normalized = value.normalize()
        if normalized == normalized.to_integral():
            return str(int(normalized))
        return format(normalized, "f").rstrip("0").rstrip(".")
    except (InvalidOperation, ValueError):
        numeric_matches = re.findall(r"-?\d+(?:\.\d+)?", cleaned)
        if len(numeric_matches) == 1:
            try:
                value = Decimal(numeric_matches[0])
                normalized = value.normalize()
                if normalized == normalized.to_integral():
                    return str(int(normalized))
                return format(normalized, "f").rstrip("0").rstrip(".")
            except (InvalidOperation, ValueError):
                pass
        collapsed = " ".join(lowered.split())
        return collapsed or None


def is_correct(prediction: str | None, gold: str | None) -> bool:
    pred_norm = normalize_answer(prediction)
    gold_norm = normalize_answer(gold)
    return pred_norm is not None and pred_norm == gold_norm


ARM_LABELS = ["seed", "think2", "think4", "think6", "think8", "think10", "think12"]
MATH_VERIFY_PARSE = None
MATH_VERIFY_VERIFY = None


@dataclass(frozen=True)
class ParsedAnswer:
    kind: str
    value: Any


def _read_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if path.suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]
    if isinstance(payload, dict):
        for key in ("records", "rows"):
            value = payload.get(key)
            if isinstance(value, list):
                return [row for row in value if isinstance(row, dict)]
        for value in payload.values():
            if isinstance(value, dict):
                for key in ("records", "rows"):
                    nested = value.get(key)
                    if isinstance(nested, list):
                        return [row for row in nested if isinstance(row, dict)]
    raise ValueError(f"Could not infer records from {path}")


def _balanced_brace_content(text: str, start: int) -> tuple[str, int] | None:
    if start >= len(text) or text[start] != "{":
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


def extract_boxed_balanced(text: str | None) -> str | None:
    if not text:
        return None
    marker = r"\boxed"
    idx = str(text).rfind(marker)
    if idx < 0:
        return None
    brace = str(text).find("{", idx + len(marker))
    if brace < 0:
        return None
    result = _balanced_brace_content(str(text), brace)
    if result is None:
        return None
    return result[0].strip()


def _strip_outer_math(text: str) -> str:
    s = text.strip()
    if s.startswith("$") and s.endswith("$") and len(s) >= 2:
        s = s[1:-1].strip()
    if s.startswith(r"\(") and s.endswith(r"\)"):
        s = s[2:-2].strip()
    if s.startswith(r"\[") and s.endswith(r"\]"):
        s = s[2:-2].strip()
    return s


def _remove_latex_grouping_commas(text: str) -> str:
    s = text
    s = re.sub(r",\\!\s*", "", s)
    s = re.sub(r"(?<=\d),(?=\d{3}(?:\D|$))", "", s)
    return s


def _replace_balanced_command(text: str, command: str, repl) -> str:
    s = text
    marker = "\\" + command
    pos = 0
    while True:
        idx = s.find(marker, pos)
        if idx < 0:
            return s
        brace = s.find("{", idx + len(marker))
        if brace < 0:
            pos = idx + len(marker)
            continue
        first = _balanced_brace_content(s, brace)
        if first is None:
            pos = idx + len(marker)
            continue
        if command in {"frac", "dfrac", "tfrac"}:
            second_start = first[1]
            while second_start < len(s) and s[second_start].isspace():
                second_start += 1
            second = _balanced_brace_content(s, second_start)
            if second is None:
                pos = first[1]
                continue
            new = repl(first[0], second[0])
            s = s[:idx] + new + s[second[1] :]
            pos = idx + len(new)
        else:
            new = repl(first[0])
            s = s[:idx] + new + s[first[1] :]
            pos = idx + len(new)


def canonical_text(text: Any) -> str | None:
    if text is None:
        return None
    s = str(text).strip()
    if not s:
        return None
    boxed = extract_boxed_balanced(s)
    if boxed:
        s = boxed
    s = _strip_outer_math(s)
    s = s.replace("\u2212", "-")
    s = s.replace("：", ":")
    # Remove thousands/grouping commas, but preserve commas that separate
    # answer components such as "1,-2" or "(3, pi/2)".
    s = _remove_latex_grouping_commas(s)
    s = re.sub(r"\\(?:left|right|big|Big|bigg|Bigg)", "", s)
    s = re.sub(r"\\(?:,|;|!| )", "", s)
    s = s.replace(r"\cdot", "*").replace(r"\times", "*")
    s = s.replace(r"\div", "/")
    s = s.replace(r"\pi", "pi")
    s = s.replace(r"^\circ", "").replace(r"^{\circ}", "")
    s = s.replace("°", "")
    s = _replace_balanced_command(s, "frac", lambda a, b: f"(({a})/({b}))")
    s = _replace_balanced_command(s, "dfrac", lambda a, b: f"(({a})/({b}))")
    s = _replace_balanced_command(s, "tfrac", lambda a, b: f"(({a})/({b}))")
    s = _replace_balanced_command(s, "sqrt", lambda a: f"sqrt({a})")
    s = re.sub(r"\s+", "", s)
    s = s.strip()
    return s or None


def _top_level_split(s: str, sep: str = ",") -> list[str]:
    parts: list[str] = []
    depth = 0
    start = 0
    pairs = {"(": ")", "[": "]", "{": "}"}
    closers = set(pairs.values())
    for idx, ch in enumerate(s):
        if ch in pairs:
            depth += 1
        elif ch in closers:
            depth = max(0, depth - 1)
        elif ch == sep and depth == 0:
            parts.append(s[start:idx])
            start = idx + 1
    parts.append(s[start:])
    return [part for part in parts if part != ""]


def _strip_outer_pair(s: str) -> tuple[str, str | None]:
    pairs = {"(": ")", "[": "]", "{": "}"}
    if len(s) >= 2 and s[0] in pairs and s[-1] == pairs[s[0]]:
        depth = 0
        for idx, ch in enumerate(s):
            if ch == s[0]:
                depth += 1
            elif ch == pairs[s[0]]:
                depth -= 1
                if depth == 0 and idx != len(s) - 1:
                    return s, None
        return s[1:-1], s[0]
    return s, None


def _decimal_fraction(s: str) -> Fraction | None:
    try:
        return Fraction(Decimal(s))
    except (InvalidOperation, ValueError, ZeroDivisionError):
        return None


def _safe_numeric_eval(s: str) -> float | None:
    allowed = set("0123456789+-*/().pieE ")
    if not set(s) <= allowed:
        return None
    if re.search(r"[A-DF-Za-df-mo-z]", s.replace("pi", "")):
        return None
    expr = s.replace("^", "**")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            value = eval(expr, {"__builtins__": {}}, {"pi": math.pi, "e": math.e})
    except Exception:
        return None
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return None


def _fraction_from_expr(s: str) -> Fraction | None:
    plain = s
    while plain.startswith("(") and plain.endswith(")"):
        inner, paired = _strip_outer_pair(plain)
        if paired is None:
            break
        plain = inner
    if re.fullmatch(r"[+-]?\d+", plain):
        return Fraction(int(plain), 1)
    frac = _decimal_fraction(plain)
    if frac is not None:
        return frac
    if re.fullmatch(r"[+-]?\d+/\(?[+-]?\d+\)?", plain):
        a, b = plain.replace("(", "").replace(")", "").split("/", 1)
        try:
            return Fraction(int(a), int(b))
        except ZeroDivisionError:
            return None
    return None


def parse_answer(text: Any) -> ParsedAnswer:
    s = canonical_text(text)
    if s is None:
        return ParsedAnswer("none", None)
    lowered = s.lower()
    # Normalize common final-answer prose remnants.
    lowered = re.sub(r"^(?:finalanswer:?|answer:?|theansweris)", "", lowered)
    lowered = lowered.strip(".")
    inner, opener = _strip_outer_pair(lowered)
    if opener in {"(", "["}:
        parts = _top_level_split(inner)
        if len(parts) > 1:
            return ParsedAnswer("tuple", tuple(parse_answer(part).value for part in parts))
    if opener == "{":
        parts = _top_level_split(inner)
        if len(parts) > 1:
            return ParsedAnswer("set", frozenset(parse_answer(part).value for part in parts))
    frac = _fraction_from_expr(lowered)
    if frac is not None:
        return ParsedAnswer("number", frac)
    numeric = _safe_numeric_eval(lowered)
    if numeric is not None:
        return ParsedAnswer("float", numeric)
    # Canonical symbolic text. Remove harmless braces and normalize subtraction spacing.
    sym = lowered.replace("{", "").replace("}", "")
    sym = sym.replace("**", "^")
    sym = re.sub(r"\\[a-zA-Z]+", "", sym)
    return ParsedAnswer("symbol", sym)


def _values_equal(a: Any, b: Any) -> bool:
    if isinstance(a, Fraction) and isinstance(b, Fraction):
        return a == b
    if isinstance(a, Fraction) and isinstance(b, (int, float)):
        return abs(float(a) - float(b)) <= 1e-8
    if isinstance(b, Fraction) and isinstance(a, (int, float)):
        return abs(float(a) - float(b)) <= 1e-8
    if isinstance(a, float) and isinstance(b, float):
        return abs(a - b) <= 1e-8 * max(1.0, abs(a), abs(b))
    if isinstance(a, tuple) and isinstance(b, tuple) and len(a) == len(b):
        return all(_values_equal(x, y) for x, y in zip(a, b))
    if isinstance(a, frozenset) and isinstance(b, frozenset):
        if len(a) != len(b):
            return False
        unmatched = list(b)
        for x in a:
            for idx, y in enumerate(unmatched):
                if _values_equal(x, y):
                    unmatched.pop(idx)
                    break
            else:
                return False
        return True
    return a == b


def audit_equivalent(prediction: Any, gold: Any) -> bool:
    # Keep the project evaluator as a subset so existing exact string matches survive.
    if is_correct(str(prediction) if prediction is not None else None, str(gold) if gold is not None else None):
        return True
    pred = parse_answer(prediction)
    target = parse_answer(gold)
    if pred.kind == "none" or target.kind == "none":
        return False
    return _values_equal(pred.value, target.value)


def _load_math_verify() -> bool:
    global MATH_VERIFY_PARSE, MATH_VERIFY_VERIFY
    if MATH_VERIFY_PARSE is not None and MATH_VERIFY_VERIFY is not None:
        return True
    try:
        from math_verify import parse, verify  # type: ignore
    except Exception:
        return False
    MATH_VERIFY_PARSE = parse
    MATH_VERIFY_VERIFY = verify
    return True


def _math_verify_preclean(text: Any) -> str | None:
    if text is None:
        return None
    s = str(text).strip()
    if not s:
        return None
    boxed = extract_boxed_balanced(s)
    if boxed:
        s = boxed
    s = _strip_outer_math(s)
    s = s.replace("\u2212", "-")
    s = _remove_latex_grouping_commas(s)
    s = re.sub(r"\\(?:left|right|big|Big|bigg|Bigg)", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s or None


def _strip_text_commands(text: str) -> str:
    s = text
    for command in ("mbox", "text", "mathrm"):
        marker = "\\" + command
        pos = 0
        while True:
            idx = s.find(marker, pos)
            if idx < 0:
                break
            brace = s.find("{", idx + len(marker))
            if brace < 0:
                pos = idx + len(marker)
                continue
            content = _balanced_brace_content(s, brace)
            if content is None:
                pos = idx + len(marker)
                continue
            s = s[:idx] + s[content[1] :]
            pos = idx
    return s


def _safe_numeric_domain_for_math_verify(gold: Any) -> bool:
    """Avoid math_verify false positives on symbolic MATH answers.

    math_verify's parser is intentionally permissive and may extract a scalar
    from a symbolic expression (e.g. comparing ``5r^5`` with ``5``). For this
    audit we only trust it on scalar numeric-style answers.
    """
    s = _math_verify_preclean(gold)
    if s is None:
        return False
    lowered = s.lower()
    if any(token in lowered for token in ("pmatrix", "matrix", r"\begin", r"\end", "&", r"\\")):
        return False
    if r"\pm" in lowered or "±" in lowered:
        return False
    if re.search(r"_\{?\d", lowered):
        return False
    if re.search(r"(?<![a-zA-Z])i(?![a-zA-Z])", lowered):
        return False
    scrub = _strip_text_commands(lowered)
    scrub = re.sub(r"\\(?:frac|dfrac|tfrac|sqrt|pi|circ|left|right|cdot|times|div)\b", "", scrub)
    scrub = scrub.replace("pi", "")
    scrub = re.sub(r"\\[a-zA-Z]+", "", scrub)
    return not bool(re.search(r"[a-zA-Z]", scrub))


def _plain_integer_literal(text: Any) -> bool:
    s = _math_verify_preclean(text)
    if s is None:
        return False
    return bool(re.fullmatch(r"[+-]?\d+", s.strip()))


def _has_top_level_separator(text: Any) -> bool:
    s = canonical_text(text)
    if s is None:
        return False
    inner, opener = _strip_outer_pair(s.lower())
    candidate = inner if opener in {"(", "[", "{"} else s.lower()
    return len(_top_level_split(candidate)) > 1


def math_verify_equivalent(prediction: Any, gold: Any) -> bool | None:
    """Use math_verify with light pre-cleaning and a structure guard.

    Returns None when math_verify is unavailable. The guard avoids a known
    pitfall where tuple/list answers such as ``1,-2`` may be parsed as the
    first scalar only.
    """
    if not _load_math_verify():
        return None
    if is_correct(str(prediction) if prediction is not None else None, str(gold) if gold is not None else None):
        return True
    if _has_top_level_separator(gold) or _has_top_level_separator(prediction):
        # Use the local composite-aware audit rather than letting math_verify
        # silently compare only the first scalar in a tuple/list answer.
        return audit_equivalent(prediction, gold)
    if not _safe_numeric_domain_for_math_verify(gold):
        return audit_equivalent(prediction, gold)
    gold_text_for_guard = _math_verify_preclean(gold) or ""
    pred_text_for_guard = _math_verify_preclean(prediction) or ""
    if (
        (r"\sqrt" in gold_text_for_guard or r"\pi" in gold_text_for_guard or "pi" in gold_text_for_guard)
        and not (r"\sqrt" in pred_text_for_guard or r"\pi" in pred_text_for_guard or "pi" in pred_text_for_guard)
        and _plain_integer_literal(prediction)
    ):
        return audit_equivalent(prediction, gold)
    pred_text = _math_verify_preclean(prediction)
    gold_text = _math_verify_preclean(gold)
    if pred_text is None or gold_text is None:
        return False
    try:
        gold_parsed = MATH_VERIFY_PARSE(gold_text, raise_on_error=False)  # type: ignore[misc]
        pred_parsed = MATH_VERIFY_PARSE(pred_text, raise_on_error=False)  # type: ignore[misc]
        return bool(MATH_VERIFY_VERIFY(gold_parsed, pred_parsed, raise_on_error=False))  # type: ignore[misc]
    except Exception:
        return False


def _complexity_flags(gold: Any) -> list[str]:
    s = str(gold or "")
    flags: list[str] = []
    checks = [
        ("frac", r"\\frac|/"),
        ("sqrt", r"\\sqrt|sqrt"),
        ("pi", r"\\pi|\bpi\b"),
        ("tuple", r"[()]"),
        ("set", r"[{}]"),
        ("symbol", r"[a-zA-Z]"),
        ("degree", r"\\circ|°"),
        ("comma", r","),
    ]
    for name, pattern in checks:
        if re.search(pattern, s):
            flags.append(name)
    return flags


def _answer_for_label(row: dict[str, Any], label: str) -> Any:
    if label == "seed":
        return row.get("seed_answer")
    return row.get(f"{label}_answer")


def _final_answer_for_label(row: dict[str, Any], label: str) -> Any:
    if label == "seed":
        return row.get("seed_answer")
    return row.get(f"{label}_answer") if row.get(f"{label}_gate_pass") else row.get("seed_answer")


def _audit_rows(
    rows: list[dict[str, Any]],
    labels: list[str],
    *,
    use_math_verify: bool = False,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    summary: list[dict[str, Any]] = []
    mismatches: list[dict[str, Any]] = []
    math_verify_mismatches: list[dict[str, Any]] = []
    for mode in ("raw", "final"):
        for label in labels:
            if label != "seed" and not any(f"{label}_answer" in row for row in rows):
                continue
            n = 0
            strict_correct = audit_correct = false_neg = false_pos = 0
            math_verify_correct = math_verify_false_neg = math_verify_false_pos = 0
            stored_disagree = 0
            complex_count = 0
            flag_counts: dict[str, int] = {}
            for row in rows:
                gold = row.get("gold_answer") or row.get("answer")
                if gold is None:
                    continue
                answer = _answer_for_label(row, label) if mode == "raw" else _final_answer_for_label(row, label)
                strict = is_correct(str(answer) if answer is not None else None, str(gold))
                audit = audit_equivalent(answer, gold)
                math_ok = math_verify_equivalent(answer, gold) if use_math_verify else None
                stored = None
                if label == "seed":
                    stored = row.get("seed_correct")
                elif mode == "raw":
                    stored = row.get(f"{label}_raw_correct")
                else:
                    stored = row.get(f"{label}_final_correct")
                if stored is not None and bool(stored) != strict:
                    stored_disagree += 1
                flags = _complexity_flags(gold)
                if flags:
                    complex_count += 1
                for flag in flags:
                    flag_counts[flag] = flag_counts.get(flag, 0) + 1
                n += 1
                strict_correct += int(strict)
                audit_correct += int(audit)
                false_neg += int((not strict) and audit)
                false_pos += int(strict and (not audit))
                if math_ok is not None:
                    math_verify_correct += int(math_ok)
                    math_verify_false_neg += int((not strict) and math_ok)
                    math_verify_false_pos += int(strict and (not math_ok))
                if strict != audit:
                    mismatches.append(
                        {
                            "qid": row.get("qid"),
                            "seed": row.get("seed"),
                            "mode": mode,
                            "label": label,
                            "gold_answer": gold,
                            "answer": answer,
                            "strict_correct": strict,
                            "audit_correct": audit,
                            "strict_norm_answer": normalize_answer(str(answer) if answer is not None else None),
                            "strict_norm_gold": normalize_answer(str(gold)),
                            "parsed_answer": repr(parse_answer(answer).value),
                            "parsed_gold": repr(parse_answer(gold).value),
                            "gold_flags": flags,
                        }
                    )
                if math_ok is not None and strict != math_ok:
                    math_verify_mismatches.append(
                        {
                            "qid": row.get("qid"),
                            "seed": row.get("seed"),
                            "mode": mode,
                            "label": label,
                            "gold_answer": gold,
                            "answer": answer,
                            "strict_correct": strict,
                            "math_verify_correct": math_ok,
                            "strict_norm_answer": normalize_answer(str(answer) if answer is not None else None),
                            "strict_norm_gold": normalize_answer(str(gold)),
                            "math_verify_pred_text": _math_verify_preclean(answer),
                            "math_verify_gold_text": _math_verify_preclean(gold),
                            "gold_flags": flags,
                        }
                    )
            if n:
                summary.append(
                    {
                        "mode": mode,
                        "label": label,
                        "n": n,
                        "strict_correct": strict_correct,
                        "strict_accuracy": strict_correct / n,
                        "audit_correct": audit_correct,
                        "audit_accuracy": audit_correct / n,
                        "audit_minus_strict": audit_correct - strict_correct,
                        "false_negative_strict": false_neg,
                        "false_positive_strict": false_pos,
                        "math_verify_correct": math_verify_correct if use_math_verify else "",
                        "math_verify_accuracy": (math_verify_correct / n) if use_math_verify else "",
                        "math_verify_minus_strict": (math_verify_correct - strict_correct) if use_math_verify else "",
                        "math_verify_false_negative_strict": math_verify_false_neg if use_math_verify else "",
                        "math_verify_false_positive_strict": math_verify_false_pos if use_math_verify else "",
                        "stored_vs_recomputed_disagree": stored_disagree,
                        "complex_gold_count": complex_count,
                        **{f"gold_{k}_count": v for k, v in sorted(flag_counts.items())},
                    }
                )
    return summary, mismatches, math_verify_mismatches


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_examples(path: Path, mismatches: list[dict[str, Any]], *, limit: int) -> None:
    lines = ["# Math Answer Evaluation Audit Examples", ""]
    for row in mismatches[:limit]:
        lines.extend(
            [
                f"## {row.get('qid')} seed={row.get('seed')} {row.get('mode')}:{row.get('label')}",
                "",
                f"- gold: `{row.get('gold_answer')}`",
                f"- answer: `{row.get('answer')}`",
                f"- strict: `{row.get('strict_correct')}`; audit: `{row.get('audit_correct')}`",
                f"- strict norm: pred=`{row.get('strict_norm_answer')}`, gold=`{row.get('strict_norm_gold')}`",
                f"- parsed: pred=`{row.get('parsed_answer')}`, gold=`{row.get('parsed_gold')}`",
                f"- flags: `{','.join(row.get('gold_flags') or [])}`",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit strict normalized EM against a stronger lightweight math-equivalence evaluator.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("artifacts/remote_stage_main/independent_checkers_20260517/karmed_arm_rows_with_checkers.jsonl"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/eval_audit/math500_answer_eval"))
    parser.add_argument("--labels", nargs="*", default=ARM_LABELS)
    parser.add_argument("--example-limit", type=int, default=80)
    parser.add_argument("--use-math-verify", action="store_true")
    args = parser.parse_args()

    rows = _read_rows(args.input)
    if args.use_math_verify and not _load_math_verify():
        raise SystemExit("math_verify is not installed in this Python environment.")
    summary, mismatches, math_verify_mismatches = _audit_rows(
        rows,
        args.labels,
        use_math_verify=args.use_math_verify,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output_dir / "summary.csv", summary)
    _write_jsonl(args.output_dir / "mismatches.jsonl", mismatches)
    if args.use_math_verify:
        _write_jsonl(args.output_dir / "math_verify_mismatches.jsonl", math_verify_mismatches)
    _write_examples(args.output_dir / "mismatch_examples.md", mismatches, limit=args.example_limit)
    (args.output_dir / "summary.json").write_text(
        json.dumps(
            {
                "input": str(args.input),
                "rows": len(rows),
                "labels": args.labels,
                "summary": summary,
                "mismatch_count": len(mismatches),
                "math_verify_mismatch_count": len(math_verify_mismatches) if args.use_math_verify else None,
                "note": (
                    "This audit compares current normalized exact match with a lightweight "
                    "zero-dependency math-equivalence heuristic. It can identify likely strict-EM "
                    "misjudgments but is not a formal symbolic verifier."
                ),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"wrote {args.output_dir}")
    for row in summary:
        if row["mode"] == "raw" and row["label"] in {"seed", "think8", "think12"}:
            print(
                f"{row['mode']}:{row['label']} strict={row['strict_correct']}/{row['n']} "
                f"audit={row['audit_correct']}/{row['n']} "
                f"delta={row['audit_minus_strict']} fn={row['false_negative_strict']} fp={row['false_positive_strict']}"
                + (
                    f" math_verify={row['math_verify_correct']}/{row['n']} "
                    f"mv_delta={row['math_verify_minus_strict']}"
                    if args.use_math_verify
                    else ""
                )
            )


if __name__ == "__main__":
    main()
