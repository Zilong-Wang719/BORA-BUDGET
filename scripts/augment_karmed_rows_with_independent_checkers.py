from __future__ import annotations

import argparse
import ast
import json
import math
import operator
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bora.common import load_config, load_problem_split, normalize_answer


_ALLOWED_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
}
_ALLOWED_UNARYOPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}


def _load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def _records(payload: dict[str, Any], method: str | None = None) -> list[dict[str, Any]]:
    if isinstance(payload.get("records"), list):
        return payload["records"]
    if isinstance(payload.get("rows"), list):
        return payload["rows"]
    if method is not None:
        block = payload.get(method)
        if isinstance(block, dict) and isinstance(block.get("records"), list):
            return block["records"]
        raise KeyError(f"Method {method!r} not found in JSON payload.")
    candidates = [
        value.get("records")
        for value in payload.values()
        if isinstance(value, dict) and isinstance(value.get("records"), list)
    ]
    if len(candidates) == 1:
        return candidates[0]
    raise ValueError("Could not infer records block.")


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _seed_maps(items: list[str], method: str | None) -> dict[int, dict[str, dict[str, Any]]]:
    out: dict[int, dict[str, dict[str, Any]]] = {}
    for item in items:
        seed_text, path_text = item.split(":", 1)
        seed = int(seed_text)
        out[seed] = {str(row["qid"]): row for row in _records(_load_json(path_text), method)}
    return out


def _float_answer(value: Any) -> float | None:
    norm = normalize_answer(str(value) if value is not None else None)
    if norm is None:
        return None
    try:
        return float(norm.replace(",", ""))
    except Exception:
        return None


def _sign(value: float | None) -> int:
    if value is None or abs(value) < 1e-12:
        return 0
    return 1 if value > 0 else -1


def _numbers(text: str) -> list[float]:
    out: list[float] = []
    for raw in re.findall(r"(?<![A-Za-z])-?\d+(?:\.\d+)?", text or ""):
        try:
            out.append(float(raw.replace(",", "")))
        except ValueError:
            pass
    return out


def _numeric_tokens(text: str) -> list[str]:
    return re.findall(r"(?<![A-Za-z])-?\d+(?:\.\d+)?", text or "")


def _is_numeric_only(answer: Any) -> bool:
    norm = normalize_answer(str(answer) if answer is not None else None)
    return _float_answer(norm) is not None


def _latex_to_plain(expr: str) -> str:
    expr = expr or ""
    expr = re.sub(r"\\frac\s*\{([^{}]+)\}\s*\{([^{}]+)\}", r"((\1)/(\2))", expr)
    expr = re.sub(r"\\sqrt\s*\{([^{}]+)\}", r"sqrt(\1)", expr)
    expr = expr.replace("\\cdot", "*").replace("\\times", "*").replace("\\div", "/")
    expr = expr.replace("\\left", "").replace("\\right", "")
    expr = expr.replace("{", "(").replace("}", ")")
    expr = expr.replace("^", "**")
    expr = expr.replace("−", "-").replace("–", "-")
    expr = re.sub(r"(?<=\d),(?=\d)", "", expr)
    expr = re.sub(r"\s+", "", expr)
    return expr


def _eval_ast(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _eval_ast(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_BINOPS:
        left = _eval_ast(node.left)
        right = _eval_ast(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > 8:
            raise ValueError("exponent too large")
        return float(_ALLOWED_BINOPS[type(node.op)](left, right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_UNARYOPS:
        return float(_ALLOWED_UNARYOPS[type(node.op)](_eval_ast(node.operand)))
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "sqrt":
        if len(node.args) != 1:
            raise ValueError("sqrt arity")
        value = _eval_ast(node.args[0])
        if value < 0:
            raise ValueError("sqrt negative")
        return math.sqrt(value)
    if isinstance(node, ast.Name) and node.id == "pi":
        return math.pi
    raise ValueError(f"unsupported expression: {ast.dump(node)[:80]}")


def _safe_eval_numeric(expr: str) -> float | None:
    expr = _latex_to_plain(expr)
    if not expr or len(expr) > 120:
        return None
    if re.search(r"[^0-9+\-*/().%pieqsrt]", expr):
        return None
    # Avoid interpreting stray prose fragments as expressions.
    if not re.search(r"\d", expr):
        return None
    try:
        tree = ast.parse(expr, mode="eval")
        value = _eval_ast(tree)
    except Exception:
        return None
    if not math.isfinite(value) or abs(value) > 1e12:
        return None
    return value


def _equation_segments(text: str, *, max_segments: int = 80) -> list[list[str]]:
    segments: list[list[str]] = []
    # Keep this intentionally conservative: only inspect compact math-ish lines.
    for line in (text or "").splitlines():
        if "=" not in line:
            continue
        if len(line) > 260:
            continue
        if not re.search(r"\d", line):
            continue
        cleaned = re.sub(r"\$+", "", line)
        parts = [part.strip(" .,:;") for part in cleaned.split("=")]
        parts = [part for part in parts if part]
        if len(parts) >= 2:
            segments.append(parts[:4])
        if len(segments) >= max_segments:
            break
    return segments


def _arithmetic_checker_features(trace: str) -> dict[str, float]:
    checked = mismatched = parsed_exprs = 0
    max_abs_error = 0.0
    for chain in _equation_segments(trace):
        values: list[float] = []
        for expr in chain:
            value = _safe_eval_numeric(expr)
            if value is not None:
                values.append(value)
        parsed_exprs += len(values)
        if len(values) < 2:
            continue
        for left, right in zip(values, values[1:]):
            checked += 1
            abs_error = abs(left - right)
            max_abs_error = max(max_abs_error, abs_error)
            tol = max(1e-6, 1e-6 * max(abs(left), abs(right), 1.0))
            if abs_error > tol:
                mismatched += 1
    rate = mismatched / checked if checked else 0.0
    return {
        "checker_arith_equation_checks": float(checked),
        "checker_arith_parsed_exprs": float(parsed_exprs),
        "checker_arith_mismatch_count": float(mismatched),
        "checker_arith_mismatch_rate": float(rate),
        "checker_arith_has_mismatch": 1.0 if mismatched else 0.0,
        "checker_arith_max_abs_error_log1p": math.log1p(max_abs_error),
    }


def _format_checker_features(question: str, seed_answer: Any, trace: str, seed_record: dict[str, Any] | None) -> dict[str, float]:
    q = (question or "").lower()
    answer_text = str(seed_answer or "")
    answer_norm = normalize_answer(answer_text)
    seed_value = _float_answer(answer_text)
    answer_numeric_only = _is_numeric_only(answer_text)
    answer_numbers = _numbers(answer_text)
    question_numbers = _numbers(question)
    question_number_set = {round(value, 12) for value in question_numbers}

    wants_tuple = bool(re.search(r"coordinate|ordered pair|polar|cartesian|point|\\left\s*\(", q))
    wants_set_or_all = bool(re.search(r"find all|all values|solutions|roots|set of|ordered pairs|triples", q))
    wants_expression = bool(re.search(r"in terms of|simplify|expression|formula|polynomial|function", q))
    wants_fraction = bool(re.search(r"fraction|ratio|probability|percent|\\frac|rational", q))
    wants_angle_pi = bool(re.search(r"\\pi|pi|radian|angle|polar", q))
    wants_modular = bool(re.search(r"modulo|mod |remainder|congruent|divisible", q))
    wants_geometry = bool(re.search(r"triangle|circle|angle|radius|diameter|area|perimeter|volume|geometry", q))

    has_tuple_answer = any(token in answer_text for token in ["(", ")", ",", "\\left", "\\right"])
    has_pi_answer = bool(re.search(r"\\pi|pi", answer_text.lower()))
    has_fraction_answer = "/" in answer_text or "\\frac" in answer_text
    has_set_answer = any(token in answer_text for token in ["{", "}", "[", "]"]) or "," in answer_text

    seed_num_rounded = round(seed_value, 12) if seed_value is not None else None
    seed_copies_problem_number = seed_num_rounded in question_number_set if seed_num_rounded is not None else False
    small_integer = seed_value is not None and abs(seed_value - round(seed_value)) < 1e-9 and abs(seed_value) <= 3
    many_question_nums = len(question_numbers) >= 6

    metadata = (seed_record or {}).get("metadata") or {}
    completion_tokens = float(metadata.get("completion_tokens") or (seed_record or {}).get("solver_tokens") or 0.0)
    max_new_tokens = float(metadata.get("max_new_tokens") or 0.0)
    near_cap = max_new_tokens > 0 and completion_tokens >= max_new_tokens * 0.95
    completion_head = str(metadata.get("completion_head") or trace or "")
    trace_ends_incomplete = bool(re.search(r"(but|therefore|so|because|then|hence|thus|=|\\+|-|,)\s*$", completion_head.strip(), flags=re.I))
    explicit_answer_present = bool(re.search(r"final answer|answer\s*(is|:)|\\boxed", completion_head, flags=re.I))

    mismatch_count = 0
    mismatch_count += int(wants_tuple and answer_numeric_only and not has_tuple_answer)
    mismatch_count += int(wants_angle_pi and not has_pi_answer and answer_numeric_only)
    mismatch_count += int(wants_fraction and answer_numeric_only and not has_fraction_answer)
    mismatch_count += int(wants_set_or_all and answer_numeric_only and not has_set_answer)
    mismatch_count += int(wants_expression and answer_numeric_only)

    return {
        "checker_answer_numeric_only": 1.0 if answer_numeric_only else 0.0,
        "checker_answer_number_count": float(len(answer_numbers)),
        "checker_seed_answer_sign": float(_sign(seed_value)),
        "checker_seed_answer_abs_log1p": math.log1p(abs(seed_value)) if seed_value is not None else 0.0,
        "checker_question_number_count": float(len(question_numbers)),
        "checker_many_question_numbers": 1.0 if many_question_nums else 0.0,
        "checker_seed_copies_problem_number": 1.0 if seed_copies_problem_number else 0.0,
        "checker_small_integer_answer": 1.0 if small_integer else 0.0,
        "checker_wants_tuple": 1.0 if wants_tuple else 0.0,
        "checker_wants_set_or_all": 1.0 if wants_set_or_all else 0.0,
        "checker_wants_expression": 1.0 if wants_expression else 0.0,
        "checker_wants_fraction": 1.0 if wants_fraction else 0.0,
        "checker_wants_angle_pi": 1.0 if wants_angle_pi else 0.0,
        "checker_wants_modular": 1.0 if wants_modular else 0.0,
        "checker_wants_geometry": 1.0 if wants_geometry else 0.0,
        "checker_tuple_mismatch": 1.0 if wants_tuple and answer_numeric_only and not has_tuple_answer else 0.0,
        "checker_pi_mismatch": 1.0 if wants_angle_pi and not has_pi_answer and answer_numeric_only else 0.0,
        "checker_fraction_mismatch": 1.0 if wants_fraction and answer_numeric_only and not has_fraction_answer else 0.0,
        "checker_set_mismatch": 1.0 if wants_set_or_all and answer_numeric_only and not has_set_answer else 0.0,
        "checker_expression_mismatch": 1.0 if wants_expression and answer_numeric_only else 0.0,
        "checker_format_mismatch_count": float(mismatch_count),
        "checker_has_format_mismatch": 1.0 if mismatch_count else 0.0,
        "checker_completion_near_cap": 1.0 if near_cap else 0.0,
        "checker_trace_ends_incomplete": 1.0 if trace_ends_incomplete else 0.0,
        "checker_explicit_answer_present_in_head": 1.0 if explicit_answer_present else 0.0,
    }


def _checker_features(problem: dict[str, Any], row: dict[str, Any], seed_record: dict[str, Any] | None) -> dict[str, float]:
    metadata = (seed_record or {}).get("metadata") or {}
    trace = str(metadata.get("completion_head") or "")
    question = str(problem.get("question") or row.get("question") or "")
    seed_answer = row.get("seed_answer")
    features = {}
    features.update(_format_checker_features(question, seed_answer, trace, seed_record))
    features.update(_arithmetic_checker_features(trace))
    # A compact aggregate risk feature, deliberately simple and interpretable.
    risk_terms = [
        features.get("checker_has_format_mismatch", 0.0),
        features.get("checker_arith_has_mismatch", 0.0),
        features.get("checker_completion_near_cap", 0.0),
        features.get("checker_trace_ends_incomplete", 0.0),
        features.get("checker_seed_copies_problem_number", 0.0) * features.get("checker_many_question_numbers", 0.0),
        features.get("checker_small_integer_answer", 0.0) * features.get("checker_wants_geometry", 0.0),
    ]
    features["checker_independent_risk_score"] = float(sum(risk_terms))
    return features


def main() -> None:
    parser = argparse.ArgumentParser(description="Add non-LLM checker features to K-armed budget rows.")
    parser.add_argument("--arm-rows", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", choices=["train", "dev", "test"], default="dev")
    parser.add_argument(
        "--seed-result",
        action="append",
        default=[],
        help="Mapping in the form seed:path/to/seed_result.json. Repeat once per seed.",
    )
    parser.add_argument("--seed-method", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    problems = {str(row["qid"]): row for row in load_problem_split(config, args.split)}
    seed_by_seed = _seed_maps(args.seed_result, args.seed_method)
    rows = _load_jsonl(args.arm_rows)

    updated = 0
    feature_hits: dict[str, int] = {}
    for row in rows:
        seed = int(row["seed"])
        qid = str(row["qid"])
        problem = problems.get(qid, {})
        seed_record = seed_by_seed.get(seed, {}).get(qid)
        checker = _checker_features(problem, row, seed_record)
        features = dict(row.get("features") or {})
        for key, value in checker.items():
            if isinstance(value, (int, float)) and math.isfinite(float(value)):
                features[key] = float(value)
                feature_hits[key] = feature_hits.get(key, 0) + 1
        row["features"] = features
        row["independent_checker_available"] = seed_record is not None and bool(problem)
        updated += 1

    _write_jsonl(args.output, rows)
    print(
        f"wrote {args.output} updated={updated}/{len(rows)} "
        f"features={json.dumps(feature_hits, sort_keys=True)}"
    )


if __name__ == "__main__":
    main()
