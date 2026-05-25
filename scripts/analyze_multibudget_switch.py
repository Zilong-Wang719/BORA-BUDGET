from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics as stats
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from bora.common import dump_json, is_correct, load_jsonl

from analyze_trigger_frontier import _feature_names, _fit_scores, _matrix  # type: ignore
from summarize_bora_switch_calibrated import (  # type: ignore
    OPPORTUNITY_PATHS,
    SEEDS,
    THINK8K_PATHS,
    _load_json,
    _mean,
    _records,
    _std,
    _write_csv,
)


def _baseline_records(path: Path) -> dict[str, dict[str, Any]]:
    payload = _load_json(path)
    return {str(row["qid"]): row for row in _records(payload)}


def _num(value: Any) -> float | None:
    try:
        return float(str(value).replace(",", ""))
    except Exception:
        return None


def _strict_pass(seed_answer: Any, think_answer: Any, total_tokens: float, max_tokens: int) -> bool:
    seed = _num(seed_answer)
    think = _num(think_answer)
    if think is None:
        return False
    if total_tokens >= 0.98 * max_tokens:
        return False
    if seed is not None and seed != 0 and think != 0:
        if (seed > 0) != (think > 0):
            return False
    return True


def _arm_rows(root: Path, seed: int) -> list[dict[str, Any]]:
    rows12 = load_jsonl(root / OPPORTUNITY_PATHS[seed])
    rows8_by_qid = _baseline_records(root / THINK8K_PATHS[seed])
    out = []
    for row12 in rows12:
        qid = str(row12["qid"])
        row8 = rows8_by_qid[qid]
        seed_correct = bool(row12["seed_correct"])
        seed_answer = row12.get("seed_answer")
        gold = row12.get("gold_answer")

        ans8 = row8.get("prediction")
        tok8 = float(row8.get("total_tokens") or 0.0)
        pass8 = _strict_pass(seed_answer, ans8, tok8, 8192)
        final8 = ans8 if pass8 else seed_answer
        correct8 = bool(is_correct(final8, gold))

        ans12 = row12.get("think_answer")
        tok12 = float(row12.get("think_total_tokens") or 0.0)
        pass12 = bool(row12.get("strict_filter_pass"))
        final12 = ans12 if pass12 else seed_answer
        correct12 = bool(is_correct(final12, gold))

        out.append(
            {
                **row12,
                "think8_answer": ans8,
                "think8_correct_raw": bool(row8.get("correct")),
                "think8_total_tokens": tok8,
                "think8_strict_pass": pass8,
                "think8_final_correct": correct8,
                "think12_final_correct": correct12,
                "think8_helpful": (not seed_correct) and correct8,
                "think12_helpful": (not seed_correct) and correct12,
                "think8_harmful": seed_correct and (not correct8),
                "think12_harmful": seed_correct and (not correct12),
            }
        )
    return out


def _simulate(rows: list[dict[str, Any]], actions: dict[str, str]) -> dict[str, Any]:
    final_correct = 0
    tokens = []
    helpful = harmful = wrong_to_wrong = 0
    counts = {"accept": 0, "think8": 0, "think12": 0}
    for row in rows:
        action = actions.get(str(row["qid"]), "accept")
        counts[action] += 1
        seed_correct = bool(row["seed_correct"])
        seed_tokens = float(row.get("seed_total_tokens") or 0.0)
        if action == "think8":
            correct = bool(row["think8_final_correct"])
            total = seed_tokens + float(row.get("think8_total_tokens") or 0.0)
        elif action == "think12":
            correct = bool(row["think12_final_correct"])
            total = seed_tokens + float(row.get("think_total_tokens") or 0.0)
        else:
            correct = seed_correct
            total = seed_tokens
        final_correct += int(correct)
        tokens.append(total)
        helpful += int((not seed_correct) and correct)
        harmful += int(seed_correct and (not correct))
        wrong_to_wrong += int((not seed_correct) and (not correct) and action != "accept")
    n = max(1, len(rows))
    return {
        "count": len(rows),
        "correct": final_correct,
        "accuracy": final_correct / n,
        "avg_tokens": _mean(tokens),
        "helpful": helpful,
        "harmful": harmful,
        "wrong_to_wrong": wrong_to_wrong,
        "accept": counts["accept"],
        "think8": counts["think8"],
        "think12": counts["think12"],
        "think_rate": (counts["think8"] + counts["think12"]) / n,
    }


def _oracle_actions(rows: list[dict[str, Any]], max_avg_tokens: float | None = None) -> dict[str, str]:
    # Greedy by utility per extra token; enough for a diagnostic upper bound.
    actions = {str(row["qid"]): "accept" for row in rows}
    base = _simulate(rows, actions)
    candidates: list[tuple[float, float, str, str]] = []
    for row in rows:
        qid = str(row["qid"])
        seed_correct = int(bool(row["seed_correct"]))
        seed_tokens = float(row.get("seed_total_tokens") or 0.0)
        for action, correct_key, token_key in [
            ("think8", "think8_final_correct", "think8_total_tokens"),
            ("think12", "think12_final_correct", "think_total_tokens"),
        ]:
            gain = int(bool(row[correct_key])) - seed_correct
            extra = float(row.get(token_key) or 0.0)
            if gain <= 0 or extra <= 0:
                continue
            candidates.append((gain / extra, extra, qid, action))
    candidates.sort(reverse=True)
    total_tokens = base["avg_tokens"] * len(rows)
    budget_total = None if max_avg_tokens is None else max_avg_tokens * len(rows)
    for _ratio, extra, qid, action in candidates:
        if actions[qid] != "accept":
            continue
        if budget_total is not None and total_tokens + extra > budget_total:
            continue
        actions[qid] = action
        total_tokens += extra
    if max_avg_tokens is None:
        for row in rows:
            qid = str(row["qid"])
            if actions[qid] != "accept":
                continue
            c8 = bool(row["think8_final_correct"])
            c12 = bool(row["think12_final_correct"])
            seed = bool(row["seed_correct"])
            if c12 and not seed:
                actions[qid] = "think12"
            elif c8 and not seed:
                actions[qid] = "think8"
    return actions


def _fit_arm_scores(
    train_rows: list[dict[str, Any]],
    eval_rows: list[dict[str, Any]],
    *,
    arm: str,
    random_seed: int,
) -> list[float]:
    train = []
    eval_ = []
    for row in train_rows:
        train.append({**row, "helpful": bool(row[f"{arm}_helpful"]), "harmful": bool(row[f"{arm}_harmful"])})
    for row in eval_rows:
        eval_.append({**row, "helpful": bool(row[f"{arm}_helpful"]), "harmful": bool(row[f"{arm}_harmful"])})
    names = _feature_names(train + eval_)
    _train_scores, eval_scores, _harm, _info = _fit_scores(
        train,
        eval_,
        model_name="gbdt",
        feature_names=names,
        harm_weight=2.0,
        random_seed=random_seed,
    )
    return eval_scores


def _learned_budget_actions(
    train_rows: list[dict[str, Any]],
    eval_rows: list[dict[str, Any]],
    *,
    target_rate: float,
    random_seed: int,
    lambda_cost: float,
) -> dict[str, str]:
    score8 = _fit_arm_scores(train_rows, eval_rows, arm="think8", random_seed=random_seed)
    score12 = _fit_arm_scores(train_rows, eval_rows, arm="think12", random_seed=random_seed + 101)
    items = []
    for idx, row in enumerate(eval_rows):
        s8 = score8[idx] - lambda_cost * float(row.get("think8_total_tokens") or 0.0)
        s12 = score12[idx] - lambda_cost * float(row.get("think_total_tokens") or 0.0)
        if s8 <= 0 and s12 <= 0:
            continue
        action = "think12" if s12 >= s8 else "think8"
        best = max(s8, s12)
        items.append((best, str(row["qid"]), action))
    items.sort(reverse=True)
    k = int(round(len(eval_rows) * target_rate))
    return {qid: action for _score, qid, action in items[:k]}


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/paper_tables"))
    parser.add_argument("--lambda-cost", type=float, default=0.0)
    args = parser.parse_args()

    rows_by_seed = {seed: _arm_rows(args.root, seed) for seed in SEEDS}
    budget_points = {
        "oracle_unconstrained": None,
        "oracle_match_think8k": 3867.8,
        "oracle_match_gbdt50": 3589.6,
        "oracle_match_gbdt30": 2681.7,
    }
    rows_out = []
    for name, budget in budget_points.items():
        per = []
        for seed, rows in rows_by_seed.items():
            sim = _simulate(rows, _oracle_actions(rows, max_avg_tokens=budget))
            per.append(sim)
            rows_out.append({"method": name, "seed": seed, **sim})
        rows_out.append(
            {
                "method": name,
                "seed": "mean",
                "accuracy": _mean([r["accuracy"] for r in per]),
                "std_accuracy": _std([r["accuracy"] for r in per]),
                "avg_tokens": _mean([r["avg_tokens"] for r in per]),
                "helpful": sum(r["helpful"] for r in per),
                "harmful": sum(r["harmful"] for r in per),
                "wrong_to_wrong": sum(r["wrong_to_wrong"] for r in per),
                "accept": sum(r["accept"] for r in per),
                "think8": sum(r["think8"] for r in per),
                "think12": sum(r["think12"] for r in per),
            }
        )

    learned_rows = []
    for rate in (0.3, 0.5):
        per = []
        for seed in SEEDS:
            train_rows = [row for other in SEEDS if other != seed for row in rows_by_seed[other]]
            eval_rows = rows_by_seed[seed]
            actions = _learned_budget_actions(
                train_rows,
                eval_rows,
                target_rate=rate,
                random_seed=seed,
                lambda_cost=float(args.lambda_cost),
            )
            sim = _simulate(eval_rows, actions)
            per.append(sim)
            learned_rows.append({"method": f"learned_multiarm_top{int(rate*100)}", "seed": seed, **sim})
        learned_rows.append(
            {
                "method": f"learned_multiarm_top{int(rate*100)}",
                "seed": "mean",
                "accuracy": _mean([r["accuracy"] for r in per]),
                "std_accuracy": _std([r["accuracy"] for r in per]),
                "avg_tokens": _mean([r["avg_tokens"] for r in per]),
                "helpful": sum(r["helpful"] for r in per),
                "harmful": sum(r["harmful"] for r in per),
                "wrong_to_wrong": sum(r["wrong_to_wrong"] for r in per),
                "accept": sum(r["accept"] for r in per),
                "think8": sum(r["think8"] for r in per),
                "think12": sum(r["think12"] for r in per),
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output_dir / "math500_multibudget_oracle.csv", rows_out)
    _write_csv(args.output_dir / "math500_multibudget_learned.csv", learned_rows)
    payload = {"oracle": rows_out, "learned": learned_rows}
    dump_json(args.output_dir / "math500_multibudget_switch.json", payload)

    print("oracle means:")
    for row in rows_out:
        if row["seed"] == "mean":
            print(row["method"], f"acc={100*row['accuracy']:.2f}", f"tok={row['avg_tokens']:.1f}", "8k", row["think8"], "12k", row["think12"], "harm", row["harmful"])
    print("learned means:")
    for row in learned_rows:
        if row["seed"] == "mean":
            print(row["method"], f"acc={100*row['accuracy']:.2f}", f"tok={row['avg_tokens']:.1f}", "8k", row["think8"], "12k", row["think12"], "harm", row["harmful"])


if __name__ == "__main__":
    main()
