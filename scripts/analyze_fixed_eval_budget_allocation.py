from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics as stats
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor


SEEDS = [17, 7, 23]
DEFAULT_EXTRA_BUDGETS = [
    200_000.0,
    300_000.0,
    400_000.0,
    500_000.0,
    750_000.0,
    1_000_000.0,
    1_250_000.0,
    1_500_000.0,
    1_628_500.0,
]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _std(values: list[float]) -> float:
    return stats.stdev(values) if len(values) > 1 else 0.0


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _feature_names(rows: list[dict[str, Any]]) -> list[str]:
    names: set[str] = set()
    for row in rows:
        names.update((row.get("features") or {}).keys())
    return sorted(names)


def _matrix(rows: list[dict[str, Any]], names: list[str]) -> np.ndarray:
    return np.array(
        [[_as_float((row.get("features") or {}).get(name), 0.0) for name in names] for row in rows],
        dtype=np.float64,
    )


def _labels(rows: list[dict[str, Any]], key: str) -> np.ndarray:
    return np.array([int(bool(row.get(key))) for row in rows], dtype=np.int64)


def _seed_tokens(row: dict[str, Any]) -> float:
    return _as_float(row.get("seed_total_tokens"))


def _think_tokens(row: dict[str, Any]) -> float:
    return _as_float(row.get("think_total_tokens"))


def _trace_score(row: dict[str, Any]) -> float:
    features = row.get("features") or {}
    return (
        _as_float(features.get("sf_trace_words"))
        or _as_float(features.get("seed_total_tokens"))
        or _seed_tokens(row)
    )


def _fit_prob(
    train_rows: list[dict[str, Any]],
    eval_rows: list[dict[str, Any]],
    names: list[str],
    key: str,
    *,
    seed: int,
) -> np.ndarray:
    y = _labels(train_rows, key)
    if len(set(y.tolist())) < 2:
        p = float(np.mean(y)) if len(y) else 0.0
        return np.full(len(eval_rows), p)
    model = HistGradientBoostingClassifier(
        max_iter=200,
        learning_rate=0.05,
        l2_regularization=0.01,
        random_state=seed,
    )
    # Match scripts/analyze_trigger_frontier.py exactly for the paper-facing
    # fixed-eval trigger. Earlier budget-allocation audits used class weights;
    # this script intentionally does not, so top30/top50 references align with
    # the current main trigger-frontier summary.
    model.fit(_matrix(train_rows, names), y)
    return model.predict_proba(_matrix(eval_rows, names))[:, 1]


def _fit_cost(
    train_rows: list[dict[str, Any]],
    eval_rows: list[dict[str, Any]],
    names: list[str],
    *,
    seed: int,
) -> np.ndarray:
    y = np.array([_think_tokens(row) for row in train_rows], dtype=np.float64)
    model = HistGradientBoostingRegressor(
        max_iter=200,
        learning_rate=0.05,
        l2_regularization=0.01,
        random_state=seed,
        loss="squared_error",
    )
    model.fit(_matrix(train_rows, names), y)
    return np.clip(model.predict(_matrix(eval_rows, names)), 1.0, None)


def _strict_final_correct(row: dict[str, Any]) -> bool:
    if bool(row.get("strict_filter_pass")):
        return bool(row.get("think_correct"))
    return bool(row.get("seed_correct"))


def _simulate(rows: list[dict[str, Any]], selected: set[int]) -> dict[str, Any]:
    correct = helpful = harmful = wrong_to_wrong = adopted = 0
    total_tokens: list[float] = []
    actual_extra = 0.0
    for idx, row in enumerate(rows):
        seed_correct = bool(row.get("seed_correct"))
        tokens = _seed_tokens(row)
        final_correct = seed_correct
        if idx in selected:
            tokens += _think_tokens(row)
            actual_extra += _think_tokens(row)
            if bool(row.get("strict_filter_pass")):
                adopted += 1
                final_correct = bool(row.get("think_correct"))
                helpful += int((not seed_correct) and final_correct)
                harmful += int(seed_correct and (not final_correct))
                wrong_to_wrong += int((not seed_correct) and (not final_correct))
        correct += int(final_correct)
        total_tokens.append(tokens)
    n = max(1, len(rows))
    return {
        "count": len(rows),
        "correct": correct,
        "accuracy": correct / n,
        "avg_tokens": _mean(total_tokens),
        "total_tokens_500q": _mean(total_tokens) * 500.0,
        "actual_extra_tokens": actual_extra,
        "actual_extra_tokens_per_q": actual_extra / n,
        "trigger_count": len(selected),
        "trigger_rate": len(selected) / n,
        "adoption_count": adopted,
        "adoption_rate": adopted / n,
        "helpful": helpful,
        "harmful": harmful,
        "wrong_to_wrong": wrong_to_wrong,
    }


def _select_top_rate(scores: np.ndarray, rate: float) -> set[int]:
    if rate <= 0:
        return set()
    k = min(len(scores), max(1, int(round(rate * len(scores)))))
    return {int(idx) for idx in np.argsort(-scores)[:k]}


def _select_value_per_token(
    *,
    scores: np.ndarray,
    costs: np.ndarray,
    extra_budget: float,
    positive_only: bool = True,
) -> tuple[set[int], float, float]:
    ranked: list[tuple[float, int, float]] = []
    for idx, (score, cost) in enumerate(zip(scores, costs, strict=True)):
        cost = max(1.0, float(cost))
        if positive_only and float(score) <= 0:
            continue
        ranked.append((float(score) / cost, idx, cost))
    ranked.sort(reverse=True)
    selected: set[int] = set()
    pred_spent = 0.0
    lambda_b = 0.0
    for ratio, idx, cost in ranked:
        if pred_spent + cost <= extra_budget + 1e-9:
            selected.add(idx)
            pred_spent += cost
            lambda_b = ratio
    return selected, pred_spent, lambda_b


def _select_oracle_binary(rows: list[dict[str, Any]], extra_budget: float) -> set[int]:
    candidates: list[tuple[float, int]] = []
    for idx, row in enumerate(rows):
        if bool(row.get("seed_correct")):
            continue
        if not bool(row.get("strict_filter_pass")):
            continue
        if not bool(row.get("think_correct")):
            continue
        candidates.append((_think_tokens(row), idx))
    candidates.sort()
    selected: set[int] = set()
    spent = 0.0
    for cost, idx in candidates:
        if spent + cost <= extra_budget + 1e-9:
            selected.add(idx)
            spent += cost
    return selected


def _load_seed_rows(root: Path, seed: int) -> list[dict[str, Any]]:
    path = root / f"seed{seed}" / f"opportunity_seed{seed}_think12k_fixed.jsonl"
    rows = _read_jsonl(path)
    rows.sort(key=lambda row: str(row.get("qid")))
    return rows


def _run_split(
    *,
    root: Path,
    eval_seed: int,
    extra_budgets: list[float],
    random_trials: int,
    alpha: float,
) -> list[dict[str, Any]]:
    eval_rows = _load_seed_rows(root, eval_seed)
    train_rows: list[dict[str, Any]] = []
    for seed in SEEDS:
        if seed != eval_seed:
            train_rows.extend(_load_seed_rows(root, seed))
    names = _feature_names(train_rows + eval_rows)
    p_help = _fit_prob(train_rows, eval_rows, names, "helpful", seed=eval_seed)
    p_harm = _fit_prob(train_rows, eval_rows, names, "harmful", seed=eval_seed + 101)
    scores = p_help - alpha * p_harm
    learned_costs = _fit_cost(train_rows, eval_rows, names, seed=eval_seed + 1009)
    actual_costs = np.array([_think_tokens(row) for row in eval_rows], dtype=np.float64)
    trace_scores = np.array([_trace_score(row) for row in eval_rows], dtype=np.float64)
    seed_avg = _mean([_seed_tokens(row) for row in eval_rows])

    out: list[dict[str, Any]] = []

    for method, selected, pred_spent, lambda_b, budget in [
        ("accept_only", set(), "", "", ""),
        ("all_trigger_think12_strict", set(range(len(eval_rows))), "", "", ""),
        ("score_top30", _select_top_rate(scores, 0.3), "", "", ""),
        ("score_top50", _select_top_rate(scores, 0.5), "", "", ""),
    ]:
        sim = _simulate(eval_rows, selected)
        out.append(
            {
                "eval_seed": eval_seed,
                "method": method,
                "target_extra_budget": budget,
                "predicted_extra_spent": pred_spent,
                "lambda_b": lambda_b,
                "seed_avg_tokens": seed_avg,
                **sim,
            }
        )

    rng = random.Random(eval_seed + 20260521)
    for budget in extra_budgets:
        selections: list[tuple[str, set[int], float, float]] = []
        oracle = _select_oracle_binary(eval_rows, budget)
        selections.append(("oracle_binary_12k", oracle, "", ""))
        learned, pred_spent, lambda_b = _select_value_per_token(
            scores=scores, costs=learned_costs, extra_budget=budget
        )
        selections.append(("learned_vpt_learned_cost", learned, pred_spent, lambda_b))
        learned_actual, pred_spent_actual, lambda_actual = _select_value_per_token(
            scores=scores, costs=actual_costs, extra_budget=budget
        )
        selections.append(("learned_vpt_actual_cost_diagnostic", learned_actual, pred_spent_actual, lambda_actual))
        trace, trace_spent, trace_lambda = _select_value_per_token(
            scores=trace_scores, costs=actual_costs, extra_budget=budget, positive_only=False
        )
        selections.append(("trace_vpt_actual_cost", trace, trace_spent, trace_lambda))

        for method, selected, pred_spent, lambda_b in selections:
            sim = _simulate(eval_rows, selected)
            out.append(
                {
                    "eval_seed": eval_seed,
                    "method": method,
                    "target_extra_budget": budget,
                    "predicted_extra_spent": pred_spent,
                    "lambda_b": lambda_b,
                    "seed_avg_tokens": seed_avg,
                    **sim,
                }
            )

        # Random matched-budget allocation, averaged within each seed.
        random_sims: list[dict[str, Any]] = []
        for _ in range(random_trials):
            order = list(range(len(eval_rows)))
            rng.shuffle(order)
            selected: set[int] = set()
            spent = 0.0
            for idx in order:
                cost = _think_tokens(eval_rows[idx])
                if spent + cost <= budget + 1e-9:
                    selected.add(idx)
                    spent += cost
            random_sims.append(_simulate(eval_rows, selected))
        mean_random: dict[str, Any] = {
            "eval_seed": eval_seed,
            "method": "random_actual_cost",
            "target_extra_budget": budget,
            "predicted_extra_spent": "",
            "lambda_b": "",
            "seed_avg_tokens": seed_avg,
        }
        keys = [key for key in random_sims[0].keys() if isinstance(random_sims[0][key], (int, float))]
        for key in keys:
            mean_random[key] = _mean([float(sim[key]) for sim in random_sims])
        out.append(mean_random)

    return out


def _summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault((str(row["method"]), str(row["target_extra_budget"])), []).append(row)
    out: list[dict[str, Any]] = []
    metric_keys = [
        "accuracy",
        "avg_tokens",
        "actual_extra_tokens",
        "actual_extra_tokens_per_q",
        "trigger_rate",
        "adoption_rate",
        "helpful",
        "harmful",
        "wrong_to_wrong",
    ]
    for (method, budget), group in sorted(groups.items(), key=lambda item: (item[0][0], str(item[0][1]))):
        row: dict[str, Any] = {"method": method, "target_extra_budget": budget}
        for key in metric_keys:
            vals = [float(g[key]) for g in group if key in g and g[key] != ""]
            if not vals:
                continue
            row[key] = _mean(vals)
            row[f"{key}_std"] = _std(vals)
        out.append(row)
    return out


def _format_pct(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def _write_markdown(path: Path, summary: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    wanted = [
        "accept_only",
        "score_top30",
        "score_top50",
        "all_trigger_think12_strict",
        "oracle_binary_12k",
        "learned_vpt_learned_cost",
        "learned_vpt_actual_cost_diagnostic",
        "trace_vpt_actual_cost",
        "random_actual_cost",
    ]
    lines = [
        "# Fixed-Eval BORA-Budget Allocation",
        "",
        "Action set: `ACCEPT / THINK@12k`. Correctness and adoption use the fixed full-completion evaluator rows.",
        "",
        "## Reference / frontier rows",
        "",
        "| Method | Acc | Avg tokens | Trigger | Harmful |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for method in ["accept_only", "score_top30", "score_top50", "all_trigger_think12_strict"]:
        row = next(r for r in summary if r["method"] == method and r["target_extra_budget"] == "")
        lines.append(
            f"| `{method}` | {_format_pct(float(row['accuracy']))} | {float(row['avg_tokens']):.1f} | "
            f"{_format_pct(float(row['trigger_rate']))} | {float(row['harmful']):.1f} |"
        )
    lines.extend(
        [
            "",
            "## Total extra-budget allocation",
            "",
            "| Extra budget | Oracle12 | Learned cost | Learned actual-cost diag | Trace | Random | Learned tokens | Learned trigger | Learned harmful |",
            "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    budgets = sorted(
        {
            str(r["target_extra_budget"])
            for r in summary
            if r["target_extra_budget"] not in ("", None)
        },
        key=lambda x: float(x),
    )
    by_key = {(r["method"], str(r["target_extra_budget"])): r for r in summary}
    for budget in budgets:
        oracle = by_key.get(("oracle_binary_12k", budget))
        learned = by_key.get(("learned_vpt_learned_cost", budget))
        learned_actual = by_key.get(("learned_vpt_actual_cost_diagnostic", budget))
        trace = by_key.get(("trace_vpt_actual_cost", budget))
        random_row = by_key.get(("random_actual_cost", budget))
        if not all([oracle, learned, learned_actual, trace, random_row]):
            continue
        lines.append(
            f"| {float(budget):.0f} | {_format_pct(float(oracle['accuracy']))} | "
            f"{_format_pct(float(learned['accuracy']))} | {_format_pct(float(learned_actual['accuracy']))} | "
            f"{_format_pct(float(trace['accuracy']))} | {_format_pct(float(random_row['accuracy']))} | "
            f"{float(learned['avg_tokens']):.1f} | {_format_pct(float(learned['trigger_rate']))} | "
            f"{float(learned['harmful']):.1f} |"
        )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- `learned_vpt_learned_cost` is the deployable row: it uses learned opportunity scores and a learned cost model.",
            "- `learned_vpt_actual_cost_diagnostic` uses held-out actual costs only to isolate cost-model error.",
            "- `oracle_binary_12k` is an upper bound for the binary action set and uses gold correctness labels.",
            "- Multi-arm `ACCEPT/THINK@8k/THINK@12k` fixed-eval allocation should be rerun after the missing fixed `THINK@8k` seed completes.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("artifacts/remote_stage_eval_audit/math500_main_algorithm_fixed_20260521"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/remote_stage_eval_audit/math500_fixed_budget_allocation_20260521"),
    )
    parser.add_argument("--extra-budgets", type=float, nargs="*", default=DEFAULT_EXTRA_BUDGETS)
    parser.add_argument("--alpha", type=float, default=2.0)
    parser.add_argument("--random-trials", type=int, default=100)
    args = parser.parse_args()

    rows: list[dict[str, Any]] = []
    for seed in SEEDS:
        rows.extend(
            _run_split(
                root=args.root,
                eval_seed=seed,
                extra_budgets=args.extra_budgets,
                random_trials=args.random_trials,
                alpha=args.alpha,
            )
        )
    summary = _summarize(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output_dir / "fixed_budget_allocation_by_seed.csv", rows)
    _write_csv(args.output_dir / "fixed_budget_allocation_summary.csv", summary)
    _write_markdown(args.output_dir / "FIXED_BUDGET_ALLOCATION_SUMMARY.md", summary)
    print(args.output_dir / "FIXED_BUDGET_ALLOCATION_SUMMARY.md")


if __name__ == "__main__":
    main()
