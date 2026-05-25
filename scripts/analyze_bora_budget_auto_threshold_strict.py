from __future__ import annotations

import argparse
import csv
import json
import math
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
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


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


def _rows_by_seed(rows: list[dict[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    out = {seed: [] for seed in SEEDS}
    for row in rows:
        seed = int(row["seed"])
        if seed in out:
            out[seed].append(row)
    for seed in out:
        out[seed].sort(key=lambda row: str(row["qid"]))
    return out


def _seed_tokens(row: dict[str, Any]) -> float:
    return _as_float(row.get("seed_total_tokens"))


def _think_tokens(row: dict[str, Any], arm: str) -> float:
    return _as_float(row.get(f"{arm}_total_tokens"))


def _fit_prob(
    train_rows: list[dict[str, Any]],
    eval_rows: list[dict[str, Any]],
    names: list[str],
    key: str,
    *,
    seed: int,
    positive_weight: float,
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
    sample_weight = np.where(y == 1, positive_weight, 1.0)
    model.fit(_matrix(train_rows, names), y, sample_weight=sample_weight)
    return model.predict_proba(_matrix(eval_rows, names))[:, 1]


def _predict_costs(
    *,
    train_rows: list[dict[str, Any]],
    eval_rows: list[dict[str, Any]],
    names: list[str],
    arm: str,
    mode: str,
    seed: int,
) -> np.ndarray:
    train_costs = np.array([_think_tokens(row, arm) for row in train_rows], dtype=np.float64)
    if mode == "actual_oracle":
        return np.array([_think_tokens(row, arm) for row in eval_rows], dtype=np.float64)
    if mode == "constant_mean":
        return np.full(len(eval_rows), float(np.mean(train_costs)))
    if mode == "constant_p90":
        return np.full(len(eval_rows), float(np.percentile(train_costs, 90)))
    if mode == "learned_gbdt":
        model = HistGradientBoostingRegressor(
            max_iter=200,
            learning_rate=0.05,
            l2_regularization=0.01,
            random_state=seed,
            loss="squared_error",
        )
        model.fit(_matrix(train_rows, names), train_costs)
        pred = model.predict(_matrix(eval_rows, names))
        return np.clip(pred, 1.0, None)
    raise ValueError(f"unknown cost mode: {mode}")


def _simulate(rows: list[dict[str, Any]], selected: set[int], *, arm: str) -> dict[str, Any]:
    correct = helpful = harmful = wrong_to_wrong = adopted = 0
    total_tokens: list[float] = []
    actual_extra_tokens = 0.0
    for idx, row in enumerate(rows):
        seed_correct = bool(row.get("seed_correct"))
        total = _seed_tokens(row)
        if idx in selected:
            final_correct = bool(row.get(f"{arm}_final_correct"))
            extra = _think_tokens(row, arm)
            total += extra
            actual_extra_tokens += extra
            adopted += int(bool(row.get(f"{arm}_gate_pass")))
            helpful += int((not seed_correct) and final_correct)
            harmful += int(seed_correct and (not final_correct))
            wrong_to_wrong += int((not seed_correct) and (not final_correct) and bool(row.get(f"{arm}_gate_pass")))
        else:
            final_correct = seed_correct
        correct += int(final_correct)
        total_tokens.append(total)
    n = max(1, len(rows))
    return {
        "count": len(rows),
        "correct": correct,
        "accuracy": correct / n,
        "avg_tokens": _mean(total_tokens),
        "total_tokens_500q": _mean(total_tokens) * 500.0,
        "actual_extra_tokens": actual_extra_tokens,
        "actual_extra_tokens_per_q": actual_extra_tokens / n,
        "trigger_count": len(selected),
        "trigger_rate": len(selected) / n,
        "adoption_count": adopted,
        "adoption_rate": adopted / n,
        "helpful": helpful,
        "harmful": harmful,
        "wrong_to_wrong": wrong_to_wrong,
    }


def _select_by_budget(
    *,
    scores: np.ndarray,
    p_harm: np.ndarray,
    pred_costs: np.ndarray,
    extra_budget: float,
    harm_threshold: float | None,
) -> tuple[set[int], float, float]:
    eligible = []
    for idx, (score, harm, cost) in enumerate(zip(scores, p_harm, pred_costs, strict=True)):
        cost = max(1.0, float(cost))
        if float(score) <= 0:
            continue
        if harm_threshold is not None and float(harm) > harm_threshold:
            continue
        eligible.append((float(score) / cost, idx, cost))
    eligible.sort(reverse=True)
    selected: set[int] = set()
    pred_spent = 0.0
    lambda_b = 0.0
    for ratio, idx, cost in eligible:
        if pred_spent + cost <= extra_budget + 1e-9:
            selected.add(idx)
            pred_spent += cost
            lambda_b = ratio
    return selected, pred_spent, lambda_b


def _select_top_rate(scores: np.ndarray, rate: float) -> set[int]:
    k = min(len(scores), max(1, int(round(rate * len(scores)))))
    return set(int(idx) for idx in np.argsort(-scores)[:k])


def _run_one_split(
    *,
    train_rows: list[dict[str, Any]],
    eval_rows: list[dict[str, Any]],
    eval_seed: int,
    feature_names: list[str],
    arm: str,
    alpha: float,
    cost_modes: list[str],
    extra_budgets: list[float],
    harm_thresholds: list[float | None],
) -> list[dict[str, Any]]:
    p_help = _fit_prob(train_rows, eval_rows, feature_names, f"{arm}_helpful", seed=eval_seed, positive_weight=8.0)
    p_harm = _fit_prob(train_rows, eval_rows, feature_names, f"{arm}_harmful", seed=eval_seed + 101, positive_weight=12.0)
    scores = p_help - alpha * p_harm

    out: list[dict[str, Any]] = []
    for rate in (0.3, 0.5):
        selected = _select_top_rate(scores, rate)
        out.append(
            {
                "method": f"score_top{int(rate * 100)}",
                "eval_seed": eval_seed,
                "arm": arm,
                "cost_mode": "none",
                "harm_threshold": "none",
                "target_extra_budget": "",
                "predicted_extra_spent": "",
                "lambda_b": "",
                **_simulate(eval_rows, selected, arm=arm),
            }
        )

    for mode in cost_modes:
        pred_costs = _predict_costs(
            train_rows=train_rows,
            eval_rows=eval_rows,
            names=feature_names,
            arm=arm,
            mode=mode,
            seed=eval_seed + 1009,
        )
        for harm_threshold in harm_thresholds:
            htag = "none" if harm_threshold is None else f"{harm_threshold:.4g}"
            for budget in extra_budgets:
                selected, pred_spent, lambda_b = _select_by_budget(
                    scores=scores,
                    p_harm=p_harm,
                    pred_costs=pred_costs,
                    extra_budget=budget,
                    harm_threshold=harm_threshold,
                )
                out.append(
                    {
                        "method": "auto_threshold_value_per_token",
                        "eval_seed": eval_seed,
                        "arm": arm,
                        "cost_mode": mode,
                        "harm_threshold": htag,
                        "target_extra_budget": budget,
                        "predicted_extra_spent": pred_spent,
                        "lambda_b": lambda_b,
                        **_simulate(eval_rows, selected, arm=arm),
                    }
                )
    return out


def _summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        key = (
            str(row["method"]),
            str(row["arm"]),
            str(row["cost_mode"]),
            str(row["harm_threshold"]),
            str(row.get("target_extra_budget") or "frontier"),
        )
        groups.setdefault(key, []).append(row)
    out: list[dict[str, Any]] = []
    for (method, arm, cost_mode, harm_threshold, budget), items in sorted(groups.items()):
        out.append(
            {
                "method": method,
                "arm": arm,
                "cost_mode": cost_mode,
                "harm_threshold": harm_threshold,
                "target_extra_budget": budget,
                "seeds": len(items),
                "mean_acc": _mean([float(row["accuracy"]) for row in items]),
                "std_acc": _std([float(row["accuracy"]) for row in items]),
                "avg_tokens": _mean([float(row["avg_tokens"]) for row in items]),
                "actual_extra_tokens": _mean([float(row["actual_extra_tokens"]) for row in items]),
                "actual_extra_tokens_per_q": _mean([float(row["actual_extra_tokens_per_q"]) for row in items]),
                "predicted_extra_spent": _mean([_as_float(row.get("predicted_extra_spent")) for row in items]),
                "trigger_rate": _mean([float(row["trigger_rate"]) for row in items]),
                "adoption_rate": _mean([float(row["adoption_rate"]) for row in items]),
                "helpful": _mean([float(row["helpful"]) for row in items]),
                "harmful": _mean([float(row["harmful"]) for row in items]),
                "wrong_to_wrong": _mean([float(row["wrong_to_wrong"]) for row in items]),
                "lambda_b": _mean([_as_float(row.get("lambda_b")) for row in items if row.get("lambda_b") not in {"", None}]),
            }
        )
    return out


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _write_summary(path: Path, summary: list[dict[str, Any]], *, budgets: list[float]) -> None:
    lines = [
        "# BORA-Budget AutoThreshold Strict Deployment Audit",
        "",
        "Policy: sort by predicted value per predicted token, `r_i=(P_help-alpha P_harm)/c_hat_i`, and trigger until the total extra thinking budget is exhausted. This is equivalent to a token-price threshold for the binary `ACCEPT/THINK@12k` action set.",
        "",
        "## Main learned-cost policy",
        "",
        "| Extra budget | Acc | Avg tokens | Actual extra | Trigger | Harmful | Lambda_B |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    wanted = {
        ("auto_threshold_value_per_token", "think12", "learned_gbdt", "none", f"{budget:.1f}")
        for budget in budgets
    }
    for row in summary:
        key = (
            row["method"],
            row["arm"],
            row["cost_mode"],
            row["harm_threshold"],
            row["target_extra_budget"],
        )
        if key in wanted:
            lines.append(
                f"| {float(row['target_extra_budget']):.0f} | {100*float(row['mean_acc']):.2f}% | "
                f"{float(row['avg_tokens']):.1f} | {float(row['actual_extra_tokens']):.0f} | "
                f"{100*float(row['trigger_rate']):.1f}% | {float(row['harmful']):.1f} | "
                f"{float(row['lambda_b']):.8f} |"
            )
    lines.extend(
        [
            "",
            "## Cost-model comparison at key budgets",
            "",
            "| Cost model | Extra budget | Acc | Avg tokens | Actual extra | Trigger | Harmful |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    key_budgets = {400_000.0, 1_000_000.0, 1_500_000.0, 1_628_500.0}
    for row in summary:
        if (
            row["method"] == "auto_threshold_value_per_token"
            and row["arm"] == "think12"
            and row["harm_threshold"] == "none"
            and _as_float(row["target_extra_budget"]) in key_budgets
        ):
            lines.append(
                f"| {row['cost_mode']} | {float(row['target_extra_budget']):.0f} | "
                f"{100*float(row['mean_acc']):.2f}% | {float(row['avg_tokens']):.1f} | "
                f"{float(row['actual_extra_tokens']):.0f} | {100*float(row['trigger_rate']):.1f}% | "
                f"{float(row['harmful']):.1f} |"
            )
    lines.extend(
        [
            "",
            "## Frontier references",
            "",
            "| Method | Acc | Avg tokens | Trigger | Harmful |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in summary:
        if row["method"] in {"score_top30", "score_top50"}:
            lines.append(
                f"| {row['method']} | {100*float(row['mean_acc']):.2f}% | "
                f"{float(row['avg_tokens']):.1f} | {100*float(row['trigger_rate']):.1f}% | "
                f"{float(row['harmful']):.1f} |"
            )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Strict B_extra BORA-Budget AutoThreshold offline evaluation.")
    parser.add_argument(
        "--arm-rows",
        type=Path,
        default=Path("artifacts/remote_stage_main/independent_checkers_20260517/karmed_arm_rows_with_checkers.jsonl"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/bora_budget_autothreshold_strict"))
    parser.add_argument("--arm", type=str, default="think12")
    parser.add_argument("--alpha", type=float, default=2.0)
    parser.add_argument(
        "--extra-budgets",
        type=str,
        default=",".join(str(int(x)) for x in DEFAULT_EXTRA_BUDGETS),
    )
    parser.add_argument(
        "--cost-modes",
        type=str,
        default="learned_gbdt,constant_mean,constant_p90,actual_oracle",
    )
    parser.add_argument(
        "--harm-thresholds",
        type=str,
        default="none,0.01,0.03,0.05",
    )
    args = parser.parse_args()

    rows = _read_jsonl(args.arm_rows)
    by_seed = _rows_by_seed(rows)
    names = _feature_names(rows)
    budgets = [float(part) for part in args.extra_budgets.split(",") if part.strip()]
    cost_modes = [part.strip() for part in args.cost_modes.split(",") if part.strip()]
    harm_thresholds: list[float | None] = []
    for part in args.harm_thresholds.split(","):
        part = part.strip().lower()
        if not part:
            continue
        harm_thresholds.append(None if part == "none" else float(part))

    all_rows: list[dict[str, Any]] = []
    for eval_seed in SEEDS:
        train_rows = [row for seed in SEEDS if seed != eval_seed for row in by_seed[seed]]
        eval_rows = by_seed[eval_seed]
        all_rows.extend(
            _run_one_split(
                train_rows=train_rows,
                eval_rows=eval_rows,
                eval_seed=eval_seed,
                feature_names=names,
                arm=args.arm,
                alpha=args.alpha,
                cost_modes=cost_modes,
                extra_budgets=budgets,
                harm_thresholds=harm_thresholds,
            )
        )

    summary = _summarize(all_rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output_dir / "autothreshold_by_seed.csv", all_rows)
    _write_csv(args.output_dir / "autothreshold_summary.csv", summary)
    _write_summary(args.output_dir / "AUTOTHRESHOLD_STRICT_SUMMARY.md", summary, budgets=budgets)
    (args.output_dir / "config.json").write_text(
        json.dumps(
            {
                "arm_rows": str(args.arm_rows),
                "arm": args.arm,
                "alpha": args.alpha,
                "extra_budgets": budgets,
                "cost_modes": cost_modes,
                "harm_thresholds": ["none" if item is None else item for item in harm_thresholds],
                "seeds": SEEDS,
                "feature_count": len(names),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(args.output_dir / "AUTOTHRESHOLD_STRICT_SUMMARY.md")
    for row in summary:
        if (
            row["method"] == "auto_threshold_value_per_token"
            and row["cost_mode"] == "learned_gbdt"
            and row["harm_threshold"] == "none"
            and row["target_extra_budget"] in {"400000.0", "1000000.0", "1500000.0", "1628500.0"}
        ):
            print(
                row["cost_mode"],
                row["target_extra_budget"],
                f"acc={100*float(row['mean_acc']):.2f}",
                f"tok={float(row['avg_tokens']):.1f}",
                f"actual_extra={float(row['actual_extra_tokens']):.0f}",
                f"trigger={100*float(row['trigger_rate']):.1f}",
                f"harm={float(row['harmful']):.1f}",
            )


if __name__ == "__main__":
    main()
