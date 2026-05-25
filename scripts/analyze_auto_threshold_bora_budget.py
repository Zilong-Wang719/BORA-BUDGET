from __future__ import annotations

import argparse
import csv
import json
import math
import statistics as stats
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier


SEEDS = [17, 7, 23]
DEFAULT_AVG_BUDGETS = [1000.0, 1200.0, 1300.0, 1500.0, 2000.0, 2681.7, 3589.6, 3867.8]


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


def _fit_prob(train_rows: list[dict[str, Any]], eval_rows: list[dict[str, Any]], names: list[str], key: str, *, seed: int, positive_weight: float) -> tuple[np.ndarray, np.ndarray]:
    y = _labels(train_rows, key)
    if len(set(y.tolist())) < 2:
        p = float(np.mean(y)) if len(y) else 0.0
        return np.full(len(train_rows), p), np.full(len(eval_rows), p)
    x_train = _matrix(train_rows, names)
    x_eval = _matrix(eval_rows, names)
    sample_weight = np.where(y == 1, positive_weight, 1.0)
    model = HistGradientBoostingClassifier(
        max_iter=200,
        learning_rate=0.05,
        l2_regularization=0.01,
        random_state=seed,
    )
    model.fit(x_train, y, sample_weight=sample_weight)
    return model.predict_proba(x_train)[:, 1], model.predict_proba(x_eval)[:, 1]


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


def _think_tokens(row: dict[str, Any]) -> float:
    return _as_float(row.get("think12_total_tokens"))


def _simulate(rows: list[dict[str, Any]], selected: set[int]) -> dict[str, Any]:
    correct = helpful = harmful = wrong_to_wrong = adopted = 0
    tokens: list[float] = []
    for idx, row in enumerate(rows):
        seed_correct = bool(row.get("seed_correct"))
        if idx in selected:
            final_correct = bool(row.get("think12_final_correct"))
            total = _seed_tokens(row) + _think_tokens(row)
            adopted += int(bool(row.get("think12_gate_pass")))
            helpful += int((not seed_correct) and final_correct)
            harmful += int(seed_correct and (not final_correct))
            wrong_to_wrong += int((not seed_correct) and (not final_correct) and bool(row.get("think12_gate_pass")))
        else:
            final_correct = seed_correct
            total = _seed_tokens(row)
        correct += int(final_correct)
        tokens.append(total)
    n = max(1, len(rows))
    return {
        "count": len(rows),
        "correct": correct,
        "accuracy": correct / n,
        "avg_tokens": _mean(tokens),
        "total_tokens_500q": _mean(tokens) * 500.0,
        "trigger_count": len(selected),
        "trigger_rate": len(selected) / n,
        "adoption_count": adopted,
        "adoption_rate": adopted / n,
        "helpful": helpful,
        "harmful": harmful,
        "wrong_to_wrong": wrong_to_wrong,
    }


def _selected_by_positive_utility(scores: np.ndarray, costs: np.ndarray, lam_per_1k: float) -> set[int]:
    utility = scores - lam_per_1k * (costs / 1000.0)
    return {idx for idx, value in enumerate(utility.tolist()) if value > 0}


def _selected_by_threshold(scores: np.ndarray, tau: float) -> set[int]:
    return {idx for idx, value in enumerate(scores.tolist()) if value >= tau}


def _selected_top_rate(scores: np.ndarray, rate: float) -> set[int]:
    if rate <= 0:
        return set()
    k = min(len(scores), max(1, int(round(rate * len(scores)))))
    order = np.argsort(-scores)
    return set(int(idx) for idx in order[:k])


def _selected_budget_greedy(scores: np.ndarray, costs: np.ndarray, rows: list[dict[str, Any]], avg_budget: float) -> set[int]:
    total = sum(_seed_tokens(row) for row in rows)
    budget = avg_budget * max(1, len(rows))
    selected: set[int] = set()
    order = sorted(range(len(rows)), key=lambda idx: scores[idx] / max(1.0, costs[idx]), reverse=True)
    for idx in order:
        cost = float(costs[idx])
        if cost <= 0:
            continue
        if total + cost <= budget + 1e-9:
            selected.add(idx)
            total += cost
    return selected


def _actual_utility(sim: dict[str, Any], seed_correct: int, *, token_price_per_1k: float, harm_penalty: float) -> float:
    delta = float(sim["correct"]) - float(seed_correct)
    extra = max(0.0, float(sim["avg_tokens"]) - float(sim.get("seed_avg_tokens", 0.0))) * float(sim["count"]) / 1000.0
    return delta - token_price_per_1k * extra - harm_penalty * float(sim["harmful"])


def _train_eval_one(
    *,
    train_rows: list[dict[str, Any]],
    eval_rows: list[dict[str, Any]],
    eval_seed: int,
    feature_names: list[str],
    alpha: float,
    token_price_grid: list[float],
    utility_token_price: float,
    harm_penalty: float,
    avg_budgets: list[float],
    random_seed: int,
) -> list[dict[str, Any]]:
    train_help, eval_help = _fit_prob(train_rows, eval_rows, feature_names, "think12_helpful", seed=random_seed, positive_weight=8.0)
    train_harm, eval_harm = _fit_prob(train_rows, eval_rows, feature_names, "think12_harmful", seed=random_seed + 101, positive_weight=12.0)
    train_scores = train_help - alpha * train_harm
    eval_scores = eval_help - alpha * eval_harm
    train_costs = np.array([_think_tokens(row) for row in train_rows], dtype=np.float64)
    eval_costs = np.array([_think_tokens(row) for row in eval_rows], dtype=np.float64)
    train_seed_correct = sum(int(bool(row.get("seed_correct"))) for row in train_rows)
    eval_seed_avg = _mean([_seed_tokens(row) for row in eval_rows])

    out: list[dict[str, Any]] = []

    # Frontier references using learned ranking.
    for rate in (0.3, 0.5):
        selected = _selected_top_rate(eval_scores, rate)
        sim = _simulate(eval_rows, selected)
        sim["seed_avg_tokens"] = eval_seed_avg
        out.append({"method": f"learned_top{int(rate*100)}", "eval_seed": eval_seed, "selected_lambda": "", "target_budget": "", **sim})

    # Auto utility: choose lambda on training seeds by actual validation utility.
    best_lam = None
    best_train_utility = -1e18
    best_train_trigger = 0.0
    for lam in token_price_grid:
        selected = _selected_by_positive_utility(train_scores, train_costs, lam)
        sim = _simulate(train_rows, selected)
        sim["seed_avg_tokens"] = _mean([_seed_tokens(row) for row in train_rows])
        util = _actual_utility(sim, train_seed_correct, token_price_per_1k=utility_token_price, harm_penalty=harm_penalty)
        if util > best_train_utility:
            best_train_utility = util
            best_lam = lam
            best_train_trigger = sim["trigger_rate"]
    assert best_lam is not None
    selected = _selected_by_positive_utility(eval_scores, eval_costs, best_lam)
    sim = _simulate(eval_rows, selected)
    sim["seed_avg_tokens"] = eval_seed_avg
    out.append(
        {
            "method": "auto_utility_positive",
            "eval_seed": eval_seed,
            "selected_lambda": best_lam,
            "train_utility": best_train_utility,
            "train_trigger_rate": best_train_trigger,
            "target_budget": "",
            **sim,
        }
    )

    # Budget-price threshold: choose lambda so train cost fits each target budget.
    for avg_budget in avg_budgets:
        candidates: list[tuple[float, float, float, set[int]]] = []
        for lam in token_price_grid:
            selected = _selected_by_positive_utility(train_scores, train_costs, lam)
            sim = _simulate(train_rows, selected)
            candidates.append((abs(float(sim["avg_tokens"]) - avg_budget), -float(sim["accuracy"]), lam, selected))
        _gap, _neg_acc, chosen_lam, _train_selected = min(candidates)
        selected_eval = _selected_by_positive_utility(eval_scores, eval_costs, chosen_lam)
        sim = _simulate(eval_rows, selected_eval)
        sim["seed_avg_tokens"] = eval_seed_avg
        out.append(
            {
                "method": "budget_price_positive",
                "eval_seed": eval_seed,
                "selected_lambda": chosen_lam,
                "target_budget": avg_budget,
                **sim,
            }
        )

        selected_eval = _selected_budget_greedy(eval_scores, eval_costs, eval_rows, avg_budget)
        sim = _simulate(eval_rows, selected_eval)
        sim["seed_avg_tokens"] = eval_seed_avg
        out.append(
            {
                "method": "learned_value_per_token_greedy",
                "eval_seed": eval_seed,
                "selected_lambda": "",
                "target_budget": avg_budget,
                **sim,
            }
        )
    return out


def _summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        key = (str(row["method"]), str(row.get("target_budget") or "auto"))
        groups.setdefault(key, []).append(row)
    out: list[dict[str, Any]] = []
    for (method, target_budget), items in sorted(groups.items()):
        out.append(
            {
                "method": method,
                "target_budget": target_budget,
                "seeds": len(items),
                "mean_acc": _mean([float(row["accuracy"]) for row in items]),
                "std_acc": _std([float(row["accuracy"]) for row in items]),
                "avg_tokens": _mean([float(row["avg_tokens"]) for row in items]),
                "total_tokens_500q": _mean([float(row["total_tokens_500q"]) for row in items]),
                "trigger_rate": _mean([float(row["trigger_rate"]) for row in items]),
                "adoption_rate": _mean([float(row["adoption_rate"]) for row in items]),
                "helpful": _mean([float(row["helpful"]) for row in items]),
                "harmful": _mean([float(row["harmful"]) for row in items]),
                "wrong_to_wrong": _mean([float(row["wrong_to_wrong"]) for row in items]),
                "selected_lambda": _mean([float(row["selected_lambda"]) for row in items if row.get("selected_lambda") not in {"", None}]),
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
        for row in rows:
            writer.writerow(row)


def _write_summary(path: Path, summary: list[dict[str, Any]]) -> None:
    lines = ["# Auto-Threshold BORA-Budget", ""]
    lines.append("Policy: trigger `THINK@12k` when `P_help - alpha P_harm - lambda C > 0`; lambda is chosen on calibration seeds, so trigger rate is induced by the score distribution.")
    lines.append("")
    lines.append("| Method | Target budget | Acc | Tokens | Trigger | Harmful | Lambda |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in summary:
        if row["method"] in {"auto_utility_positive", "budget_price_positive", "learned_value_per_token_greedy", "learned_top30", "learned_top50"}:
            target = row["target_budget"]
            lines.append(
                f"| {row['method']} | {target} | {100*float(row['mean_acc']):.2f}% | "
                f"{float(row['avg_tokens']):.1f} | {100*float(row['trigger_rate']):.1f}% | "
                f"{float(row['harmful']):.1f} | {float(row['selected_lambda'] or 0.0):.6f} |"
            )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Auto-threshold / token-price BORA-Budget offline evaluation.")
    parser.add_argument(
        "--arm-rows",
        type=Path,
        default=Path("artifacts/remote_stage_main/independent_checkers_20260517/karmed_arm_rows_with_checkers.jsonl"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/auto_threshold_bora_budget"))
    parser.add_argument("--alpha", type=float, default=2.0)
    parser.add_argument("--utility-token-price", type=float, default=0.0)
    parser.add_argument("--harm-penalty", type=float, default=2.0)
    parser.add_argument("--avg-budgets", type=str, default=",".join(str(x) for x in DEFAULT_AVG_BUDGETS))
    args = parser.parse_args()

    rows = _read_jsonl(args.arm_rows)
    by_seed = _rows_by_seed(rows)
    feature_names = _feature_names(rows)
    avg_budgets = [float(part) for part in args.avg_budgets.split(",") if part.strip()]
    token_price_grid = [0.0]
    token_price_grid.extend([10 ** exp for exp in np.linspace(-6, -1, 101)])
    all_rows: list[dict[str, Any]] = []
    for eval_seed in SEEDS:
        train_rows = [row for seed in SEEDS if seed != eval_seed for row in by_seed[seed]]
        eval_rows = by_seed[eval_seed]
        all_rows.extend(
            _train_eval_one(
                train_rows=train_rows,
                eval_rows=eval_rows,
                eval_seed=eval_seed,
                feature_names=feature_names,
                alpha=float(args.alpha),
                token_price_grid=token_price_grid,
                utility_token_price=float(args.utility_token_price),
                harm_penalty=float(args.harm_penalty),
                avg_budgets=avg_budgets,
                random_seed=eval_seed,
            )
        )
    summary = _summarize(all_rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output_dir / "auto_threshold_by_seed.csv", all_rows)
    _write_csv(args.output_dir / "auto_threshold_summary.csv", summary)
    _write_summary(args.output_dir / "AUTO_THRESHOLD_SUMMARY.md", summary)
    (args.output_dir / "config.json").write_text(
        json.dumps(
            {
                "arm_rows": str(args.arm_rows),
                "alpha": args.alpha,
                "utility_token_price": args.utility_token_price,
                "harm_penalty": args.harm_penalty,
                "avg_budgets": avg_budgets,
                "feature_count": len(feature_names),
                "seeds": SEEDS,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(args.output_dir / "AUTO_THRESHOLD_SUMMARY.md")
    for row in summary:
        if row["method"] in {"auto_utility_positive", "learned_top30", "learned_top50"}:
            print(row["method"], row["target_budget"], f"acc={100*float(row['mean_acc']):.2f}", f"tok={float(row['avg_tokens']):.1f}", f"trigger={100*float(row['trigger_rate']):.1f}", f"lambda={float(row['selected_lambda'] or 0.0):.6g}")
    for row in summary:
        if row["method"] == "budget_price_positive" and row["target_budget"] in {"1300.0", "2681.7", "3589.6"}:
            print(row["method"], row["target_budget"], f"acc={100*float(row['mean_acc']):.2f}", f"tok={float(row['avg_tokens']):.1f}", f"trigger={100*float(row['trigger_rate']):.1f}", f"lambda={float(row['selected_lambda'] or 0.0):.6g}")


if __name__ == "__main__":
    main()
