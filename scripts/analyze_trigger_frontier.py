from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

try:
    from bora.common import dump_json, load_jsonl
except ModuleNotFoundError:
    def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        with Path(path).open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows

    def dump_json(path: str | Path, payload: Any) -> None:
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(out):
        return default
    return out


def _feature_names(rows: list[dict[str, Any]]) -> list[str]:
    names: set[str] = set()
    for row in rows:
        names.update((row.get("features") or {}).keys())
    return sorted(names)


def _matrix(rows: list[dict[str, Any]], names: list[str]) -> list[list[float]]:
    return [[_as_float((row.get("features") or {}).get(name), 0.0) for name in names] for row in rows]


def _labels(rows: list[dict[str, Any]], name: str) -> list[int]:
    return [int(bool(row.get(name))) for row in rows]


def _constant_prob(rows: list[dict[str, Any]], label: str) -> list[float]:
    values = _labels(rows, label)
    p = sum(values) / max(1, len(values))
    return [p for _ in rows]


def _fit_scores(
    train_rows: list[dict[str, Any]],
    eval_rows: list[dict[str, Any]],
    *,
    model_name: str,
    feature_names: list[str],
    harm_weight: float,
    random_seed: int,
) -> tuple[list[float], list[float], list[float], dict[str, Any]]:
    if model_name == "trace_length":
        scores = [
            _as_float((row.get("features") or {}).get("sf_trace_words"), 0.0)
            or _as_float((row.get("features") or {}).get("seed_total_tokens"), 0.0)
            for row in eval_rows
        ]
        train_scores = [
            _as_float((row.get("features") or {}).get("sf_trace_words"), 0.0)
            or _as_float((row.get("features") or {}).get("seed_total_tokens"), 0.0)
            for row in train_rows
        ]
        return train_scores, scores, [0.0 for _ in scores], {"kind": "trace_length"}

    try:
        from sklearn.ensemble import HistGradientBoostingClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
    except Exception as exc:  # pragma: no cover - exercised on remote if sklearn missing.
        raise RuntimeError("scikit-learn is required for learned trigger models.") from exc

    x_train = _matrix(train_rows, feature_names)
    x_eval = _matrix(eval_rows, feature_names)

    def fit_predict(label: str) -> tuple[list[float], list[float], dict[str, Any]]:
        y_train = _labels(train_rows, label)
        if len(set(y_train)) < 2:
            p = sum(y_train) / max(1, len(y_train))
            return [p for _ in train_rows], [p for _ in eval_rows], {"constant": p}
        if model_name == "logistic":
            model = make_pipeline(
                StandardScaler(),
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=random_seed,
                ),
            )
        elif model_name == "gbdt":
            model = HistGradientBoostingClassifier(
                max_iter=200,
                learning_rate=0.05,
                l2_regularization=0.01,
                random_state=random_seed,
            )
        else:
            raise KeyError(f"Unknown model: {model_name}")
        model.fit(x_train, y_train)
        train_prob = [float(row[1]) for row in model.predict_proba(x_train)]
        eval_prob = [float(row[1]) for row in model.predict_proba(x_eval)]
        return train_prob, eval_prob, {"classes": [0, 1]}

    train_help, eval_help, help_info = fit_predict("helpful")
    train_harm, eval_harm, harm_info = fit_predict("harmful")
    train_scores = [h - harm_weight * r for h, r in zip(train_help, train_harm)]
    eval_scores = [h - harm_weight * r for h, r in zip(eval_help, eval_harm)]
    return train_scores, eval_scores, eval_harm, {
        "kind": model_name,
        "helpful_model": help_info,
        "harmful_model": harm_info,
        "harm_weight": harm_weight,
    }


def _filter_pass(row: dict[str, Any], filter_name: str) -> bool:
    if filter_name == "none":
        return True
    if filter_name == "main":
        return bool(row.get("main_filter_pass"))
    if filter_name == "strict":
        return bool(row.get("strict_filter_pass"))
    if filter_name == "main_delta2":
        return bool(row.get("main_filter_pass")) and _answer_delta(row) >= 2.0
    if filter_name == "strict_delta2":
        return bool(row.get("strict_filter_pass")) and _answer_delta(row) >= 2.0
    raise KeyError(f"Unknown adoption filter: {filter_name}")


def _answer_delta(row: dict[str, Any]) -> float:
    def parse(value: Any) -> float | None:
        try:
            return float(str(value).replace(",", ""))
        except Exception:
            return None

    seed = parse(row.get("seed_answer_normalized"))
    think = parse(row.get("think_answer_normalized"))
    if seed is None or think is None:
        return 0.0
    return abs(think - seed)


def _select_topk(scores: list[float], rate: float) -> set[int]:
    if rate <= 0:
        return set()
    k = min(len(scores), max(1, int(round(len(scores) * rate))))
    ordered = sorted(range(len(scores)), key=lambda idx: scores[idx], reverse=True)
    return set(ordered[:k])


def _threshold_from_train(train_scores: list[float], rate: float) -> float:
    if not train_scores:
        return float("inf")
    if rate <= 0:
        return max(train_scores) + 1e-9
    k = min(len(train_scores), max(1, int(round(len(train_scores) * rate))))
    ordered = sorted(train_scores, reverse=True)
    return ordered[k - 1]


def _simulate(
    rows: list[dict[str, Any]],
    *,
    selected: set[int],
    adoption_filter: str,
) -> dict[str, Any]:
    final_correct = 0
    seed_correct_count = 0
    think_correct_count = 0
    triggered = 0
    adopted = 0
    helpful = 0
    harmful = 0
    wrong_to_wrong = 0
    total_tokens: list[float] = []
    triggered_extra_tokens: list[float] = []
    helpful_qids: list[str] = []
    harmful_qids: list[str] = []

    for idx, row in enumerate(rows):
        seed_correct = bool(row.get("seed_correct"))
        think_correct = bool(row.get("think_correct"))
        seed_correct_count += int(seed_correct)
        think_correct_count += int(think_correct)
        seed_tokens = _as_float(row.get("seed_total_tokens"))
        think_tokens = _as_float(row.get("think_total_tokens"))
        use_think = idx in selected
        tokens = seed_tokens
        row_final_correct = seed_correct
        if use_think:
            triggered += 1
            tokens += think_tokens
            triggered_extra_tokens.append(think_tokens)
            if _filter_pass(row, adoption_filter):
                adopted += 1
                row_final_correct = think_correct
                if (not seed_correct) and think_correct:
                    helpful += 1
                    helpful_qids.append(str(row.get("qid")))
                if seed_correct and (not think_correct):
                    harmful += 1
                    harmful_qids.append(str(row.get("qid")))
                if (not seed_correct) and (not think_correct):
                    wrong_to_wrong += 1
        final_correct += int(row_final_correct)
        total_tokens.append(tokens)

    n = max(1, len(rows))
    return {
        "count": len(rows),
        "seed_correct": seed_correct_count,
        "think_correct": think_correct_count,
        "final_correct": final_correct,
        "accuracy": final_correct / n,
        "delta_correct": final_correct - seed_correct_count,
        "trigger_count": triggered,
        "trigger_rate": triggered / n,
        "adoption_count": adopted,
        "adoption_rate": adopted / n,
        "helpful": helpful,
        "harmful": harmful,
        "wrong_to_wrong": wrong_to_wrong,
        "avg_total_tokens": sum(total_tokens) / len(total_tokens) if total_tokens else None,
        "avg_triggered_extra_tokens": (
            sum(triggered_extra_tokens) / len(triggered_extra_tokens)
            if triggered_extra_tokens
            else 0.0
        ),
        "helpful_qids": helpful_qids,
        "harmful_qids": harmful_qids,
    }


def _heuristic_scores(rows: list[dict[str, Any]]) -> list[float]:
    return [1.0 if bool(row.get("old_trigger_nonbranch_or_long300")) else 0.0 for row in rows]


def _evaluate_scores(
    *,
    method: str,
    train_scores: list[float],
    eval_scores: list[float],
    eval_rows: list[dict[str, Any]],
    rates: list[float],
    filters: list[str],
    selection_modes: list[str],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for rate in rates:
        for mode in selection_modes:
            if mode == "topk_eval":
                selected = _select_topk(eval_scores, rate)
                threshold = None
            elif mode == "threshold_from_train":
                threshold = _threshold_from_train(train_scores, rate)
                selected = {idx for idx, score in enumerate(eval_scores) if score >= threshold}
            elif mode == "heuristic_binary":
                selected = {idx for idx, score in enumerate(eval_scores) if score > 0}
                threshold = 0.0
            else:
                raise KeyError(mode)
            for adoption_filter in filters:
                row = _simulate(eval_rows, selected=selected, adoption_filter=adoption_filter)
                row.update(
                    {
                        "method": method,
                        "target_rate": rate,
                        "selection_mode": mode,
                        "threshold": threshold,
                        "adoption_filter": adoption_filter,
                    }
                )
                out.append(row)
    return out


def _random_results(
    *,
    eval_rows: list[dict[str, Any]],
    rates: list[float],
    filters: list[str],
    trials: int,
    random_seed: int,
) -> list[dict[str, Any]]:
    rng = random.Random(random_seed)
    rows: list[dict[str, Any]] = []
    n = len(eval_rows)
    for rate in rates:
        k = min(n, max(1, int(round(n * rate)))) if rate > 0 else 0
        for adoption_filter in filters:
            trial_rows = []
            for _ in range(max(1, trials)):
                selected = set(rng.sample(range(n), k)) if k > 0 else set()
                trial_rows.append(_simulate(eval_rows, selected=selected, adoption_filter=adoption_filter))
            mean_row: dict[str, Any] = {
                "method": "random",
                "target_rate": rate,
                "selection_mode": "topk_eval",
                "adoption_filter": adoption_filter,
                "random_trials": max(1, trials),
            }
            for key in [
                "accuracy",
                "delta_correct",
                "trigger_count",
                "trigger_rate",
                "adoption_count",
                "adoption_rate",
                "helpful",
                "harmful",
                "wrong_to_wrong",
                "avg_total_tokens",
                "avg_triggered_extra_tokens",
            ]:
                values = [float(row[key]) for row in trial_rows]
                mean_row[key] = sum(values) / len(values)
                mean_row[f"{key}_std"] = (
                    math.sqrt(sum((v - mean_row[key]) ** 2 for v in values) / len(values))
                    if len(values) > 1
                    else 0.0
                )
            mean_row["count"] = n
            mean_row["seed_correct"] = trial_rows[0]["seed_correct"]
            mean_row["think_correct"] = trial_rows[0]["think_correct"]
            mean_row["final_correct"] = mean_row["accuracy"] * n
            rows.append(mean_row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train calibrated BORA-Switch opportunity triggers and evaluate accuracy-cost frontiers."
    )
    parser.add_argument("--train-rollouts", required=True)
    parser.add_argument("--eval-rollouts", required=True)
    parser.add_argument("--models", default="logistic,gbdt,trace_length,heuristic,random")
    parser.add_argument("--rates", default="0.1,0.2,0.3,0.5")
    parser.add_argument("--filters", default="main,strict")
    parser.add_argument("--selection-modes", default="topk_eval,threshold_from_train")
    parser.add_argument("--harm-weight", type=float, default=2.0)
    parser.add_argument("--random-trials", type=int, default=50)
    parser.add_argument("--random-seed", type=int, default=0)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    train_rows = load_jsonl(args.train_rollouts)
    eval_rows = load_jsonl(args.eval_rollouts)
    if not train_rows or not eval_rows:
        raise ValueError("Both train and eval rollouts must be non-empty.")
    feature_names = _feature_names(train_rows + eval_rows)
    rates = [float(item) for item in args.rates.split(",") if item.strip()]
    filters = [item.strip() for item in args.filters.split(",") if item.strip()]
    selection_modes = [item.strip() for item in args.selection_modes.split(",") if item.strip()]
    models = [item.strip() for item in args.models.split(",") if item.strip()]

    results: list[dict[str, Any]] = []
    model_info: dict[str, Any] = {}
    for model_name in models:
        if model_name == "random":
            results.extend(
                _random_results(
                    eval_rows=eval_rows,
                    rates=rates,
                    filters=filters,
                    trials=int(args.random_trials),
                    random_seed=int(args.random_seed),
                )
            )
            continue
        if model_name == "heuristic":
            train_scores = _heuristic_scores(train_rows)
            eval_scores = _heuristic_scores(eval_rows)
            model_info[model_name] = {"kind": "old_trigger_nonbranch_or_long300"}
            results.extend(
                _evaluate_scores(
                    method=model_name,
                    train_scores=train_scores,
                    eval_scores=eval_scores,
                    eval_rows=eval_rows,
                    rates=[0.0],
                    filters=filters,
                    selection_modes=["heuristic_binary"],
                )
            )
            continue
        train_scores, eval_scores, eval_harm_prob, info = _fit_scores(
            train_rows,
            eval_rows,
            model_name=model_name,
            feature_names=feature_names,
            harm_weight=float(args.harm_weight),
            random_seed=int(args.random_seed),
        )
        model_info[model_name] = info
        results.extend(
            _evaluate_scores(
                method=model_name,
                train_scores=train_scores,
                eval_scores=eval_scores,
                eval_rows=eval_rows,
                rates=rates,
                filters=filters,
                selection_modes=selection_modes,
            )
        )

    results.sort(key=lambda row: (row["adoption_filter"], row["selection_mode"], row["target_rate"], row["method"]))
    payload = {
        "inputs": vars(args),
        "feature_names": feature_names,
        "model_info": model_info,
        "train_summary": {
            "count": len(train_rows),
            "seed_correct": sum(bool(row.get("seed_correct")) for row in train_rows),
            "think_correct": sum(bool(row.get("think_correct")) for row in train_rows),
            "helpful": sum(bool(row.get("helpful")) for row in train_rows),
            "harmful": sum(bool(row.get("harmful")) for row in train_rows),
        },
        "eval_summary": {
            "count": len(eval_rows),
            "seed_correct": sum(bool(row.get("seed_correct")) for row in eval_rows),
            "think_correct": sum(bool(row.get("think_correct")) for row in eval_rows),
            "helpful": sum(bool(row.get("helpful")) for row in eval_rows),
            "harmful": sum(bool(row.get("harmful")) for row in eval_rows),
        },
        "results": results,
    }
    dump_json(args.output, payload)

    print(f"wrote frontier to {args.output}")
    print("top zero-harm rows by accuracy then tokens:")
    zero_harm = [row for row in results if float(row.get("harmful", 0)) <= 1e-9]
    zero_harm.sort(key=lambda row: (-float(row["accuracy"]), float(row["avg_total_tokens"] or 0)))
    for row in zero_harm[:20]:
        print(
            f"{row['method']} {row['selection_mode']} rate={row['target_rate']} "
            f"filter={row['adoption_filter']} acc={row['accuracy']*100:.2f}% "
            f"delta={row['delta_correct']} trigger={row['trigger_count']} "
            f"helpful={row['helpful']} harmful={row['harmful']} "
            f"tokens={row['avg_total_tokens']:.2f}"
        )


if __name__ == "__main__":
    main()
