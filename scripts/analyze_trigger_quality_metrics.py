from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(out):
        return default
    return out


def _load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _dump_json(path: str | Path, payload: Any) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _feature_names(rows: list[dict[str, Any]]) -> list[str]:
    names: set[str] = set()
    for row in rows:
        names.update((row.get("features") or {}).keys())
    return sorted(names)


def _matrix(rows: list[dict[str, Any]], names: list[str]) -> list[list[float]]:
    return [[_as_float((row.get("features") or {}).get(name), 0.0) for name in names] for row in rows]


def _labels(rows: list[dict[str, Any]], name: str) -> list[int]:
    return [int(bool(row.get(name))) for row in rows]


def _extra_costs(rows: list[dict[str, Any]]) -> list[float]:
    return [_as_float(row.get("think_total_tokens"), 0.0) for row in rows]


def _trace_scores(rows: list[dict[str, Any]]) -> list[float]:
    return [
        _as_float((row.get("features") or {}).get("sf_trace_words"), 0.0)
        or _as_float((row.get("features") or {}).get("seed_total_tokens"), 0.0)
        for row in rows
    ]


def _problem_shape_feature_names(names: list[str]) -> list[str]:
    prefixes = (
        "question_has_",
        "question_numeric_count",
        "question_char_len",
        "question_word_len",
    )
    return [name for name in names if name.startswith(prefixes)]


def _parse_feature_names(names: list[str]) -> list[str]:
    prefixes = ("seed_answer_",)
    return [name for name in names if name.startswith(prefixes)]


def _trace_feature_names(names: list[str]) -> list[str]:
    needles = ("trace", "token", "word", "char", "numeric_count")
    return [name for name in names if any(needle in name for needle in needles)]


def _select_features(all_names: list[str], group: str) -> list[str]:
    if group == "all":
        return all_names
    if group == "trace":
        return _trace_feature_names(all_names)
    if group == "parse":
        return _parse_feature_names(all_names)
    if group == "problem_shape":
        return _problem_shape_feature_names(all_names)
    if group == "all_minus_trace":
        trace = set(_trace_feature_names(all_names))
        return [name for name in all_names if name not in trace]
    raise KeyError(f"Unknown feature group: {group}")


def _constant_scores(rows: list[dict[str, Any]], label: str) -> list[float]:
    y = _labels(rows, label)
    p = sum(y) / max(1, len(y))
    return [p for _ in rows]


def _fit_classifier_scores(
    train_rows: list[dict[str, Any]],
    eval_rows: list[dict[str, Any]],
    *,
    model_name: str,
    feature_names: list[str],
    label: str,
    random_seed: int,
) -> tuple[list[float], dict[str, Any]]:
    y_train = _labels(train_rows, label)
    if model_name == "trace":
        raw = _trace_scores(eval_rows)
        if not raw:
            return [], {"kind": "trace"}
        lo, hi = min(raw), max(raw)
        if hi <= lo:
            return [0.5 for _ in raw], {"kind": "trace", "constant": True}
        return [(value - lo) / (hi - lo) for value in raw], {"kind": "trace_minmax"}
    if len(set(y_train)) < 2 or not feature_names:
        return _constant_scores(train_rows, label)[:0] + _constant_scores(eval_rows, label), {
            "kind": "constant",
            "p": sum(y_train) / max(1, len(y_train)),
        }

    try:
        from sklearn.ensemble import HistGradientBoostingClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("scikit-learn is required for trigger quality metrics.") from exc

    x_train = _matrix(train_rows, feature_names)
    x_eval = _matrix(eval_rows, feature_names)
    if model_name == "logistic":
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, class_weight="balanced", random_state=random_seed),
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
    return [float(row[1]) for row in model.predict_proba(x_eval)], {
        "kind": model_name,
        "features": len(feature_names),
    }


def _fit_cost_predictions(
    train_rows: list[dict[str, Any]],
    eval_rows: list[dict[str, Any]],
    *,
    model_name: str,
    feature_names: list[str],
    random_seed: int,
) -> tuple[list[float], dict[str, Any]]:
    y_train = _extra_costs(train_rows)
    if model_name == "trace":
        # Simple proportional trace baseline. Fit one scalar slope on train.
        x_train = _trace_scores(train_rows)
        x_eval = _trace_scores(eval_rows)
        denom = sum(value * value for value in x_train)
        slope = sum(x * y for x, y in zip(x_train, y_train)) / denom if denom > 0 else 0.0
        if slope <= 0:
            slope = sum(y_train) / max(1, len(y_train))
            return [slope for _ in eval_rows], {"kind": "trace_constant"}
        return [max(0.0, slope * value) for value in x_eval], {"kind": "trace_slope", "slope": slope}
    if not feature_names:
        mean_cost = sum(y_train) / max(1, len(y_train))
        return [mean_cost for _ in eval_rows], {"kind": "constant", "mean": mean_cost}

    try:
        from sklearn.ensemble import HistGradientBoostingRegressor
        from sklearn.linear_model import Ridge
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("scikit-learn is required for cost prediction metrics.") from exc

    x_train = _matrix(train_rows, feature_names)
    x_eval = _matrix(eval_rows, feature_names)
    if model_name == "logistic":
        model = make_pipeline(StandardScaler(), Ridge(alpha=1.0, random_state=random_seed))
    elif model_name == "gbdt":
        model = HistGradientBoostingRegressor(
            max_iter=200,
            learning_rate=0.05,
            l2_regularization=0.01,
            random_state=random_seed,
        )
    else:
        raise KeyError(f"Unknown cost model: {model_name}")
    model.fit(x_train, y_train)
    return [max(0.0, float(value)) for value in model.predict(x_eval)], {
        "kind": model_name,
        "features": len(feature_names),
    }


def _binary_metrics(y_true: list[int], scores: list[float]) -> dict[str, Any]:
    try:
        from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("scikit-learn is required for trigger quality metrics.") from exc

    positives = sum(y_true)
    out: dict[str, Any] = {
        "positives": positives,
        "prevalence": positives / max(1, len(y_true)),
        "n": len(y_true),
    }
    if len(set(y_true)) < 2:
        out.update({"auroc": None, "auprc": None})
    else:
        out["auroc"] = float(roc_auc_score(y_true, scores))
        out["auprc"] = float(average_precision_score(y_true, scores))
    clipped = [min(1.0, max(0.0, float(score))) for score in scores]
    out["brier"] = float(brier_score_loss(y_true, clipped)) if y_true else None
    out["ece_10"] = _ece(y_true, clipped, bins=10)
    for rate in (0.1, 0.2, 0.3, 0.5):
        out[f"precision_at_{int(rate * 100)}"] = _precision_at_rate(y_true, scores, rate)
        out[f"recall_at_{int(rate * 100)}"] = _recall_at_rate(y_true, scores, rate)
    return out


def _precision_at_rate(y_true: list[int], scores: list[float], rate: float) -> float | None:
    if not y_true:
        return None
    k = max(1, min(len(y_true), int(round(len(y_true) * rate))))
    top = sorted(range(len(scores)), key=lambda idx: scores[idx], reverse=True)[:k]
    return sum(y_true[idx] for idx in top) / k


def _recall_at_rate(y_true: list[int], scores: list[float], rate: float) -> float | None:
    positives = sum(y_true)
    if positives <= 0:
        return None
    k = max(1, min(len(y_true), int(round(len(y_true) * rate))))
    top = sorted(range(len(scores)), key=lambda idx: scores[idx], reverse=True)[:k]
    return sum(y_true[idx] for idx in top) / positives


def _ece(y_true: list[int], scores: list[float], *, bins: int) -> float | None:
    if not y_true:
        return None
    total = 0.0
    n = len(y_true)
    for idx in range(bins):
        lo = idx / bins
        hi = (idx + 1) / bins
        bucket = [
            j
            for j, score in enumerate(scores)
            if (lo <= score < hi) or (idx == bins - 1 and score == hi)
        ]
        if not bucket:
            continue
        conf = sum(scores[j] for j in bucket) / len(bucket)
        acc = sum(y_true[j] for j in bucket) / len(bucket)
        total += len(bucket) / n * abs(conf - acc)
    return total


def _regression_metrics(y_true: list[float], pred: list[float]) -> dict[str, Any]:
    if not y_true:
        return {"n": 0, "mae": None, "rmse": None, "spearman": None}
    errors = [p - y for p, y in zip(pred, y_true)]
    mae = sum(abs(err) for err in errors) / len(errors)
    rmse = math.sqrt(sum(err * err for err in errors) / len(errors))
    return {
        "n": len(y_true),
        "mae": mae,
        "rmse": rmse,
        "spearman": _spearman(y_true, pred),
    }


def _spearman(a: list[float], b: list[float]) -> float | None:
    if len(a) < 2:
        return None
    try:
        from scipy.stats import spearmanr
    except Exception:
        return None
    value = spearmanr(a, b).correlation
    return None if value is None or not math.isfinite(float(value)) else float(value)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = sorted({key for row in rows for key in row.keys()})
    lines = [",".join(keys)]
    for row in rows:
        values = []
        for key in keys:
            value = row.get(key)
            text = "" if value is None else json.dumps(value) if isinstance(value, (dict, list)) else str(value)
            if "," in text or "\n" in text or '"' in text:
                text = '"' + text.replace('"', '""') + '"'
            values.append(text)
        lines.append(",".join(values))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Report AUROC/AUPRC/calibration/cost metrics for BORA trigger models."
    )
    parser.add_argument("--train-rollouts", nargs="+", required=True)
    parser.add_argument("--eval-rollouts", nargs="+", required=True)
    parser.add_argument("--models", default="gbdt,logistic,trace")
    parser.add_argument("--feature-groups", default="all,trace,parse,problem_shape,all_minus_trace")
    parser.add_argument("--random-seed", type=int, default=0)
    parser.add_argument("--output", required=True)
    parser.add_argument("--output-csv", default=None)
    args = parser.parse_args()

    train_rows = [row for path in args.train_rollouts for row in _load_jsonl(path)]
    eval_rows = [row for path in args.eval_rollouts for row in _load_jsonl(path)]
    if not train_rows or not eval_rows:
        raise ValueError("Both train and eval rollouts must be non-empty.")

    all_feature_names = _feature_names(train_rows + eval_rows)
    models = [item.strip() for item in args.models.split(",") if item.strip()]
    feature_groups = [item.strip() for item in args.feature_groups.split(",") if item.strip()]

    rows: list[dict[str, Any]] = []
    for feature_group in feature_groups:
        names = _select_features(all_feature_names, feature_group)
        for model in models:
            if model == "trace" and feature_group != "trace":
                continue
            for label in ("helpful", "harmful"):
                scores, model_info = _fit_classifier_scores(
                    train_rows,
                    eval_rows,
                    model_name=model,
                    feature_names=names,
                    label=label,
                    random_seed=int(args.random_seed),
                )
                metrics = _binary_metrics(_labels(eval_rows, label), scores)
                rows.append(
                    {
                        "task": label,
                        "model": model,
                        "feature_group": feature_group,
                        "feature_count": len(names),
                        "model_info": model_info,
                        **metrics,
                    }
                )

            pred_cost, cost_info = _fit_cost_predictions(
                train_rows,
                eval_rows,
                model_name=model,
                feature_names=names,
                random_seed=int(args.random_seed),
            )
            rows.append(
                {
                    "task": "cost",
                    "model": model,
                    "feature_group": feature_group,
                    "feature_count": len(names),
                    "model_info": cost_info,
                    **_regression_metrics(_extra_costs(eval_rows), pred_cost),
                }
            )

    payload = {
        "inputs": vars(args),
        "train_summary": {
            "count": len(train_rows),
            "helpful": sum(_labels(train_rows, "helpful")),
            "harmful": sum(_labels(train_rows, "harmful")),
            "avg_extra_cost": sum(_extra_costs(train_rows)) / max(1, len(train_rows)),
        },
        "eval_summary": {
            "count": len(eval_rows),
            "helpful": sum(_labels(eval_rows, "helpful")),
            "harmful": sum(_labels(eval_rows, "harmful")),
            "avg_extra_cost": sum(_extra_costs(eval_rows)) / max(1, len(eval_rows)),
        },
        "results": rows,
    }
    _dump_json(args.output, payload)
    if args.output_csv:
        _write_csv(Path(args.output_csv), rows)

    print(f"wrote trigger quality metrics to {args.output}")
    for row in rows:
        if row["task"] in {"helpful", "harmful"} and row.get("auprc") is not None:
            print(
                f"{row['task']} {row['model']}:{row['feature_group']} "
                f"auroc={row.get('auroc')} auprc={row.get('auprc')} "
                f"p@30={row.get('precision_at_30')}"
            )


if __name__ == "__main__":
    main()
