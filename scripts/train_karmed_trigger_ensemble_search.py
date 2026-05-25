from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from bora.common import dump_json

from train_karmed_budget_selector_fast import (  # type: ignore
    _arm_label,
    _attach_questions,
    _fit_split_scores,
    _load_jsonl,
    _mean,
    _oracle_actions,
    _prepare_rows,
    _simulate,
    _std,
)


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
        writer.writerows(rows)


def _feature_value(row: dict[str, Any], *names: str) -> float:
    features = row.get("features") or {}
    for name in names:
        value = features.get(name)
        if value is None:
            value = row.get(name)
        try:
            out = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(out):
            return out
    return 0.0


def _normalize(values: list[float]) -> list[float]:
    if not values:
        return []
    mu = sum(values) / len(values)
    var = sum((value - mu) ** 2 for value in values) / max(1, len(values) - 1)
    sigma = math.sqrt(var)
    if sigma < 1e-12:
        return [0.0 for _ in values]
    return [(value - mu) / sigma for value in values]


def _pair_best(scores: dict[str, Any], qid: str, arms: list[int]) -> tuple[float, int]:
    return max((float(scores["pair"][(qid, arm)]), arm) for arm in arms)


def _component_matrix(rows: list[dict[str, Any]], scores: dict[str, Any], arms: list[int]) -> dict[str, list[float]]:
    pair = []
    trace = []
    seed_tokens = []
    for row in rows:
        qid = str(row["qid"])
        pair.append(_pair_best(scores, qid, arms)[0])
        trace.append(_feature_value(row, "sf_trace_words", "seed_total_tokens"))
        seed_tokens.append(float(row.get("seed_total_tokens") or 0.0))
    return {
        "q_any": _normalize([float(value) for value in scores["q_any"]]),
        "q_12": _normalize([float(value) for value in scores["q_12"]]),
        "pair": _normalize(pair),
        "trace": _normalize(trace),
        "seed_tokens": _normalize(seed_tokens),
    }


def _blend_scores(components: dict[str, list[float]], weights: dict[str, float]) -> list[float]:
    n = len(next(iter(components.values()))) if components else 0
    out = [0.0 for _ in range(n)]
    for name, weight in weights.items():
        values = components.get(name)
        if values is None:
            continue
        out = [score + float(weight) * value for score, value in zip(out, values)]
    return out


def _actions_from_scores(
    rows: list[dict[str, Any]],
    scores: dict[str, Any],
    arms: list[int],
    blended: list[float],
    *,
    rate: float,
    arm_mode: str,
) -> dict[str, int | None]:
    k = min(len(rows), max(1, int(round(len(rows) * rate))))
    order = sorted(range(len(rows)), key=lambda idx: blended[idx], reverse=True)
    actions: dict[str, int | None] = {}
    for idx in order[:k]:
        row = rows[idx]
        qid = str(row["qid"])
        if arm_mode == "12k":
            actions[qid] = 12
        elif arm_mode == "pair":
            pair_score, arm = _pair_best(scores, qid, arms)
            actions[qid] = arm if pair_score > 0 else None
        elif arm_mode == "pair_or_12":
            pair_score, arm = _pair_best(scores, qid, arms)
            actions[qid] = arm if pair_score > 0 else 12
        else:
            raise KeyError(f"Unknown arm mode: {arm_mode}")
    return actions


def _objective(sim: dict[str, Any], *, token_weight: float) -> float:
    return (
        float(sim.get("helpful") or 0.0)
        - 2.0 * float(sim.get("harmful") or 0.0)
        - token_weight * float(sim.get("avg_tokens") or 0.0)
    )


def _search_weights(
    rows: list[dict[str, Any]],
    scores: dict[str, Any],
    arms: list[int],
    *,
    rate: float,
    arm_mode: str,
    grid: list[float],
    token_weight: float,
) -> tuple[dict[str, float], dict[str, Any]]:
    components = _component_matrix(rows, scores, arms)
    best_weights: dict[str, float] | None = None
    best_sim: dict[str, Any] | None = None
    best_value = -1e18
    names = ["q_any", "q_12", "pair", "trace", "seed_tokens"]
    for q_any in grid:
        for q_12 in grid:
            for pair in grid:
                for trace in grid:
                    for seed_tokens in grid:
                        weights = {
                            "q_any": q_any,
                            "q_12": q_12,
                            "pair": pair,
                            "trace": trace,
                            "seed_tokens": seed_tokens,
                        }
                        if all(abs(weights[name]) < 1e-12 for name in names):
                            continue
                        blended = _blend_scores(components, weights)
                        actions = _actions_from_scores(
                            rows,
                            scores,
                            arms,
                            blended,
                            rate=rate,
                            arm_mode=arm_mode,
                        )
                        sim = _simulate(rows, actions, arms)
                        value = _objective(sim, token_weight=token_weight)
                        if value > best_value:
                            best_value = value
                            best_weights = weights
                            best_sim = sim
    if best_weights is None or best_sim is None:
        raise RuntimeError("Weight search did not evaluate any candidate.")
    return best_weights, best_sim


def _apply_weights(
    rows: list[dict[str, Any]],
    scores: dict[str, Any],
    arms: list[int],
    *,
    weights: dict[str, float],
    rate: float,
    arm_mode: str,
) -> dict[str, int | None]:
    components = _component_matrix(rows, scores, arms)
    blended = _blend_scores(components, weights)
    return _actions_from_scores(rows, scores, arms, blended, rate=rate, arm_mode=arm_mode)


def _actions_threshold_downgrade(
    rows: list[dict[str, Any]],
    scores: dict[str, Any],
    arms: list[int],
    *,
    rate: float,
    threshold: float,
    trigger: str,
) -> dict[str, int | None]:
    if trigger == "q12":
        trigger_scores = [float(value) for value in scores["q_12"]]
    elif trigger == "qany":
        trigger_scores = [float(value) for value in scores["q_any"]]
    else:
        raise KeyError(f"Unknown trigger: {trigger}")
    k = min(len(rows), max(1, int(round(len(rows) * rate))))
    order = sorted(range(len(rows)), key=lambda idx: trigger_scores[idx], reverse=True)
    actions: dict[str, int | None] = {}
    fallback = max(arms)
    lower_arms = [arm for arm in arms if arm < fallback]
    for idx in order[:k]:
        qid = str(rows[idx]["qid"])
        chosen = fallback
        for arm in lower_arms:
            if float(scores["pair"][(qid, arm)]) >= threshold:
                chosen = arm
                break
        actions[qid] = chosen
    return actions


def _search_threshold_downgrade(
    rows: list[dict[str, Any]],
    scores: dict[str, Any],
    arms: list[int],
    *,
    rate: float,
    thresholds: list[float],
    trigger: str,
    token_weight: float,
) -> tuple[float, dict[str, Any]]:
    best_threshold = thresholds[0]
    best_sim: dict[str, Any] | None = None
    best_value = -1e18
    for threshold in thresholds:
        actions = _actions_threshold_downgrade(
            rows,
            scores,
            arms,
            rate=rate,
            threshold=threshold,
            trigger=trigger,
        )
        sim = _simulate(rows, actions, arms)
        value = _objective(sim, token_weight=token_weight)
        if value > best_value:
            best_value = value
            best_threshold = threshold
            best_sim = sim
    if best_sim is None:
        raise RuntimeError("Threshold search did not evaluate any candidate.")
    return best_threshold, best_sim


def _summary(method: str, per_seed: list[dict[str, Any]], arms: list[int]) -> dict[str, Any]:
    keys = ["accept"] + [_arm_label(arm) for arm in arms]
    return {
        "method": method,
        "seed": "mean",
        "accuracy": _mean([row["accuracy"] for row in per_seed]),
        "std_accuracy": _std([row["accuracy"] for row in per_seed]),
        "avg_tokens": _mean([row["avg_tokens"] for row in per_seed]),
        "helpful": sum(int(row["helpful"]) for row in per_seed),
        "harmful": sum(int(row["harmful"]) for row in per_seed),
        "wrong_to_wrong": sum(int(row["wrong_to_wrong"]) for row in per_seed),
        "trigger_count": sum(int(row["trigger_count"]) for row in per_seed),
        **{key: sum(int(row.get(key, 0)) for row in per_seed) for key in keys},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm-rows", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", default="17,7,23")
    parser.add_argument("--arms", default="2,4,6,8,10,12")
    parser.add_argument("--rates", default="0.12,0.2,0.3,0.5")
    parser.add_argument("--positive-weight", type=float, default=40.0)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--use-text", action="store_true")
    parser.add_argument("--grid", default="0,0.25,0.5,1,2,4")
    parser.add_argument("--threshold-grid", default="0.05,0.1,0.15,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9")
    parser.add_argument("--token-weight", type=float, default=0.0)
    args = parser.parse_args()

    seeds = [int(item) for item in args.seeds.split(",") if item.strip()]
    arms = sorted(int(item) for item in args.arms.split(",") if item.strip())
    rates = [float(item) for item in args.rates.split(",") if item.strip()]
    grid = [float(item) for item in args.grid.split(",") if item.strip()]
    threshold_grid = [float(item) for item in args.threshold_grid.split(",") if item.strip()]

    rows = _load_jsonl(args.arm_rows)
    if args.use_text:
        _attach_questions(rows, args.root)
    _prepare_rows(rows, arms)
    by_seed = {seed: [row for row in rows if int(row["seed"]) == seed] for seed in seeds}

    split_scores: dict[int, dict[str, Any]] = {}
    calibration_scores: dict[int, tuple[list[dict[str, Any]], dict[str, Any]]] = {}
    for eval_seed in seeds:
        train_seeds = [seed for seed in seeds if seed != eval_seed]
        train_rows = [row for seed in train_seeds for row in by_seed[seed]]
        eval_rows = by_seed[eval_seed]
        split_scores[eval_seed] = _fit_split_scores(
            train_rows,
            eval_rows,
            arms,
            positive_weight=float(args.positive_weight),
            random_seed=eval_seed,
            use_text=bool(args.use_text),
        )
        # Out-of-fold calibration on the two training seeds.  Each calibration
        # seed is scored by a model trained on the other calibration seed, so the
        # weight search does not use in-sample labels for its score components.
        cal_rows: list[dict[str, Any]] = []
        cal_scores_parts: list[dict[str, Any]] = []
        for cal_seed in train_seeds:
            support_seed = [seed for seed in train_seeds if seed != cal_seed][0]
            support_rows = by_seed[support_seed]
            part_rows = by_seed[cal_seed]
            part_scores = _fit_split_scores(
                support_rows,
                part_rows,
                arms,
                positive_weight=float(args.positive_weight),
                random_seed=eval_seed * 100 + cal_seed,
                use_text=bool(args.use_text),
            )
            cal_rows.extend(part_rows)
            cal_scores_parts.append(part_scores)
        merged_scores: dict[str, Any] = {"q_any": [], "q_12": [], "pair": {}, "distill": {}, "qid_prior": {}}
        for part_rows, part_scores in zip([by_seed[s] for s in train_seeds], cal_scores_parts):
            offset_q_any = part_scores["q_any"]
            offset_q_12 = part_scores["q_12"]
            merged_scores["q_any"].extend(offset_q_any)
            merged_scores["q_12"].extend(offset_q_12)
            for row in part_rows:
                qid = str(row["qid"])
                for arm in arms:
                    merged_scores["pair"][(qid, arm)] = part_scores["pair"][(qid, arm)]
        calibration_scores[eval_seed] = (cal_rows, merged_scores)

    result_rows: list[dict[str, Any]] = []
    for rate in rates:
        for arm_mode in ["12k", "pair", "pair_or_12"]:
            method = f"calib_ensemble_{arm_mode}_top{int(round(rate * 100))}"
            per_seed: list[dict[str, Any]] = []
            for eval_seed in seeds:
                cal_rows, cal_scores = calibration_scores[eval_seed]
                weights, cal_sim = _search_weights(
                    cal_rows,
                    cal_scores,
                    arms,
                    rate=rate,
                    arm_mode=arm_mode,
                    grid=grid,
                    token_weight=float(args.token_weight),
                )
                eval_rows = by_seed[eval_seed]
                actions = _apply_weights(
                    eval_rows,
                    split_scores[eval_seed],
                    arms,
                    weights=weights,
                    rate=rate,
                    arm_mode=arm_mode,
                )
                sim = _simulate(eval_rows, actions, arms)
                row = {
                    "method": method,
                    "seed": eval_seed,
                    **sim,
                    "weights": json.dumps(weights, sort_keys=True),
                    "calibration_helpful": cal_sim.get("helpful"),
                    "calibration_accuracy": cal_sim.get("accuracy"),
                    "calibration_tokens": cal_sim.get("avg_tokens"),
                }
                result_rows.append(row)
                per_seed.append(row)
            result_rows.append(_summary(method, per_seed, arms))

        for trigger in ["q12", "qany"]:
            method = f"calib_thresh_{trigger}_top{int(round(rate * 100))}"
            per_seed = []
            for eval_seed in seeds:
                cal_rows, cal_scores = calibration_scores[eval_seed]
                threshold, cal_sim = _search_threshold_downgrade(
                    cal_rows,
                    cal_scores,
                    arms,
                    rate=rate,
                    thresholds=threshold_grid,
                    trigger=trigger,
                    token_weight=float(args.token_weight),
                )
                eval_rows = by_seed[eval_seed]
                actions = _actions_threshold_downgrade(
                    eval_rows,
                    split_scores[eval_seed],
                    arms,
                    rate=rate,
                    threshold=threshold,
                    trigger=trigger,
                )
                sim = _simulate(eval_rows, actions, arms)
                row = {
                    "method": method,
                    "seed": eval_seed,
                    **sim,
                    "threshold": threshold,
                    "calibration_helpful": cal_sim.get("helpful"),
                    "calibration_accuracy": cal_sim.get("accuracy"),
                    "calibration_tokens": cal_sim.get("avg_tokens"),
                }
                result_rows.append(row)
                per_seed.append(row)
            result_rows.append(_summary(method, per_seed, arms))

    oracle_per = []
    for seed in seeds:
        eval_rows = by_seed[seed]
        actions = _oracle_actions(eval_rows, arms)
        sim = _simulate(eval_rows, actions, arms)
        row = {"method": "oracle_minimal", "seed": seed, **sim}
        oracle_per.append(row)
        result_rows.append(row)
    result_rows.append(_summary("oracle_minimal", oracle_per, arms))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "math500_karmed_ensemble_search_results.csv"
    json_path = args.output_dir / "math500_karmed_ensemble_search_results.json"
    _write_csv(csv_path, result_rows)
    dump_json(
        json_path,
        {
            "config": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
            "results": result_rows,
        },
    )
    print(f"Wrote {csv_path}")
    for row in result_rows:
        if row["seed"] == "mean":
            print(
                row["method"],
                f"acc={100*row['accuracy']:.2f}",
                f"tok={row['avg_tokens']:.1f}",
                f"help={row['helpful']}",
                f"harm={row['harmful']}",
                "arms=" + ",".join(f"{key}:{row.get(key,0)}" for key in ["accept"] + [_arm_label(arm) for arm in arms]),
            )


if __name__ == "__main__":
    main()
