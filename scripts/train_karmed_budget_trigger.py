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

from bora.common import dump_json, is_correct, load_jsonl, normalize_answer

from analyze_trigger_frontier import _feature_names, _fit_scores  # type: ignore
from summarize_bora_switch_calibrated import (  # type: ignore
    NO_THINK_PATHS,
    OPPORTUNITY_PATHS,
    SEEDS,
    THINK8K_PATHS,
    THINK12K_PATHS,
    _load_json,
    _mean,
    _records,
    _std,
)


ARM_CAPS = {
    2: 2048,
    4: 4096,
    6: 6144,
    8: 8192,
    10: 10240,
    12: 12288,
}


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


def _load_result_records(path: Path) -> dict[str, dict[str, Any]]:
    payload = _load_json(path)
    return {str(row.get("qid") or row.get("id")): row for row in _records(payload)}


def _num(value: Any) -> float | None:
    norm = normalize_answer(value)
    if norm is None:
        return None
    try:
        return float(norm.replace(",", ""))
    except Exception:
        return None


def _sign(value: float | None) -> int:
    if value is None or value == 0:
        return 0
    return 1 if value > 0 else -1


def _token_value(record: dict[str, Any], *keys: str) -> float:
    metadata = record.get("metadata") or record.get("execution_metadata") or {}
    for key in keys:
        value = record.get(key)
        if value is None:
            value = metadata.get(key)
        try:
            out = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(out):
            return out
    return 0.0


def _record_answer(record: dict[str, Any]) -> Any:
    for key in ("prediction", "thinking_answer", "answer", "final_answer"):
        if record.get(key) is not None:
            return record.get(key)
    return None


def _near_cap(record: dict[str, Any], cap_tokens: int, *, cap_ratio: float, cap_margin: int) -> bool:
    metadata = record.get("metadata") or record.get("execution_metadata") or {}
    max_new_tokens = int(metadata.get("max_new_tokens") or cap_tokens)
    completion_tokens = _token_value(record, "solver_tokens", "extra_total_tokens", "completion_tokens")
    if completion_tokens <= 0:
        completion_tokens = _token_value(record, "total_tokens")
    if max_new_tokens <= 0:
        return False
    return completion_tokens >= int(max_new_tokens * cap_ratio) or (
        cap_margin > 0 and completion_tokens >= max_new_tokens - cap_margin
    )


def _strict_gate(seed_answer: Any, think_answer: Any, record: dict[str, Any], cap_tokens: int, args: argparse.Namespace) -> bool:
    seed_value = _num(seed_answer)
    think_value = _num(think_answer)
    if think_value is None:
        return False
    if _near_cap(record, cap_tokens, cap_ratio=float(args.cap_ratio), cap_margin=int(args.cap_margin_tokens)):
        return False
    if bool(args.sign_guard):
        seed_sign = _sign(seed_value)
        think_sign = _sign(think_value)
        if seed_sign != 0 and think_sign != 0 and seed_sign != think_sign:
            return False
    return True


def _lowercap_path(root: Path, lowercap_root: Path, arm: int, seed: int) -> Path:
    return root / lowercap_root / f"think{arm}k_seed{seed}" / f"standard_direct_cot_think{arm}k_math500_seed{seed}.json"


def _arm_result_path(root: Path, lowercap_root: Path, arm: int, seed: int) -> Path:
    if arm == 8:
        return root / THINK8K_PATHS[seed]
    if arm == 12:
        return root / THINK12K_PATHS[seed]
    return _lowercap_path(root, lowercap_root, arm, seed)


def _arm_label(arm: int) -> str:
    return f"think{arm}"


def _build_rows(root: Path, lowercap_root: Path, arms: list[int], seed: int, args: argparse.Namespace) -> list[dict[str, Any]]:
    base_rows = load_jsonl(root / OPPORTUNITY_PATHS[seed])
    seed_records = _load_result_records(root / NO_THINK_PATHS[seed])
    arm_records: dict[int, dict[str, dict[str, Any]]] = {}
    for arm in arms:
        path = _arm_result_path(root, lowercap_root, arm, seed)
        if not path.exists():
            raise FileNotFoundError(f"Missing arm result for seed={seed}, arm={arm}k: {path}")
        arm_records[arm] = _load_result_records(path)

    out: list[dict[str, Any]] = []
    for row in base_rows:
        qid = str(row["qid"])
        seed_record = seed_records.get(qid, {})
        seed_answer = row.get("seed_answer")
        gold = row.get("gold_answer")
        seed_correct = bool(row.get("seed_correct"))
        seed_tokens = float(row.get("seed_total_tokens") or _token_value(seed_record, "total_tokens", "solver_tokens"))
        item: dict[str, Any] = {
            "qid": qid,
            "seed": seed,
            "gold_answer": gold,
            "seed_answer": seed_answer,
            "seed_answer_normalized": normalize_answer(seed_answer),
            "seed_correct": seed_correct,
            "seed_total_tokens": seed_tokens,
            "features": row.get("features") or {},
            "old_trigger_nonbranch_or_long300": bool(row.get("old_trigger_nonbranch_or_long300")),
        }
        for arm in arms:
            label = _arm_label(arm)
            record = arm_records[arm].get(qid)
            if record is None:
                raise KeyError(f"Missing qid={qid} in arm {arm}k seed={seed}")
            think_answer = _record_answer(record)
            think_tokens = _token_value(record, "total_tokens", "solver_tokens", "extra_total_tokens")
            cap_tokens = ARM_CAPS.get(arm, arm * 1024)
            gate_pass = _strict_gate(seed_answer, think_answer, record, cap_tokens, args)
            final_answer = think_answer if gate_pass else seed_answer
            raw_correct = bool(record.get("correct")) if record.get("correct") is not None else is_correct(think_answer, gold)
            final_correct = is_correct(final_answer, gold)
            item.update(
                {
                    f"{label}_answer": think_answer,
                    f"{label}_answer_normalized": normalize_answer(think_answer),
                    f"{label}_raw_correct": raw_correct,
                    f"{label}_gate_pass": gate_pass,
                    f"{label}_final_correct": final_correct,
                    f"{label}_total_tokens": think_tokens,
                    f"{label}_helpful": (not seed_correct) and final_correct,
                    f"{label}_harmful": seed_correct and (not final_correct),
                    f"{label}_wrong_to_wrong": (not seed_correct) and (not final_correct) and gate_pass,
                }
            )
        out.append(item)
    return out


def _simulate(rows: list[dict[str, Any]], actions: dict[str, int | None], arms: list[int]) -> dict[str, Any]:
    correct = 0
    helpful = harmful = wrong_to_wrong = adopted = triggered = 0
    tokens: list[float] = []
    arm_counts = {f"think{arm}": 0 for arm in arms}
    arm_counts["accept"] = 0
    helpful_qids: list[str] = []
    harmful_qids: list[str] = []
    for row in rows:
        qid = str(row["qid"])
        arm = actions.get(qid)
        seed_correct = bool(row["seed_correct"])
        seed_tokens = float(row.get("seed_total_tokens") or 0.0)
        if arm is None:
            arm_counts["accept"] += 1
            final_correct = seed_correct
            total = seed_tokens
        else:
            label = _arm_label(int(arm))
            arm_counts[label] += 1
            triggered += 1
            final_correct = bool(row[f"{label}_final_correct"])
            total = seed_tokens + float(row.get(f"{label}_total_tokens") or 0.0)
            if bool(row.get(f"{label}_gate_pass")):
                adopted += 1
            if (not seed_correct) and final_correct:
                helpful += 1
                helpful_qids.append(qid)
            if seed_correct and (not final_correct):
                harmful += 1
                harmful_qids.append(qid)
            if (not seed_correct) and (not final_correct) and bool(row.get(f"{label}_gate_pass")):
                wrong_to_wrong += 1
        correct += int(final_correct)
        tokens.append(total)
    n = max(1, len(rows))
    return {
        "count": len(rows),
        "correct": correct,
        "accuracy": correct / n,
        "avg_tokens": _mean(tokens),
        "trigger_count": triggered,
        "trigger_rate": triggered / n,
        "adoption_count": adopted,
        "adoption_rate": adopted / n,
        "helpful": helpful,
        "harmful": harmful,
        "wrong_to_wrong": wrong_to_wrong,
        "helpful_qids": helpful_qids,
        "harmful_qids": harmful_qids,
        **arm_counts,
    }


def _oracle_minimal(rows: list[dict[str, Any]], arms: list[int]) -> dict[str, int | None]:
    actions: dict[str, int | None] = {}
    for row in rows:
        qid = str(row["qid"])
        action: int | None = None
        for arm in arms:
            if bool(row.get(f"{_arm_label(arm)}_helpful")):
                action = arm
                break
        actions[qid] = action
    return actions


def _oracle_budget_greedy(rows: list[dict[str, Any]], arms: list[int], max_avg_tokens: float) -> dict[str, int | None]:
    actions = {str(row["qid"]): None for row in rows}
    total_tokens = sum(float(row.get("seed_total_tokens") or 0.0) for row in rows)
    budget_total = max_avg_tokens * max(1, len(rows))
    candidates: list[tuple[float, float, int, str]] = []
    for row in rows:
        seed_gain_base = int(bool(row["seed_correct"]))
        for arm in arms:
            label = _arm_label(arm)
            gain = int(bool(row[f"{label}_final_correct"])) - seed_gain_base
            extra = float(row.get(f"{label}_total_tokens") or 0.0)
            if gain <= 0 or extra <= 0:
                continue
            # Prefer lower budgets on ties.
            candidates.append((gain / extra, -extra, arm, str(row["qid"])))
    candidates.sort(reverse=True)
    for _ratio, neg_extra, arm, qid in candidates:
        if actions[qid] is not None:
            continue
        extra = -neg_extra
        if total_tokens + extra > budget_total:
            continue
        actions[qid] = arm
        total_tokens += extra
    return actions


def _clone_with_labels(rows: list[dict[str, Any]], *, helpful_key: str, harmful_key: str) -> list[dict[str, Any]]:
    return [{**row, "helpful": bool(row.get(helpful_key)), "harmful": bool(row.get(harmful_key))} for row in rows]


def _fit_binary_score(
    train_rows: list[dict[str, Any]],
    eval_rows: list[dict[str, Any]],
    *,
    helpful_key: str,
    harmful_key: str,
    harm_weight: float,
    random_seed: int,
) -> list[float]:
    train = _clone_with_labels(train_rows, helpful_key=helpful_key, harmful_key=harmful_key)
    eval_ = _clone_with_labels(eval_rows, helpful_key=helpful_key, harmful_key=harmful_key)
    feature_names = _feature_names(train + eval_)
    _train_scores, eval_scores, _eval_harm, _info = _fit_scores(
        train,
        eval_,
        model_name="gbdt",
        feature_names=feature_names,
        harm_weight=harm_weight,
        random_seed=random_seed,
    )
    return eval_scores


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _matrix_from_features(rows: list[dict[str, Any]], feature_names: list[str]) -> list[list[float]]:
    return [[_as_float((row.get("features") or {}).get(name), 0.0) for name in feature_names] for row in rows]


def _fit_probabilities(
    train_rows: list[dict[str, Any]],
    eval_rows: list[dict[str, Any]],
    *,
    label_key: str,
    feature_names: list[str],
    random_seed: int,
    positive_weight: float,
) -> tuple[list[float], list[float]]:
    try:
        from sklearn.ensemble import HistGradientBoostingClassifier
    except Exception as exc:  # pragma: no cover - remote dependency check.
        raise RuntimeError("scikit-learn is required for K-armed trigger training.") from exc

    y_train = [int(bool(row.get(label_key))) for row in train_rows]
    if len(set(y_train)) < 2:
        p = sum(y_train) / max(1, len(y_train))
        return [p for _ in train_rows], [p for _ in eval_rows]

    x_train = _matrix_from_features(train_rows, feature_names)
    x_eval = _matrix_from_features(eval_rows, feature_names)
    sample_weight = [positive_weight if y else 1.0 for y in y_train]
    models = [
        HistGradientBoostingClassifier(
            max_iter=180,
            learning_rate=0.05,
            l2_regularization=0.02,
            random_state=random_seed,
        ),
    ]
    train_probs = [0.0 for _ in train_rows]
    eval_probs = [0.0 for _ in eval_rows]
    used = 0
    for model in models:
        model.fit(x_train, y_train, sample_weight=sample_weight)
        train_pred = [float(row[1]) for row in model.predict_proba(x_train)]
        eval_pred = [float(row[1]) for row in model.predict_proba(x_eval)]
        train_probs = [a + b for a, b in zip(train_probs, train_pred)]
        eval_probs = [a + b for a, b in zip(eval_probs, eval_pred)]
        used += 1
    return [p / used for p in train_probs], [p / used for p in eval_probs]


def _minimal_helpful_arm(row: dict[str, Any], arms: list[int]) -> int | None:
    for arm in arms:
        if bool(row.get(f"{_arm_label(arm)}_helpful")):
            return arm
    return None


def _arm_pair_rows(rows: list[dict[str, Any]], arms: list[int]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    max_arm = max(arms)
    for row in rows:
        base_features = dict(row.get("features") or {})
        for arm in arms:
            label = _arm_label(arm)
            features = dict(base_features)
            features.update(
                {
                    "arm_budget_k": float(arm),
                    "arm_budget_frac": float(arm) / float(max_arm),
                    "arm_budget_log": math.log1p(float(arm)),
                    "arm_is_low": 1.0 if arm <= 4 else 0.0,
                    "arm_is_mid": 1.0 if 6 <= arm <= 8 else 0.0,
                    "arm_is_high": 1.0 if arm >= 10 else 0.0,
                }
            )
            for candidate in arms:
                features[f"arm_eq_{candidate}k"] = 1.0 if arm == candidate else 0.0
            out.append(
                {
                    "qid": row["qid"],
                    "seed": row["seed"],
                    "arm": arm,
                    "features": features,
                    "helpful": bool(row.get(f"{label}_helpful")),
                    "harmful": bool(row.get(f"{label}_harmful")),
                    "wrong_to_wrong": bool(row.get(f"{label}_wrong_to_wrong")),
                    "arm_tokens": float(row.get(f"{label}_total_tokens") or 0.0),
                }
            )
    return out


def _fit_pairwise_arm_scores(
    train_rows: list[dict[str, Any]],
    eval_rows: list[dict[str, Any]],
    arms: list[int],
    *,
    harm_weight: float,
    lambda_cost_per_1k: float,
    random_seed: int,
    positive_weight: float,
) -> dict[tuple[str, int], float]:
    train_pairs = _arm_pair_rows(train_rows, arms)
    eval_pairs = _arm_pair_rows(eval_rows, arms)
    feature_names = _feature_names(train_pairs + eval_pairs)
    _train_help, eval_help = _fit_probabilities(
        train_pairs,
        eval_pairs,
        label_key="helpful",
        feature_names=feature_names,
        random_seed=random_seed,
        positive_weight=positive_weight,
    )
    _train_harm, eval_harm = _fit_probabilities(
        train_pairs,
        eval_pairs,
        label_key="harmful",
        feature_names=feature_names,
        random_seed=random_seed + 101,
        positive_weight=max(positive_weight, 8.0),
    )
    out: dict[tuple[str, int], float] = {}
    for pair, p_help, p_harm in zip(eval_pairs, eval_help, eval_harm):
        score = p_help - harm_weight * p_harm - lambda_cost_per_1k * (float(pair.get("arm_tokens") or 0.0) / 1000.0)
        out[(str(pair["qid"]), int(pair["arm"]))] = score
    return out


def _train_pairwise_actions(
    train_rows: list[dict[str, Any]],
    eval_rows: list[dict[str, Any]],
    arms: list[int],
    *,
    rate: float,
    harm_weight: float,
    lambda_cost_per_1k: float,
    random_seed: int,
    positive_weight: float,
) -> dict[str, int | None]:
    pair_scores = _fit_pairwise_arm_scores(
        train_rows,
        eval_rows,
        arms,
        harm_weight=harm_weight,
        lambda_cost_per_1k=lambda_cost_per_1k,
        random_seed=random_seed,
        positive_weight=positive_weight,
    )
    ranked: list[tuple[float, str, int]] = []
    for row in eval_rows:
        qid = str(row["qid"])
        score, arm = max((pair_scores[(qid, arm)], arm) for arm in arms)
        ranked.append((score, qid, arm))
    ranked.sort(reverse=True)
    k = min(len(eval_rows), max(1, int(round(len(eval_rows) * rate))))
    return {qid: arm for score, qid, arm in ranked[:k] if score > 0}


def _train_conservative_downgrade_actions(
    train_rows: list[dict[str, Any]],
    eval_rows: list[dict[str, Any]],
    arms: list[int],
    *,
    rate: float,
    harm_weight: float,
    lambda_cost_per_1k: float,
    random_seed: int,
    positive_weight: float,
    margin: float,
) -> dict[str, int | None]:
    # Preserve the strong 12k trigger for recall, then only downgrade when a cheaper
    # arm is predicted to be nearly as useful. This targets same-accuracy/lower-token
    # improvements rather than risky full multi-arm replacement.
    trigger_actions = _train_12k_only_actions(
        train_rows,
        eval_rows,
        rate=rate,
        harm_weight=harm_weight,
        random_seed=random_seed,
    )
    pair_scores = _fit_pairwise_arm_scores(
        train_rows,
        eval_rows,
        arms,
        harm_weight=harm_weight,
        lambda_cost_per_1k=lambda_cost_per_1k,
        random_seed=random_seed + 211,
        positive_weight=positive_weight,
    )
    actions: dict[str, int | None] = {}
    for row in eval_rows:
        qid = str(row["qid"])
        if qid not in trigger_actions:
            actions[qid] = None
            continue
        score12 = pair_scores[(qid, 12)] if 12 in arms else max(pair_scores[(qid, arm)] for arm in arms)
        eligible = [
            arm
            for arm in arms
            if pair_scores[(qid, arm)] >= score12 - margin and pair_scores[(qid, arm)] > 0
        ]
        actions[qid] = min(eligible) if eligible else 12
    return actions


def _train_oracle_distill_actions(
    train_rows: list[dict[str, Any]],
    eval_rows: list[dict[str, Any]],
    arms: list[int],
    *,
    rate: float,
    random_seed: int,
    positive_weight: float,
) -> dict[str, int | None]:
    try:
        from sklearn.ensemble import HistGradientBoostingClassifier
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("scikit-learn is required for oracle distillation.") from exc

    class_values = [0] + arms
    train = []
    eval_ = []
    for row in train_rows:
        arm = _minimal_helpful_arm(row, arms)
        train.append({**row, "oracle_class": int(arm or 0)})
    for row in eval_rows:
        eval_.append({**row, "oracle_class": 0})
    feature_names = _feature_names(train + eval_)
    x_train = _matrix_from_features(train, feature_names)
    x_eval = _matrix_from_features(eval_, feature_names)
    y_train = [int(row["oracle_class"]) for row in train]
    sample_weight = [positive_weight if y != 0 else 1.0 for y in y_train]
    models = [
        HistGradientBoostingClassifier(
            max_iter=200,
            learning_rate=0.05,
            l2_regularization=0.02,
            random_state=random_seed,
        ),
    ]
    eval_class_scores = [{value: 0.0 for value in class_values} for _ in eval_rows]
    used = 0
    for model in models:
        if len(set(y_train)) < 2:
            continue
        model.fit(x_train, y_train, sample_weight=sample_weight)
        classes = [int(value) for value in model.classes_]
        for idx, probs in enumerate(model.predict_proba(x_eval)):
            for cls, prob in zip(classes, probs):
                eval_class_scores[idx][cls] += float(prob)
        used += 1
    if used == 0:
        return {}
    ranked: list[tuple[float, str, int]] = []
    for idx, row in enumerate(eval_rows):
        scores = {cls: value / used for cls, value in eval_class_scores[idx].items()}
        accept_prob = scores.get(0, 0.0)
        nonaccept = [(scores.get(arm, 0.0), arm) for arm in arms]
        best_prob, best_arm = max(nonaccept)
        ranked.append((1.0 - accept_prob + best_prob, str(row["qid"]), best_arm))
    ranked.sort(reverse=True)
    k = min(len(eval_rows), max(1, int(round(len(eval_rows) * rate))))
    return {qid: arm for score, qid, arm in ranked[:k] if score > 0}


def _top_by_scores(rows: list[dict[str, Any]], scores: list[float], rate: float) -> set[str]:
    k = min(len(rows), max(1, int(round(len(rows) * rate))))
    ordered = sorted(range(len(rows)), key=lambda idx: scores[idx], reverse=True)
    return {str(rows[idx]["qid"]) for idx in ordered[:k]}


def _train_hierarchical_actions(
    train_rows: list[dict[str, Any]],
    eval_rows: list[dict[str, Any]],
    arms: list[int],
    *,
    rate: float,
    harm_weight: float,
    lambda_cost_per_1k: float,
    random_seed: int,
    selector: str,
) -> dict[str, int | None]:
    for row in train_rows + eval_rows:
        row["any_budget_helpful"] = any(bool(row.get(f"{_arm_label(arm)}_helpful")) for arm in arms)
        row["any_budget_harmful"] = any(bool(row.get(f"{_arm_label(arm)}_harmful")) for arm in arms)
    think_scores = _fit_binary_score(
        train_rows,
        eval_rows,
        helpful_key="any_budget_helpful",
        harmful_key="any_budget_harmful",
        harm_weight=harm_weight,
        random_seed=random_seed,
    )
    selected = _top_by_scores(eval_rows, think_scores, rate)
    arm_scores: dict[int, list[float]] = {}
    train_avg_cost: dict[int, float] = {}
    for arm in arms:
        label = _arm_label(arm)
        arm_scores[arm] = _fit_binary_score(
            train_rows,
            eval_rows,
            helpful_key=f"{label}_helpful",
            harmful_key=f"{label}_harmful",
            harm_weight=harm_weight,
            random_seed=random_seed + arm,
        )
        train_avg_cost[arm] = _mean([float(row.get(f"{label}_total_tokens") or 0.0) for row in train_rows])

    actions: dict[str, int | None] = {}
    for idx, row in enumerate(eval_rows):
        qid = str(row["qid"])
        if qid not in selected:
            actions[qid] = None
            continue
        scored = [
            (
                arm_scores[arm][idx] - lambda_cost_per_1k * (train_avg_cost[arm] / 1000.0),
                arm,
            )
            for arm in arms
        ]
        if selector == "argmax":
            best_score, best_arm = max(scored, key=lambda item: item[0])
            actions[qid] = best_arm if best_score > 0 else None
        elif selector == "min_positive":
            positive = [(score, arm) for score, arm in scored if score > 0]
            actions[qid] = min((arm for _score, arm in positive), default=max(scored, key=lambda item: item[0])[1])
        else:
            raise KeyError(f"Unknown selector: {selector}")
    return actions


def _train_per_arm_actions(
    train_rows: list[dict[str, Any]],
    eval_rows: list[dict[str, Any]],
    arms: list[int],
    *,
    rate: float,
    harm_weight: float,
    lambda_cost_per_1k: float,
    random_seed: int,
) -> dict[str, int | None]:
    arm_scores: dict[int, list[float]] = {}
    train_avg_cost: dict[int, float] = {}
    for arm in arms:
        label = _arm_label(arm)
        arm_scores[arm] = _fit_binary_score(
            train_rows,
            eval_rows,
            helpful_key=f"{label}_helpful",
            harmful_key=f"{label}_harmful",
            harm_weight=harm_weight,
            random_seed=random_seed + arm,
        )
        train_avg_cost[arm] = _mean([float(row.get(f"{label}_total_tokens") or 0.0) for row in train_rows])
    ranked: list[tuple[float, str, int]] = []
    for idx, row in enumerate(eval_rows):
        scored = [
            (
                arm_scores[arm][idx] - lambda_cost_per_1k * (train_avg_cost[arm] / 1000.0),
                arm,
            )
            for arm in arms
        ]
        best_score, best_arm = max(scored, key=lambda item: item[0])
        ranked.append((best_score, str(row["qid"]), best_arm))
    ranked.sort(reverse=True)
    k = min(len(eval_rows), max(1, int(round(len(eval_rows) * rate))))
    return {qid: arm for score, qid, arm in ranked[:k] if score > 0}


def _train_12k_only_actions(
    train_rows: list[dict[str, Any]],
    eval_rows: list[dict[str, Any]],
    *,
    rate: float,
    harm_weight: float,
    random_seed: int,
) -> dict[str, int | None]:
    scores = _fit_binary_score(
        train_rows,
        eval_rows,
        helpful_key="think12_helpful",
        harmful_key="think12_harmful",
        harm_weight=harm_weight,
        random_seed=random_seed,
    )
    selected = _top_by_scores(eval_rows, scores, rate)
    return {str(row["qid"]): 12 for row in eval_rows if str(row["qid"]) in selected}


def _trace_actions(eval_rows: list[dict[str, Any]], *, rate: float, arm: int) -> dict[str, int | None]:
    def score(row: dict[str, Any]) -> float:
        features = row.get("features") or {}
        return float(features.get("sf_trace_words") or features.get("seed_total_tokens") or row.get("seed_total_tokens") or 0.0)

    k = min(len(eval_rows), max(1, int(round(len(eval_rows) * rate))))
    ordered = sorted(eval_rows, key=score, reverse=True)
    return {str(row["qid"]): arm for row in ordered[:k]}


def _random_actions(eval_rows: list[dict[str, Any]], *, rate: float, arm: int, random_seed: int) -> dict[str, int | None]:
    rng = random.Random(random_seed)
    rows = list(eval_rows)
    rng.shuffle(rows)
    k = min(len(rows), max(1, int(round(len(rows) * rate))))
    return {str(row["qid"]): arm for row in rows[:k]}


def _summarize_method(rows: list[dict[str, Any]], method: str, per_seed: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "method": method,
        "seed": "mean",
        "accuracy": _mean([row["accuracy"] for row in per_seed]),
        "std_accuracy": _std([row["accuracy"] for row in per_seed]),
        "avg_tokens": _mean([row["avg_tokens"] for row in per_seed]),
        "trigger_count": sum(int(row["trigger_count"]) for row in per_seed),
        "adoption_count": sum(int(row["adoption_count"]) for row in per_seed),
        "helpful": sum(int(row["helpful"]) for row in per_seed),
        "harmful": sum(int(row["harmful"]) for row in per_seed),
        "wrong_to_wrong": sum(int(row["wrong_to_wrong"]) for row in per_seed),
        **{
            key: sum(int(row.get(key, 0)) for row in per_seed)
            for key in rows[0].keys()
            if key.startswith("think") or key == "accept"
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train/evaluate hierarchical K-armed budget triggers for BORA-Switch."
    )
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument(
        "--lowercap-root",
        type=Path,
        default=Path("artifacts/remote_stage_main/math500_lowercap_think_20260516_mainpush"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/remote_stage_main/math500_karmed_bandit_20260516"))
    parser.add_argument("--seeds", default="17,7,23")
    parser.add_argument("--arms", default="2,4,6,8,12")
    parser.add_argument("--rates", default="0.1,0.2,0.3,0.5")
    parser.add_argument("--harm-weight", type=float, default=2.0)
    parser.add_argument("--lambda-cost-per-1k", type=float, default=0.0)
    parser.add_argument("--positive-weight", type=float, default=6.0)
    parser.add_argument("--cap-ratio", type=float, default=0.95)
    parser.add_argument("--cap-margin-tokens", type=int, default=128)
    parser.add_argument("--sign-guard", action="store_true", default=True)
    args = parser.parse_args()

    seeds = [int(item) for item in args.seeds.split(",") if item.strip()]
    arms = sorted(int(item) for item in args.arms.split(",") if item.strip())
    rates = [float(item) for item in args.rates.split(",") if item.strip()]

    rows_by_seed = {
        seed: _build_rows(args.root, args.lowercap_root, arms, seed, args)
        for seed in seeds
    }

    arm_rows_path = args.output_dir / "karmed_arm_rows.jsonl"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with arm_rows_path.open("w", encoding="utf-8") as handle:
        for seed in seeds:
            for row in rows_by_seed[seed]:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    result_rows: list[dict[str, Any]] = []

    # Oracle diagnostics: minimal helpful budget and several token-budget greedy oracles.
    oracle_specs: list[tuple[str, float | None]] = [
        ("oracle_minimal_helpful", None),
        ("oracle_budget_2500", 2500.0),
        ("oracle_budget_3000", 3000.0),
        ("oracle_budget_3589", 3589.6),
        ("oracle_budget_3868", 3867.8),
    ]
    for method, budget in oracle_specs:
        per: list[dict[str, Any]] = []
        for seed in seeds:
            eval_rows = rows_by_seed[seed]
            if budget is None:
                actions = _oracle_minimal(eval_rows, arms)
            else:
                actions = _oracle_budget_greedy(eval_rows, arms, budget)
            sim = _simulate(eval_rows, actions, arms)
            row = {"method": method, "seed": seed, **sim}
            per.append(row)
            result_rows.append(row)
        result_rows.append(_summarize_method(result_rows[-1:], method, per))

    for rate in rates:
        method_builders = [
            (
                f"karmed_pairwise_top{int(rate * 100)}",
                lambda train, eval_, seed: _train_pairwise_actions(
                    train,
                    eval_,
                    arms,
                    rate=rate,
                    harm_weight=float(args.harm_weight),
                    lambda_cost_per_1k=float(args.lambda_cost_per_1k),
                    random_seed=seed + 503,
                    positive_weight=float(args.positive_weight),
                ),
            ),
            (
                f"karmed_oracle_distill_top{int(rate * 100)}",
                lambda train, eval_, seed: _train_oracle_distill_actions(
                    train,
                    eval_,
                    arms,
                    rate=rate,
                    random_seed=seed + 607,
                    positive_weight=float(args.positive_weight),
                ),
            ),
            (
                f"karmed_12ktrigger_downgrade_m02_top{int(rate * 100)}",
                lambda train, eval_, seed: _train_conservative_downgrade_actions(
                    train,
                    eval_,
                    arms,
                    rate=rate,
                    harm_weight=float(args.harm_weight),
                    lambda_cost_per_1k=float(args.lambda_cost_per_1k),
                    random_seed=seed + 701,
                    positive_weight=float(args.positive_weight),
                    margin=0.02,
                ),
            ),
            (
                f"karmed_12ktrigger_downgrade_m05_top{int(rate * 100)}",
                lambda train, eval_, seed: _train_conservative_downgrade_actions(
                    train,
                    eval_,
                    arms,
                    rate=rate,
                    harm_weight=float(args.harm_weight),
                    lambda_cost_per_1k=float(args.lambda_cost_per_1k),
                    random_seed=seed + 709,
                    positive_weight=float(args.positive_weight),
                    margin=0.05,
                ),
            ),
            (
                f"karmed_hier_argmax_top{int(rate * 100)}",
                lambda train, eval_, seed: _train_hierarchical_actions(
                    train,
                    eval_,
                    arms,
                    rate=rate,
                    harm_weight=float(args.harm_weight),
                    lambda_cost_per_1k=float(args.lambda_cost_per_1k),
                    random_seed=seed,
                    selector="argmax",
                ),
            ),
            (
                f"karmed_hier_minpos_top{int(rate * 100)}",
                lambda train, eval_, seed: _train_hierarchical_actions(
                    train,
                    eval_,
                    arms,
                    rate=rate,
                    harm_weight=float(args.harm_weight),
                    lambda_cost_per_1k=float(args.lambda_cost_per_1k),
                    random_seed=seed + 13,
                    selector="min_positive",
                ),
            ),
            (
                f"karmed_perarm_top{int(rate * 100)}",
                lambda train, eval_, seed: _train_per_arm_actions(
                    train,
                    eval_,
                    arms,
                    rate=rate,
                    harm_weight=float(args.harm_weight),
                    lambda_cost_per_1k=float(args.lambda_cost_per_1k),
                    random_seed=seed + 29,
                ),
            ),
            (
                f"gbdt_12k_only_top{int(rate * 100)}",
                lambda train, eval_, seed: _train_12k_only_actions(
                    train,
                    eval_,
                    rate=rate,
                    harm_weight=float(args.harm_weight),
                    random_seed=seed + 41,
                ),
            ),
        ]

        for method, builder in method_builders:
            per = []
            for seed in seeds:
                train_rows = [row for other in seeds if other != seed for row in rows_by_seed[other]]
                eval_rows = rows_by_seed[seed]
                actions = builder(train_rows, eval_rows, seed)
                sim = _simulate(eval_rows, actions, arms)
                row = {"method": method, "seed": seed, **sim}
                per.append(row)
                result_rows.append(row)
            result_rows.append(_summarize_method(result_rows[-1:], method, per))

        for method, action_builder in [
            (f"trace_12k_top{int(rate * 100)}", lambda eval_, seed: _trace_actions(eval_, rate=rate, arm=12)),
            (f"random_12k_top{int(rate * 100)}", lambda eval_, seed: _random_actions(eval_, rate=rate, arm=12, random_seed=seed + 719)),
        ]:
            per = []
            for seed in seeds:
                eval_rows = rows_by_seed[seed]
                actions = action_builder(eval_rows, seed)
                sim = _simulate(eval_rows, actions, arms)
                row = {"method": method, "seed": seed, **sim}
                per.append(row)
                result_rows.append(row)
            result_rows.append(_summarize_method(result_rows[-1:], method, per))

    out_csv = args.output_dir / "math500_karmed_loso_results.csv"
    out_json = args.output_dir / "math500_karmed_loso_results.json"
    _write_csv(out_csv, result_rows)
    dump_json(
        out_json,
        {
            "config": {
                "seeds": seeds,
                "arms": arms,
                "rates": rates,
                "harm_weight": args.harm_weight,
                "lambda_cost_per_1k": args.lambda_cost_per_1k,
                "positive_weight": args.positive_weight,
                "cap_ratio": args.cap_ratio,
                "cap_margin_tokens": args.cap_margin_tokens,
                "sign_guard": args.sign_guard,
            },
            "results": result_rows,
        },
    )

    print(f"Wrote {out_csv}")
    for row in result_rows:
        if row["seed"] == "mean":
            print(
                row["method"],
                f"acc={100 * row['accuracy']:.2f}",
                f"tok={row['avg_tokens']:.1f}",
                f"help={row['helpful']}",
                f"harm={row['harmful']}",
                "arms=" + ",".join(
                    f"{key}:{row.get(key, 0)}"
                    for key in ["accept"] + [_arm_label(arm) for arm in arms]
                ),
            )


if __name__ == "__main__":
    main()
