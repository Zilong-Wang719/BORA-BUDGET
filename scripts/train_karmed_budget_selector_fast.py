from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from bora.common import dump_json, load_jsonl
from summarize_bora_switch_calibrated import OPPORTUNITY_PATHS  # type: ignore


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
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
    if len(values) < 2:
        return 0.0
    mu = _mean(values)
    return math.sqrt(sum((value - mu) ** 2 for value in values) / (len(values) - 1))


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


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _arm_label(arm: int) -> str:
    return f"think{arm}"


def _feature_names(rows: list[dict[str, Any]]) -> list[str]:
    names: set[str] = set()
    for row in rows:
        names.update((row.get("features") or {}).keys())
    return sorted(names)


def _matrix(rows: list[dict[str, Any]], names: list[str]) -> list[list[float]]:
    return [[_as_float((row.get("features") or {}).get(name), 0.0) for name in names] for row in rows]


def _attach_questions(rows: list[dict[str, Any]], root: Path) -> None:
    by_seed_qid: dict[tuple[int, str], str] = {}
    for seed, rel in OPPORTUNITY_PATHS.items():
        path = root / rel
        if not path.exists():
            continue
        for row in load_jsonl(path):
            by_seed_qid[(int(seed), str(row["qid"]))] = str(row.get("question") or "")
    for row in rows:
        row["question"] = by_seed_qid.get((int(row["seed"]), str(row["qid"])), "")


def _texts(rows: list[dict[str, Any]]) -> list[str]:
    return [str(row.get("question") or "") for row in rows]


def _fit_text_binary_probs(
    train_rows: list[dict[str, Any]],
    eval_rows: list[dict[str, Any]],
    *,
    label_key: str,
    feature_names: list[str],
    positive_weight: float,
    random_seed: int,
) -> list[float]:
    from scipy.sparse import csr_matrix, hstack
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import MaxAbsScaler

    y_train = [int(bool(row.get(label_key))) for row in train_rows]
    if len(set(y_train)) < 2:
        p = sum(y_train) / max(1, len(y_train))
        return [p for _ in eval_rows]

    vectorizer = TfidfVectorizer(max_features=1600, ngram_range=(1, 2), min_df=2)
    x_text_train = vectorizer.fit_transform(_texts(train_rows))
    x_text_eval = vectorizer.transform(_texts(eval_rows))
    scaler = MaxAbsScaler()
    x_num_train = scaler.fit_transform(csr_matrix(_matrix(train_rows, feature_names)))
    x_num_eval = scaler.transform(csr_matrix(_matrix(eval_rows, feature_names)))
    x_train = hstack([x_num_train, x_text_train], format="csr")
    x_eval = hstack([x_num_eval, x_text_eval], format="csr")
    weights = [positive_weight if y else 1.0 for y in y_train]
    model = LogisticRegression(
        C=0.7,
        max_iter=2000,
        class_weight="balanced",
        solver="liblinear",
        random_state=random_seed,
    )
    model.fit(x_train, y_train, sample_weight=weights)
    return [float(row[1]) for row in model.predict_proba(x_eval)]


def _fit_text_multiclass_probs(
    train_rows: list[dict[str, Any]],
    eval_rows: list[dict[str, Any]],
    *,
    label_key: str,
    feature_names: list[str],
    positive_weight: float,
    random_seed: int,
) -> tuple[list[int], list[list[float]]]:
    from scipy.sparse import csr_matrix, hstack
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import MaxAbsScaler

    y_train = [int(row.get(label_key) or 0) for row in train_rows]
    classes = sorted(set(y_train))
    if len(classes) < 2:
        return classes, [[1.0] for _ in eval_rows]
    vectorizer = TfidfVectorizer(max_features=1600, ngram_range=(1, 2), min_df=2)
    x_text_train = vectorizer.fit_transform(_texts(train_rows))
    x_text_eval = vectorizer.transform(_texts(eval_rows))
    scaler = MaxAbsScaler()
    x_num_train = scaler.fit_transform(csr_matrix(_matrix(train_rows, feature_names)))
    x_num_eval = scaler.transform(csr_matrix(_matrix(eval_rows, feature_names)))
    x_train = hstack([x_num_train, x_text_train], format="csr")
    x_eval = hstack([x_num_eval, x_text_eval], format="csr")
    weights = [positive_weight if y else 1.0 for y in y_train]
    model = LogisticRegression(
        C=0.7,
        max_iter=3000,
        class_weight="balanced",
        solver="saga",
        multi_class="auto",
        random_state=random_seed,
    )
    model.fit(x_train, y_train, sample_weight=weights)
    return [int(cls) for cls in model.classes_], [
        [float(value) for value in row] for row in model.predict_proba(x_eval)
    ]


def _fit_binary_probs(
    train_rows: list[dict[str, Any]],
    eval_rows: list[dict[str, Any]],
    *,
    label_key: str,
    feature_names: list[str],
    positive_weight: float,
    random_seed: int,
    use_text: bool,
) -> list[float]:
    if use_text:
        return _fit_text_binary_probs(
            train_rows,
            eval_rows,
            label_key=label_key,
            feature_names=feature_names,
            positive_weight=positive_weight,
            random_seed=random_seed,
        )
    from sklearn.ensemble import HistGradientBoostingClassifier

    y_train = [int(bool(row.get(label_key))) for row in train_rows]
    if len(set(y_train)) < 2:
        p = sum(y_train) / max(1, len(y_train))
        return [p for _ in eval_rows]
    model = HistGradientBoostingClassifier(
        max_iter=90,
        learning_rate=0.07,
        l2_regularization=0.03,
        random_state=random_seed,
    )
    weights = [positive_weight if y else 1.0 for y in y_train]
    model.fit(_matrix(train_rows, feature_names), y_train, sample_weight=weights)
    return [float(row[1]) for row in model.predict_proba(_matrix(eval_rows, feature_names))]


def _fit_multiclass_probs(
    train_rows: list[dict[str, Any]],
    eval_rows: list[dict[str, Any]],
    *,
    label_key: str,
    feature_names: list[str],
    positive_weight: float,
    random_seed: int,
    use_text: bool,
) -> tuple[list[int], list[list[float]]]:
    if use_text:
        return _fit_text_multiclass_probs(
            train_rows,
            eval_rows,
            label_key=label_key,
            feature_names=feature_names,
            positive_weight=positive_weight,
            random_seed=random_seed,
        )
    from sklearn.ensemble import HistGradientBoostingClassifier

    y_train = [int(row.get(label_key) or 0) for row in train_rows]
    classes = sorted(set(y_train))
    if len(classes) < 2:
        return classes, [[1.0] for _ in eval_rows]
    model = HistGradientBoostingClassifier(
        max_iter=110,
        learning_rate=0.07,
        l2_regularization=0.03,
        random_state=random_seed,
    )
    weights = [positive_weight if y else 1.0 for y in y_train]
    model.fit(_matrix(train_rows, feature_names), y_train, sample_weight=weights)
    return [int(cls) for cls in model.classes_], [
        [float(value) for value in row] for row in model.predict_proba(_matrix(eval_rows, feature_names))
    ]


def _minimal_helpful_arm(row: dict[str, Any], arms: list[int]) -> int:
    for arm in arms:
        if bool(row.get(f"{_arm_label(arm)}_helpful")):
            return arm
    return 0


def _prepare_rows(rows: list[dict[str, Any]], arms: list[int]) -> None:
    for row in rows:
        row["any_helpful"] = any(bool(row.get(f"{_arm_label(arm)}_helpful")) for arm in arms)
        row["any_harmful"] = any(bool(row.get(f"{_arm_label(arm)}_harmful")) for arm in arms)
        row["oracle_arm"] = _minimal_helpful_arm(row, arms)
        row["think12_helpful_label"] = bool(row.get("think12_helpful"))
        row["think12_harmful_label"] = bool(row.get("think12_harmful"))


def _pair_rows(rows: list[dict[str, Any]], arms: list[int]) -> list[dict[str, Any]]:
    out = []
    max_arm = max(arms)
    for row in rows:
        base = dict(row.get("features") or {})
        for arm in arms:
            label = _arm_label(arm)
            features = dict(base)
            features["arm_budget_k"] = float(arm)
            features["arm_budget_frac"] = float(arm) / max_arm
            features["arm_budget_log"] = math.log1p(float(arm))
            features["arm_low"] = 1.0 if arm <= 4 else 0.0
            features["arm_mid"] = 1.0 if 6 <= arm <= 8 else 0.0
            features["arm_high"] = 1.0 if arm >= 10 else 0.0
            for candidate in arms:
                features[f"arm_eq_{candidate}"] = 1.0 if arm == candidate else 0.0
            out.append(
                {
                    "qid": row["qid"],
                    "seed": row["seed"],
                    "arm": arm,
                    "question": row.get("question") or "",
                    "features": features,
                    "helpful": bool(row.get(f"{label}_helpful")),
                    "harmful": bool(row.get(f"{label}_harmful")),
                    "arm_tokens": float(row.get(f"{label}_total_tokens") or 0.0),
                }
            )
    return out


def _simulate(rows: list[dict[str, Any]], actions: dict[str, int | None], arms: list[int]) -> dict[str, Any]:
    correct = helpful = harmful = wrong_to_wrong = triggered = adopted = 0
    tokens: list[float] = []
    counts = {"accept": 0, **{_arm_label(arm): 0 for arm in arms}}
    for row in rows:
        qid = str(row["qid"])
        arm = actions.get(qid)
        seed_correct = bool(row["seed_correct"])
        seed_tokens = float(row.get("seed_total_tokens") or 0.0)
        if arm is None:
            counts["accept"] += 1
            final_correct = seed_correct
            total = seed_tokens
        else:
            label = _arm_label(arm)
            counts[label] += 1
            triggered += 1
            final_correct = bool(row.get(f"{label}_final_correct"))
            total = seed_tokens + float(row.get(f"{label}_total_tokens") or 0.0)
            if bool(row.get(f"{label}_gate_pass")):
                adopted += 1
            helpful += int((not seed_correct) and final_correct)
            harmful += int(seed_correct and (not final_correct))
            wrong_to_wrong += int((not seed_correct) and (not final_correct) and bool(row.get(f"{label}_gate_pass")))
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
        **counts,
    }


def _top_qids(rows: list[dict[str, Any]], scores: list[float], rate: float) -> set[str]:
    k = min(len(rows), max(1, int(round(len(rows) * rate))))
    order = sorted(range(len(rows)), key=lambda idx: scores[idx], reverse=True)
    return {str(rows[idx]["qid"]) for idx in order[:k]}


def _oracle_actions(rows: list[dict[str, Any]], arms: list[int]) -> dict[str, int | None]:
    return {str(row["qid"]): (_minimal_helpful_arm(row, arms) or None) for row in rows}


def _fit_split_scores(
    train_rows: list[dict[str, Any]],
    eval_rows: list[dict[str, Any]],
    arms: list[int],
    *,
    positive_weight: float,
    random_seed: int,
    use_text: bool,
) -> dict[str, Any]:
    base_names = _feature_names(train_rows + eval_rows)
    q_any = _fit_binary_probs(
        train_rows,
        eval_rows,
        label_key="any_helpful",
        feature_names=base_names,
        positive_weight=positive_weight,
        random_seed=random_seed,
        use_text=use_text,
    )
    q_12 = _fit_binary_probs(
        train_rows,
        eval_rows,
        label_key="think12_helpful_label",
        feature_names=base_names,
        positive_weight=positive_weight,
        random_seed=random_seed + 11,
        use_text=use_text,
    )
    pair_train = _pair_rows(train_rows, arms)
    pair_eval = _pair_rows(eval_rows, arms)
    pair_names = _feature_names(pair_train + pair_eval)
    pair_help = _fit_binary_probs(
        pair_train,
        pair_eval,
        label_key="helpful",
        feature_names=pair_names,
        positive_weight=positive_weight,
        random_seed=random_seed + 23,
        use_text=use_text,
    )
    pair_scores: dict[tuple[str, int], float] = {}
    for pair, score in zip(pair_eval, pair_help):
        pair_scores[(str(pair["qid"]), int(pair["arm"]))] = score
    classes, probs = _fit_multiclass_probs(
        train_rows,
        eval_rows,
        label_key="oracle_arm",
        feature_names=base_names,
        positive_weight=positive_weight,
        random_seed=random_seed + 37,
        use_text=use_text,
    )
    distill: dict[str, dict[int, float]] = {}
    for row, prob_row in zip(eval_rows, probs):
        score_map = {int(cls): 0.0 for cls in [0] + arms}
        for cls, prob in zip(classes, prob_row):
            score_map[int(cls)] = float(prob)
        distill[str(row["qid"])] = score_map
    return {
        "q_any": q_any,
        "q_12": q_12,
        "pair": pair_scores,
        "distill": distill,
        "qid_prior": _qid_prior_stats(train_rows, arms),
    }


def _qid_prior_stats(rows: list[dict[str, Any]], arms: list[int]) -> dict[str, dict[str, Any]]:
    stats: dict[str, dict[str, Any]] = {}
    for row in rows:
        qid = str(row["qid"])
        stat = stats.setdefault(
            qid,
            {
                "count": 0,
                "any_helpful": 0,
                "any_harmful": 0,
                "arms": {
                    arm: {"helpful": 0, "harmful": 0, "tokens": []}
                    for arm in arms
                },
            },
        )
        stat["count"] += 1
        stat["any_helpful"] += int(bool(row.get("any_helpful")))
        stat["any_harmful"] += int(bool(row.get("any_harmful")))
        for arm in arms:
            label = _arm_label(arm)
            arm_stat = stat["arms"][arm]
            arm_stat["helpful"] += int(bool(row.get(f"{label}_helpful")))
            arm_stat["harmful"] += int(bool(row.get(f"{label}_harmful")))
            arm_stat["tokens"].append(float(row.get(f"{label}_total_tokens") or 0.0))
    return stats


def _qid_prior_best_arm(
    stat: dict[str, Any] | None,
    arms: list[int],
    *,
    harm_weight: float,
    cost_weight: float,
) -> tuple[float, int | None]:
    if not stat:
        return 0.0, None
    best_score = -1e9
    best_arm: int | None = None
    for arm in arms:
        arm_stat = stat["arms"][arm]
        mean_tokens = _mean([float(value) for value in arm_stat.get("tokens") or []])
        # Count-based empirical Bayes prior over repeated stochastic seeds.  The
        # token term is intentionally weak: it breaks ties toward lower budgets
        # without letting cheap but unhelpful arms dominate.
        score = (
            float(arm_stat.get("helpful") or 0)
            - harm_weight * float(arm_stat.get("harmful") or 0)
            - cost_weight * (mean_tokens / 1000.0)
        )
        if score > best_score or (abs(score - best_score) < 1e-12 and best_arm is not None and arm < best_arm):
            best_score = score
            best_arm = arm
    any_score = float(stat.get("any_helpful") or 0) - harm_weight * float(stat.get("any_harmful") or 0)
    score = max(best_score, any_score)
    if best_arm is None or score <= 0:
        return score, None
    return score, best_arm


def _actions_qany_pair(eval_rows: list[dict[str, Any]], arms: list[int], scores: dict[str, Any], rate: float) -> dict[str, int | None]:
    selected = _top_qids(eval_rows, scores["q_any"], rate)
    actions = {}
    for row in eval_rows:
        qid = str(row["qid"])
        if qid not in selected:
            actions[qid] = None
            continue
        score, arm = max((scores["pair"][(qid, arm)], arm) for arm in arms)
        actions[qid] = arm if score > 0 else None
    return actions


def _actions_pair_global(eval_rows: list[dict[str, Any]], arms: list[int], scores: dict[str, Any], rate: float) -> dict[str, int | None]:
    ranked = []
    for row in eval_rows:
        qid = str(row["qid"])
        score, arm = max((scores["pair"][(qid, arm)], arm) for arm in arms)
        ranked.append((score, qid, arm))
    ranked.sort(reverse=True)
    k = min(len(eval_rows), max(1, int(round(len(eval_rows) * rate))))
    return {qid: arm for score, qid, arm in ranked[:k] if score > 0}


def _actions_qid_prior(
    eval_rows: list[dict[str, Any]],
    arms: list[int],
    scores: dict[str, Any],
    rate: float,
    *,
    harm_weight: float,
    cost_weight: float,
) -> dict[str, int | None]:
    ranked = []
    priors = scores["qid_prior"]
    for row in eval_rows:
        qid = str(row["qid"])
        prior_score, arm = _qid_prior_best_arm(
            priors.get(qid),
            arms,
            harm_weight=harm_weight,
            cost_weight=cost_weight,
        )
        ranked.append((prior_score, qid, arm))
    ranked.sort(reverse=True)
    k = min(len(eval_rows), max(1, int(round(len(eval_rows) * rate))))
    return {qid: arm for score, qid, arm in ranked[:k] if score > 0 and arm is not None}


def _actions_qid_blend(
    eval_rows: list[dict[str, Any]],
    arms: list[int],
    scores: dict[str, Any],
    rate: float,
    *,
    harm_weight: float,
    cost_weight: float,
    prior_weight: float,
) -> dict[str, int | None]:
    ranked = []
    priors = scores["qid_prior"]
    for idx, row in enumerate(eval_rows):
        qid = str(row["qid"])
        prior_score, prior_arm = _qid_prior_best_arm(
            priors.get(qid),
            arms,
            harm_weight=harm_weight,
            cost_weight=cost_weight,
        )
        pair_score, pair_arm = max((scores["pair"][(qid, arm)], arm) for arm in arms)
        blended = scores["q_any"][idx] + prior_weight * max(0.0, prior_score)
        arm = prior_arm if prior_arm is not None and prior_score > 0 else pair_arm
        ranked.append((blended, qid, arm, pair_score))
    ranked.sort(reverse=True)
    k = min(len(eval_rows), max(1, int(round(len(eval_rows) * rate))))
    return {qid: arm for score, qid, arm, pair_score in ranked[:k] if score > 0 and (arm is not None)}


def _actions_distill(eval_rows: list[dict[str, Any]], arms: list[int], scores: dict[str, Any], rate: float) -> dict[str, int | None]:
    ranked = []
    for row in eval_rows:
        qid = str(row["qid"])
        probs = scores["distill"][qid]
        arm_prob, arm = max((probs.get(arm, 0.0), arm) for arm in arms)
        ranked.append((1.0 - probs.get(0, 0.0) + arm_prob, qid, arm))
    ranked.sort(reverse=True)
    k = min(len(eval_rows), max(1, int(round(len(eval_rows) * rate))))
    return {qid: arm for score, qid, arm in ranked[:k] if score > 0}


def _actions_12k(eval_rows: list[dict[str, Any]], scores: dict[str, Any], rate: float) -> dict[str, int | None]:
    selected = _top_qids(eval_rows, scores["q_12"], rate)
    return {str(row["qid"]): 12 for row in eval_rows if str(row["qid"]) in selected}


def _actions_12k_downgrade(
    eval_rows: list[dict[str, Any]],
    arms: list[int],
    scores: dict[str, Any],
    rate: float,
    *,
    margin: float,
) -> dict[str, int | None]:
    selected = _top_qids(eval_rows, scores["q_12"], rate)
    actions = {}
    for row in eval_rows:
        qid = str(row["qid"])
        if qid not in selected:
            actions[qid] = None
            continue
        score12 = scores["pair"][(qid, 12)]
        eligible = [arm for arm in arms if scores["pair"][(qid, arm)] >= score12 - margin and scores["pair"][(qid, arm)] > 0]
        actions[qid] = min(eligible) if eligible else 12
    return actions


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
    parser.add_argument("--rates", default="0.12,0.3,0.5")
    parser.add_argument("--positive-weight", type=float, default=10.0)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--use-text", action="store_true")
    parser.add_argument("--qid-prior-harm-weight", type=float, default=2.0)
    parser.add_argument("--qid-prior-cost-weight", type=float, default=0.02)
    parser.add_argument("--qid-prior-blend-weight", type=float, default=1.5)
    args = parser.parse_args()

    seeds = [int(item) for item in args.seeds.split(",") if item.strip()]
    arms = sorted(int(item) for item in args.arms.split(",") if item.strip())
    rates = [float(item) for item in args.rates.split(",") if item.strip()]
    rows = _load_jsonl(args.arm_rows)
    if args.use_text:
        _attach_questions(rows, args.root)
    _prepare_rows(rows, arms)
    by_seed = {seed: [row for row in rows if int(row["seed"]) == seed] for seed in seeds}

    split_scores = {}
    for seed in seeds:
        train_rows = [row for other in seeds if other != seed for row in by_seed[other]]
        eval_rows = by_seed[seed]
        split_scores[seed] = _fit_split_scores(
            train_rows,
            eval_rows,
            arms,
            positive_weight=float(args.positive_weight),
            random_seed=seed,
            use_text=bool(args.use_text),
        )

    result_rows = []
    methods = {
        "oracle_minimal": lambda eval_rows, scores, rate: _oracle_actions(eval_rows, arms),
        "gbdt_12k": lambda eval_rows, scores, rate: _actions_12k(eval_rows, scores, rate),
        "qany_pair": lambda eval_rows, scores, rate: _actions_qany_pair(eval_rows, arms, scores, rate),
        "pair_global": lambda eval_rows, scores, rate: _actions_pair_global(eval_rows, arms, scores, rate),
        "qid_prior": lambda eval_rows, scores, rate: _actions_qid_prior(
            eval_rows,
            arms,
            scores,
            rate,
            harm_weight=float(args.qid_prior_harm_weight),
            cost_weight=float(args.qid_prior_cost_weight),
        ),
        "qid_blend": lambda eval_rows, scores, rate: _actions_qid_blend(
            eval_rows,
            arms,
            scores,
            rate,
            harm_weight=float(args.qid_prior_harm_weight),
            cost_weight=float(args.qid_prior_cost_weight),
            prior_weight=float(args.qid_prior_blend_weight),
        ),
        "oracle_distill": lambda eval_rows, scores, rate: _actions_distill(eval_rows, arms, scores, rate),
        "downgrade_m02": lambda eval_rows, scores, rate: _actions_12k_downgrade(eval_rows, arms, scores, rate, margin=0.02),
        "downgrade_m05": lambda eval_rows, scores, rate: _actions_12k_downgrade(eval_rows, arms, scores, rate, margin=0.05),
        "downgrade_m10": lambda eval_rows, scores, rate: _actions_12k_downgrade(eval_rows, arms, scores, rate, margin=0.10),
    }
    for rate in rates:
        for method, builder in methods.items():
            if method == "oracle_minimal" and rate != rates[0]:
                continue
            name = method if method == "oracle_minimal" else f"{method}_top{int(round(rate * 100))}"
            per_seed = []
            for seed in seeds:
                eval_rows = by_seed[seed]
                actions = builder(eval_rows, split_scores[seed], rate)
                sim = _simulate(eval_rows, actions, arms)
                row = {"method": name, "seed": seed, **sim}
                result_rows.append(row)
                per_seed.append(row)
            result_rows.append(_summary(name, per_seed, arms))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "math500_karmed_fast_results.csv"
    json_path = args.output_dir / "math500_karmed_fast_results.json"
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
            arms_msg = ",".join(f"{key}:{row.get(key, 0)}" for key in ["accept"] + [_arm_label(arm) for arm in arms])
            print(row["method"], f"acc={100*row['accuracy']:.2f}", f"tok={row['avg_tokens']:.1f}", f"help={row['helpful']}", f"harm={row['harmful']}", arms_msg)


if __name__ == "__main__":
    main()
