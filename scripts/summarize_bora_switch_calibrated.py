from __future__ import annotations

import argparse
import csv
import json
import random
import statistics as stats
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from bora.common import dump_json, is_correct, load_jsonl

from analyze_trigger_frontier import (  # type: ignore
    _feature_names,
    _filter_pass,
    _fit_scores,
    _heuristic_scores,
    _select_topk,
    _simulate,
)


SEEDS = [17, 7, 23]


OPPORTUNITY_PATHS = {
    17: "artifacts/remote_stage_c/calibrated_switch_20260515/opportunity_math500_seed17_think12k.jsonl",
    7: "artifacts/remote_stage_c/calibrated_switch_20260515/opportunity_math500_seed7_think12k.jsonl",
    23: "artifacts/remote_stage_c/calibrated_switch_20260515/opportunity_math500_seed23_think12k.jsonl",
}


FRONTIER_LOSO_PATHS = {
    17: "artifacts/remote_stage_c/calibrated_switch_20260515/frontier_loso_train7_23_eval17.json",
    7: "artifacts/remote_stage_c/calibrated_switch_20260515/frontier_loso_train17_23_eval7.json",
    23: "artifacts/remote_stage_c/calibrated_switch_20260515/frontier_loso_train17_7_eval23.json",
}


THINK8K_PATHS = {
    17: "artifacts/remote_stage_c/math500_think8k_baseline_20260512/standard_direct_cot_think8k_math500_seed17_merged.json",
    7: "artifacts/remote_stage_c/math500_think8k_seed7_20260515/standard_direct_cot_think8k_math500_seed7.json",
    23: "artifacts/remote_stage_c/math500_think8k_seed23_20260515/standard_direct_cot_think8k_math500_seed23.json",
}


NO_THINK_PATHS = {
    17: "artifacts/remote_stage_c/math500_switch_20260512/standard_direct_cot_math500_seed17.json",
    7: "artifacts/remote_stage_c/math500_seed_repeat_seed7_20260513/standard_direct_cot_no_think_math500_seed7.json",
    23: "artifacts/remote_stage_c/math500_seed_repeat_seed23_20260515/standard_direct_cot_no_think_math500_seed23.json",
}


THINK12K_PATHS = {
    17: "artifacts/remote_stage_c/math500_think12k_baseline_20260512/standard_direct_cot_think12k_math500_seed17_merged.json",
    7: "artifacts/remote_stage_c/math500_think12k_seed7_20260515/standard_direct_cot_think12k_math500_seed7.json",
    23: "artifacts/remote_stage_c/math500_think12k_seed23_20260515/standard_direct_cot_think12k_math500_seed23.json",
}


SC3_PATHS = {
    17: "artifacts/remote_stage_c/math500_sc3_seed17_20260515/self_consistency3_no_think_math500_seed17.json",
    7: "artifacts/remote_stage_c/math500_sc3_seed7_20260515/self_consistency3_no_think_math500_seed7.json",
    23: "artifacts/remote_stage_c/math500_sc3_seed23_20260515/self_consistency3_no_think_math500_seed23.json",
}


COLLEGE_FRONTIER_PATHS = {
    17: "artifacts/remote_stage_c/external_calibration_20260516/frontier_train_college2400_eval_math500_seed17.json",
    7: "artifacts/remote_stage_c/external_calibration_20260516/frontier_train_college2400_eval_math500_seed7.json",
    23: "artifacts/remote_stage_c/external_calibration_20260516/frontier_train_college2400_eval_math500_seed23.json",
}


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def _records(payload: dict[str, Any]) -> list[dict[str, Any]]:
    if isinstance(payload.get("records"), list):
        return payload["records"]
    if isinstance(payload.get("rows"), list):
        return payload["rows"]
    for value in payload.values():
        if isinstance(value, dict) and isinstance(value.get("records"), list):
            return value["records"]
        if isinstance(value, dict) and isinstance(value.get("rows"), list):
            return value["rows"]
    raise ValueError("Could not infer records block.")


def _nested_summary(payload: dict[str, Any]) -> dict[str, Any]:
    if isinstance(payload.get("summary"), dict):
        return payload["summary"]
    for value in payload.values():
        if isinstance(value, dict) and isinstance(value.get("summary"), dict):
            return value["summary"]
    return {}


def _total_tokens(row: dict[str, Any]) -> float:
    for key in ("total_tokens", "avg_total_tokens", "tokens"):
        if row.get(key) is not None:
            return float(row[key])
    usage = row.get("usage") or {}
    for key in ("total_tokens", "completion_tokens"):
        if usage.get(key) is not None:
            return float(usage[key])
    return 0.0


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _std(values: list[float]) -> float:
    return stats.stdev(values) if len(values) > 1 else 0.0


def _format_pct(value: float) -> str:
    return f"{100 * value:.2f}"


def _baseline_from_records(path: Path) -> dict[str, Any]:
    payload = _load_json(path)
    records = _records(payload)
    summary = _nested_summary(payload)
    n = len(records)
    correct = sum(bool(row.get("correct")) for row in records)
    avg_tokens = summary.get("avg_total_tokens")
    if avg_tokens is None:
        avg_tokens = _mean([_total_tokens(row) for row in records])
    qid_correct = {str(row.get("qid") or row.get("id")): bool(row.get("correct")) for row in records}
    return {
        "count": n,
        "correct": correct,
        "accuracy": correct / max(1, n),
        "avg_total_tokens": float(avg_tokens or 0.0),
        "qid_correct": qid_correct,
    }


def _latency_by_qid(path: Path) -> tuple[dict[str, float], float]:
    payload = _load_json(path)
    records = _records(payload)
    summary = _nested_summary(payload)
    by_qid = {
        str(row.get("qid") or row.get("id")): float(row.get("latency_ms") or 0.0)
        for row in records
    }
    avg = summary.get("avg_latency_ms")
    if avg is None:
        avg = _mean(list(by_qid.values()))
    return by_qid, float(avg or 0.0)


def _seed_from_opportunity(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    correct = sum(bool(row.get("seed_correct")) for row in rows)
    return {
        "count": n,
        "correct": correct,
        "accuracy": correct / max(1, n),
        "avg_total_tokens": _mean([float(row.get("seed_total_tokens") or 0) for row in rows]),
        "qid_correct": {str(row["qid"]): bool(row.get("seed_correct")) for row in rows},
    }


def _think12k_from_opportunity(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    correct = sum(bool(row.get("think_correct")) for row in rows)
    return {
        "count": n,
        "correct": correct,
        "accuracy": correct / max(1, n),
        "avg_total_tokens": _mean([float(row.get("think_total_tokens") or 0) for row in rows]),
        "qid_correct": {str(row["qid"]): bool(row.get("think_correct")) for row in rows},
    }


def _frontier_rows(path: Path) -> list[dict[str, Any]]:
    payload = _load_json(path)
    rows = payload.get("results")
    if not isinstance(rows, list):
        raise ValueError(f"Missing results list in {path}")
    return rows


def _pick_frontier_row(
    rows: list[dict[str, Any]],
    *,
    method: str,
    rate: float | None = None,
    adoption_filter: str = "strict",
    selection_mode: str = "topk_eval",
) -> dict[str, Any]:
    candidates = []
    for row in rows:
        if row.get("method") != method:
            continue
        if row.get("adoption_filter") != adoption_filter:
            continue
        if row.get("selection_mode") != selection_mode:
            continue
        if rate is not None and abs(float(row.get("target_rate") or 0.0) - rate) > 1e-9:
            continue
        candidates.append(row)
    if not candidates:
        raise KeyError(
            f"No frontier row for method={method}, rate={rate}, filter={adoption_filter}, mode={selection_mode}"
        )
    return max(candidates, key=lambda row: (float(row["accuracy"]), -float(row["avg_total_tokens"])))


def _row_summary(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "count": int(row["count"]),
        "correct": int(round(float(row["final_correct"]))),
        "accuracy": float(row["accuracy"]),
        "avg_total_tokens": float(row["avg_total_tokens"]),
        "trigger_count": int(round(float(row.get("trigger_count", 0)))),
        "trigger_rate": float(row.get("trigger_rate", 0.0)),
        "adoption_count": int(round(float(row.get("adoption_count", 0)))),
        "adoption_rate": float(row.get("adoption_rate", 0.0)),
        "helpful": int(round(float(row.get("helpful", 0)))),
        "harmful": int(round(float(row.get("harmful", 0)))),
        "wrong_to_wrong": int(round(float(row.get("wrong_to_wrong", 0)))),
        "source_row": row,
    }


def _simulate_selected(
    train_rows: list[dict[str, Any]],
    eval_rows: list[dict[str, Any]],
    *,
    method: str,
    rate: float,
    adoption_filter: str,
    random_seed: int,
) -> tuple[dict[str, Any], dict[str, bool]]:
    if method == "trace_length":
        scores = [
            float((row.get("features") or {}).get("sf_trace_words") or 0)
            or float((row.get("features") or {}).get("seed_total_tokens") or 0)
            for row in eval_rows
        ]
    elif method == "heuristic":
        scores = _heuristic_scores(eval_rows)
    elif method in {"gbdt", "logistic"}:
        feature_names = _feature_names(train_rows + eval_rows)
        _train_scores, scores, _harm, _info = _fit_scores(
            train_rows,
            eval_rows,
            model_name=method,
            feature_names=feature_names,
            harm_weight=2.0,
            random_seed=random_seed,
        )
    else:
        raise KeyError(method)

    if method == "heuristic":
        selected = {idx for idx, score in enumerate(scores) if score > 0}
    else:
        selected = _select_topk(scores, rate)
    summary = _simulate(eval_rows, selected=selected, adoption_filter=adoption_filter)
    correctness: dict[str, bool] = {}
    for idx, row in enumerate(eval_rows):
        seed_correct = bool(row.get("seed_correct"))
        final_answer = row.get("seed_answer")
        if idx in selected and _filter_pass(row, adoption_filter):
            final_answer = row.get("think_answer")
        correctness[str(row["qid"])] = bool(is_correct(final_answer, row.get("gold_answer")))
        if str(row["qid"]) not in correctness:
            correctness[str(row["qid"])] = seed_correct
    return summary, correctness


def _simulate_selected_with_latency(
    train_rows: list[dict[str, Any]],
    eval_rows: list[dict[str, Any]],
    *,
    method: str,
    rate: float,
    adoption_filter: str,
    random_seed: int,
    seed_latency: dict[str, float],
    think_latency: dict[str, float],
) -> tuple[dict[str, Any], dict[str, bool], set[str]]:
    if method == "trace_length":
        scores = [
            float((row.get("features") or {}).get("sf_trace_words") or 0)
            or float((row.get("features") or {}).get("seed_total_tokens") or 0)
            for row in eval_rows
        ]
    elif method == "heuristic":
        scores = _heuristic_scores(eval_rows)
    elif method in {"gbdt", "logistic"}:
        feature_names = _feature_names(train_rows + eval_rows)
        _train_scores, scores, _harm, _info = _fit_scores(
            train_rows,
            eval_rows,
            model_name=method,
            feature_names=feature_names,
            harm_weight=2.0,
            random_seed=random_seed,
        )
    else:
        raise KeyError(method)

    if method == "heuristic":
        selected = {idx for idx, score in enumerate(scores) if score > 0}
    else:
        selected = _select_topk(scores, rate)

    summary = _simulate(eval_rows, selected=selected, adoption_filter=adoption_filter)
    correctness: dict[str, bool] = {}
    selected_qids: set[str] = set()
    latencies: list[float] = []
    for idx, row in enumerate(eval_rows):
        qid = str(row["qid"])
        seed_correct = bool(row.get("seed_correct"))
        final_answer = row.get("seed_answer")
        latency = seed_latency.get(qid, 0.0)
        if idx in selected:
            selected_qids.add(qid)
            latency += think_latency.get(qid, 0.0)
            if _filter_pass(row, adoption_filter):
                final_answer = row.get("think_answer")
        correctness[qid] = bool(is_correct(final_answer, row.get("gold_answer")))
        if qid not in correctness:
            correctness[qid] = seed_correct
        latencies.append(latency)
    summary["avg_latency_ms_estimated"] = _mean(latencies)
    return summary, correctness, selected_qids


def _feature_subset_names(all_names: list[str], subset: str) -> list[str]:
    trace_names = {
        name
        for name in all_names
        if name
        in {
            "seed_total_tokens",
            "seed_completion_tokens",
            "seed_prompt_tokens",
            "sf_trace_words",
            "sf_trace_chars",
        }
    }
    parse_names = {
        name
        for name in all_names
        if name.startswith("seed_answer_")
        or name
        in {
            "sf_answer_format_quality",
            "sf_answer_parse_success",
            "sf_explicit_answer_present",
            "sf_final_answer_position",
            "sf_multiple_explicit_answers",
            "sf_multiple_numeric_answers",
            "seed_completion_head_numeric_count",
        }
    }
    old_bora_names = {
        name
        for name in all_names
        if name.startswith("old_")
        or name.startswith("old_rescue_action_")
        or name.startswith("sf_verifier_")
        or name
        in {
            "sf_high_risk_clean_seed",
            "sf_arithmetic_pattern_risk",
            "sf_seed_confidence",
            "sf_seed_done",
            "sf_short_or_degenerate_reasoning",
            "sf_trigger_malformed_final_answer",
        }
    }
    problem_shape_names = {
        name
        for name in all_names
        if name.startswith("question_")
    }
    groups = {
        "trace_only": trace_names,
        "parse_only": parse_names,
        "old_bora_only": old_bora_names,
        "problem_shape_only": problem_shape_names,
        "all_minus_trace": set(all_names) - trace_names,
        "all_features": set(all_names),
    }
    out = sorted(groups[subset])
    if not out:
        raise ValueError(f"Feature subset {subset} is empty.")
    return out


def _feature_ablation_summary(
    *,
    opportunities: dict[int, list[dict[str, Any]]],
    output_dir: Path,
) -> list[dict[str, Any]]:
    subsets = [
        "trace_only",
        "parse_only",
        "old_bora_only",
        "problem_shape_only",
        "all_minus_trace",
        "all_features",
    ]
    rows_out: list[dict[str, Any]] = []
    for subset in subsets:
        per_rate: dict[float, list[dict[str, Any]]] = {0.3: [], 0.5: []}
        for seed in SEEDS:
            train_rows = [row for other in SEEDS if other != seed for row in opportunities[other]]
            eval_rows = opportunities[seed]
            feature_names = _feature_subset_names(_feature_names(train_rows + eval_rows), subset)
            _train_scores, eval_scores, _harm, _info = _fit_scores(
                train_rows,
                eval_rows,
                model_name="gbdt",
                feature_names=feature_names,
                harm_weight=2.0,
                random_seed=seed,
            )
            for rate in (0.3, 0.5):
                selected = _select_topk(eval_scores, rate)
                sim = _simulate(eval_rows, selected=selected, adoption_filter="strict")
                per_rate[rate].append(sim)
        for rate, sims in per_rate.items():
            rows_out.append(
                {
                    "feature_subset": subset,
                    "rate": rate,
                    "mean_acc": 100 * _mean([sim["accuracy"] for sim in sims]),
                    "std_acc": 100 * _std([sim["accuracy"] for sim in sims]),
                    "avg_tokens": _mean([sim["avg_total_tokens"] for sim in sims]),
                    "helpful": sum(int(sim["helpful"]) for sim in sims),
                    "harmful": sum(int(sim["harmful"]) for sim in sims),
                    "wrong_to_wrong": sum(int(sim["wrong_to_wrong"]) for sim in sims),
                    "feature_count": len(
                        _feature_subset_names(
                            _feature_names([row for rows in opportunities.values() for row in rows]),
                            subset,
                        )
                    ),
                }
            )
    _write_csv(output_dir / "math500_feature_ablation.csv", rows_out)
    return rows_out


def _qid_bootstrap_delta(
    method_correct: dict[int, dict[str, bool]],
    baseline_correct: dict[int, dict[str, bool]],
    *,
    samples: int,
    seed: int,
) -> dict[str, float]:
    observations: list[tuple[int, str]] = []
    for eval_seed, qid_map in method_correct.items():
        for qid in qid_map:
            if qid in baseline_correct.get(eval_seed, {}):
                observations.append((eval_seed, qid))
    rng = random.Random(seed)
    observed = [
        int(method_correct[s][q]) - int(baseline_correct[s][q])
        for s, q in observations
    ]
    draws = []
    n = len(observations)
    for _ in range(samples):
        total = 0
        for _ in range(n):
            s, q = observations[rng.randrange(n)]
            total += int(method_correct[s][q]) - int(baseline_correct[s][q])
        draws.append(total / n)
    draws.sort()
    return {
        "n": n,
        "mean": _mean(observed),
        "ci_low": draws[int(0.025 * (len(draws) - 1))],
        "ci_high": draws[int(0.975 * (len(draws) - 1))],
        "p_le_zero": sum(value <= 0 for value in draws) / len(draws),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys and key != "source_row":
                keys.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in keys})


def _markdown_table(rows: list[list[Any]], headers: list[str]) -> str:
    out = ["| " + " | ".join(headers) + " |"]
    out.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        out.append("| " + " | ".join(str(item) for item in row) + " |")
    return "\n".join(out)


def summarize(root: Path, output_dir: Path, bootstrap_samples: int) -> dict[str, Any]:
    opportunities = {seed: load_jsonl(root / OPPORTUNITY_PATHS[seed]) for seed in SEEDS}
    seed_latency = {seed: _latency_by_qid(root / NO_THINK_PATHS[seed]) for seed in SEEDS}
    think_latency = {seed: _latency_by_qid(root / THINK12K_PATHS[seed]) for seed in SEEDS}

    baselines: dict[str, dict[int, dict[str, Any]]] = defaultdict(dict)
    for seed in SEEDS:
        baselines["no_think"][seed] = _seed_from_opportunity(opportunities[seed])
        baselines["think12k"][seed] = _think12k_from_opportunity(opportunities[seed])
        baselines["think8k"][seed] = _baseline_from_records(root / THINK8K_PATHS[seed])
        baselines["sc3_no_think"][seed] = _baseline_from_records(root / SC3_PATHS[seed])
        baselines["no_think"][seed]["avg_latency_ms"] = seed_latency[seed][1]
        baselines["think12k"][seed]["avg_latency_ms"] = think_latency[seed][1]
        baselines["think8k"][seed]["avg_latency_ms"] = _nested_summary(_load_json(root / THINK8K_PATHS[seed])).get("avg_latency_ms", 0.0)
        baselines["sc3_no_think"][seed]["avg_latency_ms"] = _nested_summary(_load_json(root / SC3_PATHS[seed])).get("avg_latency_ms", 0.0)

    loso_train_sets = {
        17: [opportunities[7], opportunities[23]],
        7: [opportunities[17], opportunities[23]],
        23: [opportunities[17], opportunities[7]],
    }

    method_defs = {
        "GBDT top30 strict": ("gbdt", 0.3, "strict"),
        "GBDT top50 strict": ("gbdt", 0.5, "strict"),
        "GBDT top30 main": ("gbdt", 0.3, "main"),
        "GBDT top50 main": ("gbdt", 0.5, "main"),
        "Trace top30 strict": ("trace_length", 0.3, "strict"),
        "Trace top50 strict": ("trace_length", 0.5, "strict"),
        "Random top30 strict": ("random", 0.3, "strict"),
        "Random top50 strict": ("random", 0.5, "strict"),
        "Heuristic strict": ("heuristic", None, "strict"),
    }

    frontier_summaries: dict[str, dict[int, dict[str, Any]]] = defaultdict(dict)
    recomputed_correct: dict[str, dict[int, dict[str, bool]]] = defaultdict(dict)
    streaming_summaries: dict[str, dict[int, dict[str, Any]]] = defaultdict(dict)
    for seed in SEEDS:
        rows = _frontier_rows(root / FRONTIER_LOSO_PATHS[seed])
        train_rows = [row for part in loso_train_sets[seed] for row in part]
        for label, (method, rate, adoption_filter) in method_defs.items():
            if method == "random":
                row = _pick_frontier_row(
                    rows,
                    method=method,
                    rate=float(rate),
                    adoption_filter=adoption_filter,
                    selection_mode="topk_eval",
                )
                frontier_summaries[label][seed] = _row_summary(row)
                frontier_summaries[label][seed]["avg_latency_ms_estimated"] = (
                    seed_latency[seed][1]
                    + float(frontier_summaries[label][seed]["trigger_rate"]) * think_latency[seed][1]
                )
                continue
            if method == "heuristic":
                row = _pick_frontier_row(
                    rows,
                    method=method,
                    rate=None,
                    adoption_filter=adoption_filter,
                    selection_mode="heuristic_binary",
                )
                frontier_summaries[label][seed] = _row_summary(row)
                summary, correctness, selected_qids = _simulate_selected_with_latency(
                    train_rows,
                    opportunities[seed],
                    method=method,
                    rate=0.0,
                    adoption_filter=adoption_filter,
                    random_seed=seed,
                    seed_latency=seed_latency[seed][0],
                    think_latency=think_latency[seed][0],
                )
            else:
                row = _pick_frontier_row(
                    rows,
                    method=method,
                    rate=float(rate),
                    adoption_filter=adoption_filter,
                    selection_mode="topk_eval",
                )
                frontier_summaries[label][seed] = _row_summary(row)
                summary, correctness, selected_qids = _simulate_selected_with_latency(
                    train_rows,
                    opportunities[seed],
                    method=method,
                    rate=float(rate),
                    adoption_filter=adoption_filter,
                    random_seed=seed,
                    seed_latency=seed_latency[seed][0],
                    think_latency=think_latency[seed][0],
                )
            # Keep the recomputed summary nearby as a guard against accidental row mismatch.
            frontier_summaries[label][seed]["recomputed_accuracy"] = summary["accuracy"]
            frontier_summaries[label][seed]["avg_latency_ms_estimated"] = summary.get("avg_latency_ms_estimated", 0.0)
            frontier_summaries[label][seed]["selected_qids"] = sorted(selected_qids) if "selected_qids" in locals() else []
            recomputed_correct[label][seed] = correctness

        for label, rate in [("GBDT threshold30 strict", 0.3), ("GBDT threshold50 strict", 0.5)]:
            row = _pick_frontier_row(
                rows,
                method="gbdt",
                rate=rate,
                adoption_filter="strict",
                selection_mode="threshold_from_train",
            )
            streaming_summaries[label][seed] = _row_summary(row)

    table_methods = [
        ("`/no_think`", baselines["no_think"]),
        ("SC@3 `/no_think`", baselines["sc3_no_think"]),
        ("always `/think@8k`", baselines["think8k"]),
        ("always `/think@12k`", baselines["think12k"]),
        ("GBDT top30 strict", frontier_summaries["GBDT top30 strict"]),
        ("GBDT top50 strict", frontier_summaries["GBDT top50 strict"]),
        ("Trace top50 strict", frontier_summaries["Trace top50 strict"]),
        ("Random top50 strict", frontier_summaries["Random top50 strict"]),
        ("Heuristic strict", frontier_summaries["Heuristic strict"]),
    ]

    main_rows: list[dict[str, Any]] = []
    for method, per_seed in table_methods:
        accs = [per_seed[seed]["accuracy"] for seed in SEEDS]
        toks = [per_seed[seed]["avg_total_tokens"] for seed in SEEDS]
        row: dict[str, Any] = {
            "method": method,
            "seed17_acc": 100 * per_seed[17]["accuracy"],
            "seed7_acc": 100 * per_seed[7]["accuracy"],
            "seed23_acc": 100 * per_seed[23]["accuracy"],
            "mean_acc": 100 * _mean(accs),
            "std_acc": 100 * _std(accs),
            "avg_tokens": _mean(toks),
            "harmful": sum(int(per_seed[seed].get("harmful", 0)) for seed in SEEDS),
            "helpful": sum(int(per_seed[seed].get("helpful", 0)) for seed in SEEDS),
            "trigger_rate": _mean([float(per_seed[seed].get("trigger_rate", 0.0)) for seed in SEEDS]),
            "adoption_rate": _mean([float(per_seed[seed].get("adoption_rate", 0.0)) for seed in SEEDS]),
            "avg_latency_ms": _mean([float(per_seed[seed].get("avg_latency_ms", per_seed[seed].get("avg_latency_ms_estimated", 0.0))) for seed in SEEDS]),
        }
        main_rows.append(row)

    safety_rows: list[dict[str, Any]] = []
    for label in ["GBDT top30 main", "GBDT top30 strict", "GBDT top50 main", "GBDT top50 strict"]:
        per_seed = frontier_summaries[label]
        safety_rows.append(
            {
                "method": label,
                "mean_acc": 100 * _mean([per_seed[seed]["accuracy"] for seed in SEEDS]),
                "avg_tokens": _mean([per_seed[seed]["avg_total_tokens"] for seed in SEEDS]),
                "helpful": sum(per_seed[seed]["helpful"] for seed in SEEDS),
                "harmful": sum(per_seed[seed]["harmful"] for seed in SEEDS),
                "wrong_to_wrong": sum(per_seed[seed]["wrong_to_wrong"] for seed in SEEDS),
                "trigger_rate": _mean([per_seed[seed]["trigger_rate"] for seed in SEEDS]),
                "adoption_rate": _mean([per_seed[seed]["adoption_rate"] for seed in SEEDS]),
            }
        )

    frontier_rows: list[dict[str, Any]] = []
    for label in [
        "GBDT top30 strict",
        "GBDT top50 strict",
        "Trace top30 strict",
        "Trace top50 strict",
        "Random top30 strict",
        "Random top50 strict",
        "Heuristic strict",
    ]:
        per_seed = frontier_summaries[label]
        frontier_rows.append(
            {
                "method": label,
                "mean_acc": 100 * _mean([per_seed[seed]["accuracy"] for seed in SEEDS]),
                "std_acc": 100 * _std([per_seed[seed]["accuracy"] for seed in SEEDS]),
                "avg_tokens": _mean([per_seed[seed]["avg_total_tokens"] for seed in SEEDS]),
                "trigger_rate": _mean([per_seed[seed]["trigger_rate"] for seed in SEEDS]),
                "helpful": sum(per_seed[seed]["helpful"] for seed in SEEDS),
                "harmful": sum(per_seed[seed]["harmful"] for seed in SEEDS),
                "wrong_to_wrong": sum(per_seed[seed]["wrong_to_wrong"] for seed in SEEDS),
                "avg_latency_ms": _mean([float(per_seed[seed].get("avg_latency_ms_estimated", 0.0)) for seed in SEEDS]),
            }
        )

    streaming_rows: list[dict[str, Any]] = []
    for label in ["GBDT threshold30 strict", "GBDT threshold50 strict"]:
        per_seed = streaming_summaries[label]
        streaming_rows.append(
            {
                "method": label,
                "target_rate": 0.3 if "30" in label else 0.5,
                "actual_trigger_rate": _mean([per_seed[seed]["trigger_rate"] for seed in SEEDS]),
                "mean_acc": 100 * _mean([per_seed[seed]["accuracy"] for seed in SEEDS]),
                "std_acc": 100 * _std([per_seed[seed]["accuracy"] for seed in SEEDS]),
                "avg_tokens": _mean([per_seed[seed]["avg_total_tokens"] for seed in SEEDS]),
                "helpful": sum(per_seed[seed]["helpful"] for seed in SEEDS),
                "harmful": sum(per_seed[seed]["harmful"] for seed in SEEDS),
                "wrong_to_wrong": sum(per_seed[seed]["wrong_to_wrong"] for seed in SEEDS),
            }
        )

    # External calibration source ablation.
    college_rows: list[dict[str, Any]] = []
    for seed in SEEDS:
        rows = _frontier_rows(root / COLLEGE_FRONTIER_PATHS[seed])
        for label, method, rate, adoption_filter in [
            ("College2400 GBDT top50 strict", "gbdt", 0.5, "strict"),
            ("College2400 Trace top50 strict", "trace_length", 0.5, "strict"),
            ("College2400 Heuristic strict", "heuristic", None, "strict"),
        ]:
            row = _pick_frontier_row(
                rows,
                method=method,
                rate=rate,
                adoption_filter=adoption_filter,
                selection_mode="heuristic_binary" if method == "heuristic" else "topk_eval",
            )
            college_rows.append({"seed": seed, "method": label, **_row_summary(row)})

    college_summary: list[dict[str, Any]] = []
    for label in sorted({row["method"] for row in college_rows}):
        subset = [row for row in college_rows if row["method"] == label]
        college_summary.append(
            {
                "method": label,
                "mean_acc": 100 * _mean([row["accuracy"] for row in subset]),
                "avg_tokens": _mean([row["avg_total_tokens"] for row in subset]),
                "helpful": sum(row["helpful"] for row in subset),
                "harmful": sum(row["harmful"] for row in subset),
            }
        )

    # Paired bootstrap for deterministic methods.
    baseline_correct = {
        "no_think": {seed: baselines["no_think"][seed]["qid_correct"] for seed in SEEDS},
        "sc3_no_think": {seed: baselines["sc3_no_think"][seed]["qid_correct"] for seed in SEEDS},
        "think8k": {seed: baselines["think8k"][seed]["qid_correct"] for seed in SEEDS},
    }
    bootstrap = {
        "gbdt50_vs_no_think": _qid_bootstrap_delta(
            recomputed_correct["GBDT top50 strict"],
            baseline_correct["no_think"],
            samples=bootstrap_samples,
            seed=123,
        ),
        "gbdt50_vs_sc3": _qid_bootstrap_delta(
            recomputed_correct["GBDT top50 strict"],
            baseline_correct["sc3_no_think"],
            samples=bootstrap_samples,
            seed=124,
        ),
        "gbdt50_vs_think8k": _qid_bootstrap_delta(
            recomputed_correct["GBDT top50 strict"],
            baseline_correct["think8k"],
            samples=bootstrap_samples,
            seed=125,
        ),
        "gbdt50_vs_trace50": _qid_bootstrap_delta(
            recomputed_correct["GBDT top50 strict"],
            recomputed_correct["Trace top50 strict"],
            samples=bootstrap_samples,
            seed=126,
        ),
        "gbdt30_vs_trace30": _qid_bootstrap_delta(
            recomputed_correct["GBDT top30 strict"],
            recomputed_correct["Trace top30 strict"],
            samples=bootstrap_samples,
            seed=127,
        ),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "math500_main_table.csv", main_rows)
    _write_csv(output_dir / "math500_trigger_frontier.csv", frontier_rows)
    _write_csv(output_dir / "math500_safety_gate_table.csv", safety_rows)
    _write_csv(output_dir / "math500_collegemath_calibration.csv", college_summary)
    _write_csv(output_dir / "math500_streaming_threshold.csv", streaming_rows)
    feature_ablation = _feature_ablation_summary(opportunities=opportunities, output_dir=output_dir)

    payload = {
        "main_table": main_rows,
        "trigger_frontier": frontier_rows,
        "safety_gate": safety_rows,
        "college_calibration": college_summary,
        "streaming_threshold": streaming_rows,
        "feature_ablation": feature_ablation,
        "bootstrap": bootstrap,
    }
    dump_json(output_dir / "bora_switch_calibrated_summary.json", payload)

    md: list[str] = []
    md.append("# BORA-Switch-Calibrated Summary\n")
    md.append("## MATH500 Main Table\n")
    md.append(
        _markdown_table(
            [
                [
                    row["method"],
                    f"{row['seed17_acc']:.2f}",
                    f"{row['seed7_acc']:.2f}",
                    f"{row['seed23_acc']:.2f}",
                    f"{row['mean_acc']:.2f}±{row['std_acc']:.2f}",
                    f"{row['avg_tokens']:.1f}",
                    f"{row['avg_latency_ms']/1000:.2f}",
                    row["helpful"] or "-",
                    row["harmful"] or "0",
                ]
                for row in main_rows
            ],
            ["Method", "Seed17", "Seed7", "Seed23", "Mean±std", "Avg tokens", "Latency(s)", "Helpful", "Harmful"],
        )
    )
    md.append("\n## Trigger Frontier\n")
    md.append(
        _markdown_table(
            [
                [
                    row["method"],
                    f"{row['mean_acc']:.2f}±{row['std_acc']:.2f}",
                    f"{row['avg_tokens']:.1f}",
                    f"{100*row['trigger_rate']:.1f}",
                    row["helpful"],
                    row["harmful"],
                    row["wrong_to_wrong"],
                ]
                for row in frontier_rows
            ],
            ["Method", "Acc", "Avg tokens", "Trigger %", "Helpful", "Harmful", "Wrong→wrong"],
        )
    )
    md.append("\n## Safety Gate Tradeoff\n")
    md.append(
        _markdown_table(
            [
                [
                    row["method"],
                    f"{row['mean_acc']:.2f}",
                    f"{row['avg_tokens']:.1f}",
                    row["helpful"],
                    row["harmful"],
                    row["wrong_to_wrong"],
                    f"{100*row['adoption_rate']:.1f}",
                ]
                for row in safety_rows
            ],
            ["Method", "Mean acc", "Avg tokens", "Helpful", "Harmful", "Wrong→wrong", "Adopt %"],
        )
    )
    md.append("\n## Streaming Threshold Evaluation\n")
    md.append(
        _markdown_table(
            [
                [
                    row["method"],
                    f"{100*row['target_rate']:.1f}",
                    f"{100*row['actual_trigger_rate']:.1f}",
                    f"{row['mean_acc']:.2f}±{row['std_acc']:.2f}",
                    f"{row['avg_tokens']:.1f}",
                    row["helpful"],
                    row["harmful"],
                ]
                for row in streaming_rows
            ],
            ["Method", "Target %", "Actual trigger %", "Acc", "Avg tokens", "Helpful", "Harmful"],
        )
    )
    md.append("\n## Feature Ablation\n")
    md.append(
        _markdown_table(
            [
                [
                    row["feature_subset"],
                    int(row["feature_count"]),
                    f"{100*row['rate']:.0f}",
                    f"{row['mean_acc']:.2f}±{row['std_acc']:.2f}",
                    f"{row['avg_tokens']:.1f}",
                    row["helpful"],
                    row["harmful"],
                ]
                for row in feature_ablation
            ],
            ["Feature subset", "#feat", "Rate %", "Acc", "Avg tokens", "Helpful", "Harmful"],
        )
    )
    md.append("\n## Calibration Source Ablation\n")
    md.append(
        _markdown_table(
            [
                [
                    row["method"],
                    f"{row['mean_acc']:.2f}",
                    f"{row['avg_tokens']:.1f}",
                    row["helpful"],
                    row["harmful"],
                ]
                for row in college_summary
            ],
            ["Method", "Mean acc", "Avg tokens", "Helpful", "Harmful"],
        )
    )
    md.append("\n## Paired Bootstrap Deltas\n")
    md.append(
        _markdown_table(
            [
                [
                    name,
                    f"{100*row['mean']:.2f} pp",
                    f"[{100*row['ci_low']:.2f}, {100*row['ci_high']:.2f}]",
                    f"{row['p_le_zero']:.3f}",
                ]
                for name, row in bootstrap.items()
            ],
            ["Comparison", "Mean delta", "95% CI", "P(delta<=0)"],
        )
    )
    (output_dir / "bora_switch_calibrated_summary.md").write_text("\n\n".join(md), encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/paper_tables"))
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    args = parser.parse_args()
    payload = summarize(args.root, args.output_dir, args.bootstrap_samples)
    print((args.output_dir / "bora_switch_calibrated_summary.md").resolve())
    for row in payload["main_table"]:
        print(f"{row['method']}: {row['mean_acc']:.2f}% tokens={row['avg_tokens']:.1f}")


if __name__ == "__main__":
    main()
