from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from analyze_trigger_frontier import (  # type: ignore
    _as_float,
    _evaluate_scores,
    _feature_names,
    _fit_scores,
)
from bora.common import load_jsonl


TRACE_FEATURES = {
    "seed_total_tokens",
    "seed_completion_tokens",
    "seed_prompt_tokens",
    "seed_completion_head_numeric_count",
    "sf_trace_words",
    "sf_trace_chars",
    "sf_trace_sentences",
}


def _is_answer_parse(name: str) -> bool:
    return name.startswith("seed_answer_")


def _is_trace(name: str) -> bool:
    return name in TRACE_FEATURES or name.startswith("sf_trace_")


def _is_old_bora(name: str) -> bool:
    return (
        name.startswith("old_")
        or name.startswith("sf_trigger_")
        or name.startswith("sf_verifier_")
        or name.startswith("sf_rescue_")
    )


def _is_problem_shape(name: str) -> bool:
    return name.startswith("question_")


def _is_seed_consistency(name: str) -> bool:
    return name.startswith("seed_sc_")


def _feature_sets(all_names: list[str]) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = {
        "all_features": list(all_names),
        "trace_stats_only": [n for n in all_names if _is_trace(n)],
        "seed_consistency_only": [n for n in all_names if _is_seed_consistency(n)],
        "answer_parse_only": [n for n in all_names if _is_answer_parse(n)],
        "problem_shape_only": [n for n in all_names if _is_problem_shape(n)],
        "old_bora_only": [n for n in all_names if _is_old_bora(n)],
        "all_minus_trace": [n for n in all_names if not _is_trace(n)],
        "all_minus_seed_consistency": [n for n in all_names if not _is_seed_consistency(n)],
        "all_minus_answer_parse": [n for n in all_names if not _is_answer_parse(n)],
        "all_minus_problem_shape": [n for n in all_names if not _is_problem_shape(n)],
        "all_minus_old_bora": [n for n in all_names if not _is_old_bora(n)],
        "parse_plus_problem": [
            n for n in all_names if _is_answer_parse(n) or _is_problem_shape(n)
        ],
        "trace_plus_problem": [n for n in all_names if _is_trace(n) or _is_problem_shape(n)],
        "trace_plus_seed_consistency": [
            n for n in all_names if _is_trace(n) or _is_seed_consistency(n)
        ],
        "problem_plus_seed_consistency": [
            n for n in all_names if _is_problem_shape(n) or _is_seed_consistency(n)
        ],
    }
    return {name: feats for name, feats in groups.items() if feats}


def _labels(rows: list[dict[str, Any]], name: str) -> list[int]:
    return [int(bool(row.get(name))) for row in rows]


def _auc_metrics(y: list[int], scores: list[float]) -> dict[str, float | None]:
    if len(set(y)) < 2:
        return {"auroc": None, "auprc": None}
    try:
        from sklearn.metrics import average_precision_score, roc_auc_score
    except Exception:
        return {"auroc": None, "auprc": None}
    return {
        "auroc": float(roc_auc_score(y, scores)),
        "auprc": float(average_precision_score(y, scores)),
    }


def _mean(values: list[float]) -> float:
    return sum(values) / max(1, len(values))


def _std(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    return statistics.stdev(values)


def _summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    keys = sorted({(r["feature_set"], r["model"], r["selection_mode"], r["adoption_filter"], r["target_rate"]) for r in rows})
    out: list[dict[str, Any]] = []
    for feature_set, model, selection_mode, adoption_filter, target_rate in keys:
        group = [
            r
            for r in rows
            if r["feature_set"] == feature_set
            and r["model"] == model
            and r["selection_mode"] == selection_mode
            and r["adoption_filter"] == adoption_filter
            and r["target_rate"] == target_rate
        ]
        out.append(
            {
                "feature_set": feature_set,
                "model": model,
                "selection_mode": selection_mode,
                "adoption_filter": adoption_filter,
                "target_rate": target_rate,
                "n_seeds": len(group),
                "mean_accuracy": _mean([float(r["accuracy"]) for r in group]),
                "std_accuracy": _std([float(r["accuracy"]) for r in group]),
                "mean_tokens": _mean([float(r["avg_total_tokens"]) for r in group]),
                "mean_trigger_rate": _mean([float(r["trigger_rate"]) for r in group]),
                "mean_adoption_rate": _mean([float(r["adoption_rate"]) for r in group]),
                "sum_helpful": sum(int(r["helpful"]) for r in group),
                "sum_harmful": sum(int(r["harmful"]) for r in group),
                "sum_wrong_to_wrong": sum(int(r["wrong_to_wrong"]) for r in group),
                "mean_helpful_auroc": _mean([float(r["helpful_auroc"]) for r in group if r.get("helpful_auroc") is not None]),
                "mean_helpful_auprc": _mean([float(r["helpful_auprc"]) for r in group if r.get("helpful_auprc") is not None]),
                "mean_harmful_auroc": _mean([float(r["harmful_auroc"]) for r in group if r.get("harmful_auroc") is not None]),
                "mean_harmful_auprc": _mean([float(r["harmful_auprc"]) for r in group if r.get("harmful_auprc") is not None]),
            }
        )
    return out


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _pct(value: float) -> str:
    return f"{100 * value:.2f}"


def _write_md(path: Path, summary: list[dict[str, Any]], *, feature_sets: dict[str, list[str]]) -> None:
    lines: list[str] = []
    lines.append("# Qwen3-8B Trigger Feature Ablation")
    lines.append("")
    lines.append("Fixed-eval MATH500, leave-one-seed-out over seeds 17/7/23. Scores use the same GBDT opportunity model form as the main BORA-Switch controller.")
    lines.append("")
    lines.append("## Feature Sets")
    lines.append("")
    lines.append("| Feature set | # features |")
    lines.append("| --- | ---: |")
    for name, feats in feature_sets.items():
        lines.append(f"| `{name}` | {len(feats)} |")
    lines.append("")

    for mode in ["topk_eval", "threshold_from_train"]:
        lines.append(f"## Strict Gate, `{mode}`")
        lines.append("")
        rows = [
            r
            for r in summary
            if r["model"] == "gbdt"
            and r["adoption_filter"] == "strict"
            and r["selection_mode"] == mode
            and float(r["target_rate"]) in {0.3, 0.5}
        ]
        rows.sort(key=lambda r: (float(r["target_rate"]), -float(r["mean_accuracy"])))
        lines.append("| Features | Rate | Acc mean±std | Tokens | Trigger | Helpful | Harmful | Help AUROC | Help AUPRC | Harm AUROC |")
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
        for r in rows:
            lines.append(
                "| `{feature_set}` | {rate:.0f}% | {acc}±{std} | {tokens:.1f} | {trig} | {helpful} | {harmful} | {hauc:.3f} | {hap:.3f} | {rauc:.3f} |".format(
                    feature_set=r["feature_set"],
                    rate=100 * float(r["target_rate"]),
                    acc=_pct(float(r["mean_accuracy"])),
                    std=100 * float(r["std_accuracy"]),
                    tokens=float(r["mean_tokens"]),
                    trig=_pct(float(r["mean_trigger_rate"])),
                    helpful=int(r["sum_helpful"]),
                    harmful=int(r["sum_harmful"]),
                    hauc=float(r["mean_helpful_auroc"]),
                    hap=float(r["mean_helpful_auprc"]),
                    rauc=float(r["mean_harmful_auroc"]),
                )
            )
        lines.append("")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="artifacts/remote_stage_eval_audit/math500_main_algorithm_fixed_20260521")
    parser.add_argument("--seeds", default="17,7,23")
    parser.add_argument("--rates", default="0.3,0.5")
    parser.add_argument("--filters", default="strict")
    parser.add_argument("--selection-modes", default="topk_eval,threshold_from_train")
    parser.add_argument("--models", default="gbdt")
    parser.add_argument("--harm-weight", type=float, default=1.0)
    parser.add_argument("--random-seed", type=int, default=17)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    root = Path(args.root)
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    rates = [float(x) for x in args.rates.split(",") if x.strip()]
    filters = [x.strip() for x in args.filters.split(",") if x.strip()]
    selection_modes = [x.strip() for x in args.selection_modes.split(",") if x.strip()]
    models = [x.strip() for x in args.models.split(",") if x.strip()]

    by_seed: dict[int, list[dict[str, Any]]] = {}
    all_rows: list[dict[str, Any]] = []
    for seed in seeds:
        rows = load_jsonl(root / f"seed{seed}" / f"opportunity_seed{seed}_think12k_fixed.jsonl")
        by_seed[seed] = rows
        all_rows.extend(rows)
    all_names = _feature_names(all_rows)
    feature_sets = _feature_sets(all_names)

    detail_rows: list[dict[str, Any]] = []
    for eval_seed in seeds:
        eval_rows = by_seed[eval_seed]
        train_rows = [row for seed in seeds if seed != eval_seed for row in by_seed[seed]]
        for feature_set, selected_features in feature_sets.items():
            for model in models:
                train_scores, eval_scores, eval_harm_prob, info = _fit_scores(
                    train_rows,
                    eval_rows,
                    model_name=model,
                    feature_names=selected_features,
                    harm_weight=args.harm_weight,
                    random_seed=args.random_seed + eval_seed,
                )
                help_metrics = _auc_metrics(_labels(eval_rows, "helpful"), eval_scores)
                harm_metrics = _auc_metrics(_labels(eval_rows, "harmful"), eval_harm_prob)
                results = _evaluate_scores(
                    method=model,
                    train_scores=train_scores,
                    eval_scores=eval_scores,
                    eval_rows=eval_rows,
                    rates=rates,
                    filters=filters,
                    selection_modes=selection_modes,
                )
                for result in results:
                    row = {
                        "eval_seed": eval_seed,
                        "feature_set": feature_set,
                        "feature_count": len(selected_features),
                        "model": model,
                        "helpful_auroc": help_metrics["auroc"],
                        "helpful_auprc": help_metrics["auprc"],
                        "harmful_auroc": harm_metrics["auroc"],
                        "harmful_auprc": harm_metrics["auprc"],
                        **{
                            key: value
                            for key, value in result.items()
                            if key not in {"helpful_qids", "harmful_qids"}
                        },
                    }
                    detail_rows.append(row)

    summary_rows = _summarize(detail_rows)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "feature_sets.json").write_text(json.dumps(feature_sets, indent=2), encoding="utf-8")
    (out_dir / "details.json").write_text(json.dumps(detail_rows, indent=2), encoding="utf-8")
    (out_dir / "summary.json").write_text(json.dumps(summary_rows, indent=2), encoding="utf-8")
    _write_csv(out_dir / "details.csv", detail_rows)
    _write_csv(out_dir / "summary.csv", summary_rows)
    _write_md(out_dir / "FEATURE_ABLATION_SUMMARY.md", summary_rows, feature_sets=feature_sets)
    print(f"wrote {out_dir}")


if __name__ == "__main__":
    main()
