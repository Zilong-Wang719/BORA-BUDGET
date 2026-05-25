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
    from bora.common import dump_json, dump_jsonl, load_jsonl
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

    def dump_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, sort_keys=True) + "\n")

import analyze_trigger_frontier as frontier


def _qid(row: dict[str, Any]) -> str:
    return str(row.get("qid"))


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _std(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    mu = _mean(values)
    return math.sqrt(sum((value - mu) ** 2 for value in values) / (len(values) - 1))


def _load_many(paths: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        rows.extend(load_jsonl(path))
    return rows


def _stratified_qid_folds(
    rows: list[dict[str, Any]],
    *,
    num_folds: int,
    random_seed: int,
) -> list[set[str]]:
    by_qid: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        qid = _qid(row)
        if qid:
            by_qid.setdefault(qid, []).append(row)

    qid_stats: list[tuple[str, int, float, float]] = []
    for qid, qrows in by_qid.items():
        seed_acc = _mean([1.0 if bool(row.get("seed_correct")) else 0.0 for row in qrows])
        helpful = _mean([1.0 if bool(row.get("helpful")) else 0.0 for row in qrows])
        trace_len = _mean(
            [
                frontier._as_float((row.get("features") or {}).get("sf_trace_words"), 0.0)
                or frontier._as_float((row.get("features") or {}).get("seed_total_tokens"), 0.0)
                for row in qrows
            ]
        )
        # Low seed accuracy and high trace length are put in different buckets
        # before round-robin assignment, which keeps folds balanced without
        # depending on task-specific metadata.
        qid_stats.append((qid, int(round(seed_acc * 10)), helpful, trace_len))

    rng = random.Random(random_seed)
    rng.shuffle(qid_stats)
    qid_stats.sort(key=lambda item: (item[1], item[2], item[3], item[0]))

    folds = [set() for _ in range(num_folds)]
    for idx, (qid, *_rest) in enumerate(qid_stats):
        folds[idx % num_folds].add(qid)
    return folds


def _summarize_fold_results(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        key = (
            row.get("method"),
            row.get("selection_mode"),
            row.get("target_rate"),
            row.get("adoption_filter"),
        )
        groups.setdefault(key, []).append(row)

    summary: list[dict[str, Any]] = []
    for (method, selection_mode, target_rate, adoption_filter), group in groups.items():
        count = sum(int(row.get("count", 0)) for row in group)
        mean_fields = [
            "accuracy",
            "delta_correct",
            "trigger_rate",
            "adoption_rate",
            "helpful",
            "harmful",
            "wrong_to_wrong",
            "avg_total_tokens",
            "avg_triggered_extra_tokens",
        ]
        out: dict[str, Any] = {
            "method": method,
            "selection_mode": selection_mode,
            "target_rate": target_rate,
            "adoption_filter": adoption_filter,
            "folds": len(group),
            "count": count,
        }
        for field in mean_fields:
            values = [float(row.get(field, 0.0) or 0.0) for row in group]
            out[f"{field}_mean"] = _mean(values)
            out[f"{field}_std"] = _std(values)
        summary.append(out)

    summary.sort(
        key=lambda row: (
            str(row["adoption_filter"]),
            str(row["selection_mode"]),
            float(row["target_rate"] or 0),
            str(row["method"]),
        )
    )
    return summary


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = list(rows[0].keys())
    lines = [",".join(keys)]
    for row in rows:
        values = []
        for key in keys:
            value = row.get(key)
            text = "" if value is None else str(value)
            if "," in text or "\n" in text or '"' in text:
                text = '"' + text.replace('"', '""') + '"'
            values.append(text)
        lines.append(",".join(values))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate BORA-Switch trigger frontiers on qid-disjoint folds."
    )
    parser.add_argument("--rollouts", nargs="+", required=True)
    parser.add_argument("--num-folds", type=int, default=5)
    parser.add_argument("--models", default="gbdt,trace_length,random")
    parser.add_argument("--rates", default="0.3,0.5")
    parser.add_argument("--filters", default="strict")
    parser.add_argument("--selection-modes", default="topk_eval,threshold_from_train")
    parser.add_argument("--harm-weight", type=float, default=2.0)
    parser.add_argument("--random-trials", type=int, default=50)
    parser.add_argument("--random-seed", type=int, default=0)
    parser.add_argument("--output", required=True)
    parser.add_argument("--output-csv", default=None)
    parser.add_argument("--work-dir", default=None)
    args = parser.parse_args()

    all_rows = _load_many(args.rollouts)
    if not all_rows:
        raise ValueError("No rollout rows loaded.")

    folds = _stratified_qid_folds(
        all_rows,
        num_folds=max(2, int(args.num_folds)),
        random_seed=int(args.random_seed),
    )
    feature_names = frontier._feature_names(all_rows)
    rates = [float(item) for item in args.rates.split(",") if item.strip()]
    filters = [item.strip() for item in args.filters.split(",") if item.strip()]
    selection_modes = [item.strip() for item in args.selection_modes.split(",") if item.strip()]
    models = [item.strip() for item in args.models.split(",") if item.strip()]

    work_dir = Path(args.work_dir) if args.work_dir else Path(args.output).with_suffix("").parent / "problem_disjoint_folds"
    work_dir.mkdir(parents=True, exist_ok=True)

    fold_payloads: list[dict[str, Any]] = []
    flat_results: list[dict[str, Any]] = []
    for fold_idx, eval_qids in enumerate(folds):
        train_rows = [row for row in all_rows if _qid(row) not in eval_qids]
        eval_rows = [row for row in all_rows if _qid(row) in eval_qids]
        if not train_rows or not eval_rows:
            continue

        # Keep fold materialized for auditability and easy reruns with the
        # single-split frontier script.
        train_path = work_dir / f"fold{fold_idx}_train.jsonl"
        eval_path = work_dir / f"fold{fold_idx}_eval.jsonl"
        dump_jsonl(train_path, train_rows)
        dump_jsonl(eval_path, eval_rows)

        results: list[dict[str, Any]] = []
        model_info: dict[str, Any] = {}
        for model_name in models:
            if model_name == "random":
                results.extend(
                    frontier._random_results(
                        eval_rows=eval_rows,
                        rates=rates,
                        filters=filters,
                        trials=int(args.random_trials),
                        random_seed=int(args.random_seed) + fold_idx,
                    )
                )
                continue
            if model_name == "heuristic":
                train_scores = frontier._heuristic_scores(train_rows)
                eval_scores = frontier._heuristic_scores(eval_rows)
                model_info[model_name] = {"kind": "old_trigger_nonbranch_or_long300"}
                results.extend(
                    frontier._evaluate_scores(
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

            train_scores, eval_scores, _eval_harm, info = frontier._fit_scores(
                train_rows,
                eval_rows,
                model_name=model_name,
                feature_names=feature_names,
                harm_weight=float(args.harm_weight),
                random_seed=int(args.random_seed) + fold_idx,
            )
            model_info[model_name] = info
            results.extend(
                frontier._evaluate_scores(
                    method=model_name,
                    train_scores=train_scores,
                    eval_scores=eval_scores,
                    eval_rows=eval_rows,
                    rates=rates,
                    filters=filters,
                    selection_modes=selection_modes,
                )
            )

        for row in results:
            row["fold"] = fold_idx
            row["eval_qids"] = len(eval_qids)
        flat_results.extend(results)
        fold_payloads.append(
            {
                "fold": fold_idx,
                "train_count": len(train_rows),
                "eval_count": len(eval_rows),
                "eval_qids": sorted(eval_qids),
                "model_info": model_info,
                "results": results,
            }
        )

    summary = _summarize_fold_results(flat_results)
    payload = {
        "inputs": vars(args),
        "total_rows": len(all_rows),
        "total_qids": len({ _qid(row) for row in all_rows if _qid(row) }),
        "feature_names": feature_names,
        "folds": fold_payloads,
        "summary": summary,
    }
    dump_json(args.output, payload)
    if args.output_csv:
        _write_csv(Path(args.output_csv), summary)

    print(f"wrote qid-disjoint frontier to {args.output}")
    for row in sorted(summary, key=lambda item: (-float(item.get("accuracy_mean", 0)), float(item.get("avg_total_tokens_mean", 0))))[:12]:
        print(
            f"{row['method']} {row['selection_mode']} rate={row['target_rate']} "
            f"filter={row['adoption_filter']} acc={100*row['accuracy_mean']:.2f}±{100*row['accuracy_std']:.2f} "
            f"harm={row['harmful_mean']:.2f} tokens={row['avg_total_tokens_mean']:.1f}"
        )


if __name__ == "__main__":
    main()
