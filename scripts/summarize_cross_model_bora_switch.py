from __future__ import annotations

import argparse
import json
import statistics as stats
from pathlib import Path
from typing import Any


SEEDS = [17, 7, 23]


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def _records(payload: dict[str, Any]) -> list[dict[str, Any]]:
    if isinstance(payload.get("records"), list):
        return payload["records"]
    for value in payload.values():
        if isinstance(value, dict) and isinstance(value.get("records"), list):
            return value["records"]
    raise ValueError("Could not find records in payload")


def _summary_from_result(path: Path) -> dict[str, Any]:
    payload = _load_json(path)
    records = _records(payload)
    n = len(records)
    correct = sum(bool(row.get("correct")) for row in records)
    tokens = [float(row.get("total_tokens") or row.get("solver_tokens") or 0.0) for row in records]
    return {
        "count": n,
        "correct": correct,
        "accuracy": correct / max(1, n),
        "avg_total_tokens": sum(tokens) / max(1, len(tokens)),
    }


def _first_existing(paths: list[Path]) -> Path:
    for path in paths:
        if path.exists():
            return path
    raise FileNotFoundError("None of the candidate result files exists: " + ", ".join(str(p) for p in paths))


def _frontier_row(path: Path, method: str, rate: float) -> dict[str, Any]:
    payload = _load_json(path)
    rows = payload.get("results")
    if not isinstance(rows, list):
        raise ValueError(f"No results list in {path}")
    candidates = [
        row
        for row in rows
        if row.get("method") == method
        and row.get("selection_mode") == "topk_eval"
        and row.get("adoption_filter") == "strict"
        and abs(float(row.get("target_rate") or 0.0) - rate) < 1e-9
    ]
    if not candidates:
        raise KeyError(f"Missing {method} topk strict rate={rate} in {path}")
    return candidates[0]


def _mean(values: list[float]) -> float:
    return sum(values) / max(1, len(values))


def _std(values: list[float]) -> float:
    return stats.stdev(values) if len(values) > 1 else 0.0


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "accuracy_mean": _mean([float(row["accuracy"]) for row in rows]),
        "accuracy_std": _std([float(row["accuracy"]) for row in rows]),
        "avg_total_tokens": _mean([float(row["avg_total_tokens"]) for row in rows]),
        "correct_total": sum(int(row.get("correct", row.get("final_correct", 0))) for row in rows),
        "count_total": sum(int(row.get("count", 0)) for row in rows),
        "helpful_total": sum(int(row.get("helpful", 0)) for row in rows),
        "harmful_total": sum(int(row.get("harmful", 0)) for row in rows),
        "trigger_total": sum(int(row.get("trigger_count", 0)) for row in rows),
        "adoption_total": sum(int(row.get("adoption_count", 0)) for row in rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-tag", required=True)
    parser.add_argument("--root", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    args = parser.parse_args()

    root = Path(args.root)
    per_seed: dict[str, dict[str, Any]] = {}
    groups: dict[str, list[dict[str, Any]]] = {
        "no_think": [],
        "think12k": [],
        "gbdt_top30_strict": [],
        "gbdt_top50_strict": [],
        "trace_top50_strict": [],
        "random_top50_strict": [],
    }

    for seed in SEEDS:
        seed_dir = root / f"seed{seed}"
        no_think = _summary_from_result(
            _first_existing(
                [
                    seed_dir / f"no_think_seed{seed}.fixed.json",
                    seed_dir / f"no_think_seed{seed}.json",
                ]
            )
        )
        think12 = _summary_from_result(
            _first_existing(
                [
                    seed_dir / f"think12k_seed{seed}.fixed.json",
                    seed_dir / f"think12k_seed{seed}.json",
                    seed_dir / f"think_seed{seed}.fixed.json",
                    seed_dir / f"think_seed{seed}.json",
                ]
            )
        )
        frontier = root / f"frontier_eval{seed}.json"
        gbdt30 = _frontier_row(frontier, "gbdt", 0.3)
        gbdt50 = _frontier_row(frontier, "gbdt", 0.5)
        trace50 = _frontier_row(frontier, "trace_length", 0.5)
        random50 = _frontier_row(frontier, "random", 0.5)
        per_seed[str(seed)] = {
            "no_think": no_think,
            "think12k": think12,
            "gbdt_top30_strict": gbdt30,
            "gbdt_top50_strict": gbdt50,
            "trace_top50_strict": trace50,
            "random_top50_strict": random50,
        }
        groups["no_think"].append(no_think)
        groups["think12k"].append(think12)
        groups["gbdt_top30_strict"].append(gbdt30)
        groups["gbdt_top50_strict"].append(gbdt50)
        groups["trace_top50_strict"].append(trace50)
        groups["random_top50_strict"].append(random50)

    aggregate = {name: _aggregate(rows) for name, rows in groups.items()}
    payload = {
        "model_tag": args.model_tag,
        "root": str(root),
        "seeds": SEEDS,
        "per_seed": per_seed,
        "aggregate": aggregate,
    }
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_json).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [f"# Cross-Model BORA-Switch Summary: `{args.model_tag}`", ""]
    lines.append("| Method | Mean acc | Std | Avg tokens | Helpful | Harmful |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
    for name, row in aggregate.items():
        lines.append(
            f"| `{name}` | {100*row['accuracy_mean']:.2f} | "
            f"{100*row['accuracy_std']:.2f} | {row['avg_total_tokens']:.1f} | "
            f"{row['helpful_total']} | {row['harmful_total']} |"
        )
    lines.append("")
    lines.append("Primary check: `gbdt_top50_strict` should outperform `random_top50_strict` and `trace_top50_strict` at comparable trigger rate, and should recover a meaningful fraction of the `think12k - no_think` gap.")
    Path(args.output_md).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(Path(args.output_md).read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
