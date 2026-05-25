from __future__ import annotations

from statistics import mean
from typing import Any

import numpy as np


def summarize_records(records: list[dict[str, Any]]) -> dict[str, float]:
    if not records:
        return {
            "accuracy": 0.0,
            "avg_total_tokens": 0.0,
            "avg_solver_tokens": 0.0,
            "avg_verifier_tokens": 0.0,
            "avg_latency_ms": 0.0,
            "avg_branches_used": 0.0,
            "stop_rate": 0.0,
            "negative_flip_rate": 0.0,
        }
    return {
        "accuracy": mean(float(record["correct"]) for record in records),
        "avg_total_tokens": mean(record["total_tokens"] for record in records),
        "avg_solver_tokens": mean(record["solver_tokens"] for record in records),
        "avg_verifier_tokens": mean(record["verifier_tokens"] for record in records),
        "avg_latency_ms": mean(record["latency_ms"] for record in records),
        "avg_branches_used": mean(record["branches_used"] for record in records),
        "stop_rate": mean(float("STOP" in record["actions"]) for record in records),
        "negative_flip_rate": mean(
            float(record.get("metadata", {}).get("negative_flip", False)) for record in records
        ),
    }


def bootstrap_accuracy_ci(
    records: list[dict[str, Any]],
    num_samples: int = 1000,
    seed: int = 0,
) -> dict[str, float]:
    if not records:
        return {"low": 0.0, "high": 0.0}
    rng = np.random.default_rng(seed)
    accuracies = []
    correct = np.asarray([float(record["correct"]) for record in records], dtype=float)
    for _ in range(num_samples):
        sample = rng.choice(correct, size=len(correct), replace=True)
        accuracies.append(float(np.mean(sample)))
    return {
        "low": float(np.quantile(accuracies, 0.05)),
        "high": float(np.quantile(accuracies, 0.95)),
    }


def build_frontier(summaries: dict[str, dict[str, float]]) -> list[dict[str, float | str]]:
    frontier: list[dict[str, float | str]] = []
    for name, summary in sorted(
        summaries.items(),
        key=lambda item: (item[1]["avg_total_tokens"], -item[1]["accuracy"]),
    ):
        if not frontier or summary["accuracy"] > frontier[-1]["accuracy"]:
            frontier.append(
                {
                    "name": name,
                    "accuracy": summary["accuracy"],
                    "avg_total_tokens": summary["avg_total_tokens"],
                }
            )
    return frontier


def format_summary_table(summaries: dict[str, dict[str, float]]) -> str:
    lines = [
        "name\taccuracy\tavg_total_tokens\tavg_solver_tokens\tavg_verifier_tokens\tavg_latency_ms"
    ]
    for name, summary in summaries.items():
        lines.append(
            f"{name}\t{summary['accuracy']:.3f}\t{summary['avg_total_tokens']:.1f}\t"
            f"{summary['avg_solver_tokens']:.1f}\t{summary['avg_verifier_tokens']:.1f}\t"
            f"{summary['avg_latency_ms']:.1f}"
        )
    return "\n".join(lines)
