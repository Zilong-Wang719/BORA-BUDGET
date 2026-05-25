from __future__ import annotations

import argparse
import csv
import json
import random
import statistics as stats
from pathlib import Path
from typing import Any


SEEDS = [17, 7, 23]
ARMS = [8, 12]
DEFAULT_BUDGETS = [800.0, 1000.0, 1200.0, 1300.0, 1500.0, 2000.0, 2500.0, 2681.7, 3000.0, 3500.0, 3589.6, 3867.8, 4303.1]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _std(values: list[float]) -> float:
    return stats.stdev(values) if len(values) > 1 else 0.0


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out


def _seed_tokens(row: dict[str, Any]) -> float:
    return _as_float(row.get("seed_total_tokens"))


def _arm_tokens(row: dict[str, Any], arm: int) -> float:
    return _as_float(row.get(f"think{arm}_total_tokens"))


def _arm_final_correct(row: dict[str, Any], arm: int) -> bool:
    return bool(row.get(f"think{arm}_final_correct"))


def _arm_gate_pass(row: dict[str, Any], arm: int) -> bool:
    return bool(row.get(f"think{arm}_gate_pass"))


def _trace_score(row: dict[str, Any]) -> float:
    features = row.get("features") or {}
    return (
        _as_float(features.get("sf_trace_words"))
        or _as_float(features.get("seed_total_tokens"))
        or _seed_tokens(row)
    )


def _rows_by_seed(rows: list[dict[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    grouped: dict[int, list[dict[str, Any]]] = {seed: [] for seed in SEEDS}
    for row in rows:
        seed = int(row.get("seed"))
        if seed in grouped:
            grouped[seed].append(row)
    for seed in grouped:
        grouped[seed].sort(key=lambda row: str(row.get("qid")))
    return grouped


def _simulate(rows: list[dict[str, Any]], actions: dict[str, int | None]) -> dict[str, Any]:
    correct = helpful = harmful = wrong_to_wrong = adopted = triggered = 0
    tokens: list[float] = []
    counts = {"accept": 0, "think8": 0, "think12": 0}
    helpful_qids: list[str] = []
    harmful_qids: list[str] = []
    for row in rows:
        qid = str(row["qid"])
        action = actions.get(qid)
        seed_correct = bool(row.get("seed_correct"))
        total = _seed_tokens(row)
        if action is None:
            final_correct = seed_correct
            counts["accept"] += 1
        else:
            triggered += 1
            counts[f"think{action}"] += 1
            total += _arm_tokens(row, action)
            final_correct = _arm_final_correct(row, action)
            if _arm_gate_pass(row, action):
                adopted += 1
            if (not seed_correct) and final_correct:
                helpful += 1
                helpful_qids.append(qid)
            if seed_correct and (not final_correct):
                harmful += 1
                harmful_qids.append(qid)
            if (not seed_correct) and (not final_correct) and _arm_gate_pass(row, action):
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
        "accept": counts["accept"],
        "think8": counts["think8"],
        "think12": counts["think12"],
        "helpful_qids": ",".join(helpful_qids),
        "harmful_qids": ",".join(harmful_qids),
    }


def _accept_actions(rows: list[dict[str, Any]]) -> dict[str, int | None]:
    return {str(row["qid"]): None for row in rows}


def _all_arm_actions(rows: list[dict[str, Any]], arm: int) -> dict[str, int | None]:
    return {str(row["qid"]): arm for row in rows}


def _budget_total(rows: list[dict[str, Any]], avg_budget: float) -> float:
    return avg_budget * max(1, len(rows))


def _fits(total_tokens: float, extra: float, budget_total: float) -> bool:
    return total_tokens + extra <= budget_total + 1e-9


def _oracle_knapsack_actions(rows: list[dict[str, Any]], avg_budget: float, arms: list[int] = ARMS) -> dict[str, int | None]:
    """Exact accuracy oracle for binary-gain arms.

    Under strict-gated rows, useful actions only add one correct answer and do
    not harm correct seeds. The exact max-accuracy solution is therefore to
    choose the cheapest helpful arm per qid, then take as many as fit.
    """
    actions = _accept_actions(rows)
    total_tokens = sum(_seed_tokens(row) for row in rows)
    budget_total = _budget_total(rows, avg_budget)
    candidates: list[tuple[float, str, int]] = []
    for row in rows:
        if bool(row.get("seed_correct")):
            continue
        qid = str(row["qid"])
        helpful: list[tuple[float, int]] = []
        for arm in arms:
            if _arm_final_correct(row, arm):
                helpful.append((_arm_tokens(row, arm), arm))
        if helpful:
            cost, arm = min(helpful)
            candidates.append((cost, qid, arm))
    candidates.sort()
    for cost, qid, arm in candidates:
        if _fits(total_tokens, cost, budget_total):
            actions[qid] = arm
            total_tokens += cost
    return actions


def _oracle_lagrangian_actions(rows: list[dict[str, Any]], lam: float, arms: list[int] = ARMS) -> dict[str, int | None]:
    actions = _accept_actions(rows)
    for row in rows:
        qid = str(row["qid"])
        seed_correct = int(bool(row.get("seed_correct")))
        best_score = 0.0
        best_arm: int | None = None
        for arm in arms:
            gain = int(_arm_final_correct(row, arm)) - seed_correct
            score = gain - lam * _arm_tokens(row, arm)
            if score > best_score:
                best_score = score
                best_arm = arm
        actions[qid] = best_arm
    return actions


def _best_lagrangian_under_budget(rows: list[dict[str, Any]], avg_budget: float) -> dict[str, int | None]:
    best_actions = _accept_actions(rows)
    best_correct = _simulate(rows, best_actions)["correct"]
    best_tokens = _simulate(rows, best_actions)["avg_tokens"]
    lambdas = [0.0]
    lambdas.extend([10 ** exp for exp in [-6, -5.5, -5, -4.5, -4, -3.5, -3, -2.5, -2]])
    for lam in lambdas:
        actions = _oracle_lagrangian_actions(rows, lam)
        sim = _simulate(rows, actions)
        if sim["avg_tokens"] <= avg_budget + 1e-9:
            if sim["correct"] > best_correct or (sim["correct"] == best_correct and sim["avg_tokens"] < best_tokens):
                best_actions = actions
                best_correct = int(sim["correct"])
                best_tokens = float(sim["avg_tokens"])
    return best_actions


def _score_budget_actions(
    rows: list[dict[str, Any]],
    avg_budget: float,
    *,
    score_kind: str,
    arm: int,
    rng: random.Random | None = None,
) -> dict[str, int | None]:
    actions = _accept_actions(rows)
    total_tokens = sum(_seed_tokens(row) for row in rows)
    budget_total = _budget_total(rows, avg_budget)
    if score_kind == "trace":
        ordered = sorted(rows, key=_trace_score, reverse=True)
    elif score_kind == "heuristic_trace":
        ordered = sorted(
            [row for row in rows if bool(row.get("old_trigger_nonbranch_or_long300"))],
            key=_trace_score,
            reverse=True,
        )
    elif score_kind == "random":
        ordered = list(rows)
        assert rng is not None
        rng.shuffle(ordered)
    else:
        raise KeyError(score_kind)
    for row in ordered:
        qid = str(row["qid"])
        extra = _arm_tokens(row, arm)
        if extra <= 0:
            continue
        if _fits(total_tokens, extra, budget_total):
            actions[qid] = arm
            total_tokens += extra
    return actions


def _top_rate_actions(rows: list[dict[str, Any]], *, score_kind: str, arm: int, rate: float, rng: random.Random | None = None) -> dict[str, int | None]:
    actions = _accept_actions(rows)
    k = max(0, min(len(rows), int(round(rate * len(rows)))))
    if score_kind == "trace":
        ordered = sorted(rows, key=_trace_score, reverse=True)
    elif score_kind == "random":
        ordered = list(rows)
        assert rng is not None
        rng.shuffle(ordered)
    elif score_kind == "heuristic_trace":
        ordered = sorted(
            [row for row in rows if bool(row.get("old_trigger_nonbranch_or_long300"))],
            key=_trace_score,
            reverse=True,
        )
    else:
        raise KeyError(score_kind)
    for row in ordered[:k]:
        actions[str(row["qid"])] = arm
    return actions


def _rows_for_budget_methods(rows: list[dict[str, Any]], budgets: list[float], random_repeats: int) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for budget in budgets:
        method_actions = {
            "oracle_knapsack_8_12": _oracle_knapsack_actions(rows, budget),
            "oracle_lagrangian_8_12": _best_lagrangian_under_budget(rows, budget),
            "trace12_budget": _score_budget_actions(rows, budget, score_kind="trace", arm=12),
            "trace8_budget": _score_budget_actions(rows, budget, score_kind="trace", arm=8),
            "heuristic12_budget": _score_budget_actions(rows, budget, score_kind="heuristic_trace", arm=12),
        }
        for method, actions in method_actions.items():
            out.append({"method": method, "budget": budget, "budget_name": f"{budget:.1f}", **_simulate(rows, actions)})

        for arm in ARMS:
            sims = []
            for rep in range(random_repeats):
                actions = _score_budget_actions(
                    rows,
                    budget,
                    score_kind="random",
                    arm=arm,
                    rng=random.Random(1009 + rep + 17 * arm + int(budget * 10)),
                )
                sims.append(_simulate(rows, actions))
            out.append(_average_sims(sims, method=f"random{arm}_budget", budget=budget, budget_name=f"{budget:.1f}"))
    return out


def _average_sims(sims: list[dict[str, Any]], *, method: str, budget: float, budget_name: str | None = None) -> dict[str, Any]:
    keys_float = ["correct", "accuracy", "avg_tokens", "trigger_count", "trigger_rate", "adoption_count", "adoption_rate", "helpful", "harmful", "wrong_to_wrong", "accept", "think8", "think12"]
    row: dict[str, Any] = {"method": method, "budget": budget, "budget_name": budget_name or f"{budget:.1f}", "count": sims[0]["count"] if sims else 0}
    for key in keys_float:
        row[key] = _mean([float(sim.get(key) or 0.0) for sim in sims])
    return row


def _reference_rows(rows: list[dict[str, Any]], random_repeats: int) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for arm in ARMS:
        correct = sum(int(bool(row.get(f"think{arm}_raw_correct"))) for row in rows)
        seed_correct = sum(int(bool(row.get("seed_correct"))) for row in rows)
        n = max(1, len(rows))
        out.append(
            {
                "method": f"standalone_think{arm}_raw_reference",
                "budget": _mean([_arm_tokens(row, arm) for row in rows]),
                "budget_name": "reference",
                "count": len(rows),
                "correct": correct,
                "accuracy": correct / n,
                "avg_tokens": _mean([_arm_tokens(row, arm) for row in rows]),
                "trigger_count": len(rows),
                "trigger_rate": 1.0,
                "adoption_count": len(rows),
                "adoption_rate": 1.0,
                "helpful": sum(int((not bool(row.get("seed_correct"))) and bool(row.get(f"think{arm}_raw_correct"))) for row in rows),
                "harmful": sum(int(bool(row.get("seed_correct")) and (not bool(row.get(f"think{arm}_raw_correct")))) for row in rows),
                "wrong_to_wrong": sum(int((not bool(row.get("seed_correct"))) and (not bool(row.get(f"think{arm}_raw_correct")))) for row in rows),
                "accept": 0,
                "think8": len(rows) if arm == 8 else 0,
                "think12": len(rows) if arm == 12 else 0,
                "seed_correct_reference": seed_correct,
            }
        )
    for method, actions in {
        "accept_only": _accept_actions(rows),
        "all_trigger_think8_strict_gate": _all_arm_actions(rows, 8),
        "all_trigger_think12_strict_gate": _all_arm_actions(rows, 12),
    }.items():
        sim = _simulate(rows, actions)
        out.append({"method": method, "budget": sim["avg_tokens"], "budget_name": "reference", **sim})
    for rate in (0.3, 0.5):
        for score_kind, name in [("trace", "trace12_top"), ("heuristic_trace", "heuristic12_top")]:
            actions = _top_rate_actions(rows, score_kind=score_kind, arm=12, rate=rate)
            sim = _simulate(rows, actions)
            out.append({"method": f"{name}{int(rate*100)}", "budget": sim["avg_tokens"], "budget_name": f"top{int(rate*100)}", **sim})
        sims = []
        for rep in range(random_repeats):
            actions = _top_rate_actions(rows, score_kind="random", arm=12, rate=rate, rng=random.Random(2027 + rep + int(rate * 1000)))
            sims.append(_simulate(rows, actions))
        out.append(_average_sims(sims, method=f"random12_top{int(rate*100)}", budget=_mean([sim["avg_tokens"] for sim in sims]), budget_name=f"top{int(rate*100)}"))
    return out


def _summarize_mean(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        budget_name = str(row.get("budget_name") or f"{float(row['budget']):.1f}")
        groups.setdefault((str(row["method"]), budget_name), []).append(row)
    out: list[dict[str, Any]] = []
    for (method, budget_name), items in sorted(groups.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        out.append(
            {
                "method": method,
                "budget_name": budget_name,
                "budget": _mean([float(row["budget"]) for row in items]),
                "seeds": len(items),
                "mean_acc": _mean([float(row["accuracy"]) for row in items]),
                "std_acc": _std([float(row["accuracy"]) for row in items]),
                "avg_tokens": _mean([float(row["avg_tokens"]) for row in items]),
                "trigger_rate": _mean([float(row["trigger_rate"]) for row in items]),
                "adoption_rate": _mean([float(row["adoption_rate"]) for row in items]),
                "helpful": _mean([float(row["helpful"]) for row in items]),
                "harmful": _mean([float(row["harmful"]) for row in items]),
                "wrong_to_wrong": _mean([float(row["wrong_to_wrong"]) for row in items]),
                "think8": _mean([float(row["think8"]) for row in items]),
                "think12": _mean([float(row["think12"]) for row in items]),
            }
        )
    return out


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


def _format_pct(value: float) -> str:
    return f"{100 * value:.2f}%"


def _write_summary(path: Path, mean_rows: list[dict[str, Any]]) -> None:
    def find(method: str, budget: float | None = None) -> dict[str, Any] | None:
        candidates = [row for row in mean_rows if row["method"] == method]
        if budget is not None:
            candidates = sorted(candidates, key=lambda row: abs(float(row["budget"]) - budget))
        return candidates[0] if candidates else None

    lines = ["# E2 BORA-Budget Offline Batch Allocation", ""]
    lines.append("Action set: `ACCEPT / THINK@8k / THINK@12k`; correctness uses existing strict-gated arm labels.")
    lines.append("")
    lines.append("## Reference Points")
    lines.append("")
    lines.append("| Method | Accuracy | Tokens | Trigger | Harmful |")
    lines.append("| --- | ---: | ---: | ---: | ---: |")
    for method in ["accept_only", "standalone_think8_raw_reference", "standalone_think12_raw_reference", "all_trigger_think8_strict_gate", "all_trigger_think12_strict_gate"]:
        row = find(method)
        if row:
            lines.append(
                f"| {method} | {_format_pct(float(row['mean_acc']))} | {float(row['avg_tokens']):.1f} | "
                f"{_format_pct(float(row['trigger_rate']))} | {float(row['harmful']):.1f} |"
            )
    lines.append("")
    lines.append("## Fixed-Budget Oracle/Heuristic Comparison")
    lines.append("")
    lines.append("| Budget | Oracle knapsack | Oracle tokens | Trace12 | Random12 | Heuristic12 |")
    lines.append("| ---: | ---: | ---: | ---: | ---: | ---: |")
    for budget in DEFAULT_BUDGETS:
        oracle = find("oracle_knapsack_8_12", budget)
        trace = find("trace12_budget", budget)
        rand = find("random12_budget", budget)
        heur = find("heuristic12_budget", budget)
        if oracle and trace and rand and heur:
            lines.append(
                f"| {budget:.1f} | {_format_pct(float(oracle['mean_acc']))} | {float(oracle['avg_tokens']):.1f} | "
                f"{_format_pct(float(trace['mean_acc']))} | {_format_pct(float(rand['mean_acc']))} | "
                f"{_format_pct(float(heur['mean_acc']))} |"
            )
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append("- `oracle_knapsack_8_12` is the upper bound for this action set under each token budget; it selects the cheapest helpful thinking arm per problem.")
    lines.append("- Trace/random/heuristic rows use only seed-state ordering and do not know which arm is actually helpful.")
    lines.append("- If the oracle is much better than trace/random at the same budget, remaining headroom is trigger/arm-selection quality rather than thinking capability.")
    path.write_text("\n".join(lines), encoding="utf-8")


def _load_external_frontier_refs(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    out: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            method = str(row.get("method") or "")
            if method not in {"GBDT top30 strict", "GBDT top50 strict"}:
                continue
            out.append(
                {
                    "method": "binary_" + method.lower().replace(" ", "_"),
                    "budget_name": "external_reference",
                    "budget": _as_float(row.get("avg_tokens")),
                    "seeds": 3,
                    "mean_acc": _as_float(row.get("mean_acc")) / 100.0,
                    "std_acc": _as_float(row.get("std_acc")) / 100.0,
                    "avg_tokens": _as_float(row.get("avg_tokens")),
                    "trigger_rate": _as_float(row.get("trigger_rate")),
                    "adoption_rate": "",
                    "helpful": _as_float(row.get("helpful")),
                    "harmful": _as_float(row.get("harmful")),
                    "wrong_to_wrong": _as_float(row.get("wrong_to_wrong")),
                    "think8": 0.0,
                    "think12": _as_float(row.get("trigger_rate")) * 500.0,
                    "source": str(path),
                }
            )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="E2: offline BORA-Budget batch allocation over ACCEPT/THINK@8k/THINK@12k.")
    parser.add_argument(
        "--arm-rows",
        type=Path,
        default=Path("artifacts/remote_stage_main/independent_checkers_20260517/karmed_arm_rows_with_checkers.jsonl"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/e2_bora_budget"))
    parser.add_argument("--budgets", type=str, default=",".join(str(x) for x in DEFAULT_BUDGETS))
    parser.add_argument("--random-repeats", type=int, default=100)
    parser.add_argument("--paper-frontier-csv", type=Path, default=Path("artifacts/paper_tables/math500_trigger_frontier.csv"))
    args = parser.parse_args()

    budgets = [float(part) for part in args.budgets.split(",") if part.strip()]
    rows = _read_jsonl(args.arm_rows)
    by_seed = _rows_by_seed(rows)
    by_seed_rows: list[dict[str, Any]] = []
    lambda_rows: list[dict[str, Any]] = []
    for seed, seed_rows in by_seed.items():
        for row in _reference_rows(seed_rows, args.random_repeats):
            by_seed_rows.append({"seed": seed, **row})
        for row in _rows_for_budget_methods(seed_rows, budgets, args.random_repeats):
            by_seed_rows.append({"seed": seed, **row})
        for lam in [0.0, 1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3]:
            sim = _simulate(seed_rows, _oracle_lagrangian_actions(seed_rows, lam))
            lambda_rows.append({"seed": seed, "lambda": lam, **sim})

    mean_rows = _summarize_mean(by_seed_rows)
    external_refs = _load_external_frontier_refs(args.paper_frontier_csv)
    mean_rows_with_refs = mean_rows + external_refs
    mean_lambda = _summarize_mean(
        [{**row, "method": "oracle_lagrangian_lambda", "budget": float(row["lambda"])} for row in lambda_rows]
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output_dir / "math500_e2_bora_budget_by_seed.csv", by_seed_rows)
    _write_csv(args.output_dir / "math500_e2_bora_budget_mean.csv", mean_rows_with_refs)
    _write_csv(args.output_dir / "math500_e2_external_references.csv", external_refs)
    _write_csv(args.output_dir / "math500_e2_lagrangian_lambda_by_seed.csv", lambda_rows)
    _write_csv(args.output_dir / "math500_e2_lagrangian_lambda_mean.csv", mean_lambda)
    _write_summary(args.output_dir / "E2_BORA_BUDGET_SUMMARY.md", mean_rows_with_refs)
    (args.output_dir / "config.json").write_text(
        json.dumps(
            {
                "arm_rows": str(args.arm_rows),
                "budgets": budgets,
                "random_repeats": args.random_repeats,
                "seeds": SEEDS,
                "arms": ARMS,
                "correctness": "existing strict-gated arm labels from karmed rows",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(args.output_dir / "E2_BORA_BUDGET_SUMMARY.md")
    for method in ["accept_only", "standalone_think8_raw_reference", "standalone_think12_raw_reference", "all_trigger_think8_strict_gate", "all_trigger_think12_strict_gate"]:
        row = next((item for item in mean_rows if item["method"] == method), None)
        if row:
            print(method, f"acc={100*float(row['mean_acc']):.2f}", f"tok={float(row['avg_tokens']):.1f}")
    for budget in [2681.7, 3589.6, 3867.8]:
        row = min(
            [item for item in mean_rows if item["method"] == "oracle_knapsack_8_12"],
            key=lambda item: abs(float(item["budget"]) - budget),
        )
        print("oracle_knapsack_8_12", f"budget={budget:.1f}", f"acc={100*float(row['mean_acc']):.2f}", f"tok={float(row['avg_tokens']):.1f}", f"8k={float(row['think8']):.1f}", f"12k={float(row['think12']):.1f}")


if __name__ == "__main__":
    main()
