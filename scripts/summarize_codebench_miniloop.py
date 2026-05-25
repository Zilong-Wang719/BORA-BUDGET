from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _records(path: str | Path) -> list[dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return list(payload.get("standard_direct_cot", {}).get("records") or payload.get("records") or [])


def _acc_tokens(path: str | Path) -> dict[str, float]:
    records = _records(path)
    n = max(1, len(records))
    return {
        "count": len(records),
        "accuracy": sum(bool(r.get("correct")) for r in records) / n,
        "avg_tokens": sum(float(r.get("total_tokens") or 0) for r in records) / n,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize the MBPP codebench minimal loop.")
    parser.add_argument("--seed-results", required=True)
    parser.add_argument("--think-results", required=True)
    parser.add_argument("--frontier", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    seed = _acc_tokens(args.seed_results)
    think = _acc_tokens(args.think_results)
    frontier = json.loads(Path(args.frontier).read_text(encoding="utf-8"))
    rows = frontier.get("results") or []
    keep = []
    for row in rows:
        if row.get("adoption_filter") != "strict":
            continue
        if row.get("selection_mode") != "topk_eval":
            continue
        if row.get("method") not in {"gbdt", "trace_length", "random"}:
            continue
        if float(row.get("target_rate") or 0) not in {0.3, 0.5}:
            continue
        keep.append(row)
    keep.sort(key=lambda r: (float(r.get("target_rate") or 0), str(r.get("method"))))
    summary = {
        "seed": seed,
        "think": think,
        "train_summary": frontier.get("train_summary"),
        "eval_summary": frontier.get("eval_summary"),
        "frontier_rows": keep,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote codebench summary to {args.output}")
    print(f"no_think acc={seed['accuracy']*100:.2f}% tokens={seed['avg_tokens']:.1f}")
    print(f"think acc={think['accuracy']*100:.2f}% tokens={think['avg_tokens']:.1f}")
    for row in keep:
        print(
            f"{row['method']} rate={row['target_rate']} acc={row['accuracy']*100:.2f}% "
            f"tokens={row['avg_total_tokens']:.1f} trigger={row['trigger_rate']*100:.1f}% "
            f"helpful={row['helpful']} harmful={row['harmful']}"
        )


if __name__ == "__main__":
    main()
