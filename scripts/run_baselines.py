from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bora.baselines import BASELINE_REGISTRY
from bora.common import dump_json, load_config, load_problem_split
from bora.eval import summarize_records


def parse_methods(raw: str | None) -> list[str]:
    if raw is None or not raw.strip():
        return list(BASELINE_REGISTRY)
    return [item.strip() for item in raw.split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="conf/math_mvp.yaml")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--split", choices=["train", "dev", "test"], default="test")
    parser.add_argument("--methods", default=None, help="Comma-separated baseline names.")
    parser.add_argument("--output", default=None)
    parser.add_argument("--partial-output", default=None)
    parser.add_argument("--progress-every", type=int, default=16)
    parser.add_argument("--random-seed", type=int, default=None)
    args = parser.parse_args()

    config = {**load_config(args.config), "mode": "eval"}
    if args.random_seed is not None:
        config["random_seed"] = args.random_seed
    dataset = load_problem_split(config, args.split)
    if args.limit is not None:
        dataset = dataset[: args.limit]

    partial_path = Path(args.partial_output) if args.partial_output else None
    if partial_path is not None:
        partial_path.parent.mkdir(parents=True, exist_ok=True)
        partial_path.write_text("", encoding="utf-8")

    payload: dict[str, dict[str, object]] = {}
    for name in parse_methods(args.methods):
        if name not in BASELINE_REGISTRY:
            raise KeyError(f"Unknown baseline method: {name}")
        runner = BASELINE_REGISTRY[name]
        records = []
        for offset, problem in enumerate(dataset):
            episode_config = {
                **config,
                "episode_seed": int(config.get("random_seed", 0)) + offset,
            }
            record = asdict(runner(episode_config, problem))
            records.append(record)
            if partial_path is not None:
                with partial_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps({"method": name, **record}) + "\n")
            if args.progress_every > 0 and (offset + 1) % args.progress_every == 0:
                print(
                    f"[{offset + 1}/{len(dataset)}] method={name} "
                    f"id={record['qid']} correct={record['correct']} "
                    f"tokens={record['total_tokens']}",
                    flush=True,
                )
        payload[name] = {
            "summary": summarize_records(records),
            "records": records,
        }

    output_path = args.output or config["baseline_result_path"]
    dump_json(output_path, payload)
    print(f"wrote baseline results to {output_path}")


if __name__ == "__main__":
    main()
