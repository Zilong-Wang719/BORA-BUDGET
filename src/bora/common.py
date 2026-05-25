from __future__ import annotations

import json
import math
import re
from collections import Counter
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Iterable

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def project_root() -> Path:
    return PROJECT_ROOT


def resolve_path(value: str | Path, base_dir: Path | None = None) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    base = base_dir or project_root()
    return (base / path).resolve()


def ensure_dir(path: str | Path) -> Path:
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    return target


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = resolve_path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle) or {}

    output_dir = resolve_path(cfg.get("output_dir", "artifacts"))
    data_cfg = dict(cfg.get("data", {}))
    legacy_data_path = cfg.get("data_path", "data/sample_math.jsonl")
    train_data_path = resolve_path(data_cfg.get("train_path", legacy_data_path))
    dev_data_path = resolve_path(data_cfg.get("dev_path", legacy_data_path))
    test_data_path = resolve_path(data_cfg.get("test_path", legacy_data_path))

    train_dir = output_dir / "train"
    eval_dir = output_dir / "eval"
    result_dir = eval_dir / "results"
    cfg["config_path"] = str(config_path)
    cfg["project_root"] = str(project_root())
    cfg["mode"] = str(cfg.get("mode", "eval"))
    cfg["output_dir"] = str(output_dir)
    cfg["train_dir"] = str(train_dir)
    cfg["eval_dir"] = str(eval_dir)
    cfg["result_dir"] = str(result_dir)
    cfg["train_data_path"] = str(train_data_path)
    cfg["dev_data_path"] = str(dev_data_path)
    cfg["test_data_path"] = str(test_data_path)
    cfg["data"] = {
        **data_cfg,
        "train_path": str(train_data_path),
        "dev_path": str(dev_data_path),
        "test_path": str(test_data_path),
    }
    cfg["train_rollout_path"] = str(train_dir / "rollouts_train.jsonl")
    cfg["dev_rollout_path"] = str(train_dir / "rollouts_dev.jsonl")
    cfg["rollout_path"] = cfg["train_rollout_path"]
    cfg["proxy_path"] = str(train_dir / "proxy.joblib")
    cfg["proxy_metrics_path"] = str(train_dir / "proxy_metrics.json")
    cfg["prior_path"] = str(train_dir / "bandit_prior.joblib")
    cfg["bora_result_path"] = str(result_dir / "bora.json")
    cfg["baseline_result_path"] = str(result_dir / "baselines.json")
    ensure_dir(output_dir)
    ensure_dir(train_dir)
    ensure_dir(eval_dir)
    ensure_dir(result_dir)
    return cfg


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def dump_json(path: str | Path, payload: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def dump_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def safe_float(value: str | int | float | None) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def normalize_answer(answer: str | None) -> str | None:
    if answer is None:
        return None
    cleaned = answer.strip()
    if not cleaned:
        return None
    cleaned = cleaned.replace(",", "")
    if "=" in cleaned:
        cleaned = cleaned.split("=")[-1].strip()
    lowered = cleaned.lower()
    if lowered in {"unknown", "n/a", "none"}:
        return None
    whole_hour_time = re.search(r"\b(-?\d{1,2}):00\b", cleaned)
    if whole_hour_time:
        return whole_hour_time.group(1)
    try:
        value = Decimal(cleaned)
        normalized = value.normalize()
        if normalized == normalized.to_integral():
            return str(int(normalized))
        return format(normalized, "f").rstrip("0").rstrip(".")
    except (InvalidOperation, ValueError):
        numeric_matches = re.findall(r"-?\d+(?:\.\d+)?", cleaned)
        if len(numeric_matches) == 1:
            try:
                value = Decimal(numeric_matches[0])
                normalized = value.normalize()
                if normalized == normalized.to_integral():
                    return str(int(normalized))
                return format(normalized, "f").rstrip("0").rstrip(".")
            except (InvalidOperation, ValueError):
                pass
        collapsed = " ".join(lowered.split())
        return collapsed or None


def is_correct(prediction: str | None, gold: str | None) -> bool:
    pred_norm = normalize_answer(prediction)
    gold_norm = normalize_answer(gold)
    return pred_norm is not None and pred_norm == gold_norm


def entropy_from_counts(counter: Counter[str]) -> float:
    total = sum(counter.values())
    if total <= 0:
        return 0.0
    entropy = 0.0
    for count in counter.values():
        if count <= 0:
            continue
        prob = count / total
        entropy -= prob * math.log(prob + 1e-12)
    return entropy


def weighted_choice(items: list[str], weights: list[float], rng: Any) -> str:
    total = sum(weights)
    if total <= 0:
        return items[0]
    probs = [weight / total for weight in weights]
    return str(rng.choice(items, p=probs))


def canonicalize_problem(
    row: dict[str, Any],
    *,
    split: str,
    index: int,
) -> dict[str, Any]:
    qid = row.get("qid") or row.get("problem_id") or row.get("id") or f"{split}_{index}"
    question = row.get("question") or row.get("prompt") or row.get("problem")
    if question is None:
        raise ValueError(f"Problem row {qid!r} is missing a question field.")
    metadata = dict(row.get("metadata") or {})
    if "dataset" in row and "dataset" not in metadata:
        metadata["dataset"] = row["dataset"]
    if "source" in row and "source" not in metadata:
        metadata["source"] = row["source"]
    difficulty = (
        row.get("difficulty")
        or metadata.get("difficulty")
        or metadata.get("level")
        or "unknown"
    )
    return {
        "qid": str(qid),
        "question": str(question),
        "answer": row.get("answer"),
        "difficulty": str(difficulty),
        "metadata": metadata,
    }


def _slice_rows(
    rows: list[dict[str, Any]],
    *,
    start: int = 0,
    limit: int | None = None,
    end: int | None = None,
    stride: int = 1,
) -> list[dict[str, Any]]:
    stop = end if end is not None else None
    sliced = rows[start:stop:stride]
    if limit is not None:
        sliced = sliced[:limit]
    return sliced


def load_problem_split(config: dict[str, Any], split: str) -> list[dict[str, Any]]:
    data_cfg = dict(config.get("data", {}))
    path_key = f"{split}_path"
    if path_key not in data_cfg:
        raise KeyError(f"Missing data.{path_key} in config.")
    raw_rows = load_jsonl(data_cfg[path_key])
    rows = _slice_rows(
        raw_rows,
        start=int(data_cfg.get(f"{split}_start", 0)),
        limit=(
            int(data_cfg[f"{split}_limit"])
            if data_cfg.get(f"{split}_limit") is not None
            else None
        ),
        end=(
            int(data_cfg[f"{split}_end"])
            if data_cfg.get(f"{split}_end") is not None
            else None
        ),
        stride=max(int(data_cfg.get(f"{split}_stride", 1)), 1),
    )
    return [
        canonicalize_problem(row, split=split, index=index)
        for index, row in enumerate(rows)
    ]
