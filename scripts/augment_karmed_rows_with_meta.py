from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
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
    raise ValueError("Could not infer records block.")


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _meta_map(path: Path) -> dict[str, dict[str, float]]:
    return {
        str(row["qid"]): dict(row.get("meta_features") or {})
        for row in _records(_load_json(path))
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm-rows", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--meta",
        action="append",
        default=[],
        help="Mapping in the form seed:path/to/meta.json. Repeat once per seed.",
    )
    args = parser.parse_args()

    meta_by_seed: dict[int, dict[str, dict[str, float]]] = {}
    for item in args.meta:
        seed_text, path_text = item.split(":", 1)
        meta_by_seed[int(seed_text)] = _meta_map(Path(path_text))

    rows = _load_jsonl(args.arm_rows)
    updated = 0
    feature_hits: dict[str, int] = {}
    for row in rows:
        seed = int(row["seed"])
        qid = str(row["qid"])
        meta = meta_by_seed.get(seed, {}).get(qid, {})
        if not meta:
            continue
        features = dict(row.get("features") or {})
        for key, value in meta.items():
            try:
                features[key] = float(value)
            except (TypeError, ValueError):
                continue
            feature_hits[key] = feature_hits.get(key, 0) + 1
        row["features"] = features
        row["meta_probe_available"] = True
        updated += 1
    _write_jsonl(args.output, rows)
    print(
        f"wrote {args.output} updated={updated}/{len(rows)} "
        f"features={json.dumps(feature_hits, sort_keys=True)}"
    )


if __name__ == "__main__":
    main()
