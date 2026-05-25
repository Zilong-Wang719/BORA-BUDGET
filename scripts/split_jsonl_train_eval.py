from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def _bucket(qid: str, folds: int) -> int:
    digest = hashlib.md5(qid.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % folds


def main() -> None:
    parser = argparse.ArgumentParser(description="Split JSONL rows into deterministic train/eval folds.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--train-output", required=True)
    parser.add_argument("--eval-output", required=True)
    parser.add_argument("--folds", type=int, default=2)
    parser.add_argument("--eval-fold", type=int, default=0)
    args = parser.parse_args()

    rows = [json.loads(line) for line in Path(args.input).read_text(encoding="utf-8").splitlines() if line.strip()]
    train = []
    eval_rows = []
    for row in rows:
        bucket = _bucket(str(row.get("qid")), int(args.folds))
        if bucket == int(args.eval_fold):
            eval_rows.append(row)
        else:
            train.append(row)

    for path, subset in ((args.train_output, train), (args.eval_output, eval_rows)):
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8") as handle:
            for row in subset:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"train={len(train)} eval={len(eval_rows)}")


if __name__ == "__main__":
    main()
