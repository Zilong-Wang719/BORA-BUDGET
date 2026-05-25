from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bora.common import load_problem_split


class DataLoadingTests(unittest.TestCase):
    def test_load_problem_split_canonicalizes_remote_schema_and_slicing(self) -> None:
        rows = [
            {
                "problem_id": "gsm8k_train_0",
                "dataset": "gsm8k",
                "question": "Question 0",
                "answer": "1",
                "metadata": {"level": 1},
            },
            {
                "problem_id": "gsm8k_train_1",
                "dataset": "gsm8k",
                "question": "Question 1",
                "answer": "2",
                "metadata": {"level": 2},
            },
            {
                "problem_id": "gsm8k_train_2",
                "dataset": "gsm8k",
                "question": "Question 2",
                "answer": "3",
                "metadata": {"level": 3},
            },
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "gsm8k_train.jsonl"
            with path.open("w", encoding="utf-8") as handle:
                for row in rows:
                    handle.write(json.dumps(row) + "\n")

            config = {
                "data": {
                    "train_path": str(path),
                    "train_start": 1,
                    "train_limit": 1,
                }
            }
            problems = load_problem_split(config, "train")

        self.assertEqual(len(problems), 1)
        self.assertEqual(problems[0]["qid"], "gsm8k_train_1")
        self.assertEqual(problems[0]["question"], "Question 1")
        self.assertEqual(problems[0]["answer"], "2")
        self.assertEqual(problems[0]["difficulty"], "2")
        self.assertEqual(problems[0]["metadata"]["dataset"], "gsm8k")


if __name__ == "__main__":
    unittest.main()
