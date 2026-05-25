from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts.build_rescue_bandit_prior import build_gate_rows


class RescuePriorTests(unittest.TestCase):
    def test_gate_rows_group_by_state_id_not_qid(self) -> None:
        rows = [
            {
                "qid": "same_qid",
                "state_id": "same_qid:seed1024",
                "features": [1.0, 0.0],
                "action": "ACCEPT_SEED",
                "marginal_reward": 0.0,
            },
            {
                "qid": "same_qid",
                "state_id": "same_qid:seed1024",
                "features": [1.0, 0.0],
                "action": "VERIFY_ADOPT",
                "marginal_reward": -0.1,
            },
            {
                "qid": "same_qid",
                "state_id": "same_qid:seed192",
                "features": [0.0, 1.0],
                "action": "ACCEPT_SEED",
                "marginal_reward": 0.0,
            },
            {
                "qid": "same_qid",
                "state_id": "same_qid:seed192",
                "features": [0.0, 1.0],
                "action": "VERIFY_ADOPT",
                "marginal_reward": 0.8,
            },
        ]

        gate_rows = build_gate_rows(rows)

        self.assertEqual(len(gate_rows), 4)
        rescue_rows = [row for row in gate_rows if row[1] == "RESCUE"]
        self.assertEqual([row[0] for row in rescue_rows], [[1.0, 0.0], [0.0, 1.0]])
        self.assertEqual([row[2] for row in rescue_rows], [-0.1, 0.8])


if __name__ == "__main__":
    unittest.main()
