from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bora.common import is_correct, normalize_answer


class AnswerNormalizationTests(unittest.TestCase):
    def test_numeric_answer_with_units_matches_plain_number(self) -> None:
        self.assertEqual(normalize_answer("800 cents"), "800")
        self.assertTrue(is_correct("800 cents", "800"))

    def test_currency_answer_matches_plain_number(self) -> None:
        self.assertEqual(normalize_answer("$400"), "400")
        self.assertTrue(is_correct("$400", "400"))

    def test_multiple_numbers_are_not_collapsed_to_last_number(self) -> None:
        self.assertFalse(is_correct("22 + 14 + 11 + 7", "54"))

    def test_whole_hour_time_matches_hour_answer(self) -> None:
        self.assertEqual(normalize_answer("10:00 am"), "10")
        self.assertTrue(is_correct(r"\boxed{10:00 \text{ am}}", "10"))


if __name__ == "__main__":
    unittest.main()
