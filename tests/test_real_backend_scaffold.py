from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bora.answer_extraction import extract_explicit_answer, extract_tagged_sections, parse_score
from bora.llm import get_llm_backend
from bora.solver import TransformersMathSolver
from bora.verifier import TransformersVerifier, _extract_verifier_candidate_answer, _parse_verifier_score


class RealBackendScaffoldTests(unittest.TestCase):
    def test_answer_extraction_and_tag_parsing_are_robust(self) -> None:
        raw = """
[STEP]
Compute the remaining quantity.

[CURRENT_ANSWER]
Final answer: 42

[CONFIDENCE]
0.83

[DONE]
true
"""
        sections = extract_tagged_sections(raw)
        self.assertEqual(sections["STEP"], "Compute the remaining quantity.")
        self.assertEqual(extract_explicit_answer(sections["CURRENT_ANSWER"]), "42")
        self.assertAlmostEqual(parse_score(sections["CONFIDENCE"]), 0.83)
        self.assertIsNone(parse_score(""))

    def test_whole_hour_time_answer_uses_hour_value(self) -> None:
        self.assertEqual(extract_explicit_answer(r"The answer is \boxed{10:00 am}."), "10")
        self.assertEqual(extract_explicit_answer("Final answer: 10:00 am"), "10")

    def test_hf_backends_build_without_eager_model_load(self) -> None:
        config = {
            "llm": {
                "model_name": "/tmp/fake-model",
                "device": "cpu",
                "prompt_template": "auto",
                "local_files_only": True,
                "trust_remote_code": True,
            },
            "solver": {
                "backend": "hf_transformers",
            },
            "verifier": {
                "backend": "hf_transformers",
            },
        }
        self.assertIsInstance(TransformersMathSolver(config), TransformersMathSolver)
        self.assertIsInstance(TransformersVerifier(config), TransformersVerifier)

    def test_verifier_score_fallback_does_not_parse_problem_numbers(self) -> None:
        raw = "The calculation uses 48 candy bars and 50 costs, so the answer looks plausible."
        self.assertIsNone(_parse_verifier_score({}, raw))
        self.assertAlmostEqual(_parse_verifier_score({}, "Score: 0.25"), 0.25)

    def test_verifier_candidate_fallback_extracts_prose_answer(self) -> None:
        raw = "The current answer is incorrect; the correct latest start time is 10:00 am, not 6"
        self.assertEqual(_extract_verifier_candidate_answer({}, raw), "10")

    def test_vllm_backends_build_without_eager_model_load(self) -> None:
        config = {
            "llm": {
                "backend": "vllm",
                "model_name": "/tmp/fake-model",
                "cuda_visible_devices": "0",
                "local_files_only": True,
                "trust_remote_code": True,
            },
            "solver": {
                "backend": "vllm",
            },
            "verifier": {
                "backend": "vllm",
            },
        }
        self.assertIsInstance(TransformersMathSolver(config), TransformersMathSolver)
        self.assertIsInstance(TransformersVerifier(config), TransformersVerifier)
        self.assertEqual(get_llm_backend(config, "solver").backend_name, "vllm")


if __name__ == "__main__":
    unittest.main()
