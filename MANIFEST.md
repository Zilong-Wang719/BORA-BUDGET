# Anonymous Submission Manifest

## Included

| Path | Purpose |
| --- | --- |
| `src/bora/` | Core controller, answer extraction, runtime, solver, verifier, and baseline utilities. |
| `scripts/` | Curated reproduction scripts for generation, robust evaluation, opportunity rows, trigger frontiers, budget allocation, feature ablations, transfer tasks, and K-armed diagnostics. |
| `conf/mock_smoke.yaml` | Local mock-backend smoke-test config. |
| `conf/example_math500_vllm.yaml` | Placeholder vLLM config with public/local paths. |
| `data/` | Tiny sample JSONL files for tests and smoke runs. |
| `tests/` | Unit tests for core package behavior. |
| `paper/` | Anonymous ACL/EMNLP LaTeX source only. |
| `figures/` | Paper figure/table LaTeX snippets and SVG source. |
| `artifacts/paper_tables/` | Small paper-facing summary artifacts. |

## Excluded

- `Qwen3-4B/` and all model checkpoint files.
- `.venv*`, `__pycache__`, `.DS_Store`, LaTeX build products, and logs.
- Remote launch/watch shell scripts containing private cluster paths.
- Full rollout artifacts and intermediate audit traces.
- Review logs and paper backups.

## Expected Reviewer Entry Points

1. Install package: `pip install -e ".[eval,dev]"`.
2. Run unit tests: `pytest tests`.
3. Run mock smoke baseline: `PYTHONPATH=src python scripts/run_baselines.py --config conf/mock_smoke.yaml --split dev --limit 2`.
4. Inspect generation/evaluation pipeline through `scripts/run_batched_standard_cot_vllm.py`, `scripts/evaluate_math_answers_robust.py`, `scripts/build_opportunity_rollouts.py`, and `scripts/analyze_bora_budget_auto_threshold_strict.py`.
