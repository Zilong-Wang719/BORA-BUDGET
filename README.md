# BORA-Budget Anonymous Code Release

This folder is an anonymized code package for the paper submission. It contains
the runnable controller, evaluation utilities, selected analysis scripts, paper
source, and small sample data. It intentionally excludes model checkpoints,
private server launch scripts, large generated rollouts, virtual environments,
logs, and user-specific paths.

## Contents

- `src/bora/`: core BORA/BORA-Budget controller utilities.
- `scripts/`: curated scripts for generation, answer evaluation, opportunity
  rollout construction, trigger analysis, budget allocation, transfer-task
  preprocessing, and K-armed diagnostics.
- `conf/`: anonymized example configs.
- `data/`: tiny sample JSONL files for smoke tests only.
- `tests/`: unit tests for answer extraction, budget accounting, leakage, and
  controller scaffolding.
- `artifacts/paper_tables/`: small table artifacts used for paper-facing
  summaries. Full model outputs are not included.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[eval,dev]"
```

For GPU generation with vLLM:

```bash
pip install -e ".[vllm,eval,dev]"
```

## Smoke Test

The repository includes a mock backend config that does not require a model
checkpoint:

```bash
PYTHONPATH=src python scripts/run_baselines.py --config conf/mock_smoke.yaml --split dev --limit 2
pytest tests
```

## Reproducing the Main Pipeline

Full reproduction requires downloading the chosen hybrid-thinking model and the
public benchmark data. Use local paths in `conf/example_math500_vllm.yaml`; no
private server paths are required.

Typical workflow:

```bash
# 1. Generate cheap seed outputs.
PYTHONPATH=src python scripts/run_batched_standard_cot_vllm.py \
  --config conf/example_math500_vllm.yaml \
  --split test \
  --output artifacts/no_think_seed17.json \
  --disable-thinking \
  --store-completion-text \
  --random-seed 17

# 2. Generate paid thinking outputs.
PYTHONPATH=src python scripts/run_batched_standard_cot_vllm.py \
  --config conf/example_math500_vllm.yaml \
  --split test \
  --output artifacts/think_seed17.json \
  --enable-thinking \
  --store-completion-text \
  --random-seed 17

# 3. Re-evaluate answers from full completions.
PYTHONPATH=src python scripts/evaluate_math_answers_robust.py \
  --input artifacts/no_think_seed17.json \
  --output artifacts/no_think_seed17.fixed.json

# 4. Build opportunity rows and replay learned/random/trace triggers.
PYTHONPATH=src python scripts/build_opportunity_rollouts.py --help
PYTHONPATH=src python scripts/analyze_trigger_frontier.py --help
PYTHONPATH=src python scripts/analyze_bora_budget_auto_threshold_strict.py --help
```

Some paper tables were generated from full rollouts that are too large for the
anonymous package. The included `artifacts/paper_tables/` files document the
small paper-facing summaries, while the scripts above describe the exact
pipeline used to regenerate them from public data and model outputs.

## Anonymization Notes

This package removes:

- local usernames and home-directory paths;
- remote SSH hostnames/IP addresses;
- model checkpoint files;
- generated logs, `.aux`/`.log`/`.synctex` LaTeX build products;
- large raw rollouts and cached Python bytecode.

If you add new files before submission, rerun:

```bash
grep -RInE 'LOCAL_USER_PATTERN|LOCAL_HOST_PATTERN|LOCAL_PATH_PATTERN' .
```

from inside this folder.
