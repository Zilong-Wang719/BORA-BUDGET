from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import joblib
import numpy as np
from sklearn.linear_model import Ridge

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bora.bandit import make_default_prior
from bora.common import load_config, load_jsonl
from bora.rescue_bandit import ACCEPT_SEED, GATE_ACCEPT, GATE_RESCUE


def fit_linear_prior(
    rows: list[tuple[list[float], str, float]],
    *,
    arms: list[str],
    dim: int,
    lam: float,
) -> dict[str, dict[str, np.ndarray]]:
    prior = make_default_prior(arms, dim, lam)
    for arm in arms:
        arm_rows = [(features, reward) for features, row_arm, reward in rows if row_arm == arm]
        if not arm_rows:
            continue
        matrix = np.asarray([features for features, _ in arm_rows], dtype=float)
        target = np.asarray([reward for _, reward in arm_rows], dtype=float)
        ridge = Ridge(alpha=lam, fit_intercept=False)
        ridge.fit(matrix, target)
        precision = lam * np.eye(dim) + matrix.T @ matrix
        mean = ridge.coef_
        prior[arm] = {"A": precision, "b": precision @ mean}
    return prior


def build_gate_rows(rows: list[dict[str, Any]]) -> list[tuple[list[float], str, float]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("state_id") or row["qid"])].append(row)

    gate_rows: list[tuple[list[float], str, float]] = []
    for qid_rows in grouped.values():
        features = qid_rows[0]["features"]
        gate_rows.append((features, GATE_ACCEPT, 0.0))
        rescue_rewards = [
            float(row["marginal_reward"])
            for row in qid_rows
            if row["action"] != ACCEPT_SEED
        ]
        gate_rows.append((features, GATE_RESCUE, max(rescue_rewards) if rescue_rewards else 0.0))
    return gate_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="conf/gsm8k_qwen3_stagea_rescue_bandit.yaml")
    parser.add_argument("--rollout-path", default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    config = {**load_config(args.config), "mode": "train"}
    rollout_path = args.rollout_path or str(Path(config["train_dir"]) / "rescue_rollouts_train.jsonl")
    rows = load_jsonl(rollout_path)
    if not rows:
        raise RuntimeError(f"No rescue rollout rows found at {rollout_path}")

    dim = len(rows[0]["features"])
    lam = float(config.get("rescue_bandit", {}).get("lam", config.get("bandit", {}).get("lam", 1.0)))
    rescue_actions = [
        action
        for action in config.get("rescue_bandit", {}).get("actions", [])
        if action != ACCEPT_SEED
    ]
    if not rescue_actions:
        rescue_actions = sorted({row["action"] for row in rows if row["action"] != ACCEPT_SEED})

    rescue_rows = [
        (row["features"], row["action"], float(row["marginal_reward"]))
        for row in rows
        if row["action"] in rescue_actions
    ]
    rescue_prior = fit_linear_prior(rescue_rows, arms=rescue_actions, dim=dim, lam=lam)
    gate_prior = fit_linear_prior(
        build_gate_rows(rows),
        arms=[GATE_ACCEPT, GATE_RESCUE],
        dim=dim,
        lam=lam,
    )

    output_path = args.output or str(Path(config["train_dir"]) / "rescue_bandit_prior.joblib")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "feature_names": rows[0].get("feature_names", []),
            "gate_arms": [GATE_ACCEPT, GATE_RESCUE],
            "rescue_actions": rescue_actions,
            "gate_prior": gate_prior,
            "rescue_prior": rescue_prior,
        },
        output_path,
    )
    print(f"wrote rescue bandit prior to {output_path}")


if __name__ == "__main__":
    main()
