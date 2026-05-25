from __future__ import annotations

from typing import Any

import numpy as np


class LinTS:
    def __init__(self, arms: list[str], d: int, lam: float = 1.0, ts_scale: float = 0.5) -> None:
        self.arms = list(arms)
        self.d = d
        self.lam = lam
        self.ts_scale = ts_scale
        self.A = {arm: lam * np.eye(d) for arm in self.arms}
        self.b = {arm: np.zeros(d) for arm in self.arms}

    def load_prior(self, prior: dict[str, dict[str, np.ndarray]] | None) -> None:
        if not prior:
            return
        for arm in self.arms:
            if arm not in prior:
                continue
            self.A[arm] = np.asarray(prior[arm]["A"], dtype=float)
            self.b[arm] = np.asarray(prior[arm]["b"], dtype=float)

    def select(
        self,
        features: np.ndarray,
        feasible_arms: list[str],
        rng: np.random.Generator,
    ) -> str:
        if not feasible_arms:
            return "STOP"
        scores: dict[str, float] = {}
        for arm in feasible_arms:
            precision = self.A[arm]
            inv = np.linalg.pinv(precision)
            mu = inv @ self.b[arm]
            covariance = (self.ts_scale**2) * ((inv + inv.T) / 2.0)
            covariance += 1e-6 * np.eye(self.d)
            theta = rng.multivariate_normal(mu, covariance)
            scores[arm] = float(np.dot(features, theta))
        return max(scores, key=scores.get)

    def update(self, features: np.ndarray, arm: str, reward: float) -> None:
        vector = np.asarray(features, dtype=float)
        self.A[arm] += np.outer(vector, vector)
        self.b[arm] += reward * vector


def make_default_prior(arms: list[str], d: int, lam: float) -> dict[str, dict[str, np.ndarray]]:
    return {arm: {"A": lam * np.eye(d), "b": np.zeros(d)} for arm in arms}
