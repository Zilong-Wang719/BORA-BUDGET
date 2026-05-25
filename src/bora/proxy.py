from __future__ import annotations

from typing import Any

import joblib
import numpy as np
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split


class ConstantProxy:
    def __init__(self, value: float = 0.5) -> None:
        self.value = float(value)

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        count = len(features)
        probs = np.full((count, 2), 0.0, dtype=float)
        probs[:, 1] = self.value
        probs[:, 0] = 1.0 - self.value
        return probs


class IsotonicCalibratedProxy:
    def __init__(self, base_model: Any, calibrator: IsotonicRegression) -> None:
        self.base_model = base_model
        self.calibrator = calibrator

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        base_probs = self.base_model.predict_proba(features)[:, 1]
        calibrated = np.clip(self.calibrator.predict(base_probs), 0.0, 1.0)
        probs = np.zeros((len(calibrated), 2), dtype=float)
        probs[:, 1] = calibrated
        probs[:, 0] = 1.0 - calibrated
        return probs


def _compute_metrics(labels: np.ndarray, probs: np.ndarray) -> dict[str, float | None]:
    metrics: dict[str, float | None] = {
        "num_examples": float(len(labels)),
        "positive_rate": float(np.mean(labels)) if len(labels) else None,
        "brier": None,
        "auc": None,
    }
    if len(labels):
        metrics["brier"] = float(brier_score_loss(labels, probs))
        if len(np.unique(labels)) > 1:
            metrics["auc"] = float(roc_auc_score(labels, probs))
    return metrics


def fit_proxy_model(
    features: list[list[float]],
    labels: list[int],
    config: dict[str, Any],
    validation_features: list[list[float]] | None = None,
    validation_labels: list[int] | None = None,
) -> tuple[Any, dict[str, Any]]:
    if not features:
        return ConstantProxy(0.5), {"train": {}, "validation": {}, "calibrated": False}
    unique = sorted(set(labels))
    if len(unique) < 2:
        value = float(unique[0]) if unique else 0.5
        return ConstantProxy(value), {
            "train": {"num_examples": float(len(labels)), "positive_rate": value},
            "validation": {},
            "calibrated": False,
        }

    proxy_cfg = config.get("proxy", {})
    base_model = HistGradientBoostingClassifier(
        max_depth=int(proxy_cfg.get("max_depth", 3)),
        learning_rate=float(proxy_cfg.get("learning_rate", 0.05)),
        max_iter=int(proxy_cfg.get("max_iter", 200)),
        random_state=int(config.get("random_seed", 0)),
    )
    feature_array = np.asarray(features, dtype=float)
    label_array = np.asarray(labels, dtype=int)
    validation_split = float(proxy_cfg.get("validation_split", 0.2))
    calibrate = bool(proxy_cfg.get("calibrate", False))
    metrics: dict[str, Any] = {"train": {}, "validation": {}, "calibrated": False}
    has_external_validation = validation_features is not None and validation_labels is not None

    if has_external_validation:
        X_train, y_train = feature_array, label_array
        X_val = np.asarray(validation_features, dtype=float)
        y_val = np.asarray(validation_labels, dtype=int)
    else:
        class_counts = np.bincount(label_array, minlength=2)
        can_stratify = len(np.unique(label_array)) > 1 and int(class_counts.min()) >= 2
        if 0.0 < validation_split < 1.0 and len(label_array) >= 5:
            stratify = label_array if can_stratify else None
            X_train, X_val, y_train, y_val = train_test_split(
                feature_array,
                label_array,
                test_size=validation_split,
                random_state=int(config.get("random_seed", 0)),
                stratify=stratify,
            )
        else:
            X_train, y_train = feature_array, label_array
            X_val = np.empty((0, feature_array.shape[1]), dtype=float)
            y_val = np.empty((0,), dtype=int)

    model = clone(base_model)
    model.fit(X_train, y_train)
    train_probs = model.predict_proba(X_train)[:, 1]
    metrics["train"] = _compute_metrics(y_train, train_probs)

    calibrated_model: Any = model
    if len(y_val):
        val_probs = model.predict_proba(X_val)[:, 1]
        metrics["validation"] = _compute_metrics(y_val, val_probs)
        if calibrate and len(np.unique(y_val)) > 1:
            calibrator = IsotonicRegression(out_of_bounds="clip")
            calibrator.fit(val_probs, y_val)
            calibrated_model = IsotonicCalibratedProxy(model, calibrator)
            metrics["calibrated"] = True
            metrics["calibration_examples"] = float(len(y_val))
    return calibrated_model, metrics


def predict_correctness(model: Any, features: list[float] | np.ndarray) -> float:
    vector = np.asarray(features, dtype=float).reshape(1, -1)
    return float(model.predict_proba(vector)[0, 1])


def save_proxy(model: Any, path: str) -> None:
    joblib.dump(model, path)


def load_proxy(path: str) -> Any:
    return joblib.load(path)
