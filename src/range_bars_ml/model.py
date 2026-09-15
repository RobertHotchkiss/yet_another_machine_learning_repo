"""Probability model, calibration, selective thresholding, and persistence."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import polars as pl
from lightgbm import LGBMClassifier
from sklearn.isotonic import IsotonicRegression


@dataclass(frozen=True)
class ModelConfig:
    calibration_fraction: float = 0.10
    random_state: int = 42
    n_estimators: int = 500
    learning_rate: float = 0.05
    num_leaves: int = 31
    min_child_samples: int = 20
    subsample: float = 1.0
    subsample_freq: int = 0
    colsample_bytree: float = 1.0
    reg_lambda: float = 0.0
    n_jobs: int = -1
    verbosity: int = -1


@dataclass
class SelectiveSignalModel:
    feature_columns: list[str]
    estimator: LGBMClassifier
    calibrator: IsotonicRegression
    long_threshold: float
    short_threshold: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def predict_probability(self, frame: pl.DataFrame) -> np.ndarray:
        missing = set(self.feature_columns) - set(frame.columns)
        if missing:
            raise ValueError(f"Inference data is missing feature columns: {sorted(missing)}")
        raw = self.estimator.predict_proba(frame.select(self.feature_columns).to_numpy())[:, 1]
        return self.calibrator.predict(raw)

    def predict(self, frame: pl.DataFrame) -> pl.DataFrame:
        probability = self.predict_probability(frame)
        signal = np.where(probability >= self.long_threshold, "long", np.where(probability <= self.short_threshold, "short", "abstain"))
        return pl.DataFrame({"probability_up": probability, "signal": signal})

    def save(self, path: str | Path) -> None:
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str | Path) -> "SelectiveSignalModel":
        return joblib.load(path)


def fit_selective_model(frame: pl.DataFrame, feature_names: list[str], config: ModelConfig = ModelConfig()) -> SelectiveSignalModel:
    """Fit a classifier and calibrator, holding the latest training tail for calibration."""
    if not 0 < config.calibration_fraction < 0.5:
        raise ValueError("calibration_fraction must be between 0 and 0.5")
    calibration_start = int(frame.height * (1 - config.calibration_fraction))
    if calibration_start < 100 or frame.height - calibration_start < 50:
        raise ValueError("Need at least 100 fit rows and 50 calibration rows")
    fit, calibration = frame[:calibration_start], frame[calibration_start:]
    estimator = LGBMClassifier(
        objective="binary", n_estimators=config.n_estimators, learning_rate=config.learning_rate,
        num_leaves=config.num_leaves, min_child_samples=config.min_child_samples,
        subsample=config.subsample, subsample_freq=config.subsample_freq,
        colsample_bytree=config.colsample_bytree, reg_lambda=config.reg_lambda,
        random_state=config.random_state, n_jobs=config.n_jobs, verbosity=config.verbosity,
    )
    estimator.fit(fit.select(feature_names).to_numpy(), fit["target"].to_numpy())
    raw = estimator.predict_proba(calibration.select(feature_names).to_numpy())[:, 1]
    calibrator = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip")
    calibrator.fit(raw, calibration["target"].to_numpy())
    return SelectiveSignalModel(feature_names, estimator, calibrator, 0.5, 0.5)


def choose_thresholds(
    probability: np.ndarray,
    target: np.ndarray,
    *,
    minimum_coverage: float = 0.0,
    abstention_penalty: float = -0.001,
    long_gate: float | np.ndarray = 0.5,
    short_gate: float | np.ndarray = 0.5,
    steps: int = 50,
) -> tuple[float, float, dict[str, float]]:
    """Choose thresholds maximizing net score with a small abstention penalty.

    Correct directional calls score ``+1``, incorrect calls ``-1``, and an
    abstention scores ``abstention_penalty``.  The sentinel thresholds allow
    abstaining on every row when that is the best validation decision.
    """
    if not 0 <= minimum_coverage <= 1:
        raise ValueError("minimum_coverage must be in [0, 1]")
    if abstention_penalty >= 0:
        raise ValueError("abstention_penalty must be negative")
    if len(probability) != len(target) or not len(target):
        raise ValueError("probability and target must have the same non-zero length")
    long_values = np.append(np.linspace(0.50, 1.00, steps), 1.01)
    short_values = np.append(np.linspace(0.00, 0.50, steps), -0.01)
    required = int(np.ceil(len(target) * minimum_coverage))
    best: tuple[float, int, float, float] | None = None
    for long_threshold in long_values:
        long_mask = probability >= np.maximum(long_threshold, long_gate)
        long_correct = int(np.sum(target[long_mask] == 1))
        for short_threshold in short_values:
            short_mask = probability <= np.minimum(short_threshold, short_gate)
            count = int(long_mask.sum() + short_mask.sum())
            if count < required:
                continue
            correct = long_correct + int(np.sum(target[short_mask] == 0))
            wrong = count - correct
            score = correct - wrong + (len(target) - count) * abstention_penalty
            candidate = (score, count, float(long_threshold), float(short_threshold))
            if best is None or candidate[:2] > best[:2]:
                best = candidate
    if best is None:
        raise ValueError("No threshold combination meets minimum coverage")
    _, _, long_threshold, short_threshold = best
    return long_threshold, short_threshold, signal_metrics(
        probability, target, long_threshold, short_threshold, abstention_penalty=abstention_penalty,
        long_gate=long_gate, short_gate=short_gate,
    )


def signal_metrics(
    probability: np.ndarray,
    target: np.ndarray,
    long_threshold: float,
    short_threshold: float,
    *,
    abstention_penalty: float = -0.001,
    long_gate: float | np.ndarray = 0.5,
    short_gate: float | np.ndarray = 0.5,
) -> dict[str, float]:
    """Return precision diagnostics and net scores for selective predictions."""
    long_mask = probability >= np.maximum(long_threshold, long_gate)
    short_mask = probability <= np.minimum(short_threshold, short_gate)
    signal_mask = long_mask | short_mask
    correct = int((target[long_mask] == 1).sum() + (target[short_mask] == 0).sum())
    signals = int(signal_mask.sum())
    wrong = signals - correct
    abstentions = len(target) - signals
    score = correct - wrong + abstentions * abstention_penalty
    metrics: dict[str, float] = {
        "coverage": float(signal_mask.mean()), "signals": float(signals),
        "correct": float(correct), "wrong": float(wrong), "abstentions": float(abstentions),
        "score": float(score), "mean_score": float(score / len(target)),
    }
    metrics["precision"] = float(correct / signals) if signals else float("nan")
    for name, mask, expected in (("long", long_mask, 1), ("short", short_mask, 0)):
        metrics[f"{name}_coverage"] = float(mask.mean())
        metrics[f"{name}_precision"] = float((target[mask] == expected).mean()) if mask.any() else float("nan")
        metrics[f"{name}_signals"] = float(mask.sum())
    return metrics
