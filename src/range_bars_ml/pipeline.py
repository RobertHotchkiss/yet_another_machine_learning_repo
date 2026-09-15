"""End-to-end experiment orchestration built from the public library APIs."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
import polars as pl

from .data import prepare_dataset
from .features import FeatureConfig
from .model import ModelConfig, SelectiveSignalModel, choose_thresholds, fit_selective_model, signal_metrics
from .splits import SplitPlan, walk_forward_splits


@dataclass(frozen=True)
class ExperimentConfig:
    feature: FeatureConfig = field(default_factory=FeatureConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    test_fraction: float = 0.20
    n_folds: int = 3
    minimum_coverage: float = 0.0
    abstention_penalty: float = -0.001
    edge_margin: float = 0.0


@dataclass
class ExperimentResult:
    model: SelectiveSignalModel
    split_plan: SplitPlan
    validation_metrics: dict[str, float]
    test_metrics: dict[str, float]
    feature_columns: list[str]


@dataclass
class ValidationResult:
    """Walk-forward validation output, deliberately excluding final-test metrics."""

    split_plan: SplitPlan
    validation_metrics: dict[str, float]
    long_threshold: float
    short_threshold: float
    feature_columns: list[str]
    model_config: ModelConfig
    validation_rows: int


def run_validation(frame: pl.DataFrame, config: ExperimentConfig = ExperimentConfig()) -> ValidationResult:
    """Evaluate a configuration on development folds without fitting or scoring the final test.

    The newest ``test_fraction`` rows are reserved by the split plan.  This
    function only indexes fold train/validation rows and never indexes the
    final-test rows after the split has been made.
    """
    if config.edge_margin < 0:
        raise ValueError("edge_margin must be non-negative")
    dataset, features = prepare_dataset(frame, config.feature)
    split_plan = walk_forward_splits(dataset.height, test_fraction=config.test_fraction, n_folds=config.n_folds)
    probabilities: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    long_gates: list[np.ndarray] = []
    short_gates: list[np.ndarray] = []
    for train_index, validation_index in split_plan.folds:
        train = dataset[train_index.tolist()]
        model = fit_selective_model(train, features, config.model)
        validation = dataset[validation_index.tolist()]
        probabilities.append(model.predict_probability(validation))
        targets.append(validation["target"].to_numpy())
        training_up_rate = float(train["target"].mean())
        long_gates.append(np.full(validation.height, max(0.5, training_up_rate + config.edge_margin)))
        short_gates.append(np.full(validation.height, min(0.5, training_up_rate - config.edge_margin)))
    validation_probability, validation_target = np.concatenate(probabilities), np.concatenate(targets)
    validation_long_gate, validation_short_gate = np.concatenate(long_gates), np.concatenate(short_gates)
    long_threshold, short_threshold, _ = choose_thresholds(
        validation_probability, validation_target, minimum_coverage=config.minimum_coverage,
        abstention_penalty=config.abstention_penalty,
        long_gate=validation_long_gate, short_gate=validation_short_gate,
    )
    return ValidationResult(
        split_plan=split_plan,
        validation_metrics=signal_metrics(
            validation_probability, validation_target, long_threshold, short_threshold,
            abstention_penalty=config.abstention_penalty,
            long_gate=validation_long_gate, short_gate=validation_short_gate,
        ),
        long_threshold=long_threshold,
        short_threshold=short_threshold,
        feature_columns=features,
        model_config=config.model,
        validation_rows=len(validation_target),
    )


def run_experiment(frame: pl.DataFrame, config: ExperimentConfig = ExperimentConfig()) -> ExperimentResult:
    """Build data, tune thresholds walk-forward, then report one untouched final test."""
    validation_result = run_validation(frame, config)
    dataset, features = prepare_dataset(frame, config.feature)
    split_plan = validation_result.split_plan
    long_threshold, short_threshold = validation_result.long_threshold, validation_result.short_threshold
    final_model = fit_selective_model(dataset[split_plan.development.tolist()], features, config.model)
    development_up_rate = float(dataset[split_plan.development.tolist()]["target"].mean())
    long_gate = max(0.5, development_up_rate + config.edge_margin)
    short_gate = min(0.5, development_up_rate - config.edge_margin)
    final_model.long_threshold = max(long_threshold, long_gate)
    final_model.short_threshold = min(short_threshold, short_gate)
    final_model.metadata = {
        "minimum_coverage": config.minimum_coverage,
        "abstention_penalty": config.abstention_penalty,
        "development_up_rate": development_up_rate,
        "edge_margin": config.edge_margin,
        "long_gate": long_gate,
        "short_gate": short_gate,
        "validation_rows": validation_result.validation_rows,
        "feature_config": asdict(config.feature),
        "model_config": asdict(config.model),
    }
    test = dataset[split_plan.test.tolist()]
    test_probability = final_model.predict_probability(test)
    return ExperimentResult(
        model=final_model,
        split_plan=split_plan,
        validation_metrics=validation_result.validation_metrics,
        test_metrics=signal_metrics(
            test_probability, test["target"].to_numpy(), long_threshold, short_threshold,
            abstention_penalty=config.abstention_penalty,
        ),
        feature_columns=features,
    )
