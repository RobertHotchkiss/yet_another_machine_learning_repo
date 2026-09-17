"""FX-specific experiment entry points built on the shared modelling pipeline."""

from __future__ import annotations

import polars as pl

from .fx_data import prepare_fx_dataset
from .pipeline import ExperimentConfig, ExperimentResult, ValidationResult, run_experiment, run_validation


def run_fx_validation(frame: pl.DataFrame, config: ExperimentConfig = ExperimentConfig()) -> ValidationResult:
    """Run walk-forward validation using FX renko features."""
    return run_validation(frame, config, dataset_preparer=prepare_fx_dataset)


def run_fx_experiment(frame: pl.DataFrame, config: ExperimentConfig = ExperimentConfig()) -> ExperimentResult:
    """Run the full selective-signal experiment using FX renko features."""
    return run_experiment(frame, config, dataset_preparer=prepare_fx_dataset)
