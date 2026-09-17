"""Input validation and assembly of model-ready FX renko data."""

from __future__ import annotations

from pathlib import Path

import polars as pl

from .fx_features import FxFeatureConfig, build_fx_features, build_next_fx_bar_labels, fx_feature_columns


def load_fx_renko_bars(path: str | Path) -> pl.DataFrame:
    """Read FX renko bars without embedding a machine-specific path."""
    return pl.read_parquet(path)


def prepare_fx_dataset(frame: pl.DataFrame, config: FxFeatureConfig = FxFeatureConfig()) -> tuple[pl.DataFrame, list[str]]:
    """Return labelled, finite, model-ready FX rows and their feature names."""
    labelled = build_next_fx_bar_labels(build_fx_features(frame, config))
    columns = fx_feature_columns(labelled)
    dataset = labelled.drop_nulls(columns + ["target"])
    for column in columns:
        if dataset.schema[column].is_numeric():
            dataset = dataset.filter(pl.col(column).is_finite())
    return dataset.select(columns + ["target"]), columns
