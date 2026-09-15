"""Input validation and assembly of model-ready data."""

from __future__ import annotations

from pathlib import Path

import polars as pl

from .features import FeatureConfig, build_features, build_next_bar_labels, feature_columns


def load_range_bars(path: str | Path) -> pl.DataFrame:
    """Read Parquet range bars without embedding a machine-specific path."""
    return pl.read_parquet(path)


def prepare_dataset(frame: pl.DataFrame, config: FeatureConfig = FeatureConfig()) -> tuple[pl.DataFrame, list[str]]:
    """Return labelled, finite, model-ready rows and their feature column names."""
    featured = build_features(frame, config)
    labelled = build_next_bar_labels(featured)
    columns = feature_columns(labelled)
    dataset = labelled.drop_nulls(columns + ["target"])
    # Polars represents non-finite floats rather than nulls; LightGBM accepts NaN but not inf.
    for column in columns:
        if dataset.schema[column].is_numeric():
            dataset = dataset.filter(pl.col(column).is_finite())
    # Raw market fields are construction inputs, not model/debug outputs.
    return dataset.select(columns + ["target"]), columns
