"""Causal feature and label construction for completed FX renko bars."""

from __future__ import annotations

from dataclasses import dataclass
from math import pi

import polars as pl


EPSILON = 1e-12


REQUIRED_FX_COLUMNS = {
    "timestamp", "close_timestamp", "mid_open", "mid_high", "mid_low", "mid_close",
}


@dataclass(frozen=True)
class FxFeatureConfig:
    """Feature windows expressed in completed FX renko bars."""

    windows: tuple[int, ...] = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
    time_column: str = "close_timestamp"


def _validate_columns(frame: pl.DataFrame, config: FxFeatureConfig) -> None:
    missing = REQUIRED_FX_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"FX renko data is missing required columns: {sorted(missing)}")
    if config.time_column not in frame.columns:
        raise ValueError(f"FX renko data is missing configured time column: {config.time_column}")


def fx_feature_columns(frame: pl.DataFrame) -> list[str]:
    """Return FX model inputs while excluding raw columns and labels."""
    return [name for name in frame.columns if name not in REQUIRED_FX_COLUMNS | {"target"}]


def _bar_feature_expressions(config: FxFeatureConfig) -> list[pl.Expr]:
    open_ = pl.col("mid_open")
    close = pl.col("mid_close")
    high = pl.col("mid_high")
    low = pl.col("mid_low")
    bar_range = high - low
    body_high = pl.max_horizontal(open_, close)
    body_low = pl.min_horizontal(open_, close)
    duration_ms = (pl.col("close_timestamp") - pl.col("timestamp")).dt.total_milliseconds()
    return [
        ((close / open_) - 1).alias("bar_return"),
        (bar_range / open_).alias("bar_range_pct"),
        ((close - low) / (bar_range + EPSILON)).alias("close_in_range"),
        ((high - body_high) / (bar_range + EPSILON)).alias("upper_wick_share"),
        ((body_low - low) / (bar_range + EPSILON)).alias("lower_wick_share"),
        ((close - open_).abs() / (bar_range + EPSILON)).alias("body_to_range"),
        pl.when(close > open_).then(1).when(close < open_).then(-1).otherwise(0).cast(pl.Int8).alias("bar_direction"),
        duration_ms.cast(pl.Float64).log1p().alias("log_bar_duration"),
        ((pl.col(config.time_column).dt.hour() * 2 * pi / 24).sin()).alias("close_hour_sin"),
        ((pl.col(config.time_column).dt.hour() * 2 * pi / 24).cos()).alias("close_hour_cos"),
        ((pl.col(config.time_column).dt.weekday() * 2 * pi / 7).sin()).alias("close_weekday_sin"),
        ((pl.col(config.time_column).dt.weekday() * 2 * pi / 7).cos()).alias("close_weekday_cos"),
    ]


def _window_feature_expressions(window: int) -> list[pl.Expr]:
    close = pl.col("mid_close")
    log_return = close.log() - close.shift(1).log()
    return_std = log_return.rolling_std(window_size=window, ddof=0)
    close_mean = close.rolling_mean(window_size=window)
    close_std = close.rolling_std(window_size=window, ddof=0)
    range_pct = pl.col("bar_range_pct")
    range_mean = range_pct.rolling_mean(window_size=window)
    duration = pl.col("log_bar_duration")
    duration_mean = duration.rolling_mean(window_size=window)
    duration_std = duration.rolling_std(window_size=window, ddof=0)
    direction = pl.col("bar_direction")
    reversal = direction.ne(direction.shift(1)).fill_null(False).cast(pl.Float64)
    direction_run = direction.ne(direction.shift(1)).fill_null(True).cast(pl.Int64).cum_sum()
    direction_streak = direction.ne(0).cast(pl.Int64).cum_sum().over(direction_run)
    cumulative_return = close.log() - close.shift(window).log()

    def standardized(numerator: pl.Expr, denominator: pl.Expr) -> pl.Expr:
        return pl.when(denominator.abs() > EPSILON).then(numerator / denominator).otherwise(0.0)

    return [
        cumulative_return.alias(f"log_return_{window}"),
        log_return.rolling_mean(window_size=window).alias(f"return_mean_{window}"),
        return_std.alias(f"return_std_{window}"),
        direction.cast(pl.Float64).rolling_mean(window_size=window).alias(f"direction_persistence_{window}"),
        reversal.rolling_mean(window_size=window).alias(f"reversal_rate_{window}"),
        direction_streak.clip(upper_bound=window).cast(pl.Float64).truediv(window).alias(f"direction_streak_share_{window}"),
        standardized(cumulative_return, return_std * window**0.5).alias(f"trend_strength_{window}"),
        standardized(log_return - log_return.rolling_mean(window_size=window), return_std).alias(f"return_acceleration_{window}"),
        standardized(close - close_mean, close_std).alias(f"close_mean_zscore_{window}"),
        ((close - pl.col("mid_low").rolling_min(window_size=window)) /
         (pl.col("mid_high").rolling_max(window_size=window) - pl.col("mid_low").rolling_min(window_size=window) + EPSILON)).alias(f"channel_position_{window}"),
        range_mean.alias(f"range_mean_{window}"),
        range_pct.rolling_std(window_size=window, ddof=0).alias(f"range_std_{window}"),
        standardized(range_pct - range_mean, range_mean).alias(f"range_regime_{window}"),
        duration_mean.alias(f"duration_log_mean_{window}"),
        duration_std.alias(f"duration_log_std_{window}"),
        standardized(duration - duration_mean, duration_std).alias(f"duration_log_zscore_{window}"),
    ]


def _window_bar_composition_expressions(window: int) -> list[pl.Expr]:
    bar_duration = (pl.col("close_timestamp") - pl.col("timestamp")).dt.total_milliseconds()
    window_duration = bar_duration.rolling_sum(window_size=window)
    expressions: list[pl.Expr] = []
    for lag in range(window):
        suffix = f"window_{window}_lag_{lag}"
        expressions.extend([
            (bar_duration.shift(lag) / (window_duration + EPSILON)).alias(f"bar_time_share_{suffix}"),
            (pl.col("mid_close").shift(lag) > pl.col("mid_open").shift(lag)).cast(pl.Int8).alias(f"bar_is_up_{suffix}"),
        ])
    return expressions


def build_fx_features(frame: pl.DataFrame, config: FxFeatureConfig = FxFeatureConfig()) -> pl.DataFrame:
    """Create no-volume features observable when each FX bar has closed."""
    _validate_columns(frame, config)
    if not config.windows or any(window <= 0 for window in config.windows):
        raise ValueError("Feature windows must contain positive integers")

    df = frame.sort(config.time_column).with_columns(_bar_feature_expressions(config))
    expressions: list[pl.Expr] = []
    for window in config.windows:
        expressions.extend(_window_feature_expressions(window))
        expressions.extend(_window_bar_composition_expressions(window))
    return df.with_columns(expressions)


def build_next_fx_bar_labels(frame: pl.DataFrame) -> pl.DataFrame:
    """Add next-bar direction labels; flat and final bars receive null labels."""
    _validate_columns(frame, FxFeatureConfig())
    return frame.with_columns(
        pl.col("mid_close").shift(-1).alias("_next_close"),
        pl.col("mid_open").shift(-1).alias("_next_open"),
    ).with_columns(
        pl.when(pl.col("_next_close") > pl.col("_next_open"))
        .then(pl.lit(1, dtype=pl.Int8))
        .when(pl.col("_next_close") < pl.col("_next_open"))
        .then(pl.lit(0, dtype=pl.Int8))
        .otherwise(pl.lit(None, dtype=pl.Int8))
        .alias("target")
    ).drop(["_next_close", "_next_open"])
