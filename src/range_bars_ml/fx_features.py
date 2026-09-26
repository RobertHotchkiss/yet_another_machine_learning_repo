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

    windows: tuple[int, ...] = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 50, 100)
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
    open_ = pl.col("mid_open")
    close = pl.col("mid_close")
    bar_range = pl.col("mid_high") - pl.col("mid_low")
    log_return = close.log() - close.shift(1).log()
    return_std = log_return.rolling_std(window_size=window, ddof=0)
    close_mean = close.rolling_mean(window_size=window)
    close_std = close.rolling_std(window_size=window, ddof=0)
    range_pct = pl.col("bar_range_pct")
    range_mean = range_pct.rolling_mean(window_size=window)
    duration = pl.col("log_bar_duration")
    duration_mean = duration.rolling_mean(window_size=window)
    duration_std = duration.rolling_std(window_size=window, ddof=0)
    duration_seconds = (
        (pl.col("close_timestamp") - pl.col("timestamp")).dt.total_milliseconds() / 1_000
    )
    speed = 1 / (duration_seconds + EPSILON)
    speed_mean = speed.rolling_mean(window_size=window)
    speed_std = speed.rolling_std(window_size=window, ddof=0)
    direction = pl.col("bar_direction")
    reversal = direction.ne(direction.shift(1)).fill_null(False).cast(pl.Float64)
    direction_run = direction.ne(direction.shift(1)).fill_null(True).cast(pl.Int64).cum_sum()
    direction_streak = direction.ne(0).cast(pl.Int64).cum_sum().over(direction_run)
    cumulative_return = close.log() - close.shift(window).log()
    abs_return = log_return.abs()
    prior_high = pl.col("mid_high").shift(1).rolling_max(window_size=window)
    prior_low = pl.col("mid_low").shift(1).rolling_min(window_size=window)
    rolling_high = pl.col("mid_high").rolling_max(window_size=window)
    rolling_low = pl.col("mid_low").rolling_min(window_size=window)
    ema = close.ewm_mean(span=window, adjust=False)
    up_bar = (direction > 0).cast(pl.Float64)
    down_bar = (direction < 0).cast(pl.Float64)
    up_duration_count = up_bar.rolling_sum(window_size=window)
    down_duration_count = down_bar.rolling_sum(window_size=window)
    up_duration_mean = pl.when(up_duration_count > 0).then(
        (duration * up_bar).rolling_sum(window_size=window) / up_duration_count
    ).otherwise(None)
    down_duration_mean = pl.when(down_duration_count > 0).then(
        (duration * down_bar).rolling_sum(window_size=window) / down_duration_count
    ).otherwise(None)
    fast_bar = (duration_seconds < duration_seconds.rolling_median(window_size=window)).cast(pl.Float64)
    return_per_second = log_return / (duration_seconds + EPSILON)
    signed_range_per_second = ((bar_range / (open_ + EPSILON)) * direction) / (duration_seconds + EPSILON)

    def standardized(numerator: pl.Expr, denominator: pl.Expr) -> pl.Expr:
        return pl.when(denominator.abs() > EPSILON).then(numerator / denominator).otherwise(0.0)

    def bars_since_extreme(value: pl.Expr, extreme: pl.Expr, *, is_high: bool) -> pl.Expr:
        comparator = (lambda candidate: candidate >= extreme) if is_high else (lambda candidate: candidate <= extreme)
        # Candidates are ordered from newest to oldest, so ties resolve to the most recent bar.
        return pl.coalesce([
            pl.when(comparator(value.shift(lag))).then(pl.lit(lag, dtype=pl.Int64))
            for lag in range(window)
        ])

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
        # Longer-horizon structure and EMA trend context.
        ((close / rolling_high) - 1).alias(f"drawdown_from_high_{window}"),
        ((close / rolling_low) - 1).alias(f"rebound_from_low_{window}"),
        ((close / (prior_high + EPSILON)) - 1).alias(f"distance_to_prior_high_{window}"),
        ((close / (prior_low + EPSILON)) - 1).alias(f"distance_to_prior_low_{window}"),
        (close > prior_high).cast(pl.Int8).alias(f"breakout_up_{window}"),
        (close < prior_low).cast(pl.Int8).alias(f"breakout_down_{window}"),
        (pl.col("mid_high") >= rolling_high).cast(pl.Float64).rolling_sum(window_size=window).alias(f"new_high_count_{window}"),
        (pl.col("mid_low") <= rolling_low).cast(pl.Float64).rolling_sum(window_size=window).alias(f"new_low_count_{window}"),
        bars_since_extreme(pl.col("mid_high"), rolling_high, is_high=True).alias(f"bars_since_rolling_high_{window}"),
        bars_since_extreme(pl.col("mid_low"), rolling_low, is_high=False).alias(f"bars_since_rolling_low_{window}"),
        ((close / (ema + EPSILON)) - 1).alias(f"ema_distance_{window}"),
        ((ema / (ema.shift(window) + EPSILON)) - 1).alias(f"ema_slope_{window}"),
        # Volatility level, asymmetry, extremes, and a robust jump proxy.
        abs_return.rolling_mean(window_size=window).alias(f"abs_return_mean_{window}"),
        (log_return.pow(2).rolling_mean(window_size=window)).sqrt().alias(f"realized_volatility_{window}"),
        (pl.when(log_return > 0).then(log_return.pow(2)).otherwise(0.0).rolling_mean(window_size=window)).sqrt().alias(f"upside_volatility_{window}"),
        (pl.when(log_return < 0).then(log_return.pow(2)).otherwise(0.0).rolling_mean(window_size=window)).sqrt().alias(f"downside_volatility_{window}"),
        abs_return.rolling_max(window_size=window).alias(f"max_abs_return_{window}"),
        standardized(
            abs_return - abs_return.rolling_median(window_size=window),
            (abs_return - abs_return.rolling_median(window_size=window)).abs().rolling_median(window_size=window),
        ).alias(f"return_jump_score_{window}"),
        # Directional formation speed.  These use elapsed wall-clock time, not volume.
        up_duration_mean.fill_null(0.0).alias(f"up_duration_log_mean_{window}"),
        down_duration_mean.fill_null(0.0).alias(f"down_duration_log_mean_{window}"),
        standardized(up_duration_mean.fill_null(0.0) - down_duration_mean.fill_null(0.0), duration_std).alias(f"up_down_duration_spread_{window}"),
        pl.when(direction > 0).then(standardized(duration - up_duration_mean, duration_std))
        .when(direction < 0).then(standardized(duration - down_duration_mean, duration_std))
        .otherwise(0.0).alias(f"same_direction_duration_zscore_{window}"),
        return_per_second.rolling_mean(window_size=window).alias(f"return_per_second_mean_{window}"),
        signed_range_per_second.rolling_mean(window_size=window).alias(f"signed_range_per_second_mean_{window}"),
        (fast_bar * (direction > 0).cast(pl.Float64)).rolling_mean(window_size=window).alias(f"fast_up_share_{window}"),
        (fast_bar * (direction < 0).cast(pl.Float64)).rolling_mean(window_size=window).alias(f"fast_down_share_{window}"),
        standardized(speed - speed_mean, speed_std).alias(f"formation_speed_zscore_{window}"),
    ]


def _cross_window_feature_expressions(windows: tuple[int, ...]) -> list[pl.Expr]:
    """Compare adjacent configured horizons without introducing hidden lookbacks."""
    close = pl.col("mid_close")
    log_return = close.log() - close.shift(1).log()
    expressions: list[pl.Expr] = []
    for fast, slow in zip(windows, windows[1:]):
        fast_ema = close.ewm_mean(span=fast, adjust=False)
        slow_ema = close.ewm_mean(span=slow, adjust=False)
        fast_volatility = log_return.rolling_std(window_size=fast, ddof=0)
        slow_volatility = log_return.rolling_std(window_size=slow, ddof=0)
        expressions.extend([
            ((fast_ema / (slow_ema + EPSILON)) - 1).alias(f"ema_fast_slow_gap_{fast}_{slow}"),
            pl.when(slow_volatility > EPSILON).then(fast_volatility / slow_volatility).otherwise(0.0)
            .alias(f"volatility_ratio_{fast}_{slow}"),
        ])
    return expressions


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
    if len(set(config.windows)) != len(config.windows):
        raise ValueError("Feature windows must not contain duplicates")

    df = frame.sort(config.time_column).with_columns(_bar_feature_expressions(config))
    expressions: list[pl.Expr] = []
    for window in config.windows:
        expressions.extend(_window_feature_expressions(window))
        expressions.extend(_window_bar_composition_expressions(window))
    expressions.extend(_cross_window_feature_expressions(tuple(sorted(config.windows))))
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
