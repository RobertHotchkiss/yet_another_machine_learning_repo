"""Causal feature and label construction for completed range bars."""

from __future__ import annotations

from dataclasses import dataclass
from math import pi
import polars as pl

REQUIRED_COLUMNS = {
    "open", "high", "low", "close", "timestamp", "close_time",
    "total_sold_volume", "total_bought_volume",
    "total_sold_dollar_volume", "total_bought_dollar_volume",
}


@dataclass(frozen=True)
class FeatureConfig:
    """Feature windows expressed in completed range bars."""

    windows: tuple[int, ...] = (1, 2,3,4,5,6,7,8,9,10)
    time_column: str = "close_time"


def _validate_columns(frame: pl.DataFrame) -> None:
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"Range-bar data is missing required columns: {sorted(missing)}")


def feature_columns(frame: pl.DataFrame) -> list[str]:
    """Return model inputs while excluding raw columns and training labels."""
    excluded = REQUIRED_COLUMNS | {"first_trade_id", "last_trade_id", "target"}
    return [name for name in frame.columns if name not in excluded]


def _bar_feature_expressions(config: FeatureConfig) -> list[pl.Expr]:
    """Features describing the just-completed bar itself."""
    close = pl.col("close")
    open_ = pl.col("open")
    total_volume = pl.col("total_bought_volume") + pl.col("total_sold_volume")
    total_dollar = pl.col("total_bought_dollar_volume") + pl.col("total_sold_dollar_volume")
    return [
        ((close / open_) - 1).alias("bar_return"),
        ((pl.col("high") - pl.col("low")) / open_).alias("bar_range_pct"),
        ((close - pl.col("low")) / (pl.col("high") - pl.col("low") + 1e-12)).alias("close_in_range"),
        ((pl.col("total_bought_volume") - pl.col("total_sold_volume")) / (total_volume + 1e-12)).alias("volume_imbalance"),
        ((pl.col("total_bought_dollar_volume") - pl.col("total_sold_dollar_volume")) / (total_dollar + 1e-12)).alias("dollar_imbalance"),
        ((pl.col(config.time_column).dt.hour() * 2 * pi / 24).sin()).alias("close_hour_sin"),
        ((pl.col(config.time_column).dt.hour() * 2 * pi / 24).cos()).alias("close_hour_cos"),
        ((pl.col(config.time_column).dt.weekday() * 2 * pi / 7).sin()).alias("close_weekday_sin"),
        ((pl.col(config.time_column).dt.weekday() * 2 * pi / 7).cos()).alias("close_weekday_cos"),
    ]


def _window_feature_expressions(window: int) -> list[pl.Expr]:
    """Trailing, causal features for a completed ``window``-bar move.

    All outputs in this family are dimensionless price/imbalance measures.
    Per-bar composition features are built separately below.
    """
    close = pl.col("close")
    log_return = close.log() - close.shift(1).log()
    bought_dollar = pl.col("total_bought_dollar_volume")
    sold_dollar = pl.col("total_sold_dollar_volume")
    total_dollar = bought_dollar + sold_dollar
    window_bought_dollar = bought_dollar.rolling_sum(window_size=window)
    window_sold_dollar = sold_dollar.rolling_sum(window_size=window)
    window_total_dollar = total_dollar.rolling_sum(window_size=window)
    return [
        (close.log() - close.shift(window).log()).alias(f"log_return_{window}"),
        log_return.rolling_mean(window_size=window).alias(f"return_mean_{window}"),
        log_return.rolling_std(window_size=window, ddof=0).alias(f"return_std_{window}"),
        pl.col("volume_imbalance").rolling_mean(window_size=window).alias(f"imbalance_mean_{window}"),
        # Side volume relative to the total dollar volume of the same period.
        (window_bought_dollar / (window_total_dollar + 1e-12)).alias(f"bought_dollar_share_{window}"),
        (window_sold_dollar / (window_total_dollar + 1e-12)).alias(f"sold_dollar_share_{window}"),
        ((window_bought_dollar - window_sold_dollar) / (window_total_dollar + 1e-12)).alias(f"dollar_imbalance_{window}"),
    ]


def _window_bar_composition_expressions(window: int) -> list[pl.Expr]:
    """Represent every bar in a trailing window as a normalised composition.

    ``lag_0`` is the just-closed bar and ``lag_n`` is ``n`` bars before it.
    Each share uses the total of the same trailing window as its denominator.
    """
    bar_duration = (pl.col("close_time") - pl.col("timestamp")).dt.total_milliseconds()
    bought_dollar = pl.col("total_bought_dollar_volume")
    sold_dollar = pl.col("total_sold_dollar_volume")
    total_dollar = bought_dollar + sold_dollar
    window_duration = bar_duration.rolling_sum(window_size=window)
    window_bought = bought_dollar.rolling_sum(window_size=window)
    window_sold = sold_dollar.rolling_sum(window_size=window)
    window_total = total_dollar.rolling_sum(window_size=window)
    expressions: list[pl.Expr] = []
    for lag in range(window):
        suffix = f"window_{window}_lag_{lag}"
        expressions.extend([
            (bar_duration.shift(lag) / (window_duration + 1e-12)).alias(f"bar_time_share_{suffix}"),
            (bought_dollar.shift(lag) / (window_bought + 1e-12)).alias(f"bar_bought_dollar_share_{suffix}"),
            (sold_dollar.shift(lag) / (window_sold + 1e-12)).alias(f"bar_sold_dollar_share_{suffix}"),
            (total_dollar.shift(lag) / (window_total + 1e-12)).alias(f"bar_total_dollar_share_{suffix}"),
            (pl.col("close").shift(lag) > pl.col("open").shift(lag)).cast(pl.Int8).alias(f"bar_is_up_{suffix}"),
        ])
    return expressions


def build_features(frame: pl.DataFrame, config: FeatureConfig = FeatureConfig()) -> pl.DataFrame:
    """Create features observable only when each current bar has closed."""
    _validate_columns(frame)
    if not config.windows or any(window <= 0 for window in config.windows):
        raise ValueError("Feature windows must contain positive integers")

    df = frame.sort(config.time_column)
    # Materialise base columns before referring to them in rolling expressions.
    df = df.with_columns(_bar_feature_expressions(config))
    expressions: list[pl.Expr] = []
    for window in config.windows:
        expressions.extend(_window_feature_expressions(window))
        expressions.extend(_window_bar_composition_expressions(window))
    return df.with_columns(expressions)


def build_next_bar_labels(frame: pl.DataFrame) -> pl.DataFrame:
    """Add ``target`` for next-bar direction; flat and final bars receive null labels."""
    _validate_columns(frame)
    return frame.with_columns(
        pl.col("close").shift(-1).alias("_next_close"),
        pl.col("open").shift(-1).alias("_next_open"),
    ).with_columns(
        pl.when(pl.col("_next_close") > pl.col("_next_open"))
        .then(pl.lit(1, dtype=pl.Int8))
        .when(pl.col("_next_close") < pl.col("_next_open"))
        .then(pl.lit(0, dtype=pl.Int8))
        .otherwise(pl.lit(None, dtype=pl.Int8))
        .alias("target")
    ).drop(["_next_close", "_next_open"])
