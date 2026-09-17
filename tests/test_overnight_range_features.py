from datetime import timedelta

import polars as pl

from libs.overnight_range import (
    OvernightRangeBreakoutConfig,
    OvernightRangeFeatureConfig,
    build_overnight_range_breakout_ledger,
    build_overnight_range_features,
)
from tests.test_overnight_range_breakout import _session


def test_feature_frame_uses_prior_day_history_and_excludes_trade_outcome_from_inputs():
    first_day = _session()
    second_day = first_day.with_columns(pl.col("timestamp") + timedelta(days=1))
    bars = pl.concat([first_day, second_day])
    ledger = build_overnight_range_breakout_ledger(bars, OvernightRangeBreakoutConfig(buffer_pips=0))
    config = OvernightRangeFeatureConfig(
        daily_range_windows=(1,),
        moving_average_window=1,
        minimum_range_coverage=0,
        minimum_trade_coverage=0,
    )

    dataset, columns = build_overnight_range_features(ledger, bars, config)

    assert dataset.height == 1
    assert "range_to_adr_1" in columns
    assert "range_to_previous_overnight_range" in columns
    assert "side_is_long" in columns
    assert "entry_with_previous_day_direction" in columns
    assert "gross_pnl_pips" not in columns
    assert "r_return" not in columns
    assert "exit_reason" not in columns
    assert dataset.select(columns).null_count().row(0).count(0) == len(columns)
    assert dataset["target"].to_list() == [1]


def test_feature_builder_accepts_legacy_ledger_with_only_r_multiple():
    first_day = _session()
    bars = pl.concat([first_day, first_day.with_columns(pl.col("timestamp") + timedelta(days=1))])
    ledger = build_overnight_range_breakout_ledger(bars, OvernightRangeBreakoutConfig(buffer_pips=0)).drop("r_return")
    config = OvernightRangeFeatureConfig(
        daily_range_windows=(1,), moving_average_window=1,
        minimum_range_coverage=0, minimum_trade_coverage=0,
    )

    dataset, _ = build_overnight_range_features(ledger, bars, config)

    assert dataset["r_return"].to_list() == dataset["r_multiple"].to_list()
