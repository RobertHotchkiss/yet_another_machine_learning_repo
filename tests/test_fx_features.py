from datetime import datetime, timedelta, timezone

import polars as pl
import pytest

from range_bars_ml.fx_data import prepare_fx_dataset
from range_bars_ml.fx_features import FxFeatureConfig, build_fx_features, build_next_fx_bar_labels


def fx_bars(count: int = 80) -> pl.DataFrame:
    base = datetime(2025, 1, 1, tzinfo=timezone.utc)
    opens = [1.10 + i * 0.0001 for i in range(count)]
    closes = [value + (0.00005 if i % 2 else -0.00005) for i, value in enumerate(opens)]
    return pl.DataFrame({
        "timestamp": [base + timedelta(minutes=i) for i in range(count)],
        "close_timestamp": [base + timedelta(minutes=i, seconds=30) for i in range(count)],
        "mid_open": opens,
        "mid_high": [value + 0.0001 for value in opens],
        "mid_low": [value - 0.0001 for value in opens],
        "mid_close": closes,
    })


def test_fx_next_bar_labels_are_shifted_and_final_row_is_null():
    labelled = build_next_fx_bar_labels(fx_bars(3))
    assert labelled["target"].to_list() == [1, 0, None]


def test_fx_dataset_keeps_non_volume_features_and_excludes_raw_columns():
    dataset, features = prepare_fx_dataset(fx_bars())
    assert dataset.height < 80
    assert "log_return_10" in features
    assert "upper_wick_share" in features
    assert "trend_strength_10" in features
    assert "close_mean_zscore_10" in features
    assert "duration_log_zscore_10" in features
    assert "bar_time_share_window_10_lag_0" in features
    assert "bar_is_up_window_10_lag_9" in features
    assert not any("volume" in name or "imbalance" in name for name in features)
    assert not {"timestamp", "close_timestamp", "mid_open", "mid_high", "mid_low", "mid_close"} & set(features)
    assert dataset.select(features).null_count().row(0).count(0) == len(features)


def test_fx_bar_composition_uses_timestamp_duration_and_direction():
    featured = build_fx_features(fx_bars(3), FxFeatureConfig(windows=(3,)))
    row = featured.select([
        "bar_time_share_window_3_lag_0", "bar_is_up_window_3_lag_0",
        "bar_is_up_window_3_lag_1", "bar_is_up_window_3_lag_2",
    ]).row(2, named=True)
    assert row["bar_time_share_window_3_lag_0"] == pytest.approx(1 / 3)
    assert [row[f"bar_is_up_window_3_lag_{lag}"] for lag in range(3)] == [0, 1, 0]


def test_fx_technical_features_capture_causal_regimes():
    frame = pl.DataFrame({
        "timestamp": [
            datetime(2025, 1, 1, tzinfo=timezone.utc) + timedelta(minutes=offset)
            for offset in (0, 1, 3, 6)
        ],
        "close_timestamp": [
            datetime(2025, 1, 1, tzinfo=timezone.utc) + timedelta(minutes=offset)
            for offset in (0.5, 2, 4.5, 10)
        ],
        "mid_open": [100.0, 101.0, 102.0, 103.0],
        "mid_high": [101.5, 102.5, 103.5, 104.5],
        "mid_low": [99.5, 100.5, 101.5, 101.5],
        "mid_close": [101.0, 102.0, 103.0, 102.0],
    })
    featured = build_fx_features(frame, FxFeatureConfig(windows=(3,)))
    row = featured.row(3, named=True)

    assert row["bar_direction"] == -1
    assert row["upper_wick_share"] == pytest.approx(0.5)
    assert row["lower_wick_share"] == pytest.approx(1 / 6)
    assert row["body_to_range"] == pytest.approx(1 / 3)
    assert row["direction_persistence_3"] == pytest.approx(1 / 3)
    assert row["reversal_rate_3"] == pytest.approx(1 / 3)
    assert row["direction_streak_share_3"] == pytest.approx(1 / 3)
    assert row["channel_position_3"] == pytest.approx((102.0 - 100.5) / (104.5 - 100.5))
    assert row["duration_log_zscore_3"] > 0


def test_fx_features_require_the_fx_schema():
    with pytest.raises(ValueError, match="mid_close"):
        build_fx_features(fx_bars().drop("mid_close"))
