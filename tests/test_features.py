from datetime import datetime, timedelta, timezone

import polars as pl
import pytest

from range_bars_ml.data import prepare_dataset
from range_bars_ml.features import FeatureConfig, build_features, build_next_bar_labels


def bars(count: int = 80) -> pl.DataFrame:
    base = datetime(2025, 1, 1, tzinfo=timezone.utc)
    opens = [100.0 + i for i in range(count)]
    closes = [value + (1 if i % 2 else -1) for i, value in enumerate(opens)]
    return pl.DataFrame({
        "open": opens, "high": [value + 2 for value in opens], "low": [value - 2 for value in opens], "close": closes,
        "timestamp": [base + timedelta(minutes=i) for i in range(count)], "close_time": [base + timedelta(minutes=i, seconds=30) for i in range(count)],
        "total_sold_volume": [10.0] * count, "total_bought_volume": [12.0] * count,
        "total_sold_dollar_volume": [1000.0] * count, "total_bought_dollar_volume": [1200.0] * count,
    })


def test_next_bar_labels_are_shifted_and_final_row_is_null():
    labelled = build_next_bar_labels(bars(3))
    assert labelled["target"].to_list() == [1, 0, None]


def test_prepared_data_has_no_null_features_and_excludes_raw_columns():
    dataset, features = prepare_dataset(bars())
    assert dataset.height < 80
    assert "open" not in features
    assert "log_return_10" in features
    assert "bought_dollar_share_10" in features
    assert "bar_time_share_window_10_lag_0" in features
    assert "bar_is_up_window_10_lag_9" in features
    assert "open" not in dataset.columns
    assert "close_time" not in dataset.columns
    assert dataset.select(features).null_count().row(0).count(0) == len(features)


def test_windowed_bar_composition_is_time_and_dollar_volume_normalised():
    featured = build_features(bars(3), FeatureConfig(windows=(3,)))
    columns = [
        "bar_time_share_window_3_lag_0", "bar_bought_dollar_share_window_3_lag_1",
        "bar_sold_dollar_share_window_3_lag_2", "bar_total_dollar_share_window_3_lag_0",
        "bar_is_up_window_3_lag_0", "bar_is_up_window_3_lag_1", "bar_is_up_window_3_lag_2",
    ]
    row = featured.select(columns).row(2, named=True)
    # All three bars take 30 seconds and have equal bought/sold/total dollar volume.
    assert row["bar_time_share_window_3_lag_0"] == pytest.approx(1 / 3)
    assert row["bar_bought_dollar_share_window_3_lag_1"] == pytest.approx(1 / 3)
    assert row["bar_sold_dollar_share_window_3_lag_2"] == pytest.approx(1 / 3)
    assert row["bar_total_dollar_share_window_3_lag_0"] == pytest.approx(1 / 3)
    # The current bar is down, the preceding bar is up, then the oldest is down.
    assert [row[f"bar_is_up_window_3_lag_{lag}"] for lag in range(3)] == [0, 1, 0]
