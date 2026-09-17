from datetime import date, datetime, timedelta, timezone

import polars as pl
import pytest

from libs.overnight_range import chart


def bars(start: datetime, count: int) -> pl.DataFrame:
    timestamps = [start + timedelta(minutes=index) for index in range(count)]
    values = [1.2 + index * 0.00001 for index in range(count)]
    return pl.DataFrame({
        "timestamp": timestamps,
        "bid_open": values,
        "bid_high": [value + 0.0001 for value in values],
        "bid_low": [value - 0.0001 for value in values],
        "bid_close": [value + 0.00002 for value in values],
        "ask_open": [value + 0.0002 for value in values],
        "ask_high": [value + 0.0003 for value in values],
        "ask_low": [value + 0.0001 for value in values],
        "ask_close": [value + 0.00022 for value in values],
    })


@pytest.mark.parametrize(
    ("session_date", "start", "count"),
    [
        (date(2026, 1, 15), datetime(2026, 1, 14, 22, tzinfo=timezone.utc), 1440),
        (date(2026, 7, 15), datetime(2026, 7, 14, 21, tzinfo=timezone.utc), 1440),
        (date(2026, 3, 29), datetime(2026, 3, 28, 22, tzinfo=timezone.utc), 1380),
        (date(2026, 10, 25), datetime(2026, 10, 24, 21, tzinfo=timezone.utc), 1500),
    ],
)
def test_plot_overnight_range_handles_london_dst(session_date, start, count):
    figure, output_path = chart.plot_overnight_range(session_date, bars(start, count))

    assert output_path == chart.DEFAULT_OUTPUT_DIRECTORY / f"overnight_range_{session_date.isoformat()}.html"
    assert output_path.is_file()
    assert len(figure.data) == 1
    assert list(figure.data[0].open[:2]) == [1.2001, 1.20011]


def test_plot_overnight_range_rejects_incomplete_session():
    start = datetime(2026, 1, 14, 22, tzinfo=timezone.utc)

    with pytest.raises(ValueError, match="every expected one-minute bar"):
        chart.plot_overnight_range(date(2026, 1, 15), bars(start, 1439))


def test_plot_overnight_range_rejects_missing_columns():
    with pytest.raises(ValueError, match="ask_close"):
        chart.plot_overnight_range(date(2026, 1, 15), pl.DataFrame({"timestamp": []}))
