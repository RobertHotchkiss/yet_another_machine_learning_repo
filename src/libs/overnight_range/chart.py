"""Interactive charts for London-time overnight sessions."""

from datetime import date as Date
from datetime import datetime, time, timedelta, timezone
from pathlib import Path
from typing import TypeAlias
from zoneinfo import ZoneInfo

import plotly.graph_objects as go
import polars as pl


LONDON = ZoneInfo("Europe/London")
UTC = timezone.utc
REQUIRED_COLUMNS = {
    "timestamp",
    "bid_open", "bid_high", "bid_low", "bid_close",
    "ask_open", "ask_high", "ask_low", "ask_close",
}
DEFAULT_OUTPUT_DIRECTORY = Path(__file__).resolve().parents[3] / "plots" / "overnight_range"
ChartResult: TypeAlias = tuple[go.Figure, Path]


def _coerce_session_date(value: Date | datetime | str) -> Date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, Date):
        return value
    if isinstance(value, str):
        try:
            return Date.fromisoformat(value)
        except ValueError as exc:
            raise ValueError("date must be an ISO calendar date, for example '2026-09-17'.") from exc
    raise TypeError("date must be a datetime.date, datetime.datetime, or ISO date string.")


def _session_bounds(session_date: Date) -> tuple[datetime, datetime]:
    end_london = datetime.combine(session_date, time(22), tzinfo=LONDON)
    start_london = datetime.combine(session_date - timedelta(days=1), time(22), tzinfo=LONDON)
    return start_london.astimezone(UTC), end_london.astimezone(UTC)


def _validate_frame(frame: pl.DataFrame) -> None:
    missing = sorted(REQUIRED_COLUMNS - set(frame.columns))
    if missing:
        raise ValueError(f"frame is missing required columns: {', '.join(missing)}.")


def _session_frame(frame: pl.DataFrame, start: datetime, end: datetime) -> pl.DataFrame:
    session = frame.filter(pl.col("timestamp").is_between(start, end, closed="left")).sort("timestamp")
    expected_count = int((end - start).total_seconds() // 60)
    timestamps = session.get_column("timestamp").to_list()
    expected_timestamps = [start + timedelta(minutes=index) for index in range(expected_count)]

    # if timestamps != expected_timestamps:
    #     raise ValueError(
    #         "requested London session does not contain every expected one-minute bar "
    #         f"from {start.isoformat()} to {end.isoformat()}."
    #     )
    return session


def plot_overnight_range(date: Date | datetime | str, frame: pl.DataFrame) -> ChartResult:
    """Save and return a mid-price candlestick chart for a 22:00 London session.

    ``date`` is the London calendar date on which the session ends at 22:00.
    """
    _validate_frame(frame)
    session_date = _coerce_session_date(date)
    start, end = _session_bounds(session_date)
    session = _session_frame(frame, start, end).with_columns(
        [
            ((pl.col(f"bid_{field}") + pl.col(f"ask_{field}")) / 2).alias(f"mid_{field}")
            for field in ("open", "high", "low", "close")
        ]
    )

    london_timestamps = [timestamp.astimezone(LONDON) for timestamp in session.get_column("timestamp").to_list()]
    figure = go.Figure(
        go.Candlestick(
            x=london_timestamps,
            open=session.get_column("mid_open").to_list(),
            high=session.get_column("mid_high").to_list(),
            low=session.get_column("mid_low").to_list(),
            close=session.get_column("mid_close").to_list(),
            name="Mid price",
        )
    )
    figure.update_layout(
        title=f"Mid-price candles: {session_date.isoformat()} London session (22:00-22:00)",
        xaxis_title="Time (Europe/London)",
        yaxis_title="Mid price",
        xaxis_rangeslider_visible=False,
        template="plotly_white",
    )

    DEFAULT_OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    output_path = DEFAULT_OUTPUT_DIRECTORY / f"overnight_range_{session_date.isoformat()}.html"
    figure.write_html(output_path, include_plotlyjs=True, full_html=True, auto_open=False)
    return figure, output_path
