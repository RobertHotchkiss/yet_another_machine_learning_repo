"""London overnight-range breakout research ledger."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo

import polars as pl


LONDON = ZoneInfo("Europe/London")
REQUIRED_COLUMNS = {
    "timestamp",
    "bid_open", "bid_high", "bid_low", "bid_close",
    "ask_open", "ask_high", "ask_low", "ask_close",
}


@dataclass(frozen=True)
class OvernightRangeBreakoutConfig:
    """Parameters for a first-break London overnight-range strategy.

    All times are interpreted in ``Europe/London``.  ``buffer_pips`` is an
    absolute price buffer; stop and target distances are multiples of the
    completed midpoint overnight range.
    """

    range_start: time = time(0, 0)
    range_end: time = time(7, 0)
    exit_time: time = time(17, 0)
    pip_size: float = 0.0001
    buffer_pips: float = 5.0
    stop_range_multiple: float = 1.0
    target_range_multiple: float = 1.0

    def __post_init__(self) -> None:
        if not self.range_start < self.range_end < self.exit_time:
            raise ValueError("range_start, range_end, and exit_time must be strictly increasing.")
        if self.pip_size <= 0:
            raise ValueError("pip_size must be positive.")
        if self.buffer_pips < 0:
            raise ValueError("buffer_pips cannot be negative.")
        if self.stop_range_multiple <= 0 or self.target_range_multiple <= 0:
            raise ValueError("stop_range_multiple and target_range_multiple must be positive.")


def _validate_frame(frame: pl.DataFrame) -> None:
    missing = sorted(REQUIRED_COLUMNS - set(frame.columns))
    if missing:
        raise ValueError(f"frame is missing required columns: {', '.join(missing)}.")
    timestamp_type = frame.schema["timestamp"]
    if not isinstance(timestamp_type, pl.Datetime) or timestamp_type.time_zone is None:
        raise ValueError("timestamp must be a timezone-aware Polars Datetime column.")


def _bounds(session_date: date, config: OvernightRangeBreakoutConfig) -> tuple[datetime, datetime, datetime]:
    return tuple(
        datetime.combine(session_date, session_time, tzinfo=LONDON)
        for session_time in (config.range_start, config.range_end, config.exit_time)
    )  # type: ignore[return-value]


def _empty_ledger() -> pl.DataFrame:
    return pl.DataFrame(schema={
        "session_date": pl.Date,
        "range_start": pl.Datetime(time_zone="Europe/London"),
        "range_end": pl.Datetime(time_zone="Europe/London"),
        "exit_time": pl.Datetime(time_zone="Europe/London"),
        "range_expected_minutes": pl.Int64,
        "range_observed_minutes": pl.Int64,
        "range_coverage": pl.Float64,
        "trade_expected_minutes": pl.Int64,
        "trade_observed_minutes": pl.Int64,
        "trade_coverage": pl.Float64,
        "range_high": pl.Float64,
        "range_low": pl.Float64,
        "range_open": pl.Float64,
        "range_close": pl.Float64,
        "range_size": pl.Float64,
        "range_pips": pl.Float64,
        "long_trigger": pl.Float64,
        "short_trigger": pl.Float64,
        "status": pl.String,
        "side": pl.String,
        "breakout_timestamp": pl.Datetime(time_zone="UTC"),
        "entry_timestamp": pl.Datetime(time_zone="UTC"),
        "entry_price": pl.Float64,
        "stop_price": pl.Float64,
        "target_price": pl.Float64,
        "exit_timestamp": pl.Datetime(time_zone="UTC"),
        "exit_price": pl.Float64,
        "exit_reason": pl.String,
        "gross_pnl": pl.Float64,
        "gross_pnl_pips": pl.Float64,
        "r_multiple": pl.Float64,
        "r_return": pl.Float64,
    })


def build_overnight_range_breakout_ledger(
    frame: pl.DataFrame,
    config: OvernightRangeBreakoutConfig = OvernightRangeBreakoutConfig(),
) -> pl.DataFrame:
    """Return one auditable first-breakout row for each observed London date.

    The range uses midpoint high/low values. Longs enter on the following
    contiguous minute's ask open and exit on bid prices; shorts use the mirror
    convention. Partial sessions remain in the ledger with coverage fields.
    """
    _validate_frame(frame)
    if frame.is_empty():
        return _empty_ledger()

    bars = frame.sort("timestamp").with_columns([
        ((pl.col("bid_high") + pl.col("ask_high")) / 2).alias("_mid_high"),
        ((pl.col("bid_low") + pl.col("ask_low")) / 2).alias("_mid_low"),
        pl.col("timestamp").dt.convert_time_zone("Europe/London").dt.date().alias("_session_date"),
    ])
    sessions = bars.partition_by("_session_date", as_dict=False, maintain_order=True)
    rows: list[dict[str, object]] = []

    for session in sessions:
        session_date = session.item(0, "_session_date")
        range_start, range_end, exit_time = _bounds(session_date, config)
        range_start_utc = range_start.astimezone(ZoneInfo("UTC"))
        range_end_utc = range_end.astimezone(ZoneInfo("UTC"))
        exit_time_utc = exit_time.astimezone(ZoneInfo("UTC"))
        values = session.to_dicts()
        range_bars = [bar for bar in values if range_start_utc <= bar["timestamp"] < range_end_utc]
        trade_bars = [bar for bar in values if range_end_utc <= bar["timestamp"] < exit_time_utc]
        expected_range = int((range_end_utc - range_start_utc).total_seconds() // 60)
        expected_trade = int((exit_time_utc - range_end_utc).total_seconds() // 60)
        row: dict[str, object] = {
            "session_date": session_date,
            "range_start": range_start,
            "range_end": range_end,
            "exit_time": exit_time,
            "range_expected_minutes": expected_range,
            "range_observed_minutes": len(range_bars),
            "range_coverage": len(range_bars) / expected_range,
            "trade_expected_minutes": expected_trade,
            "trade_observed_minutes": len(trade_bars),
            "trade_coverage": len(trade_bars) / expected_trade,
            "range_high": None, "range_low": None, "range_open": None, "range_close": None,
            "range_size": None, "range_pips": None,
            "long_trigger": None, "short_trigger": None,
            "status": "no_range_data", "side": None, "breakout_timestamp": None,
            "entry_timestamp": None, "entry_price": None, "stop_price": None, "target_price": None,
            "exit_timestamp": None, "exit_price": None, "exit_reason": "not_entered_no_range_data",
            "gross_pnl": None, "gross_pnl_pips": None, "r_multiple": None, "r_return": None,
        }
        if not range_bars:
            rows.append(row)
            continue

        high = max(float(bar["_mid_high"]) for bar in range_bars)
        low = min(float(bar["_mid_low"]) for bar in range_bars)
        range_open = (float(range_bars[0]["bid_open"]) + float(range_bars[0]["ask_open"])) / 2
        range_close = (float(range_bars[-1]["bid_close"]) + float(range_bars[-1]["ask_close"])) / 2
        size = high - low
        buffer = config.buffer_pips * config.pip_size
        long_trigger, short_trigger = high + buffer, low - buffer
        row.update({
            "range_high": high, "range_low": low, "range_open": range_open, "range_close": range_close,
            "range_size": size,
            "range_pips": size / config.pip_size,
            "long_trigger": long_trigger, "short_trigger": short_trigger,
            "status": "no_breakout", "exit_reason": "not_entered_no_breakout",
        })
        if size <= 0:
            row.update({"status": "zero_range", "exit_reason": "not_entered_zero_range"})
            rows.append(row)
            continue

        # A 16:59 trigger cannot be filled at a following open before the
        # configured 16:59 close-out, so 16:58 is the final eligible trigger.
        final_trigger_time = exit_time_utc - timedelta(minutes=1)
        breakout: dict[str, object] | None = None
        side: str | None = None
        for bar in trade_bars:
            if bar["timestamp"] >= final_trigger_time:
                break
            hit_long = float(bar["ask_high"]) >= long_trigger
            hit_short = float(bar["bid_low"]) <= short_trigger
            if hit_long and hit_short:
                row.update({
                    "status": "ambiguous_first_break",
                    "breakout_timestamp": bar["timestamp"],
                    "exit_reason": "not_entered_ambiguous_first_break",
                })
                break
            if hit_long or hit_short:
                breakout, side = bar, "long" if hit_long else "short"
                break
        if breakout is None:
            rows.append(row)
            continue
        if side is None:  # An ambiguous candle has already updated the row.
            rows.append(row)
            continue

        breakout_time = breakout["timestamp"]
        entry_time = breakout_time + timedelta(minutes=1)
        entry_bar = next((bar for bar in trade_bars if bar["timestamp"] == entry_time), None)
        row.update({"side": side, "breakout_timestamp": breakout_time})
        if entry_bar is None:
            row.update({"status": "missing_entry_bar", "exit_reason": "not_entered_missing_entry_bar"})
            rows.append(row)
            continue

        entry = float(entry_bar["ask_open"] if side == "long" else entry_bar["bid_open"])
        risk = size * config.stop_range_multiple
        reward = size * config.target_range_multiple
        stop = entry - risk if side == "long" else entry + risk
        target = entry + reward if side == "long" else entry - reward
        row.update({
            "status": "entered", "entry_timestamp": entry_time, "entry_price": entry,
            "stop_price": stop, "target_price": target,
        })

        active_bars = [bar for bar in trade_bars if bar["timestamp"] >= entry_time]
        exit_bar: dict[str, object] | None = None
        exit_price: float | None = None
        exit_reason: str | None = None
        for bar in active_bars:
            if side == "long":
                stop_hit = float(bar["bid_low"]) <= stop
                target_hit = float(bar["bid_high"]) >= target
            else:
                stop_hit = float(bar["ask_high"]) >= stop
                target_hit = float(bar["ask_low"]) <= target
            if stop_hit:  # Conservative when both levels are reached in one bar.
                exit_bar, exit_price, exit_reason = bar, stop, "stop"
                break
            if target_hit:
                exit_bar, exit_price, exit_reason = bar, target, "target"
                break
        if exit_bar is None and active_bars:
            exit_bar = active_bars[-1]
            exit_price = float(exit_bar["bid_close"] if side == "long" else exit_bar["ask_close"])
            exit_reason = "time"

        if exit_bar is not None and exit_price is not None:
            pnl = exit_price - entry if side == "long" else entry - exit_price
            row.update({
                "exit_timestamp": exit_bar["timestamp"], "exit_price": exit_price,
                "exit_reason": exit_reason, "gross_pnl": pnl,
                "gross_pnl_pips": pnl / config.pip_size, "r_multiple": pnl / risk,
                "r_return": pnl / risk,
            })
        else:
            row.update({"status": "missing_exit_bar", "exit_reason": "missing_exit_bar"})
        rows.append(row)

    return pl.DataFrame(rows).cast(_empty_ledger().schema, strict=False).sort("session_date")
