from datetime import datetime, timedelta, timezone

import polars as pl
import pytest

from libs.overnight_range import OvernightRangeBreakoutConfig, build_overnight_range_breakout_ledger


UTC = timezone.utc


def _bar(timestamp, *, bid_open=1.1000, bid_high=1.1000, bid_low=1.1000, bid_close=1.1000):
    spread = 0.0002
    return {
        "timestamp": timestamp,
        "bid_open": bid_open, "bid_high": bid_high, "bid_low": bid_low, "bid_close": bid_close,
        "ask_open": bid_open + spread, "ask_high": bid_high + spread,
        "ask_low": bid_low + spread, "ask_close": bid_close + spread,
    }


def _session(*, breakout_side="long", both=False, outcome="target") -> pl.DataFrame:
    date = datetime(2026, 1, 5, tzinfo=UTC)
    bars = []
    for minute in range(7 * 60):
        bars.append(_bar(date + timedelta(minutes=minute), bid_high=1.0999, bid_low=1.0991))
    # With a zero buffer, midpoint range is 1.0992 to 1.1000. The 07:00 bar
    # triggers the requested side; entry is the 07:01 executable open.
    trigger = date + timedelta(hours=7)
    if both:
        bars.append(_bar(trigger, bid_high=1.1012, bid_low=1.0988))
    elif breakout_side == "long":
        bars.append(_bar(trigger, bid_high=1.1010, bid_low=1.1002))
    else:
        bars.append(_bar(trigger, bid_high=1.0997, bid_low=1.0989))
    entry = trigger + timedelta(minutes=1)
    if outcome == "target":
        if breakout_side == "long":
            bars.append(_bar(entry, bid_open=1.1010, bid_high=1.1020, bid_low=1.1007))
        else:
            bars.append(_bar(entry, bid_open=1.0990, bid_high=1.0993, bid_low=1.0979))
    elif outcome == "both":
        bars.append(_bar(entry, bid_open=1.1010, bid_high=1.1022, bid_low=1.0998))
    else:
        bars.append(_bar(entry, bid_open=1.1010, bid_high=1.1011, bid_low=1.1008, bid_close=1.1009))
    return pl.DataFrame(bars)


def _config(**overrides):
    return OvernightRangeBreakoutConfig(buffer_pips=0, **overrides)


def test_long_breakout_uses_next_ask_open_and_bid_target_exit():
    ledger = build_overnight_range_breakout_ledger(_session(), _config())
    row = ledger.row(0, named=True)

    assert row["side"] == "long"
    assert row["status"] == "entered"
    assert row["entry_price"] == pytest.approx(1.1012)
    assert row["exit_reason"] == "target"
    assert row["exit_price"] == pytest.approx(1.1020)
    assert row["gross_pnl_pips"] == pytest.approx(8)
    assert row["r_return"] == pytest.approx(row["r_multiple"])
    assert row["range_observed_minutes"] == 420
    assert row["range_coverage"] == 1


def test_short_breakout_uses_bid_entry_and_ask_target_exit():
    ledger = build_overnight_range_breakout_ledger(_session(breakout_side="short"), _config())
    row = ledger.row(0, named=True)

    assert row["side"] == "short"
    assert row["entry_price"] == pytest.approx(1.0990)
    assert row["exit_reason"] == "target"
    assert row["exit_price"] == pytest.approx(1.0982)


def test_dual_breakout_is_ambiguous_and_has_no_trade():
    row = build_overnight_range_breakout_ledger(_session(both=True), _config()).row(0, named=True)

    assert row["status"] == "ambiguous_first_break"
    assert row["side"] is None
    assert row["entry_price"] is None
    assert row["exit_reason"] == "not_entered_ambiguous_first_break"


def test_same_bar_stop_and_target_records_conservative_stop():
    row = build_overnight_range_breakout_ledger(_session(outcome="both"), _config()).row(0, named=True)

    assert row["exit_reason"] == "stop"
    assert row["exit_price"] == pytest.approx(row["stop_price"])


def test_partial_session_is_retained_and_time_exit_uses_last_available_close():
    row = build_overnight_range_breakout_ledger(_session(outcome="time"), _config()).row(0, named=True)

    assert row["trade_observed_minutes"] == 2
    assert row["trade_coverage"] == pytest.approx(2 / 600)
    assert row["exit_reason"] == "time"
    assert row["exit_price"] == pytest.approx(1.1009)


def test_missing_immediate_entry_bar_is_reported_without_dropping_session():
    entry_time = datetime(2026, 1, 5, 7, 1, tzinfo=UTC)
    frame = _session().filter(pl.col("timestamp") != entry_time)
    row = build_overnight_range_breakout_ledger(frame, _config()).row(0, named=True)

    assert row["status"] == "missing_entry_bar"
    assert row["entry_price"] is None
    assert row["range_observed_minutes"] == 420
    assert row["exit_reason"] == "not_entered_missing_entry_bar"


def test_required_columns_and_config_values_are_validated():
    with pytest.raises(ValueError, match="ask_close"):
        build_overnight_range_breakout_ledger(pl.DataFrame({"timestamp": []}))
    with pytest.raises(ValueError, match="pip_size"):
        OvernightRangeBreakoutConfig(pip_size=0)
