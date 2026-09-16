import pytest


live_range_bars_rs = pytest.importorskip("live_range_bars_rs")


def test_renko_ish_uses_strict_boundary_and_keeps_the_trigger_trade():
    aggregator = live_range_bars_rs.RenkoIshAggregator(0.001)
    assert aggregator.process_trade(1, 100.0, 2.0, 200.0, False, 1) is None
    assert aggregator.process_trade(2, 100.1, 2.0, 200.2, False, 2) is None
    bar = aggregator.process_trade(3, 100.1001, 2.0, 200.2002, True, 3)
    assert bar.first_trade_id == 1
    assert bar.last_trade_id == 3
    assert bar.total_bought_volume == 4.0
    assert bar.total_sold_volume == 2.0


def test_renko_ish_snapshot_preserves_active_bar_and_deduplicates_ids():
    aggregator = live_range_bars_rs.RenkoIshAggregator(0.001)
    aggregator.process_trade(1, 100.0, 1.0, 100.0, False, 1)
    restored = live_range_bars_rs.RenkoIshAggregator.from_snapshot(aggregator.snapshot())
    assert restored.process_trade(1, 200.0, 1.0, 200.0, False, 2) is None
    assert restored.process_trade(2, 100.2, 1.0, 100.2, True, 3).last_trade_id == 2
