from decimal import Decimal

from range_bars_ml.live_execution import DeribitSignalState, ENTRY_USD, REVERSAL_USD


def test_first_directional_signal_enters_with_ten_usd():
    state = DeribitSignalState()

    decision = state.decide("long")

    assert decision.reason == "initial_entry"
    assert decision.intent is not None
    assert decision.intent.side == "buy"
    assert decision.intent.amount_usd == ENTRY_USD == Decimal("10")
    assert not decision.intent.reduce_only
    state.mark_filled(Decimal("10"))
    assert state.confirmed_position == "long"


def test_repeated_signal_does_not_submit_another_order():
    state = DeribitSignalState()
    state.decide("short")
    state.mark_filled(Decimal("10"))

    decision = state.decide("short")

    assert decision.intent is None
    assert decision.reason == "repeated_directional_signal"


def test_opposite_signal_reverses_with_twenty_usd():
    state = DeribitSignalState()
    state.decide("long")
    state.mark_filled(Decimal("10"))

    decision = state.decide("short")

    assert decision.intent is not None
    assert decision.intent.side == "sell"
    assert decision.intent.amount_usd == REVERSAL_USD == Decimal("20")
    assert not decision.intent.reduce_only
    state.mark_filled(Decimal("20"))
    assert state.confirmed_position == "short"


def test_abstention_flattens_with_reduce_only_ten_usd_order():
    state = DeribitSignalState()
    state.decide("short")
    state.mark_filled(Decimal("10"))

    decision = state.decide("abstain")

    assert decision.intent is not None
    assert decision.intent.side == "buy"
    assert decision.intent.amount_usd == Decimal("10")
    assert decision.intent.reduce_only
    state.mark_filled(Decimal("10"))
    assert state.confirmed_position == "flat"


def test_pending_order_defers_new_signal_and_partial_fill_halts():
    state = DeribitSignalState()
    state.decide("long")

    assert state.decide("short").reason == "pending_order"
    state.mark_filled(Decimal("5"))

    assert state.halted
    assert "partial_or_unexpected_fill" in state.halted_reason
    assert state.decide("short").reason == "halted"


def test_rejection_halts_execution():
    state = DeribitSignalState()
    state.decide("long")
    state.mark_rejected("insufficient_margin")

    assert state.halted
    assert state.pending is None
    assert state.decide("long").reason == "halted"
