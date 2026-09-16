"""Fail-closed order decisions for the Binance-data / Deribit-execution runner."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Literal


Signal = Literal["long", "short", "abstain"]
PositionState = Literal["long", "short", "flat"]
OrderSideName = Literal["buy", "sell"]

ENTRY_USD = Decimal("10")
REVERSAL_USD = Decimal("20")


@dataclass(frozen=True)
class MarketOrderIntent:
    """A single Deribit market-order request expressed in Deribit USD units."""

    signal: Signal
    side: OrderSideName
    amount_usd: Decimal
    reduce_only: bool
    target_position: PositionState
    reason: str


@dataclass(frozen=True)
class Decision:
    """The result of processing one model signal."""

    intent: MarketOrderIntent | None
    reason: str


class DeribitSignalState:
    """Tracks confirmed strategy exposure and serializes live order submission.

    The runner only changes ``confirmed_position`` after receiving a complete
    fill. Any rejection or partial fill halts automated execution; it must be
    reviewed rather than guessed through with another order.
    """

    def __init__(self) -> None:
        self.confirmed_position: PositionState = "flat"
        self.pending: MarketOrderIntent | None = None
        self.halted_reason: str | None = None

    @property
    def halted(self) -> bool:
        return self.halted_reason is not None

    def decide(self, signal: Signal) -> Decision:
        if signal not in ("long", "short", "abstain"):
            raise ValueError(f"Unknown model signal: {signal!r}")
        if self.halted:
            return Decision(None, "halted")
        if self.pending is not None:
            return Decision(None, "pending_order")

        if signal == "abstain":
            if self.confirmed_position == "flat":
                return Decision(None, "already_flat")
            intent = MarketOrderIntent(
                signal=signal,
                side="sell" if self.confirmed_position == "long" else "buy",
                amount_usd=ENTRY_USD,
                reduce_only=True,
                target_position="flat",
                reason="abstention_close",
            )
        elif self.confirmed_position == signal:
            return Decision(None, "repeated_directional_signal")
        elif self.confirmed_position == "flat":
            intent = MarketOrderIntent(
                signal=signal,
                side="buy" if signal == "long" else "sell",
                amount_usd=ENTRY_USD,
                reduce_only=False,
                target_position=signal,
                reason="initial_entry",
            )
        else:
            intent = MarketOrderIntent(
                signal=signal,
                side="buy" if signal == "long" else "sell",
                amount_usd=REVERSAL_USD,
                reduce_only=False,
                target_position=signal,
                reason="direction_reversal",
            )

        self.pending = intent
        return Decision(intent, intent.reason)

    def mark_filled(self, filled_amount_usd: Decimal) -> None:
        """Confirm a complete fill, or stop automated trading on a partial one."""
        if self.pending is None:
            self.halt("fill_without_pending_order")
            return
        if filled_amount_usd != self.pending.amount_usd:
            self.halt(
                f"partial_or_unexpected_fill: expected {self.pending.amount_usd}, "
                f"received {filled_amount_usd}",
            )
            return
        self.confirmed_position = self.pending.target_position
        self.pending = None

    def mark_rejected(self, reason: str) -> None:
        self.halt(f"order_rejected: {reason}")

    def halt(self, reason: str) -> None:
        self.pending = None
        self.halted_reason = reason

