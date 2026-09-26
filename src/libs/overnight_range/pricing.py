"""Input-schema detection and price expressions for overnight-range research."""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl


BID_ASK_COLUMNS = {
    "timestamp",
    "bid_open", "bid_high", "bid_low", "bid_close",
    "ask_open", "ask_high", "ask_low", "ask_close",
}
MIDPOINT_COLUMNS = {"timestamp", "open", "high", "low", "close"}


@dataclass(frozen=True)
class PricingSchema:
    """Column mapping for either executable bid/ask or midpoint OHLC bars."""

    mode: str

    def mid_expression(self, field: str) -> pl.Expr:
        if self.mode == "bid_ask":
            return ((pl.col(f"bid_{field}") + pl.col(f"ask_{field}")) / 2).alias(f"_mid_{field}")
        return pl.col(field).alias(f"_mid_{field}")

    def entry_column(self, side: str) -> str:
        if self.mode == "midpoint":
            return "open"
        return "ask_open" if side == "long" else "bid_open"

    def high_column(self, side: str) -> str:
        if self.mode == "midpoint":
            return "high"
        return "bid_high" if side == "long" else "ask_high"

    def low_column(self, side: str) -> str:
        if self.mode == "midpoint":
            return "low"
        return "bid_low" if side == "long" else "ask_low"

    def close_column(self, side: str) -> str:
        if self.mode == "midpoint":
            return "close"
        return "bid_close" if side == "long" else "ask_close"


def resolve_pricing_schema(frame: pl.DataFrame, mode: str = "auto") -> PricingSchema:
    """Validate *frame* and resolve its supported OHLC price convention."""
    if mode not in {"auto", "bid_ask", "midpoint"}:
        raise ValueError("price_mode must be one of: auto, bid_ask, midpoint.")
    columns = set(frame.columns)
    has_bid_ask = BID_ASK_COLUMNS <= columns
    has_midpoint = MIDPOINT_COLUMNS <= columns
    if mode == "auto":
        if has_bid_ask:
            return PricingSchema("bid_ask")
        if has_midpoint:
            return PricingSchema("midpoint")
    elif mode == "bid_ask" and has_bid_ask:
        return PricingSchema("bid_ask")
    elif mode == "midpoint" and has_midpoint:
        return PricingSchema("midpoint")
    required = BID_ASK_COLUMNS if mode == "bid_ask" else MIDPOINT_COLUMNS if mode == "midpoint" else BID_ASK_COLUMNS | MIDPOINT_COLUMNS
    missing = sorted(required - columns)
    raise ValueError(f"frame does not satisfy the requested {mode} OHLC schema; missing columns: {', '.join(missing)}.")
