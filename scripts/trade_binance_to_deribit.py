"""Score Binance range bars and execute confirmed signals on Deribit mainnet.

Binance is the *only* data source for range bars and model features. Orders
are submitted only to BTC-PERPETUAL.DERIBIT after explicit live arming.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import deque
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any, Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from nautilus_trader.adapters.binance.common.enums import BinanceAccountType
from nautilus_trader.adapters.binance.config import BinanceDataClientConfig
from nautilus_trader.adapters.binance.factories import BinanceLiveDataClientFactory
from nautilus_trader.adapters.deribit.config import DeribitExecClientConfig
from nautilus_trader.adapters.deribit.factories import DeribitLiveExecClientFactory
from nautilus_trader.config import (
    InstrumentProviderConfig,
    LiveExecEngineConfig,
    RoutingConfig,
    StrategyConfig,
    TradingNodeConfig,
)
from nautilus_trader.core.nautilus_pyo3 import DeribitEnvironment, DeribitProductType
from nautilus_trader.live.node import TradingNode
from nautilus_trader.model.data import TradeTick
from nautilus_trader.model.enums import AggressorSide, OrderSide
from nautilus_trader.model.events import (
    OrderCanceled,
    OrderDenied,
    OrderExpired,
    OrderFilled,
    OrderRejected,
    PositionChanged,
)
from nautilus_trader.model.identifiers import InstrumentId, TraderId
from nautilus_trader.trading.strategy import Strategy

from range_bars_ml import SelectiveSignalModel
from range_bars_ml.live_execution import DeribitSignalState, MarketOrderIntent
from range_bars_ml.live_support import LiveScoreboard, LiveStateStore, completed_bar_row, latest_feature_frame


DEFAULT_MODEL = PROJECT_ROOT / "src" / "libs" / "notebooks" / "range_bar_model.joblib"
BINANCE_INSTRUMENT = InstrumentId.from_str("BTCUSDT-PERP.BINANCE")
DERIBIT_INSTRUMENT = InstrumentId.from_str("BTC-PERPETUAL.DERIBIT")
BAR_BUFFER_SIZE = 64


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Use public Binance BTCUSDT range bars to execute $10/$20 Deribit BTC perpetual market orders.",
    )
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "data" / "live")
    parser.add_argument("--range-percent", type=float, default=0.001)
    parser.add_argument("--warmup-minutes", type=int, default=30)
    parser.add_argument(
        "--live-execution",
        action="store_true",
        help="Required acknowledgement before connecting the Deribit mainnet execution client.",
    )
    return parser


def require_live_execution_credentials(args: argparse.Namespace) -> None:
    if not args.live_execution:
        raise RuntimeError("Refusing to create a live Deribit execution node without --live-execution.")
    missing = [name for name in ("DERIBIT_API_KEY", "DERIBIT_API_SECRET") if not os.environ.get(name)]
    if missing:
        raise RuntimeError(f"Missing required Deribit mainnet credentials: {', '.join(missing)}")


def tick_arguments(tick: TradeTick) -> tuple[int, float, float, float, bool, int] | None:
    """Map a Binance trade representation to the Rust aggregator input."""
    if tick.aggressor_side == AggressorSide.NO_AGGRESSOR:
        return None
    price, quantity = float(tick.price), float(tick.size)
    is_buyer_maker = tick.aggressor_side == AggressorSide.SELLER
    return int(str(tick.trade_id)), price, quantity, price * quantity, is_buyer_maker, tick.ts_event // 1_000


def _deserialize_bar(row: dict[str, Any]) -> dict[str, Any]:
    return {
        **row,
        "timestamp": datetime.fromisoformat(row["timestamp"]),
        "close_time": datetime.fromisoformat(row["close_time"]),
    }


class BinanceDeribitConfig(StrategyConfig, frozen=True):
    data_instrument_id: InstrumentId
    execution_instrument_id: InstrumentId
    model_path: Path
    output_dir: Path
    range_percent: float = 0.001
    warmup_minutes: int = 30


class BinanceDeribitStrategy(Strategy):
    """Rust bars + Python model from Binance, guarded Deribit market orders."""

    def __init__(self, config: BinanceDeribitConfig) -> None:
        super().__init__(config)
        self.execution_instrument = None
        self.model = SelectiveSignalModel.load(config.model_path)
        self.store = LiveStateStore(config.output_dir)
        self.bars: deque[dict[str, Any]] = deque(maxlen=BAR_BUFFER_SIZE)
        self.aggregator: Any = None
        self.scoreboard = LiveScoreboard(self.model.metadata.get("abstention_penalty", -0.001))
        self.execution_state = DeribitSignalState()
        self.latest_signal: str = "abstain"
        self.armed = False

    def on_start(self) -> None:
        try:
            from live_range_bars_rs import RenkoIshAggregator
        except ImportError as error:
            raise RuntimeError(
                "live_range_bars_rs is not installed; run `maturin develop --release` in crates/live_range_bars_rs.",
            ) from error
        state = self.store.load()
        if state:
            self.aggregator = RenkoIshAggregator.from_snapshot(state["rust_snapshot"])
            self.bars.extend(_deserialize_bar(row) for row in state["bars"])
        else:
            self.aggregator = RenkoIshAggregator(self.config.range_percent)

        self.execution_instrument = self.cache.instrument(self.config.execution_instrument_id)
        if self.execution_instrument is None:
            raise RuntimeError(f"Missing execution instrument in cache: {self.config.execution_instrument_id}")
        self._require_flat_startup_account()
        self.armed = True
        start = self._clock.utc_now() - timedelta(minutes=self.config.warmup_minutes)
        self.request_trade_ticks(self.config.data_instrument_id, start=start)
        self.subscribe_trade_ticks(self.config.data_instrument_id)
        self.store.event(
            "started",
            {
                "data_instrument": str(self.config.data_instrument_id),
                "execution_instrument": str(self.config.execution_instrument_id),
                "model_path": str(self.config.model_path),
                "entry_usd": "10",
                "reversal_usd": "20",
            },
        )

    def _require_flat_startup_account(self) -> None:
        open_orders = self.cache.orders_open(instrument_id=self.config.execution_instrument_id)
        open_positions = self.cache.positions_open(instrument_id=self.config.execution_instrument_id)
        if open_orders or open_positions or not self.portfolio.is_flat(self.config.execution_instrument_id):
            raise RuntimeError(
                "Refusing to arm: Deribit BTC-PERPETUAL must be flat with no open orders. "
                "Cancel or close existing exposure manually before starting.",
            )

    def on_historical_data(self, data: Any) -> None:
        for item in data if isinstance(data, Iterable) and not isinstance(data, TradeTick) else (data,):
            if isinstance(item, TradeTick):
                self._process_tick(item)

    def on_trade_tick(self, tick: TradeTick) -> None:
        self._process_tick(tick)

    def _process_tick(self, tick: TradeTick) -> None:
        arguments = tick_arguments(tick)
        if arguments is None:
            self.store.event("skipped_trade", {"reason": "no_aggressor", "trade_id": str(tick.trade_id)})
            return
        closed = self.aggregator.process_trade(*arguments)
        if closed is None:
            return
        bar = completed_bar_row(closed)
        self.bars.append(bar)
        self.store.save(rust_snapshot=self.aggregator.snapshot(), bars=self.bars)
        feature = latest_feature_frame(self.bars, self.model)
        if feature is None:
            self.store.event("warmup_bar", bar)
            return
        prediction = self.model.predict(feature).row(0, named=True)
        probability, signal = float(prediction["probability_up"]), str(prediction["signal"])
        settled = self.scoreboard.settle_then_record(bar, probability, signal)
        self.latest_signal = signal
        event = {
            **bar,
            "data_instrument": str(self.config.data_instrument_id),
            "execution_instrument": str(self.config.execution_instrument_id),
            "probability_up": probability,
            "signal": signal,
        }
        self.store.event("signal", event)
        if settled:
            self.store.event("scored_signal", settled)
            comparison = self.scoreboard.comparison()
            self.store.event("scoreboard", comparison)
            self.log.info(
                f"model score={comparison['model']['score']:.3f}; "
                f"up={comparison['always_up']['score']:.3f}; "
                f"down={comparison['always_down']['score']:.3f}; {settled}",
            )
        self._apply_latest_signal()

    def _apply_latest_signal(self) -> None:
        if not self.armed or self.execution_instrument is None:
            return
        decision = self.execution_state.decide(self.latest_signal)  # type: ignore[arg-type]
        self.store.event(
            "execution_decision",
            {
                "signal": self.latest_signal,
                "reason": decision.reason,
                "confirmed_position": self.execution_state.confirmed_position,
                "pending": self.execution_state.pending is not None,
                "halted_reason": self.execution_state.halted_reason,
            },
        )
        if decision.intent is not None:
            self._submit_market_order(decision.intent)

    def _submit_market_order(self, intent: MarketOrderIntent) -> None:
        side = OrderSide.BUY if intent.side == "buy" else OrderSide.SELL
        quantity = self.execution_instrument.make_qty(intent.amount_usd)
        order = self.order_factory.market(
            self.config.execution_instrument_id,
            side,
            quantity,
            reduce_only=intent.reduce_only,
        )
        try:
            self.submit_order(order)
        except Exception as error:
            self.execution_state.halt(f"submit_failed: {error}")
            self.store.event("execution_halted", {"reason": self.execution_state.halted_reason})
            raise
        self.store.event(
            "submitted_order",
            {
                "signal": intent.signal,
                "reason": intent.reason,
                "side": intent.side,
                "amount_usd": str(intent.amount_usd),
                "reduce_only": intent.reduce_only,
                "target_position": intent.target_position,
                "instrument": str(self.config.execution_instrument_id),
            },
        )

    def on_order_filled(self, event: OrderFilled) -> None:
        self.store.event("fill", event.to_dict())
        self.execution_state.mark_filled(Decimal(str(event.last_qty)))
        if self.execution_state.halted:
            self.store.event("execution_halted", {"reason": self.execution_state.halted_reason})
            return
        self._apply_latest_signal()

    def on_order_rejected(self, event: OrderRejected) -> None:
        self.store.event("rejected", event.to_dict())
        self.execution_state.mark_rejected(str(event))
        self.store.event("execution_halted", {"reason": self.execution_state.halted_reason})

    def on_order_denied(self, event: OrderDenied) -> None:
        self.store.event("denied", event.to_dict())
        self.execution_state.mark_rejected(str(event))
        self.store.event("execution_halted", {"reason": self.execution_state.halted_reason})

    def on_order_canceled(self, event: OrderCanceled) -> None:
        self.store.event("canceled", event.to_dict())
        self.execution_state.mark_rejected(str(event))
        self.store.event("execution_halted", {"reason": self.execution_state.halted_reason})

    def on_order_expired(self, event: OrderExpired) -> None:
        self.store.event("expired", event.to_dict())
        self.execution_state.mark_rejected(str(event))
        self.store.event("execution_halted", {"reason": self.execution_state.halted_reason})

    def on_position_changed(self, event: PositionChanged) -> None:
        if (
            event.instrument_id != self.config.execution_instrument_id
            or self.execution_state.halted
            or self.execution_state.pending is not None
        ):
            return
        actual = self._actual_position_state()
        if actual != self.execution_state.confirmed_position:
            self.execution_state.halt(
                f"unexpected_position: expected {self.execution_state.confirmed_position}, observed {actual}",
            )
            self.store.event("execution_halted", {"reason": self.execution_state.halted_reason})

    def _actual_position_state(self) -> str:
        if self.portfolio.is_flat(self.config.execution_instrument_id):
            return "flat"
        if self.portfolio.is_net_long(self.config.execution_instrument_id):
            return "long"
        if self.portfolio.is_net_short(self.config.execution_instrument_id):
            return "short"
        return "unknown"

    def on_stop(self) -> None:
        if self.aggregator is not None:
            self.store.save(rust_snapshot=self.aggregator.snapshot(), bars=self.bars)
        self.store.event(
            "stopped",
            {"metrics": self.scoreboard.comparison(), "halted_reason": self.execution_state.halted_reason},
        )


def build_node(args: argparse.Namespace) -> tuple[TradingNode, BinanceDeribitStrategy]:
    require_live_execution_credentials(args)
    if not args.model_path.exists():
        raise FileNotFoundError(args.model_path)
    if args.range_percent <= 0 or args.warmup_minutes <= 0:
        raise ValueError("range percent and warmup minutes must be positive")
    config = TradingNodeConfig(
        trader_id=TraderId("BINANCE-DERIBIT-RANGE-001"),
        exec_engine=LiveExecEngineConfig(
            reconciliation=True,
            reconciliation_instrument_ids=[DERIBIT_INSTRUMENT],
        ),
        data_clients={
            "BINANCE": BinanceDataClientConfig(
                account_type=BinanceAccountType.USDT_FUTURES,
                instrument_provider=InstrumentProviderConfig(load_ids=frozenset({BINANCE_INSTRUMENT})),
                routing=RoutingConfig(default=True),
                use_agg_trade_ticks=True,
            ),
        },
        exec_clients={
            "DERIBIT": DeribitExecClientConfig(
                instrument_provider=InstrumentProviderConfig(load_ids=frozenset({DERIBIT_INSTRUMENT})),
                routing=RoutingConfig(default=True),
                product_types=(DeribitProductType.FUTURE,),
                environment=DeribitEnvironment.MAINNET,
            ),
        },
    )
    node = TradingNode(config=config)
    node.add_data_client_factory("BINANCE", BinanceLiveDataClientFactory)
    node.add_exec_client_factory("DERIBIT", DeribitLiveExecClientFactory)
    strategy = BinanceDeribitStrategy(BinanceDeribitConfig(
        data_instrument_id=BINANCE_INSTRUMENT,
        execution_instrument_id=DERIBIT_INSTRUMENT,
        model_path=args.model_path,
        output_dir=args.output_dir,
        range_percent=args.range_percent,
        warmup_minutes=args.warmup_minutes,
    ))
    node.trader.add_strategy(strategy)
    return node, strategy


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    node, _ = build_node(args)
    node.build()
    try:
        node.run(raise_exception=True)
    finally:
        node.stop()
        node.dispose()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
