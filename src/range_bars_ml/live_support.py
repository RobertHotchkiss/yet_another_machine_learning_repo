"""Shared Python-side helpers for the live Nautilus sandbox runner."""

from __future__ import annotations

import json
import math
import os
from collections import deque
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import polars as pl

from .features import FeatureConfig, build_features
from .model import SelectiveSignalModel, signal_metrics


RAW_BAR_COLUMNS = (
    "first_trade_id", "last_trade_id", "timestamp", "close_time", "open", "high", "low", "close",
    "total_sold_volume", "total_bought_volume", "total_sold_dollar_volume", "total_bought_dollar_volume",
)


def completed_bar_row(bar: Any) -> dict[str, Any]:
    """Convert the Rust extension's completed bar object into the training schema."""
    return {
        "first_trade_id": bar.first_trade_id,
        "last_trade_id": bar.last_trade_id,
        "timestamp": datetime.fromtimestamp(bar.timestamp_us / 1_000_000, tz=UTC),
        "close_time": datetime.fromtimestamp(bar.close_time_us / 1_000_000, tz=UTC),
        "open": bar.open, "high": bar.high, "low": bar.low, "close": bar.close,
        "total_sold_volume": bar.total_sold_volume, "total_bought_volume": bar.total_bought_volume,
        "total_sold_dollar_volume": bar.total_sold_dollar_volume,
        "total_bought_dollar_volume": bar.total_bought_dollar_volume,
    }


def latest_feature_frame(rows: Iterable[dict[str, Any]], model: SelectiveSignalModel) -> pl.DataFrame | None:
    """Return one finite, model-compatible live row, or ``None`` during warm-up."""
    frame = pl.DataFrame(list(rows))
    if frame.is_empty():
        return None
    feature_config = FeatureConfig(**model.metadata["feature_config"])
    featured = build_features(frame, feature_config)
    available = featured.select(model.feature_columns).drop_nulls()
    for name in model.feature_columns:
        if available.schema[name].is_numeric():
            available = available.filter(pl.col(name).is_finite())
    return available.tail(1) if available.height else None


@dataclass
class PendingSignal:
    probability_up: float
    signal: str
    bar_close_time: str


class LiveScoreboard:
    """Scores a completed prediction when the following completed bar reveals its target."""

    def __init__(self, abstention_penalty: float) -> None:
        self.abstention_penalty = abstention_penalty
        self.pending: PendingSignal | None = None
        self.rows: list[dict[str, Any]] = []

    def settle_then_record(self, bar: dict[str, Any], probability_up: float, signal: str) -> dict[str, Any] | None:
        settled: dict[str, Any] | None = None
        if self.pending is not None:
            actual_up = int(bar["close"] > bar["open"])
            if self.pending.signal == "long":
                score = 1.0 if actual_up else -1.0
            elif self.pending.signal == "short":
                score = 1.0 if not actual_up else -1.0
            else:
                score = self.abstention_penalty
            settled = {**asdict(self.pending), "target": actual_up, "score": score}
            self.rows.append(settled)
        self.pending = PendingSignal(probability_up, signal, bar["close_time"].isoformat())
        return settled

    def metrics(self) -> dict[str, float]:
        if not self.rows:
            return {"signals": 0.0, "score": 0.0, "mean_score": 0.0}
        target = np.asarray([row["target"] for row in self.rows])
        probability = np.asarray([
            1.0 if row["signal"] == "long" else 0.0 if row["signal"] == "short" else 0.5
            for row in self.rows
        ])
        return signal_metrics(probability, target, 0.75, 0.25, abstention_penalty=self.abstention_penalty)

    def comparison(self) -> dict[str, dict[str, float]]:
        """Return model and fixed-direction baselines over settled live rows."""
        if not self.rows:
            empty = self.metrics()
            return {name: empty for name in ("model", "always_up", "always_down", "always_abstain")}
        target = np.asarray([row["target"] for row in self.rows])
        return {
            "model": self.metrics(),
            "always_up": signal_metrics(np.ones(len(target)), target, 0.5, -0.01, abstention_penalty=self.abstention_penalty),
            "always_down": signal_metrics(np.zeros(len(target)), target, 1.01, 0.5, abstention_penalty=self.abstention_penalty),
            "always_abstain": signal_metrics(np.full(len(target), 0.5), target, 1.01, -0.01, abstention_penalty=self.abstention_penalty),
        }


class LiveStateStore:
    """Atomic JSON checkpoint and append-only JSONL event writer under ``data/live``."""

    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.checkpoint_path = self.directory / "state.json"
        self.events_path = self.directory / "events.jsonl"

    def save(self, *, rust_snapshot: str, bars: deque[dict[str, Any]]) -> None:
        payload = {"rust_snapshot": rust_snapshot, "bars": [self._json_row(row) for row in bars]}
        temporary = self.checkpoint_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(payload), encoding="utf-8")
        os.replace(temporary, self.checkpoint_path)

    def load(self) -> dict[str, Any] | None:
        if not self.checkpoint_path.exists():
            return None
        return json.loads(self.checkpoint_path.read_text(encoding="utf-8"))

    def event(self, kind: str, payload: dict[str, Any]) -> None:
        row = {"kind": kind, "recorded_at": datetime.now(UTC).isoformat(), **self._json_row(payload)}
        with self.events_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, allow_nan=False) + "\n")

    @staticmethod
    def _json_row(row: dict[str, Any]) -> dict[str, Any]:
        def convert(value: Any) -> Any:
            if isinstance(value, datetime):
                return value.isoformat()
            if isinstance(value, float) and not math.isfinite(value):
                return None
            if isinstance(value, Decimal):
                return str(value)
            if isinstance(value, dict):
                return {name: convert(item) for name, item in value.items()}
            if isinstance(value, (list, tuple)):
                return [convert(item) for item in value]
            return value

        return {name: convert(value) for name, value in row.items()}
