"""Monthly five-year walk-forward test for overnight-range breakout selection."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from lightgbm import LGBMRegressor


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from libs.overnight_range import (  # noqa: E402
    OvernightRangeBreakoutConfig,
    OvernightRangeFeatureConfig,
    build_overnight_range_breakout_ledger,
    build_overnight_range_features,
)


@dataclass(frozen=True)
class ModelSettings:
    n_estimators: int = 300
    learning_rate: float = 0.03
    num_leaves: int = 31
    min_child_samples: int = 30
    reg_lambda: float = 1.0
    random_state: int = 18616
    n_jobs: int = -1


def month_start(value: str) -> date:
    parsed = date.fromisoformat(value)
    if parsed.day != 1:
        raise argparse.ArgumentTypeError("dates must be the first day of a month (YYYY-MM-01).")
    return parsed


def add_months(value: date, months: int) -> date:
    index = value.year * 12 + value.month - 1 + months
    return date(index // 12, index % 12 + 1, 1)


def months_between(start: date, end: date) -> list[date]:
    if start >= end:
        raise ValueError("start-month must be before end-month.")
    result: list[date] = []
    current = start
    while current < end:
        result.append(current)
        current = add_months(current, 1)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a monthly five-year overnight-range LightGBM walk-forward backtest.")
    parser.add_argument("--parquet-path", type=Path, default=PROJECT_ROOT / "data" / "minute_bars_gbpusd.parquet")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument("--start-month", type=month_start, default=date(2020, 1, 1))
    parser.add_argument("--end-month", type=month_start, default=date(2025, 1, 1), help="Exclusive month.")
    parser.add_argument("--history-months", type=int, default=120)
    parser.add_argument("--validation-months", type=int, default=12)
    parser.add_argument("--minimum-train-rows", type=int, default=100)
    parser.add_argument("--minimum-validation-rows", type=int, default=20)
    parser.add_argument("--minimum-validation-coverage", type=float, default=0.10)
    parser.add_argument("--coverage-threshold", type=float, default=0.95)
    parser.add_argument("--buffer-pips", type=float, default=3.0)
    parser.add_argument("--stop-range-multiple", type=float, default=0.5)
    parser.add_argument("--target-range-multiple", type=float, default=2.0)
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--num-leaves", type=int, default=31)
    parser.add_argument("--min-child-samples", type=int, default=30)
    parser.add_argument("--reg-lambda", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=18616)
    parser.add_argument("--n-jobs", type=int, default=-1)
    return parser


def model_settings(args: argparse.Namespace) -> ModelSettings:
    return ModelSettings(
        n_estimators=args.n_estimators, learning_rate=args.learning_rate, num_leaves=args.num_leaves,
        min_child_samples=args.min_child_samples, reg_lambda=args.reg_lambda,
        random_state=args.seed, n_jobs=args.n_jobs,
    )


def fit_model(frame: pl.DataFrame, features: list[str], settings: ModelSettings) -> LGBMRegressor:
    return LGBMRegressor(objective="regression", verbosity=-1, **asdict(settings)).fit(
        frame.select(features).to_numpy(), frame["r_return"].to_numpy(),
    )


def select_r_gate(
    validation: pl.DataFrame,
    predicted_r: np.ndarray,
    minimum_coverage: float,
) -> tuple[float, dict[str, float]]:
    """Choose the validation predicted-R gate maximizing total realized R.

    Quantile candidates include the minimum prediction (all validation trades).
    Equal total R prefers higher coverage, then the lower gate.
    """
    if not 0 < minimum_coverage <= 1:
        raise ValueError("minimum_validation_coverage must be in (0, 1].")
    if validation.height != len(predicted_r) or validation.is_empty():
        raise ValueError("validation and predicted_r must have the same non-zero length.")
    required = int(np.ceil(validation.height * minimum_coverage))
    candidates: list[tuple[float, float, float, dict[str, float]]] = []
    for gate in np.unique(np.quantile(predicted_r, np.linspace(0.0, 0.90, 91))):
        selected = validation.filter(pl.Series(predicted_r >= gate))
        if selected.height < required:
            continue
        total_r = float(selected["r_return"].sum())
        coverage = selected.height / validation.height
        metrics = {
            "validation_selected_trades": float(selected.height),
            "validation_coverage": float(coverage),
            "validation_total_r": total_r,
            "validation_mean_r": float(selected["r_return"].mean()),
        }
        candidates.append((total_r, coverage, -float(gate), metrics))
    if not candidates:
        raise ValueError("No predicted-R gate met minimum validation coverage.")
    _, _, negative_gate, metrics = max(candidates, key=lambda item: item[:3])
    return -negative_gate, metrics


def performance(frame: pl.DataFrame, total_candidates: int) -> dict[str, float]:
    if frame.is_empty():
        return {"trades": 0.0, "coverage": 0.0, "total_r": 0.0, "mean_r": float("nan"), "win_rate": float("nan")}
    return {
        "trades": float(frame.height),
        "coverage": float(frame.height / total_candidates),
        "total_r": float(frame["r_return"].sum()),
        "mean_r": float(frame["r_return"].mean()),
        "win_rate": float(frame["target"].mean()),
    }


def _summary_row(month: date, **values: Any) -> dict[str, Any]:
    return {"score_month": month, **values}


def run_walk_forward(
    dataset: pl.DataFrame,
    feature_columns: list[str],
    *,
    start_month: date,
    end_month: date,
    history_months: int,
    validation_months: int,
    minimum_train_rows: int,
    minimum_validation_rows: int,
    minimum_validation_coverage: float,
    settings: ModelSettings,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Score one out-of-sample calendar month per five-year rolling model."""
    if history_months <= validation_months or validation_months < 1:
        raise ValueError("history_months must be greater than validation_months, which must be positive.")
    if minimum_train_rows < 1 or minimum_validation_rows < 1:
        raise ValueError("minimum row counts must be positive.")
    records: list[dict[str, Any]] = []
    scored_frames: list[pl.DataFrame] = []
    for score_month in months_between(start_month, end_month):
        history_start = add_months(score_month, -history_months)
        validation_start = add_months(score_month, -validation_months)
        score_end = add_months(score_month, 1)
        history = dataset.filter((pl.col("session_date") >= history_start) & (pl.col("session_date") < score_month))
        fit = history.filter(pl.col("session_date") < validation_start)
        validation = history.filter(pl.col("session_date") >= validation_start)
        scored = dataset.filter((pl.col("session_date") >= score_month) & (pl.col("session_date") < score_end))
        common = {
            "history_start": history_start, "validation_start": validation_start, "history_end": score_month,
            "history_rows": history.height, "fit_rows": fit.height, "validation_rows": validation.height,
            "scored_candidates": scored.height,
        }
        if fit.height < minimum_train_rows:
            records.append(_summary_row(score_month, status="skipped_insufficient_fit_rows", **common))
            continue
        if validation.height < minimum_validation_rows:
            records.append(_summary_row(score_month, status="skipped_insufficient_validation_rows", **common))
            continue
        if scored.is_empty():
            records.append(_summary_row(score_month, status="skipped_no_scored_trades", **common))
            continue

        validation_model = fit_model(fit, feature_columns, settings)
        validation_prediction = validation_model.predict(validation.select(feature_columns).to_numpy())
        try:
            gate, validation_metrics = select_r_gate(validation, validation_prediction, minimum_validation_coverage)
        except ValueError as error:
            records.append(_summary_row(score_month, status="skipped_no_valid_gate", error=str(error), **common))
            continue
        final_model = fit_model(history, feature_columns, settings)
        prediction = final_model.predict(scored.select(feature_columns).to_numpy())
        selected = scored.filter(pl.Series(prediction >= gate))
        baseline_metrics = performance(scored, scored.height)
        selected_metrics = performance(selected, scored.height)
        records.append(_summary_row(
            score_month, status="scored", selected_gate=gate, **common, **validation_metrics,
            **{f"baseline_{name}": value for name, value in baseline_metrics.items()},
            **{f"selected_{name}": value for name, value in selected_metrics.items()},
        ))
        scored_frames.append(scored.with_columns([
            pl.lit(score_month).alias("score_month"),
            pl.lit(history_start).alias("history_start"),
            pl.lit(validation_start).alias("validation_start"),
            pl.lit(gate).alias("selected_gate"),
            pl.Series("predicted_r_return", prediction),
            pl.Series("take_trade", prediction >= gate),
        ]))
    summary = pl.DataFrame(records).sort("score_month")
    trades = pl.concat(scored_frames, how="diagonal_relaxed").sort(["score_month", "session_date"]) if scored_frames else pl.DataFrame()
    return trades, summary


def _json_number(value: float) -> float | None:
    return value if np.isfinite(value) else None


def write_outputs(trades: pl.DataFrame, summary: pl.DataFrame, output_dir: Path, manifest: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in (("overnight_range_walk_forward_trades", trades), ("overnight_range_walk_forward_monthly", summary)):
        frame.write_parquet(output_dir / f"{name}.parquet")
        frame.write_csv(output_dir / f"{name}.csv")
    (output_dir / "overnight_range_walk_forward_manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not 0 <= args.coverage_threshold <= 1:
        raise ValueError("coverage-threshold must be in [0, 1].")
    bars = pl.read_parquet(args.parquet_path)
    breakout = OvernightRangeBreakoutConfig(
        buffer_pips=args.buffer_pips, stop_range_multiple=args.stop_range_multiple,
        target_range_multiple=args.target_range_multiple,
    )
    feature = OvernightRangeFeatureConfig(
        minimum_range_coverage=args.coverage_threshold,
        minimum_trade_coverage=args.coverage_threshold,
    )
    ledger = build_overnight_range_breakout_ledger(bars, breakout)
    dataset, feature_columns = build_overnight_range_features(ledger, bars, feature)
    settings = model_settings(args)
    trades, summary = run_walk_forward(
        dataset, feature_columns, start_month=args.start_month, end_month=args.end_month,
        history_months=args.history_months, validation_months=args.validation_months,
        minimum_train_rows=args.minimum_train_rows, minimum_validation_rows=args.minimum_validation_rows,
        minimum_validation_coverage=args.minimum_validation_coverage, settings=settings,
    )
    scored = summary.filter(pl.col("status") == "scored")
    selected_trades = trades.filter(pl.col("take_trade")) if not trades.is_empty() else trades
    manifest = {
        "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "feature_columns": feature_columns,
        "dataset_rows": dataset.height,
        "months_requested": len(months_between(args.start_month, args.end_month)),
        "months_scored": scored.height,
        "trade_rows_scored": trades.height,
        "selected_trade_rows": selected_trades.height,
        "baseline_total_r": _json_number(float(trades["r_return"].sum())) if not trades.is_empty() else 0.0,
        "selected_total_r": _json_number(float(selected_trades["r_return"].sum())) if not selected_trades.is_empty() else 0.0,
    }
    write_outputs(trades, summary, args.output_dir, manifest)
    print(f"Scored months: {scored.height}/{len(months_between(args.start_month, args.end_month))}")
    print(f"Trade rows: {trades.height}; selected: {selected_trades.height}")
    print(f"Outputs: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
